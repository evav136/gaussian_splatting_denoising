import { GUI } from 'dat';
import { vec3, vec4, mat4, quat } from 'glm';

import { ResizeSystem } from 'engine/systems/ResizeSystem.js';
import { UpdateSystem } from 'engine/systems/UpdateSystem.js';

import {
    Camera,
    Node,
    Transform,
} from 'engine/core.js';

import { TouchController } from 'engine/controllers/TouchController.js';

import { parseSplats } from './parseSplats.js';
import { Splat } from './Splat.js';
import { SplatRenderer } from './SplatRenderer.js';
import { Compositor } from './Compositor.js';
import { Denoiser } from './Denoiser.js';
import { DataCapture } from './DataCapture.js';
import { NeuralDenoiser } from './NeuralDenoiser.js';

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredFeatures: ['float32-blendable'] });
const canvas = document.querySelector('canvas');
const context = canvas.getContext('webgpu');
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format });

const renderer = new SplatRenderer(device, format);
const denoiser = new Denoiser(device, format);
const dataCapture = new DataCapture(device, format);
const neuralDenoiser = new NeuralDenoiser(device, format);
const compositorFloat = new Compositor(device, 'rgba32float');
const compositor = new Compositor(device, format);
compositor.gamma = 1;

let depthTexture, colorTexture, compositorTexture, depthColorTexture, denoiserTexture;
let nFrames = 0;
let denoiseEnabled = true;
let captureRequested = false;

const scene = new Node();

const splatContainer = new Node();
scene.addChild(splatContainer);

const camera = new Node();
camera.addComponent(new Transform());
camera.addComponent(new Camera());
camera.addComponent(new TouchController(camera, canvas));
scene.addChild(camera);

canvas.addEventListener('dragover', e => {
    e.preventDefault();
});

canvas.addEventListener('drop', async e => {
    e.preventDefault();

    for (const child of splatContainer.children) {
        console.log('Removing existing splats');
        child.remove();
    }

    console.log('Adding new splats');
    console.log(e.dataTransfer.files);
    const arrayBuffers = await Promise.all([...e.dataTransfer.files].map(file => file.arrayBuffer()));
    for (const arrayBuffer of arrayBuffers) {
        console.log('New splat');
        const splatData = parseSplats(arrayBuffer);
        const splatMean = splatData
            .map(splat => splat.position)
            .reduce((a, p) => vec3.add(a, a, vec3.scale(vec3.create(), p, 1 / splatData.length)), vec3.create());
        for (const splat of splatData) {
            vec3.subtract(splat.position, splat.position, splatMean);
        }

        const splat = new Node();
        splat.addComponent(new Splat(splatData));
        splatContainer.addChild(splat);
        nFrames = 0;
    }
});

function update(t, dt) {
    scene.traverse(node => {
        for (const component of node.components) {
            component.update?.(t, dt);
        }
    });
}

camera.addComponent({
    lastTransform: camera.getComponentOfType(Transform).matrix,
    update() {
        const newTransform = camera.getComponentOfType(Transform).matrix;
        if (mat4.exactEquals(newTransform, this.lastTransform)) {
            return;
        }

        this.lastTransform = newTransform;
        nFrames = 0;
    }
});

function render() {
    if (!colorTexture || colorTexture.width !== canvas.width || colorTexture.height !== canvas.height) {
        colorTexture?.destroy();
        colorTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            format: format,
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,  // potrebno za DataCapture readback
        });
    }

    if (!depthTexture || depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
        depthTexture?.destroy();
        depthTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    if (!compositorTexture || compositorTexture.width !== canvas.width || compositorTexture.height !== canvas.height) {
        compositorTexture?.destroy();
        compositorTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            format: 'rgba32float',
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,  // potrebno za DataCapture readback
        });
    }

    // globinska tekstura: r32float, berljiva v denoiserju
    // vsak piksel = linearna razdalja od kamere v enotah scene (0.0 = ozadje)
    if (!depthColorTexture || depthColorTexture.width !== canvas.width || depthColorTexture.height !== canvas.height) {
        depthColorTexture?.destroy();
        depthColorTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            format: 'r32float',
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,  // potrebno za DataCapture readback
        });
    }

    // izhod denoiserja: isti format kot colorTexture
    // compositor bo bral iz te teksture namesto iz surovega colorTexture
    if (!denoiserTexture || denoiserTexture.width !== canvas.width || denoiserTexture.height !== canvas.height) {
        denoiserTexture?.destroy();
        denoiserTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            format: format,
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    const renderTarget = {
        color: colorTexture,
        depth: depthTexture,
        depthColor: depthColorTexture,
    };

    const compositorTarget = {
        color: compositorTexture,
    };

    const canvasTarget = {
        color: context.getCurrentTexture(),
    };

    // 1. Renderer: surovi šumni okvir -> colorTexture + depthColorTexture
    renderer.render(renderTarget, scene, camera);

    // DataCapture: zajem training parov
    if (captureRequested) {
        captureRequested = false;
        nFrames = 0;
        dataCapture.trigger();
    }
    dataCapture.update(nFrames, colorTexture, depthColorTexture, compositorTexture);

    // 2. Časovna akumulacija [1]:
    //    surovi okvir -> compositorTexture (tekoče povprečje)
    //    varianca pada kot 1/N - po ~30 okvirjih večina šuma izgineta
    compositorFloat.render(compositorTarget, colorTexture, 1 / (++nFrames), 1);

    // 3. Prostorski bilateralni filter (POTEM, kot v [1]):
    //    apliciramo samo dokler časovna akumulacija še ni konvergirala (nFrames < prag).
    //    Ko je nFrames velik, je compositorTexture že čist -> spatial filter bi samo zamegljal.
    //    Prag ~20 okvirjev: pri σ≈0.3 (stohastičen šum) se residualni šum zmanjša za ~4.5×.
    const spatialActive = denoiseEnabled && nFrames <= 20;
    if (spatialActive) {
        denoiser.render({ color: denoiserTexture }, compositorTexture, depthColorTexture);
        compositor.render(canvasTarget, denoiserTexture);
    } else {
        compositor.render(canvasTarget, compositorTexture);
    }
}

function resize({ displaySize: { width, height }}) {
    camera.getComponentOfType(Camera).aspect = width / height;
    nFrames = 0;
}

new ResizeSystem({ canvas, resize }).start();
new UpdateSystem({ update, render }).start();

const gui = new GUI();
gui.add(renderer, 'splatScale', 0, 10);
gui.add(renderer, 'loBound', 0, 1);
gui.add(renderer, 'hiBound', 0, 1);

const denoiserFolder = gui.addFolder('Denoiser');
denoiserFolder.add({ denoiseEnabled }, 'denoiseEnabled').name('vklopljen').onChange(v => {
    denoiseEnabled = v;
    nFrames = 0;  // resetiramo časovno akumulacijo ob preklopu
});
denoiserFolder.add(denoiser, 'sigmaS', 0.5, 8).name('sigmaS (prostorski)').onChange(() => { nFrames = 0; });
denoiserFolder.add(denoiser, 'sigmaD', 0.01, 10).name('sigmaD').onChange(() => { nFrames = 0; });
denoiserFolder.open();

const captureFolder = gui.addFolder('Training Data');
const captureStatus = { status: 'Pripravljeno (0 parov)' };
captureFolder.add(captureStatus, 'status').name('stanje').listen();
captureFolder.add({
    capture() { captureRequested = true; }
}, 'capture').name('Zajemi par (premakni kamero najprej)');
captureFolder.open();

dataCapture.onStatusChange = msg => { captureStatus.status = msg; };

// Neural denoiser GUI
const neuralFolder = gui.addFolder('Neural Denoiser (RCNN U-Net)');
const neuralStatus = { status: 'Ni nalozeno...' };
neuralFolder.add(neuralStatus, 'status').name('stanje').listen();
neuralFolder.add({
    async apply() {
        neuralStatus.status = 'Nalagam model...';
        try {
            if (!neuralDenoiser.session) {
                await neuralDenoiser.init('denoiser.onnx');
                neuralStatus.status = 'Model nalozeno. Poganjam inferenco...';
            } else {
                neuralStatus.status = 'Poganjam inferenco...';
            }
            await neuralDenoiser.denoise(colorTexture, depthColorTexture);
            neuralStatus.status = 'Konec! (glejte panel zgoraj desno)';
        } catch(e) {
            neuralStatus.status = 'Napaka: ' + e.message;
            console.error(e);
        }
    }
}, 'apply').name('Uporabi neural denoiser');
neuralFolder.open();
