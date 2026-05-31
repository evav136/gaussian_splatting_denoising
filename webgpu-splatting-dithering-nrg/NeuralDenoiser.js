/**
 * NeuralDenoiser.js
 * Enkratno neural denoisanje z RCNN U-Net + KPCN modelom (ONNX Runtime Web).
 *
 * Ni real-time — inferenca traja ~5-15 sekund na CPU (WASM backend).
 * Uporaba: pritisni gumb v GUI, pocakaj, rezultat se prikaze v plovecen panelu.
 *
 * Pipeline:
 *   1. Preberi colorTexture (noisy RGB) + depthTexture iz GPU
 *   2. Zmnozaj na 512x512 (hitrejsa inferenca, model treniran na 512x512 cropih)
 *   3. Sestavi vhodni tenzor [1, 4, 512, 512]: RGB + normalizirana globina
 *   4. ONNX inferenca -> [1, 3, 512, 512] denoised RGB
 *   5. Prikazi before/after v plovecen panelu
 */

const INFERENCE_SIZE = 256;  // shrink -> denoise -> upsample: hitrejse kot 512, dovolj za demo
// WASM datoteke so na CDN (fetchamo z fetch(), ne z import() -> ni CORS problema)
const WASM_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';

export class NeuralDenoiser {

    constructor(device, canvasFormat) {
        this.device      = device;
        this.canvasFormat = canvasFormat;
        this.session     = null;
        this.ort         = null;
        this._panel      = null;
    }

    // Nalozi ONNX model (klici enkrat ob zagonu)
    // Predpogoj: lib/ort.min.js mora biti nalozena kot <script> v index.html
    //            -> window.ort je globalno dostopen
    async init(modelPath = 'denoiser.onnx') {
        const ort = window.ort;
        if (!ort) throw new Error('window.ort ni na voljo — preveri da je lib/ort.min.js nalozena v index.html');

        // WASM datoteke (ort-wasm.wasm, ort-wasm-simd.wasm, ...) fetchamo s CDN
        // fetch() nima CORS omejitev za javne CDN-je
        ort.env.wasm.wasmPaths = WASM_CDN;
        this.ort = ort;

        // Poskusi WebGPU (GPU inferenca), ce ne gre pade nazaj na WASM (CPU)
        this.session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['webgpu', 'wasm'],
        });
    }

    // Pozeni inferenco na trenutnem okvirju
    // colorTexture: rgba8unorm/bgra8unorm, depthTexture: r32float
    // Vrne: opis rezultata (za GUI status)
    async denoise(colorTexture, depthTexture) {
        if (!this.session) throw new Error('Model ni nalozeno. Klici init() najprej.');

        const W = colorTexture.width;
        const H = colorTexture.height;

        // 1. Preberi teksturi z GPU
        const colorData = await this._readColor(colorTexture);   // Uint8Array [H*W*4]
        const depthData = await this._readDepth(depthTexture);   // Float32Array [H*W]

        // 2. Zmnozaj na INFERENCE_SIZE x INFERENCE_SIZE
        const colorSmall = this._resizeColor(colorData, W, H, INFERENCE_SIZE, INFERENCE_SIZE);
        const depthSmall = this._resizeDepth(depthData, W, H, INFERENCE_SIZE, INFERENCE_SIZE);

        // 3. Sestavi vhodni tenzor [1, 4, S, S]: normalizirani RGB + globina
        const S  = INFERENCE_SIZE;
        const N  = S * S;
        const input = new Float32Array(4 * N);
        for (let i = 0; i < N; i++) {
            input[0 * N + i] = colorSmall[i * 4 + 0] / 255.0;  // R
            input[1 * N + i] = colorSmall[i * 4 + 1] / 255.0;  // G
            input[2 * N + i] = colorSmall[i * 4 + 2] / 255.0;  // B
            input[3 * N + i] = depthSmall[i];                    // depth [0,1]
        }

        // 4. ONNX inferenca
        const tensor = new this.ort.Tensor('float32', input, [1, 4, S, S]);
        const results = await this.session.run({ 'noisy_depth': tensor });
        const output = results['denoised'].data;   // Float32Array [1, 3, S, S]

        // 5. Prikazi before/after
        const noisyCanvas   = this._float32ToCanvas(input,  S, S, true);   // iz vhoda (RGB)
        const denoisedCanvas = this._float32ToCanvas(output, S, S, false);  // iz izhoda
        this._showPanel(noisyCanvas, denoisedCanvas);

        return 'Konec! (glejte panel)';
    }

    // ── Branje GPU tekstur ────────────────────────────────────────────────────

    async _readColor(texture) {
        const { width, height } = texture;
        const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
        const buf = this.device.createBuffer({
            size:  bytesPerRow * height,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const enc = this.device.createCommandEncoder();
        enc.copyTextureToBuffer({ texture }, { buffer: buf, bytesPerRow }, { width, height });
        this.device.queue.submit([enc.finish()]);

        await buf.mapAsync(GPUMapMode.READ);
        const raw = new Uint8Array(buf.getMappedRange().slice(0));
        buf.unmap(); buf.destroy();

        // Odstrani row padding + popravi BGRA -> RGBA (Mac vrne bgra8unorm)
        const isBGRA = this.canvasFormat === 'bgra8unorm';
        const out = new Uint8Array(width * height * 4);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const src = y * bytesPerRow + x * 4;
                const dst = (y * width + x) * 4;
                out[dst + 0] = isBGRA ? raw[src + 2] : raw[src + 0];
                out[dst + 1] = raw[src + 1];
                out[dst + 2] = isBGRA ? raw[src + 0] : raw[src + 2];
                out[dst + 3] = 255;
            }
        }
        return out;
    }

    async _readDepth(texture) {
        const { width, height } = texture;
        const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
        const buf = this.device.createBuffer({
            size:  bytesPerRow * height,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const enc = this.device.createCommandEncoder();
        enc.copyTextureToBuffer({ texture }, { buffer: buf, bytesPerRow }, { width, height });
        this.device.queue.submit([enc.finish()]);

        await buf.mapAsync(GPUMapMode.READ);
        const raw = new Float32Array(buf.getMappedRange().slice(0));
        buf.unmap(); buf.destroy();

        // Odstrani row padding in normaliziraj na [0, 1]
        const stride = bytesPerRow / 4;
        const out = new Float32Array(width * height);
        let maxD = 0;
        for (let y = 0; y < height; y++)
            for (let x = 0; x < width; x++) {
                const v = raw[y * stride + x];
                if (v > maxD) maxD = v;
                out[y * width + x] = v;
            }
        // Normalizacija: enako kot DataCapture._saveDepthPNG
        if (maxD > 0)
            for (let i = 0; i < out.length; i++) out[i] /= maxD;
        return out;
    }

    // ── Resize ────────────────────────────────────────────────────────────────

    _resizeColor(data, srcW, srcH, dstW, dstH) {
        // Resize z 2D canvas (bilinearno interpolacijo brskalnik naredi sam)
        const srcCanvas = document.createElement('canvas');
        srcCanvas.width  = srcW;
        srcCanvas.height = srcH;
        const srcCtx = srcCanvas.getContext('2d');
        const img = new ImageData(new Uint8ClampedArray(data), srcW, srcH);
        srcCtx.putImageData(img, 0, 0);

        const dstCanvas = document.createElement('canvas');
        dstCanvas.width  = dstW;
        dstCanvas.height = dstH;
        const dstCtx = dstCanvas.getContext('2d');
        dstCtx.drawImage(srcCanvas, 0, 0, dstW, dstH);
        return dstCtx.getImageData(0, 0, dstW, dstH).data;
    }

    _resizeDepth(data, srcW, srcH, dstW, dstH) {
        // Nearest-neighbor resize za globino
        const out = new Float32Array(dstW * dstH);
        for (let y = 0; y < dstH; y++) {
            for (let x = 0; x < dstW; x++) {
                const sx = Math.floor(x * srcW / dstW);
                const sy = Math.floor(y * srcH / dstH);
                out[y * dstW + x] = data[sy * srcW + sx];
            }
        }
        return out;
    }

    // ── Prikaz ────────────────────────────────────────────────────────────────

    _float32ToCanvas(data, W, H, isInput) {
        const canvas = document.createElement('canvas');
        canvas.width  = W;
        canvas.height = H;
        const ctx = canvas.getContext('2d');
        const img = ctx.createImageData(W, H);
        const N   = W * H;
        for (let i = 0; i < N; i++) {
            if (isInput) {
                // Vhod je shranjen kot [R,G,B,D, R,G,B,D, ...] -> [R plane, G plane, B plane, D plane]
                img.data[i * 4 + 0] = Math.round(data[0 * N + i] * 255);
                img.data[i * 4 + 1] = Math.round(data[1 * N + i] * 255);
                img.data[i * 4 + 2] = Math.round(data[2 * N + i] * 255);
            } else {
                // Izhod: [R plane, G plane, B plane]
                img.data[i * 4 + 0] = Math.round(Math.min(1, Math.max(0, data[0 * N + i])) * 255);
                img.data[i * 4 + 1] = Math.round(Math.min(1, Math.max(0, data[1 * N + i])) * 255);
                img.data[i * 4 + 2] = Math.round(Math.min(1, Math.max(0, data[2 * N + i])) * 255);
            }
            img.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
        return canvas;
    }

    _showPanel(noisyCanvas, denoisedCanvas) {
        // Odstrani stari panel ce obstaja
        if (this._panel) this._panel.remove();

        const panel = document.createElement('div');
        panel.style.cssText = `
            position: fixed; top: 10px; right: 10px; z-index: 1000;
            background: rgba(0,0,0,0.85); border: 1px solid #555;
            border-radius: 8px; padding: 12px; color: white;
            font-family: monospace; font-size: 12px;
        `;

        const title = document.createElement('div');
        title.textContent = 'RCNN U-Net + KPCN (neural denoiser)';
        title.style.cssText = 'margin-bottom: 8px; font-weight: bold; color: #adf';
        panel.appendChild(title);

        const row = document.createElement('div');
        row.style.cssText = 'display: flex; gap: 8px; align-items: flex-start;';

        for (const [label, canvas] of [['Surov (šumen)', noisyCanvas], ['Denoisano', denoisedCanvas]]) {
            const col = document.createElement('div');
            col.style.cssText = 'text-align: center;';
            const lbl = document.createElement('div');
            lbl.textContent = label;
            lbl.style.cssText = 'margin-bottom: 4px; color: #ccc;';
            canvas.style.cssText = 'display: block; max-width: 300px; max-height: 300px;';
            col.appendChild(lbl);
            col.appendChild(canvas);
            row.appendChild(col);
        }
        panel.appendChild(row);

        const close = document.createElement('button');
        close.textContent = '✕ Zapri';
        close.style.cssText = `
            margin-top: 10px; background: #333; color: white;
            border: 1px solid #555; border-radius: 4px;
            padding: 4px 12px; cursor: pointer; font-size: 12px;
        `;
        close.onclick = () => panel.remove();
        panel.appendChild(close);

        document.body.appendChild(panel);
        this._panel = panel;
    }
}
