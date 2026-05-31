const code = await fetch(new URL('Denoiser.wgsl', import.meta.url))
    .then(response => response.text());

export class Denoiser {

    constructor(device, format = 'rgba8unorm') {
        this.device = device;
        this.format = format;

        // Privzete vrednosti parametrov
        // sigmaS = 3.0: Gaussova prostorska utež, učinkovit polmer ~3 piksle
        //   pri hiBound=1 je šum gost -> potrebujemo večji polmer
        // sigmaD = 1.0: globinska utež med foreground piksli (oba depth > 0)
        //   vrednosti so linearna razdalja od kamere v enotah scene
        this.sigmaS = 1.5;
        this.sigmaD = 1.0;

        // Uniform buffer: 2 x f32 = 8 bajtov
        this.uniformBuffer = this.device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Sampler: non-filtering ker bomo mi sami računali uteži
        this.sampler = this.device.createSampler({
            // privzeto: nearest, clamp-to-edge — ne interpoliramo, ne ponavljamo
        });

        const module = this.device.createShaderModule({ code });

        // Bind group layout: colorTexture + depthTexture + sampler + uniforms
        this.layout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'unfilterable-float' },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    // r32float za globinsko teksturo — unfilterable
                    texture: { sampleType: 'unfilterable-float' },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: { type: 'non-filtering' },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {},
                },
            ],
        });

        this.pipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.layout],
            }),
            vertex: { module },
            fragment: {
                module,
                targets: [{ format: this.format }],
            },
        });
    }

    render(renderTarget, colorTexture, depthTexture) {
        // Posodobi parametre v uniform buffer
        this.device.queue.writeBuffer(
            this.uniformBuffer, 0,
            new Float32Array([this.sigmaS, this.sigmaD])
        );

        const bindGroup = this.device.createBindGroup({
            layout: this.layout,
            entries: [
                { binding: 0, resource: colorTexture.createView() },
                { binding: 1, resource: depthTexture.createView() },
                { binding: 2, resource: this.sampler },
                { binding: 3, resource: { buffer: this.uniformBuffer } },
            ],
        });

        const commandEncoder = this.device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: renderTarget.color.createView(),
                loadOp: 'clear',
                clearValue: [0, 0, 0, 1],
                storeOp: 'store',
            }],
        });

        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(3);  // en full-screen trikotnik
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }
}
