// DataCapture.js
// Generira (šumni, čisti) pare za trening RCNN U-Net denoiserja.
//
// Potek za vsak par:
//   1. nFrames resetira na 0 -> okvir 1 = surov šumen okvir (vhod)
//   2. čakamo 100 okvirjev -> compositorTexture konvergira (cilj)
//   3. shranimo oba kot PNG -> noisy_XXXX.png, clean_XXXX.png, depth_XXXX.png
//
// Globinska slika (depth) je pomožni vhod mreži za edge-stopping (iz [2] Chaitanya et al.)

export class DataCapture {

    constructor(device, canvasFormat) {
        this.device = device;
        this.canvasFormat = canvasFormat; // 'rgba8unorm' ali 'bgra8unorm' (odvisno od platforme)

        this.state = 'idle';        // 'idle' | 'need_noisy' | 'wait_clean'
        this.captureIndex = 0;
        this.busy = false;

        // Shranjeni podatki med čakanjem na konvergenco
        this.savedNoisy = null;
        this.savedDepth = null;

        // Koliko okvirjev čakamo na konvergenco
        // 100 okvirjev: σ_šuma zmanjšan za 10× (1/√100)
        this.convergenceFrames = 100;

        // Callback ki ga kličemo ko se stanje spremeni (za GUI prikaz)
        this.onStatusChange = null;
    }

    // Sproži zajem novega para - pokliče main.js naj resetira nFrames
    // Vrne true če je treba resetirati nFrames
    trigger() {
        if (this.state !== 'idle' || this.busy) return false;
        this.state = 'need_noisy';
        this._notify('Čakam na šumni okvir...');
        return true; // main.js resetira nFrames
    }

    // Pokliči vsak okvir iz render zanke
    // Vrne { resetFrames: bool }
    update(nFrames, colorTexture, depthColorTexture, compositorTexture) {
        if (this.busy) return;

        // Okvir 1: zajemi šumni vhod
        if (this.state === 'need_noisy' && nFrames === 1) {
            this.busy = true;
            this._notify('Zajemujem šumni okvir...');

            this._readUnorm8(colorTexture)
                .then(noisy => {
                    this.savedNoisy = { data: noisy, width: colorTexture.width, height: colorTexture.height };
                    return this._readFloat32(depthColorTexture, 1);
                })
                .then(depth => {
                    this.savedDepth = { data: depth, width: depthColorTexture.width, height: depthColorTexture.height };
                    this.state = 'wait_clean';
                    this.busy = false;
                    this._notify(`Čakam na konvergenco (0 / ${this.convergenceFrames} okvirjev)...`);
                });
        }

        // Periodično posodabljaj napredek
        if (this.state === 'wait_clean' && !this.busy) {
            const progress = Math.min(nFrames, this.convergenceFrames);
            if (nFrames % 10 === 0) {
                this._notify(`Čakam na konvergenco (${progress} / ${this.convergenceFrames} okvirjev)...`);
            }
        }

        // Okvir 100+: zajemi čisti cilj
        if (this.state === 'wait_clean' && nFrames >= this.convergenceFrames) {
            this.busy = true;
            this._notify('Zajemujem čisti okvir...');

            this._readFloat32(compositorTexture, 4)
                .then(clean => {
                    const cleanImg = { data: clean, width: compositorTexture.width, height: compositorTexture.height };
                    this._savePair(this.savedNoisy, cleanImg, this.savedDepth);
                    this.captureIndex++;
                    this.state = 'idle';
                    this.busy = false;
                    this._notify(`✓ Par ${this.captureIndex} shranjen. Premakni kamero in pritisni znova.`);
                });
        }
    }

    get statusText() {
        if (this.state === 'idle') return `Pripravljeno (${this.captureIndex} parov)`;
        return this._lastStatus || '...';
    }

    // ── Interne metode ───────────────────────────────────────────────────────

    _notify(msg) {
        this._lastStatus = msg;
        if (this.onStatusChange) this.onStatusChange(msg);
    }

    // Preberi unorm8 teksturo (colorTexture: rgba8unorm ali bgra8unorm)
    async _readUnorm8(texture) {
        const { width, height } = texture;
        const bytesPerPixel = 4;
        const bytesPerRow = Math.ceil(width * bytesPerPixel / 256) * 256;

        const buffer = this.device.createBuffer({
            size: bytesPerRow * height,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const enc = this.device.createCommandEncoder();
        enc.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, { width, height });
        this.device.queue.submit([enc.finish()]);

        await buffer.mapAsync(GPUMapMode.READ);
        const raw = new Uint8Array(buffer.getMappedRange().slice(0));
        buffer.unmap();
        buffer.destroy();

        // Odstrani row padding in popravi BGRA -> RGBA (Mac vrne bgra8unorm)
        const isBGRA = this.canvasFormat === 'bgra8unorm';
        const tight = new Uint8Array(width * height * 4);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const src = y * bytesPerRow + x * 4;
                const dst = (y * width + x) * 4;
                if (isBGRA) {
                    tight[dst + 0] = raw[src + 2]; // R ← B
                    tight[dst + 1] = raw[src + 1]; // G ← G
                    tight[dst + 2] = raw[src + 0]; // B ← R
                    tight[dst + 3] = 255;
                } else {
                    tight[dst + 0] = raw[src + 0];
                    tight[dst + 1] = raw[src + 1];
                    tight[dst + 2] = raw[src + 2];
                    tight[dst + 3] = 255;
                }
            }
        }
        return tight;
    }

    // Preberi float32 teksturo (compositorTexture: rgba32float ali depthColorTexture: r32float)
    async _readFloat32(texture, channels) {
        const { width, height } = texture;
        const bytesPerPixel = channels * 4;
        const bytesPerRow = Math.ceil(width * bytesPerPixel / 256) * 256;

        const buffer = this.device.createBuffer({
            size: bytesPerRow * height,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const enc = this.device.createCommandEncoder();
        enc.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, { width, height });
        this.device.queue.submit([enc.finish()]);

        await buffer.mapAsync(GPUMapMode.READ);
        const raw = new Float32Array(buffer.getMappedRange().slice(0));
        buffer.unmap();
        buffer.destroy();

        // Odstrani row padding
        const stride = bytesPerRow / 4; // float32 elementi na vrstico (vključno s paddingom)
        const tight = new Float32Array(width * height * channels);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                for (let c = 0; c < channels; c++) {
                    tight[(y * width + x) * channels + c] = raw[y * stride + x * channels + c];
                }
            }
        }
        return tight;
    }

    // Shrani par kot tri PNG datoteke
    _savePair(noisy, clean, depth) {
        const idx = String(this.captureIndex).padStart(4, '0');

        // Šumna barvna slika (uint8 RGBA)
        this._saveUint8PNG(noisy.data, noisy.width, noisy.height, `noisy_${idx}.png`);

        // Čista barvna slika (float32 -> clamped uint8)
        this._saveFloat32PNG(clean.data, clean.width, clean.height, 4, `clean_${idx}.png`);

        // Globinska slika (float32 -> normaliziran uint8)
        this._saveDepthPNG(depth.data, depth.width, depth.height, `depth_${idx}.png`);
    }

    _saveUint8PNG(data, width, height, filename) {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        const img = new ImageData(new Uint8ClampedArray(data), width, height);
        ctx.putImageData(img, 0, 0);
        this._triggerDownload(canvas.toDataURL('image/png'), filename);
    }

    _saveFloat32PNG(data, width, height, channels, filename) {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        const img = ctx.createImageData(width, height);
        for (let i = 0; i < width * height; i++) {
            img.data[i * 4 + 0] = Math.round(Math.min(1, Math.max(0, data[i * channels + 0])) * 255);
            img.data[i * 4 + 1] = Math.round(Math.min(1, Math.max(0, data[i * channels + 1])) * 255);
            img.data[i * 4 + 2] = Math.round(Math.min(1, Math.max(0, data[i * channels + 2])) * 255);
            img.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
        this._triggerDownload(canvas.toDataURL('image/png'), filename);
    }

    _saveDepthPNG(data, width, height, filename) {
        // Normaliziramo globino na [0, 255] za shranjevanje kot grayscale PNG
        // Ozadje (depth=0) ostane črno
        let maxDepth = 0;
        for (let i = 0; i < data.length; i++) {
            if (data[i] > maxDepth) maxDepth = data[i];
        }

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        const img = ctx.createImageData(width, height);
        for (let i = 0; i < width * height; i++) {
            const v = maxDepth > 0 ? Math.round((data[i] / maxDepth) * 255) : 0;
            img.data[i * 4 + 0] = v;
            img.data[i * 4 + 1] = v;
            img.data[i * 4 + 2] = v;
            img.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
        this._triggerDownload(canvas.toDataURL('image/png'), filename);
    }

    _triggerDownload(dataURL, filename) {
        const a = document.createElement('a');
        a.href = dataURL;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
}
