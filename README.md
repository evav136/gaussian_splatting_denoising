# Gaussian Splatting Render Denoising

Seminar project for the NRG course, University of Ljubljana FRI, 2026.  
**Eva Vidic** — ev0743@student.uni-lj.si

## What this is

A WebGPU Gaussian splatting renderer extended with two denoisers for stochastic transparency noise:

1. **Bilateral filter** — real-time depth-guided 7×7 spatial filter, runs as a WGSL compute shader
2. **RCNN U-Net + KPCN** — learned neural denoiser trained on 316 noisy/clean image pairs, exported to ONNX and integrated in the browser as a demo

## Running the renderer

Requires Chrome (WebGPU support).

```bash
cd webgpu-splatting-dithering-nrg
python3 -m http.server 8080
```

Open `http://localhost:8080`, then drag and drop a `.splat` file onto the canvas.  
The bilateral filter is active by default. The neural denoiser can be triggered via the GUI button ("Uporabi neural denoiser").

## Directory structure

```
webgpu-splatting-dithering-nrg/
  main.js               main render loop
  SplatRenderer.js/wgsl Gaussian splatting renderer
  Denoiser.js/wgsl      bilateral filter (WGSL compute shader)
  NeuralDenoiser.js     ONNX Runtime Web inference
  DataCapture.js        training data capture module
  denoiser.onnx         exported neural denoiser model (3.7 MB)
  denoiser.onnx.data    model weights

training/
  model.py              RCNN U-Net + KPCN architecture
  train.py              training script
  dataset.py            dataset and augmentation
  evaluate.py           PSNR/SSIM evaluation
  neural_eval.py        full dataset evaluation
  bilateral_eval.py     bilateral filter evaluation
  holdout_eval.py       holdout evaluation on unseen scenes
  export_onnx.py        ONNX export script
  checkpoints/best.pt   best model checkpoint (epoch 65)
  holdout_results.csv   results on 100 holdout pairs (8 unseen scenes)
  neural_results.csv    results on 316 training pairs
```

## Training (requires GPU and training data)

```bash
cd training
python train.py --data ../training_data --device cuda
```

## Evaluation

```bash
cd training
# Neural denoiser
python neural_eval.py --data ../training_data --checkpoint checkpoints/best.pt --device cuda

# Bilateral filter
python bilateral_eval.py --data ../training_data
```
