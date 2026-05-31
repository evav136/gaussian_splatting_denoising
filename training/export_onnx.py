"""
export_onnx.py
Exportira RCNN U-Net denoiser v ONNX format za uporabo v brskalniku
z onnxruntime-web.

Trik: ker hidden_states niso bile nikoli trenirane (brez sekvenčnih podatkov),
model zavijemo v wrapper ki ignorira recurrente in exportira samo
x -> denoised_rgb.

Uporaba:
  python export_onnx.py --checkpoint checkpoints/best.pt --output ../webgpu-splatting-dithering-nrg/denoiser.onnx
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from model import RCNNUNet


class DenoiserWrapper(nn.Module):
    """
    Wrapper ki skrije recurrentni del modela.
    Vhod:  [B, 4, H, W]  noisy RGB + depth
    Izhod: [B, 3, H, W]  denoised RGB v [0, 1]

    hidden_states = None (conv_hidden uteži niso bile koristno istrenirane
    ker pri treningu ni bilo sekvenčnih parov -> model deluje kot navaden CNN).
    """
    def __init__(self, model: RCNNUNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.model(x, hidden_states=None)
        return out  # [B, 3, H, W]


def export(checkpoint_path: str, output_path: str, tile_size: int = 512):
    device = 'cpu'  # ONNX export vedno na CPUju

    # Nalozi model
    print(f"Nalagam checkpoint: {checkpoint_path}")
    base_model = RCNNUNet(in_channels=4, out_channels=3)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    elif isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    base_model.load_state_dict(state)
    base_model.eval()

    model = DenoiserWrapper(base_model)
    model.eval()

    params = sum(p.numel() for p in base_model.parameters())
    print(f"Parametri modela: {params:,}")

    # Testni vhod (tile_size x tile_size)
    dummy = torch.randn(1, 4, tile_size, tile_size)

    # Preverimo forward pass
    with torch.no_grad():
        out = model(dummy)
    print(f"Test forward: {dummy.shape} -> {out.shape}  "
          f"min={out.min():.3f} max={out.max():.3f}")

    # Export v ONNX
    print(f"\nExportiram v: {output_path}")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=['noisy_depth'],   # [B, 4, H, W]: RGB + depth
        output_names=['denoised'],     # [B, 3, H, W]: clean RGB
        dynamic_axes={
            # Dinamicne dimenzije: batch + prostorske osi
            # -> model deluje na poljubni velikosti tile-a
            'noisy_depth': {0: 'batch', 2: 'height', 3: 'width'},
            'denoised':    {0: 'batch', 2: 'height', 3: 'width'},
        },
        opset_version=17,   # opset 17: podpira F.unfold (Im2Col), LayerNorm, itd.
        do_constant_folding=True,  # optimizacija: zloži konstantne izraze
        dynamo=False,       # stari stabilen exporter: pravilno vgradi vse utezi
    )

    # Preveri velikost
    import os
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Velikost ONNX datoteke: {size_mb:.1f} MB")

    # Opcijsko: preveri z onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        inp = dummy.numpy()
        result = sess.run(['denoised'], {'noisy_depth': inp})[0]
        print(f"OnnxRuntime test: {inp.shape} -> {result.shape}  OK")
    except ImportError:
        print("onnxruntime ni namesccen (pip install onnxruntime) - preskoceno")

    print("\nExport uspesen!")
    print(f"Kopiraj {output_path} v mapo webgpu-splatting-dithering-nrg/")


def main():
    parser = argparse.ArgumentParser(description='Export denoiser v ONNX')
    parser.add_argument('--checkpoint', default='checkpoints/best.pt',
                        help='Pot do best.pt')
    parser.add_argument('--output', default='../webgpu-splatting-dithering-nrg/denoiser.onnx',
                        help='Izhodni .onnx fajl')
    parser.add_argument('--tile', type=int, default=512,
                        help='Velikost testnega tile-a (privzeto 512)')
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.tile)


if __name__ == '__main__':
    main()
