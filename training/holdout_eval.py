"""
holdout_eval.py
Evalvacija na holdout test setu (slike ki niso bile v trening setu).

Struktura test_holdout/:
    noisy/   scene-7k_XXXX_noisy1.png   (RGBA, ze 512x512)
    clean/   scene-7k_XXXX_clean.png    (RGBA, ze 512x512)
    depth/   scene-7k_XXXX_depth.png    (RGBA, grayscale depth [0,255])

Evalvira:
    1. Baseline (brez denoiserja)
    2. RCNN U-Net + KPCN

Rezultate shrani v holdout_results.csv

Uporaba:
    python holdout_eval.py --data ../test_holdout --checkpoint checkpoints/best.pt
    python holdout_eval.py --data ../test_holdout --checkpoint checkpoints/best.pt --device mps
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model import RCNNUNet
from evaluate import psnr, ssim


# ── Pomocne funkcije ───────────────────────────────────────────────────────────

def load_rgb_float(path):
    """Naloži PNG kot float32 [H, W, 3] v [0, 1]. Ignorira alpha kanal."""
    return np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0

def load_depth_float(path):
    """Naloži depth PNG kot float32 [H, W] v [0, 1]. Depth je grayscale (R kanal)."""
    img = np.array(Image.open(path).convert('RGB'), dtype=np.float32)
    depth = img[:, :, 0] / 255.0  # R kanal = globina
    return depth

def float_to_uint8(img):
    """Pretvori float32 [0,1] v uint8 [0,255]."""
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def load_holdout_pairs(data_dir):
    """Najde vse pare v test_holdout/. Vrne seznam slovarjev."""
    data_dir = Path(data_dir)
    noisy_files = sorted((data_dir / 'noisy').glob('*_noisy1.png'))

    pairs = []
    for noisy_path in noisy_files:
        base = noisy_path.name.replace('_noisy1.png', '')
        clean_path = data_dir / 'clean' / f'{base}_clean.png'
        depth_path = data_dir / 'depth' / f'{base}_depth.png'

        if clean_path.exists() and depth_path.exists():
            pairs.append({'base': base, 'noisy': noisy_path,
                          'clean': clean_path, 'depth': depth_path})
        else:
            print(f'[SKIP] manjka clean ali depth za {base}')

    return pairs


def load_model(checkpoint_path, device):
    model = RCNNUNet(in_channels=4, out_channels=3)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    elif isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def infer(model, noisy_rgb, depth, device):
    """Inferenca na celi sliki. Vrne denoised float32 [H, W, 3]."""
    H, W = noisy_rgb.shape[:2]

    # Sestavi tenzor [1, 4, H, W]
    x = np.zeros((4, H, W), dtype=np.float32)
    x[0] = noisy_rgb[:, :, 0]
    x[1] = noisy_rgb[:, :, 1]
    x[2] = noisy_rgb[:, :, 2]
    x[3] = depth
    t = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        out, _ = model(t)  # [1, 3, H, W], h_0=0 (single frame)

    out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    return np.clip(out_np, 0, 1)


# ── Glavna funkcija ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       default='../test_holdout')
    parser.add_argument('--checkpoint', default='checkpoints/best.pt')
    parser.add_argument('--device',     default='cpu',
                        help='cpu | cuda | mps')
    parser.add_argument('--out',        default='holdout_results.csv')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f'Naprava: {device}')

    pairs = load_holdout_pairs(args.data)
    print(f'Najdenih parov: {len(pairs)}')
    if not pairs:
        print('Ni parov! Preveri pot do test_holdout/')
        return

    print(f'Nalagam model: {args.checkpoint}')
    model = load_model(args.checkpoint, device)

    results_baseline = []
    results_neural   = []
    times = []

    csv_lines = ['index,scene,psnr_baseline,ssim_baseline,psnr_neural,ssim_neural']

    for i, p in enumerate(pairs):
        noisy  = load_rgb_float(p['noisy'])
        clean  = load_rgb_float(p['clean'])
        depth  = load_depth_float(p['depth'])

        noisy_u8 = float_to_uint8(noisy)
        clean_u8 = float_to_uint8(clean)

        # Baseline metrike
        psnr_b = psnr(noisy_u8, clean_u8)
        ssim_b = ssim(noisy_u8, clean_u8)

        # Neural inferenca
        t0 = time.time()
        denoised = infer(model, noisy, depth, device)
        t1 = time.time()

        denoised_u8 = float_to_uint8(denoised)
        psnr_n = psnr(denoised_u8, clean_u8)
        ssim_n = ssim(denoised_u8, clean_u8)

        results_baseline.append((psnr_b, ssim_b))
        results_neural.append((psnr_n, ssim_n))
        times.append(t1 - t0)

        csv_lines.append(f"{p['base']},{p['base'].rsplit('_',1)[0]},"
                         f"{psnr_b:.4f},{ssim_b:.4f},{psnr_n:.4f},{ssim_n:.4f}")

        print(f'[{i+1:3d}/{len(pairs)}] {p["base"]:30s} '
              f'baseline={psnr_b:.2f}dB  neural={psnr_n:.2f}dB  '
              f'({t1-t0:.1f}s)')

    # Povprecja
    avg_psnr_b = np.mean([r[0] for r in results_baseline])
    avg_ssim_b = np.mean([r[1] for r in results_baseline])
    avg_psnr_n = np.mean([r[0] for r in results_neural])
    avg_ssim_n = np.mean([r[1] for r in results_neural])
    avg_time   = np.mean(times)

    print()
    print('=' * 60)
    print(f'HOLDOUT REZULTATI ({len(pairs)} parov, {len(set(p["base"].rsplit("_",1)[0] for p in pairs))} scen)')
    print('=' * 60)
    print(f'Baseline:         PSNR={avg_psnr_b:.2f} dB   SSIM={avg_ssim_b:.4f}')
    print(f'RCNN U-Net+KPCN:  PSNR={avg_psnr_n:.2f} dB   SSIM={avg_ssim_n:.4f}')
    print(f'Povprecni cas inferenc: {avg_time:.2f}s/sliko')
    print(f'Izboljsava: +{avg_psnr_n - avg_psnr_b:.2f} dB')

    # Shrani CSV
    out_path = Path(args.out)
    out_path.write_text('\n'.join(csv_lines) + '\n')
    print(f'\nRezultati shranjeni: {out_path}')


if __name__ == '__main__':
    main()
