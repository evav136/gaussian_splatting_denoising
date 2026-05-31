"""
neural_eval.py
Evalvacija RCNN U-Net + KPCN denoiserja na vseh 316 parih.

Postopek:
  1. Naloži model (best.pt ali zadnji checkpoint)
  2. Za vsak par (noisy + depth):
       - Pozene inferenco cez celo sliko (ne crop)
       - Shrani denoised_XXXX.png v training_data/neural/
  3. Izracuna PSNR/SSIM za baseline in neural
  4. Izpise tabelo + primerja z bilateral filtrom ce je csv na voljo

Opomba:
  - Inferenca na celi sliki porabi vec GPU pomnilnika kot trening (ki je delal na 512x512 cropsih)
  - Slike so do 1478x2400 px -> se poveci batch dim ne bo slo
  - Resitev: tile_size inferenca (obdelamo v 512x512 blokih z overlapping)
    ALI cela slika naenkrat ce GPU ima dovolj pomnilnika (H100 ima 80 GB, mac ne)

Uporaba na Macu (CPU):
  python neural_eval.py --data ../training_data --checkpoint checkpoints/best.pt --device cpu

Uporaba na CUDA:
  python neural_eval.py --data ../training_data --checkpoint checkpoints/best.pt --device cuda

Uporaba s tile procesiranjem (za manjse GPU/CPU):
  python neural_eval.py --data ../training_data --checkpoint checkpoints/best.pt --tile 512 --overlap 64
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Uvozi modela in metrik iz lokalnih skript
sys.path.insert(0, str(Path(__file__).parent))
from model import RCNNUNet
from evaluate import psnr, ssim, load_pairs


# ── Nalaganje ─────────────────────────────────────────────────────────────────

def load_rgb_float(path):
    """Naloži PNG kot float32 [H, W, 3] v [0, 1]."""
    return np.array(Image.open(path).convert('RGB'),  dtype=np.float32) / 255.0


def load_depth_float(path):
    """Naloži globinsko PNG kot float32 [H, W] v [0, 1]."""
    return np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0


def float_to_uint8(img_float):
    """Pretvori [0,1] float v uint8 [0, 255]."""
    return (np.clip(img_float, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def load_model(checkpoint_path, device):
    """Naloži utezi in postavi model v eval mode."""
    model = RCNNUNet(in_channels=4, out_channels=3)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Checkpoint je slovar z 'model', 'optimizer', 'epoch' ...
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    elif isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ── Inferenca ─────────────────────────────────────────────────────────────────

def infer_full(model, noisy_rgb, depth, device):
    """
    Inferenca na celi sliki naenkrat.
    Deluje na H100 (80 GB) - za Mac ce je slika prevelika se sesuje.

    noisy_rgb: [H, W, 3] float32 [0,1]
    depth:     [H, W]    float32 [0,1]
    Vrne:      [H, W, 3] float32 [0,1]
    """
    H, W = noisy_rgb.shape[:2]

    # Zdruzi noisy + depth v 4-kanalski tenzor
    depth_3d = depth[:, :, np.newaxis]                   # [H, W, 1]
    inp = np.concatenate([noisy_rgb, depth_3d], axis=2)  # [H, W, 4]

    # NumPy -> PyTorch [1, 4, H, W]
    x = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        out, _ = model(x)  # [1, 3, H, W]

    # PyTorch -> NumPy [H, W, 3]
    result = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def infer_tiled(model, noisy_rgb, depth, device, tile_size=512, overlap=64):
    """
    Inferenca v prekrivajocem se tile-ih.
    Resuje problem velikih slik na majhnih GPU/CPU.

    tile_size: velikost tile-a v pikslih
    overlap:   prekrivanje med sosednimi tile-i (za gladke prehode)
    """
    H, W = noisy_rgb.shape[:2]
    step = tile_size - overlap

    # Inicializiramo izhodno sliko in akumulator za povprecenje na robovih
    output = np.zeros((H, W, 3), dtype=np.float64)
    counts = np.zeros((H, W),    dtype=np.float64)

    # Gaussova utez za mehke robove tile-ov (da ni vidnih stikov)
    y_w = np.hanning(tile_size).reshape(-1, 1)
    x_w = np.hanning(tile_size).reshape(1, -1)
    blend_mask = (y_w * x_w)  # [tile_size, tile_size]

    ys = list(range(0, H - tile_size + 1, step))
    xs = list(range(0, W - tile_size + 1, step))

    # Zagotovimo da zajamemo skrajni desni/spodnji rob
    if ys[-1] + tile_size < H: ys.append(H - tile_size)
    if xs[-1] + tile_size < W: xs.append(W - tile_size)

    for y0 in ys:
        for x0 in xs:
            y1 = y0 + tile_size
            x1 = x0 + tile_size

            # Izrezi tile
            tile_rgb   = noisy_rgb[y0:y1, x0:x1]   # [T, T, 3]
            tile_depth = depth    [y0:y1, x0:x1]    # [T, T]

            # Inferenca na tile-u
            tile_out = infer_full(model, tile_rgb, tile_depth, device)

            # Akumuliramo z mehkim maskiranjem
            output[y0:y1, x0:x1] += tile_out * blend_mask[:, :, np.newaxis]
            counts[y0:y1, x0:x1] += blend_mask

    # Normalizacija (deli z vsoto uteži)
    counts_safe = np.maximum(counts, 1e-8)
    result = output / counts_safe[:, :, np.newaxis]
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ── Evalvacija ────────────────────────────────────────────────────────────────

def run_evaluation(data_dir, checkpoint_path, device='cpu',
                   tile_size=None, overlap=64, csv_path=None):
    """
    Evalvacija RCNN U-Net na vseh parih.

    tile_size: ce je None -> cela slika naenkrat; drugace tiled inferenca
    """
    data_dir = Path(data_dir)
    pairs    = load_pairs(data_dir)

    if not pairs:
        print("NAPAKA: Ni parov. Preverite strukturo mape.")
        return

    # Nalozi model
    print(f"\nNalagam model: {checkpoint_path}")
    model = load_model(checkpoint_path, device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parametri: {params:,}")
    print(f"Naprava:   {device}")
    if tile_size:
        print(f"Nacin:     tiled {tile_size}x{tile_size} z overlappom {overlap}px")
    else:
        print(f"Nacin:     cela slika naenkrat")
    print()

    out_dir = data_dir / 'neural'
    out_dir.mkdir(exist_ok=True)

    results_noisy  = []
    results_neural = []
    times = []

    for i, p in enumerate(pairs):
        out_path = out_dir / f"denoised_{p['index']}.png"

        # Resume: ce denoised slika ze obstaja, preskoči inferenco
        # Samo nalozi obstoječi rezultat in izracunaj metrike.
        if out_path.exists():
            noisy   = load_rgb_float(p['noisy'])
            clean_f = load_rgb_float(p['clean'])
            noisy_u8    = float_to_uint8(noisy)
            clean_u8    = float_to_uint8(clean_f)
            denoised_u8 = np.array(Image.open(out_path).convert('RGB'))
            psnr_n, ssim_n = psnr(noisy_u8, clean_u8),    ssim(noisy_u8, clean_u8)
            psnr_d, ssim_d = psnr(denoised_u8, clean_u8), ssim(denoised_u8, clean_u8)
            results_noisy.append((psnr_n, ssim_n))
            results_neural.append((psnr_d, ssim_d))
            times.append(0.0)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1:3d}/{len(pairs)}]  (ze obdelano, preskoceno)")
            continue

        # Nalozi slike
        noisy   = load_rgb_float(p['noisy'])
        clean_f = load_rgb_float(p['clean'])

        if p['depth'] is None:
            depth = np.zeros(noisy.shape[:2], dtype=np.float32)
        else:
            depth = load_depth_float(p['depth'])

        # Inferenca
        t0 = time.time()
        if tile_size:
            denoised = infer_tiled(model, noisy, depth, device, tile_size, overlap)
        else:
            denoised = infer_full(model, noisy, depth, device)
        dt = time.time() - t0
        times.append(dt)

        # Pretvori v uint8 za metrike
        noisy_u8    = float_to_uint8(noisy)
        clean_u8    = float_to_uint8(clean_f)
        denoised_u8 = float_to_uint8(denoised)

        psnr_n, ssim_n = psnr(noisy_u8, clean_u8),    ssim(noisy_u8, clean_u8)
        psnr_d, ssim_d = psnr(denoised_u8, clean_u8), ssim(denoised_u8, clean_u8)

        results_noisy.append((psnr_n, ssim_n))
        results_neural.append((psnr_d, ssim_d))

        # Shrani (format za evaluate.py)
        Image.fromarray(denoised_u8).save(out_path)

        # Napredek
        if (i + 1) % 10 == 0 or i == 0 or i == len(pairs) - 1:
            print(f"  [{i+1:3d}/{len(pairs)}]  {p['noisy'].name}  "
                  f"PSNR: {psnr_n:.2f} -> {psnr_d:.2f} dB  "
                  f"SSIM: {ssim_n:.4f} -> {ssim_d:.4f}  "
                  f"({dt:.1f}s)")

    # Povzetki
    arr_n = np.array(results_noisy)
    arr_d = np.array(results_neural)

    mean_psnr_n, mean_ssim_n = arr_n[:, 0].mean(), arr_n[:, 1].mean()
    mean_psnr_d, mean_ssim_d = arr_d[:, 0].mean(), arr_d[:, 1].mean()

    print()
    print("=" * 62)
    print(f"  {'Metoda':<22}  {'PSNR [dB]':>10}  {'SSIM':>10}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}")
    print(f"  {'Brez denoiserja':<22}  {mean_psnr_n:>10.2f}  {mean_ssim_n:>10.4f}")
    print(f"  {'RCNN U-Net + KPCN':<22}  {mean_psnr_d:>10.2f}  {mean_ssim_d:>10.4f}")
    delta_p = mean_psnr_d - mean_psnr_n
    delta_s = mean_ssim_d - mean_ssim_n
    print(f"  {'Izboljsanje':<22}  {delta_p:>+10.2f}  {delta_s:>+10.4f}")
    print("=" * 62)
    print(f"  Povprecni cas na sliko: {np.mean(times):.2f}s "
          f"(skupaj {sum(times)/60:.1f} min za {len(pairs)} slik)")
    print()

    # Shrani CSV
    if csv_path:
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'psnr_noisy', 'ssim_noisy', 'psnr_neural', 'ssim_neural'])
            for p, (pn, sn), (pd, sd) in zip(pairs, results_noisy, results_neural):
                writer.writerow([p['index'], f'{pn:.4f}', f'{sn:.4f}', f'{pd:.4f}', f'{sd:.4f}'])
        print(f"  Rezultati shranjeni: {csv_path}")

    return {
        'psnr_noisy':  mean_psnr_n,
        'ssim_noisy':  mean_ssim_n,
        'psnr_neural': mean_psnr_d,
        'ssim_neural': mean_ssim_d,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evalvacija RCNN U-Net + KPCN denoiserja: PSNR + SSIM'
    )
    parser.add_argument('--data',       required=True,
                        help='Mapa s pari: noisy/, clean/, depth/')
    parser.add_argument('--checkpoint', required=True,
                        help='Pot do .pt datoteke (npr. checkpoints/best.pt)')
    parser.add_argument('--device',     default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Naprava za inferenco (privzeto: cpu)')
    parser.add_argument('--tile',       type=int, default=None,
                        help='Tiled inferenca: velikost tile-a v pikslih (privzeto: ne -> cela slika)')
    parser.add_argument('--overlap',    type=int, default=64,
                        help='Prekrivanje med tile-i v pikslih (privzeto: 64)')
    parser.add_argument('--csv',        default=None,
                        help='Shrani rezultate kot CSV (npr. neural_results.csv)')
    args = parser.parse_args()

    try:
        from scipy.ndimage import convolve  # noqa: za ssim
    except ImportError:
        print("NAPAKA: pip install scipy")
        return

    run_evaluation(
        data_dir=args.data,
        checkpoint_path=args.checkpoint,
        device=args.device,
        tile_size=args.tile,
        overlap=args.overlap,
        csv_path=args.csv,
    )


if __name__ == '__main__':
    main()
