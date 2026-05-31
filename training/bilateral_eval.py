"""
bilateral_eval.py
Evaluacija prostorskega bilateral filtra na vseh 316 parih.

Algoritem:
  Za vsak piksel p:
    - Pogledamo 7x7 sosede q
    - Prostorska utez:    w_s = exp(-||p-q||^2 / (2*sigma_s^2))      <- 2D razdalja na zaslonu
    - Globinska utez:     w_d = exp(-(D_p - D_q)^2 / (2*sigma_d^2)) <- 3D razdalja v globino
      Posebni primeri:    D_p==0 in D_q==0  -> w_d = 1  (ozadje prosto meša)
                          drugace D==0       -> w_d = 0  (ozadje/ospredje ne mešamo)
    - Firefly suppresija: omeji luminanco soseda na 4*L_center + 0.01
    - Filtered(p) = sum(w_s * w_d * sosed) / sum(w_s * w_d)

Shrani rezultate kot training_data/bilateral/denoised_XXXX.png
Izpiše PSNR/SSIM za baseline (brez filtra) in bilateral (s filtrom).

Uporaba:
  python bilateral_eval.py --data ../training_data
  python bilateral_eval.py --data ../training_data --sigma_s 1.5 --sigma_d 0.1
  python bilateral_eval.py --data ../training_data --csv bilateral_results.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Uvozi skupne funkcije iz evaluate.py (psnr, ssim, load_pairs)
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import psnr, ssim, load_pairs


# ── Bilateral filter ──────────────────────────────────────────────────────────

def bilateral_filter(noisy_rgb, depth, sigma_s=1.5, sigma_d=0.1, K=7):
    """
    7x7 prostorski bilateral filter z globinskim ustavljanjem robov.

    noisy_rgb: [H, W, 3] float32 v [0, 1]
    depth:     [H, W]    float32 v [0, 1], 0 = ozadje / zavrzeni fragment
    sigma_s:   sirina prostorske Gaussove uteži [piksli]
    sigma_d:   sirina globinske Gaussove uteži [enote normalizirane globine 0-1]
    K:         velikost filtrirnega okna (7 -> 7x7)

    Vrne: [H, W, 3] float32 v [0, 1] - filtirana slika
    """
    H, W = noisy_rgb.shape[:2]
    pad = K // 2

    # Paddanje: 'reflect' za barve (gladko na robovih), 'edge' za globino
    rgb_pad   = np.pad(noisy_rgb, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    depth_pad = np.pad(depth,     ((pad, pad), (pad, pad)),          mode='edge')

    # Luminanca za firefly suppresijo: povprecje RGB
    lum_pad    = rgb_pad.mean(axis=2)   # [H+2p, W+2p]
    lum_center = noisy_rgb.mean(axis=2) # [H, W]

    # Akumulatorji
    acc        = np.zeros((H, W, 3), dtype=np.float64)
    weight_sum = np.zeros((H, W),    dtype=np.float64)

    # Globina centralnega piksla (brez paddinga)
    depth_center = depth  # [H, W]

    # Iteriramo cez vse odmike v KxK oknu
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):

            # 1. Prostorska utez: odvisna samo od 2D razdalje na zaslonu
            w_s = float(np.exp(-(dy ** 2 + dx ** 2) / (2.0 * sigma_s ** 2)))

            # Koordinate soseda v padded sliki
            ny = pad + dy
            nx = pad + dx

            # Sosednje vrednosti
            neighbor_rgb   = rgb_pad  [ny:ny + H, nx:nx + W, :]  # [H, W, 3]
            neighbor_depth = depth_pad[ny:ny + H, nx:nx + W]     # [H, W]
            neighbor_lum   = lum_pad  [ny:ny + H, nx:nx + W]     # [H, W]

            # 2. Firefly suppresija: omeji luminanco soseda na 4*L_center + 0.01
            max_lum      = 4.0 * lum_center + 0.01               # [H, W]
            clamp_factor = np.where(
                neighbor_lum > max_lum,
                max_lum / (neighbor_lum + 1e-8),
                1.0
            )  # [H, W] - faktor s katerim skaliramo barve soseda
            clamped_rgb = neighbor_rgb * clamp_factor[:, :, np.newaxis]

            # 3. Globinska utez
            #    Oba ospredna piksla -> exp padec
            #    Oba ozadna piksla   -> prosto mesanje (w_d = 1)
            #    Mesano               -> brez mesanja (w_d = 0)
            both_fg = (depth_center > 0) & (neighbor_depth > 0)
            both_bg = (depth_center == 0) & (neighbor_depth == 0)

            depth_diff_sq = (depth_center - neighbor_depth) ** 2
            w_d_fg = np.exp(-depth_diff_sq / (2.0 * sigma_d ** 2))

            w_d = np.where(both_fg, w_d_fg,
                  np.where(both_bg, 1.0, 0.0))  # [H, W]

            # 4. Skupna utez in akumulacija
            w = w_s * w_d  # [H, W]

            acc        += clamped_rgb * w[:, :, np.newaxis]
            weight_sum += w

    # Normalizacija (zascita pred delitvijo z 0 - ne bi se smelo zgoditi)
    weight_sum_safe = np.maximum(weight_sum, 1e-8)
    result = acc / weight_sum_safe[:, :, np.newaxis]

    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ── Nalaganje slik ────────────────────────────────────────────────────────────

def load_rgb_float(path):
    """Naloži PNG kot float32 [H, W, 3] v [0, 1]."""
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def load_depth_float(path):
    """Naloži globinsko PNG kot float32 [H, W] v [0, 1]."""
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float32) / 255.0


def float_to_uint8(img_float):
    """Pretvori [0,1] float v uint8 [0, 255]."""
    return (np.clip(img_float, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ── Metrike ───────────────────────────────────────────────────────────────────

def compute_metrics(img, ref):
    """Izracuna PSNR in SSIM med uint8 slikama [H, W, 3]."""
    return psnr(img, ref), ssim(img, ref)


# ── Evalvacija ────────────────────────────────────────────────────────────────

def run_evaluation(data_dir, sigma_s=1.5, sigma_d=0.1, K=7, csv_path=None):
    """
    Za vsak par:
      1. Naloži noisy, clean, depth
      2. Filtrira z bilateral filtrom
      3. Shrani denoised sliko
      4. Izracuna PSNR/SSIM za baseline in bilateral

    Shrani denoised slike v data_dir/bilateral/denoised_XXXX.png
    (format ki ga pricakuje evaluate.py --denoised bilateral/)
    """
    data_dir = Path(data_dir)
    pairs    = load_pairs(data_dir)

    if not pairs:
        print("NAPAKA: Ni najdenih parov. Preverite strukturo mape.")
        return

    out_dir = data_dir / 'bilateral'
    out_dir.mkdir(exist_ok=True)

    print(f"\nBilateral filter evalvacija")
    print(f"  Slike:   {len(pairs)} parov")
    print(f"  sigma_s: {sigma_s}  sigma_d: {sigma_d}  K: {K}x{K}")
    print(f"  Izhod:   {out_dir}/")
    print()

    results_noisy    = []
    results_bilateral = []
    times = []

    for i, p in enumerate(pairs):
        # Nalozi slike
        noisy = load_rgb_float(p['noisy'])     # [H, W, 3] float32 [0,1]
        clean_f = load_rgb_float(p['clean'])   # [H, W, 3] float32 [0,1]

        if p['depth'] is None:
            # Brez globine: bilateral degradira na cistoprostorski filter
            depth = np.zeros(noisy.shape[:2], dtype=np.float32)
        else:
            depth = load_depth_float(p['depth'])  # [H, W] float32 [0,1]

        # Bilateral filter
        t0 = time.time()
        filtered = bilateral_filter(noisy, depth, sigma_s=sigma_s, sigma_d=sigma_d, K=K)
        dt = time.time() - t0
        times.append(dt)

        # Pretvori v uint8 za metrike in shranjevanje
        noisy_u8    = float_to_uint8(noisy)
        clean_u8    = float_to_uint8(clean_f)
        filtered_u8 = float_to_uint8(filtered)

        # Metrike
        psnr_n, ssim_n = compute_metrics(noisy_u8,    clean_u8)
        psnr_b, ssim_b = compute_metrics(filtered_u8, clean_u8)

        results_noisy.append((psnr_n, ssim_n))
        results_bilateral.append((psnr_b, ssim_b))

        # Shrani denoised (format za evaluate.py: denoised_XXXX.png)
        out_path = out_dir / f"denoised_{p['index']}.png"
        Image.fromarray(filtered_u8).save(out_path)

        # Napredek vsak 10. par (316 parov)
        if (i + 1) % 10 == 0 or i == 0 or i == len(pairs) - 1:
            print(f"  [{i+1:3d}/{len(pairs)}]  {p['noisy'].name}  "
                  f"PSNR: {psnr_n:.2f} -> {psnr_b:.2f} dB  "
                  f"SSIM: {ssim_n:.4f} -> {ssim_b:.4f}  "
                  f"({dt:.1f}s)")

    # Povzetki
    arr_n = np.array(results_noisy)
    arr_b = np.array(results_bilateral)

    mean_psnr_n, mean_ssim_n = arr_n[:, 0].mean(), arr_n[:, 1].mean()
    mean_psnr_b, mean_ssim_b = arr_b[:, 0].mean(), arr_b[:, 1].mean()

    d_psnr = mean_psnr_b - mean_psnr_n
    d_ssim = mean_ssim_b - mean_ssim_n

    print()
    print("=" * 62)
    print(f"  {'Metoda':<22}  {'PSNR [dB]':>10}  {'SSIM':>10}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}")
    print(f"  {'Brez denoiserja':<22}  {mean_psnr_n:>10.2f}  {mean_ssim_n:>10.4f}")
    print(f"  {'Bilateral filter':<22}  {mean_psnr_b:>10.2f}  {mean_ssim_b:>10.4f}")
    print(f"  {'Izboljsanje':<22}  {d_psnr:>+10.2f}  {d_ssim:>+10.4f}")
    print("=" * 62)
    print(f"  Povprecni cas na sliko: {np.mean(times):.2f}s "
          f"(skupaj {sum(times)/60:.1f} min za {len(pairs)} slik)")
    print()

    # Shrani CSV
    if csv_path:
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'psnr_noisy', 'ssim_noisy', 'psnr_bilateral', 'ssim_bilateral'])
            for p, (pn, sn), (pb, sb) in zip(pairs, results_noisy, results_bilateral):
                writer.writerow([p['index'], f'{pn:.4f}', f'{sn:.4f}', f'{pb:.4f}', f'{sb:.4f}'])
        print(f"  Rezultati shranjeni: {csv_path}")

    return {
        'psnr_noisy':    mean_psnr_n,
        'ssim_noisy':    mean_ssim_n,
        'psnr_bilateral': mean_psnr_b,
        'ssim_bilateral': mean_ssim_b,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evalvacija bilateral denoiserja: PSNR + SSIM na vseh 316 parih'
    )
    parser.add_argument('--data',    required=True,
                        help='Mapa s pari: noisy/, clean/, depth/')
    parser.add_argument('--sigma_s', type=float, default=1.5,
                        help='Prostorska sirina Gaussove uteži [piksli] (privzeto: 1.5)')
    parser.add_argument('--sigma_d', type=float, default=0.1,
                        help='Globinska sirina Gaussove uteži [0-1 norm.] (privzeto: 0.1)')
    parser.add_argument('--kernel',  type=int,   default=7,
                        help='Velikost filtrirnega okna (privzeto: 7)')
    parser.add_argument('--csv',     default=None,
                        help='Shrani rezultate kot CSV (npr. bilateral_results.csv)')
    args = parser.parse_args()

    try:
        from scipy.ndimage import convolve  # noqa: potrebno za ssim v evaluate.py
    except ImportError:
        print("NAPAKA: scipy ni namesccen. Pozeni: pip install scipy")
        return

    run_evaluation(
        data_dir=args.data,
        sigma_s=args.sigma_s,
        sigma_d=args.sigma_d,
        K=args.kernel,
        csv_path=args.csv,
    )


if __name__ == '__main__':
    main()
