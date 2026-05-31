"""
evaluate.py
Merjenje uspešnosti denoiserja s standardnimi metrikami PSNR in SSIM.

Uporablja pare (šumna, čista) slika zajete z DataCapture.js.

Metriki:
  PSNR - Peak Signal-to-Noise Ratio [dB], višji = boljše
  SSIM - Structural Similarity Index [0–1], višji = boljše

Uporaba:
  python evaluate.py --data ../training_data
  python evaluate.py --data ../training_data --denoised ../training_data/denoised
"""

import argparse
import os
import re
import numpy as np
from pathlib import Path
from PIL import Image


# ── Metrike ──────────────────────────────────────────────────────────────────

def mse(a, b):
    """Mean Squared Error med dvema slikama [0,1]."""
    return np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)


def psnr(a, b, max_val=255.0):
    """
    Peak Signal-to-Noise Ratio [dB].
    PSNR = 20 * log10(MAX / sqrt(MSE))
    Višji = manj šuma v primerjavi z referenco.
    """
    err = mse(a, b)
    if err == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(err))


def ssim(a, b, window_size=11, sigma=1.5):
    """
    Structural Similarity Index [0, 1].
    Meri strukturno podobnost - bolje korelira s človeško percepcijo kot PSNR.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    # Konstante za numerično stabilnost (Wang et al. 2004)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Gaussian kernel
    half = window_size // 2
    x = np.arange(-half, half + 1)
    kern_1d = np.exp(-x**2 / (2 * sigma**2))
    kern_1d /= kern_1d.sum()
    kern = np.outer(kern_1d, kern_1d)

    def convolve_channel(img):
        from scipy.ndimage import convolve
        return convolve(img, kern, mode='reflect')

    ssim_vals = []

    # Računamo po kanalih (RGB)
    for c in range(a.shape[2] if a.ndim == 3 else 1):
        ac = a[:, :, c] if a.ndim == 3 else a
        bc = b[:, :, c] if b.ndim == 3 else b

        mu_a = convolve_channel(ac)
        mu_b = convolve_channel(bc)
        mu_a2 = mu_a ** 2
        mu_b2 = mu_b ** 2
        mu_ab = mu_a * mu_b

        sigma_a2 = convolve_channel(ac ** 2) - mu_a2
        sigma_b2 = convolve_channel(bc ** 2) - mu_b2
        sigma_ab = convolve_channel(ac * bc) - mu_ab

        numerator   = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
        denominator = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)

        ssim_map = numerator / (denominator + 1e-8)
        ssim_vals.append(ssim_map.mean())

    return float(np.mean(ssim_vals))


# ── Nalaganje podatkov ────────────────────────────────────────────────────────

def load_pairs(data_dir):
    """
    Poišče vse pare (noisy_XXXX.png, clean_XXXX.png) v mapi.
    Pričakuje strukturo:
        data_dir/
          noisy/  noisy_0000.png ...
          clean/  clean_0000.png ...
    Ali flat:
        data_dir/  noisy_0000.png, clean_0000.png, ...
    """
    data_dir = Path(data_dir)

    # Poskusi organized subfolders najprej
    noisy_dir = data_dir / 'noisy'
    clean_dir = data_dir / 'clean'

    if noisy_dir.exists() and clean_dir.exists():
        # Organizirana struktura po organize.py: 0000.png, 0001.png ...
        noisy_files = sorted(noisy_dir.glob('*.png'))
    else:
        noisy_dir = data_dir
        clean_dir = data_dir
        noisy_files = sorted(data_dir.glob('noisy_*.png'))

    pairs = []
    for nf in noisy_files:
        # Podpira oba formata: '0000.png' in 'noisy_0000.png'
        stem = nf.stem  # '0000' ali 'noisy_0000'
        idx = re.search(r'(\d+)$', stem)
        if not idx:
            continue

        number = idx.group(1)
        cf = clean_dir / nf.name        # isti stem: 0000.png
        df = (data_dir / 'depth' / nf.name)

        if cf.exists():
            pairs.append({
                'index': number,
                'noisy': nf,
                'clean': cf,
                'depth': df if df.exists() else None,
            })

    return pairs


def load_image(path):
    """Naloži PNG kot numpy array uint8 [H, W, 3]."""
    img = Image.open(path).convert('RGB')
    return np.array(img)


# ── Evalvacija ────────────────────────────────────────────────────────────────

def evaluate_pairs(pairs, denoised_dir=None):
    """
    Za vsak par izračuna:
      - PSNR(noisy, clean) <- baseline brez denoiserja
      - SSIM(noisy, clean) <- baseline brez denoiserja
      - PSNR(denoised, clean) <- naša metoda (če je denoised_dir podan)
      - SSIM(denoised, clean) <- naša metoda
    """
    results = []

    for p in pairs:
        noisy = load_image(p['noisy'])
        clean = load_image(p['clean'])

        row = {
            'index': p['index'],
            'psnr_noisy':    psnr(noisy, clean),
            'ssim_noisy':    ssim(noisy, clean),
            'psnr_denoised': None,
            'ssim_denoised': None,
        }

        # Če imamo izhod denoiserja, ga primerjamo
        if denoised_dir:
            den_path = Path(denoised_dir) / f'denoised_{p["index"]}.png'
            if den_path.exists():
                denoised = load_image(den_path)
                row['psnr_denoised'] = psnr(denoised, clean)
                row['ssim_denoised'] = ssim(denoised, clean)

        results.append(row)

    return results


def print_results(results):
    """Izpiše tabelo rezultatov + povzetke."""
    has_denoised = any(r['psnr_denoised'] is not None for r in results)

    # Glava tabele
    if has_denoised:
        print(f"\n{'Index':>6}  {'PSNR noisy':>12}  {'SSIM noisy':>11}  {'PSNR denoised':>14}  {'SSIM denoised':>14}")
        print("-" * 65)
    else:
        print(f"\n{'Index':>6}  {'PSNR noisy':>12}  {'SSIM noisy':>11}")
        print("-" * 32)

    for r in results:
        if has_denoised:
            pd = f"{r['psnr_denoised']:>14.2f}" if r['psnr_denoised'] else f"{'—':>14}"
            sd = f"{r['ssim_denoised']:>14.4f}" if r['ssim_denoised'] else f"{'—':>14}"
            print(f"{r['index']:>6}  {r['psnr_noisy']:>12.2f}  {r['ssim_noisy']:>11.4f}  {pd}  {sd}")
        else:
            print(f"{r['index']:>6}  {r['psnr_noisy']:>12.2f}  {r['ssim_noisy']:>11.4f}")

    # Povzetki
    psnr_n = [r['psnr_noisy'] for r in results if np.isfinite(r['psnr_noisy'])]
    ssim_n = [r['ssim_noisy'] for r in results]

    print("-" * (65 if has_denoised else 32))
    if has_denoised:
        psnr_d = [r['psnr_denoised'] for r in results if r['psnr_denoised'] is not None]
        ssim_d = [r['ssim_denoised'] for r in results if r['ssim_denoised'] is not None]
        print(f"\n{'POVPREČJE':>6}  {np.mean(psnr_n):>12.2f}  {np.mean(ssim_n):>11.4f}  "
              f"{np.mean(psnr_d) if psnr_d else 0:>14.2f}  {np.mean(ssim_d) if ssim_d else 0:>14.4f}")
        print(f"{'STD DEV':>6}  {np.std(psnr_n):>12.2f}  {np.std(ssim_n):>11.4f}  "
              f"{np.std(psnr_d) if psnr_d else 0:>14.2f}  {np.std(ssim_d) if ssim_d else 0:>14.4f}")
        if psnr_d:
            delta = np.mean(psnr_d) - np.mean(psnr_n)
            print(f"\n  -> Izboljšanje PSNR: {delta:+.2f} dB")
            print(f"  -> Izboljšanje SSIM: {np.mean(ssim_d) - np.mean(ssim_n):+.4f}")
    else:
        print(f"\n{'POVPREČJE':>6}  {np.mean(psnr_n):>12.2f}  {np.mean(ssim_n):>11.4f}")
        print(f"{'STD DEV':>6}  {np.std(psnr_n):>12.2f}  {np.std(ssim_n):>11.4f}")

    print(f"\n  Skupaj parov: {len(results)}")


def save_csv(results, out_path):
    """Shrani rezultate kot CSV za poročilo."""
    import csv
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'psnr_noisy', 'ssim_noisy', 'psnr_denoised', 'ssim_denoised'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Rezultati shranjeni: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evalvacija denoiserja: PSNR + SSIM')
    parser.add_argument('--data', required=True,
                        help='Mapa s training pari (noisy_*.png + clean_*.png)')
    parser.add_argument('--denoised', default=None,
                        help='Mapa z izhodi denoiserja (denoised_*.png) — opcijsko')
    parser.add_argument('--csv', default=None,
                        help='Shrani rezultate kot CSV (npr. results.csv)')
    args = parser.parse_args()

    print(f"Iščem pare v: {args.data}")
    pairs = load_pairs(args.data)

    if not pairs:
        print("NAPAKA: Ni najdenih parov. Pričakujem noisy_XXXX.png + clean_XXXX.png")
        return

    print(f"Najdeno {len(pairs)} parov.")

    try:
        from scipy.ndimage import convolve  # noqa: preverimo odvisnost
    except ImportError:
        print("NAPAKA: pip install scipy")
        return

    results = evaluate_pairs(pairs, args.denoised)
    print_results(results)

    if args.csv:
        save_csv(results, args.csv)


if __name__ == '__main__':
    main()
