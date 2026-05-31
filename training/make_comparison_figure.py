"""
make_comparison_figure.py
Ustvari comparison figuro za porocilo: noisy | bilateral | neural | clean

Zbere N najzanimivejsih slik (po razliki PSNR noisy vs bilateral) in
za vsako ustvari pasico cez full crop polja.

Uporaba (po koncu treninga in evalvacije):
  python make_comparison_figure.py \\
      --data   ../training_data \\
      --output ../report/figures/comparison.png

Opcijsko:
  --index 0042     prikaze tocno dolocen par (po stevilki)
  --n_rows 3       koliko parov prikazemo (privzeto: 3)
  --crop 512       velikost prikazanega okna (privzeto: 512x512)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import load_pairs, psnr


# ── Pomocne funkcije ──────────────────────────────────────────────────────────

def load_rgb(path):
    """Naloži RGB PNG."""
    return np.array(Image.open(path).convert('RGB'), dtype=np.uint8)


def find_content_center(clean_rgb):
    """
    Poisce center vsebine (non-black pikslov) v čisti sliki.
    Vrne (cy, cx) - koordinati centra.
    """
    brightness = clean_rgb.mean(axis=2)
    mask = brightness > 12  # > 12/255 ~ vsebina (ne ozadje)

    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]

    if len(rows) == 0:
        # Fallback: sredisce slike
        return clean_rgb.shape[0] // 2, clean_rgb.shape[1] // 2

    cy = int((rows[0] + rows[-1]) / 2)
    cx = int((cols[0] + cols[-1]) / 2)
    return cy, cx


def center_crop(img, cy, cx, size):
    """
    Izreze kvadratni crop okoli (cy, cx).
    """
    H, W = img.shape[:2]
    half = size // 2

    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(H, y0 + size)
    x1 = min(W, x0 + size)

    # Prilagodi ce smo blizu roba
    if y1 - y0 < size:
        y0 = max(0, y1 - size)
    if x1 - x0 < size:
        x0 = max(0, x1 - size)

    return img[y0:y0+size, x0:x0+size]


def add_label(img_array, text, font_size=20):
    """Doda belo besedilo z crno senco v spodnji del slike."""
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    W, H = img.size
    # Preprost font (PIL privzet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Besedilo na dno slike
    x, y = 8, H - font_size - 8

    # Crna senca
    for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1),(0,-1),(0,1),(-1,0),(1,0)]:
        draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)

    # Belo besedilo
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return np.array(img)


def psnr_label(img_u8, ref_u8):
    """Vrne PSNR kot string za labelo."""
    p = psnr(img_u8, ref_u8)
    return f"{p:.1f} dB"


def make_row(noisy_path, bilateral_path, neural_path, clean_path,
             crop_size=512, font_size=20):
    """
    Ustvari eno vrstico primerjave: noisy | bilateral | neural | clean
    Vsi cropani na isti crop_size x crop_size regijo.
    Vrne numpy array [crop_size, 4*crop_size + 3*3, 3] (3px razmik med stolpci)
    """
    noisy    = load_rgb(noisy_path)
    clean    = load_rgb(clean_path)

    # Najdi center vsebine iz čiste slike
    cy, cx = find_content_center(clean)

    # Skupni crop za vse metode
    noisy_c    = center_crop(noisy, cy, cx, crop_size)
    clean_c    = center_crop(clean, cy, cx, crop_size)

    # Dodamo labele z vrednostmi PSNR
    noisy_l = add_label(noisy_c,   f"Noisy  {psnr_label(noisy_c, clean_c)}", font_size)
    clean_l = add_label(clean_c,    "Clean (ref)",                            font_size)

    panels = [noisy_l]

    # Bilateral (opcijsko - ce ni available, prikazi placeholder)
    if bilateral_path and Path(bilateral_path).exists():
        bilat   = load_rgb(bilateral_path)
        bilat_c = center_crop(bilat, cy, cx, crop_size)
        bilat_l = add_label(bilat_c, f"Bilateral  {psnr_label(bilat_c, clean_c)}", font_size)
        panels.append(bilat_l)
    else:
        placeholder = np.full((crop_size, crop_size, 3), 40, dtype=np.uint8)
        panels.append(add_label(placeholder, "Bilateral (N/A)", font_size))

    # Neural (opcijsko)
    if neural_path and Path(neural_path).exists():
        neural   = load_rgb(neural_path)
        neural_c = center_crop(neural, cy, cx, crop_size)
        neural_l = add_label(neural_c, f"Neural  {psnr_label(neural_c, clean_c)}", font_size)
        panels.append(neural_l)
    else:
        placeholder = np.full((crop_size, crop_size, 3), 40, dtype=np.uint8)
        panels.append(add_label(placeholder, "Neural (N/A)", font_size))

    panels.append(clean_l)

    # Zlepimo horizontalno z 3px sivim razmikom
    sep = np.full((crop_size, 3, 3), 128, dtype=np.uint8)
    row_parts = []
    for k, panel in enumerate(panels):
        row_parts.append(panel[:crop_size, :crop_size])
        if k < len(panels) - 1:
            row_parts.append(sep)

    return np.concatenate(row_parts, axis=1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Ustvari comparison figuro za porocilo'
    )
    parser.add_argument('--data',    required=True,
                        help='Mapa s pari: noisy/, clean/, bilateral/, neural/')
    parser.add_argument('--output',  default='../report/figures/comparison.png',
                        help='Pot za shranjevanje figure (privzeto: ../report/figures/comparison.png)')
    parser.add_argument('--index',   default=None,
                        help='Tocno dolocen par po stevilki (npr. 0042)')
    parser.add_argument('--n_rows',  type=int, default=3,
                        help='Stevilo prikazanih parov (privzeto: 3)')
    parser.add_argument('--crop',    type=int, default=512,
                        help='Velikost prikazanega okna [px] (privzeto: 512)')
    args = parser.parse_args()

    data_dir = Path(args.data)
    pairs    = load_pairs(data_dir)

    if not pairs:
        print("NAPAKA: Ni parov.")
        return

    bilat_dir  = data_dir / 'bilateral'
    neural_dir = data_dir / 'neural'

    # Izberi katere pare prikazati
    if args.index:
        selected = [p for p in pairs if p['index'] == args.index]
        if not selected:
            print(f"NAPAKA: Par {args.index} ni najden.")
            return
    else:
        # Izberi tiste z najvecjo razliko bilateral - noisy (najbol vizualno zanimive)
        scored = []
        for p in pairs:
            noisy_u8 = np.array(Image.open(p['noisy']).convert('RGB'), dtype=np.uint8)
            clean_u8 = np.array(Image.open(p['clean']).convert('RGB'), dtype=np.uint8)
            psnr_n = psnr(noisy_u8, clean_u8)

            bilat_path = bilat_dir / f"denoised_{p['index']}.png"
            if bilat_path.exists():
                bilat_u8 = np.array(Image.open(bilat_path).convert('RGB'), dtype=np.uint8)
                psnr_b = psnr(bilat_u8, clean_u8)
                delta = psnr_b - psnr_n
            else:
                delta = 0.0

            scored.append((delta, p))

        # Sortiramo po padajoci razliki: najvec izboljsave = najbolj zanimiva slika
        scored.sort(key=lambda x: -x[0])
        selected = [p for _, p in scored[:args.n_rows]]

    print(f"Ustvarjam figuro z {len(selected)} pari...")

    rows = []
    for p in selected:
        idx = p['index']
        bilat_path  = bilat_dir  / f"denoised_{idx}.png"
        neural_path = neural_dir / f"denoised_{idx}.png"

        row = make_row(
            noisy_path=p['noisy'],
            bilateral_path=bilat_path if bilat_path.exists()  else None,
            neural_path=neural_path   if neural_path.exists() else None,
            clean_path=p['clean'],
            crop_size=args.crop,
        )
        rows.append(row)
        print(f"  Par {idx}: {row.shape}")

    # Zlepimo vrstice vertikalno z 6px razmikom
    sep_h = np.full((6, rows[0].shape[1], 3), 180, dtype=np.uint8)
    col_parts = []
    for k, row in enumerate(rows):
        col_parts.append(row)
        if k < len(rows) - 1:
            col_parts.append(sep_h)

    final = np.concatenate(col_parts, axis=0)

    # Shrani
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final).save(out_path)
    print(f"\nFigura shranjena: {out_path}  ({final.shape[1]}x{final.shape[0]} px)")


if __name__ == '__main__':
    main()
