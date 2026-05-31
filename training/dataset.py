"""
dataset.py
PyTorch dataset za nalaganje (šumnih, čistih, globinskih) parov.

Vhod mreže: noisy RGB (3 kanali) + depth (1 kanal) = 4 kanali  [0, 1]
Cilj mreže: clean RGB (3 kanali)  [0, 1]

Augmentacija (med treningom):
  - content-aware crop 512x512: cropamo znotraj bounding boxa vsebine,
    da se izognemo večinoma črnim izrezom (ozadje Gaussian splatting scen)
  - naključna 90° rotacija (0°, 90°, 180°, 270°)
  - naključni horizontalni zrcalni odraz (50%)
Isti naključni parametri se aplicirajo na noisy, clean IN depth.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class DenoisingDataset(Dataset):
    """
    Naloži pare iz strukture:
        root/
          noisy/  0000.png ...
          clean/  0000.png ...
          depth/  0000.png ...

    crops_per_image: koliko naključnih izrezov vzamemo iz vsake slike na epoho.
    316 parov x 8 izrezov = 2528 vzorcev na epoho.
    """

    def __init__(self, root, crop_size=512, augment=True, crops_per_image=8):
        self.root = Path(root)
        self.crop_size = crop_size
        self.augment = augment
        self.crops_per_image = crops_per_image if augment else 1

        self.noisy_dir = self.root / 'noisy'
        self.clean_dir = self.root / 'clean'
        self.depth_dir = self.root / 'depth'

        # Poišči vse veljavne pare
        self.pairs = []
        for f in sorted(self.noisy_dir.glob('*.png')):
            c = self.clean_dir / f.name
            d = self.depth_dir / f.name
            if c.exists() and d.exists():
                self.pairs.append(f.stem)

        if not self.pairs:
            raise FileNotFoundError(f"Ni parov v {root}. Poženite najprej organize.py.")

        n_effective = len(self.pairs) * self.crops_per_image
        print(f"Dataset: {len(self.pairs)} parov × {self.crops_per_image} izrezov = {n_effective} vzorcev/epoho")

    def __len__(self):
        return len(self.pairs) * self.crops_per_image

    def __getitem__(self, idx):
        # Iz zaporednega indeksa izpelji kateri par in kateri izrez
        pair_idx = idx // self.crops_per_image
        stem = self.pairs[pair_idx]

        # Naloži vse tri slike
        noisy = np.array(Image.open(self.noisy_dir / f'{stem}.png').convert('RGB'), dtype=np.float32) / 255.0
        clean = np.array(Image.open(self.clean_dir / f'{stem}.png').convert('RGB'), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depth_dir / f'{stem}.png').convert('L'),   dtype=np.float32) / 255.0

        # depth: [H, W] -> [H, W, 1]
        depth = depth[:, :, np.newaxis]

        # Vhod mreže: noisy (3) + depth (1) = 4 kanali
        inp = np.concatenate([noisy, depth], axis=2)  # [H, W, 4]
        tgt = clean                                   # [H, W, 3]

        # Augmentacija - isti parametri za inp IN tgt
        if self.augment:
            inp, tgt = self._augment(inp, tgt)
        else:
            inp, tgt = self._center_crop(inp, tgt)

        # NumPy [H, W, C] -> PyTorch [C, H, W]
        inp = torch.from_numpy(inp.transpose(2, 0, 1))
        tgt = torch.from_numpy(tgt.transpose(2, 0, 1))

        return inp, tgt

    def _find_content_bbox(self, clean):
        """
        Poišče bounding box vsebine (non-black pikslov) v čisti sliki.
        Vrne (y0, y1, x0, x1) ali None če je slika popolnoma črna.

        Uporablja clean sliko (ne noisy), ker je zanesljivejša -
        noisy slika ima naključne svetle pike tudi na ozadju.
        """
        brightness = clean.mean(axis=2)          # [H, W], povprečje RGB
        mask = brightness > 0.05                 # 5/255 ≈ minimalna vsebina

        rows = np.any(mask, axis=1)              # katera vrstica ima vsebino
        cols = np.any(mask, axis=0)              # kateri stolpec ima vsebino

        if not rows.any():
            return None                          # celotna slika je črna

        y0 = int(np.where(rows)[0][0])
        y1 = int(np.where(rows)[0][-1])
        x0 = int(np.where(cols)[0][0])
        x1 = int(np.where(cols)[0][-1])

        return y0, y1, x0, x1

    def _augment(self, inp, tgt):
        """Content-aware crop + rejection sampling + rotacija + horizontalni zrcalni odraz.

        1. Poišče bbox vsebine + 1024px padding -> široko iskalno območje
        2. Poskusi do 20x najti crop kjer >= 2/3 pikslov ni črnih
        3. Če noben ne zadosti -> vzame crop z največ vsebine (best_frac)
        """
        H, W = inp.shape[:2]
        cs = self.crop_size

        # ── Iskalno območje: bbox vsebine + velik padding ─────────────────
        bbox = self._find_content_bbox(tgt[:, :, :3])
        if bbox is not None:
            y0, y1, x0, x1 = bbox
            pad = 1024
            sy0 = max(0, y0 - pad)
            sy1 = min(H, y1 + pad)
            sx0 = max(0, x0 - pad)
            sx1 = min(W, x1 + pad)
        else:
            sy0, sy1, sx0, sx1 = 0, H, 0, W   # fallback: cela slika

        # Zagotovi da je iskalno območje vsaj crop_size
        if sy1 - sy0 < cs: sy0, sy1 = 0, H
        if sx1 - sx0 < cs: sx0, sx1 = 0, W

        # ── Rejection sampling: do 20 poskusov, ohrani najboljši ─────────
        best_y, best_x, best_frac = 0, 0, -1.0

        for _ in range(20):
            y = random.randint(sy0, sy1 - cs)
            x = random.randint(sx0, sx1 - cs)

            patch = tgt[y:y+cs, x:x+cs]
            frac_content = float((patch.mean(axis=2) > 0.05).mean())

            if frac_content > best_frac:
                best_y, best_x, best_frac = y, x, frac_content

            if frac_content >= 2/3:   # >= 2/3 vsebine sprejmemo takoj
                break
        # Če noben poskus ni dosegel praga -> vzamemo tistega z največ vsebine

        inp = inp[best_y:best_y+cs, best_x:best_x+cs]
        tgt = tgt[best_y:best_y+cs, best_x:best_x+cs]

        # ── 90° rotacija (0°, 90°, 180°, 270°) ───────────────────────────
        k = random.randint(0, 3)
        if k > 0:
            inp = np.rot90(inp, k=k).copy()
            tgt = np.rot90(tgt, k=k).copy()

        # ── Horizontalni zrcalni odraz (50%) ──────────────────────────────
        if random.random() > 0.5:
            inp = inp[:, ::-1].copy()
            tgt = tgt[:, ::-1].copy()

        return inp, tgt

    def _center_crop(self, inp, tgt):
        """Center crop na crop_size x crop_size."""
        H, W = inp.shape[:2]
        cs = self.crop_size
        y = max(0, (H - cs) // 2)
        x = max(0, (W - cs) // 2)
        return inp[y:y+cs, x:x+cs], tgt[y:y+cs, x:x+cs]
