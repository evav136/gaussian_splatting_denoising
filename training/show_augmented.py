"""
show_augmented.py
Prikaže N primerov augmentiranih izrezov iz dataseta.
Za vsak primer: noisy crop | clean crop | depth crop

Uporaba:
  python show_augmented.py --data ../training_data --n 20
"""

import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import DenoisingDataset


def show_samples(data_dir, n=20, crop_size=512, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    ds = DenoisingDataset(data_dir, crop_size=crop_size, augment=True, crops_per_image=1)

    # Vzorčimo n naključnih indeksov
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    cols = 3  # noisy | clean | depth
    rows = n
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(f'Content-aware augmented crops ({crop_size}×{crop_size})', fontsize=14, y=1.01)

    col_titles = ['Noisy (vhod)', 'Clean (cilj)', 'Depth']
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=11, fontweight='bold')

    for row, idx in enumerate(indices):
        inp, tgt = ds[idx]  # inp: [4,H,W], tgt: [3,H,W]

        noisy_np = inp[:3].permute(1, 2, 0).numpy()   # [H,W,3]
        depth_np = inp[3].numpy()                       # [H,W]
        clean_np = tgt.permute(1, 2, 0).numpy()        # [H,W,3]

        axes[row, 0].imshow(np.clip(noisy_np, 0, 1))
        axes[row, 1].imshow(np.clip(clean_np, 0, 1))
        axes[row, 2].imshow(depth_np, cmap='plasma', vmin=0, vmax=1)

        # Statistika za debug: povprečna svetlost clean
        mean_brightness = clean_np.mean()
        axes[row, 0].set_ylabel(f'#{idx}\nμ={mean_brightness:.3f}', fontsize=7, rotation=0, labelpad=40, va='center')

        for c in range(cols):
            axes[row, c].axis('off')

    plt.tight_layout()
    out = Path(data_dir).parent / 'training' / 'augmented_examples.png'
    plt.savefig(out, dpi=80, bbox_inches='tight')
    print(f"Shranjeno: {out}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',      default='../training_data')
    parser.add_argument('--n',         type=int, default=20)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    show_samples(args.data, n=args.n, crop_size=args.crop_size, seed=args.seed)
