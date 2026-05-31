"""
train.py
Trening RCNN U-Net denoiserja.

Izgubna funkcija (iz [2] Chaitanya et al.):
  L = L1_spatial + λ * L1_gradient
  L1_spatial:  |output - clean|       ... piksel-za-pikslom
  L1_gradient: |∇output - ∇clean|     ... ohrani robove

Optimizer: Adam (kot v [2])
Learning rate: 1e-4 z exponential decay

Uporaba:
  python train.py --data ../training_data --epochs 100
  python train.py --data ../training_data --epochs 100 --device mps   (Mac Apple Silicon)
  python train.py --data ../training_data --epochs 100 --device cuda  (šolski strežnik)
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import DenoisingDataset
from model import RCNNUNet, count_parameters


# ── Izgubna funkcija ──────────────────────────────────────────────────────────

def gradient_loss(pred, target):
    """
    L1 izguba na gradientih (robovih) slike.
    Iz [2] Chaitanya et al. - kaznuje razlike v strukturi robov.
    ∇x = razlika med sosednjimi piksli v x smeri
    ∇y = razlika med sosednjimi piksli v y smeri
    """
    def grad(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]   # [B, C, H, W-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]   # [B, C, H-1, W]
        return dx, dy

    pred_dx, pred_dy = grad(pred)
    tgt_dx,  tgt_dy  = grad(target)

    return torch.mean(torch.abs(pred_dx - tgt_dx)) + \
           torch.mean(torch.abs(pred_dy - tgt_dy))


def total_loss(pred, target, lambda_grad=0.1):
    """
    Skupna izguba = L1_spatial + λ * L1_gradient
    λ=0.1: gradient izguba prispeva 10% (robovi so pomembni a ne dominantni)
    """
    l1 = torch.mean(torch.abs(pred - target))
    lg = gradient_loss(pred, target)
    return l1 + lambda_grad * lg, l1.item(), lg.item()


# ── Trening ───────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss_sum = 0.0
    total_l1 = 0.0
    total_lg = 0.0

    for inp, tgt in loader:
        inp = inp.to(device)   # [B, 4, H, W]
        tgt = tgt.to(device)   # [B, 3, H, W]

        optimizer.zero_grad()

        # Forward pass (brez rekurence med treningom — vsak batch je neodvisen)
        # Med inferenco bi prenašali hidden states čez okvirje
        pred, _ = model(inp, hidden_states=None)

        loss, l1, lg = total_loss(pred, tgt)
        loss.backward()

        # Gradient clipping — prepreči eksplodirajoče gradiante
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss_sum += loss.item()
        total_l1       += l1
        total_lg       += lg

    n = len(loader)
    return total_loss_sum / n, total_l1 / n, total_lg / n


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total = 0.0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        pred, _ = model(inp)
        loss, _, _ = total_loss(pred, tgt)
        total += loss.item()
    return total / len(loader)


def psnr_batch(pred, target):
    """PSNR za batch — za monitoring med treningom."""
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.mean().item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Trening RCNN U-Net denoiserja')
    parser.add_argument('--data',        default='../training_data', help='Mapa s training pari')
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--crop_size',   type=int,   default=128)
    parser.add_argument('--val_split',   type=float, default=0.1,  help='Delež validacijskega seta')
    parser.add_argument('--device',      default='auto',           help='auto | cpu | cuda | mps')
    parser.add_argument('--save_dir',    default='checkpoints',    help='Mapa za shranjevanje modela')
    parser.add_argument('--resume',      default=None,             help='Pot do checkpointa za nadaljevanje')
    args = parser.parse_args()

    # Izbira naprave
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Naprava: {device}")

    # Dataset
    full_dataset = DenoisingDataset(args.data, crop_size=args.crop_size, augment=True)
    n_val   = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # Validacijski set brez augmentacije
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"Train: {n_train} parov | Val: {n_val} parov")

    # Model
    model = RCNNUNet(in_channels=4, out_channels=3, base_filters=32).to(device)
    count_parameters(model)

    # Optimizer + scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # Checkpoint
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    start_epoch = 0
    best_val = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val = ckpt.get('best_val', float('inf'))
        print(f"Nadaljujem od epohe {start_epoch}")

    # Trening zanka
    print(f"\nZačenjam trening: {args.epochs} epoh, batch={args.batch_size}, lr={args.lr}")
    print(f"{'Epoha':>6}  {'Train loss':>11}  {'L1':>8}  {'Grad':>8}  {'Val loss':>10}  {'LR':>8}  {'Čas':>6}")
    print("-" * 70)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, l1, lg = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(f"{epoch+1:>6}  {train_loss:>11.5f}  {l1:>8.5f}  {lg:>8.5f}  {val_loss:>10.5f}  {lr:>8.6f}  {elapsed:>5.1f}s")

        # Shrani najboljši model
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val':  best_val,
                'args':      vars(args),
            }, save_dir / 'best.pt')
            print(f"         Shranjen best.pt (val_loss={best_val:.5f})")

        # Shrani zadnji checkpoint vsakih 10 epoh
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val':  best_val,
            }, save_dir / f'epoch_{epoch+1:04d}.pt')

    print(f"\nTrening končan. Najboljši val_loss: {best_val:.5f}")
    print(f"Model shranjen: {save_dir}/best.pt")
    print(f"\nNaslednji korak: python export_onnx.py --checkpoint {save_dir}/best.pt")


if __name__ == '__main__':
    main()
