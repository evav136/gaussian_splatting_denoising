"""
model.py
RCNN U-Net denoiser s KPCN izhodnim slojem.

Inspiriran z:
  [2] Chaitanya et al. SIGGRAPH 2017 - RCNN arhitektura za temporalni denoising
  [3] Bako et al. SIGGRAPH 2017      - Kernel-Predicting Convolutional Networks (KPCN)

Arhitektura:
  Encoder: 3 nivoji z MaxPool, vsak nivo ima RCNN blok
  Bottleneck: RCNN blok na najnižji resoluciji
  Decoder: 3 nivoji z bilinearnim upsamplingom + skip connections

RCNN blok (Recurrent CNN):
  Kombinira konvolucijo trenutnega vhoda s konvolucijo prejšnjega skritega stanja.
  Omogoča časovno koherenco čez okvirje (recurrence).

KPCN izhod (namesto direktne napovedi RGB):
  Mreža napove 7x7=49 uteži za vsak piksel.
  Izhod = utežena vsota 7x7 sosedov iz šumne slike.
  Prednost: mreža ne more "halucinirati" barv - meša samo obstoječe piksle.
  Robovi so ostrejši kot pri direktni napovedi (ni težnje k povprečenju).

Vhod:  [B, 4, H, W]   noisy RGB (3) + depth (1)
Izhod: [B, 3, H, W]   clean RGB (3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RCNNBlock(nn.Module):
    """
    Recurrent CNN blok iz [2] Chaitanya et al.

    Za vsak okvir:
      hidden_new = ReLU( conv_input(x) + conv_hidden(hidden_prev) )

    Conv na vhodu + conv na skritemu stanju = 10 vrstic kode.
    Skrito stanje se prenaša med okvirji -> mreža "pomni" pretekle okvirje.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_input  = nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1)
        self.conv_hidden = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, hidden=None):
        """
        x:      [B, in_channels, H, W]
        hidden: [B, out_channels, H, W] ali None (prvi okvir)
        vrne:   (izhod, novo_skrito_stanje)
        """
        out = self.conv_input(x)
        if hidden is not None:
            out = out + self.conv_hidden(hidden)
        out = F.relu(self.bn(out))
        return out, out  # skrito stanje = izhod tega bloka


class EncoderBlock(nn.Module):
    """Encoder nivo: RCNN + MaxPool."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.rcnn = RCNNBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, hidden=None):
        feat, hidden_new = self.rcnn(x, hidden)
        pooled = self.pool(feat)
        return pooled, feat, hidden_new
        # pooled -> naslednji encoder nivo
        # feat -> skip connection za decoder
        # hidden_new -> skrito stanje za naslednji okvir


class DecoderBlock(nn.Module):
    """Decoder nivo: upsample + skip connection + konvolucija."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RCNNUNet(nn.Module):
    """
    RCNN U-Net s KPCN izhodnim slojem.

    Velikosti filtrov (manjši kot [2] za hitrost):
      Encoder: 4 -> 32 -> 64 -> 128
      Bottleneck: 128 -> 256
      Decoder: 256 -> 128 -> 64 ->32
      Izhod: 32 -> 49 (7x7 kernel uteži) -> aplicira na šumno sliko -> 3 (RGB)

    Skupaj ~1.8M parametrov.
    """

    def __init__(self, in_channels=4, out_channels=3, base_filters=32, kernel_size=7):
        super().__init__()
        f = base_filters   # 32
        self.kernel_size = kernel_size
        K2 = kernel_size * kernel_size   # 49

        # Encoder (3 nivoji)
        self.enc1 = EncoderBlock(in_channels, f)      # 4 -> 32
        self.enc2 = EncoderBlock(f,           f*2)    # 32 -> 64
        self.enc3 = EncoderBlock(f*2,         f*4)    # 64 -> 128

        # Bottleneck
        self.bottleneck = RCNNBlock(f*4, f*8)         # 128 -> 256

        # Decoder (3 nivoji)
        self.dec3 = DecoderBlock(f*8, f*4, f*4)       # 256+128 -> 128
        self.dec2 = DecoderBlock(f*4, f*2, f*2)       # 128+64 -> 64
        self.dec1 = DecoderBlock(f*2, f,   f)         # 64+32 -> 32

        # KPCN izhodni sloj: 32 -> K² uteži (brez aktivacije - softmax pride v _apply_kernel)
        self.output_conv = nn.Conv2d(f, K2, kernel_size=1)

    def _apply_kernel(self, noisy, weights):
        """
        KPCN: aplicira napovedane kernele na šumno sliko.

        noisy:   [B, 3, H, W]  - šumni RGB kanali iz vhoda
        weights: [B, K2, H, W] - surove uteži iz output_conv

        Postopek:
          1. Softmax normalizacija -> uteži seštejejo v 1 za vsak piksel
          2. F.unfold izvleče KxK sosede za vsak piksel v eno matriko
          3. Utežena vsota sosedov -> čist piksel

        Opomba: F.unfold ni podprt na Apple MPS (pada na CPU).
                Na CUDA (šolski strežnik) je to nativna, hitra operacija.

        Vrne: [B, 3, H, W] - denoisana slika v [0, 1]
        """
        B, C, H, W = noisy.shape
        K  = self.kernel_size
        K2 = K * K

        # 1. Normalizacija utezi: softmax čez K2 dimenzijo
        weights = F.softmax(weights, dim=1)            # [B, K2, H, W]

        # 2. Izvlecemo KxK sosede za vsak piksel
        #    F.unfold: [B, 3, H, W] -> [B, 3*K2, H*W]
        patches = F.unfold(noisy, kernel_size=K, padding=K // 2)

        # 3. Preuredimo za množenje z utežmi
        #    patches: [B, 3, K2, H*W]
        #    weights: [B, 1, K2, H*W]
        patches = patches.view(B, C, K2, H * W)
        w = weights.view(B, 1, K2, H * W)

        # 4. Utežena vsota: [B, 3, H*W] -> [B, 3, H, W]
        out = (patches * w).sum(dim=2).view(B, C, H, W)

        return torch.clamp(out, 0.0, 1.0)

    def forward(self, x, hidden_states=None):
        """
        x:             [B, 4, H, W]   šumni okvir (RGB) + globina
        hidden_states: slovar s skritimi stanji iz prejšnjega okvirja (ali None)

        Vrne: (denoised [B,3,H,W], new_hidden_states)
        """
        if hidden_states is None:
            hidden_states = {}

        # Shranimo šumni RGB za kernel aplikacijo na koncu
        noisy_rgb = x[:, :3, :, :]   # [B, 3, H, W]

        # Encoder
        x1, skip1, h1 = self.enc1(x,  hidden_states.get('enc1'))
        x2, skip2, h2 = self.enc2(x1, hidden_states.get('enc2'))
        x3, skip3, h3 = self.enc3(x2, hidden_states.get('enc3'))

        # Bottleneck
        b, hb = self.bottleneck(x3, hidden_states.get('bottleneck'))

        # Decoder (skip connections iz encoderja)
        d3 = self.dec3(b,  skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        # KPCN: napovedi kernel uteži, apliciraj na šumno sliko
        kernel_weights = self.output_conv(d1)                # [B, 49, H, W]
        out = self._apply_kernel(noisy_rgb, kernel_weights)  # [B, 3, H, W]

        new_hidden = {
            'enc1': h1,
            'enc2': h2,
            'enc3': h3,
            'bottleneck': hb,
        }

        return out, new_hidden


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri skupaj:    {total:,}")
    print(f"Parametri (trening): {trainable:,}")
    return trainable


if __name__ == '__main__':
    model = RCNNUNet()
    count_parameters(model)

    # Test forward pass
    x = torch.randn(2, 4, 128, 128)
    out, hidden = model(x)
    print(f"Vhod:  {x.shape}")
    print(f"Izhod: {out.shape}  (pričakovano: [2, 3, 128, 128])")
    print(f"Izhod min/max: {out.min():.3f} / {out.max():.3f}  (mora biti v [0,1])")
    print(f"Skrita stanja: {list(hidden.keys())}")

    # Drugi okvir z rekurenco
    x2 = torch.randn(2, 4, 128, 128)
    out2, hidden2 = model(x2, hidden)
    print(f"Izhod 2 (z rekurenco): {out2.shape}")
