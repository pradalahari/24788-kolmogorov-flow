"""
models/unet.py — U-Net for next-step vorticity prediction.

Standard encoder-decoder with skip connections.
4 levels: 64 → 128 → 256 → 512
Input/Output: [B, 1, 160, 160]
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive (Conv2d → BatchNorm → ReLU) blocks."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downsample: MaxPool → DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upsample → concatenate skip → DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatch due to odd spatial dimensions
        diffH = skip.size(2) - x.size(2)
        diffW = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diffW // 2, diffW - diffW // 2,
                                   diffH // 2, diffH - diffH // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net with 4 encoder/decoder levels.

    Input:  [B, 1, 160, 160]
    Output: [B, 1, 160, 160]

    Channel progression: 1 → 64 → 128 → 256 → 512 → 256 → 128 → 64 → 1
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()
        c = base_channels  # 64

        self.inc = DoubleConv(in_channels, c)          # 1 → 64
        self.down1 = Down(c, c * 2)                     # 64 → 128
        self.down2 = Down(c * 2, c * 4)                 # 128 → 256
        self.down3 = Down(c * 4, c * 8)                 # 256 → 512

        self.up1 = Up(c * 8, c * 4)                     # 512 → 256
        self.up2 = Up(c * 4, c * 2)                     # 256 → 128
        self.up3 = Up(c * 2, c)                         # 128 → 64

        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)  # 64 → 1

    def forward(self, x):
        x1 = self.inc(x)       # [B, 64, 160, 160]
        x2 = self.down1(x1)    # [B, 128, 80, 80]
        x3 = self.down2(x2)    # [B, 256, 40, 40]
        x4 = self.down3(x3)    # [B, 512, 20, 20]

        x = self.up1(x4, x3)   # [B, 256, 40, 40]
        x = self.up2(x, x2)    # [B, 128, 80, 80]
        x = self.up3(x, x1)    # [B, 64, 160, 160]

        return self.outc(x)     # [B, 1, 160, 160]


if __name__ == "__main__":
    # Quick shape test
    model = UNet()
    x = torch.randn(2, 1, 160, 160)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
