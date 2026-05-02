"""
models/fno.py — Fourier Neural Operator for next-step vorticity prediction.

Based on: Li et al., "Fourier Neural Operator for Parametric PDEs," ICLR 2021.
https://arxiv.org/abs/2010.08895

Architecture:
  1. Lifting layer: 1 channel → hidden channels
  2. N Fourier layers (default 4), each:
     - FFT → multiply by learnable weights (keep lowest k_max modes) → iFFT
     - Add pointwise linear transform (1×1 conv)
     - GELU activation
  3. Projection layer: hidden channels → 1

Input/Output: [B, 1, 160, 160]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D Spectral convolution layer.
    Applies a learnable linear transform in Fourier space,
    keeping only the lowest k_max modes in each dimension.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # number of Fourier modes to keep (height)
        self.modes2 = modes2  # number of Fourier modes to keep (width)

        scale = 1.0 / (in_channels * out_channels)
        # Complex-valued learnable weights for two quadrants of the spectrum
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        """Complex multiplication: (B, C_in, H, W) x (C_in, C_out, H, W) → (B, C_out, H, W)"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B = x.shape[0]

        # 2D real FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            B, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Top-left corner (low freq modes)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        # Bottom-left corner (negative freq modes in dim 1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # Inverse FFT back to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FourierLayer(nn.Module):
    """Single Fourier layer: spectral conv + pointwise linear + GELU."""

    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral_conv(x) + self.pointwise(x)))


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator.

    Args:
        modes1, modes2: number of Fourier modes to keep per dimension (default 16)
        width: hidden channel width (default 32)
        n_layers: number of Fourier layers (default 4)
        in_channels: input channels (default 1)
        out_channels: output channels (default 1)
    """

    def __init__(self, modes1=16, modes2=16, width=32, n_layers=4,
                 in_channels=1, out_channels=1):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers

        # Lifting: project input channels to hidden width
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)

        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer(width, modes1, modes2)
            for _ in range(n_layers)
        ])

        # Projection: hidden width → output channels
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, kernel_size=1),
        )

    def forward(self, x):
        # x: [B, 1, 160, 160]
        x = self.lift(x)  # [B, width, 160, 160]

        for layer in self.fourier_layers:
            x = layer(x)

        x = self.project(x)  # [B, 1, 160, 160]
        return x


if __name__ == "__main__":
    # Quick shape test
    model = FNO2d(modes1=16, modes2=16, width=32, n_layers=4)
    x = torch.randn(2, 1, 160, 160)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
