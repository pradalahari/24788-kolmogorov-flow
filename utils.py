"""
utils.py — Metrics, normalization, and plotting utilities for Kolmogorov Flow project.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────

class Normalizer:
    """Simple mean/std normalizer. Compute stats on training data, apply to all."""

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def encode(self, x):
        return (x - self.mean) / (self.std + 1e-8)

    def decode(self, x):
        return x * (self.std + 1e-8) + self.mean

    def save(self, path):
        torch.save({"mean": self.mean, "std": self.std}, path)

    @staticmethod
    def load(path):
        d = torch.load(path, weights_only=True)
        return Normalizer(d["mean"], d["std"])


def compute_normalizer(h5_file, train_indices, num_samples_for_stats=20):
    """
    Compute mean and std from a subset of training samples.
    Loads only a few samples to avoid memory issues.
    """
    import h5py

    with h5py.File(h5_file, "r") as f:
        u = f["valid"]["u"]
        subset = train_indices[:num_samples_for_stats]
        frames = []
        for idx in subset:
            # shape: [200, 160, 160] — take all timesteps for this sample
            sample = torch.tensor(u[idx], dtype=torch.float32)
            frames.append(sample)
        frames = torch.cat(frames, dim=0)  # [num_samples * 200, 160, 160]
        mean = frames.mean().item()
        std = frames.std().item()

    return Normalizer(mean, std)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def nrmse_field(pred, target, eps=1e-8):
    """
    Normalized RMSE for a single field.
    pred, target: shape [B, 1, H, W] or [1, H, W] or [H, W]
    Returns scalar NRMSE averaged over the batch.
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    B = pred.shape[0]
    # Flatten spatial dims
    pred_flat = pred.reshape(B, -1)
    target_flat = target.reshape(B, -1)
    nx_ny = pred_flat.shape[1]

    # ||u||_2 = sqrt(1/(nx*ny) * sum |u(i,j)|^2)
    rmse = torch.sqrt(torch.mean((pred_flat - target_flat) ** 2, dim=1))
    norm = torch.sqrt(torch.mean(target_flat ** 2, dim=1)) + eps

    return (rmse / norm).mean().item()


def rollout_nrmse(model, initial_frame, ground_truth_frames, normalizer, device, num_steps=100):
    """
    Autoregressive rollout and compute NRMSE at each step.

    Args:
        model: trained model
        initial_frame: shape [1, 160, 160] (raw, unnormalized)
        ground_truth_frames: shape [T, 160, 160] (raw, unnormalized) — frames at t+1, t+2, ...
        normalizer: Normalizer object
        device: torch device
        num_steps: number of rollout steps

    Returns:
        nrmse_per_step: list of NRMSE values at each rollout step
    """
    model.eval()
    num_steps = min(num_steps, ground_truth_frames.shape[0])

    current = initial_frame.unsqueeze(0).to(device)  # [1, 1, 160, 160]
    nrmse_per_step = []

    with torch.no_grad():
        for t in range(num_steps):
            # Normalize input, predict, denormalize output
            inp = normalizer.encode(current)
            pred_norm = model(inp)
            pred = normalizer.decode(pred_norm)

            gt = ground_truth_frames[t].unsqueeze(0).unsqueeze(0).to(device)
            nrmse_per_step.append(nrmse_field(pred, gt))

            # Feed prediction back as next input
            current = pred

    return nrmse_per_step


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_training_curves(train_losses, val_losses, title="Training Curves", save_path=None):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_rollout_comparison(nrmse_unet, nrmse_fno, save_path=None):
    """Plot rollout NRMSE vs timestep for both models."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(nrmse_unet, label="U-Net", linewidth=2)
    ax.plot(nrmse_fno, label="FNO", linewidth=2)
    ax.set_xlabel("Rollout Timestep")
    ax.set_ylabel("NRMSE")
    ax.set_title("Autoregressive Rollout: NRMSE vs Timestep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_vorticity_comparison(pred, target, timestep, model_name="Model", save_path=None):
    """
    Side-by-side vorticity field visualization.
    pred, target: shape [160, 160] numpy arrays
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin = min(target.min(), pred.min())
    vmax = max(target.max(), pred.max())

    im0 = axes[0].imshow(target, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Ground Truth (t={timestep})")
    axes[0].axis("off")

    im1 = axes[1].imshow(pred, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{model_name} Prediction (t={timestep})")
    axes[1].axis("off")

    diff = np.abs(pred - target)
    im2 = axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("Absolute Error")
    axes[2].axis("off")

    fig.colorbar(im0, ax=axes[:2], shrink=0.8, label="Vorticity")
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label="|Error|")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ──────────────────────────────────────────────
# Log-TKE Spectrum (Bonus)
# ──────────────────────────────────────────────

def compute_energy_spectrum(field):
    """
    Compute radially averaged energy spectrum from a 2D vorticity field.
    field: shape [H, W] numpy array
    Returns: (wavenumbers, energy_spectrum)
    """
    H, W = field.shape
    # 2D FFT
    fft2 = np.fft.fft2(field)
    power = np.abs(fft2) ** 2 / (H * W)

    # Wavenumber grids
    kx = np.fft.fftfreq(W, d=1.0 / W)
    ky = np.fft.fftfreq(H, d=1.0 / H)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    # Radial binning
    k_max = int(min(H, W) / 2)
    k_bins = np.arange(0.5, k_max + 0.5, 1.0)
    energy = np.zeros(len(k_bins))

    for i, k in enumerate(k_bins):
        mask = (K >= k - 0.5) & (K < k + 0.5)
        energy[i] = power[mask].sum()

    return k_bins, energy


def plot_tke_comparison(gt_field, pred_unet, pred_fno, save_path=None):
    """
    Plot log-TKE spectra for ground truth vs both model predictions.
    All inputs: shape [160, 160] numpy arrays
    """
    k, E_gt = compute_energy_spectrum(gt_field)
    _, E_unet = compute_energy_spectrum(pred_unet)
    _, E_fno = compute_energy_spectrum(pred_fno)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.loglog(k, E_gt, "k-", label="Ground Truth", linewidth=2)
    ax.loglog(k, E_unet, "b--", label="U-Net", linewidth=1.5)
    ax.loglog(k, E_fno, "r--", label="FNO", linewidth=1.5)
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("Energy E(k)")
    ax.set_title("Log Turbulent Kinetic Energy Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
