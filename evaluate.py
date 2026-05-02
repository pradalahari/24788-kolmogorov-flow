"""
evaluate.py — Evaluation script for trained models.

Computes:
  1. One-step NRMSE on validation set
  2. Autoregressive rollout NRMSE over time
  3. Log-TKE spectrum comparison (bonus)

Usage:
    python evaluate.py --data_path ./data/KolmFlow_valid_256.h5 --ckpt_dir ./checkpoints --fig_dir ./figures
"""

import argparse
import os
import torch
import numpy as np

from dataset import KolmFlowDataset, KolmFlowRolloutDataset, get_data_splits
from models.unet import UNet
from models.fno import FNO2d
from utils import (
    Normalizer, nrmse_field, rollout_nrmse,
    plot_rollout_comparison, plot_vorticity_comparison, plot_tke_comparison,
)


def load_model(model_name, ckpt_path, device):
    """Load a trained model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if model_name == "unet":
        model = UNet(in_channels=1, out_channels=1, base_channels=64)
    elif model_name == "fno":
        model = FNO2d(modes1=16, modes2=16, width=32, n_layers=4)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Strip _orig_mod. prefix added by torch.compile
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Loaded {model_name} from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt['val_loss']:.6f}, val_nrmse={ckpt.get('val_nrmse', 'N/A')})")
    return model


@torch.no_grad()
def eval_one_step(model, val_loader, device):
    """Compute average one-step NRMSE on validation set."""
    model.eval()
    total_nrmse = 0.0
    n = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_nrmse += nrmse_field(pred, y) * x.shape[0]
        n += x.shape[0]
    return total_nrmse / n


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_idx, val_idx = get_data_splits(n_samples=256, n_train=200, seed=42)

    norm_path = os.path.join(os.path.dirname(args.data_path), "normalizer.pt")
    normalizer = Normalizer.load(norm_path)
    print(f"Normalizer: mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")

    # One-step validation loader
    val_ds = KolmFlowDataset(args.data_path, val_idx, normalizer=normalizer)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # Rollout dataset (raw, unnormalized)
    rollout_ds = KolmFlowRolloutDataset(args.data_path, val_idx)

    unet = load_model("unet", os.path.join(args.ckpt_dir, "best_unet.pt"), device)
    fno = load_model("fno", os.path.join(args.ckpt_dir, "best_fno.pt"), device)

    print("\n=== One-Step NRMSE ===")
    nrmse_unet_1step = eval_one_step(unet, val_loader, device)
    nrmse_fno_1step = eval_one_step(fno, val_loader, device)
    print(f"U-Net: {nrmse_unet_1step:.6f}")
    print(f"FNO:   {nrmse_fno_1step:.6f}")

    print("\n=== Autoregressive Rollout ===")
    n_rollout_samples = min(5, len(val_idx))  # Average over a few samples
    rollout_steps = min(100, 199)

    all_unet_rollouts = []
    all_fno_rollouts = []

    for i in range(n_rollout_samples):
        frames = rollout_ds[i]  # [200, 160, 160]
        initial = frames[0].unsqueeze(0)    # [1, 160, 160]
        gt_future = frames[1:]              # [199, 160, 160]

        unet_nrmse = rollout_nrmse(unet, initial, gt_future, normalizer, device, rollout_steps)
        fno_nrmse_vals = rollout_nrmse(fno, initial, gt_future, normalizer, device, rollout_steps)

        all_unet_rollouts.append(unet_nrmse)
        all_fno_rollouts.append(fno_nrmse_vals)
        print(f"  Sample {i}: U-Net final={unet_nrmse[-1]:.4f}, FNO final={fno_nrmse_vals[-1]:.4f}")

    # Average rollout curves
    avg_unet = np.mean(all_unet_rollouts, axis=0)
    avg_fno = np.mean(all_fno_rollouts, axis=0)

    os.makedirs(args.fig_dir, exist_ok=True)
    plot_rollout_comparison(
        avg_unet, avg_fno,
        save_path=os.path.join(args.fig_dir, "rollout_nrmse.png"),
    )

    print("\n=== Vorticity Visualizations ===")
    # Use first validation sample
    frames = rollout_ds[0]  # [200, 160, 160]

    for model, model_name in [(unet, "U-Net"), (fno, "FNO")]:
        current = frames[0].unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 160, 160]

        for vis_step in [1, 10, 25, 50]:
            if vis_step >= frames.shape[0]:
                break

            # Roll forward to vis_step
            model.eval()
            pred = frames[0].unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                for t in range(vis_step):
                    inp = normalizer.encode(pred)
                    pred = normalizer.decode(model(inp))

            gt = frames[vis_step].cpu().numpy()
            pr = pred.squeeze().cpu().numpy()

            plot_vorticity_comparison(
                pr, gt, timestep=vis_step, model_name=model_name,
                save_path=os.path.join(args.fig_dir, f"vorticity_{model_name.lower()}_t{vis_step}.png"),
            )

    print("\n=== Log-TKE Spectrum ===")
    # Get predictions at a specific rollout step (e.g., t=20)
    tke_step = 20
    gt_field = frames[tke_step].cpu().numpy()

    # Roll forward with each model
    for_tke = {}
    for model, name in [(unet, "unet"), (fno, "fno")]:
        pred = frames[0].unsqueeze(0).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            for t in range(tke_step):
                inp = normalizer.encode(pred)
                pred = normalizer.decode(model(inp))
        for_tke[name] = pred.squeeze().cpu().numpy()

    plot_tke_comparison(
        gt_field, for_tke["unet"], for_tke["fno"],
        save_path=os.path.join(args.fig_dir, "tke_spectrum.png"),
    )

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'Metric':<30} {'U-Net':<12} {'FNO':<12}")
    print("-" * 50)
    print(f"{'One-step NRMSE':<30} {nrmse_unet_1step:<12.6f} {nrmse_fno_1step:<12.6f}")
    print(f"{'Rollout NRMSE (t=50)':<30} {avg_unet[49]:<12.4f} {avg_fno[49]:<12.4f}")
    print(f"{'Rollout NRMSE (t=100)':<30} {avg_unet[-1]:<12.4f} {avg_fno[-1]:<12.4f}")
    print("=" * 50)

    # Clean up
    val_ds.close()
    rollout_ds.close()
    print("\nDone! Figures saved to:", args.fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models on Kolmogorov Flow")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--fig_dir", type=str, default="./figures")
    args = parser.parse_args()
    main(args)
