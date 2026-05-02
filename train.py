"""
train.py — Shared training script for U-Net and FNO.

Usage (from Colab or command line):
    python train.py --model unet --data_path ./data/KolmFlow_valid_256.h5 --epochs 50
    python train.py --model fno  --data_path ./data/KolmFlow_valid_256.h5 --epochs 50
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import KolmFlowDataset, get_data_splits
from models.unet import UNet
from models.fno import FNO2d
from utils import compute_normalizer, nrmse_field, plot_training_curves


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_nrmse = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item()
        total_nrmse += nrmse_field(pred, y)
        n_batches += 1

    return total_loss / n_batches, total_nrmse / n_batches


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_idx, val_idx = get_data_splits(n_samples=256, n_train=200, seed=42)
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

    norm_path = os.path.join(os.path.dirname(args.data_path), "normalizer.pt")
    if os.path.exists(norm_path):
        normalizer = __import__("utils").Normalizer.load(norm_path)
        print(f"Loaded normalizer: mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")
    else:
        print("Computing normalizer stats from training data...")
        normalizer = compute_normalizer(args.data_path, train_idx)
        normalizer.save(norm_path)
        print(f"Saved normalizer: mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")

    train_ds = KolmFlowDataset(args.data_path, train_idx, normalizer=normalizer, stride=args.stride)
    val_ds = KolmFlowDataset(args.data_path, val_idx, normalizer=normalizer, stride=args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train pairs: {len(train_ds)}, Val pairs: {len(val_ds)}")

    if args.model == "unet":
        model = UNet(in_channels=1, out_channels=1, base_channels=64)
    elif args.model == "fno":
        model = FNO2d(modes1=16, modes2=16, width=32, n_layers=4)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Parameters: {n_params:,}")

    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile()")
    except Exception:
        print("torch.compile not available, using eager mode")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_nrmse = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val NRMSE: {val_nrmse:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_nrmse": val_nrmse,
                "args": vars(args),
            }, os.path.join(ckpt_dir, f"best_{args.model}.pt"))
            print(f"  → Saved best model (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    fig_dir = args.fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    plot_training_curves(
        train_losses, val_losses,
        title=f"{args.model.upper()} Training Curves",
        save_path=os.path.join(fig_dir, f"training_curves_{args.model}.png"),
    )

    # Clean up
    train_ds.close()
    val_ds.close()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net or FNO on Kolmogorov Flow")
    parser.add_argument("--model", type=str, required=True, choices=["unet", "fno"])
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to KolmFlow_valid_256.h5")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--fig_dir", type=str, default="./figures")
    parser.add_argument("--stride", type=int, default=2,
                        help="Timestep stride (2 = use every other pair, halves data)")
    args = parser.parse_args()
    main(args)
