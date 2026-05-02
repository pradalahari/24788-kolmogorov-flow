# Neural PDE Surrogates for Kolmogorov Flow: U-Net vs. Fourier Neural Operator

**Course:** 24-788 Introduction to Deep Learning (Spring 2026, CMU)  
**Author:** Praditya Alahari

## Overview

This project compares a U-Net and a Fourier Neural Operator (FNO) for next-timestep vorticity prediction on 2D Kolmogorov flow. Both models achieve similar one-step accuracy (NRMSE ≈ 0.022), but the FNO significantly outperforms the U-Net in autoregressive rollout stability.

## Repository Structure

```
├── README.md
├── dataset.py                 # PyTorch Dataset for Kolmogorov Flow (HDF5)
├── train.py                   # Training script (shared for both models)
├── evaluate.py                # Evaluation: one-step NRMSE, rollout, TKE spectrum
├── utils.py                   # Normalization, metrics, plotting utilities
├── models/
│   ├── __init__.py
│   ├── unet.py                # U-Net implementation
│   └── fno.py                 # FNO implementation
├── checkpoints/
│   ├── best_unet.pt           # Trained U-Net weights
│   └── best_fno.pt            # Trained FNO weights
├── figures/                   # Generated plots for the report
├── reproduce_results.ipynb    # Notebook to reproduce all figures and metrics
└── data/
    └── normalizer.pt          # Precomputed normalization statistics
```

## Setup

### Dependencies

This project runs on Google Colab Pro. The following packages are used (all pre-installed on Colab):

- Python 3.10+
- PyTorch 2.0+
- h5py
- matplotlib
- numpy

### Data Download

Download `KolmFlow_valid_256.h5` (~5 GB) from HuggingFace and place it in `data/`:

```
https://huggingface.co/datasets/ayz2/temporal_pdes/tree/main/valid
```

## Reproducing Results

### Option 1: Run the reproduce notebook (recommended)

Open `reproduce_results.ipynb` in Google Colab, mount your Drive, and run all cells. This loads the saved checkpoints and regenerates all figures and metrics from the report without retraining.

### Option 2: Retrain from scratch

```bash
# Train U-Net
python train.py --model unet --data_path ./data/KolmFlow_valid_256.h5 --epochs 50 --batch_size 32 --stride 2

# Train FNO
python train.py --model fno --data_path ./data/KolmFlow_valid_256.h5 --epochs 50 --batch_size 32 --stride 2

# Evaluate both models
python evaluate.py --data_path ./data/KolmFlow_valid_256.h5 --ckpt_dir ./checkpoints --fig_dir ./figures
```

## Results Summary

| Model | Parameters | One-Step NRMSE | Rollout NRMSE (t=50) | Rollout NRMSE (t=100) |
|-------|-----------|---------------|---------------------|----------------------|
| U-Net | 7.70M | 0.0225 | 2.24 | 1.69 × 10⁶ |
| FNO   | 2.10M | 0.0216 | 0.58 | 0.96 |

## References

- Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations," ICLR 2021
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015
