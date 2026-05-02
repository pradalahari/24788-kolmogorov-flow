"""
dataset.py — PyTorch Dataset for Kolmogorov Flow (HDF5).

Lazily loads individual (frame_t, frame_t+1) pairs from the HDF5 file.
Splits on samples (not timesteps) for proper train/val separation.
"""

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class KolmFlowDataset(Dataset):
    """
    Dataset for next-step vorticity prediction.

    Each item returns:
        x: vorticity at time t,   shape [1, 160, 160]
        y: vorticity at time t+1, shape [1, 160, 160]

    Args:
        h5_path: path to KolmFlow_valid_256.h5
        sample_indices: list/array of sample indices to use (e.g., 0-199 for train)
        normalizer: optional Normalizer object to normalize data on the fly
        n_timesteps: number of timesteps per sample (default 200)
    """

    def __init__(self, h5_path, sample_indices, normalizer=None, n_timesteps=200, stride=1):
        self.h5_path = h5_path
        self.sample_indices = np.array(sample_indices)
        self.normalizer = normalizer
        self.n_timesteps = n_timesteps
        self.stride = stride

        # Build list of valid timestep indices with stride
        self.valid_timesteps = list(range(0, n_timesteps - 1, stride))
        self.pairs_per_sample = len(self.valid_timesteps)
        self.total_pairs = len(self.sample_indices) * self.pairs_per_sample

        self._file = None

    def __len__(self):
        return self.total_pairs

    def __getitem__(self, idx):
        sample_local = idx // self.pairs_per_sample
        t_local = idx % self.pairs_per_sample
        t = self.valid_timesteps[t_local]

        sample_idx = self.sample_indices[sample_local]

        # Lazy open (for num_workers=0 usage; for num_workers>0, open/close each call)
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

        u = self._file["valid"]["u"]

        x = torch.tensor(u[sample_idx, t], dtype=torch.float32).unsqueeze(0)       # [1, 160, 160]
        y = torch.tensor(u[sample_idx, t + 1], dtype=torch.float32).unsqueeze(0)    # [1, 160, 160]

        if self.normalizer is not None:
            x = self.normalizer.encode(x)
            y = self.normalizer.encode(y)

        return x, y

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


class KolmFlowRolloutDataset(Dataset):
    """
    Dataset for rollout evaluation.
    Returns an entire trajectory for a given sample.

    Each item returns:
        frames: shape [200, 160, 160] — all timesteps (unnormalized)
    """

    def __init__(self, h5_path, sample_indices):
        self.h5_path = h5_path
        self.sample_indices = np.array(sample_indices)
        self._file = None

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]

        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

        frames = torch.tensor(
            self._file["valid"]["u"][sample_idx], dtype=torch.float32
        )  # [200, 160, 160]

        return frames

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def get_data_splits(n_samples=256, n_train=200, seed=42):
    """
    Returns train and val sample indices.
    Split is on samples, NOT on timesteps.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:].tolist()
    return train_indices, val_indices
