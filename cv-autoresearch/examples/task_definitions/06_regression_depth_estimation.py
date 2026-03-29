"""
Task Definition: Regression / Depth Estimation
===============================================

Use case: predict a continuous scalar value from an image.
Dataset: Synthetic in-memory dataset — no download required.

Each sample is a 64×64 RGB image containing a solid-coloured disk.
The regression target is the disk's radius (normalised to [0, 1]).
A model that learns to detect and measure the disk can achieve low RMSE.

This synthetic setup demonstrates the task description pattern for any
scalar regression task (depth, age, quality score, angle, count, etc.)
without requiring external data downloads.

Generated metric config
-----------------------
Inserting the task_description below into the system causes Claude's
metric_generator to produce a YAML similar to:

    _target_: torchmetrics.MeanSquaredError
    squared: false

squared: false means the metric returns RMSE (not MSE).
higher_is_better is set to False because we minimise error.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset

from cv_autoresearch import run_autoresearch
from cv_autoresearch.config.schema import SearchConfig


# ── Task description ──────────────────────────────────────────────────────────

TASK_DESCRIPTION = (
    "Scalar regression: predict the radius of a coloured disk drawn on a 64×64 RGB image. "
    "Radius values are normalised to [0, 1]. "
    "Minimise root mean squared error (RMSE) on the validation set."
)

# ── Dataset ───────────────────────────────────────────────────────────────────

_RNG = np.random.default_rng(seed=42)
_IMG_SIZE = 64


def _make_disk_image(radius: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a 64×64 RGB image with a disk of given normalised radius."""
    img = np.zeros((_IMG_SIZE, _IMG_SIZE, 3), dtype=np.float32)
    cx = cy = _IMG_SIZE // 2
    r_px = int(radius * _IMG_SIZE * 0.45) + 1  # Map [0,1] → [1, 29] pixels
    color = rng.uniform(0.3, 1.0, size=3).astype(np.float32)

    ys, xs = np.ogrid[:_IMG_SIZE, :_IMG_SIZE]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r_px ** 2
    img[mask] = color
    return img  # (H, W, 3), float32 in [0, 1]


class SyntheticDiskDataset(Dataset):
    """Synthetic dataset of disk images with scalar radius targets.

    Generated deterministically in memory — no file I/O or download.
    """

    def __init__(self, n_samples: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._radii = rng.uniform(0.05, 0.95, size=n_samples).astype(np.float32)
        self._images = [_make_disk_image(r, rng) for r in self._radii]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Image: (H, W, C) float32 → (C, H, W)
        image = torch.from_numpy(self._images[idx]).permute(2, 0, 1)
        target = torch.tensor(self._radii[idx])  # Scalar float32
        return image, target


def get_datasets() -> tuple[Dataset, Dataset]:
    """Return (train_dataset, val_dataset) — fully synthetic, no download."""
    return SyntheticDiskDataset(n_samples=5000, seed=0), SyntheticDiskDataset(n_samples=1000, seed=1)


# ── Model ─────────────────────────────────────────────────────────────────────


class DiskRadiusRegressor(pl.LightningModule):
    """Small CNN that predicts the radius of a disk in a 64×64 image."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(1)  # (B,)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, radii = batch
        preds = self(images)
        return F.mse_loss(preds, radii)

    def configure_optimizers(self):
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_ds, val_ds = get_datasets()

    config = SearchConfig(
        task_description=TASK_DESCRIPTION,
        total_trials=40,
        epochs_per_trial=8,
        device="cuda",
        output_dir="./output/06_regression_depth_estimation",
    )

    result = run_autoresearch(DiskRadiusRegressor(), train_ds, val_ds, config)
    print(f"Best RMSE: {result['best_metric']['value']:.4f}")
