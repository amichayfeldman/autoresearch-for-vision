"""
Task Definition: Anomaly Detection
====================================

Use case: classify images as normal or anomalous, with severe class imbalance.
Dataset: MNIST (auto-downloaded via torchvision).
         Normal  → label 0: digits 0-8  (~90 % of data after downsampling)
         Anomaly → label 1: digit 9     (~10 % of data after downsampling)

Digit 9 acts as the rare anomaly class; digits 0-8 are the "normal" population.
Downsampling is applied so the train split has a realistic 9:1 imbalance ratio.

Generated metric config
-----------------------
Inserting the task_description below into the system causes Claude's
metric_generator to produce a YAML similar to:

    _target_: torchmetrics.AUROC
    task: binary

AUROC is the standard metric for anomaly detection because:
- No classification threshold needs to be chosen in advance.
- It is robust to heavy class imbalance.
- It measures the model's ability to rank anomalies above normal samples.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import pytorch_lightning as pl
from torch.utils.data import Dataset, Subset
import numpy as np

from cv_autoresearch import run_autoresearch
from cv_autoresearch.config.schema import SearchConfig


# ── Task description ──────────────────────────────────────────────────────────

TASK_DESCRIPTION = (
    "Binary anomaly detection on handwritten digit images (MNIST). "
    "Digits 0-8 are treated as normal samples (label 0); digit 9 is the rare anomaly (label 1). "
    "Images are 28×28 grayscale converted to single-channel. "
    "Training imbalance is approximately 9:1 (normal:anomaly). "
    "Optimise AUROC to avoid requiring a hard classification threshold."
)

# ── Dataset ───────────────────────────────────────────────────────────────────

_TRANSFORM = T.Compose([T.ToTensor()])


class MNISTAnomaly(Dataset):
    """MNIST re-labelled as binary anomaly detection.

    Normal (0) = digits 0-8  |  Anomaly (1) = digit 9

    The normal class is downsampled so that the train split has a 9:1 ratio,
    mimicking realistic industrial defect inspection datasets.
    """

    def __init__(self, train: bool, imbalance_ratio: float = 9.0) -> None:
        base = torchvision.datasets.MNIST(
            root="./data", train=train, download=True, transform=_TRANSFORM
        )
        targets = np.array(base.targets)

        normal_idx = np.where(targets != 9)[0]
        anomaly_idx = np.where(targets == 9)[0]

        if train:
            # Downsample normal class to achieve the target ratio
            n_anomaly = len(anomaly_idx)
            n_normal = int(n_anomaly * imbalance_ratio)
            rng = np.random.default_rng(seed=42)
            normal_idx = rng.choice(normal_idx, size=n_normal, replace=False)

        self._indices = np.concatenate([normal_idx, anomaly_idx])
        self._labels = np.where(targets[self._indices] == 9, 1, 0)
        self._base = base

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, _ = self._base[self._indices[idx]]
        return image, int(self._labels[idx])


def get_datasets() -> tuple[Dataset, Dataset]:
    """Return (train_dataset, val_dataset) for MNIST anomaly detection."""
    return MNISTAnomaly(train=True), MNISTAnomaly(train=False, imbalance_ratio=9.0)


# ── Model ─────────────────────────────────────────────────────────────────────


class AnomalyDetector(pl.LightningModule):
    """Compact CNN binary classifier for MNIST anomaly detection."""

    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        # Upweight anomaly class (positive weight = imbalance ratio)
        weight = torch.tensor([1.0, 9.0], device=self.device)
        return F.cross_entropy(logits, labels, weight=weight)

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
        output_dir="./output/07_anomaly_detection",
        aug_overrides={
            # Grayscale images: colour augmentations are irrelevant
            "ColorJitter": None,
            "RandomBrightnessContrast": None,
            "CLAHE": None,
        },
    )

    result = run_autoresearch(AnomalyDetector(), train_ds, val_ds, config)
    print(f"Best AUROC: {result['best_metric']['value']:.4f}")
