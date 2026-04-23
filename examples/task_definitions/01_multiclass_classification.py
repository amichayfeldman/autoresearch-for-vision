"""
Task Definition: Multi-class Image Classification
==================================================

Use case: classify images into N mutually exclusive categories.
Dataset: CIFAR-10 (auto-downloaded via torchvision).

No LightningModule subclassing required — just a plain nn.Module,
a loss_fn, and a predict function mapping logits to integer labels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

from cv_autoresearch.autoresearch import run
from cv_autoresearch.task import TaskDef


# ── Dataset factory ───────────────────────────────────────────────────────────

_TRANSFORM = T.Compose([T.ToTensor()])


def get_datasets() -> tuple[Dataset, Dataset]:
    """Download and return (train_dataset, val_dataset) for CIFAR-10."""
    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=_TRANSFORM
    )
    val_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=_TRANSFORM
    )
    return train_ds, val_ds


# ── Model ─────────────────────────────────────────────────────────────────────


class SmallCNN(nn.Module):
    """Lightweight CNN for 32×32 CIFAR input. Plain nn.Module — no LightningModule."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_ds, val_ds = get_datasets()

    task = TaskDef(
        model=SmallCNN(),
        train_dataset=train_ds,
        val_dataset=val_ds,
        loss_fn=lambda logits, batch: F.cross_entropy(logits, batch[1]),
        predict=lambda logits, batch: (logits.argmax(1).tolist(), batch[1].tolist()),
        num_classes=10,
        description="10-class image classification on CIFAR-10 (32×32 RGB). Optimise macro F1.",
    )

    result = run(task, total_directives=10, epochs_per_trial=5, device="cpu")
    print(f"Best F1: {result['best_f1']:.4f}")
    print(f"Best hyperparams: {result['best_hp']}")
