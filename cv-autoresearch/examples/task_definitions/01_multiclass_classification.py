"""
Task Definition: Multi-class Image Classification
==================================================

Use case: classify images into N mutually exclusive categories.
Dataset: CIFAR-10 (auto-downloaded via torchvision).

Generated metric config
-----------------------
Inserting the task_description below into the system causes Claude's
metric_generator to produce a YAML similar to:

    _target_: torchmetrics.Accuracy
    task: multiclass
    num_classes: 10
    top_k: 1

The system instantiates torchmetrics.Accuracy(task="multiclass",
num_classes=10, top_k=1) and uses top-1 accuracy during validation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import pytorch_lightning as pl
from torch.utils.data import Dataset

from cv_autoresearch import run_autoresearch
from cv_autoresearch.config.schema import SearchConfig


# ── Task description ──────────────────────────────────────────────────────────

TASK_DESCRIPTION = (
    "10-class image classification on CIFAR-10 "
    "(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). "
    "Images are 32×32 RGB. Optimise top-1 multiclass accuracy."
)

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


class SmallCNN(pl.LightningModule):
    """Lightweight CNN for 32×32 CIFAR input."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        return F.cross_entropy(self(images), labels)

    def configure_optimizers(self):
        pass  # Managed by AutoResearchModule


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_ds, val_ds = get_datasets()

    config = SearchConfig(
        task_description=TASK_DESCRIPTION,
        total_trials=40,
        epochs_per_trial=5,
        device="cuda",
        output_dir="./output/01_multiclass_classification",
    )

    result = run_autoresearch(SmallCNN(), train_ds, val_ds, config)
    print(f"Best accuracy: {result['best_metric']['value']:.4f}")
    print(f"Best hyperparams: {result['best_hyperparams']}")
