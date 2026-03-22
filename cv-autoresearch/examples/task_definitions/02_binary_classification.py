"""
Task Definition: Binary Classification
=======================================

Use case: two-class problems, often with class imbalance.
Dataset: CIFAR-10 binary split — 5 animal classes vs 5 vehicle classes
         (auto-downloaded via torchvision, no CSV required).

  Animals  → label 0: bird, cat, deer, dog, frog, horse
  Vehicles → label 1: airplane, automobile, ship, truck

Generated metric config
-----------------------
Inserting the task_description below into the system causes Claude's
metric_generator to produce a YAML similar to:

    _target_: torchmetrics.AUROC
    task: binary

The system instantiates torchmetrics.AUROC(task="binary") and uses
area-under-ROC as the primary metric.  AUROC is threshold-free and robust
to class imbalance, which is why Claude selects it when the description
mentions an imbalanced binary problem.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import pytorch_lightning as pl
from torch.utils.data import Dataset, Subset

from cv_autoresearch import run_autoresearch
from cv_autoresearch.config.schema import SearchConfig


# ── Task description ──────────────────────────────────────────────────────────

TASK_DESCRIPTION = (
    "Binary image classification: distinguish living creatures (birds, cats, deer, "
    "dogs, frogs, horses — label 0) from man-made vehicles (airplanes, automobiles, "
    "ships, trucks — label 1) in 32×32 RGB CIFAR-10 images. "
    "Classes are balanced 60:40 in this split. Optimise AUROC."
)

# ── Dataset ───────────────────────────────────────────────────────────────────

# CIFAR-10 class indices
_ANIMAL_CLASSES = {2, 3, 4, 5, 6, 7}    # bird, cat, deer, dog, frog, horse
_VEHICLE_CLASSES = {0, 1, 8, 9}          # airplane, automobile, ship, truck

_TRANSFORM = T.Compose([T.ToTensor()])


class CIFAR10Binary(Dataset):
    """CIFAR-10 re-labelled as a binary animal-vs-vehicle task.

    No external files required — downloads CIFAR-10 automatically.
    """

    def __init__(self, train: bool) -> None:
        self._base = torchvision.datasets.CIFAR10(
            root="./data", train=train, download=True, transform=_TRANSFORM
        )

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, cifar_label = self._base[idx]
        binary_label = 0 if cifar_label in _ANIMAL_CLASSES else 1
        return image, binary_label


def get_datasets() -> tuple[Dataset, Dataset]:
    """Return (train_dataset, val_dataset) for binary CIFAR-10."""
    return CIFAR10Binary(train=True), CIFAR10Binary(train=False)


# ── Model ─────────────────────────────────────────────────────────────────────


class BinaryClassifier(pl.LightningModule):
    """MobileNetV3-Small for binary CIFAR-10 classification."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import mobilenet_v3_small
        self.backbone = mobilenet_v3_small(pretrained=False)
        # Replace final linear: in_features=1024 → 2 outputs
        self.backbone.classifier[-1] = torch.nn.Linear(1024, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        return F.cross_entropy(self(images), labels)

    def configure_optimizers(self):
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_ds, val_ds = get_datasets()

    config = SearchConfig(
        task_description=TASK_DESCRIPTION,
        total_trials=40,
        epochs_per_trial=5,
        device="cuda",
        output_dir="./output/02_binary_classification",
    )

    result = run_autoresearch(BinaryClassifier(), train_ds, val_ds, config)
    print(f"Best AUROC: {result['best_metric']['value']:.4f}")
