"""
Task Definition: Large-scale Classification with Top-5 Accuracy
===============================================================

Use case: classification over many classes where the top-1 prediction is
too strict and top-5 captures user-acceptable correctness.
Dataset: CIFAR-100 (auto-downloaded via torchvision).
         100 fine-grained classes across 20 superclass groups.

With 100 classes, top-5 accuracy is a meaningful metric: the correct class
must appear in the model's five highest-confidence predictions.  This mirrors
the standard ImageNet-1K evaluation protocol at a more accessible scale.

Generated metric config
-----------------------
Inserting the task_description below into the system causes Claude's
metric_generator to produce a YAML similar to:

    _target_: torchmetrics.Accuracy
    task: multiclass
    num_classes: 100
    top_k: 5

The system instantiates torchmetrics.Accuracy(task="multiclass",
num_classes=100, top_k=5) and uses top-5 accuracy as the primary metric.
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
    "Large-scale 100-class image classification on CIFAR-100. "
    "Classes span 20 supergroups (animals, vehicles, household objects, plants, etc.) "
    "with 500 training images per class. Images are 32×32 RGB. "
    "Optimise top-5 accuracy: the correct class must appear in the five "
    "highest-confidence predictions."
)

# ── Dataset factory ───────────────────────────────────────────────────────────

_TRANSFORM = T.Compose([T.ToTensor()])


def get_datasets() -> tuple[Dataset, Dataset]:
    """Download and return (train_dataset, val_dataset) for CIFAR-100."""
    train_ds = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=_TRANSFORM
    )
    val_ds = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=_TRANSFORM
    )
    return train_ds, val_ds


# ── Model ─────────────────────────────────────────────────────────────────────


class CIFAR100Classifier(pl.LightningModule):
    """ResNet-50 for 100-class CIFAR-100 classification."""

    NUM_CLASSES = 100

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import resnet50
        self.backbone = resnet50(pretrained=False)
        # CIFAR-100 is 32×32; replace the 7×7 stem with a 3×3 conv
        self.backbone.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = torch.nn.Identity()
        self.backbone.fc = torch.nn.Linear(2048, self.NUM_CLASSES)

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
        total_trials=60,
        epochs_per_trial=10,
        device="cuda",
        output_dir="./output/09_top5_accuracy",
        hp_overrides={
            "batch_size": 128,  # Larger batches stabilise training on CIFAR-100
        },
    )

    result = run_autoresearch(CIFAR100Classifier(), train_ds, val_ds, config)
    print(f"Best top-5 accuracy: {result['best_metric']['value']:.4f}")
