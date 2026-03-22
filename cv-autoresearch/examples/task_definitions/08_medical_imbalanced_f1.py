"""
Task Definition: Imbalanced Multi-class Classification (Macro F1)
=================================================================

Use case: multi-class classification with severe class imbalance where
every class matters equally regardless of frequency.
Dataset: SVHN (Street View House Numbers, auto-downloaded via torchvision).
         10 digit classes with naturally unequal frequencies.
         Digit '1' dominates (~25 % of train samples);
         digit '0' is the rarest (~5 %).

Although SVHN is a digit recognition dataset, it serves as a drop-in
demonstration for any imbalanced multi-class task (medical imaging,
rare disease classification, quality inspection).

Generated metric config
-----------------------
Inserting the task_description below into the system causes Claude's
metric_generator to produce a YAML similar to:

    _target_: torchmetrics.F1Score
    task: multiclass
    num_classes: 10
    average: macro

Claude selects macro-averaged F1 (not weighted or micro) because:
- The description explicitly mentions class imbalance.
- Macro averaging weights every class equally, penalising models that
  ignore minority classes.
- Accuracy and weighted-F1 would both be dominated by the majority class.
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
    "Multi-class classification of 32×32 RGB street-view digit images into 10 classes "
    "(SVHN dataset). Classes are naturally imbalanced: digit '1' appears ~5× more "
    "frequently than digit '0'. All 10 classes must be recognised reliably. "
    "Optimise macro-averaged F1 score so that rare classes receive equal weight to "
    "frequent ones."
)

# ── Dataset factory ───────────────────────────────────────────────────────────

_TRANSFORM = T.Compose([T.ToTensor()])


def get_datasets() -> tuple[Dataset, Dataset]:
    """Download and return (train_dataset, val_dataset) for SVHN."""
    train_ds = torchvision.datasets.SVHN(
        root="./data/SVHN", split="train", download=True, transform=_TRANSFORM
    )
    val_ds = torchvision.datasets.SVHN(
        root="./data/SVHN", split="test", download=True, transform=_TRANSFORM
    )
    return train_ds, val_ds


# ── Model ─────────────────────────────────────────────────────────────────────


class SVHNClassifier(pl.LightningModule):
    """ResNet-18 for SVHN 10-class digit classification."""

    NUM_CLASSES = 10

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import resnet18
        self.backbone = resnet18(pretrained=False)
        self.backbone.fc = torch.nn.Linear(512, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        # SVHN labels are in [1, 10]; remap digit '10' → 0
        labels = labels % 10
        return F.cross_entropy(logits, labels)

    def configure_optimizers(self):
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_ds, val_ds = get_datasets()

    config = SearchConfig(
        task_description=TASK_DESCRIPTION,
        total_trials=50,
        epochs_per_trial=8,
        device="cuda",
        output_dir="./output/08_medical_imbalanced_f1",
    )

    result = run_autoresearch(SVHNClassifier(), train_ds, val_ds, config)
    print(f"Best macro F1: {result['best_metric']['value']:.4f}")
