"""
Task Definition: Fine-grained Visual Recognition
=================================================

Use case: distinguish highly similar subcategories within a single semantic
domain where inter-class differences are subtle.
Dataset: STL-10 (auto-downloaded via torchvision).
         10 classes at 96×96 resolution with only 500 training images per class.
         The small training set and high resolution make it a challenging
         fine-grained proxy comparable to CUB-200 or Stanford Cars.

STL-10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
Several pairs (e.g. bird↔airplane, car↔truck) are visually similar and require
fine-grained feature discrimination to separate reliably.

Generated metric config
-----------------------
Inserting the task_description below into the system causes Claude's
metric_generator to produce a YAML similar to:

    _target_: torchmetrics.Accuracy
    task: multiclass
    num_classes: 10
    top_k: 1

Fine-grained recognition is evaluated with top-1 accuracy (not top-5) because
confusing one subcategory with another is a genuine prediction error.
The system will typically direct the search toward strong regularisation
(weight decay, label smoothing, dropout) and aggressive augmentation
(RandomResizedCrop, ColorJitter) to combat the small training set.
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
    "Fine-grained visual classification of 10 object categories on STL-10. "
    "Images are 96×96 RGB photographs with only 500 training images per class. "
    "Visually similar pairs (bird vs airplane, car vs truck) require subtle "
    "feature discrimination. "
    "Strong data augmentation and regularisation are recommended to prevent "
    "overfitting. Optimise top-1 accuracy."
)

# ── Dataset factory ───────────────────────────────────────────────────────────

_TRANSFORM = T.Compose([T.ToTensor()])


def get_datasets() -> tuple[Dataset, Dataset]:
    """Download and return (train_dataset, val_dataset) for STL-10.

    STL-10 uses 'train' and 'test' splits (no 'val').
    The 'test' split (8,000 images) is used as the validation set.
    """
    train_ds = torchvision.datasets.STL10(
        root="./data/STL10", split="train", download=True, transform=_TRANSFORM
    )
    val_ds = torchvision.datasets.STL10(
        root="./data/STL10", split="test", download=True, transform=_TRANSFORM
    )
    return train_ds, val_ds


# ── Model ─────────────────────────────────────────────────────────────────────


class FineGrainedClassifier(pl.LightningModule):
    """ResNet-34 for fine-grained STL-10 classification.

    ResNet-34 is chosen deliberately over ResNet-50 to reduce overfitting
    on the small 500-images-per-class training set.
    """

    NUM_CLASSES = 10

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import resnet34
        self.backbone = resnet34(pretrained=False)
        self.backbone.fc = torch.nn.Linear(512, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        # Label smoothing helps prevent overconfident predictions on small datasets
        return F.cross_entropy(logits, labels, label_smoothing=0.1)

    def configure_optimizers(self):
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_ds, val_ds = get_datasets()

    config = SearchConfig(
        task_description=TASK_DESCRIPTION,
        total_trials=80,
        epochs_per_trial=20,  # Fine-grained models need more epochs to converge
        device="cuda",
        output_dir="./output/10_finegrained_recognition",
        aug_overrides={
            "Normalize": {   # Normalise with standard ImageNet statistics
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    )

    result = run_autoresearch(FineGrainedClassifier(), train_ds, val_ds, config)
    print(f"Best accuracy: {result['best_metric']['value']:.4f}")
