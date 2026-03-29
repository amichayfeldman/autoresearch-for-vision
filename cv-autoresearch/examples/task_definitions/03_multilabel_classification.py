"""
Task Definition: Multi-label Classification
============================================

Use case: each image simultaneously belongs to multiple classes.
Dataset: PASCAL VOC 2012 Detection (auto-downloaded via torchvision).
         The XML annotation for each image lists all objects present;
         we convert that into a 20-dim multi-hot vector.

Generated metric config
-----------------------
Inserting the task_description below into the system causes Claude's
metric_generator to produce a YAML similar to:

    _target_: torchmetrics.AveragePrecision
    task: multilabel
    num_labels: 20
    average: macro

The system instantiates torchmetrics.AveragePrecision(task="multilabel",
num_labels=20, average="macro") and uses mean average precision (mAP)
as the primary metric.
"""

from __future__ import annotations

from typing import Any

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
    "Multi-label image classification on PASCAL VOC 2012 (20 object categories). "
    "Each 224×224 RGB image may contain zero or more of the 20 classes simultaneously. "
    "Optimise macro mean average precision (mAP) so that rare classes receive equal weight."
)

# ── VOC class list (alphabetical, matches torchvision ordering) ───────────────

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
_CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}

_TRANSFORM = T.Compose([T.Resize((224, 224)), T.ToTensor()])


# ── Dataset ───────────────────────────────────────────────────────────────────


class VOCMultiLabel(Dataset):
    """PASCAL VOC 2012 wrapped as a multi-label classification dataset.

    Downloads the dataset automatically on first use (~2 GB).
    Each sample returns a float multi-hot vector of shape (20,) as the label.
    """

    def __init__(self, image_set: str) -> None:
        self._voc = torchvision.datasets.VOCDetection(
            root="./data/VOC",
            year="2012",
            image_set=image_set,
            download=True,
        )

    def __len__(self) -> int:
        return len(self._voc)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_pil, annotation = self._voc[idx]
        image = _TRANSFORM(image_pil)

        # annotation['annotation']['object'] is a dict for one object
        # or a list of dicts for multiple objects
        objects = annotation["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        target = torch.zeros(len(VOC_CLASSES), dtype=torch.float32)
        for obj in objects:
            class_name = obj["name"]
            if class_name in _CLASS_TO_IDX:
                target[_CLASS_TO_IDX[class_name]] = 1.0

        return image, target


def get_datasets() -> tuple[Dataset, Dataset]:
    """Return (train_dataset, val_dataset) for VOC multi-label."""
    return VOCMultiLabel("train"), VOCMultiLabel("val")


# ── Model ─────────────────────────────────────────────────────────────────────


class MultiLabelResNet(pl.LightningModule):
    """ResNet-50 with sigmoid multi-label output head."""

    NUM_CLASSES = 20

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import resnet50
        self.backbone = resnet50(pretrained=False)
        self.backbone.fc = torch.nn.Linear(2048, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # Raw logits; sigmoid applied in loss/metric

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, targets = batch
        logits = self(images)
        return F.binary_cross_entropy_with_logits(logits, targets)

    def configure_optimizers(self):
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_ds, val_ds = get_datasets()

    config = SearchConfig(
        task_description=TASK_DESCRIPTION,
        total_trials=50,
        epochs_per_trial=10,
        device="cuda",
        output_dir="./output/03_multilabel_classification",
    )

    result = run_autoresearch(MultiLabelResNet(), train_ds, val_ds, config)
    print(f"Best mAP: {result['best_metric']['value']:.4f}")
