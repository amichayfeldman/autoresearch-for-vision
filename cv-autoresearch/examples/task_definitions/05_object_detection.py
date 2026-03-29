"""
Task Definition: Object Detection
==================================

Use case: predict bounding boxes and class labels for all objects in an image.
Dataset: PASCAL VOC 2007 Detection (auto-downloaded via torchvision).

Generated metric config
-----------------------
Inserting the task_description below into the system causes Claude's
metric_generator to produce a YAML similar to:

    _target_: torchmetrics.detection.MeanAveragePrecision
    iou_thresholds:
      - 0.5
    class_metrics: false

NOTE — Evaluator compatibility
-------------------------------
torchmetrics.detection.MeanAveragePrecision expects predictions and targets
as lists of dicts ({"boxes", "scores", "labels"}), not stacked tensors.
The standard cv-autoresearch evaluator returns stacked tensors and therefore
does not support this metric directly.

This example demonstrates the correct task_description wording that generates
the detection metric config.  The runnable model below uses a classification
proxy — predicting the *dominant object class* in each image — so it can
run end-to-end with the standard evaluator.  For true mAP evaluation,
override the evaluator with custom detection inference logic.

Dataset: VOCDetection
---------------------
torchvision.datasets.VOCDetection returns (PIL.Image, annotation_dict).
The annotation dict contains an 'object' key with per-box info.
We extract the class of the most-frequent object in each image as the label.
"""

from __future__ import annotations

from collections import Counter

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
    "Object detection on PASCAL VOC 2007 (20 object classes). "
    "Predict axis-aligned bounding boxes and class labels for all objects in 224×224 RGB images. "
    "Optimise mAP at IoU threshold 0.50 (standard PASCAL VOC metric)."
)

# ── VOC class list ────────────────────────────────────────────────────────────

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
_CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}

_TRANSFORM = T.Compose([T.Resize((224, 224)), T.ToTensor()])


# ── Dataset ───────────────────────────────────────────────────────────────────


class VOCDominantClass(Dataset):
    """VOC 2007 Detection re-framed as single-label classification.

    For each image the label is the index of the most frequently occurring
    object class.  This proxy task lets the example run end-to-end with the
    standard evaluator while demonstrating the detection task description.

    Downloads the dataset automatically on first use (~870 MB).
    """

    def __init__(self, image_set: str) -> None:
        self._voc = torchvision.datasets.VOCDetection(
            root="./data/VOC",
            year="2007",
            image_set=image_set,
            download=True,
        )

    def __len__(self) -> int:
        return len(self._voc)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_pil, annotation = self._voc[idx]
        image = _TRANSFORM(image_pil)

        objects = annotation["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        # Count object classes and pick the most frequent one
        class_counts: Counter = Counter()
        for obj in objects:
            name = obj["name"]
            if name in _CLASS_TO_IDX:
                class_counts[_CLASS_TO_IDX[name]] += 1

        dominant = class_counts.most_common(1)[0][0] if class_counts else 0
        return image, dominant


def get_datasets() -> tuple[Dataset, Dataset]:
    """Return (train_dataset, val_dataset) for VOC dominant-class proxy."""
    return VOCDominantClass("train"), VOCDominantClass("val")


# ── Model ─────────────────────────────────────────────────────────────────────


class DominantClassifier(pl.LightningModule):
    """MobileNetV2 classifier for the dominant-object proxy task."""

    NUM_CLASSES = 20

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import mobilenet_v2
        self.backbone = mobilenet_v2(pretrained=False)
        self.backbone.classifier[1] = torch.nn.Linear(1280, self.NUM_CLASSES)

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
        epochs_per_trial=10,
        device="cuda",
        output_dir="./output/05_object_detection",
        aug_overrides={
            "RandomResizedCrop": None,  # Would invalidate box coordinates in true detection
            "Perspective": None,
        },
    )

    result = run_autoresearch(DominantClassifier(), train_ds, val_ds, config)
    print(f"Best metric: {result['best_metric']['value']:.4f}")
