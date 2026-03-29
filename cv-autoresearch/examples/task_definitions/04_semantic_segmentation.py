"""
Task Definition: Semantic Segmentation
=======================================

Use case: assign a class label to every pixel of an image.
Dataset: PASCAL VOC 2012 Segmentation (auto-downloaded via torchvision).
         21 classes (20 object categories + background).
         Pixel value 255 = void / ignore.

Generated metric config
-----------------------
Inserting the task_description below into the system causes Claude's
metric_generator to produce a YAML similar to:

    _target_: torchmetrics.JaccardIndex
    task: multiclass
    num_classes: 21
    ignore_index: 255

The system instantiates torchmetrics.JaccardIndex(task="multiclass",
num_classes=21, ignore_index=255) and uses mean IoU (mIoU) as the
primary metric.

Dataset __getitem__ contract for segmentation
---------------------------------------------
The label returned by __getitem__ is a 2-D int64 tensor of shape (H, W)
where each value is a class index.  The model's forward() must return
logits of shape (B, C, H, W).  torchmetrics.JaccardIndex handles this
shape automatically.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import pytorch_lightning as pl
from torch.utils.data import Dataset
from PIL import Image

from cv_autoresearch import run_autoresearch
from cv_autoresearch.config.schema import SearchConfig


# ── Task description ──────────────────────────────────────────────────────────

TASK_DESCRIPTION = (
    "Semantic segmentation of PASCAL VOC 2012 scenes into 21 classes "
    "(20 object categories + background). Images are resized to 320×320 RGB. "
    "Pixel label 255 is void/ignore. Optimise mean IoU (mIoU)."
)

# ── Constants ─────────────────────────────────────────────────────────────────

_IMAGE_SIZE = (320, 320)
_IMG_TRANSFORM = T.Compose([T.Resize(_IMAGE_SIZE), T.ToTensor()])

# ── Dataset ───────────────────────────────────────────────────────────────────


class VOCSegmentation(Dataset):
    """PASCAL VOC 2012 Segmentation wrapped for cv-autoresearch.

    Downloads the dataset automatically on first use (~2 GB).
    Returns (image_tensor, mask_tensor) where mask is int64 (H, W).
    """

    def __init__(self, image_set: str) -> None:
        self._voc = torchvision.datasets.VOCSegmentation(
            root="./data/VOC",
            year="2012",
            image_set=image_set,
            download=True,
        )

    def __len__(self) -> int:
        return len(self._voc)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_pil, mask_pil = self._voc[idx]

        image = _IMG_TRANSFORM(image_pil)

        # Resize mask with nearest-neighbour to preserve class indices
        mask_pil = mask_pil.resize((_IMAGE_SIZE[1], _IMAGE_SIZE[0]), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask_pil, dtype=np.int64))  # (H, W)

        return image, mask


def get_datasets() -> tuple[Dataset, Dataset]:
    """Return (train_dataset, val_dataset) for VOC segmentation."""
    return VOCSegmentation("train"), VOCSegmentation("val")


# ── Model ─────────────────────────────────────────────────────────────────────


class FCNSegmenter(pl.LightningModule):
    """FCN-ResNet50 for 21-class PASCAL VOC semantic segmentation."""

    NUM_CLASSES = 21

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models.segmentation import fcn_resnet50
        self.net = fcn_resnet50(pretrained=False, num_classes=self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torchvision segmentation models return an OrderedDict
        return self.net(x)["out"]  # (B, NUM_CLASSES, H, W)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, masks = batch
        logits = self(images)
        return F.cross_entropy(logits, masks, ignore_index=255)

    def configure_optimizers(self):
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_ds, val_ds = get_datasets()

    config = SearchConfig(
        task_description=TASK_DESCRIPTION,
        total_trials=50,
        epochs_per_trial=15,
        device="cuda",
        output_dir="./output/04_semantic_segmentation",
        aug_overrides={
            "VerticalFlip": None,  # Vertical flip is rarely helpful for scene parsing
        },
    )

    result = run_autoresearch(FCNSegmenter(), train_ds, val_ds, config)
    print(f"Best mIoU: {result['best_metric']['value']:.4f}")
