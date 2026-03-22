# Custom Dataset & Model Integration Guide

This guide explains how to bring your own dataset and model into **cv-autoresearch**,
the AI-directed computer-vision hyperparameter and augmentation search system.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Dataset Requirements](#2-dataset-requirements)
3. [CSV / DataFrame-Based Datasets](#3-csv--dataframe-based-datasets)
4. [Model Requirements](#4-model-requirements)
5. [SearchConfig Reference](#5-searchconfig-reference)
6. [Writing a Good Task Description](#6-writing-a-good-task-description)
7. [Complete Worked Example](#7-complete-worked-example)
8. [Running via the CLI](#8-running-via-the-cli)
9. [Advanced: Overriding Hyperparameters and Augmentations](#9-advanced-overriding-hyperparameters-and-augmentations)
10. [Output Files](#10-output-files)

---

## 1. System Overview

cv-autoresearch runs an **AI-directed search loop** over both hyperparameters and image
augmentations.  At every iteration:

1. **Claude** reads the experiment history and decides which single parameter to vary next.
2. **Optuna** proposes a concrete value for that parameter.
3. The system trains your model for `epochs_per_trial` epochs, evaluates it with the metric
   Claude chose from your task description, and records the result.
4. If the trial improved the primary metric the baseline is updated and the loop continues.

You are responsible for two things only:

| What you provide | Purpose |
|---|---|
| A `torch.utils.data.Dataset` (train + val) | Data to learn from |
| A `pytorch_lightning.LightningModule` | Model & loss to optimise |

Everything else—optimiser, scheduler, augmentations, batch size, metric—is managed by the system.

---

## 2. Dataset Requirements

Your dataset must implement the standard PyTorch `Dataset` interface:

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __len__(self) -> int:
        """Return total number of samples."""
        ...

    def __getitem__(self, idx: int) -> tuple:
        """Return (image, label) for the sample at position idx."""
        ...
```

### `__getitem__` return contract

The system expects every item to be a **2-tuple `(image, label)`**.

| Field | Accepted types | Notes |
|---|---|---|
| `image` | `PIL.Image.Image`, `numpy.ndarray` (H×W×C, uint8), `torch.Tensor` (C×H×W, float) | The system applies Albumentations augmentations on the train split before converting to a tensor. **Do not** apply random augmentations yourself on the train split—the system will handle them. |
| `label` | `int`, `torch.Tensor` (scalar or vector) | For classification: a single integer class index. For regression / segmentation / detection: whatever your `training_step` and metric expect. |

### What the train split receives

The `AutoResearchDataModule` wraps your `train_dataset` with an Albumentations pipeline that
is rebuilt every trial.  Your dataset's `__getitem__` is called first, and the resulting
image is passed through the pipeline.  Therefore:

- Apply only **deterministic preprocessing** (resize, normalisation with fixed stats) inside
  `__getitem__`.
- Do **not** apply random flips, crops, colour jitter, etc.—those are searched by the system.

The `val_dataset` is **never augmented** beyond what your `__getitem__` returns.

---

## 3. CSV / DataFrame-Based Datasets

Many real-world datasets are organised as a CSV file that maps image paths to labels.
The table below lists the **minimum columns** a well-formed index CSV must contain,
followed by task-specific additions.

### 3.1 Universal columns (all tasks)

| Column name | dtype | Description |
|---|---|---|
| `image_path` | `str` | Absolute or relative path to the image file. |
| `split` | `str` | `"train"` or `"val"` (or `"test"`). Used to split the DataFrame. |

### 3.2 Classification columns

| Column name | dtype | Description |
|---|---|---|
| `label` | `int` | Zero-based integer class index, e.g. `0 … N-1`. |
| `class_name` | `str` *(optional)* | Human-readable class name; not used by the system but useful for debugging. |

### 3.3 Regression / depth estimation columns

| Column name | dtype | Description |
|---|---|---|
| `target` | `float` | Scalar regression target (e.g. depth in metres, age in years). |

### 3.4 Semantic segmentation columns

| Column name | dtype | Description |
|---|---|---|
| `mask_path` | `str` | Path to the PNG segmentation mask (each pixel = integer class index). |

### 3.5 Object detection columns

Because each image can contain multiple boxes, store one **row per bounding box**:

| Column name | dtype | Description |
|---|---|---|
| `x_min`, `y_min`, `x_max`, `y_max` | `float` | Box coordinates in pixel space. |
| `class_id` | `int` | Zero-based class index. |

You will then group by `image_path` inside `__getitem__` to build the target dict.

### 3.6 Boilerplate CSV Dataset wrapper

```python
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class CSVImageDataset(Dataset):
    """Generic CSV-backed image classification dataset.

    Args:
        csv_path: Path to the index CSV.
        split: One of "train" or "val".
        image_root: Optional prefix prepended to relative image_path values.
    """

    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        image_root: str | Path | None = None,
    ) -> None:
        self._df = pd.read_csv(csv_path)
        self._df = self._df[self._df["split"] == split].reset_index(drop=True)
        self._image_root = Path(image_root) if image_root else Path(".")

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        row = self._df.iloc[idx]
        image_path = self._image_root / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        label = int(row["label"])
        return image, label
```

Usage:

```python
train_ds = CSVImageDataset("dataset.csv", split="train", image_root="/data/images")
val_ds   = CSVImageDataset("dataset.csv", split="val",   image_root="/data/images")
```

---

## 4. Model Requirements

Your model must be a `pytorch_lightning.LightningModule` with two methods.

### 4.1 `forward(self, x: torch.Tensor) -> torch.Tensor`

- Input `x`: batch tensor of shape `(B, C, H, W)`, already on the correct device.
- Output: raw **logits** (unnormalised scores).  For classification this is shape
  `(B, num_classes)`.

### 4.2 `training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor`

- `batch` is a `(images, labels)` tuple provided by the dataloader.
- Must return a **scalar loss tensor** that PyTorch Lightning calls `.backward()` on.

### 4.3 `configure_optimizers` — leave it empty

The system **overrides** `configure_optimizers` by wrapping your module in
`AutoResearchModule`.  You can define the method, but its return value is ignored.  Either
omit it or define a no-op:

```python
def configure_optimizers(self):
    pass  # Managed by AutoResearchModule
```

### 4.4 Minimal model template

```python
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class MyModel(pl.LightningModule):
    """Your model — only forward and training_step are required."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Define your architecture here
        import torchvision.models as models
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = torch.nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        return F.cross_entropy(logits, labels)

    def configure_optimizers(self):
        pass  # Overridden by AutoResearchModule
```

### 4.5 The `dropout_rate` hyperparameter convention

The system searches a `dropout_rate` parameter in `[0.0, 0.7]`.  If you want the search
to actually affect your model you must consume it.  The injected value is available via
`self.hparams.dropout_rate` inside `AutoResearchModule`.  Pass it to your own module's
`__init__` if you want to use it:

```python
class MyModel(pl.LightningModule):
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.drop = torch.nn.Dropout(p=dropout_rate)
        ...
```

You do not have to support `dropout_rate`—unused hyperparameters are silently ignored.

---

## 5. SearchConfig Reference

`SearchConfig` is a plain Python dataclass.  All fields except the first three have
sensible defaults.

```python
from cv_autoresearch.config.schema import SearchConfig

config = SearchConfig(
    # ── Required ──────────────────────────────────────────────────────────
    task_description = "...",   # Free-text description Claude reads to choose metrics
    primary_metric   = "...",   # Name of the metric to optimise, e.g. "accuracy"
    higher_is_better = True,    # True for accuracy/F1/mAP; False for loss/RMSE

    # ── Trial budget ──────────────────────────────────────────────────────
    total_trials               = 80,  # Total Optuna trials across all iterations
    epochs_per_trial           = 10,  # Training epochs per trial
    exploit_trials_per_directive = 5, # Optuna trials per Claude directive

    # ── Infrastructure ────────────────────────────────────────────────────
    optuna_storage = "sqlite:///autoresearch.db",
    optuna_seed    = 42,
    device         = "cuda",  # "cuda" | "cpu" | "mps"
    num_workers    = 4,       # DataLoader worker processes

    # ── Output ────────────────────────────────────────────────────────────
    output_dir = "./autoresearch_output",

    # ── Optional overrides ────────────────────────────────────────────────
    hp_overrides  = {},  # Fix hyperparams; see §9
    aug_overrides = {},  # Fix augmentation params; see §9
)
```

### Field quick-reference

| Field | Type | When to change |
|---|---|---|
| `total_trials` | `int` | Increase for longer runs (default 80 ≈ 16 Claude directives × 5 trials). |
| `epochs_per_trial` | `int` | Reduce for fast datasets; increase if convergence is slow. |
| `exploit_trials_per_directive` | `int` | Controls how many Optuna trials run per single Claude directive. |
| `device` | `str` | Use `"mps"` on Apple Silicon, `"cpu"` for debugging. |
| `num_workers` | `int` | Set to `0` on Windows or if you encounter dataloader hangs. |

---

## 6. Writing a Good Task Description

`task_description` is the only free-form text Claude sees.  It drives two decisions:

1. **Metric selection** — Claude calls `metric_generator` to choose the right
   `torchmetrics` class and its parameters.
2. **Search strategy** — Claude's `search_director` uses the task description plus
   experiment history to pick the next parameter to tune.

### Checklist for a good description

| Include | Example |
|---|---|
| Task type | "binary classification", "10-class classification", "semantic segmentation" |
| Domain / dataset | "chest X-ray", "satellite imagery", "indoor scenes" |
| Number of classes | "7 categories", "1000 classes" |
| Class imbalance | "highly imbalanced: 95 % normal, 5 % defective" |
| Primary metric intent | "maximise macro F1", "minimise RMSE", "report mAP@50" |
| Input resolution hint | "224×224 RGB images", "512×512 grayscale" |

### Examples

**Good:**
```
"Multi-class skin lesion classification into 7 categories (HAM10000 dataset).
 Images are 224×224 RGB. The dataset is highly imbalanced; optimise for
 macro-averaged F1 score."
```

**Too vague:**
```
"Image classification"
```

---

## 7. Complete Worked Example

Below is a self-contained script for a custom 5-class flower dataset stored in a CSV.

```python
# my_flower_run.py
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
from PIL import Image
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

from cv_autoresearch import run_autoresearch
from cv_autoresearch.config.schema import SearchConfig


# ── 1. Dataset ────────────────────────────────────────────────────────────────

class FlowerDataset(Dataset):
    """5-class flower classification dataset backed by a CSV index.

    The CSV must have columns: image_path, label (int 0-4), split (train/val).
    """

    def __init__(self, csv_path: str, split: str, image_root: str) -> None:
        df = pd.read_csv(csv_path)
        self._df = df[df["split"] == split].reset_index(drop=True)
        self._root = Path(image_root)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        row = self._df.iloc[idx]
        img = Image.open(self._root / row["image_path"]).convert("RGB")
        return img, int(row["label"])


# ── 2. Model ──────────────────────────────────────────────────────────────────

class FlowerClassifier(pl.LightningModule):
    """MobileNetV3-Small fine-tuned for 5-class flower classification."""

    NUM_CLASSES = 5

    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = torch.nn.Linear(in_features, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        return F.cross_entropy(self(images), labels)

    def configure_optimizers(self):
        pass  # Managed by AutoResearchModule


# ── 3. Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_ds = FlowerDataset("flowers.csv", split="train", image_root="./data")
    val_ds   = FlowerDataset("flowers.csv", split="val",   image_root="./data")

    config = SearchConfig(
        task_description=(
            "5-class flower image classification (daisy, dandelion, rose, sunflower, tulip). "
            "Images are 224×224 RGB. Optimise top-1 multiclass accuracy."
        ),
        primary_metric   = "accuracy",
        higher_is_better = True,
        total_trials     = 60,
        epochs_per_trial = 8,
        device           = "cuda",
        output_dir       = "./flower_search",
    )

    result = run_autoresearch(FlowerClassifier(), train_ds, val_ds, config)

    print(f"\nBest accuracy : {result['best_metric']['value']:.4f}")
    print(f"Best hyperparams : {result['best_hyperparams']}")
    print(f"Best augmentations: {result['best_augmentations']}")
```

Expected CSV format (`flowers.csv`):

```
image_path,label,class_name,split
flowers/daisy/001.jpg,0,daisy,train
flowers/daisy/002.jpg,0,daisy,train
flowers/rose/001.jpg,2,rose,train
flowers/daisy/099.jpg,0,daisy,val
...
```

---

## 8. Running via the CLI

If your dataset factory and model are importable (i.e., on `PYTHONPATH`), you can run
without writing a script:

```bash
# Factory function must return (train_dataset, val_dataset)
cv-autoresearch run \
  --trainer-module    mypackage.models.FlowerClassifier \
  --train-dataset     mypackage.data.get_flower_datasets \
  --task              "5-class flower classification, optimise accuracy" \
  --metric            accuracy \
  --higher-is-better \
  --total-trials      60 \
  --epochs            8 \
  --device            cuda \
  --output            flower_result.json
```

The `--train-dataset` argument must point to a **factory function** that accepts no
arguments and returns a `(train_dataset, val_dataset)` tuple.

```python
# mypackage/data.py
def get_flower_datasets():
    train_ds = FlowerDataset("flowers.csv", split="train", image_root="./data")
    val_ds   = FlowerDataset("flowers.csv", split="val",   image_root="./data")
    return train_ds, val_ds
```

To resume an interrupted run:

```bash
cv-autoresearch resume \
  --storage  sqlite:///autoresearch.db \
  --trainer-module  mypackage.models.FlowerClassifier \
  --train-dataset   mypackage.data.get_flower_datasets \
  --task            "5-class flower classification, optimise accuracy" \
  --metric          accuracy \
  --higher-is-better
```

---

## 9. Advanced: Overriding Hyperparameters and Augmentations

Use `hp_overrides` and `aug_overrides` to **fix** parameters you do not want the system
to search.  This narrows the search space and speeds up convergence.

### Fix specific hyperparameters

```python
config = SearchConfig(
    ...,
    hp_overrides={
        "batch_size":    64,      # Always use batch size 64
        "optimizer_type": "adamw", # Lock optimizer
    },
)
```

### Disable specific augmentations

```python
config = SearchConfig(
    ...,
    aug_overrides={
        "VerticalFlip": None,          # Disable (not useful for natural images)
        "Normalize": {                 # Fix normalisation statistics
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225],
        },
    },
)
```

All parameters **not** in the override dicts remain searchable.

---

## 10. Output Files

After a run completes, the `output_dir` contains:

```
output_dir/
├── checkpoints/               # PyTorch Lightning checkpoints per trial
├── autoresearch.jsonl         # One JSON line per trial (trial_id, metric, config, …)
├── improvement_curve.png      # Plot of best metric vs. trial number
└── config/
    └── metrics/
        └── generated.yaml     # Claude-generated torchmetrics instantiation config
```

The `autoresearch.jsonl` file can be parsed to reconstruct the full experiment history:

```python
import json

with open("autoresearch_output/autoresearch.jsonl") as f:
    trials = [json.loads(line) for line in f]

best = max(trials, key=lambda t: t["primary_metric_value"] or -float("inf"))
print(best)
```
