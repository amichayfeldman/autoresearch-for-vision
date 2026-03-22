# Task Definition Examples

Each file in this folder is a **fully self-contained, end-to-end runnable** script.
All datasets are downloaded automatically — no CSV files, no manual data preparation.

The core of every example is the `task_description` string you pass to `SearchConfig`.
Claude reads it to decide:

1. Which `torchmetrics` class to instantiate (**metric generation**)
2. Which hyperparameter or augmentation to vary next (**search direction**)

---

## Quick reference: task description → generated metric

| # | File | Dataset (auto-downloaded) | Task description excerpt | Generated metric |
|---|------|--------------------------|--------------------------|-----------------|
| 1 | `01_multiclass_classification.py` | CIFAR-10 | "10-class … top-1 accuracy" | `Accuracy(task="multiclass", num_classes=10, top_k=1)` |
| 2 | `02_binary_classification.py` | CIFAR-10 (binary split) | "binary … AUROC" | `AUROC(task="binary")` |
| 3 | `03_multilabel_classification.py` | PASCAL VOC 2012 Detection | "multi-label … 20 classes … macro mAP" | `AveragePrecision(task="multilabel", num_labels=20, average="macro")` |
| 4 | `04_semantic_segmentation.py` | PASCAL VOC 2012 Segmentation | "21 classes … void=255 … mIoU" | `JaccardIndex(task="multiclass", num_classes=21, ignore_index=255)` |
| 5 | `05_object_detection.py` | PASCAL VOC 2007 Detection | "bounding boxes … mAP@IoU 0.50" | `MeanAveragePrecision(iou_thresholds=[0.5])` |
| 6 | `06_regression_depth_estimation.py` | Synthetic (in-memory, no download) | "scalar regression … minimise RMSE" | `MeanSquaredError(squared=False)` |
| 7 | `07_anomaly_detection.py` | MNIST (digit 9 = anomaly) | "9:1 imbalance … AUROC" | `AUROC(task="binary")` |
| 8 | `08_medical_imbalanced_f1.py` | SVHN | "naturally imbalanced … macro F1" | `F1Score(task="multiclass", num_classes=10, average="macro")` |
| 9 | `09_top5_accuracy_imagenet.py` | CIFAR-100 | "100-class … top-5 accuracy" | `Accuracy(task="multiclass", num_classes=100, top_k=5)` |
| 10 | `10_finegrained_recognition.py` | STL-10 | "500 images/class … subtle differences … top-1" | `Accuracy(task="multiclass", num_classes=10, top_k=1)` |

---

## Why each metric is chosen

| Metric | When Claude selects it |
|--------|----------------------|
| `Accuracy(top_k=1)` | Standard balanced multi-class classification |
| `Accuracy(top_k=5)` | Many classes (≥50) where near-misses are acceptable |
| `AUROC` | Binary task with class imbalance or when no threshold should be fixed |
| `AveragePrecision` | Multi-label — each image belongs to multiple classes |
| `F1Score(average="macro")` | Multi-class with imbalance — all classes equally important |
| `JaccardIndex` | Pixel-level segmentation (mIoU) |
| `MeanAveragePrecision` | Object detection — boxes + scores + classes |
| `MeanSquaredError(squared=False)` | Scalar or dense regression (RMSE) |

---

## Running an example

```bash
cd /path/to/cv-autoresearch
python examples/task_definitions/01_multiclass_classification.py
```

Data is downloaded to `./data/` on first run.  Each example writes results to
`./output/<example_name>/`.

---

## What each file contains

```
# Module docstring
#   - Use case and dataset description
#   - Generated metric YAML and reasoning

TASK_DESCRIPTION = "..."   # String inserted into SearchConfig

def get_datasets()         # Factory returning (train_ds, val_ds) — no CSV, no local paths

class Model(...)           # LightningModule with forward + training_step

if __name__ == "__main__": # Calls run_autoresearch with SearchConfig
```

---

## Adapting an example to your own data

1. Copy the closest example and rename it.
2. Edit `TASK_DESCRIPTION` — be specific about task type, class count, imbalance,
   and desired metric.  See `docs/custom_usage_guide.md` for a checklist.
3. Replace `get_datasets()` with a function that returns your own
   `torch.utils.data.Dataset` objects (see guide for the `(image, label)` contract).
4. Replace the `Model` class with your architecture.
5. Adjust `total_trials`, `epochs_per_trial`, and `higher_is_better` to match
   your metric direction and compute budget.

---

## Notes on specific examples

**Example 5 (object detection):** `torchmetrics.detection.MeanAveragePrecision`
expects list-of-dict predictions, not stacked tensors.  The standard evaluator
returns stacked tensors.  This example runs as a dominant-class classification
proxy; see the file's docstring for how to extend it to true mAP evaluation.

**Examples 3–5 (VOC):** Data is downloaded from `host.robots.ox.ac.uk`.
If that server is temporarily unavailable, the download will fail.
Pre-downloaded data can be placed in `./data/VOC/` to skip the download.
