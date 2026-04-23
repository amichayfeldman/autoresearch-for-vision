"""Universal metric utilities for cv-autoresearch.

F1 is the default optimization target. Users can override with any MetricFn
(mIoU, mAP, binary F1, etc.) via TaskDef.metric_fn.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchmetrics.functional as F_tm


@dataclass
class Metrics:
    """Evaluation result for a single trial.

    Args:
        f1: Macro F1 score (primary optimization target when metric_fn is None).
        precision: Macro precision.
        recall: Macro recall.
    """

    f1: float
    precision: float
    recall: float


def compute(preds: list[int], targets: list[int], num_classes: int) -> Metrics:
    """Compute multiclass macro F1/precision/recall from integer labels.

    Stateless — no metric object state to reset between trials.
    Used as the default metric when TaskDef.metric_fn is None.

    Args:
        preds: Flat list of integer predicted class indices.
        targets: Flat list of integer ground-truth class indices.
        num_classes: Total number of classes.

    Returns:
        Metrics dataclass with f1, precision, and recall.
    """
    p, t = torch.tensor(preds), torch.tensor(targets)
    kw = dict(task="multiclass", num_classes=num_classes, average="macro")
    return Metrics(
        f1=F_tm.f1_score(p, t, **kw).item(),
        precision=F_tm.precision(p, t, **kw).item(),
        recall=F_tm.recall(p, t, **kw).item(),
    )


def make_multiclass_f1(num_classes: int):
    """Return a MetricFn for multiclass macro F1.

    Convenience factory so users can pass metric_fn explicitly when they
    want to be self-documenting rather than relying on the default.

    Args:
        num_classes: Number of classes.

    Returns:
        Callable (preds, targets) -> float.
    """
    def _fn(preds: list[int], targets: list[int]) -> float:
        return compute(preds, targets, num_classes).f1

    return _fn


def make_binary_f1(threshold: float = 0.5):
    """Return a MetricFn for binary F1 (segmentation, binary classification).

    Args:
        threshold: Decision threshold (unused here since preds are already int).

    Returns:
        Callable (preds, targets) -> float.
    """
    def _fn(preds: list[int], targets: list[int]) -> float:
        p, t = torch.tensor(preds), torch.tensor(targets)
        return F_tm.f1_score(p, t, task="binary").item()

    return _fn


def make_multilabel_f1(num_labels: int):
    """Return a MetricFn for multilabel macro F1 (multilabel classification).

    Args:
        num_labels: Number of labels.

    Returns:
        Callable (preds, targets) -> float.
    """
    def _fn(preds: list[int], targets: list[int]) -> float:
        p, t = torch.tensor(preds), torch.tensor(targets)
        kw = dict(task="multilabel", num_labels=num_labels, average="macro")
        return F_tm.f1_score(p, t, **kw).item()

    return _fn
