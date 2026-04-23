"""TaskDef dataclass — the user-facing contract for cv-autoresearch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset

# task.loss_fn(logits, batch) -> scalar loss tensor
LossFn = Callable[[Tensor, tuple], Tensor]

# task.predict(logits_cpu, batch) -> (int_preds, int_targets)
# Bridge from task-specific outputs to integer labels consumed by metric_fn.
# For segmentation: flatten (B,H,W) pixel preds/targets, filter void.
# For detection: argmax over class scores per proposal.
PredictFn = Callable[[Tensor, tuple], tuple[list[int], list[int]]]

# task.metric_fn(preds, targets) -> scalar float (higher = better).
# If None, autoresearch defaults to macro F1 via metrics.compute().
# Override for binary F1, mIoU, mAP, or any other scalar driving the search.
MetricFn = Callable[[list[int], list[int]], float]


@dataclass
class TaskDef:
    """User-facing task definition — the only thing the user needs to provide.

    Args:
        model: Plain nn.Module (no LightningModule subclassing required).
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        loss_fn: Task-specific loss callable: (logits, batch) -> scalar tensor.
        predict: Converts model outputs to integer preds/targets for metric_fn.
        num_classes: Number of classes (needed by default macro F1 metric).
        description: Free-text context for Claude's search director.
        metric_fn: Optional scalar metric from (preds, targets) -> float.
            Defaults to macro F1 when None. Override for segmentation (mIoU),
            detection (mAP), binary tasks, or any custom scoring function.
    """

    model: nn.Module
    train_dataset: Dataset
    val_dataset: Dataset
    loss_fn: LossFn
    predict: PredictFn
    num_classes: int
    description: str = field(default="")
    metric_fn: MetricFn | None = field(default=None)
