"""Default agent-editable task wiring.

The manager prompts an agent to adapt this module, and related training/data/model
code, for the user's task prompt. The stable runtime contract is:

- ``build_task(config)`` returns a datamodule and LightningModule.
- ``pretrain_evaluate(config)`` verifies wiring by returning at least precision and recall.
- ``evaluate_after_training(task, trainer, max_epochs)`` returns final metrics
  and epoch metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from pytorch_lightning import Trainer, seed_everything

from cv_autoresearch.engine.data import VisionDataModule
from cv_autoresearch.engine.training.module import VisionLightningModule


@dataclass
class AgentWiredTask:
    """Runtime objects built by agent-wired task code."""

    datamodule: VisionDataModule
    module: VisionLightningModule


def build_task(config: Any) -> AgentWiredTask:
    """Build task-specific data/model/training objects.

    The default is a tiny synthetic prompt-flow task so the framework can be
    tested offline. Real task wiring is expected to be supplied by the agent.
    """
    seed_everything(int(config.get("seed", 123)), workers=True)
    datamodule = VisionDataModule(config)
    module = VisionLightningModule(config)
    return AgentWiredTask(datamodule=datamodule, module=module)


def pretrain_evaluate(config: Any) -> dict[str, float]:
    """Run the task-wiring verification gate and return precision/recall."""
    task = build_task(config)
    task.module.eval()
    preds, targets = _collect_predictions(task)
    metrics = _precision_recall_metrics(
        preds,
        targets,
        num_classes=int(config.evaluation.get("num_classes", 2)),
    )
    return _with_f1(metrics)


def evaluate_after_training(
    task: AgentWiredTask,
    trainer: Trainer,
    max_epochs: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Evaluate a trained task and return final and epoch metrics."""
    task.module.eval()
    preds, targets = _collect_predictions(task)
    metrics = _precision_recall_metrics(
        preds,
        targets,
        num_classes=int(task.module.config_ref.evaluation.get("num_classes", 2)),
    )
    return _with_f1(metrics), _epoch_metrics(trainer, max_epochs, metrics)


def _collect_predictions(task: AgentWiredTask) -> tuple[list[int], list[int]]:
    preds: list[int] = []
    targets: list[int] = []
    for batch in task.datamodule.val_dataloader():
        batch_preds, batch_targets = task.module.predict_batch(batch)
        preds.extend(batch_preds)
        targets.extend(batch_targets)
    return preds, targets


def _precision_recall_metrics(
    preds: list[int],
    targets: list[int],
    *,
    num_classes: int,
) -> dict[str, float]:
    """Tiny synthetic fallback; real task metric extraction is agent-wired."""
    pred_tensor = torch.as_tensor(preds, dtype=torch.long)
    target_tensor = torch.as_tensor(targets, dtype=torch.long)
    if pred_tensor.numel() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for cls in range(num_classes):
        pred_pos = pred_tensor == cls
        target_pos = target_tensor == cls
        tp = torch.logical_and(pred_pos, target_pos).sum().item()
        fp = torch.logical_and(pred_pos, ~target_pos).sum().item()
        fn = torch.logical_and(~pred_pos, target_pos).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))
    return {
        "precision": sum(precisions) / num_classes,
        "recall": sum(recalls) / num_classes,
        "f1": sum(f1s) / num_classes,
    }


def _with_f1(metrics: dict[str, float]) -> dict[str, float]:
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    if "f1" not in metrics and precision is not None and recall is not None:
        metrics = dict(metrics)
        metrics["f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return metrics


def _epoch_metrics(
    trainer: Trainer,
    max_epochs: int,
    final_metrics: dict[str, float],
) -> list[dict[str, float]]:
    f1 = final_metrics.get("f1")
    if f1 is None:
        precision = final_metrics.get("precision", 0.0)
        recall = final_metrics.get("recall", 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return [{"epoch": epoch, "f1": float(f1)} for epoch in range(1, max_epochs + 1)]
