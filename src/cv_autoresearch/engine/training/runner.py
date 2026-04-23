"""Single configured training/evaluation execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Trainer

from cv_autoresearch.engine.task_wiring import build_task, evaluate_after_training, pretrain_evaluate
from cv_autoresearch.engine.utils import ensure_dir, to_plain_config, write_json


@dataclass(frozen=True)
class TrainingRunResult:
    """Artifacts and metrics from one training run."""

    output_dir: Path
    checkpoint_path: Path
    config_path: Path
    metrics_path: Path
    metrics: dict[str, float]
    epoch_metrics: list[dict[str, float]]
    pretrain_metrics: dict[str, float]


def run_training(config: Any, *, output_dir: str | Path | None = None) -> TrainingRunResult:
    """Train/evaluate one resolved Hydra config and save artifacts."""
    run_dir = ensure_dir(output_dir or config.history.output_dir)
    write_json(run_dir / "resolved_config.json", to_plain_config(config))
    pretrain_metrics = validate_pretrain_metrics(pretrain_evaluate(config))
    write_json(run_dir / "pretrain_metrics.json", pretrain_metrics)

    task = build_task(config)
    max_time = None
    if config.iteration.get("max_time_minutes") is not None:
        max_time = {"minutes": float(config.iteration.max_time_minutes)}

    trainer = Trainer(
        max_epochs=int(config.iteration.max_epochs),
        max_time=max_time,
        accelerator=str(config.trainer.get("accelerator", "cpu")),
        devices=int(config.trainer.get("devices", 1)),
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=bool(config.trainer.get("enable_progress_bar", False)),
        logger=False,
        deterministic=bool(config.trainer.get("deterministic", True)),
        limit_train_batches=config.trainer.get("limit_train_batches", 1.0),
        limit_val_batches=config.trainer.get("limit_val_batches", 1.0),
    )
    trainer.fit(task.module, datamodule=task.datamodule)

    metrics, epoch_metrics = evaluate_after_training(task, trainer, int(config.iteration.max_epochs))
    metrics = _with_derived_f1(metrics)
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save({"state_dict": task.module.state_dict(), "config": to_plain_config(config)}, checkpoint_path)
    metrics_path = write_json(run_dir / "metrics.json", metrics)
    return TrainingRunResult(
        output_dir=run_dir,
        checkpoint_path=checkpoint_path,
        config_path=run_dir / "resolved_config.json",
        metrics_path=metrics_path,
        metrics=metrics,
        epoch_metrics=epoch_metrics,
        pretrain_metrics=pretrain_metrics,
    )


def validate_pretrain_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Require numeric precision and recall for the task-wiring gate."""
    normalized = {key: float(value) for key, value in dict(metrics).items()}
    missing = [name for name in ("precision", "recall") if name not in normalized]
    if missing:
        raise ValueError(f"task wiring verification missing required metric(s): {', '.join(missing)}")
    invalid = [name for name in ("precision", "recall") if not math.isfinite(normalized[name])]
    if invalid:
        raise ValueError(f"task wiring verification has non-finite metric(s): {', '.join(invalid)}")
    return _with_derived_f1(normalized)


def _with_derived_f1(metrics: dict[str, float]) -> dict[str, float]:
    if "f1" in metrics:
        return metrics
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    if precision is None or recall is None:
        return metrics
    metrics = dict(metrics)
    metrics["f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return metrics
