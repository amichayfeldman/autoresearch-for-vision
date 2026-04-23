"""Lean autoresearch runner — train loop + search in one file.

User provides a TaskDef; this module handles everything else.
One universal metric (macro F1 by default, overridable) drives all decisions.
"""

from __future__ import annotations

from typing import Any

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from cv_autoresearch.advisor import get_next_directive
from cv_autoresearch.lightning.datamodule import AutoResearchDataModule
from cv_autoresearch.lightning.module import AutoResearchModule
from cv_autoresearch.metrics import compute
from cv_autoresearch.search.augmentations import build_augmentation_pipeline
from cv_autoresearch.search.history import HistoryEntry, SearchHistory
from cv_autoresearch.search.optimizer import create_study, run_exploit_phase
from cv_autoresearch.search.space import PARAM_REGISTRY
from cv_autoresearch.task import TaskDef
from cv_autoresearch.types import Directive, TrialId, TrialStatus

# HP param names produced by the search space — used to split full_config.
_KNOWN_HP: frozenset[str] = frozenset({
    "learning_rate", "weight_decay", "batch_size", "optimizer_type", "lr_scheduler",
    "lr_scheduler_gamma", "lr_scheduler_step_size", "warmup_epochs", "warmup_momentum",
    "gradient_clip_val", "label_smoothing", "dropout_rate", "mixed_precision", "ema_decay",
    "momentum", "beta1", "beta2",
})

# Augmentation transform names accepted by build_augmentation_pipeline.
_KNOWN_AUG: frozenset[str] = frozenset({
    "HorizontalFlip", "VerticalFlip", "RandomBrightnessContrast", "ColorJitter",
    "GaussianBlur", "GaussNoise", "RandomResizedCrop", "RandomScale", "Rotate",
    "Perspective", "CoarseDropout", "Sharpen", "CLAHE", "RandomGamma", "Normalize",
})

_DEFAULT_HP: dict[str, Any] = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "optimizer_type": "adam",
    "lr_scheduler": "cosine",
    "lr_scheduler_step_size": 7,  # overridden to epochs_per_trial at runtime
    "lr_scheduler_gamma": 0.1,
}


def run(
    task: TaskDef,
    *,
    total_directives: int = 20,
    trials_per_directive: int = 5,
    epochs_per_trial: int = 7,
    device: str = "cuda",
    num_workers: int = 4,
    optuna_storage: str = "sqlite:///autoresearch.db",
    optuna_seed: int = 42,
) -> dict[str, Any]:
    """Run AI-directed hyperparameter + augmentation search.

    Claude picks which parameter to vary; Optuna samples the value.
    One scalar metric (task.metric_fn or macro F1) drives all decisions.

    Args:
        task: TaskDef with model, datasets, loss_fn, predict, and optional metric_fn.
        total_directives: Number of Claude-directed search phases.
        trials_per_directive: Optuna trials per directive phase.
        epochs_per_trial: Training epochs per trial.
        device: PyTorch device string ("cuda" or "cpu").
        num_workers: DataLoader worker count.
        optuna_storage: Optuna storage URL.
        optuna_seed: Random seed for TPE sampler.

    Returns:
        Dict with best_f1, best_hp, best_aug, and total_trials.
    """
    metric_fn = task.metric_fn or (lambda p, t: compute(p, t, task.num_classes).f1)
    history = SearchHistory()
    study = create_study("cv_search", "maximize", optuna_storage, optuna_seed)

    # Baseline with default hyperparams and no augmentation.
    default_hp = {**_DEFAULT_HP, "lr_scheduler_step_size": epochs_per_trial}
    best_f1 = _train_and_eval(task, default_hp, {}, epochs_per_trial, device, num_workers, metric_fn)
    best_hp: dict[str, Any] = default_hp
    best_aug: dict[str, Any] = {}

    for directive_id in range(total_directives):
        directive: Directive = get_next_directive(task, history, best_f1)
        baseline_cfg = {**best_hp, **best_aug}
        param_range = list(
            directive.target_range or PARAM_REGISTRY.get(directive.target_param, [0.0, 1.0])
        )

        def trial_fn(full_cfg: dict[str, Any], _did: int = directive_id, _d: Directive = directive) -> float:
            nonlocal best_f1, best_hp, best_aug
            hp = {k: v for k, v in full_cfg.items() if k in _KNOWN_HP}
            aug = {k: v for k, v in full_cfg.items() if k in _KNOWN_AUG}
            trial_id = TrialId(len(history.entries))
            try:
                f1 = _train_and_eval(task, hp, aug, epochs_per_trial, device, num_workers, metric_fn)
                improved = f1 > best_f1
                history.record(HistoryEntry(
                    trial_id=trial_id,
                    directive_id=_did,
                    mode=_d.mode,
                    directive=_d,
                    param_name=_d.target_param,
                    param_value=full_cfg.get(_d.target_param),
                    metric_before=best_f1,
                    metric_after=f1,
                    optuna_objective_value=f1,
                    improved=improved,
                    status=TrialStatus.SUCCESS,
                    error_message=None,
                ))
                if improved:
                    best_f1, best_hp, best_aug = f1, hp, aug
                return f1
            except Exception as exc:  # noqa: BLE001
                history.record(HistoryEntry(
                    trial_id=trial_id,
                    directive_id=_did,
                    mode=_d.mode,
                    directive=_d,
                    param_name=_d.target_param,
                    param_value=None,
                    metric_before=best_f1,
                    metric_after=None,
                    optuna_objective_value=None,
                    improved=False,
                    status=TrialStatus.FAILED,
                    error_message=str(exc),
                ))
                return 0.0

        run_exploit_phase(
            study, trial_fn, history, baseline_cfg,
            directive.target_param, param_range, trials_per_directive,
        )

    return {
        "best_f1": best_f1,
        "best_hp": best_hp,
        "best_aug": best_aug,
        "total_trials": len(history.entries),
    }


def _train_and_eval(
    task: TaskDef,
    hp_config: dict[str, Any],
    aug_config: dict[str, Any],
    epochs: int,
    device: str,
    num_workers: int,
    metric_fn: Any,
) -> float:
    """Train for N epochs then run val inference, returning a scalar metric.

    Args:
        task: TaskDef with model, datasets, loss_fn, and predict.
        hp_config: Hyperparameter dict for this trial.
        aug_config: Augmentation config (nested transform-name → kwargs dict).
        epochs: Number of training epochs.
        device: PyTorch device string.
        num_workers: DataLoader worker count.
        metric_fn: Callable (preds, targets) -> float.

    Returns:
        Scalar metric value (higher = better).
    """
    aug = build_augmentation_pipeline(aug_config)
    wrapped = AutoResearchModule(task, hp_config)
    dm = AutoResearchDataModule(
        task.train_dataset,
        task.val_dataset,
        aug,
        hp_config.get("batch_size", 32),
        num_workers,
    )

    accelerator = "gpu" if device == "cuda" else device
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(wrapped, dm.train_dataloader())

    model_device = torch.device(device)
    model = wrapped.model.to(model_device).eval()
    loader = DataLoader(
        task.val_dataset,
        batch_size=hp_config.get("batch_size", 32),
        num_workers=num_workers,
        shuffle=False,
    )
    all_preds: list[int] = []
    all_targets: list[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch[0].to(model_device))
            p, t = task.predict(logits.cpu(), batch)
            all_preds.extend(p)
            all_targets.extend(t)
    model.train()

    return metric_fn(all_preds, all_targets)
