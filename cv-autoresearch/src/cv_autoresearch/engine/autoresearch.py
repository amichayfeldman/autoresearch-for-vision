"""Main autoresearch orchestrator."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import Dataset

from cv_autoresearch.advisor.metric_generator import generate_metric_config
from cv_autoresearch.advisor.search_director import get_next_directive
from cv_autoresearch.config.schema import SearchConfig
from cv_autoresearch.engine.baseline import should_update_baseline, update_baseline
from cv_autoresearch.engine.evaluator import (
    compute_primary_metric,
    instantiate_metric,
    run_val_inference,
)
from cv_autoresearch.engine.logger import RunLogger
from cv_autoresearch.engine.plotting import plot_improvement_curve
from cv_autoresearch.engine.reporting import generate_summary
from cv_autoresearch.lightning.datamodule import AutoResearchDataModule
from cv_autoresearch.lightning.module import AutoResearchModule
from cv_autoresearch.search.augmentations import build_augmentation_pipeline
from cv_autoresearch.search.history import HistoryEntry, SearchHistory
from cv_autoresearch.search.optimizer import create_study, run_exploit_phase
from cv_autoresearch.search.space import PARAM_REGISTRY
from cv_autoresearch.types import (
    Baseline,
    Directive,
    TrialId,
    TrialStatus,
)
from cv_autoresearch.vlm.hooks import VLMHooks

# Parameters produced by suggest_hyperparams — used to split full_config.
_KNOWN_HP_PARAMS: frozenset[str] = frozenset({
    "learning_rate", "weight_decay", "batch_size", "optimizer_type", "lr_scheduler",
    "lr_scheduler_gamma", "lr_scheduler_step_size", "warmup_epochs", "warmup_momentum",
    "gradient_clip_val", "label_smoothing", "dropout_rate", "mixed_precision", "ema_decay",
    "momentum", "beta1", "beta2",
})

# Augmentation transform names accepted by build_augmentation_pipeline.
_KNOWN_AUG_TRANSFORMS: frozenset[str] = frozenset({
    "HorizontalFlip", "VerticalFlip", "RandomBrightnessContrast", "ColorJitter",
    "GaussianBlur", "GaussNoise", "RandomResizedCrop", "RandomScale", "Rotate",
    "Perspective", "CoarseDropout", "Sharpen", "CLAHE", "RandomGamma", "Normalize",
})


def run_autoresearch(
    user_module: LightningModule,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: SearchConfig,
    vlm_hooks: VLMHooks | None = None,
) -> dict[str, Any]:
    """Run the AI-directed single-loop hyperparameter and augmentation search.

    Claude selects which parameter to vary each iteration; Optuna samples the
    magnitude for that one parameter.  Every trial changes exactly one
    parameter from the current baseline.

    Args:
        user_module: User's LightningModule with training_step and forward.
        train_dataset: Training dataset (augmentation injected by system).
        val_dataset: Validation dataset (no augmentation).
        config: SearchConfig controlling trial budgets and settings.
        vlm_hooks: Optional VLM hooks for analysis callbacks.

    Returns:
        Summary dict with best_metric, best_hyperparams, best_augmentations,
        total_trials, failed_trials, and top_improvements.
    """
    hooks = vlm_hooks or VLMHooks()
    history = SearchHistory()

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    metric_cfg, primary_metric, higher_is_better = generate_metric_config(
        config.task_description,
        config.metric_config_path,
    )
    metric = instantiate_metric(metric_cfg)

    with RunLogger(config.log_path) as logger:
        logger.log_run_start(
            config.task_description,
            primary_metric,
            higher_is_better,
            config.total_trials,
        )

        baseline = _run_initial_baseline(
            user_module, train_dataset, val_dataset, metric, config, logger
        )

        study = create_study(
            name=f"{primary_metric}_search",
            direction="maximize" if higher_is_better else "minimize",
            storage=config.optuna_storage,
            seed=config.optuna_seed,
        )

        directive_counter = 0
        while directive_counter < config.total_trials:
            directive = get_next_directive(
                config.task_description, history, baseline, config
            )
            directive_id = directive_counter
            param_range = list(directive.target_range or PARAM_REGISTRY.get(
                directive.target_param, [0.0, 1.0]
            ))
            full_baseline = {**baseline.hyperparams, **baseline.augmentation_config}
            run_exploit_phase(
                study,
                lambda cfg, _did=directive_id: _run_trial(
                    cfg, user_module, train_dataset, val_dataset, metric, config,
                    history, baseline, directive, hooks, logger, higher_is_better,
                    directive_id=_did,
                ),
                history,
                full_baseline,
                directive.target_param,
                param_range,
                config.exploit_trials_per_directive,
            )
            directive_counter += 1
            baseline = _refresh_baseline(history, baseline, higher_is_better)

        hooks.analyze_experiment(history, baseline)
        logger.log_run_end(baseline, len(history.entries))

    plot_improvement_curve(
        history,
        primary_metric,
        higher_is_better,
        config.plot_path,
    )

    return generate_summary(baseline, history, primary_metric)


def _run_initial_baseline(
    user_module: LightningModule,
    train_dataset: Dataset,
    val_dataset: Dataset,
    metric: Any,
    config: SearchConfig,
    logger: RunLogger,
) -> Baseline:
    """Train with default hyperparams and compute initial baseline metric.

    Args:
        user_module: User's LightningModule.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        metric: Instantiated torchmetrics.Metric.
        config: SearchConfig.
        logger: RunLogger (unused here, kept for future baseline logging).

    Returns:
        Initial Baseline with default hyperparams and zero augmentation.
    """
    default_hp: dict[str, Any] = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "optimizer_type": "adam",
        "lr_scheduler": "cosine",
        "lr_scheduler_step_size": config.epochs_per_trial,
        "lr_scheduler_gamma": 0.1,
    }
    default_aug: dict[str, Any] = {}

    value = _train_and_evaluate(
        user_module, default_hp, default_aug, train_dataset, val_dataset, metric, config
    )
    return Baseline(
        primary_metric_value=value,
        hyperparams=default_hp,
        augmentation_config=default_aug,
        trial_id=None,
    )


def _run_trial(
    full_config: dict[str, Any],
    user_module: LightningModule,
    train_dataset: Dataset,
    val_dataset: Dataset,
    metric: Any,
    config: SearchConfig,
    history: SearchHistory,
    baseline: Baseline,
    directive: Directive,
    hooks: VLMHooks,
    logger: RunLogger,
    higher_is_better: bool,
    directive_id: int = 0,
) -> float:
    """Execute one trial: train + evaluate, record result, call hooks.

    Splits ``full_config`` into HP and aug components, trains the model,
    and records the outcome.  On RuntimeError (OOM, NaN, etc.), records a
    FAILED entry and returns the worst possible value.

    Args:
        full_config: Flat config dict produced by exploit_space (HP + aug keys merged).
        user_module: User's LightningModule.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        metric: Instantiated torchmetrics.Metric.
        config: SearchConfig.
        history: SearchHistory to record into.
        baseline: Current baseline for computing delta.
        directive: Directive that produced this trial.
        hooks: VLM hooks for analysis.
        logger: RunLogger for JSONL event writing.
        higher_is_better: Optimization direction from metric generator.
        directive_id: ID of the directive step that spawned this trial. All
            Bayesian trials within one directive share the same directive_id.

    Returns:
        Metric value (or worst possible on failure).
    """
    trial_id = TrialId(len(history.entries))
    worst = -math.inf if higher_is_better else math.inf

    hp_config = {k: v for k, v in full_config.items() if k in _KNOWN_HP_PARAMS}
    aug_config = {k: v for k, v in full_config.items() if k in _KNOWN_AUG_TRANSFORMS}

    try:
        value = _train_and_evaluate(
            user_module, hp_config, aug_config, train_dataset, val_dataset, metric, config
        )
        improved = should_update_baseline(value, baseline.primary_metric_value, higher_is_better)
        param_value = full_config.get(directive.target_param)
        entry = HistoryEntry(
            trial_id=trial_id,
            directive_id=directive_id,
            mode=directive.mode,
            directive=directive,
            param_name=directive.target_param,
            param_value=param_value,
            metric_before=baseline.primary_metric_value,
            metric_after=value,
            optuna_objective_value=value,
            improved=improved,
            status=TrialStatus.SUCCESS,
            error_message=None,
        )
        history.record(entry)
        logger.log_trial(entry)
        hooks.analyze_iteration(entry, history, baseline)
        return value
    except Exception as exc:  # noqa: BLE001
        entry = HistoryEntry(
            trial_id=trial_id,
            directive_id=directive_id,
            mode=directive.mode,
            directive=directive,
            param_name=directive.target_param,
            param_value=None,
            metric_before=baseline.primary_metric_value,
            metric_after=None,
            optuna_objective_value=None,
            improved=False,
            status=TrialStatus.FAILED,
            error_message=str(exc),
        )
        history.record(entry)
        logger.log_trial(entry)
        hooks.analyze_iteration(entry, history, baseline)
        return worst


def _train_and_evaluate(
    user_module: LightningModule,
    hp_config: dict[str, Any],
    aug_config: dict[str, Any],
    train_dataset: Dataset,
    val_dataset: Dataset,
    metric: Any,
    config: SearchConfig,
) -> float:
    """Train model with given config and compute the primary metric.

    Args:
        user_module: User's LightningModule.
        hp_config: Hyperparameter config.
        aug_config: Augmentation config (may be empty for no augmentation).
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        metric: Instantiated torchmetrics.Metric.
        config: SearchConfig.

    Returns:
        Primary metric value on the validation set.
    """
    aug_pipeline = build_augmentation_pipeline(aug_config)
    batch_size = hp_config.get("batch_size", 32)

    wrapped = AutoResearchModule(user_module, hp_config)
    dm = AutoResearchDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        augmentation_pipeline=aug_pipeline,
        batch_size=batch_size,
        num_workers=config.num_workers,
    )

    trainer = Trainer(
        max_epochs=config.epochs_per_trial,
        accelerator=config.device if config.device != "cuda" else "gpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(wrapped, datamodule=dm)

    preds, targets = run_val_inference(
        wrapped,
        val_dataset,
        batch_size=batch_size,
        device=config.device,
        num_workers=config.num_workers,
    )
    return compute_primary_metric(preds, targets, metric)


def _refresh_baseline(
    history: SearchHistory,
    baseline: Baseline,
    higher_is_better: bool,
) -> Baseline:
    """Update baseline from history if a better trial was just recorded.

    Args:
        history: Full history including the latest entry.
        baseline: Current baseline.
        higher_is_better: Optimization direction.

    Returns:
        Updated baseline, or original if no improvement.
    """
    if not history.entries:
        return baseline
    last = history.entries[-1]
    if last.metric_after is None:
        return baseline
    if should_update_baseline(last.metric_after, baseline.primary_metric_value, higher_is_better):
        return update_baseline(
            baseline,
            last.metric_after,
            new_hp=None,
            new_aug=None,
            trial_id=last.trial_id,
        )
    return baseline
