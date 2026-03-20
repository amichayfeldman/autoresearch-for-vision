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
from cv_autoresearch.search.optimizer import create_study, run_exploit_phase, run_explore_trial
from cv_autoresearch.search.space import suggest_augmentations, suggest_hyperparams
from cv_autoresearch.types import (
    Baseline,
    Directive,
    SearchMode,
    SearchPhase,
    TrialId,
    TrialStatus,
)
from cv_autoresearch.vlm.hooks import VLMHooks


def run_autoresearch(
    user_module: LightningModule,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: SearchConfig,
    vlm_hooks: VLMHooks | None = None,
) -> dict[str, Any]:
    """Run the full AI-directed hyperparameter and augmentation search.

    Orchestrates the two-phase search:
    1. Hyperparameter phase: Claude directs Optuna to EXPLORE/EXPLOIT
       hyperparameters while keeping augmentations fixed.
    2. Augmentation phase: Claude directs Optuna to EXPLORE/EXPLOIT
       augmentation transforms with hyperparams fixed to best found.

    Args:
        user_module: User's LightningModule with training_step and forward.
        train_dataset: Training dataset (augmentation injected by system).
        val_dataset: Validation dataset (no augmentation).
        config: SearchConfig controlling trial budgets and settings.
        vlm_hooks: Optional VLM hooks for analysis callbacks.

    Returns:
        Summary dict with best_metric, best_hyperparams, best_augmentations,
        total_trials, top_improvements, and phase_breakdown.
    """
    hooks = vlm_hooks or VLMHooks()
    history = SearchHistory()

    # Ensure output directory tree exists
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    with RunLogger(config.log_path) as logger:
        logger.log_run_start(
            config.task_description,
            config.primary_metric,
            config.higher_is_better,
            config.hp_trials,
            config.aug_trials,
        )

        # Startup: generate metric config via Claude (once)
        metric_cfg = generate_metric_config(
            config.task_description,
            config.primary_metric,
            config.metric_config_path,
        )
        metric = instantiate_metric(metric_cfg)

        # Run initial baseline (default config, no augmentation)
        baseline = _run_initial_baseline(
            user_module, train_dataset, val_dataset, metric, config, logger
        )

        hp_study = create_study(
            name=f"{config.primary_metric}_hp",
            direction="maximize" if config.higher_is_better else "minimize",
            storage=config.optuna_storage,
            seed=config.optuna_seed,
        )

        # Phase 1: Hyperparameter search
        hp_budget = config.hp_trials
        trial_counter = 0

        while trial_counter < hp_budget:
            directive = get_next_directive(
                config.task_description, history, baseline, SearchPhase.HYPERPARAMETER, config
            )

            if directive.mode == SearchMode.EXPLOIT and directive.target_param:
                param_range = list(directive.target_range or [0.0, 1.0])
                results = run_exploit_phase(
                    hp_study,
                    lambda cfg: _run_trial(
                        cfg, baseline.augmentation_config, user_module, train_dataset,
                        val_dataset, metric, config, history, baseline, directive,
                        SearchPhase.HYPERPARAMETER, hooks, logger,
                    ),
                    history,
                    baseline.hyperparams,
                    directive.target_param,
                    param_range,
                    config.exploit_trials_per_directive,
                )
                trial_counter += len(results)
                baseline = _refresh_baseline(history, baseline, config.higher_is_better)
            else:
                run_explore_trial(
                    hp_study,
                    lambda cfg: _run_trial(
                        cfg, baseline.augmentation_config, user_module, train_dataset,
                        val_dataset, metric, config, history, baseline, directive,
                        SearchPhase.HYPERPARAMETER, hooks, logger,
                    ),
                    history,
                    lambda trial: suggest_hyperparams(trial, config.hp_overrides),
                )
                trial_counter += 1
                baseline = _refresh_baseline(history, baseline, config.higher_is_better)

        hp_trial_count = trial_counter
        hooks.analyze_phase(SearchPhase.HYPERPARAMETER, hp_study, baseline)
        logger.log_phase_end(
            SearchPhase.HYPERPARAMETER,
            baseline.primary_metric_value,
            sum(1 for e in history.entries if e.phase == SearchPhase.HYPERPARAMETER),
        )

        aug_study = create_study(
            name=f"{config.primary_metric}_aug",
            direction="maximize" if config.higher_is_better else "minimize",
            storage=config.optuna_storage,
            seed=config.optuna_seed,
        )

        # Phase 2: Augmentation search (hyperparams fixed to best)
        aug_budget = config.aug_trials
        aug_counter = 0

        while aug_counter < aug_budget:
            directive = get_next_directive(
                config.task_description, history, baseline, SearchPhase.AUGMENTATION, config
            )

            if directive.mode == SearchMode.EXPLOIT and directive.target_param:
                param_range = list(directive.target_range or [0.0, 1.0])
                results = run_exploit_phase(
                    aug_study,
                    lambda cfg: _run_trial(
                        baseline.hyperparams, cfg, user_module, train_dataset,
                        val_dataset, metric, config, history, baseline, directive,
                        SearchPhase.AUGMENTATION, hooks, logger,
                    ),
                    history,
                    baseline.augmentation_config,
                    directive.target_param,
                    param_range,
                    config.exploit_trials_per_directive,
                )
                aug_counter += len(results)
                baseline = _refresh_baseline(history, baseline, config.higher_is_better)
            else:
                run_explore_trial(
                    aug_study,
                    lambda cfg: _run_trial(
                        baseline.hyperparams, cfg, user_module, train_dataset,
                        val_dataset, metric, config, history, baseline, directive,
                        SearchPhase.AUGMENTATION, hooks, logger,
                    ),
                    history,
                    lambda trial: suggest_augmentations(trial, config.aug_overrides),
                )
                aug_counter += 1
                baseline = _refresh_baseline(history, baseline, config.higher_is_better)

        hooks.analyze_phase(SearchPhase.AUGMENTATION, aug_study, baseline)
        hooks.analyze_experiment(history, baseline)
        logger.log_phase_end(
            SearchPhase.AUGMENTATION,
            baseline.primary_metric_value,
            sum(1 for e in history.entries if e.phase == SearchPhase.AUGMENTATION),
        )
        logger.log_run_end(baseline, len(history.entries))

    # Generate improvement plot after logger is closed
    plot_improvement_curve(
        history,
        config.primary_metric,
        config.higher_is_better,
        config.plot_path,
        hp_trial_count,
    )

    return generate_summary(baseline, history, config)


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
    hp_config: dict[str, Any],
    aug_config: dict[str, Any],
    user_module: LightningModule,
    train_dataset: Dataset,
    val_dataset: Dataset,
    metric: Any,
    config: SearchConfig,
    history: SearchHistory,
    baseline: Baseline,
    directive: Directive,
    phase: SearchPhase,
    hooks: VLMHooks,
    logger: RunLogger,
) -> float:
    """Execute one trial: train + evaluate, record result, call hooks.

    On RuntimeError (OOM, NaN, etc.), records a FAILED entry and returns
    the worst possible value so Optuna deprioritizes that region.

    Args:
        hp_config: Hyperparameter config for this trial.
        aug_config: Augmentation config for this trial.
        user_module: User's LightningModule.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        metric: Instantiated torchmetrics.Metric.
        config: SearchConfig.
        history: SearchHistory to record into.
        baseline: Current baseline for computing delta.
        directive: Directive that produced this trial.
        phase: Current search phase.
        hooks: VLM hooks for analysis.

    Returns:
        Metric value (or worst possible on failure).
    """
    trial_id = TrialId(len(history.entries))
    worst = -math.inf if config.higher_is_better else math.inf

    try:
        value = _train_and_evaluate(
            user_module, hp_config, aug_config, train_dataset, val_dataset, metric, config
        )
        improved = should_update_baseline(value, baseline.primary_metric_value, config.higher_is_better)
        entry = HistoryEntry(
            trial_id=trial_id,
            phase=phase,
            mode=directive.mode,
            directive=directive,
            param_name=directive.target_param,
            param_value=hp_config.get(directive.target_param) if directive.target_param else None,
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
            phase=phase,
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
