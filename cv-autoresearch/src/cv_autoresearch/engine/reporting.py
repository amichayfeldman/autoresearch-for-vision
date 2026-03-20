"""Summary report generation for completed autoresearch runs."""

from __future__ import annotations

from typing import Any

from cv_autoresearch.config.schema import SearchConfig
from cv_autoresearch.search.history import SearchHistory
from cv_autoresearch.types import Baseline, SearchPhase


def generate_summary(
    baseline: Baseline,
    history: SearchHistory,
    config: SearchConfig,
) -> dict[str, Any]:
    """Generate a summary dict for a completed autoresearch run.

    Args:
        baseline: Final best baseline found.
        history: Full experiment history.
        config: SearchConfig used for the run.

    Returns:
        Dict with best_metric, best_hyperparams, best_augmentations,
        total_trials, top_improvements, and phase_breakdown.
    """
    top_entries = history.best_entries(top_k=10)
    hp_trials = sum(1 for e in history.entries if e.phase == SearchPhase.HYPERPARAMETER)
    aug_trials = sum(1 for e in history.entries if e.phase == SearchPhase.AUGMENTATION)
    failed_count = len(history.failed_entries())

    return {
        "best_metric": {
            "name": config.primary_metric,
            "value": baseline.primary_metric_value,
        },
        "best_hyperparams": baseline.hyperparams,
        "best_augmentations": baseline.augmentation_config,
        "total_trials": len(history.entries),
        "top_improvements": [
            {
                "trial_id": e.trial_id,
                "param_name": e.param_name,
                "delta": e.delta,
                "metric_after": e.metric_after,
            }
            for e in top_entries
        ],
        "phase_breakdown": {
            "hyperparameter_trials": hp_trials,
            "augmentation_trials": aug_trials,
            "failed_trials": failed_count,
        },
    }
