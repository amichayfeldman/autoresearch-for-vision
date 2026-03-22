"""Summary report generation for completed autoresearch runs."""

from __future__ import annotations

from typing import Any

from cv_autoresearch.search.history import SearchHistory
from cv_autoresearch.types import Baseline


def generate_summary(
    baseline: Baseline,
    history: SearchHistory,
    primary_metric: str,
) -> dict[str, Any]:
    """Generate a summary dict for a completed autoresearch run.

    Args:
        baseline: Final best baseline found.
        history: Full experiment history.
        primary_metric: Name of the metric as inferred by the metric generator.

    Returns:
        Dict with best_metric, best_hyperparams, best_augmentations,
        total_trials, failed_trials, and top_improvements.
    """
    top_entries = history.best_entries(top_k=10)
    failed_count = len(history.failed_entries())

    return {
        "best_metric": {
            "name": primary_metric,
            "value": baseline.primary_metric_value,
        },
        "best_hyperparams": baseline.hyperparams,
        "best_augmentations": baseline.augmentation_config,
        "total_trials": len(history.entries),
        "failed_trials": failed_count,
        "top_improvements": [
            {
                "trial_id": e.trial_id,
                "param_name": e.param_name,
                "delta": e.delta,
                "metric_after": e.metric_after,
            }
            for e in top_entries
        ],
    }
