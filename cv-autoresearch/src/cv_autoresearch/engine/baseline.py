"""Pure functions for baseline management."""

from __future__ import annotations

from typing import Any

from cv_autoresearch.types import Baseline, TrialId


def should_update_baseline(
    current_value: float,
    baseline_value: float,
    higher_is_better: bool,
) -> bool:
    """Determine whether a new metric value should replace the baseline.

    Args:
        current_value: Metric value from the latest trial.
        baseline_value: Current best metric value.
        higher_is_better: True if larger metric values are preferred.

    Returns:
        True if current_value is strictly better than baseline_value.
    """
    if higher_is_better:
        return current_value > baseline_value
    return current_value < baseline_value


def update_baseline(
    baseline: Baseline,
    new_value: float,
    new_hp: dict[str, Any] | None = None,
    new_aug: dict[str, Any] | None = None,
    trial_id: TrialId | None = None,
) -> Baseline:
    """Return a new Baseline with updated fields.

    The original Baseline is never mutated.

    Args:
        baseline: Current baseline to update from.
        new_value: New primary metric value.
        new_hp: New hyperparams (if None, keeps existing).
        new_aug: New augmentation config (if None, keeps existing).
        trial_id: Trial ID that produced the improvement.

    Returns:
        New Baseline instance with updated values.
    """
    return Baseline(
        primary_metric_value=new_value,
        hyperparams=new_hp if new_hp is not None else dict(baseline.hyperparams),
        augmentation_config=new_aug if new_aug is not None else dict(baseline.augmentation_config),
        trial_id=trial_id,
    )
