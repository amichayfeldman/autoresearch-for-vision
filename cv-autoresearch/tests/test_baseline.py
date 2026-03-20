"""Tests for cv_autoresearch.engine.baseline."""

from __future__ import annotations

import pytest

from cv_autoresearch.engine.baseline import should_update_baseline, update_baseline
from cv_autoresearch.types import Baseline, TrialId


@pytest.fixture()
def base() -> Baseline:
    return Baseline(
        primary_metric_value=0.8,
        hyperparams={"lr": 1e-3},
        augmentation_config={"flip": True},
        trial_id=TrialId(1),
    )


@pytest.mark.parametrize(
    "current, baseline_val, higher_is_better, expected",
    [
        (0.9, 0.8, True, True),    # higher is better: 0.9 > 0.8
        (0.7, 0.8, True, False),   # higher is better: 0.7 < 0.8
        (0.8, 0.8, True, False),   # equal: no improvement
        (0.1, 0.8, False, True),   # lower is better: 0.1 < 0.8
        (0.9, 0.8, False, False),  # lower is better: 0.9 > 0.8
        (0.8, 0.8, False, False),  # equal: no improvement
    ],
)
def test_should_update_baseline(
    current: float,
    baseline_val: float,
    higher_is_better: bool,
    expected: bool,
) -> None:
    assert should_update_baseline(current, baseline_val, higher_is_better) == expected


def test_update_baseline_returns_new_instance(base: Baseline) -> None:
    """update_baseline must not mutate the original Baseline."""
    new = update_baseline(base, new_value=0.9, trial_id=TrialId(2))
    assert new is not base
    assert base.primary_metric_value == 0.8  # original unchanged


def test_update_baseline_new_value(base: Baseline) -> None:
    new = update_baseline(base, new_value=0.95)
    assert new.primary_metric_value == 0.95


def test_update_baseline_preserves_hyperparams_when_none(base: Baseline) -> None:
    new = update_baseline(base, new_value=0.9)
    assert new.hyperparams == base.hyperparams


def test_update_baseline_replaces_hyperparams(base: Baseline) -> None:
    new_hp = {"lr": 5e-4, "batch_size": 64}
    new = update_baseline(base, new_value=0.9, new_hp=new_hp)
    assert new.hyperparams == new_hp


def test_update_baseline_preserves_aug_when_none(base: Baseline) -> None:
    new = update_baseline(base, new_value=0.9)
    assert new.augmentation_config == base.augmentation_config


def test_update_baseline_replaces_aug(base: Baseline) -> None:
    new_aug = {"rotate": True}
    new = update_baseline(base, new_value=0.9, new_aug=new_aug)
    assert new.augmentation_config == new_aug


def test_update_baseline_sets_trial_id(base: Baseline) -> None:
    new = update_baseline(base, new_value=0.9, trial_id=TrialId(7))
    assert new.trial_id == TrialId(7)
