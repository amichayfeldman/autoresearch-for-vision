"""Tests for cv_autoresearch.types."""

from __future__ import annotations

import pytest

from cv_autoresearch.types import (
    Baseline,
    Directive,
    IterationResult,
    SearchMode,
    SearchPhase,
    TrialId,
    TrialStatus,
)


def test_trial_id_is_int_newtype():
    tid = TrialId(42)
    assert isinstance(tid, int)
    assert tid == 42


def test_search_phase_values():
    assert SearchPhase.HYPERPARAMETER.value == "hyperparameter"
    assert SearchPhase.AUGMENTATION.value == "augmentation"


def test_search_mode_values():
    assert SearchMode.EXPLORE.value == "explore"
    assert SearchMode.EXPLOIT.value == "exploit"


def test_trial_status_values():
    assert TrialStatus.SUCCESS.value == "success"
    assert TrialStatus.FAILED.value == "failed"
    assert TrialStatus.PRUNED.value == "pruned"


def test_directive_is_frozen():
    directive = Directive(
        mode=SearchMode.EXPLORE,
        target_param="lr",
        target_range=(1e-4, 1e-2),
        phase=SearchPhase.HYPERPARAMETER,
        reason="Initial exploration",
    )
    with pytest.raises(Exception):
        directive.mode = SearchMode.EXPLOIT  # type: ignore[misc]


def test_directive_all_fields():
    directive = Directive(
        mode=SearchMode.EXPLOIT,
        target_param="dropout",
        target_range=[0.1, 0.5],
        phase=SearchPhase.AUGMENTATION,
        reason="Exploiting known good region",
    )
    assert directive.mode == SearchMode.EXPLOIT
    assert directive.target_param == "dropout"
    assert directive.target_range == [0.1, 0.5]
    assert directive.phase == SearchPhase.AUGMENTATION
    assert directive.reason == "Exploiting known good region"


def test_directive_target_param_none():
    directive = Directive(
        mode=SearchMode.EXPLORE,
        target_param=None,
        target_range=None,
        phase=SearchPhase.HYPERPARAMETER,
        reason="Broad search",
    )
    assert directive.target_param is None
    assert directive.target_range is None


def test_baseline_is_mutable():
    baseline = Baseline(
        primary_metric_value=0.85,
        hyperparams={"lr": 1e-3},
        augmentation_config={"flip": True},
    )
    baseline.primary_metric_value = 0.90
    assert baseline.primary_metric_value == 0.90


def test_baseline_trial_id_defaults_to_none():
    baseline = Baseline(
        primary_metric_value=0.75,
        hyperparams={},
        augmentation_config={},
    )
    assert baseline.trial_id is None


def test_baseline_with_trial_id():
    baseline = Baseline(
        primary_metric_value=0.80,
        hyperparams={"lr": 1e-4},
        augmentation_config={"rotate": 30},
        trial_id=TrialId(7),
    )
    assert baseline.trial_id == 7


def test_iteration_result_is_frozen():
    directive = Directive(
        mode=SearchMode.EXPLORE,
        target_param=None,
        target_range=None,
        phase=SearchPhase.HYPERPARAMETER,
        reason="test",
    )
    result = IterationResult(
        trial_id=TrialId(1),
        phase=SearchPhase.HYPERPARAMETER,
        mode=SearchMode.EXPLORE,
        status=TrialStatus.SUCCESS,
        metrics={"val_acc": 0.88},
        primary_metric_value=0.88,
        optuna_objective_value=0.88,
        config_snapshot={"lr": 1e-3},
        improved=True,
        error_message=None,
        directive=directive,
    )
    with pytest.raises(Exception):
        result.improved = False  # type: ignore[misc]


def test_iteration_result_all_fields():
    directive = Directive(
        mode=SearchMode.EXPLOIT,
        target_param="lr",
        target_range=(1e-4, 1e-2),
        phase=SearchPhase.HYPERPARAMETER,
        reason="exploit lr",
    )
    result = IterationResult(
        trial_id=TrialId(3),
        phase=SearchPhase.HYPERPARAMETER,
        mode=SearchMode.EXPLOIT,
        status=TrialStatus.PRUNED,
        metrics={"val_loss": 0.5},
        primary_metric_value=None,
        optuna_objective_value=None,
        config_snapshot={},
        improved=False,
        error_message="pruned early",
        directive=directive,
    )
    assert result.trial_id == 3
    assert result.status == TrialStatus.PRUNED
    assert result.primary_metric_value is None
    assert result.error_message == "pruned early"
    assert result.directive is directive
