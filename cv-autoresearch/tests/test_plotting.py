"""Tests for cv_autoresearch.engine.plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cv_autoresearch.engine.plotting import _build_plot_data, _make_label
from cv_autoresearch.search.history import HistoryEntry, SearchHistory
from cv_autoresearch.types import (
    Directive,
    SearchMode,
    TrialId,
    TrialStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIRECTIVE = Directive(
    mode=SearchMode.EXPLORE,
    target_param="learning_rate",
    target_range=None,
    reason="test",
)


def _make_entry(
    trial_id: int,
    param_name: str | None = None,
    param_value: Any = None,
    metric_after: float | None = None,
    improved: bool = False,
    status: TrialStatus = TrialStatus.SUCCESS,
) -> HistoryEntry:
    return HistoryEntry(
        trial_id=TrialId(trial_id),
        mode=SearchMode.EXPLORE,
        directive=_DIRECTIVE,
        param_name=param_name,
        param_value=param_value,
        metric_before=0.5,
        metric_after=metric_after,
        optuna_objective_value=metric_after,
        improved=improved,
        status=status,
        error_message=None,
    )


# ---------------------------------------------------------------------------
# _make_label
# ---------------------------------------------------------------------------


def test_make_label_baseline_trial_zero() -> None:
    entry = _make_entry(trial_id=0)
    assert _make_label(entry) == "baseline"


def test_make_label_no_param_name() -> None:
    entry = _make_entry(trial_id=1, param_name=None)
    assert _make_label(entry) == "baseline"


def test_make_label_hp_float_value() -> None:
    entry = _make_entry(trial_id=1, param_name="learning_rate", param_value=0.001234)
    label = _make_label(entry)
    assert label == "learning_rate=0.001234"


def test_make_label_hp_integer_value() -> None:
    entry = _make_entry(trial_id=1, param_name="batch_size", param_value=64)
    label = _make_label(entry)
    assert label == "batch_size=64"


def test_make_label_aug_enabled() -> None:
    entry = _make_entry(
        trial_id=1,
        param_name="HorizontalFlip_enabled",
        param_value=0.8,
    )
    assert _make_label(entry) == "HorizontalFlip_enabled=0.8"


def test_make_label_aug_disabled() -> None:
    entry = _make_entry(
        trial_id=1,
        param_name="HorizontalFlip_enabled",
        param_value=0.2,
    )
    assert _make_label(entry) == "HorizontalFlip_enabled=0.2"


# ---------------------------------------------------------------------------
# _build_plot_data — higher_is_better=True
# ---------------------------------------------------------------------------


def test_build_plot_data_empty_entries_returns_empty() -> None:
    trial_ids, running_best, imp_ids, imp_vals, labels = _build_plot_data([], True)
    assert trial_ids == []
    assert running_best == []
    assert imp_ids == []
    assert imp_vals == []
    assert labels == []


def test_build_plot_data_single_improvement_higher() -> None:
    entries = [_make_entry(0, metric_after=0.7, improved=True)]
    trial_ids, running_best, imp_ids, imp_vals, labels = _build_plot_data(entries, True)
    assert trial_ids == [0]
    assert running_best == [pytest.approx(0.7)]
    assert imp_ids == [0]
    assert imp_vals == [pytest.approx(0.7)]


def test_build_plot_data_running_best_monotonically_increases(higher: bool = True) -> None:
    entries = [
        _make_entry(0, metric_after=0.6, improved=True),
        _make_entry(1, metric_after=0.5, improved=False),   # no improvement
        _make_entry(2, metric_after=0.8, improved=True),
    ]
    _, running_best, imp_ids, _, _ = _build_plot_data(entries, True)
    assert running_best == [pytest.approx(0.6), pytest.approx(0.6), pytest.approx(0.8)]
    assert imp_ids == [0, 2]


def test_build_plot_data_failed_trial_uses_previous_best() -> None:
    entries = [
        _make_entry(0, metric_after=0.6, improved=True),
        _make_entry(1, status=TrialStatus.FAILED, metric_after=None, improved=False),
    ]
    _, running_best, _, _, _ = _build_plot_data(entries, True)
    # Failed trial should carry forward the previous best
    assert running_best[1] == pytest.approx(0.6)


def test_build_plot_data_failed_first_trial_returns_nan() -> None:
    import math

    entries = [_make_entry(0, status=TrialStatus.FAILED, metric_after=None, improved=False)]
    _, running_best, _, _, _ = _build_plot_data(entries, True)
    assert math.isnan(running_best[0])


# ---------------------------------------------------------------------------
# _build_plot_data — higher_is_better=False (lower is better, e.g. loss)
# ---------------------------------------------------------------------------


def test_build_plot_data_lower_is_better_improvement_marked() -> None:
    """For loss metrics, a decrease should be an improvement."""
    entries = [
        _make_entry(0, metric_after=2.0, improved=True),   # baseline was higher
        _make_entry(1, metric_after=1.5, improved=True),   # lower = better
        _make_entry(2, metric_after=1.8, improved=False),  # higher than 1.5 = worse
    ]
    _, running_best, imp_ids, imp_vals, _ = _build_plot_data(entries, False)
    assert imp_ids == [0, 1]
    assert imp_vals == [pytest.approx(2.0), pytest.approx(1.5)]
    # Running best should go from 2.0 → 1.5 → 1.5
    assert running_best == [pytest.approx(2.0), pytest.approx(1.5), pytest.approx(1.5)]


def test_build_plot_data_lower_is_better_all_trials_id_tracked() -> None:
    entries = [
        _make_entry(0, metric_after=1.0, improved=True),
        _make_entry(1, metric_after=0.8, improved=True),
        _make_entry(2, metric_after=0.9, improved=False),
    ]
    trial_ids, _, _, _, _ = _build_plot_data(entries, False)
    assert trial_ids == [0, 1, 2]


# ---------------------------------------------------------------------------
# plot_improvement_curve — smoke test (file created, no crash)
# ---------------------------------------------------------------------------


def test_plot_improvement_curve_creates_file(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from cv_autoresearch.engine.plotting import plot_improvement_curve

    history = SearchHistory()
    for i, (val, imp) in enumerate([(0.5, True), (0.4, False), (0.3, True)]):
        history.entries.append(_make_entry(i, metric_after=val, improved=imp))

    output = tmp_path / "curve.png"
    plot_improvement_curve(history, "val_loss", False, str(output))
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_improvement_curve_empty_history_no_crash(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from cv_autoresearch.engine.plotting import plot_improvement_curve

    history = SearchHistory()
    output = tmp_path / "curve.png"
    # Empty history should return silently without creating a file
    plot_improvement_curve(history, "val_acc", True, str(output))
    assert not output.exists()


def test_plot_improvement_curve_higher_is_better_title(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cv_autoresearch.engine.plotting import plot_improvement_curve

    history = SearchHistory()
    history.entries.append(_make_entry(0, metric_after=0.8, improved=True))
    output = tmp_path / "curve.png"
    plot_improvement_curve(history, "accuracy", True, str(output))
    # File created successfully with higher_is_better=True
    assert output.exists()


def test_plot_improvement_curve_lower_is_better_title(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from cv_autoresearch.engine.plotting import plot_improvement_curve

    history = SearchHistory()
    history.entries.append(_make_entry(0, metric_after=0.3, improved=True))
    output = tmp_path / "curve.png"
    plot_improvement_curve(history, "val_loss", False, str(output))
    assert output.exists()
