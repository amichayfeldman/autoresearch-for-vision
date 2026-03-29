"""Tests for cv_autoresearch.search.history."""

from __future__ import annotations

from typing import Any

import pytest

from cv_autoresearch.search.history import (
    HistoryEntry,
    SearchHistory,
    _fingerprint,
)
from cv_autoresearch.types import (
    Directive,
    SearchMode,
    TrialId,
    TrialStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DIRECTIVE = Directive(
    mode=SearchMode.EXPLORE,
    target_param="learning_rate",
    target_range=None,
    reason="test",
)


def _make_entry(
    trial_id: int = 1,
    mode: SearchMode = SearchMode.EXPLORE,
    param_name: str | None = None,
    param_value: Any = None,
    metric_before: float | None = None,
    metric_after: float | None = None,
    optuna_objective_value: float | None = None,
    improved: bool = False,
    status: TrialStatus = TrialStatus.SUCCESS,
    error_message: str | None = None,
) -> HistoryEntry:
    return HistoryEntry(
        trial_id=TrialId(trial_id),
        directive_id=0,
        mode=mode,
        directive=_DIRECTIVE,
        param_name=param_name,
        param_value=param_value,
        metric_before=metric_before,
        metric_after=metric_after,
        optuna_objective_value=optuna_objective_value,
        improved=improved,
        status=status,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# _fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_same_config_different_order() -> None:
    """SHA-256 fingerprint must be identical regardless of insertion order."""
    config_a = {"lr": 1e-3, "batch_size": 32, "dropout": 0.3}
    config_b = {"dropout": 0.3, "batch_size": 32, "lr": 1e-3}
    assert _fingerprint(config_a) == _fingerprint(config_b)


def test_fingerprint_different_configs_differ() -> None:
    """Different configs must produce different fingerprints."""
    config_a = {"lr": 1e-3}
    config_b = {"lr": 2e-3}
    assert _fingerprint(config_a) != _fingerprint(config_b)


def test_fingerprint_returns_hex_string() -> None:
    """Fingerprint must be a non-empty hex string."""
    fp = _fingerprint({"a": 1})
    assert isinstance(fp, str)
    assert len(fp) == 64  # SHA-256 hex digest length


# ---------------------------------------------------------------------------
# SearchHistory.is_duplicate / register
# ---------------------------------------------------------------------------


def test_is_duplicate_returns_false_for_new_config() -> None:
    history = SearchHistory()
    assert history.is_duplicate({"lr": 1e-3}) is False


def test_is_duplicate_returns_true_after_register() -> None:
    history = SearchHistory()
    config = {"lr": 1e-3, "bs": 32}
    history.register(config)
    assert history.is_duplicate(config) is True


def test_is_duplicate_order_independent() -> None:
    """Duplicate detection must be order-independent."""
    history = SearchHistory()
    history.register({"a": 1, "b": 2})
    assert history.is_duplicate({"b": 2, "a": 1}) is True


def test_register_different_config_not_duplicate() -> None:
    history = SearchHistory()
    history.register({"lr": 1e-3})
    assert history.is_duplicate({"lr": 2e-3}) is False


# ---------------------------------------------------------------------------
# SearchHistory.record
# ---------------------------------------------------------------------------


def test_record_appends_entry() -> None:
    history = SearchHistory()
    entry = _make_entry(trial_id=1)
    history.record(entry)
    assert len(history.entries) == 1
    assert history.entries[0] is entry


def test_record_multiple_entries() -> None:
    history = SearchHistory()
    for i in range(5):
        history.record(_make_entry(trial_id=i))
    assert len(history.entries) == 5


# ---------------------------------------------------------------------------
# HistoryEntry.delta
# ---------------------------------------------------------------------------


def test_history_entry_delta_when_both_metrics_present() -> None:
    entry = _make_entry(metric_before=0.80, metric_after=0.85)
    assert entry.delta == pytest.approx(0.05)


def test_history_entry_delta_negative_improvement() -> None:
    entry = _make_entry(metric_before=0.85, metric_after=0.80)
    assert entry.delta == pytest.approx(-0.05)


def test_history_entry_delta_when_metric_before_is_none() -> None:
    entry = _make_entry(metric_before=None, metric_after=0.85)
    assert entry.delta == 0.0


def test_history_entry_delta_when_metric_after_is_none() -> None:
    entry = _make_entry(metric_before=0.80, metric_after=None)
    assert entry.delta == 0.0


def test_history_entry_delta_when_both_none() -> None:
    entry = _make_entry(metric_before=None, metric_after=None)
    assert entry.delta == 0.0


# ---------------------------------------------------------------------------
# SearchHistory.to_text
# ---------------------------------------------------------------------------


def test_to_text_includes_step() -> None:
    history = SearchHistory()
    history.record(_make_entry(trial_id=1, param_name="lr"))
    text = history.to_text()
    assert "step=0" in text


def test_to_text_includes_failed_entry_with_all_failed() -> None:
    history = SearchHistory()
    history.record(
        _make_entry(
            trial_id=7,
            status=TrialStatus.FAILED,
            error_message="CUDA out of memory",
        )
    )
    text = history.to_text()
    assert "all_failed=True" in text


def test_to_text_includes_best_value() -> None:
    history = SearchHistory()
    history.record(
        _make_entry(
            trial_id=3,
            param_value=0.8765,
            metric_before=0.5,
            metric_after=0.8765,
            optuna_objective_value=0.8765,
        )
    )
    text = history.to_text()
    # New format: one aggregated line per directive step
    assert "best_value=0.8765" in text
    assert "best_delta=+0.376500" in text


def test_to_text_max_entries_limits_output() -> None:
    history = SearchHistory()
    for i in range(30):
        entry = HistoryEntry(
            trial_id=TrialId(i),
            directive_id=i,
            mode=SearchMode.EXPLORE,
            directive=_DIRECTIVE,
            param_name=None,
            param_value=None,
            metric_before=None,
            metric_after=None,
            optuna_objective_value=None,
            improved=False,
            status=TrialStatus.SUCCESS,
            error_message=None,
        )
        history.record(entry)
    text = history.to_text(max_entries=5)
    # Most recent 5 directive steps: ids 25-29
    assert "step=29" in text
    # Earliest directive step should not appear
    assert "step=0" not in text


def test_to_text_most_recent_first() -> None:
    """Most recent directive steps must appear first in the text output."""
    history = SearchHistory()
    for i in range(3):
        entry = HistoryEntry(
            trial_id=TrialId(i),
            directive_id=i,
            mode=SearchMode.EXPLORE,
            directive=_DIRECTIVE,
            param_name=None,
            param_value=None,
            metric_before=None,
            metric_after=None,
            optuna_objective_value=None,
            improved=False,
            status=TrialStatus.SUCCESS,
            error_message=None,
        )
        history.record(entry)
    text = history.to_text()
    # Most recent directive_id (2) should appear before earlier ones
    assert text.index("step=2") < text.index("step=0")


# ---------------------------------------------------------------------------
# SearchHistory.exploit_objective_values
# ---------------------------------------------------------------------------


def test_exploit_objective_values_returns_trajectory_for_param() -> None:
    history = SearchHistory()
    history.record(
        _make_entry(
            trial_id=1,
            mode=SearchMode.EXPLOIT,
            param_name="learning_rate",
            optuna_objective_value=0.81,
        )
    )
    history.record(
        _make_entry(
            trial_id=2,
            mode=SearchMode.EXPLOIT,
            param_name="learning_rate",
            optuna_objective_value=0.84,
        )
    )
    # Different param - should be excluded
    history.record(
        _make_entry(
            trial_id=3,
            mode=SearchMode.EXPLOIT,
            param_name="dropout_rate",
            optuna_objective_value=0.77,
        )
    )
    values = history.exploit_objective_values("learning_rate")
    assert values == pytest.approx([0.81, 0.84])


def test_exploit_objective_values_skips_none_values() -> None:
    history = SearchHistory()
    history.record(
        _make_entry(
            trial_id=1,
            mode=SearchMode.EXPLOIT,
            param_name="lr",
            optuna_objective_value=0.80,
        )
    )
    history.record(
        _make_entry(
            trial_id=2,
            mode=SearchMode.EXPLOIT,
            param_name="lr",
            optuna_objective_value=None,
        )
    )
    values = history.exploit_objective_values("lr")
    assert len(values) == 1
    assert values[0] == pytest.approx(0.80)


def test_exploit_objective_values_excludes_explore_entries() -> None:
    history = SearchHistory()
    history.record(
        _make_entry(
            trial_id=1,
            mode=SearchMode.EXPLORE,
            param_name="lr",
            optuna_objective_value=0.80,
        )
    )
    values = history.exploit_objective_values("lr")
    assert values == []


# ---------------------------------------------------------------------------
# SearchHistory.failed_entries / best_entries
# ---------------------------------------------------------------------------


def test_failed_entries_returns_only_failed() -> None:
    history = SearchHistory()
    history.record(_make_entry(trial_id=1, status=TrialStatus.SUCCESS))
    history.record(_make_entry(trial_id=2, status=TrialStatus.FAILED, error_message="err"))
    history.record(_make_entry(trial_id=3, status=TrialStatus.PRUNED))
    failed = history.failed_entries()
    assert len(failed) == 1
    assert failed[0].trial_id == 2


def test_best_entries_excludes_failed() -> None:
    history = SearchHistory()
    history.record(_make_entry(trial_id=1, status=TrialStatus.FAILED, metric_before=0.5, metric_after=0.9))
    history.record(_make_entry(trial_id=2, status=TrialStatus.SUCCESS, metric_before=0.7, metric_after=0.8))
    best = history.best_entries(top_k=10)
    ids = [e.trial_id for e in best]
    assert 1 not in ids
    assert 2 in ids


def test_best_entries_sorted_by_abs_delta_descending() -> None:
    history = SearchHistory()
    history.record(_make_entry(trial_id=1, status=TrialStatus.SUCCESS, metric_before=0.7, metric_after=0.75))
    history.record(_make_entry(trial_id=2, status=TrialStatus.SUCCESS, metric_before=0.5, metric_after=0.9))
    history.record(_make_entry(trial_id=3, status=TrialStatus.SUCCESS, metric_before=0.8, metric_after=0.82))
    best = history.best_entries(top_k=10)
    deltas = [abs(e.delta) for e in best]
    assert deltas == sorted(deltas, reverse=True)


def test_best_entries_respects_top_k() -> None:
    history = SearchHistory()
    for i in range(10):
        history.record(_make_entry(trial_id=i, status=TrialStatus.SUCCESS, metric_before=float(i) * 0.1, metric_after=float(i) * 0.1 + 0.05))
    best = history.best_entries(top_k=3)
    assert len(best) == 3
