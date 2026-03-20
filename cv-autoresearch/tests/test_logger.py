"""Tests for cv_autoresearch.engine.logger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cv_autoresearch.engine.logger import RunLogger, _serialise
from cv_autoresearch.search.history import HistoryEntry
from cv_autoresearch.types import (
    Baseline,
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
    trial_id: int = 0,
    mode: SearchMode = SearchMode.EXPLORE,
    param_name: str | None = "learning_rate",
    param_value: Any = 1e-3,
    metric_before: float = 0.5,
    metric_after: float | None = 0.6,
    improved: bool = True,
    status: TrialStatus = TrialStatus.SUCCESS,
    error_message: str | None = None,
) -> HistoryEntry:
    return HistoryEntry(
        trial_id=TrialId(trial_id),
        mode=mode,
        directive=_DIRECTIVE,
        param_name=param_name,
        param_value=param_value,
        metric_before=metric_before,
        metric_after=metric_after,
        optuna_objective_value=metric_after,
        improved=improved,
        status=status,
        error_message=error_message,
    )


def _make_baseline(value: float = 0.5) -> Baseline:
    return Baseline(
        primary_metric_value=value,
        hyperparams={"learning_rate": 1e-3},
        augmentation_config={},
        trial_id=None,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Context manager protocol
# ---------------------------------------------------------------------------


def test_run_logger_context_manager_creates_file(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    with RunLogger(str(log_path)):
        pass
    assert log_path.exists()


def test_run_logger_creates_parent_dirs(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "dir" / "run.jsonl"
    with RunLogger(str(log_path)):
        pass
    assert log_path.exists()


# ---------------------------------------------------------------------------
# log_run_start
# ---------------------------------------------------------------------------


def test_log_run_start_writes_correct_event(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    with RunLogger(str(log_path)) as logger:
        logger.log_run_start("classify cats", "val_acc", True, 50)

    records = _read_jsonl(log_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["event"] == "run_start"
    assert rec["task"] == "classify cats"
    assert rec["metric"] == "val_acc"
    assert rec["higher_is_better"] is True
    assert rec["total_trials"] == 50
    assert "ts" in rec


# ---------------------------------------------------------------------------
# log_trial
# ---------------------------------------------------------------------------


def test_log_trial_success_writes_correct_fields(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    entry = _make_entry(trial_id=1, metric_before=0.5, metric_after=0.65, improved=True)
    with RunLogger(str(log_path)) as logger:
        logger.log_trial(entry)

    records = _read_jsonl(log_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["event"] == "trial"
    assert rec["trial_id"] == 1
    assert "phase" not in rec
    assert rec["mode"] == "explore"
    assert rec["status"] == "success"
    assert rec["metric_before"] == pytest.approx(0.5)
    assert rec["metric_after"] == pytest.approx(0.65)
    assert rec["improved"] is True
    assert rec["error"] is None
    assert "ts" in rec


def test_log_trial_failed_writes_error(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    entry = _make_entry(
        metric_after=None,
        improved=False,
        status=TrialStatus.FAILED,
        error_message="CUDA OOM",
    )
    with RunLogger(str(log_path)) as logger:
        logger.log_trial(entry)

    records = _read_jsonl(log_path)
    rec = records[0]
    assert rec["status"] == "failed"
    assert rec["metric_after"] is None
    assert rec["error"] == "CUDA OOM"


def test_log_trial_delta_present(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    entry = _make_entry(metric_before=0.4, metric_after=0.7, improved=True)
    with RunLogger(str(log_path)) as logger:
        logger.log_trial(entry)

    records = _read_jsonl(log_path)
    assert records[0]["delta"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# log_run_end
# ---------------------------------------------------------------------------


def test_log_run_end_writes_correct_event(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    baseline = _make_baseline(0.91)
    with RunLogger(str(log_path)) as logger:
        logger.log_run_end(baseline, total_trials=42)

    records = _read_jsonl(log_path)
    rec = records[0]
    assert rec["event"] == "run_end"
    assert rec["best_metric"] == pytest.approx(0.91)
    assert rec["total_trials"] == 42
    assert "best_hyperparams" in rec
    assert "best_augmentations" in rec


# ---------------------------------------------------------------------------
# Full run sequence produces ordered events
# ---------------------------------------------------------------------------


def test_full_run_event_sequence(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    entry = _make_entry(trial_id=0)
    baseline = _make_baseline(0.8)

    with RunLogger(str(log_path)) as logger:
        logger.log_run_start("task", "val_acc", True, 10)
        logger.log_trial(entry)
        logger.log_run_end(baseline, 1)

    records = _read_jsonl(log_path)
    assert [r["event"] for r in records] == [
        "run_start",
        "trial",
        "run_end",
    ]


def test_log_is_flushed_after_each_write(tmp_path: Path) -> None:
    """File must be readable (non-empty) while logger is still open."""
    log_path = tmp_path / "run.jsonl"
    with RunLogger(str(log_path)) as logger:
        logger.log_run_start("task", "val_acc", True, 10)
        records = _read_jsonl(log_path)
        assert len(records) == 1


# ---------------------------------------------------------------------------
# _serialise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, 1),
        (1.5, 1.5),
        ("hello", "hello"),
        (True, True),
        (None, None),
        ([1, 2], [1, 2]),
        ({"a": 1}, {"a": 1}),
    ],
)
def test_serialise_json_native_types(value: Any, expected: Any) -> None:
    assert _serialise(value) == expected


def test_serialise_non_serialisable_becomes_string() -> None:
    class Unserializable:
        def __repr__(self) -> str:
            return "custom"

    result = _serialise(Unserializable())
    assert isinstance(result, str)


def test_serialise_nested_dict() -> None:
    value = {"a": {"b": [1, 2]}}
    assert _serialise(value) == {"a": {"b": [1, 2]}}
