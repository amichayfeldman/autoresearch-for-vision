"""Structured JSONL run logger for autoresearch sessions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cv_autoresearch.search.history import HistoryEntry
from cv_autoresearch.types import Baseline


def _now() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


class RunLogger:
    """Writes structured JSONL events to a log file during an autoresearch run.

    Each line in the output file is a self-contained JSON object with an
    ``event`` field and a ``ts`` (ISO-8601 UTC timestamp). The log can be
    loaded for offline analysis with:

    .. code-block:: python

        import pandas as pd
        df = pd.read_json("autoresearch.jsonl", lines=True)

    Event types:
        - ``run_start``: written once at the beginning of a run.
        - ``trial``: written after every trial (success or failure).
        - ``run_end``: written once at the end of a run.
    """

    def __init__(self, log_path: str) -> None:
        """Open the log file for appending.

        Args:
            log_path: Path to the ``.jsonl`` log file. Parent directories
                are created automatically.
        """
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def log_run_start(
        self,
        task_description: str,
        primary_metric: str,
        higher_is_better: bool,
        total_trials: int,
    ) -> None:
        """Log the beginning of a run.

        Args:
            task_description: User's free-text task description.
            primary_metric: Name of the metric being optimized.
            higher_is_better: Optimization direction.
            total_trials: Total trial budget for the run.
        """
        self._write({
            "event": "run_start",
            "ts": _now(),
            "task": task_description,
            "metric": primary_metric,
            "higher_is_better": higher_is_better,
            "total_trials": total_trials,
        })

    def log_trial(self, entry: HistoryEntry) -> None:
        """Log the outcome of a single trial.

        Args:
            entry: Completed HistoryEntry from the search loop.
        """
        self._write({
            "event": "trial",
            "ts": _now(),
            "trial_id": int(entry.trial_id),
            "mode": entry.mode.value,
            "status": entry.status.value,
            "param_name": entry.param_name,
            "param_value": entry.param_value,
            "metric_before": entry.metric_before,
            "metric_after": entry.metric_after,
            "delta": entry.delta,
            "improved": entry.improved,
            "directive_reason": entry.directive.reason,
            "config": _serialise(entry.param_value),
            "error": entry.error_message,
        })

    def log_run_end(
        self,
        baseline: Baseline,
        total_trials: int,
    ) -> None:
        """Log the end of the full run.

        Args:
            baseline: Final best baseline.
            total_trials: Total number of trials executed.
        """
        self._write({
            "event": "run_end",
            "ts": _now(),
            "best_metric": baseline.primary_metric_value,
            "total_trials": total_trials,
            "best_hyperparams": _serialise(baseline.hyperparams),
            "best_augmentations": _serialise(baseline.augmentation_config),
        })

    def close(self) -> None:
        """Flush and close the log file."""
        self._file.flush()
        self._file.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()


def _serialise(value: Any) -> Any:
    """Make a value JSON-serialisable (best-effort).

    Args:
        value: Any Python value.

    Returns:
        JSON-compatible representation.
    """
    if isinstance(value, dict):
        return {k: _serialise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(v) for v in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)
