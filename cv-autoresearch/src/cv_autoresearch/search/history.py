"""Search history tracking for cv-autoresearch."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from cv_autoresearch.types import Directive, SearchMode, SearchPhase, TrialId, TrialStatus


@dataclass(frozen=True)
class HistoryEntry:
    """Immutable record of a single trial's outcome.

    Args:
        trial_id: Unique identifier for the trial.
        phase: Search phase (hyperparameter or augmentation).
        mode: Strategy mode (explore or exploit).
        directive: The directive that guided this trial.
        param_name: Name of the parameter being varied (exploit mode only).
        param_value: Value used for param_name in this trial.
        metric_before: Baseline metric value before this trial.
        metric_after: Metric value achieved in this trial.
        optuna_objective_value: Raw Optuna objective returned by the trial.
        improved: Whether this trial improved over the baseline.
        status: Final trial status (success, failed, pruned).
        error_message: Error details if status is FAILED.
    """

    trial_id: TrialId
    phase: SearchPhase
    mode: SearchMode
    directive: Directive
    param_name: str | None
    param_value: Any
    metric_before: float | None
    metric_after: float | None
    optuna_objective_value: float | None
    improved: bool
    status: TrialStatus
    error_message: str | None

    @property
    def delta(self) -> float:
        """Compute metric improvement (metric_after - metric_before).

        Returns:
            Difference between after and before metrics, or 0.0 if either is None.
        """
        if self.metric_before is None or self.metric_after is None:
            return 0.0
        return self.metric_after - self.metric_before


def _fingerprint(config: dict[str, Any]) -> str:
    """Compute a SHA-256 fingerprint for a configuration dict.

    Keys are sorted to ensure order-independence.

    Args:
        config: Hyperparameter or augmentation configuration dict.

    Returns:
        64-character hex digest string.
    """
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass
class SearchHistory:
    """Maintains an ordered record of all trial outcomes.

    Attributes:
        entries: Ordered list of all recorded HistoryEntry objects.
    """

    entries: list[HistoryEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Private set tracking fingerprints of already-evaluated configs.
        self._seen_fingerprints: set[str] = set()

    def is_duplicate(self, config: dict[str, Any]) -> bool:
        """Check whether a config has already been evaluated.

        Args:
            config: Configuration dict to check.

        Returns:
            True if the config has been registered before.
        """
        return _fingerprint(config) in self._seen_fingerprints

    def register(self, config: dict[str, Any]) -> None:
        """Mark a config as seen to prevent duplicate evaluation.

        Args:
            config: Configuration dict to register.
        """
        self._seen_fingerprints.add(_fingerprint(config))

    def record(self, entry: HistoryEntry) -> None:
        """Append a completed trial entry to history.

        Args:
            entry: The HistoryEntry to record.
        """
        self.entries.append(entry)

    def best_entries(self, top_k: int = 10) -> list[HistoryEntry]:
        """Return the top-k entries by absolute metric delta, excluding failed trials.

        Args:
            top_k: Maximum number of entries to return.

        Returns:
            List of entries sorted by |delta| descending, length <= top_k.
        """
        candidates = [e for e in self.entries if e.status != TrialStatus.FAILED]
        return sorted(candidates, key=lambda e: abs(e.delta), reverse=True)[:top_k]

    def failed_entries(self) -> list[HistoryEntry]:
        """Return all entries with FAILED status.

        Returns:
            List of entries where status == TrialStatus.FAILED.
        """
        return [e for e in self.entries if e.status == TrialStatus.FAILED]

    def exploit_objective_values(self, param_name: str) -> list[float]:
        """Return Optuna objective values for all EXPLOIT entries on a given param.

        Args:
            param_name: The parameter name to filter on.

        Returns:
            List of objective values (None values are skipped).
        """
        return [
            e.optuna_objective_value
            for e in self.entries
            if (
                e.mode == SearchMode.EXPLOIT
                and e.param_name == param_name
                and e.optuna_objective_value is not None
            )
        ]

    def to_text(self, max_entries: int = 20) -> str:
        """Format the most recent entries as human-readable text for Claude prompts.

        Entries are ordered most-recent first.

        Args:
            max_entries: Maximum number of entries to include.

        Returns:
            Multi-line string representation of recent history.
        """
        recent = list(reversed(self.entries))[:max_entries]
        lines: list[str] = []
        for e in recent:
            parts = [
                f"trial_id={e.trial_id}",
                f"phase={e.phase.value}",
                f"mode={e.mode.value}",
                f"param_name={e.param_name}",
                f"param_value={e.param_value}",
                f"delta={e.delta:.6f}",
                f"improved={e.improved}",
                f"status={e.status.value}",
            ]
            if e.optuna_objective_value is not None:
                parts.append(f"optuna_objective={e.optuna_objective_value}")
            if e.status == TrialStatus.FAILED and e.error_message is not None:
                parts.append(f"error={e.error_message}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)
