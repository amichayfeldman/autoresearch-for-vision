"""Structured iteration history records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BaselineState:
    """Current promoted baseline artifacts."""

    iteration_id: int
    primary_metric_value: float
    checkpoint_path: str
    config_path: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class IterationRecord:
    """One immutable manager record for baseline or agent iteration."""

    iteration_id: int
    status: str
    parent_baseline_id: int | None
    changed_files: list[str]
    patch: str
    one_change_summary: str
    frozen_config: dict[str, Any]
    checkpoint_path: str | None
    metrics: dict[str, float]
    primary_metric: str
    primary_metric_before: float | None
    primary_metric_after: float | None
    improved: bool
    promoted: bool
    insight: str
    error_message: str | None = None
    epoch_metrics: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready record."""
        return asdict(self)
