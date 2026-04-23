"""Evaluation result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics produced by an immutable evaluator."""

    metrics: dict[str, float]
    epoch_metrics: list[dict[str, float]]
    artifacts: dict[str, Any] | None = None

    def primary_value(self, primary_metric: str) -> float | None:
        """Return the selected primary metric, if present."""
        return self.metrics.get(primary_metric)
