"""Core types for cv-autoresearch."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, NewType

# NewType prevents accidental mixing of raw int with trial identifiers at the type-checker level.
TrialId = NewType("TrialId", int)


class SearchMode(Enum):
    """Strategy mode: broad exploration or exploitation of known good regions."""

    EXPLORE = "explore"
    EXPLOIT = "exploit"


class TrialStatus(Enum):
    """Outcome of a single Optuna trial."""

    SUCCESS = "success"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass(frozen=True)
class Directive:
    """Decision returned by Claude at each iteration."""

    mode: SearchMode
    target_param: str
    target_range: list[float] | None
    reason: str


@dataclass
class Baseline:
    """Current best configuration."""

    primary_metric_value: float
    hyperparams: dict[str, Any]
    augmentation_config: dict[str, Any]
    trial_id: TrialId | None = None


@dataclass(frozen=True)
class IterationResult:
    """Result of a single trial iteration."""

    trial_id: TrialId
    mode: SearchMode
    status: TrialStatus
    metrics: dict[str, float]
    primary_metric_value: float | None
    optuna_objective_value: float | None
    config_snapshot: dict[str, Any]
    improved: bool
    error_message: str | None
    directive: Directive
