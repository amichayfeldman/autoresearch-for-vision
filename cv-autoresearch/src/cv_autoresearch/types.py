"""Core types for cv-autoresearch."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, NewType

TrialId = NewType("TrialId", int)


class SearchPhase(Enum):
    HYPERPARAMETER = "hyperparameter"
    AUGMENTATION = "augmentation"


class SearchMode(Enum):
    EXPLORE = "explore"
    EXPLOIT = "exploit"


class TrialStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass(frozen=True)
class Directive:
    """Decision returned by Claude at each iteration."""

    mode: SearchMode
    target_param: str | None
    target_range: tuple | list | None
    phase: SearchPhase
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
    phase: SearchPhase
    mode: SearchMode
    status: TrialStatus
    metrics: dict[str, float]
    primary_metric_value: float | None
    optuna_objective_value: float | None
    config_snapshot: dict[str, Any]
    improved: bool
    error_message: str | None
    directive: Directive
