"""Configurable losses."""

from __future__ import annotations

from typing import Any

from torch import nn


def build_loss(config: Any) -> nn.Module:
    """Build a loss module for the task type."""
    return nn.CrossEntropyLoss(label_smoothing=float(config.loss.get("label_smoothing", 0.0)))
