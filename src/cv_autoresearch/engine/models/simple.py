"""Small default models for examples and offline tests."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TinyPromptModel(nn.Module):
    """A compact CNN for prompt-flow smoke runs."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(config: Any) -> nn.Module:
    """Build the configured model."""
    channels = int(config.data.get("channels", 3))
    return TinyPromptModel(
        in_channels=channels,
        num_classes=int(config.evaluation.get("num_classes", 2)),
        dropout=float(config.model.get("dropout", 0.0)),
    )
