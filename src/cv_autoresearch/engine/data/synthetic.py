"""Tiny offline datasets used by tests and smoke runs."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticPromptDataset(Dataset):
    """Deterministic image-label fixture for prompt-flow smoke tests."""

    def __init__(
        self,
        *,
        size: int = 32,
        num_classes: int = 2,
        image_size: int = 16,
        channels: int = 3,
    ) -> None:
        generator = torch.Generator().manual_seed(1234 + size + num_classes)
        self.images = torch.rand(size, channels, image_size, image_size, generator=generator)
        means = self.images.mean(dim=(1, 2, 3))
        self.labels = ((means * num_classes).long() % num_classes).clamp(max=num_classes - 1)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]
