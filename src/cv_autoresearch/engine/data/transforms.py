"""Configurable train-time transforms."""

from __future__ import annotations

import torch


class RandomHorizontalFlip:
    """Torch-only horizontal flip transform."""

    def __init__(self, probability: float = 0.0) -> None:
        self.probability = probability

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.probability <= 0:
            return image, target
        if torch.rand(()) < self.probability:
            image = torch.flip(image, dims=(-1,))
            if target.ndim >= 2:
                target = torch.flip(target, dims=(-1,))
        return image, target


class TransformDataset(torch.utils.data.Dataset):
    """Apply an image/target transform to a dataset."""

    def __init__(self, dataset: torch.utils.data.Dataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        return self.transform(image, target)


def build_train_transform(config) -> RandomHorizontalFlip:
    """Build train transform from Hydra augmentation config."""
    probability = 0.0
    if config is not None:
        probability = float(config.get("horizontal_flip_probability", 0.0))
    return RandomHorizontalFlip(probability=probability)
