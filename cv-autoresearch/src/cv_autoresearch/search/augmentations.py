"""Albumentations pipeline construction for cv-autoresearch."""

from __future__ import annotations

from typing import Any

import albumentations as A

# Mapping from transform name strings to Albumentations classes.
_TRANSFORM_REGISTRY: dict[str, type] = {
    "HorizontalFlip": A.HorizontalFlip,
    "VerticalFlip": A.VerticalFlip,
    "RandomBrightnessContrast": A.RandomBrightnessContrast,
    "ColorJitter": A.ColorJitter,
    "GaussianBlur": A.GaussianBlur,
    "GaussNoise": A.GaussNoise,
    "RandomResizedCrop": A.RandomResizedCrop,
    "RandomScale": A.RandomScale,
    "Rotate": A.Rotate,
    "Perspective": A.Perspective,
    "CoarseDropout": A.CoarseDropout,
    "Sharpen": A.Sharpen,
    "CLAHE": A.CLAHE,
    "RandomGamma": A.RandomGamma,
    "Normalize": A.Normalize,
}


def build_augmentation_pipeline(
    aug_params: dict[str, dict[str, Any] | None],
) -> A.Compose:
    """Build an Albumentations Compose pipeline from a parameter dict.

    Each key in aug_params is a transform name; the corresponding value is
    either a dict of keyword arguments for that transform, or None to skip it.

    Args:
        aug_params: Dict mapping transform name to kwargs dict (or None to disable).

    Returns:
        An A.Compose containing all active (non-None) transforms.

    Raises:
        KeyError: If a transform name is not found in the registry.
    """
    transforms: list[A.BasicTransform] = []

    for name, params in aug_params.items():
        if params is None:
            continue
        transform_cls = _TRANSFORM_REGISTRY[name]
        transforms.append(transform_cls(**params))

    return A.Compose(transforms)
