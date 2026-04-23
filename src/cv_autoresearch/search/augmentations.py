"""Albumentations pipeline construction for cv-autoresearch."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
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

# Default kwargs for currently-supported train-time augmentations.
_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "HorizontalFlip": {"p": 1.0},
    "VerticalFlip": {"p": 1.0},
    "RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 1.0},
    "ColorJitter": {"p": 1.0},
    "GaussianBlur": {"blur_limit": 3, "p": 1.0},
    "GaussNoise": {"p": 1.0},
    "RandomResizedCrop": {"height": 224, "width": 224, "p": 1.0},
    "RandomScale": {"scale_limit": 0.1, "p": 1.0},
    "Rotate": {"limit": 15, "p": 1.0},
    "Perspective": {"p": 1.0},
    "CoarseDropout": {"p": 1.0},
    "Sharpen": {"p": 1.0},
    "CLAHE": {"p": 1.0},
    "RandomGamma": {"p": 1.0},
    "Normalize": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225), "p": 1.0},
}


def _normalize_transform_params(name: str, params: Any) -> dict[str, Any] | None:
    """Normalize mixed config forms into kwargs dict or disabled (None).

    Accepted forms:
    - None / False / <=0.5 numeric -> disabled
    - True / >0.5 numeric -> defaults for that transform
    - dict -> defaults merged with provided kwargs
    """
    if params is None:
        return None
    if isinstance(params, bool):
        return deepcopy(_DEFAULT_PARAMS[name]) if params else None
    if isinstance(params, Number):
        return deepcopy(_DEFAULT_PARAMS[name]) if float(params) > 0.5 else None
    if isinstance(params, dict):
        merged = deepcopy(_DEFAULT_PARAMS[name])
        merged.update(params)
        return merged
    raise TypeError(f"Unsupported params type for {name}: {type(params).__name__}")


def build_augmentation_pipeline(
    aug_params: dict[str, Any],
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
        TypeError: If transform parameters are not in a supported format.
    """
    transforms: list[A.BasicTransform] = []

    for name, params in aug_params.items():
        if name not in _TRANSFORM_REGISTRY:
            supported = ", ".join(sorted(_TRANSFORM_REGISTRY))
            raise KeyError(f"Unknown transform '{name}'. Supported transforms: {supported}")

        normalized = _normalize_transform_params(name, params)
        if normalized is None:
            continue
        transforms.append(_TRANSFORM_REGISTRY[name](**normalized))

    return A.Compose(transforms)
