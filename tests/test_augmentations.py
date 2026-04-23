"""Tests for cv_autoresearch.search.augmentations."""

from __future__ import annotations

import albumentations as A
import pytest

from cv_autoresearch.search.augmentations import build_augmentation_pipeline

# ---------------------------------------------------------------------------
# build_augmentation_pipeline
# ---------------------------------------------------------------------------


def test_build_augmentation_pipeline_returns_compose() -> None:
    """build_augmentation_pipeline must return an A.Compose instance."""
    aug_params: dict = {
        "HorizontalFlip": {"p": 0.5},
        "VerticalFlip": {"p": 0.3},
    }
    pipeline = build_augmentation_pipeline(aug_params)
    assert isinstance(pipeline, A.Compose)


def test_build_augmentation_pipeline_skips_none_entries() -> None:
    """Entries with None value must not appear in the pipeline transforms."""
    aug_params: dict = {
        "HorizontalFlip": {"p": 0.5},
        "VerticalFlip": None,
        "Rotate": None,
    }
    pipeline = build_augmentation_pipeline(aug_params)
    # Only HorizontalFlip should be included
    transform_types = [type(t).__name__ for t in pipeline.transforms]
    assert "HorizontalFlip" in transform_types
    assert "VerticalFlip" not in transform_types
    assert "Rotate" not in transform_types


def test_build_augmentation_pipeline_empty_dict_returns_empty_compose() -> None:
    """Empty aug_params must produce an A.Compose with no transforms."""
    pipeline = build_augmentation_pipeline({})
    assert isinstance(pipeline, A.Compose)
    assert len(pipeline.transforms) == 0


def test_build_augmentation_pipeline_all_none_returns_empty_compose() -> None:
    """All-None entries must produce an A.Compose with no transforms."""
    aug_params = {
        "HorizontalFlip": None,
        "VerticalFlip": None,
        "Rotate": None,
    }
    pipeline = build_augmentation_pipeline(aug_params)
    assert len(pipeline.transforms) == 0


def test_build_augmentation_pipeline_horizontal_flip_instantiated() -> None:
    """HorizontalFlip must be correctly instantiated with the provided params."""
    aug_params = {"HorizontalFlip": {"p": 0.7}}
    pipeline = build_augmentation_pipeline(aug_params)
    transform_types = [type(t).__name__ for t in pipeline.transforms]
    assert "HorizontalFlip" in transform_types
    # Verify the transform has the correct probability
    flip_transforms = [t for t in pipeline.transforms if isinstance(t, A.HorizontalFlip)]
    assert len(flip_transforms) == 1
    assert flip_transforms[0].p == pytest.approx(0.7)


def test_build_augmentation_pipeline_random_brightness_contrast_instantiated() -> None:
    """RandomBrightnessContrast must be instantiated with brightness/contrast limits."""
    aug_params = {
        "RandomBrightnessContrast": {
            "brightness_limit": 0.2,
            "contrast_limit": 0.2,
            "p": 0.5,
        }
    }
    pipeline = build_augmentation_pipeline(aug_params)
    transform_types = [type(t).__name__ for t in pipeline.transforms]
    assert "RandomBrightnessContrast" in transform_types


def test_build_augmentation_pipeline_multiple_active_transforms() -> None:
    """All non-None entries must be added to the pipeline."""
    aug_params = {
        "HorizontalFlip": {"p": 0.5},
        "VerticalFlip": {"p": 0.3},
        "Rotate": {"limit": 30, "p": 0.4},
        "GaussianBlur": None,
    }
    pipeline = build_augmentation_pipeline(aug_params)
    transform_types = [type(t).__name__ for t in pipeline.transforms]
    assert "HorizontalFlip" in transform_types
    assert "VerticalFlip" in transform_types
    assert "Rotate" in transform_types
    assert "GaussianBlur" not in transform_types
    assert len(pipeline.transforms) == 3


def test_build_augmentation_pipeline_normalize_instantiated() -> None:
    """Normalize transform must be correctly instantiated."""
    aug_params = {
        "Normalize": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "p": 1.0,
        }
    }
    pipeline = build_augmentation_pipeline(aug_params)
    transform_types = [type(t).__name__ for t in pipeline.transforms]
    assert "Normalize" in transform_types


@pytest.mark.parametrize(
    "transform_name, params",
    [
        ("HorizontalFlip", {"p": 0.5}),
        ("VerticalFlip", {"p": 0.5}),
        ("Sharpen", {"p": 0.5}),
        ("CLAHE", {"p": 0.5}),
        ("RandomGamma", {"p": 0.5}),
        ("Perspective", {"p": 0.5}),
    ],
)
def test_build_augmentation_pipeline_named_transforms(transform_name: str, params: dict) -> None:
    """Each named transform must be correctly instantiated in the pipeline."""
    pipeline = build_augmentation_pipeline({transform_name: params})
    transform_types = [type(t).__name__ for t in pipeline.transforms]
    assert transform_name in transform_types
