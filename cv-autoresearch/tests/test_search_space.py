"""Tests for cv_autoresearch.search.space."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import optuna
import pytest

from cv_autoresearch.search.space import (
    _suggest_or_override,
    exploit_space,
    suggest_augmentations,
    suggest_hyperparams,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_HYPERPARAM_KEYS = {
    "learning_rate",
    "weight_decay",
    "batch_size",
    "optimizer_type",
    "lr_scheduler",
    "lr_scheduler_gamma",
    "lr_scheduler_step_size",
    "warmup_epochs",
    "warmup_momentum",
    "gradient_clip_val",
    "label_smoothing",
    "dropout_rate",
    "mixed_precision",
    "ema_decay",
}

_SGD_ONLY_KEYS = {"momentum"}
_ADAM_KEYS = {"beta1", "beta2"}


def _fixed_trial(params: dict[str, Any]) -> optuna.trial.FixedTrial:
    """Return an Optuna FixedTrial for deterministic testing."""
    return optuna.trial.FixedTrial(params)


# ---------------------------------------------------------------------------
# suggest_hyperparams — required keys present
# ---------------------------------------------------------------------------


def test_suggest_hyperparams_returns_all_required_keys_for_sgd() -> None:
    """All 17 params including momentum must be present when optimizer_type=sgd."""
    params = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "optimizer_type": "sgd",
        "lr_scheduler": "cosine",
        "lr_scheduler_gamma": 0.5,
        "lr_scheduler_step_size": 5,
        "warmup_epochs": 2,
        "warmup_momentum": 0.5,
        "gradient_clip_val": 1.0,
        "label_smoothing": 0.1,
        "dropout_rate": 0.3,
        "mixed_precision": True,
        "ema_decay": 0.999,
        "momentum": 0.9,
    }
    trial = _fixed_trial(params)
    result = suggest_hyperparams(trial)
    assert _REQUIRED_HYPERPARAM_KEYS | _SGD_ONLY_KEYS <= result.keys()


def test_suggest_hyperparams_returns_all_required_keys_for_adam() -> None:
    """All 17 params including beta1/beta2 must be present when optimizer_type=adam."""
    params = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "optimizer_type": "adam",
        "lr_scheduler": "cosine",
        "lr_scheduler_gamma": 0.5,
        "lr_scheduler_step_size": 5,
        "warmup_epochs": 2,
        "warmup_momentum": 0.5,
        "gradient_clip_val": 1.0,
        "label_smoothing": 0.1,
        "dropout_rate": 0.3,
        "mixed_precision": True,
        "ema_decay": 0.999,
        "beta1": 0.9,
        "beta2": 0.999,
    }
    trial = _fixed_trial(params)
    result = suggest_hyperparams(trial)
    assert _REQUIRED_HYPERPARAM_KEYS | _ADAM_KEYS <= result.keys()


def test_suggest_hyperparams_with_adamw_includes_beta1_beta2() -> None:
    """beta1 and beta2 must appear when optimizer_type=adamw."""
    params = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "optimizer_type": "adamw",
        "lr_scheduler": "cosine",
        "lr_scheduler_gamma": 0.5,
        "lr_scheduler_step_size": 5,
        "warmup_epochs": 2,
        "warmup_momentum": 0.5,
        "gradient_clip_val": 1.0,
        "label_smoothing": 0.1,
        "dropout_rate": 0.3,
        "mixed_precision": False,
        "ema_decay": 0.999,
        "beta1": 0.9,
        "beta2": 0.999,
    }
    trial = _fixed_trial(params)
    result = suggest_hyperparams(trial)
    assert "beta1" in result
    assert "beta2" in result


def test_suggest_hyperparams_sgd_does_not_include_beta_keys() -> None:
    """momentum is included for sgd; beta1/beta2 should not be required."""
    params = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "optimizer_type": "sgd",
        "lr_scheduler": "step",
        "lr_scheduler_gamma": 0.3,
        "lr_scheduler_step_size": 10,
        "warmup_epochs": 0,
        "warmup_momentum": 0.0,
        "gradient_clip_val": 5.0,
        "label_smoothing": 0.0,
        "dropout_rate": 0.0,
        "mixed_precision": False,
        "ema_decay": 0.999,
        "momentum": 0.85,
    }
    trial = _fixed_trial(params)
    result = suggest_hyperparams(trial)
    assert "momentum" in result
    # beta1/beta2 must not be present for SGD
    assert "beta1" not in result
    assert "beta2" not in result


def test_suggest_hyperparams_overrides_are_respected() -> None:
    """When overrides are provided, those values should appear in the result."""
    params = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "optimizer_type": "adam",
        "lr_scheduler": "cosine",
        "lr_scheduler_gamma": 0.5,
        "lr_scheduler_step_size": 5,
        "warmup_epochs": 2,
        "warmup_momentum": 0.5,
        "gradient_clip_val": 1.0,
        "label_smoothing": 0.1,
        "dropout_rate": 0.3,
        "mixed_precision": True,
        "ema_decay": 0.999,
        "beta1": 0.9,
        "beta2": 0.999,
    }
    trial = _fixed_trial(params)
    overrides = {"learning_rate": 0.042, "batch_size": 128}
    result = suggest_hyperparams(trial, overrides=overrides)
    assert result["learning_rate"] == 0.042
    assert result["batch_size"] == 128


# ---------------------------------------------------------------------------
# exploit_space
# ---------------------------------------------------------------------------

_BASELINE_CONFIG: dict[str, Any] = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "dropout_rate": 0.3,
}


def test_exploit_space_two_element_range_varies_only_target_param() -> None:
    """exploit_space with 2-element float range varies only the target param."""
    params = {"learning_rate": 5e-4}
    trial = _fixed_trial(params)
    result = exploit_space(trial, "learning_rate", [1e-5, 1e-1], _BASELINE_CONFIG)
    # All other keys must equal baseline values
    for key in _BASELINE_CONFIG:
        if key != "learning_rate":
            assert result[key] == _BASELINE_CONFIG[key], f"key {key!r} was unexpectedly changed"


def test_exploit_space_target_param_is_suggested() -> None:
    """exploit_space should return the suggested value for the target param."""
    params = {"learning_rate": 7.5e-4}
    trial = _fixed_trial(params)
    result = exploit_space(trial, "learning_rate", [1e-5, 1e-1], _BASELINE_CONFIG)
    assert result["learning_rate"] == 7.5e-4


def test_exploit_space_categorical_range_selects_from_choices() -> None:
    """exploit_space with >2 elements treats param_range as categorical."""
    params = {"batch_size": 64}
    trial = _fixed_trial(params)
    result = exploit_space(trial, "batch_size", [8, 16, 32, 64, 128], _BASELINE_CONFIG)
    assert result["batch_size"] == 64


def test_exploit_space_preserves_all_other_baseline_keys() -> None:
    """All baseline keys except the target param must remain identical."""
    params = {"dropout_rate": 0.5}
    trial = _fixed_trial(params)
    result = exploit_space(trial, "dropout_rate", [0.0, 0.7], _BASELINE_CONFIG)
    expected_preserved = {k: v for k, v in _BASELINE_CONFIG.items() if k != "dropout_rate"}
    for key, val in expected_preserved.items():
        assert result[key] == val, f"key {key!r}: expected {val!r}, got {result[key]!r}"


def test_exploit_space_returns_new_dict_not_mutating_baseline() -> None:
    """exploit_space must not mutate the original baseline_config."""
    original_lr = _BASELINE_CONFIG["learning_rate"]
    params = {"learning_rate": 1e-2}
    trial = _fixed_trial(params)
    exploit_space(trial, "learning_rate", [1e-5, 1e-1], _BASELINE_CONFIG)
    assert _BASELINE_CONFIG["learning_rate"] == original_lr


# ---------------------------------------------------------------------------
# _suggest_or_override
# ---------------------------------------------------------------------------


def test_suggest_or_override_uses_override_when_present() -> None:
    """_suggest_or_override must return override value without calling suggest_fn."""
    trial = MagicMock()
    suggest_fn = MagicMock(return_value=42.0)
    result = _suggest_or_override(trial, "lr", {"lr": 0.001}, suggest_fn)
    assert result == 0.001
    suggest_fn.assert_not_called()


def test_suggest_or_override_calls_suggest_fn_when_no_override() -> None:
    """_suggest_or_override must delegate to suggest_fn when key absent from overrides."""
    trial = MagicMock()
    suggest_fn = MagicMock(return_value=0.005)
    result = _suggest_or_override(trial, "lr", {}, suggest_fn)
    assert result == 0.005
    suggest_fn.assert_called_once()


# ---------------------------------------------------------------------------
# suggest_augmentations
# ---------------------------------------------------------------------------


def test_suggest_augmentations_returns_dict_with_known_transforms() -> None:
    """suggest_augmentations must return a dict containing expected transform names."""
    params = {
        "HorizontalFlip_enabled": 0.8,
        "VerticalFlip_enabled": 0.2,
        "RandomBrightnessContrast_enabled": 0.7,
        "RandomBrightnessContrast_brightness_limit": 0.2,
        "RandomBrightnessContrast_contrast_limit": 0.2,
        "ColorJitter_enabled": 0.0,
        "GaussianBlur_enabled": 0.6,
        "GaussianBlur_blur_limit": 7,
        "GaussNoise_enabled": 0.0,
        "RandomResizedCrop_enabled": 0.7,
        "RandomResizedCrop_height": 224,
        "RandomResizedCrop_width": 224,
        "RandomScale_enabled": 0.6,
        "RandomScale_scale_limit": 0.1,
        "Rotate_enabled": 0.7,
        "Rotate_limit": 30,
        "Perspective_enabled": 0.0,
        "CoarseDropout_enabled": 0.0,
        "Sharpen_enabled": 0.6,
        "CLAHE_enabled": 0.0,
        "RandomGamma_enabled": 0.0,
        "Normalize_enabled": 0.9,
        "Normalize_mean": (0.485, 0.456, 0.406),
        "Normalize_std": (0.229, 0.224, 0.225),
    }
    trial = _fixed_trial(params)
    result = suggest_augmentations(trial)
    assert isinstance(result, dict)
    assert "HorizontalFlip" in result
    assert "Normalize" in result


def test_suggest_augmentations_disabled_transform_returns_none() -> None:
    """Transforms with enabled score <= 0.5 must map to None."""
    params = {
        "HorizontalFlip_enabled": 0.1,
        "VerticalFlip_enabled": 0.1,
        "RandomBrightnessContrast_enabled": 0.1,
        "RandomBrightnessContrast_brightness_limit": 0.2,
        "RandomBrightnessContrast_contrast_limit": 0.2,
        "ColorJitter_enabled": 0.1,
        "GaussianBlur_enabled": 0.1,
        "GaussianBlur_blur_limit": 7,
        "GaussNoise_enabled": 0.1,
        "RandomResizedCrop_enabled": 0.1,
        "RandomResizedCrop_height": 224,
        "RandomResizedCrop_width": 224,
        "RandomScale_enabled": 0.1,
        "RandomScale_scale_limit": 0.1,
        "Rotate_enabled": 0.1,
        "Rotate_limit": 30,
        "Perspective_enabled": 0.1,
        "CoarseDropout_enabled": 0.1,
        "Sharpen_enabled": 0.1,
        "CLAHE_enabled": 0.1,
        "RandomGamma_enabled": 0.1,
        "Normalize_enabled": 0.1,
        "Normalize_mean": (0.485, 0.456, 0.406),
        "Normalize_std": (0.229, 0.224, 0.225),
    }
    trial = _fixed_trial(params)
    result = suggest_augmentations(trial)
    # All should be None because enabled < 0.5
    for name, value in result.items():
        assert value is None, f"{name!r} should be None when enabled=0.1, got {value!r}"
