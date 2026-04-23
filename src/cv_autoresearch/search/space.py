"""Search-space registry and one-parameter exploit logic for cv-autoresearch."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any

import optuna

# Canonical train-time augmentations supported by the current autoresearch loop.
_AUG_DEFAULTS: dict[str, dict[str, Any]] = {
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

# Source of truth for parameters exposed to the advisor.
# Augmentation entries use "<Transform>_enabled" and "<Transform>_<field>".
PARAM_REGISTRY: dict[str, list[Any]] = {
    # Hyperparameters
    "learning_rate": [1e-5, 1e-1],
    "weight_decay": [1e-6, 1e-2],
    "batch_size": [8, 16, 32, 64, 128, 256],
    "optimizer_type": ["adam", "adamw", "sgd"],
    "lr_scheduler": ["cosine", "step", "onecycle", "cosine_with_restarts"],
    "lr_scheduler_gamma": [0.1, 0.9],
    "lr_scheduler_step_size": [1, 20],
    "warmup_epochs": [0, 10],
    "warmup_momentum": [0.0, 0.95],
    "gradient_clip_val": [0.1, 10.0],
    "label_smoothing": [0.0, 0.2],
    "dropout_rate": [0.0, 0.7],
    "mixed_precision": [True, False],
    "ema_decay": [0.99, 0.9999],
    "momentum": [0.8, 0.99],
    "beta1": [0.85, 0.95],
    "beta2": [0.99, 0.9999],
    # Augmentation toggles and knobs
    "HorizontalFlip_enabled": [0.0, 1.0],
    "VerticalFlip_enabled": [0.0, 1.0],
    "RandomBrightnessContrast_enabled": [0.0, 1.0],
    "RandomBrightnessContrast_brightness_limit": [0.0, 0.4],
    "RandomBrightnessContrast_contrast_limit": [0.0, 0.4],
    "ColorJitter_enabled": [0.0, 1.0],
    "GaussianBlur_enabled": [0.0, 1.0],
    "GaussianBlur_blur_limit": [3, 5, 7, 9, 11],
    "GaussNoise_enabled": [0.0, 1.0],
    "RandomResizedCrop_enabled": [0.0, 1.0],
    "RandomResizedCrop_height": [128, 224, 256, 384],
    "RandomResizedCrop_width": [128, 224, 256, 384],
    "RandomScale_enabled": [0.0, 1.0],
    "RandomScale_scale_limit": [0.0, 0.5],
    "Rotate_enabled": [0.0, 1.0],
    "Rotate_limit": [5, 90],
    "Perspective_enabled": [0.0, 1.0],
    "CoarseDropout_enabled": [0.0, 1.0],
    "Sharpen_enabled": [0.0, 1.0],
    "CLAHE_enabled": [0.0, 1.0],
    "RandomGamma_enabled": [0.0, 1.0],
    "Normalize_enabled": [0.0, 1.0],
}

_LOG_SCALE_PARAMS: frozenset[str] = frozenset({"learning_rate", "weight_decay"})


def _suggest_value(trial: optuna.Trial, param_name: str, param_range: list[Any]) -> Any:
    """Suggest one value from param_range with type-aware Optuna APIs."""
    if len(param_range) == 2 and all(isinstance(v, Number) and not isinstance(v, bool) for v in param_range):
        low, high = param_range
        if all(isinstance(v, int) for v in param_range):
            return trial.suggest_int(param_name, int(low), int(high))
        return trial.suggest_float(param_name, float(low), float(high), log=(param_name in _LOG_SCALE_PARAMS))
    return trial.suggest_categorical(param_name, param_range)


def _split_aug_param(param_name: str) -> tuple[str | None, str | None]:
    """Parse '<Transform>_enabled' or '<Transform>_<field>' parameter naming."""
    if param_name.endswith("_enabled"):
        transform = param_name[: -len("_enabled")]
        if transform in _AUG_DEFAULTS:
            return transform, "enabled"
        return None, None

    for transform in _AUG_DEFAULTS:
        prefix = f"{transform}_"
        if param_name.startswith(prefix):
            return transform, param_name[len(prefix):]

    return None, None


def _coerce_aug_value(field_name: str, value: Any) -> Any:
    """Keep augmentation parameter types stable after Optuna sampling."""
    if field_name in {"blur_limit", "height", "width", "limit"}:
        return int(value)
    return value


def _apply_aug_param(config: dict[str, Any], param_name: str, sampled_value: Any) -> bool:
    """Apply one augmentation search parameter to top-level config in-place."""
    transform_name, field_name = _split_aug_param(param_name)
    if transform_name is None or field_name is None:
        return False

    if field_name == "enabled":
        config[transform_name] = deepcopy(_AUG_DEFAULTS[transform_name]) if float(sampled_value) > 0.5 else None
        return True

    current = config.get(transform_name)
    if not isinstance(current, dict):
        current = deepcopy(_AUG_DEFAULTS[transform_name])
    else:
        current = dict(current)

    current[field_name] = _coerce_aug_value(field_name, sampled_value)
    if "p" not in current:
        current["p"] = 1.0
    config[transform_name] = current
    return True


def exploit_space(
    trial: optuna.Trial,
    param_name: str,
    param_range: list[Any],
    baseline_config: dict[str, Any],
) -> dict[str, Any]:
    """Return a trial config with only one parameter changed from baseline.

    For hyperparameters this updates `param_name` directly. For augmentation
    params named `<Transform>_enabled` or `<Transform>_<field>`, this mutates the
    corresponding top-level transform key to keep downstream logic consistent.
    """
    result = dict(baseline_config)
    sampled = _suggest_value(trial, param_name, param_range)
    if _apply_aug_param(result, param_name, sampled):
        return result
    result[param_name] = sampled
    return result
