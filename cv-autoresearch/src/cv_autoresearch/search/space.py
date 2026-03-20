"""Search space definitions for cv-autoresearch hyperparameter optimization."""

from __future__ import annotations

from typing import Any, Callable

import optuna


def _suggest_or_override(
    trial: optuna.Trial,
    name: str,
    overrides: dict[str, Any] | None,
    suggest_fn: Callable[[], Any],
) -> Any:
    """Return override value if present, otherwise call suggest_fn.

    Args:
        trial: Active Optuna trial object.
        name: Parameter name to look up in overrides.
        overrides: Optional dict of fixed parameter values.
        suggest_fn: Callable that asks Optuna to suggest a value.

    Returns:
        The override value if name is in overrides, else the suggested value.
    """
    if overrides and name in overrides:
        return overrides[name]
    return suggest_fn()


def suggest_hyperparams(
    trial: optuna.Trial,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Suggest a full set of training hyperparameters from Optuna.

    Suggests 15 parameters unconditionally plus optimizer-specific
    parameters: momentum (SGD only), beta1/beta2 (Adam/AdamW only).

    Args:
        trial: Active Optuna trial used for parameter suggestion.
        overrides: Optional dict of fixed values that bypass Optuna suggestion.

    Returns:
        Dict mapping parameter names to suggested (or overridden) values.
    """
    params: dict[str, Any] = {}

    params["learning_rate"] = _suggest_or_override(
        trial, "learning_rate", overrides,
        lambda: trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
    )
    params["weight_decay"] = _suggest_or_override(
        trial, "weight_decay", overrides,
        lambda: trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    )
    params["batch_size"] = _suggest_or_override(
        trial, "batch_size", overrides,
        lambda: trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256]),
    )
    params["optimizer_type"] = _suggest_or_override(
        trial, "optimizer_type", overrides,
        lambda: trial.suggest_categorical("optimizer_type", ["adam", "adamw", "sgd"]),
    )
    params["lr_scheduler"] = _suggest_or_override(
        trial, "lr_scheduler", overrides,
        lambda: trial.suggest_categorical(
            "lr_scheduler", ["cosine", "step", "onecycle", "cosine_with_restarts"]
        ),
    )
    params["lr_scheduler_gamma"] = _suggest_or_override(
        trial, "lr_scheduler_gamma", overrides,
        lambda: trial.suggest_float("lr_scheduler_gamma", 0.1, 0.9),
    )
    params["lr_scheduler_step_size"] = _suggest_or_override(
        trial, "lr_scheduler_step_size", overrides,
        lambda: trial.suggest_int("lr_scheduler_step_size", 1, 20),
    )
    params["warmup_epochs"] = _suggest_or_override(
        trial, "warmup_epochs", overrides,
        lambda: trial.suggest_int("warmup_epochs", 0, 10),
    )
    params["warmup_momentum"] = _suggest_or_override(
        trial, "warmup_momentum", overrides,
        lambda: trial.suggest_float("warmup_momentum", 0.0, 0.95),
    )
    params["gradient_clip_val"] = _suggest_or_override(
        trial, "gradient_clip_val", overrides,
        lambda: trial.suggest_float("gradient_clip_val", 0.1, 10.0),
    )
    params["label_smoothing"] = _suggest_or_override(
        trial, "label_smoothing", overrides,
        lambda: trial.suggest_float("label_smoothing", 0.0, 0.2),
    )
    params["dropout_rate"] = _suggest_or_override(
        trial, "dropout_rate", overrides,
        lambda: trial.suggest_float("dropout_rate", 0.0, 0.7),
    )
    params["mixed_precision"] = _suggest_or_override(
        trial, "mixed_precision", overrides,
        lambda: trial.suggest_categorical("mixed_precision", [True, False]),
    )
    params["ema_decay"] = _suggest_or_override(
        trial, "ema_decay", overrides,
        lambda: trial.suggest_float("ema_decay", 0.99, 0.9999),
    )

    optimizer = params["optimizer_type"]

    if optimizer == "sgd":
        params["momentum"] = _suggest_or_override(
            trial, "momentum", overrides,
            lambda: trial.suggest_float("momentum", 0.8, 0.99),
        )
    else:
        params["beta1"] = _suggest_or_override(
            trial, "beta1", overrides,
            lambda: trial.suggest_float("beta1", 0.85, 0.95),
        )
        params["beta2"] = _suggest_or_override(
            trial, "beta2", overrides,
            lambda: trial.suggest_float("beta2", 0.99, 0.9999),
        )

    return params


def suggest_augmentations(
    trial: optuna.Trial,
    overrides: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any] | None]:
    """Suggest an augmentation configuration from Optuna.

    For each of 15 transforms, suggests an enabled probability. Transforms
    with enabled score <= 0.5 are returned as None (disabled). Active
    transforms include relevant hyperparameters.

    Args:
        trial: Active Optuna trial used for parameter suggestion.
        overrides: Optional dict of fixed values that bypass Optuna suggestion.

    Returns:
        Dict mapping transform names to parameter dicts, or None if disabled.
    """
    result: dict[str, dict[str, Any] | None] = {}

    def _enabled(name: str) -> float:
        return _suggest_or_override(
            trial, f"{name}_enabled", overrides,
            lambda: trial.suggest_float(f"{name}_enabled", 0.0, 1.0),
        )

    # HorizontalFlip
    if _enabled("HorizontalFlip") > 0.5:
        result["HorizontalFlip"] = {"p": 1.0}
    else:
        result["HorizontalFlip"] = None

    # VerticalFlip
    if _enabled("VerticalFlip") > 0.5:
        result["VerticalFlip"] = {"p": 1.0}
    else:
        result["VerticalFlip"] = None

    # RandomBrightnessContrast
    if _enabled("RandomBrightnessContrast") > 0.5:
        brightness_limit = _suggest_or_override(
            trial, "RandomBrightnessContrast_brightness_limit", overrides,
            lambda: trial.suggest_float("RandomBrightnessContrast_brightness_limit", 0.0, 0.4),
        )
        contrast_limit = _suggest_or_override(
            trial, "RandomBrightnessContrast_contrast_limit", overrides,
            lambda: trial.suggest_float("RandomBrightnessContrast_contrast_limit", 0.0, 0.4),
        )
        result["RandomBrightnessContrast"] = {
            "brightness_limit": brightness_limit,
            "contrast_limit": contrast_limit,
            "p": 1.0,
        }
    else:
        result["RandomBrightnessContrast"] = None

    # ColorJitter
    if _enabled("ColorJitter") > 0.5:
        result["ColorJitter"] = {"p": 1.0}
    else:
        result["ColorJitter"] = None

    # GaussianBlur
    if _enabled("GaussianBlur") > 0.5:
        blur_limit = _suggest_or_override(
            trial, "GaussianBlur_blur_limit", overrides,
            lambda: trial.suggest_int("GaussianBlur_blur_limit", 3, 11, step=2),
        )
        result["GaussianBlur"] = {"blur_limit": blur_limit, "p": 1.0}
    else:
        result["GaussianBlur"] = None

    # GaussNoise
    if _enabled("GaussNoise") > 0.5:
        result["GaussNoise"] = {"p": 1.0}
    else:
        result["GaussNoise"] = None

    # RandomResizedCrop
    if _enabled("RandomResizedCrop") > 0.5:
        height = _suggest_or_override(
            trial, "RandomResizedCrop_height", overrides,
            lambda: trial.suggest_categorical("RandomResizedCrop_height", [128, 224, 256, 384]),
        )
        width = _suggest_or_override(
            trial, "RandomResizedCrop_width", overrides,
            lambda: trial.suggest_categorical("RandomResizedCrop_width", [128, 224, 256, 384]),
        )
        result["RandomResizedCrop"] = {"height": height, "width": width, "p": 1.0}
    else:
        result["RandomResizedCrop"] = None

    # RandomScale
    if _enabled("RandomScale") > 0.5:
        scale_limit = _suggest_or_override(
            trial, "RandomScale_scale_limit", overrides,
            lambda: trial.suggest_float("RandomScale_scale_limit", 0.0, 0.5),
        )
        result["RandomScale"] = {"scale_limit": scale_limit, "p": 1.0}
    else:
        result["RandomScale"] = None

    # Rotate
    if _enabled("Rotate") > 0.5:
        limit = _suggest_or_override(
            trial, "Rotate_limit", overrides,
            lambda: trial.suggest_int("Rotate_limit", 5, 90),
        )
        result["Rotate"] = {"limit": limit, "p": 1.0}
    else:
        result["Rotate"] = None

    # Perspective
    if _enabled("Perspective") > 0.5:
        result["Perspective"] = {"p": 1.0}
    else:
        result["Perspective"] = None

    # CoarseDropout
    if _enabled("CoarseDropout") > 0.5:
        result["CoarseDropout"] = {"p": 1.0}
    else:
        result["CoarseDropout"] = None

    # Sharpen
    if _enabled("Sharpen") > 0.5:
        result["Sharpen"] = {"p": 1.0}
    else:
        result["Sharpen"] = None

    # CLAHE
    if _enabled("CLAHE") > 0.5:
        result["CLAHE"] = {"p": 1.0}
    else:
        result["CLAHE"] = None

    # RandomGamma
    if _enabled("RandomGamma") > 0.5:
        result["RandomGamma"] = {"p": 1.0}
    else:
        result["RandomGamma"] = None

    # Normalize
    if _enabled("Normalize") > 0.5:
        mean = _suggest_or_override(
            trial, "Normalize_mean", overrides,
            lambda: (0.485, 0.456, 0.406),
        )
        std = _suggest_or_override(
            trial, "Normalize_std", overrides,
            lambda: (0.229, 0.224, 0.225),
        )
        result["Normalize"] = {"mean": mean, "std": std, "p": 1.0}
    else:
        result["Normalize"] = None

    return result


def exploit_space(
    trial: optuna.Trial,
    param_name: str,
    param_range: list[float],
    baseline_config: dict[str, Any],
) -> dict[str, Any]:
    """Return a copy of baseline_config with only param_name varied.

    If param_range contains exactly 2 elements, they are treated as [low, high]
    for a continuous float suggestion. If param_range contains more than 2 elements,
    the values are treated as categorical choices.

    Args:
        trial: Active Optuna trial for parameter suggestion.
        param_name: The parameter to vary in this exploit trial.
        param_range: Either [low, high] for float range, or list of categorical choices.
        baseline_config: Current best configuration to use as a base.

    Returns:
        New dict identical to baseline_config except for param_name.
    """
    result = dict(baseline_config)

    if len(param_range) == 2:
        result[param_name] = trial.suggest_float(param_name, float(param_range[0]), float(param_range[1]))
    else:
        result[param_name] = trial.suggest_categorical(param_name, param_range)

    return result
