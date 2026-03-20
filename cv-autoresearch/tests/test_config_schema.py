"""Tests for cv_autoresearch.config.schema."""

from __future__ import annotations

import pytest

from cv_autoresearch.config.schema import SearchConfig


@pytest.mark.parametrize(
    "field_name, expected_value",
    [
        ("total_trials", 80),
        ("epochs_per_trial", 10),
        ("exploit_trials_per_directive", 5),
        ("optuna_storage", "sqlite:///autoresearch.db"),
        ("optuna_seed", 42),
        ("device", "cuda"),
        ("num_workers", 4),
        ("output_dir", "./autoresearch_output"),
    ],
)
def test_search_config_defaults(field_name: str, expected_value: object):
    cfg = SearchConfig(
        task_description="classify cats vs dogs",
        primary_metric="val_acc",
        higher_is_better=True,
    )
    assert getattr(cfg, field_name) == expected_value


@pytest.mark.parametrize(
    "output_dir, prop, expected",
    [
        ("./out", "checkpoint_dir", "out/checkpoints"),
        ("./out", "metric_config_path", "out/config/metrics/generated.yaml"),
        ("./out", "log_path", "out/autoresearch.jsonl"),
        ("./out", "plot_path", "out/improvement_curve.png"),
        ("/abs/path", "checkpoint_dir", "/abs/path/checkpoints"),
        ("/abs/path", "log_path", "/abs/path/autoresearch.jsonl"),
    ],
)
def test_derived_paths(output_dir: str, prop: str, expected: str):
    cfg = SearchConfig(
        task_description="task",
        primary_metric="val_acc",
        higher_is_better=True,
        output_dir=output_dir,
    )
    assert getattr(cfg, prop) == expected


def test_required_task_description():
    with pytest.raises(TypeError):
        SearchConfig(primary_metric="val_acc", higher_is_better=True)  # type: ignore[call-arg]


def test_required_primary_metric():
    with pytest.raises(TypeError):
        SearchConfig(task_description="task", higher_is_better=True)  # type: ignore[call-arg]


def test_required_higher_is_better():
    with pytest.raises(TypeError):
        SearchConfig(task_description="task", primary_metric="val_acc")  # type: ignore[call-arg]


def test_hp_overrides_defaults_to_empty_dict():
    cfg = SearchConfig(
        task_description="task",
        primary_metric="val_acc",
        higher_is_better=True,
    )
    assert cfg.hp_overrides == {}


def test_aug_overrides_defaults_to_empty_dict():
    cfg = SearchConfig(
        task_description="task",
        primary_metric="val_acc",
        higher_is_better=True,
    )
    assert cfg.aug_overrides == {}


def test_hp_overrides_not_shared_between_instances():
    cfg1 = SearchConfig(
        task_description="task1",
        primary_metric="val_acc",
        higher_is_better=True,
    )
    cfg2 = SearchConfig(
        task_description="task2",
        primary_metric="val_loss",
        higher_is_better=False,
    )
    cfg1.hp_overrides["lr"] = 1e-3
    assert "lr" not in cfg2.hp_overrides


def test_aug_overrides_not_shared_between_instances():
    cfg1 = SearchConfig(
        task_description="task1",
        primary_metric="val_acc",
        higher_is_better=True,
    )
    cfg2 = SearchConfig(
        task_description="task2",
        primary_metric="val_loss",
        higher_is_better=False,
    )
    cfg1.aug_overrides["flip"] = True
    assert "flip" not in cfg2.aug_overrides


def test_custom_values():
    cfg = SearchConfig(
        task_description="segment tumors",
        primary_metric="val_dice",
        higher_is_better=True,
        total_trials=120,
        epochs_per_trial=5,
        device="cpu",
    )
    assert cfg.task_description == "segment tumors"
    assert cfg.primary_metric == "val_dice"
    assert cfg.higher_is_better is True
    assert cfg.total_trials == 120
    assert cfg.epochs_per_trial == 5
    assert cfg.device == "cpu"
