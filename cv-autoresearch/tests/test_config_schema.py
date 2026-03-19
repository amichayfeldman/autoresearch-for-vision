"""Tests for cv_autoresearch.config.schema."""

from __future__ import annotations

import pytest

from cv_autoresearch.config.schema import SearchConfig


def test_default_hp_trials():
    cfg = SearchConfig(
        task_description="classify cats vs dogs",
        primary_metric="val_acc",
        higher_is_better=True,
    )
    assert cfg.hp_trials == 50


def test_default_aug_trials():
    cfg = SearchConfig(
        task_description="classify cats vs dogs",
        primary_metric="val_acc",
        higher_is_better=True,
    )
    assert cfg.aug_trials == 30


def test_default_epochs_per_trial():
    cfg = SearchConfig(
        task_description="classify cats vs dogs",
        primary_metric="val_acc",
        higher_is_better=True,
    )
    assert cfg.epochs_per_trial == 10


def test_default_exploit_trials_per_directive():
    cfg = SearchConfig(
        task_description="classify cats vs dogs",
        primary_metric="val_acc",
        higher_is_better=True,
    )
    assert cfg.exploit_trials_per_directive == 5


def test_default_optuna_storage():
    cfg = SearchConfig(
        task_description="classify cats vs dogs",
        primary_metric="val_acc",
        higher_is_better=True,
    )
    assert cfg.optuna_storage == "sqlite:///autoresearch.db"


def test_default_optuna_seed():
    cfg = SearchConfig(
        task_description="classify cats vs dogs",
        primary_metric="val_acc",
        higher_is_better=True,
    )
    assert cfg.optuna_seed == 42


def test_default_device():
    cfg = SearchConfig(
        task_description="classify cats vs dogs",
        primary_metric="val_acc",
        higher_is_better=True,
    )
    assert cfg.device == "cuda"


def test_default_num_workers():
    cfg = SearchConfig(
        task_description="classify cats vs dogs",
        primary_metric="val_acc",
        higher_is_better=True,
    )
    assert cfg.num_workers == 4


def test_default_checkpoint_dir():
    cfg = SearchConfig(
        task_description="classify cats vs dogs",
        primary_metric="val_acc",
        higher_is_better=True,
    )
    assert cfg.checkpoint_dir == "./checkpoints"


def test_default_metric_config_path():
    cfg = SearchConfig(
        task_description="classify cats vs dogs",
        primary_metric="val_acc",
        higher_is_better=True,
    )
    assert cfg.metric_config_path == "config/metrics/generated.yaml"


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
        hp_trials=100,
        aug_trials=60,
        epochs_per_trial=5,
        device="cpu",
    )
    assert cfg.task_description == "segment tumors"
    assert cfg.primary_metric == "val_dice"
    assert cfg.higher_is_better is True
    assert cfg.hp_trials == 100
    assert cfg.aug_trials == 60
    assert cfg.epochs_per_trial == 5
    assert cfg.device == "cpu"
