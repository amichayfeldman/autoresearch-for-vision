"""Tests for cv_autoresearch.lightning modules."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
)

from cv_autoresearch.lightning.datamodule import AugmentedDataset, AutoResearchDataModule
from cv_autoresearch.lightning.module import AutoResearchModule
from cv_autoresearch.task import TaskDef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Minimal model for optimizer tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _make_task(loss_fn=None) -> TaskDef:
    """Build a minimal TaskDef for module tests."""
    model = TinyModel()
    if loss_fn is None:
        loss_fn = lambda logits, batch: torch.tensor(0.5, requires_grad=True)  # noqa: E731
    return TaskDef(
        model=model,
        train_dataset=[],  # type: ignore[arg-type]
        val_dataset=[],    # type: ignore[arg-type]
        loss_fn=loss_fn,
        predict=lambda logits, batch: ([], []),
        num_classes=2,
    )


# ---------------------------------------------------------------------------
# AutoResearchModule
# ---------------------------------------------------------------------------


def test_training_step_calls_task_loss_fn() -> None:
    """training_step must call task.loss_fn(logits, batch) and return its result."""
    called_with: list = []

    def loss_fn(logits, batch):
        called_with.append((logits, batch))
        return torch.tensor(0.5, requires_grad=True)

    task = _make_task(loss_fn=loss_fn)
    module = AutoResearchModule(task, {"learning_rate": 1e-3, "optimizer_type": "adam"})
    batch = (torch.zeros(2, 4), torch.zeros(2, dtype=torch.long))

    loss = module.training_step(batch, 0)

    assert len(called_with) == 1
    assert torch.isclose(loss, torch.tensor(0.5))


@pytest.mark.parametrize(
    "optimizer_type, expected_class",
    [
        ("adam", Adam),
        ("adamw", AdamW),
        ("sgd", SGD),
    ],
)
def test_configure_optimizers_optimizer_type(
    optimizer_type: str, expected_class: type
) -> None:
    """configure_optimizers must build the correct optimizer class."""
    task = _make_task()
    module = AutoResearchModule(
        task,
        {"learning_rate": 1e-3, "optimizer_type": optimizer_type, "momentum": 0.9},
    )
    result = module.configure_optimizers()
    assert isinstance(result["optimizer"], expected_class)


@pytest.mark.parametrize(
    "scheduler_name, expected_class",
    [
        ("cosine", CosineAnnealingLR),
        ("step", StepLR),
        ("cosine_with_restarts", CosineAnnealingWarmRestarts),
    ],
)
def test_configure_optimizers_scheduler_type(
    scheduler_name: str, expected_class: type
) -> None:
    """configure_optimizers must build the correct scheduler class."""
    task = _make_task()
    module = AutoResearchModule(
        task,
        {
            "learning_rate": 1e-3,
            "optimizer_type": "adam",
            "lr_scheduler": scheduler_name,
            "lr_scheduler_step_size": 5,
        },
    )
    result = module.configure_optimizers()
    scheduler = result["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, expected_class)


def test_configure_optimizers_returns_pl_compatible_dict() -> None:
    """configure_optimizers must return dict with 'optimizer' and 'lr_scheduler' keys."""
    task = _make_task()
    module = AutoResearchModule(task, {"learning_rate": 1e-3, "optimizer_type": "adam"})
    result = module.configure_optimizers()
    assert "optimizer" in result
    assert "lr_scheduler" in result
    assert "scheduler" in result["lr_scheduler"]


# ---------------------------------------------------------------------------
# AugmentedDataset
# ---------------------------------------------------------------------------


def test_augmented_dataset_len_matches_underlying() -> None:
    """__len__ must return the underlying dataset's length."""
    base = [(np.zeros((4, 4, 3), dtype=np.uint8), i) for i in range(10)]
    transform = MagicMock(return_value={"image": np.zeros((4, 4, 3), dtype=np.uint8)})
    ds = AugmentedDataset(base, transform)
    assert len(ds) == 10


def test_augmented_dataset_calls_transform_with_image_kwarg() -> None:
    """transform must be called with image= keyword argument."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    aug_image = np.ones((4, 4, 3), dtype=np.uint8)
    transform = MagicMock(return_value={"image": aug_image})
    base = [(image, 42)]
    ds = AugmentedDataset(base, transform)

    result_image, result_label = ds[0]

    transform.assert_called_once()
    call_kwargs = transform.call_args.kwargs
    assert "image" in call_kwargs
    np.testing.assert_array_equal(call_kwargs["image"], image)
    np.testing.assert_array_equal(result_image, aug_image)
    assert result_label == 42


# ---------------------------------------------------------------------------
# AutoResearchDataModule
# ---------------------------------------------------------------------------


def _make_dummy_dataset(size: int = 4) -> list:
    return [(np.zeros((4, 4, 3), dtype=np.uint8), i) for i in range(size)]


def test_val_dataloader_uses_raw_dataset() -> None:
    """val_dataloader must NOT wrap val_dataset in AugmentedDataset."""
    train_ds = _make_dummy_dataset(4)
    val_ds = _make_dummy_dataset(2)
    transform = MagicMock(return_value={"image": np.zeros((4, 4, 3))})

    dm = AutoResearchDataModule(
        train_dataset=train_ds,
        val_dataset=val_ds,
        augmentation_pipeline=transform,
        batch_size=2,
        num_workers=0,
    )
    val_loader = dm.val_dataloader()
    # Underlying dataset must be the raw val_ds, not an AugmentedDataset
    assert val_loader.dataset is val_ds


def test_train_dataloader_wraps_in_augmented_dataset() -> None:
    """train_dataloader must wrap train_dataset in AugmentedDataset."""
    train_ds = _make_dummy_dataset(4)
    val_ds = _make_dummy_dataset(2)
    transform = MagicMock(return_value={"image": np.zeros((4, 4, 3))})

    dm = AutoResearchDataModule(
        train_dataset=train_ds,
        val_dataset=val_ds,
        augmentation_pipeline=transform,
        batch_size=2,
        num_workers=0,
    )
    train_loader = dm.train_dataloader()
    assert isinstance(train_loader.dataset, AugmentedDataset)
    assert train_loader.dataset.dataset is train_ds
