"""Tests for cv_autoresearch.engine.evaluator module."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
import torch
import torchmetrics
from torch import nn
from torch.utils.data import Dataset

from cv_autoresearch.engine.evaluator import (
    compute_primary_metric,
    run_val_inference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ToyDataset(Dataset):
    """Minimal dataset returning (image_tensor, label) pairs."""

    def __init__(self, size: int = 8, label_offset: int = 0) -> None:
        self.size = size
        self.label_offset = label_offset

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        label = torch.tensor((idx + self.label_offset) % 2, dtype=torch.long)
        return torch.zeros(3, 4, 4), label


class ToyModel(nn.Module):
    """Simple model that flattens input and projects to num_classes."""

    def __init__(self, in_features: int = 3 * 4 * 4, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(start_dim=1))


# ---------------------------------------------------------------------------
# compute_primary_metric tests
# ---------------------------------------------------------------------------


def test_compute_primary_metric_reset_called_before_update() -> None:
    """metric.reset() must be called before metric.update() to avoid state leakage."""
    mock_metric = MagicMock()
    mock_metric.compute.return_value = torch.tensor(0.5)

    preds = torch.rand(4, 2)
    targets = torch.randint(0, 2, (4,))

    compute_primary_metric(preds, targets, mock_metric)

    call_order = [c[0] for c in mock_metric.method_calls]
    assert "reset" in call_order, "reset() was never called"
    assert "update" in call_order, "update() was never called"
    reset_idx = call_order.index("reset")
    update_idx = call_order.index("update")
    assert reset_idx < update_idx, "reset() must be called before update()"


def test_compute_primary_metric_returns_float() -> None:
    """Return value must be a Python float."""
    mock_metric = MagicMock()
    mock_metric.compute.return_value = torch.tensor(0.75)

    preds = torch.rand(4, 2)
    targets = torch.randint(0, 2, (4,))

    result = compute_primary_metric(preds, targets, mock_metric)

    assert isinstance(result, float), f"Expected float, got {type(result)}"


@pytest.mark.parametrize(
    "preds,targets,expected",
    [
        # All correct: accuracy = 1.0
        (
            torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
            torch.tensor([0, 1]),
            1.0,
        ),
        # All wrong: accuracy = 0.0
        (
            torch.tensor([[0.1, 0.9], [0.9, 0.1]]),
            torch.tensor([0, 1]),
            0.0,
        ),
        # Half correct: accuracy = 0.5
        (
            torch.tensor([[0.9, 0.1], [0.9, 0.1]]),
            torch.tensor([0, 1]),
            0.5,
        ),
    ],
)
def test_compute_primary_metric_accuracy_values(
    preds: torch.Tensor, targets: torch.Tensor, expected: float
) -> None:
    """Real torchmetrics.Accuracy returns correct scalar for known inputs."""
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=2)
    result = compute_primary_metric(preds, targets, metric)
    assert result == pytest.approx(expected, abs=1e-5)


def test_compute_primary_metric_resets_accumulated_state() -> None:
    """Metric state from a previous call must not bleed into next call."""
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    # First call — all correct
    preds_correct = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    targets = torch.tensor([0, 1])
    compute_primary_metric(preds_correct, targets, metric)

    # Second call — all wrong; should not be influenced by first call
    preds_wrong = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    result = compute_primary_metric(preds_wrong, targets, metric)

    assert result == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# run_val_inference tests
# ---------------------------------------------------------------------------


def test_run_val_inference_output_shapes() -> None:
    """Returned tensors must have first dim equal to dataset size."""
    dataset = ToyDataset(size=8)
    model = ToyModel()

    preds, targets = run_val_inference(
        model=model,
        val_dataset=dataset,
        batch_size=4,
        device="cpu",
        num_workers=0,
    )

    assert preds.shape[0] == len(dataset), f"Expected {len(dataset)} predictions, got {preds.shape[0]}"
    assert targets.shape[0] == len(dataset), f"Expected {len(dataset)} targets, got {targets.shape[0]}"


def test_run_val_inference_model_set_to_eval() -> None:
    """model.eval() must be called during inference."""
    dataset = ToyDataset(size=4)
    model = ToyModel()

    with patch.object(model, "eval", wraps=model.eval) as mock_eval:
        run_val_inference(
            model=model,
            val_dataset=dataset,
            batch_size=4,
            device="cpu",
            num_workers=0,
        )
        mock_eval.assert_called_once()


def test_run_val_inference_returns_cpu_tensors() -> None:
    """Returned tensors must reside on CPU."""
    dataset = ToyDataset(size=6)
    model = ToyModel()

    preds, targets = run_val_inference(
        model=model,
        val_dataset=dataset,
        batch_size=3,
        device="cpu",
        num_workers=0,
    )

    assert preds.device.type == "cpu"
    assert targets.device.type == "cpu"


def test_run_val_inference_no_grad_active() -> None:
    """Gradients must not be computed during inference (requires_grad=False)."""
    dataset = ToyDataset(size=4)
    model = ToyModel()

    preds, _ = run_val_inference(
        model=model,
        val_dataset=dataset,
        batch_size=4,
        device="cpu",
        num_workers=0,
    )

    assert not preds.requires_grad, "Predictions should not have requires_grad=True"


def test_run_val_inference_full_dataset_collected() -> None:
    """All samples from the dataset must appear in output (multiple batches)."""
    size = 10
    dataset = ToyDataset(size=size)
    model = ToyModel()

    preds, targets = run_val_inference(
        model=model,
        val_dataset=dataset,
        batch_size=3,  # Intentionally not a divisor of size
        device="cpu",
        num_workers=0,
    )

    assert preds.shape[0] == size
    assert targets.shape[0] == size
