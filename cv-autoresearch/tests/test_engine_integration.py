"""Integration tests for the autoresearch engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset

from cv_autoresearch.config.schema import SearchConfig
from cv_autoresearch.engine.autoresearch import run_autoresearch
from cv_autoresearch.engine.baseline import should_update_baseline, update_baseline
from cv_autoresearch.engine.reporting import generate_summary
from cv_autoresearch.search.history import HistoryEntry, SearchHistory
from cv_autoresearch.types import Baseline, Directive, SearchMode, TrialId, TrialStatus


# ---------------------------------------------------------------------------
# Toy fixtures
# ---------------------------------------------------------------------------


class TinyDataset(Dataset):
    """Minimal dataset returning tiny tensors for fast tests."""

    def __init__(self, size: int = 8) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(3, 8, 8), torch.tensor(0, dtype=torch.long)


class TinyModel(LightningModule):
    """Minimal LightningModule for integration tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3 * 8 * 8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.view(x.size(0), -1))

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        return torch.nn.functional.cross_entropy(self(x), y)

    def configure_optimizers(self) -> None:
        pass  # Overridden by AutoResearchModule


@pytest.fixture()
def tiny_config() -> SearchConfig:
    return SearchConfig(
        task_description="Binary classification test",
        primary_metric="accuracy",
        higher_is_better=True,
        total_trials=4,
        epochs_per_trial=1,
        exploit_trials_per_directive=1,
        optuna_storage="sqlite:///test_autoresearch.db",
        device="cpu",
        num_workers=0,
    )


@pytest.fixture()
def fake_directive() -> Directive:
    return Directive(
        mode=SearchMode.EXPLORE,
        target_param="learning_rate",
        target_range=None,
        reason="Test directive",
    )


# ---------------------------------------------------------------------------
# Baseline module tests
# ---------------------------------------------------------------------------


def test_should_update_higher_is_better() -> None:
    assert should_update_baseline(0.9, 0.8, higher_is_better=True) is True
    assert should_update_baseline(0.7, 0.8, higher_is_better=True) is False


def test_should_update_lower_is_better() -> None:
    assert should_update_baseline(0.1, 0.8, higher_is_better=False) is True
    assert should_update_baseline(0.9, 0.8, higher_is_better=False) is False


def test_update_baseline_immutability() -> None:
    original = Baseline(
        primary_metric_value=0.5,
        hyperparams={"lr": 1e-3},
        augmentation_config={},
        trial_id=None,
    )
    updated = update_baseline(original, new_value=0.7)
    assert original.primary_metric_value == 0.5
    assert updated.primary_metric_value == 0.7


# ---------------------------------------------------------------------------
# Reporting tests
# ---------------------------------------------------------------------------


def _make_entry(
    trial_id: int,
    delta: float,
    status: TrialStatus,
    directive: Directive | None = None,
) -> HistoryEntry:
    if directive is None:
        directive = Directive(SearchMode.EXPLORE, "learning_rate", None, "test")
    before = 0.5
    after = before + delta if status != TrialStatus.FAILED else None
    return HistoryEntry(
        trial_id=TrialId(trial_id),
        mode=SearchMode.EXPLORE,
        directive=directive,
        param_name=None,
        param_value=None,
        metric_before=before,
        metric_after=after,
        optuna_objective_value=after,
        improved=delta > 0,
        status=status,
        error_message="OOM" if status == TrialStatus.FAILED else None,
    )


def test_generate_summary_contains_required_keys() -> None:
    history = SearchHistory()
    history.record(_make_entry(1, 0.1, TrialStatus.SUCCESS))
    history.record(_make_entry(2, -0.05, TrialStatus.SUCCESS))
    baseline = Baseline(0.6, {"lr": 1e-3}, {})
    config = SearchConfig("task", "accuracy", True)

    result = generate_summary(baseline, history, config)

    assert "best_metric" in result
    assert result["best_metric"]["name"] == "accuracy"
    assert result["best_metric"]["value"] == 0.6
    assert "best_hyperparams" in result
    assert "best_augmentations" in result
    assert "total_trials" in result
    assert result["total_trials"] == 2
    assert "top_improvements" in result
    assert "failed_trials" in result


def test_generate_summary_failed_trials_counted() -> None:
    history = SearchHistory()
    history.record(_make_entry(1, 0.1, TrialStatus.SUCCESS))
    history.record(_make_entry(2, 0.0, TrialStatus.FAILED))
    baseline = Baseline(0.6, {}, {})
    config = SearchConfig("task", "accuracy", True)

    result = generate_summary(baseline, history, config)

    assert result["failed_trials"] == 1


# ---------------------------------------------------------------------------
# Integration: run_autoresearch with mocks
# ---------------------------------------------------------------------------


def test_run_autoresearch_returns_summary_keys(tiny_config: SearchConfig) -> None:
    """run_autoresearch with mocked claude and mocked training returns valid summary."""
    train_ds = TinyDataset(8)
    val_ds = TinyDataset(4)
    model = TinyModel()

    mock_metric_value = 0.75

    with (
        patch("cv_autoresearch.engine.autoresearch.generate_metric_config") as mock_gen,
        patch("cv_autoresearch.engine.autoresearch.instantiate_metric") as mock_inst,
        patch("cv_autoresearch.engine.autoresearch._train_and_evaluate") as mock_train,
        patch("cv_autoresearch.advisor.search_director._call_claude") as mock_claude,
    ):
        # Mock metric generation
        mock_gen.return_value = MagicMock()
        mock_metric = MagicMock()
        mock_metric.compute.return_value = torch.tensor(mock_metric_value)
        mock_inst.return_value = mock_metric

        # Mock training always returns a fixed metric
        mock_train.return_value = mock_metric_value

        # Mock Claude returning EXPLORE directive with a known param
        mock_claude.return_value = (
            "MODE: EXPLORE\nPARAM: learning_rate\nRANGE: NONE\nREASON: Exploring."
        )

        result = run_autoresearch(model, train_ds, val_ds, tiny_config)

    assert "best_metric" in result
    assert "best_hyperparams" in result
    assert "best_augmentations" in result
    assert "total_trials" in result
    assert "failed_trials" in result


def test_run_autoresearch_failed_trial_does_not_update_baseline(
    tiny_config: SearchConfig,
) -> None:
    """A RuntimeError during training must record FAILED and leave baseline unchanged."""
    train_ds = TinyDataset(8)
    val_ds = TinyDataset(4)
    model = TinyModel()

    initial_metric = 0.6
    call_count = 0

    def flaky_train(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:  # first calls are baseline + first trial
            if call_count == 1:
                return initial_metric
            raise RuntimeError("CUDA out of memory")
        return initial_metric + 0.1

    with (
        patch("cv_autoresearch.engine.autoresearch.generate_metric_config") as mock_gen,
        patch("cv_autoresearch.engine.autoresearch.instantiate_metric") as mock_inst,
        patch("cv_autoresearch.engine.autoresearch._train_and_evaluate", side_effect=flaky_train),
        patch("cv_autoresearch.advisor.search_director._call_claude") as mock_claude,
    ):
        mock_gen.return_value = MagicMock()
        mock_inst.return_value = MagicMock()
        mock_claude.return_value = (
            "MODE: EXPLORE\nPARAM: learning_rate\nRANGE: NONE\nREASON: Exploring."
        )

        result = run_autoresearch(model, train_ds, val_ds, tiny_config)

    # Some trials should be FAILED
    assert "failed_trials" in result
    assert result["failed_trials"] >= 1
