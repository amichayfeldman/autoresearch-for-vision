"""Tests for the cv_autoresearch CLI."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cv_autoresearch.cli import _import_dotted_path, main


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# _import_dotted_path
# ---------------------------------------------------------------------------


def test_import_dotted_path_valid() -> None:
    """Should import a known stdlib class without error."""
    cls = _import_dotted_path("collections.OrderedDict")
    import collections
    assert cls is collections.OrderedDict


def test_import_dotted_path_invalid_module(runner: CliRunner) -> None:
    """Invalid module path must call sys.exit(1) with a clean message."""
    with pytest.raises(SystemExit) as exc_info:
        _import_dotted_path("nonexistent_module_xyz.Foo")
    assert exc_info.value.code == 1


def test_import_dotted_path_invalid_attr(runner: CliRunner) -> None:
    """Valid module but missing attribute must call sys.exit(1)."""
    with pytest.raises(SystemExit) as exc_info:
        _import_dotted_path("collections.NonExistentClass")
    assert exc_info.value.code == 1


def test_import_dotted_path_no_dot(runner: CliRunner) -> None:
    """Path without a dot must call sys.exit(1)."""
    with pytest.raises(SystemExit) as exc_info:
        _import_dotted_path("nodot")
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------


def test_run_command_writes_output_json(runner: CliRunner, tmp_path) -> None:
    """run command must write a JSON file at --output path."""
    output_file = str(tmp_path / "results.json")
    fake_result = {
        "best_metric": {"name": "accuracy", "value": 0.85},
        "best_hyperparams": {"lr": 1e-3},
        "best_augmentations": {},
        "total_trials": 4,
        "failed_trials": 0,
        "top_improvements": [],
    }

    with (
        patch("cv_autoresearch.cli._import_dotted_path") as mock_import,
        patch("cv_autoresearch.cli.run_autoresearch", return_value=fake_result),
    ):
        mock_model_cls = MagicMock(return_value=MagicMock())
        mock_dataset_fn = MagicMock(return_value=(MagicMock(), MagicMock()))
        mock_import.side_effect = [mock_model_cls, mock_dataset_fn]

        result = runner.invoke(main, [
            "run",
            "--trainer-module", "mypackage.models.MyModel",
            "--train-dataset", "mypackage.data.get_datasets",
            "--task", "Test task",
            "--metric", "accuracy",
            "--output", output_file,
            "--storage", "sqlite:///test.db",
            "--device", "cpu",
        ])

    assert result.exit_code == 0, result.output
    with open(output_file) as f:
        saved = json.load(f)
    assert saved["best_metric"]["name"] == "accuracy"


def test_run_command_invalid_trainer_module(runner: CliRunner, tmp_path) -> None:
    """Invalid --trainer-module path must produce a clean error message."""
    output_file = str(tmp_path / "results.json")

    result = runner.invoke(main, [
        "run",
        "--trainer-module", "totally_invalid_module.Foo",
        "--train-dataset", "collections.OrderedDict",
        "--task", "Test task",
        "--metric", "accuracy",
        "--output", output_file,
        "--storage", "sqlite:///test.db",
    ])

    # Should exit with non-zero, output should not be a raw traceback
    assert result.exit_code != 0
    assert "Cannot import" in (result.output + str(result.exception or ""))


# ---------------------------------------------------------------------------
# history command
# ---------------------------------------------------------------------------


def test_history_command_no_studies(runner: CliRunner, tmp_path) -> None:
    """history command with empty storage should print graceful message."""
    db_path = str(tmp_path / "empty.db")

    with patch("optuna.get_all_study_names", return_value=[]):
        result = runner.invoke(main, [
            "history",
            "--storage", f"sqlite:///{db_path}",
        ])

    assert result.exit_code == 0
    assert "No studies found" in result.output


# ---------------------------------------------------------------------------
# resume command
# ---------------------------------------------------------------------------


def test_resume_command_calls_run_autoresearch(runner: CliRunner) -> None:
    """resume command must call run_autoresearch (which uses load_if_exists=True internally)."""
    fake_result = {
        "best_metric": {"name": "accuracy", "value": 0.9},
        "best_hyperparams": {},
        "best_augmentations": {},
        "total_trials": 2,
        "failed_trials": 0,
        "top_improvements": [],
    }

    with (
        patch("cv_autoresearch.cli._import_dotted_path") as mock_import,
        patch("cv_autoresearch.cli.run_autoresearch", return_value=fake_result) as mock_run,
    ):
        mock_model_cls = MagicMock(return_value=MagicMock())
        mock_dataset_fn = MagicMock(return_value=(MagicMock(), MagicMock()))
        mock_import.side_effect = [mock_model_cls, mock_dataset_fn]

        result = runner.invoke(main, [
            "resume",
            "--storage", "sqlite:///existing.db",
            "--trainer-module", "mypackage.models.MyModel",
            "--train-dataset", "mypackage.data.get_datasets",
            "--task", "Resume task",
            "--device", "cpu",
        ])

    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()
    # Verify storage was passed through to config
    call_args = mock_run.call_args
    config_arg = call_args[0][3]  # 4th positional arg is SearchConfig
    assert config_arg.optuna_storage == "sqlite:///existing.db"
