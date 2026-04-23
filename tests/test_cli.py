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
        "best_f1": 0.85,
        "best_hp": {"learning_rate": 1e-3},
        "best_aug": {},
        "total_trials": 4,
    }

    fake_task = MagicMock()
    fake_factory = MagicMock(return_value=fake_task)

    with (
        patch("cv_autoresearch.cli._import_dotted_path", return_value=fake_factory),
        patch("cv_autoresearch.cli.run", return_value=fake_result),
    ):
        result = runner.invoke(main, [
            "run",
            "--task-factory", "mypackage.tasks.make_cifar10_task",
            "--output", output_file,
            "--storage", "sqlite:///test.db",
            "--device", "cpu",
        ])

    assert result.exit_code == 0, result.output
    with open(output_file) as f:
        saved = json.load(f)
    assert saved["best_f1"] == 0.85


def test_run_command_invalid_task_factory(runner: CliRunner, tmp_path) -> None:
    """Invalid --task-factory path must produce a clean error message."""
    output_file = str(tmp_path / "results.json")

    result = runner.invoke(main, [
        "run",
        "--task-factory", "totally_invalid_module.make_task",
        "--output", output_file,
        "--storage", "sqlite:///test.db",
    ])

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
