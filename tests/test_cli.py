"""Tests for the cv_autoresearch CLI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner
from omegaconf import OmegaConf

from cv_autoresearch.cli import _apply_task_options, main


def test_run_command_invokes_iteration_manager(tmp_path: Path) -> None:
    config = OmegaConf.create(
        {
            "task": {"prompt": "", "model_path": None},
            "history": {"output_dir": str(tmp_path / "run")},
        }
    )

    with patch("cv_autoresearch.cli._compose_config", return_value=config) as compose_config:
        with patch("cv_autoresearch.cli.manage_iterations", return_value=[SimpleNamespace()]) as manage:
            result = CliRunner().invoke(
                main,
                [
                    "run",
                    "--task-prompt",
                    "Classify surface defects.",
                    "iteration.max_iterations=1",
                ],
            )

    assert result.exit_code == 0, result.output
    compose_config.assert_called_once_with(["iteration.max_iterations=1"])
    manage.assert_called_once()
    assert config.task.prompt == "Classify surface defects."
    assert "iterations: 1" in result.output
    assert str(tmp_path / "run") in result.output


def test_apply_task_options_prefers_prompt_file(tmp_path: Path) -> None:
    prompt_file = tmp_path / "task.md"
    prompt_file.write_text("Detect loose fasteners.")
    model_path = tmp_path / "model.pt"
    config = OmegaConf.create({"task": {"prompt": "", "model_path": None}})

    _apply_task_options(config, "ignored inline prompt", prompt_file, model_path)

    assert config.task.prompt == "Detect loose fasteners."
    assert config.task.prompt_file == str(prompt_file)
    assert config.task.model_path == str(model_path)
