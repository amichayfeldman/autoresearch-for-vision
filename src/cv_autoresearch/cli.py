"""cv-autoresearch CLI entry point."""

from __future__ import annotations

from pathlib import Path

import click
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from cv_autoresearch.engine.manager import manage_iterations


@click.group()
def main() -> None:
    """Run agent-managed computer-vision autoresearch."""


@main.command("run", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option("--task-prompt", default=None, help="Free-text CV task for the wiring agent.")
@click.option(
    "--task-prompt-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="File containing the CV task prompt.",
)
@click.option(
    "--model-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional .pt model path for the task-wiring phase.",
)
@click.pass_context
def run_cmd(
    ctx: click.Context,
    task_prompt: str | None,
    task_prompt_file: Path | None,
    model_path: Path | None,
) -> None:
    """Run baseline plus agent-controlled iterations.

    Unknown trailing arguments are passed through as Hydra overrides, for example:
    ``iteration.max_iterations=3 optimizer.learning_rate=0.001``.
    """
    config = _compose_config(list(ctx.args))
    _apply_task_options(config, task_prompt, task_prompt_file, model_path)
    records = manage_iterations(config, repo_root=_repo_root())
    click.echo(f"iterations: {len(records)}")
    click.echo(f"output: {config.history.output_dir}")


def _compose_config(overrides: list[str]) -> DictConfig:
    config_dir = _repo_root() / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="prompt_task", overrides=overrides)


def _apply_task_options(
    config: DictConfig,
    task_prompt: str | None,
    task_prompt_file: Path | None,
    model_path: Path | None,
) -> None:
    if task_prompt_file is not None:
        config.task.prompt = task_prompt_file.read_text()
        config.task.prompt_file = str(task_prompt_file)
    elif task_prompt is not None:
        config.task.prompt = task_prompt
    if model_path is not None:
        config.task.model_path = str(model_path)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
