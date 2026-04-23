"""cv-autoresearch CLI entry point."""

from __future__ import annotations

import importlib
import json
import sys
from typing import Any

import click

from cv_autoresearch.autoresearch import run


@click.group()
def main() -> None:
    """cv-autoresearch: AI-directed CV hyperparameter and augmentation search."""


@main.command("run")
@click.option("--task-factory", required=True,
              help="Dotted import path to a zero-arg function returning a TaskDef, "
                   "e.g. mypackage.tasks.make_cifar10_task")
@click.option("--total-directives", default=20, show_default=True)
@click.option("--trials-per-directive", default=5, show_default=True)
@click.option("--epochs", default=7, show_default=True, help="Epochs per trial")
@click.option("--device", default="cuda", show_default=True)
@click.option("--output", default="./autoresearch_results.json", show_default=True,
              help="Path to write final summary JSON")
@click.option("--storage", default="sqlite:///autoresearch.db", show_default=True,
              help="Optuna storage URL")
def run_cmd(
    task_factory: str,
    total_directives: int,
    trials_per_directive: int,
    epochs: int,
    device: str,
    output: str,
    storage: str,
) -> None:
    """Run the full autoresearch pipeline from the command line."""
    factory = _import_dotted_path(task_factory)
    task = factory()

    result = run(
        task,
        total_directives=total_directives,
        trials_per_directive=trials_per_directive,
        epochs_per_trial=epochs,
        device=device,
        optuna_storage=storage,
    )

    with open(output, "w") as f:
        json.dump(result, f, indent=2, default=str)

    click.echo(f"Results written to {output}")
    click.echo(f"Best F1: {result['best_f1']:.4f}")


@main.command("history")
@click.option("--storage", default="sqlite:///autoresearch.db", show_default=True)
@click.option("--top", default=10, show_default=True, help="Show top N improvements")
def history_cmd(storage: str, top: int) -> None:
    """Print experiment history from an existing Optuna storage."""
    import optuna
    from cv_autoresearch.search.history import SearchHistory

    try:
        study_names = optuna.get_all_study_names(storage)
    except Exception as exc:
        click.echo(f"Could not load storage: {exc}", err=True)
        sys.exit(1)

    if not study_names:
        click.echo("No studies found in storage.")
        return

    history = SearchHistory()
    click.echo(f"Studies found: {', '.join(study_names)}")
    click.echo(f"\nTop {top} improvements (by |delta|):")
    for entry in history.best_entries(top_k=top):
        click.echo(
            f"  trial={entry.trial_id} mode={entry.mode.value} "
            f"delta={entry.delta:.4f} param={entry.param_name}"
        )

    failed = history.failed_entries()
    if failed:
        click.echo(f"\nFailed trials ({len(failed)}):")
        for entry in failed:
            click.echo(f"  trial={entry.trial_id}: {entry.error_message}")


def _import_dotted_path(dotted_path: str) -> Any:
    """Import a class or function from a dotted Python path.

    Args:
        dotted_path: Dotted import path, e.g. "mypackage.module.ClassName".

    Returns:
        The imported class or function.

    Raises:
        SystemExit: If the path cannot be imported, with a clean error message.
    """
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        click.echo(f"Invalid import path: '{dotted_path}' (expected 'module.ClassName')", err=True)
        sys.exit(1)
    module_path, attr_name = parts
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        click.echo(f"Cannot import module '{module_path}': {exc}", err=True)
        sys.exit(1)
    try:
        return getattr(module, attr_name)
    except AttributeError:
        click.echo(f"Module '{module_path}' has no attribute '{attr_name}'", err=True)
        sys.exit(1)
