"""cv-autoresearch CLI entry point."""

from __future__ import annotations

import importlib
import json
import sys
from typing import Any

import click

from cv_autoresearch.config.schema import SearchConfig
from cv_autoresearch.engine.autoresearch import run_autoresearch


@click.group()
def main() -> None:
    """cv-autoresearch: AI-directed CV hyperparameter and augmentation search."""


@main.command("run")
@click.option("--trainer-module", required=True,
              help="Dotted import path to a LightningModule class, e.g. mypackage.models.MyResNet")
@click.option("--train-dataset", required=True,
              help="Dotted import path to a factory function returning (train_ds, val_ds)")
@click.option("--task", required=True,
              help="Free-text task description passed to Claude for metric generation")
@click.option("--total-trials", default=80, show_default=True)
@click.option("--epochs", default=10, show_default=True, help="Epochs per trial")
@click.option("--device", default="cuda", show_default=True)
@click.option("--output", default="./autoresearch_results.json", show_default=True,
              help="Path to write final summary JSON")
@click.option("--storage", default="sqlite:///autoresearch.db", show_default=True,
              help="Optuna storage URL")
def run_cmd(
    trainer_module: str,
    train_dataset: str,
    task: str,
    total_trials: int,
    epochs: int,
    device: str,
    output: str,
    storage: str,
) -> None:
    """Run the full autoresearch pipeline from the command line."""
    model_cls = _import_dotted_path(trainer_module)
    dataset_factory = _import_dotted_path(train_dataset)

    model = model_cls()
    train_ds, val_ds = dataset_factory()

    config = SearchConfig(
        task_description=task,
        total_trials=total_trials,
        epochs_per_trial=epochs,
        device=device,
        optuna_storage=storage,
    )

    result = run_autoresearch(model, train_ds, val_ds, config)

    with open(output, "w") as f:
        json.dump(result, f, indent=2, default=str)

    click.echo(f"Results written to {output}")
    click.echo(f"Best {result['best_metric']['name']}: {result['best_metric']['value']:.4f}")


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


@main.command("resume")
@click.option("--storage", default="sqlite:///autoresearch.db", show_default=True)
@click.option("--trainer-module", required=True)
@click.option("--train-dataset", required=True)
@click.option("--task", required=True, help="Task description for metric generation")
@click.option("--total-trials", default=80, show_default=True)
@click.option("--epochs", default=10, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.pass_context
def resume_cmd(
    ctx: click.Context,
    storage: str,
    trainer_module: str,
    train_dataset: str,
    task: str,
    total_trials: int,
    epochs: int,
    device: str,
) -> None:
    """Resume a previous autoresearch run from stored state.

    Optuna's load_if_exists=True means studies continue from where they left off.
    """
    model_cls = _import_dotted_path(trainer_module)
    dataset_factory = _import_dotted_path(train_dataset)

    model = model_cls()
    train_ds, val_ds = dataset_factory()

    config = SearchConfig(
        task_description=task,
        total_trials=total_trials,
        epochs_per_trial=epochs,
        device=device,
        optuna_storage=storage,
    )

    result = run_autoresearch(model, train_ds, val_ds, config)
    click.echo(f"Best {result['best_metric']['name']}: {result['best_metric']['value']:.4f}")


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
