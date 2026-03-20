"""Optuna study creation and trial execution for cv-autoresearch."""

from __future__ import annotations

from typing import Any, Callable

import optuna

from cv_autoresearch.search.history import SearchHistory
from cv_autoresearch.search.space import exploit_space


def create_study(
    name: str,
    direction: str,
    storage: str,
    seed: int,
) -> optuna.Study:
    """Create (or load) an Optuna study with a seeded TPE sampler.

    Args:
        name: Study name used for identification in storage.
        direction: Optimization direction, either "minimize" or "maximize".
        storage: Optuna storage URL (e.g. "sqlite:///study.db").
        seed: Random seed for the TPE sampler for reproducibility.

    Returns:
        An optuna.Study instance.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    return optuna.create_study(
        study_name=name,
        direction=direction,
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
    )


def run_exploit_phase(
    study: optuna.Study,
    objective_fn: Callable[[dict[str, Any]], float],
    history: SearchHistory,
    baseline_config: dict[str, Any],
    param_name: str,
    param_range: list[float],
    n_trials: int,
) -> list[float]:
    """Run multiple exploit trials varying a single parameter around its range.

    Each trial uses exploit_space to build a config from baseline_config with
    only param_name varied over param_range.

    Args:
        study: The Optuna study to run trials on.
        objective_fn: Function that receives a config dict and returns a float metric.
        history: SearchHistory used for duplicate detection and registration.
        baseline_config: Current best configuration to vary from.
        param_name: The parameter to sweep over.
        param_range: Either [low, high] float range, or list of categorical choices.
        n_trials: Number of exploit trials to run.

    Returns:
        List of objective values from each completed trial.
    """
    objective_values: list[float] = []

    def _objective(trial: optuna.Trial) -> float:
        config = exploit_space(trial, param_name, param_range, baseline_config)
        if history.is_duplicate(config):
            raise optuna.TrialPruned()
        history.register(config)
        value = objective_fn(config)
        objective_values.append(value)
        return value

    study.optimize(_objective, n_trials=n_trials)
    return objective_values
