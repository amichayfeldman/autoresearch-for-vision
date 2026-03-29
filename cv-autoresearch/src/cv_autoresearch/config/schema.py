"""SearchConfig dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SearchConfig:
    """Configuration for an autoresearch run.

    All file outputs (checkpoints, log, plot, metric config) are written
    under ``output_dir``. The derived paths are:

    - ``{output_dir}/checkpoints/``
    - ``{output_dir}/autoresearch.jsonl``
    - ``{output_dir}/improvement_curve.png``
    - ``{output_dir}/config/metrics/generated.yaml``
    """

    task_description: str
    # Number of directive steps (each step runs exploit_trials_per_directive Bayesian trials).
    # Actual training runs = total_trials × exploit_trials_per_directive.
    total_trials: int = 80
    epochs_per_trial: int = 7
    exploit_trials_per_directive: int = 10
    optuna_storage: str = "sqlite:///autoresearch.db"
    optuna_seed: int = 42
    hp_overrides: dict[str, Any] = field(default_factory=dict)
    aug_overrides: dict[str, Any] = field(default_factory=dict)
    # Accepts "cuda", "cpu", or "mps"; defaults to "cuda" for GPU-first training.
    device: str = "cuda"
    num_workers: int = 4
    output_dir: str = "./autoresearch_output"

    # Derived path helpers — read-only, computed from output_dir.
    @property
    def checkpoint_dir(self) -> str:
        return str(Path(self.output_dir) / "checkpoints")

    @property
    def metric_config_path(self) -> str:
        return str(Path(self.output_dir) / "config" / "metrics" / "generated.yaml")

    @property
    def log_path(self) -> str:
        return str(Path(self.output_dir) / "autoresearch.jsonl")

    @property
    def plot_path(self) -> str:
        return str(Path(self.output_dir) / "improvement_curve.png")
