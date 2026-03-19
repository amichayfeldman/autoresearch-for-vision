"""SearchConfig dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchConfig:
    """Configuration for an autoresearch run."""

    task_description: str
    primary_metric: str
    higher_is_better: bool
    hp_trials: int = 50
    aug_trials: int = 30
    epochs_per_trial: int = 10
    exploit_trials_per_directive: int = 5
    optuna_storage: str = "sqlite:///autoresearch.db"
    optuna_seed: int = 42
    hp_overrides: dict = field(default_factory=dict)
    aug_overrides: dict = field(default_factory=dict)
    device: str = "cuda"
    num_workers: int = 4
    checkpoint_dir: str = "./checkpoints"
    metric_config_path: str = "config/metrics/generated.yaml"
