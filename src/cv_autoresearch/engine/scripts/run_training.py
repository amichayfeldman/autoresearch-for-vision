"""Console entry for one configured training run."""

from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig

from cv_autoresearch.engine.scripts.prompt_args import apply_prompt_args, consume_prompt_args
from cv_autoresearch.engine.training import run_training

_PROMPT_ARGS = consume_prompt_args(sys.argv)


@hydra.main(version_base=None, config_path="../../../../../configs", config_name="prompt_task")
def main(config: DictConfig) -> None:
    apply_prompt_args(config, _PROMPT_ARGS)
    result = run_training(config)
    print(f"metrics: {result.metrics}")
    print(f"checkpoint: {result.checkpoint_path}")


if __name__ == "__main__":
    main()
