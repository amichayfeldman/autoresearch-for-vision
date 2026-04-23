"""Console entry for managed training iterations."""

from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig

from cv_autoresearch.engine.manager import manage_iterations
from cv_autoresearch.engine.scripts.prompt_args import apply_prompt_args, consume_prompt_args

_PROMPT_ARGS = consume_prompt_args(sys.argv)


@hydra.main(version_base=None, config_path="../../../../../configs", config_name="prompt_task")
def main(config: DictConfig) -> None:
    apply_prompt_args(config, _PROMPT_ARGS)
    records = manage_iterations(config)
    print(f"iterations: {len(records)}")
    print(f"output: {config.history.output_dir}")


if __name__ == "__main__":
    main()
