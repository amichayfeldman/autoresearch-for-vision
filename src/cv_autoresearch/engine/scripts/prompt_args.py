"""Prompt/model CLI arguments consumed before Hydra parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, open_dict


@dataclass(frozen=True)
class PromptArgs:
    """User-supplied task prompt inputs."""

    task_prompt: str | None = None
    task_prompt_file: str | None = None
    model_path: str | None = None


def consume_prompt_args(argv: list[str]) -> PromptArgs:
    """Remove prompt-specific args from argv so Hydra accepts the rest."""
    task_prompt: str | None = None
    task_prompt_file: str | None = None
    model_path: str | None = None
    kept = [argv[0]]
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg in {"--task-prompt", "--task-prompt-file", "--model-path"}:
            if index + 1 >= len(argv):
                raise SystemExit(f"{arg} requires a value")
            value = argv[index + 1]
            index += 2
        elif arg.startswith("--task-prompt="):
            value = arg.split("=", 1)[1]
            arg = "--task-prompt"
            index += 1
        elif arg.startswith("--task-prompt-file="):
            value = arg.split("=", 1)[1]
            arg = "--task-prompt-file"
            index += 1
        elif arg.startswith("--model-path="):
            value = arg.split("=", 1)[1]
            arg = "--model-path"
            index += 1
        else:
            kept.append(arg)
            index += 1
            continue

        if arg == "--task-prompt":
            task_prompt = value
        elif arg == "--task-prompt-file":
            task_prompt_file = value
        elif arg == "--model-path":
            model_path = value

    argv[:] = kept
    return PromptArgs(task_prompt=task_prompt, task_prompt_file=task_prompt_file, model_path=model_path)


def apply_prompt_args(config: DictConfig, args: PromptArgs) -> None:
    """Write resolved prompt/model fields into the Hydra config."""
    prompt = args.task_prompt
    if args.task_prompt_file:
        prompt = Path(args.task_prompt_file).read_text()
    with open_dict(config):
        if "task" not in config:
            config.task = {}
        if prompt is not None:
            config.task.prompt = prompt
            config.task.prompt_file = args.task_prompt_file
        if args.model_path is not None:
            config.task.model_path = args.model_path
