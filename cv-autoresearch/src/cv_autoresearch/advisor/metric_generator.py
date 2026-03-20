"""Metric config generator using claude -p."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf


def generate_metric_config(
    task_description: str,
    primary_metric: str,
    output_path: str,
) -> DictConfig:
    """Generate a Hydra metric config using claude -p.

    Calls claude -p once at startup with the task description and asks it to
    produce a Hydra _target_ config for a torchmetrics Metric class.
    Writes the result to output_path as YAML.
    Loads and returns it as an OmegaConf DictConfig.

    Args:
        task_description: Free-text description of the CV task.
        primary_metric: Name of the metric to be generated.
        output_path: Path to write the generated YAML.

    Returns:
        OmegaConf DictConfig ready for hydra.utils.instantiate.
    """
    prompt = _build_metric_prompt(task_description, primary_metric)
    response = _call_claude(prompt)
    config_dict = _parse_yaml_from_response(response)
    return _write_and_load_config(config_dict, output_path)


def _build_metric_prompt(task_description: str, primary_metric: str) -> str:
    """Build the claude -p prompt for metric generation.

    Returns:
        Prompt string requesting Hydra _target_ YAML for a torchmetrics metric.
    """
    return f"""You are an expert in PyTorch and torchmetrics.

Task description: {task_description}
Requested metric name: {primary_metric}

Generate a valid Hydra _target_ YAML config that instantiates a torchmetrics metric
for this task. The config must be valid YAML with a `_target_` key pointing to a
torchmetrics class (e.g. torchmetrics.Accuracy, torchmetrics.detection.MeanAveragePrecision, etc.)
and any required constructor arguments.

Output ONLY the YAML config, no explanation.

Example output format:
_target_: torchmetrics.Accuracy
task: multiclass
num_classes: 10
"""


def _call_claude(prompt: str) -> str:
    """Run `claude -p <prompt>` and return stdout.

    Args:
        prompt: The prompt to pass to claude.

    Returns:
        stdout from claude as a string.

    Raises:
        RuntimeError: If claude -p returns a non-zero exit code.
    """
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed: {result.stderr}")
    return result.stdout.strip()


def _parse_yaml_from_response(response: str) -> dict:
    """Extract and parse YAML from claude response.

    Strips markdown code fences (```yaml ... ``` or ``` ... ```) if present,
    then parses with yaml.safe_load.

    Args:
        response: Raw string response from claude.

    Returns:
        Parsed dict from YAML content.
    """
    text = response.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove opening fence (```yaml or ```)
        lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return yaml.safe_load(text)


def _write_and_load_config(config_dict: dict, output_path: str) -> DictConfig:
    """Write dict to YAML file and return as OmegaConf DictConfig.

    Args:
        config_dict: Dict to write.
        output_path: File path to write YAML to.

    Returns:
        OmegaConf DictConfig loaded from the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    return OmegaConf.load(path)
