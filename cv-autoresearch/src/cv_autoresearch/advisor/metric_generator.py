"""Metric config generator using claude -p."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf


def generate_metric_config(
    task_description: str,
    output_path: str,
) -> tuple[DictConfig, str, bool]:
    """Generate a Hydra metric config using claude -p.

    Calls claude -p once at startup with the task description. Claude selects
    the appropriate torchmetrics metric, a human-readable name, and whether
    higher values are better.

    Args:
        task_description: Free-text description of the CV task.
        output_path: Path to write the generated YAML (metric_config only).

    Returns:
        Tuple of (DictConfig ready for hydra.utils.instantiate, metric_name, higher_is_better).
    """
    prompt = _build_metric_prompt(task_description)
    response = _call_claude(prompt)
    metric_cfg_dict, metric_name, higher_is_better = _parse_metric_response(response)
    cfg = _write_and_load_config(metric_cfg_dict, output_path)
    return cfg, metric_name, higher_is_better


def _build_metric_prompt(task_description: str) -> str:
    """Build the claude -p prompt for metric generation.

    Returns:
        Prompt string requesting structured YAML with metric config, name, and direction.
    """
    return f"""You are an expert in PyTorch and torchmetrics.

Task description: {task_description}

Select the most appropriate primary metric for this computer vision task.
Output EXACTLY this YAML structure, no explanation:

metric_config:
  _target_: torchmetrics.Accuracy
  task: multiclass
  num_classes: 10
metric_name: accuracy
higher_is_better: true

Rules:
- metric_config must have a _target_ key pointing to a torchmetrics class
- metric_name is a short snake_case string (e.g. accuracy, auroc, miou, map)
- higher_is_better is true if larger values mean better performance, false for error metrics (rmse, mae, loss)
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


def _parse_metric_response(response: str) -> tuple[dict, str, bool]:
    """Extract metric config, name, and direction from claude's structured response.

    Strips markdown code fences if present, parses the top-level YAML which
    must contain metric_config, metric_name, and higher_is_better keys.

    Args:
        response: Raw string response from claude.

    Returns:
        Tuple of (metric_cfg_dict, metric_name, higher_is_better).

    Raises:
        KeyError: If required keys are missing from the response.
        yaml.YAMLError: If the response cannot be parsed as YAML.
    """
    full = _parse_yaml_from_response(response)
    return full["metric_config"], str(full["metric_name"]), bool(full["higher_is_better"])


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
