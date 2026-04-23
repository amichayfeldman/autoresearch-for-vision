"""Shared engine utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def to_plain_config(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Convert Hydra/OmegaConf config to JSON-serializable containers."""
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return json.loads(json.dumps(config, default=str))


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON with stable formatting."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file if it exists."""
    target = Path(path)
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text().splitlines() if line.strip()]
