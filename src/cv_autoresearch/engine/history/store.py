"""JSONL and artifact persistence for training iterations."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from cv_autoresearch.engine.history.records import BaselineState, IterationRecord
from cv_autoresearch.engine.utils import ensure_dir, read_jsonl, write_json


class HistoryStore:
    """Persist JSONL records and per-iteration artifacts."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.iterations_dir = ensure_dir(self.output_dir / "iterations")
        self.baseline_dir = ensure_dir(self.output_dir / "baseline")
        self.figures_dir = ensure_dir(self.output_dir / "figures")
        self.jsonl_path = self.output_dir / "history.jsonl"

    def iteration_dir(self, iteration_id: int) -> Path:
        if iteration_id < 0:
            return ensure_dir(self.iterations_dir / "task_wiring")
        return ensure_dir(self.iterations_dir / f"{iteration_id:04d}")

    def append(self, record: IterationRecord) -> None:
        """Append a record to JSONL and save the same payload per iteration."""
        payload = record.to_dict()
        with self.jsonl_path.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        write_json(self.iteration_dir(record.iteration_id) / "record.json", payload)
        if record.patch:
            (self.iteration_dir(record.iteration_id) / "change.patch").write_text(record.patch)

    def records(self) -> list[dict[str, Any]]:
        return read_jsonl(self.jsonl_path)

    def promote_baseline(self, record: IterationRecord) -> BaselineState:
        """Copy promoted artifacts into the stable baseline directory."""
        if record.checkpoint_path is None:
            raise ValueError("Cannot promote a record without checkpoint_path")
        checkpoint_src = Path(record.checkpoint_path)
        config_src = self.iteration_dir(record.iteration_id) / "resolved_config.json"
        checkpoint_dst = self.baseline_dir / "checkpoint.pt"
        config_dst = self.baseline_dir / "resolved_config.json"
        shutil.copy2(checkpoint_src, checkpoint_dst)
        if config_src.exists():
            shutil.copy2(config_src, config_dst)
        write_json(self.baseline_dir / "metrics.json", record.metrics)
        state = BaselineState(
            iteration_id=record.iteration_id,
            primary_metric_value=float(record.primary_metric_after or 0.0),
            checkpoint_path=str(checkpoint_dst),
            config_path=str(config_dst),
            metrics=record.metrics,
        )
        write_json(self.baseline_dir / "state.json", state.__dict__)
        return state
