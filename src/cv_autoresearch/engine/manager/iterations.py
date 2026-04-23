"""Baseline plus agent-iteration manager."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cv_autoresearch.engine.history import BaselineState, HistoryStore, IterationRecord
from cv_autoresearch.engine.manager.agent import (
    build_agent_prompt,
    build_task_wiring_prompt,
    run_agent,
)
from cv_autoresearch.engine.manager.boundaries import (
    detect_multiple_unrelated_changes,
    validate_changed_files,
)
from cv_autoresearch.engine.manager.gitdiff import changed_files, diff_for_files, new_changes
from cv_autoresearch.engine.manager.visualization import plot_f1_progress
from cv_autoresearch.engine.task_wiring import pretrain_evaluate
from cv_autoresearch.engine.training import run_training
from cv_autoresearch.engine.training.runner import validate_pretrain_metrics
from cv_autoresearch.engine.utils import to_plain_config, write_json


class IterationManager:
    """Run baseline and agent-controlled training iterations."""

    def __init__(self, config: Any, *, repo_root: str | Path = ".") -> None:
        self.config = config
        self.repo_root = Path(repo_root)
        self.store = HistoryStore(config.history.output_dir)
        self.primary_metric = str(config.evaluation.primary_metric)
        self.previous_insight = ""

    def run(self) -> list[IterationRecord]:
        """Run iteration 0 baseline and configured agent iterations."""
        records: list[IterationRecord] = []
        wiring = self._run_task_wiring_phase()
        records.append(wiring)
        if wiring.status != "wired":
            self._write_summary(records)
            return records

        baseline = self._run_iteration(0, parent=None, changed=[], patch="", summary="baseline")
        records.append(baseline)
        if baseline.status != "success" or not baseline.promoted:
            self._write_summary(records)
            return records
        baseline_state = self.store.promote_baseline(baseline)

        for iteration_id in range(1, int(self.config.iteration.max_iterations) + 1):
            before = changed_files(self.repo_root)
            prompt = build_agent_prompt(
                history_text="\n".join(r.one_change_summary for r in records),
                baseline=baseline_state.__dict__,
                previous_insight=self.previous_insight,
                metrics=baseline_state.metrics,
                editable_paths=list(self.config.agent.editable_paths),
                forbidden_paths=list(self.config.agent.forbidden_paths),
            )
            agent_result = run_agent(
                str(self.config.agent.command),
                prompt,
                int(self.config.agent.timeout_seconds),
            )
            after = changed_files(self.repo_root)
            changed = new_changes(before, after)
            patch = diff_for_files(changed, self.repo_root)
            validation = validate_changed_files(
                changed,
                editable_paths=list(self.config.agent.editable_paths),
                forbidden_paths=list(self.config.agent.forbidden_paths),
            )
            reasons = list(validation.reasons)
            if detect_multiple_unrelated_changes(changed):
                reasons.append("multiple unrelated changes detected")
            if agent_result.returncode != 0:
                reasons.append(agent_result.text or "agent command failed")
            if reasons:
                rejected = self._rejected_record(iteration_id, baseline_state, changed, patch, reasons)
                self.store.append(rejected)
                records.append(rejected)
                continue

            record = self._run_iteration(
                iteration_id,
                parent=baseline_state,
                changed=changed,
                patch=patch,
                summary=_summarize_agent_text(agent_result.text),
            )
            records.append(record)
            if record.promoted:
                baseline_state = self.store.promote_baseline(record)
            self.previous_insight = record.insight

        self._write_summary(records)
        return records

    def _run_task_wiring_phase(self) -> IterationRecord:
        before = changed_files(self.repo_root)
        task_prompt = _task_prompt(self.config)
        model_path = _model_path(self.config)
        prompt = build_task_wiring_prompt(
            task_prompt=task_prompt,
            model_path=model_path,
            editable_paths=list(self.config.agent.editable_paths),
            forbidden_paths=list(self.config.agent.forbidden_paths),
        )
        agent_result = run_agent(
            str(self.config.agent.command),
            prompt,
            int(self.config.agent.timeout_seconds),
        )
        after = changed_files(self.repo_root)
        changed = new_changes(before, after)
        patch = diff_for_files(changed, self.repo_root)
        validation = validate_changed_files(
            changed,
            editable_paths=list(self.config.agent.editable_paths),
            forbidden_paths=list(self.config.agent.forbidden_paths),
        )
        reasons = list(validation.reasons)
        if agent_result.returncode != 0:
            reasons.append(agent_result.text or "agent command failed")
        if reasons:
            record = IterationRecord(
                iteration_id=-1,
                status="rejected",
                parent_baseline_id=None,
                changed_files=changed,
                patch=patch,
                one_change_summary="rejected task wiring",
                frozen_config=to_plain_config(self.config),
                checkpoint_path=None,
                metrics={},
                primary_metric=self.primary_metric,
                primary_metric_before=None,
                primary_metric_after=None,
                improved=False,
                promoted=False,
                insight="; ".join(reasons),
                error_message="; ".join(reasons),
            )
            self.store.append(record)
            return record

        try:
            metrics = validate_pretrain_metrics(pretrain_evaluate(self.config))
            record = IterationRecord(
                iteration_id=-1,
                status="wired",
                parent_baseline_id=None,
                changed_files=changed,
                patch=patch,
                one_change_summary=_summarize_agent_text(agent_result.text) or "task wiring",
                frozen_config=to_plain_config(self.config),
                checkpoint_path=None,
                metrics=metrics,
                primary_metric=self.primary_metric,
                primary_metric_before=None,
                primary_metric_after=None,
                improved=False,
                promoted=False,
                insight=f"task wiring verification passed: precision={metrics['precision']}, recall={metrics['recall']}",
                error_message=None,
                epoch_metrics=[],
            )
            self.store.append(record)
            return record
        except Exception as exc:  # noqa: BLE001
            record = IterationRecord(
                iteration_id=-1,
                status="failed",
                parent_baseline_id=None,
                changed_files=changed,
                patch=patch,
                one_change_summary="task wiring verification failed",
                frozen_config=to_plain_config(self.config),
                checkpoint_path=None,
                metrics={},
                primary_metric=self.primary_metric,
                primary_metric_before=None,
                primary_metric_after=None,
                improved=False,
                promoted=False,
                insight="task wiring verification failed",
                error_message=str(exc),
            )
            self.store.append(record)
            return record

    def _run_iteration(
        self,
        iteration_id: int,
        *,
        parent: BaselineState | None,
        changed: list[str],
        patch: str,
        summary: str,
    ) -> IterationRecord:
        iteration_dir = self.store.iteration_dir(iteration_id)
        try:
            result = run_training(self.config, output_dir=iteration_dir)
            primary_after = result.metrics.get(self.primary_metric)
            if primary_after is None and self.primary_metric == "f1":
                primary_after = _derive_f1(result.metrics)
            primary_before = parent.primary_metric_value if parent else None
            improved = primary_after is not None and (parent is None or primary_after > parent.primary_metric_value)
            record = IterationRecord(
                iteration_id=iteration_id,
                status="success",
                parent_baseline_id=parent.iteration_id if parent else None,
                changed_files=changed,
                patch=patch,
                one_change_summary=summary,
                frozen_config=to_plain_config(self.config),
                checkpoint_path=str(result.checkpoint_path),
                metrics=result.metrics,
                primary_metric=self.primary_metric,
                primary_metric_before=primary_before,
                primary_metric_after=primary_after,
                improved=improved,
                promoted=improved,
                insight=f"{summary}: primary {self.primary_metric}={primary_after}",
                epoch_metrics=result.epoch_metrics,
            )
            self.store.append(record)
            return record
        except Exception as exc:  # noqa: BLE001
            record = IterationRecord(
                iteration_id=iteration_id,
                status="failed",
                parent_baseline_id=parent.iteration_id if parent else None,
                changed_files=changed,
                patch=patch,
                one_change_summary=summary,
                frozen_config=to_plain_config(self.config),
                checkpoint_path=None,
                metrics={},
                primary_metric=self.primary_metric,
                primary_metric_before=parent.primary_metric_value if parent else None,
                primary_metric_after=None,
                improved=False,
                promoted=False,
                insight="iteration failed",
                error_message=str(exc),
            )
            self.store.append(record)
            return record

    def _rejected_record(
        self,
        iteration_id: int,
        baseline: BaselineState,
        changed: list[str],
        patch: str,
        reasons: list[str],
    ) -> IterationRecord:
        return IterationRecord(
            iteration_id=iteration_id,
            status="rejected",
            parent_baseline_id=baseline.iteration_id,
            changed_files=changed,
            patch=patch,
            one_change_summary="rejected agent change",
            frozen_config=to_plain_config(self.config),
            checkpoint_path=None,
            metrics={},
            primary_metric=self.primary_metric,
            primary_metric_before=baseline.primary_metric_value,
            primary_metric_after=None,
            improved=False,
            promoted=False,
            insight="; ".join(reasons),
            error_message="; ".join(reasons),
        )

    def _write_summary(self, records: list[IterationRecord]) -> None:
        figure = plot_f1_progress(
            [record.to_dict() for record in records],
            self.store.figures_dir / "f1_progress.png",
        )
        write_json(self.store.output_dir / "summary.json", {"figure": str(figure), "records": len(records)})


def manage_iterations(config: Any, *, repo_root: str | Path = ".") -> list[IterationRecord]:
    """Run the configured iteration manager."""
    return IterationManager(config, repo_root=repo_root).run()


def _summarize_agent_text(text: str) -> str:
    if not text.strip():
        return "agent change"
    first = text.strip().splitlines()[0]
    return first[:160]


def _task_prompt(config: Any) -> str:
    task = config.get("task", {}) if hasattr(config, "get") else getattr(config, "task", {})
    prompt = task.get("prompt", "") if hasattr(task, "get") else getattr(task, "prompt", "")
    return str(prompt or "No task prompt provided.")


def _model_path(config: Any) -> str | None:
    task = config.get("task", {}) if hasattr(config, "get") else getattr(config, "task", {})
    model_path = task.get("model_path", None) if hasattr(task, "get") else getattr(task, "model_path", None)
    return str(model_path) if model_path else None


def _derive_f1(metrics: dict[str, float]) -> float | None:
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
