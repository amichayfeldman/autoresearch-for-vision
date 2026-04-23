"""Tests for the Hydra-driven training iteration engine."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from cv_autoresearch.engine.history import HistoryStore, IterationRecord
from cv_autoresearch.engine.manager.boundaries import (
    detect_multiple_unrelated_changes,
    validate_changed_files,
)
from cv_autoresearch.engine.manager.iterations import IterationManager
from cv_autoresearch.engine.manager.visualization import plot_f1_progress
from cv_autoresearch.engine.scripts.prompt_args import apply_prompt_args, consume_prompt_args
from cv_autoresearch.engine.training.runner import validate_pretrain_metrics


def test_hydra_configs_load_for_examples() -> None:
    with initialize(version_base=None, config_path="../configs"):
        names = [
            "config",
            "prompt_task",
            "examples/prompt_task",
        ]
        configs = [compose(config_name=name) for name in names]

    assert [cfg.evaluation.primary_metric for cfg in configs] == [
        "f1",
        "f1",
        "f1",
    ]


def test_boundary_validation_rejects_forbidden_and_uneditable_paths() -> None:
    result = validate_changed_files(
        ["src/cv_autoresearch/engine/manager/iterations.py"],
        editable_paths=["configs/model/", "src/cv_autoresearch/engine/evaluation/"],
        forbidden_paths=["src/cv_autoresearch/engine/manager/"],
    )

    assert result.valid is False
    assert any("forbidden" in reason for reason in result.reasons)
    assert detect_multiple_unrelated_changes(["configs/model/a.yaml", "configs/data/b.yaml"])


def test_history_store_records_and_promotes_baseline(tmp_path: Path) -> None:
    checkpoint = tmp_path / "iteration-checkpoint.pt"
    checkpoint.write_text("checkpoint")
    store = HistoryStore(tmp_path / "run")
    iteration_dir = store.iteration_dir(0)
    (iteration_dir / "resolved_config.json").write_text("{}\n")
    record = IterationRecord(
        iteration_id=0,
        status="success",
        parent_baseline_id=None,
        changed_files=[],
        patch="",
        one_change_summary="baseline",
        frozen_config={},
        checkpoint_path=str(checkpoint),
        metrics={"f1": 0.5},
        primary_metric="f1",
        primary_metric_before=None,
        primary_metric_after=0.5,
        improved=True,
        promoted=True,
        insight="baseline",
        epoch_metrics=[{"epoch": 1, "f1": 0.5}],
    )

    store.append(record)
    baseline = store.promote_baseline(record)

    assert store.jsonl_path.exists()
    assert baseline.primary_metric_value == 0.5
    assert Path(baseline.checkpoint_path).read_text() == "checkpoint"


def test_iteration_manager_promotes_only_improvements(monkeypatch, tmp_path: Path) -> None:
    config = SimpleNamespace(
        history=SimpleNamespace(output_dir=str(tmp_path / "run")),
        evaluation=SimpleNamespace(primary_metric="f1"),
        task=SimpleNamespace(prompt="Classify test images", model_path=None),
        iteration=SimpleNamespace(max_iterations=1),
        agent=SimpleNamespace(
            command="",
            timeout_seconds=1,
            editable_paths=["configs/model/", "src/cv_autoresearch/engine/task_wiring/"],
            forbidden_paths=["src/cv_autoresearch/engine/manager/"],
        ),
    )
    calls = iter([0.4, 0.3])

    def fake_run_training(cfg, output_dir):
        metric = next(calls)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = out / "checkpoint.pt"
        checkpoint.write_text(str(metric))
        (out / "resolved_config.json").write_text("{}\n")
        return SimpleNamespace(
            checkpoint_path=checkpoint,
            metrics={"f1": metric},
            epoch_metrics=[{"epoch": 1, "f1": metric}],
        )

    monkeypatch.setattr("cv_autoresearch.engine.manager.iterations.run_training", fake_run_training)
    monkeypatch.setattr(
        "cv_autoresearch.engine.manager.iterations.pretrain_evaluate",
        lambda cfg: {"precision": 0.5, "recall": 0.5},
    )
    monkeypatch.setattr("cv_autoresearch.engine.manager.iterations.changed_files", lambda repo: [])
    monkeypatch.setattr("cv_autoresearch.engine.manager.iterations.diff_for_files", lambda files, repo: "")

    records = IterationManager(config).run()

    assert [record.status for record in records] == ["wired", "success", "success"]
    assert [record.promoted for record in records] == [False, True, False]
    assert records[0].primary_metric_after is None
    assert records[0].insight.startswith("task wiring verification passed")
    state = json.loads((tmp_path / "run" / "baseline" / "state.json").read_text())
    assert state["iteration_id"] == 0


def test_prompt_args_accept_inline_file_and_model_path(tmp_path: Path) -> None:
    prompt_file = tmp_path / "task.md"
    prompt_file.write_text("Detect missing screws.")
    argv = [
        "manage_iterations.py",
        "--task-prompt",
        "Classify defects.",
        "--task-prompt-file",
        str(prompt_file),
        "--model-path",
        "model.pt",
        "iteration.max_epochs=1",
    ]

    args = consume_prompt_args(argv)
    config = OmegaConf.create({"task": {}})
    apply_prompt_args(config, args)

    assert argv == ["manage_iterations.py", "iteration.max_epochs=1"]
    assert config.task.prompt == "Detect missing screws."
    assert config.task.prompt_file == str(prompt_file)
    assert config.task.model_path == "model.pt"


@pytest.mark.parametrize("metrics", [{}, {"precision": 0.5}, {"recall": 0.5}])
def test_pretrain_gate_requires_precision_and_recall(metrics: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        validate_pretrain_metrics(metrics)


def test_pretrain_gate_derives_f1() -> None:
    metrics = validate_pretrain_metrics({"precision": 0.25, "recall": 0.5})

    assert metrics["f1"] == pytest.approx(1 / 3)


def test_visualization_writes_png(tmp_path: Path) -> None:
    output = plot_f1_progress(
        [
            {
                "epoch_metrics": [{"epoch": 1, "f1": 0.2}],
                "promoted": True,
                "metrics": {"f1": 0.2},
                "primary_metric_after": 0.2,
                "one_change_summary": "baseline",
            }
        ],
        tmp_path / "f1_progress.png",
    )

    assert output.exists()
    assert output.read_bytes().startswith(b"\x89PNG")


def test_project_skills_contain_boundaries() -> None:
    iteration_text = Path("agents/skills/cv-training-iteration/SKILL.md").read_text()
    wiring_text = Path("agents/skills/cv-task-wiring/SKILL.md").read_text()

    assert "Allowed Paths" in iteration_text
    assert "Forbidden Paths" in iteration_text
    assert "src/cv_autoresearch/engine/evaluation/" in iteration_text
    assert "Allowed Paths" in wiring_text
    assert "Forbidden Paths" in wiring_text
    assert "src/cv_autoresearch/engine/evaluation/" in wiring_text
