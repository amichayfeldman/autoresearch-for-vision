"""YOLO-pose training via the autoresearch iteration loop.

Integrates Ultralytics YOLO-pose training with the autoresearch system's
baseline-comparison infrastructure.  Each iteration is a full YOLO training
run (capped at ``--epochs`` or ``--max-time-minutes``).  Only iterations that
improve the primary metric promote their checkpoint as the new baseline.

Bypasses Lightning (YOLO owns its own trainer) and the agent-edits-code loop,
but uses ``HistoryStore``, ``IterationRecord``, ``BaselineState``, and
``plot_f1_progress`` from the autoresearch engine.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pyrootutils

from cv_autoresearch.engine.history import BaselineState, HistoryStore, IterationRecord
from cv_autoresearch.engine.manager.visualization import plot_f1_progress
from cv_autoresearch.engine.utils import write_json


DEFAULT_ULTRALYTICS_REPO = Path("/home/afeldman/projects/ultralytics")
DEFAULT_OUTPUT_DIR = Path("outputs/yolo_pose")
MAX_YOLO_EPOCHS = 5
MIN_YOLO_BATCH = 64
YOLO_WARMUP_EPOCHS = 0


@dataclass(frozen=True)
class YoloPoseRunConfig:
    """Resolved settings for a YOLO-pose autoresearch run."""

    data: Path
    model: str
    ultralytics_repo: Path
    output_dir: Path
    ultralytics_branch: str | None
    allow_dirty_ultralytics: bool
    name: str
    max_iterations: int
    epochs: int
    batch: int
    warmup_epochs: int
    imgsz: int
    device: str
    workers: int
    patience: int
    optimizer: str
    lr0: float
    lrf: float
    seed: int
    deterministic: bool
    plots: bool
    save_period: int
    amp: bool
    half: bool
    primary_metric: str
    skip_pretrain_val: bool


def main() -> None:
    config = parse_args()
    try:
        records = run(config)
    except Exception as exc:
        raise SystemExit(f"YOLO-pose training failed: {exc}") from exc

    promoted = [r for r in records if r.promoted]
    print(f"iterations: {len(records)}, promoted: {len(promoted)}")
    if promoted:
        best = promoted[-1]
        print(f"best {config.primary_metric}: {best.primary_metric_after}")
        print(f"checkpoint: {best.checkpoint_path}")
    print(f"output_dir: {config.output_dir}")


def parse_args() -> YoloPoseRunConfig:
    parser = argparse.ArgumentParser(
        description="Train YOLO-pose via the autoresearch iteration loop.",
    )
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics pose dataset YAML.")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="YOLO pose model name or path.")
    parser.add_argument(
        "--ultralytics-repo",
        type=Path,
        default=DEFAULT_ULTRALYTICS_REPO,
        help="Path to the local Ultralytics repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for autoresearch run artifacts.",
    )
    parser.add_argument(
        "--ultralytics-branch",
        help="Branch name to create in the Ultralytics repo. Defaults to a timestamped run branch.",
    )
    parser.add_argument(
        "--allow-dirty-ultralytics",
        action="store_true",
        help="Allow creating the run branch when the Ultralytics repo has uncommitted changes.",
    )
    parser.add_argument("--name", default="train", help="Run name used by Ultralytics and branch naming.")
    parser.add_argument("--max-iterations", type=int, default=5, help="Number of training iterations.")
    parser.add_argument("--epochs", type=int, default=MAX_YOLO_EPOCHS, help="Epochs per iteration.")
    parser.add_argument("--batch", type=int, default=MIN_YOLO_BATCH)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--half", action="store_true", help="Use half precision for validation where supported.")
    parser.add_argument("--primary-metric", default="f1", help="Metric used to decide baseline promotion.")
    parser.add_argument(
        "--skip-pretrain-val",
        action="store_true",
        help="Skip initial YOLO validation and write zero-valued pretrain metrics.",
    )
    args = parser.parse_args()

    return YoloPoseRunConfig(
        data=args.data,
        model=args.model,
        ultralytics_repo=args.ultralytics_repo,
        output_dir=args.output_dir,
        ultralytics_branch=args.ultralytics_branch,
        allow_dirty_ultralytics=args.allow_dirty_ultralytics,
        name=args.name,
        max_iterations=args.max_iterations,
        epochs=min(args.epochs, MAX_YOLO_EPOCHS),
        batch=max(args.batch, MIN_YOLO_BATCH),
        warmup_epochs=YOLO_WARMUP_EPOCHS,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        seed=args.seed,
        deterministic=not args.no_deterministic,
        plots=not args.no_plots,
        save_period=args.save_period,
        amp=not args.no_amp,
        half=args.half,
        primary_metric=args.primary_metric,
        skip_pretrain_val=args.skip_pretrain_val,
    )


# ---------------------------------------------------------------------------
# Iteration loop — uses autoresearch HistoryStore / BaselineState / Records
# ---------------------------------------------------------------------------


def run(config: YoloPoseRunConfig) -> list[IterationRecord]:
    """Run the autoresearch iteration loop with YOLO-pose training."""
    validate_paths(config)

    branch_name = config.ultralytics_branch or default_branch_name(config.name)
    create_ultralytics_branch(
        config.ultralytics_repo.resolve(),
        branch_name,
        allow_dirty=config.allow_dirty_ultralytics,
    )

    disable_clearml_logging()
    yolo_cls = load_ultralytics_yolo(config.ultralytics_repo.resolve())

    store = HistoryStore(config.output_dir.resolve())
    frozen_config = {
        **serialize_dataclass(config),
        "ultralytics_branch": branch_name,
        "clearml_disabled": True,
    }
    records: list[IterationRecord] = []

    # Pretrain validation (iteration -1 equivalent)
    pretrain_metrics = run_pretrain_validation(config, yolo_cls, store.output_dir)
    write_json(store.iteration_dir(-1) / "resolved_config.json", frozen_config)
    write_json(store.iteration_dir(-1) / "pretrain_metrics.json", pretrain_metrics)
    wiring_record = IterationRecord(
        iteration_id=-1,
        status="wired",
        parent_baseline_id=None,
        changed_files=[],
        patch="",
        one_change_summary="pretrain validation",
        frozen_config=frozen_config,
        checkpoint_path=None,
        metrics=pretrain_metrics,
        primary_metric=config.primary_metric,
        primary_metric_before=None,
        primary_metric_after=pretrain_metrics.get(config.primary_metric),
        improved=False,
        promoted=False,
        insight=f"pretrain: {config.primary_metric}={pretrain_metrics.get(config.primary_metric)}",
        epoch_metrics=[],
    )
    store.append(wiring_record)
    records.append(wiring_record)

    # Iteration loop: train → compare → promote or skip
    baseline: BaselineState | None = None
    for iteration_id in range(config.max_iterations):
        record = _run_iteration(config, yolo_cls, store, frozen_config, iteration_id, baseline)
        records.append(record)
        if record.promoted:
            baseline = store.promote_baseline(record)
            print(f"[iter {iteration_id}] promoted — {config.primary_metric}={record.primary_metric_after}")
        elif record.status == "success":
            print(f"[iter {iteration_id}] not promoted — {config.primary_metric}={record.primary_metric_after}")
        else:
            print(f"[iter {iteration_id}] failed — {record.error_message}")

    _write_summary(store, records)
    return records


def _run_iteration(
    config: YoloPoseRunConfig,
    yolo_cls: Any,
    store: HistoryStore,
    frozen_config: dict[str, Any],
    iteration_id: int,
    baseline: BaselineState | None,
) -> IterationRecord:
    """Train one YOLO iteration and return the record."""
    iteration_dir = store.iteration_dir(iteration_id)
    write_json(iteration_dir / "resolved_config.json", frozen_config)
    try:
        model = yolo_cls(config.model, task="pose")
        train_metrics = model.train(**build_train_kwargs(config, iteration_dir))
        metrics = normalize_pose_metrics(train_metrics)
        checkpoint_path = resolve_checkpoint_path(model)
        # Copy checkpoint into iteration dir for HistoryStore
        stored_ckpt = iteration_dir / "checkpoint.pt"
        shutil.copy2(checkpoint_path, stored_ckpt)

        epoch_metrics = build_epoch_metrics(config.epochs, metrics)
        write_json(iteration_dir / "metrics.json", metrics)
        write_json(iteration_dir / "epoch_metrics.json", epoch_metrics)

        primary_after = metrics.get(config.primary_metric)
        primary_before = baseline.primary_metric_value if baseline else None
        improved = primary_after is not None and (
            baseline is None or primary_after > baseline.primary_metric_value
        )
        record = IterationRecord(
            iteration_id=iteration_id,
            status="success",
            parent_baseline_id=baseline.iteration_id if baseline else None,
            changed_files=[],
            patch="",
            one_change_summary=f"yolo-pose iter {iteration_id}",
            frozen_config=frozen_config,
            checkpoint_path=str(stored_ckpt),
            metrics=metrics,
            primary_metric=config.primary_metric,
            primary_metric_before=primary_before,
            primary_metric_after=primary_after,
            improved=improved,
            promoted=improved,
            insight=f"yolo-pose iter {iteration_id}: {config.primary_metric}={primary_after}",
            epoch_metrics=epoch_metrics,
        )
        store.append(record)
        return record
    except Exception as exc:  # noqa: BLE001
        record = IterationRecord(
            iteration_id=iteration_id,
            status="failed",
            parent_baseline_id=baseline.iteration_id if baseline else None,
            changed_files=[],
            patch="",
            one_change_summary=f"yolo-pose iter {iteration_id}",
            frozen_config=frozen_config,
            checkpoint_path=None,
            metrics={},
            primary_metric=config.primary_metric,
            primary_metric_before=baseline.primary_metric_value if baseline else None,
            primary_metric_after=None,
            improved=False,
            promoted=False,
            insight="iteration failed",
            error_message=str(exc),
        )
        store.append(record)
        return record


def _write_summary(store: HistoryStore, records: list[IterationRecord]) -> None:
    figure = plot_f1_progress(
        [r.to_dict() for r in records],
        store.figures_dir / "f1_progress.png",
    )
    write_json(store.output_dir / "summary.json", {"figure": str(figure), "records": len(records)})


def run_pretrain_validation(
    config: YoloPoseRunConfig,
    yolo_cls: Any,
    output_dir: Path,
) -> dict[str, float]:
    if config.skip_pretrain_val:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    model = yolo_cls(config.model, task="pose")
    metrics_obj = model.val(**build_val_kwargs(config, output_dir))
    return normalize_pose_metrics(metrics_obj)


def validate_paths(config: YoloPoseRunConfig) -> None:
    if not config.ultralytics_repo.exists():
        raise FileNotFoundError(f"Ultralytics repo does not exist: {config.ultralytics_repo}")
    if not (config.ultralytics_repo / "ultralytics").is_dir():
        raise FileNotFoundError(f"Missing ultralytics package in: {config.ultralytics_repo}")
    if not config.data.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {config.data}")


def default_branch_name(run_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_run_name = re.sub(r"[^A-Za-z0-9._-]+", "-", run_name).strip("-").lower() or "run"
    return f"insighture/yolo-pose-{timestamp}-{safe_run_name}"


def create_ultralytics_branch(repo: Path, branch_name: str, *, allow_dirty: bool) -> None:
    if not allow_dirty:
        status = run_git(repo, "status", "--porcelain")
        if status.stdout.strip():
            raise RuntimeError(
                "Ultralytics repo has uncommitted changes. Commit them, clean the repo, "
                "or pass --allow-dirty-ultralytics."
            )
    run_git(repo, "switch", "-c", branch_name)


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        command = " ".join(["git", *args])
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{command} failed in {repo}: {message}")
    return result


def disable_clearml_logging() -> None:
    os.environ.setdefault("CLEARML_OFFLINE_MODE", "1")
    os.environ.setdefault("CLEARML_NO_DEFAULT_SERVER", "1")
    os.environ.setdefault("TRAINS_OFFLINE_MODE", "1")


def load_ultralytics_yolo(repo: Path) -> Any:
    pyrootutils.setup_root(
        search_from=repo / "ultralytics",
        indicator=".git",
        pythonpath=True,
    )

    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    SETTINGS["clearml"] = False
    return YOLO


def build_train_kwargs(config: YoloPoseRunConfig, output_dir: Path) -> dict[str, Any]:
    return {
        **shared_yolo_kwargs(config),
        "epochs": config.epochs,
        "warmup_epochs": config.warmup_epochs,
        "project": str(output_dir / "ultralytics"),
        "name": config.name,
        "exist_ok": True,
        "patience": config.patience,
        "optimizer": config.optimizer,
        "lr0": config.lr0,
        "lrf": config.lrf,
        "plots": config.plots,
        "save": True,
        "save_period": config.save_period,
        "do_export": False,
    }


def build_val_kwargs(config: YoloPoseRunConfig, output_dir: Path) -> dict[str, Any]:
    return {
        **shared_yolo_kwargs(config),
        "project": str(output_dir / "pretrain"),
        "name": "val",
        "exist_ok": True,
        "plots": False,
    }


def shared_yolo_kwargs(config: YoloPoseRunConfig) -> dict[str, Any]:
    return {
        "data": str(config.data.resolve()),
        "batch": config.batch,
        "imgsz": config.imgsz,
        "device": config.device,
        "workers": config.workers,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "amp": config.amp,
        "half": config.half,
        "task": "pose",
    }


def normalize_pose_metrics(metrics_obj: Any) -> dict[str, float]:
    raw_metrics = extract_results_dict(metrics_obj)
    pose_precision = as_float(raw_metrics.get("metrics/precision(P)", 0.0))
    pose_recall = as_float(raw_metrics.get("metrics/recall(P)", 0.0))
    f1 = (
        2 * pose_precision * pose_recall / (pose_precision + pose_recall)
        if pose_precision + pose_recall
        else 0.0
    )

    return {
        "precision": pose_precision,
        "recall": pose_recall,
        "f1": float(f1),
        "box_precision": as_float(raw_metrics.get("metrics/precision(B)", 0.0)),
        "box_recall": as_float(raw_metrics.get("metrics/recall(B)", 0.0)),
        "box_map50": as_float(raw_metrics.get("metrics/mAP50(B)", 0.0)),
        "box_map50_95": as_float(raw_metrics.get("metrics/mAP50-95(B)", 0.0)),
        "pose_map50": as_float(raw_metrics.get("metrics/mAP50(P)", 0.0)),
        "pose_map50_95": as_float(raw_metrics.get("metrics/mAP50-95(P)", 0.0)),
        "fitness": as_float(raw_metrics.get("fitness", 0.0)),
    }


def extract_results_dict(metrics_obj: Any) -> dict[str, Any]:
    if metrics_obj is None:
        return {}
    results = getattr(metrics_obj, "results_dict", None)
    if results is None:
        if isinstance(metrics_obj, dict):
            return metrics_obj
        return {}
    return dict(results)


def resolve_checkpoint_path(model: Any) -> Path:
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("Ultralytics model did not expose a trainer after training.")

    best = Path(getattr(trainer, "best", ""))
    if best.exists():
        return best.resolve()

    last = Path(getattr(trainer, "last", ""))
    if last.exists():
        return last.resolve()

    raise FileNotFoundError("No best.pt or last.pt checkpoint was found after training.")


def build_epoch_metrics(max_epochs: int, final_metrics: dict[str, float]) -> list[dict[str, float]]:
    return [{"epoch": epoch, "f1": float(final_metrics["f1"])} for epoch in range(1, max_epochs + 1)]


def as_float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def serialize_dataclass(config: YoloPoseRunConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


if __name__ == "__main__":
    main()
