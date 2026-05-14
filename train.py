"""One-file CV autoresearch program.

The human edits ``program.md``. The coding agent edits this file. The default
task is intentionally tiny, but the training stack is the real CV stack this
repo expects agents to adapt: imported/preconfigured models, PyTorch Lightning
for training, and Albumentations for augmentation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


# Agents: after the task is wired, ordinary research iterations should change
# one clear lever in this block or in the hook functions below.
RUN_NAME = "baseline"
SEED = 123
OUTPUT_DIR = Path("outputs") / RUN_NAME
IMAGE_SIZE = 32
TRAIN_SIZE = 96
VAL_SIZE = 48
BATCH_SIZE = 16
MAX_EPOCHS = 3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
THRESHOLD = 0.5
AUGMENT_HFLIP = False
MODEL_SOURCE = "local_tiny_cnn"
MODEL_CHECKPOINT: str | None = None
EXTERNAL_REPO_PATH: str | None = None
MODEL_IMPORT: str | None = None  # Optional "module.submodule:factory" from a package or repo.
TRAINING_BACKEND = "lightning"  # Use "external" when another repo owns the train loop.
EXTERNAL_TRAIN_IMPORT: str | None = None  # Optional "module.submodule:train_fn".
EXTERNAL_PREDICT_IMPORT: str | None = None  # Optional "module.submodule:predict_fn".


@dataclass
class RunResult:
    """Stable summary returned by `run()` and printed by the CLI."""

    run_name: str
    output_dir: str
    precision: float
    recall: float
    f1: float
    accuracy: float
    loss: float
    checkpoint: str


def missing_runtime_packages() -> list[str]:
    """Return optional runtime packages that are needed to execute training."""
    packages = {
        "torch": "torch",
        "albumentations": "albumentations",
        "numpy": "numpy",
    }
    if TRAINING_BACKEND == "lightning":
        packages["lightning"] = "lightning"
    return [name for name, module in packages.items() if importlib.util.find_spec(module) is None]


def require_runtime() -> None:
    """Fail before setup work begins if the CV training stack is unavailable."""

    missing = missing_runtime_packages()
    if missing:
        raise RuntimeError(
            "Missing CV runtime package(s): "
            + ", ".join(missing)
            + ". Install this project with `pip install -e .` before running train.py."
        )


def make_dataset(size: int, *, image_size: int, seed: int):
    """Create a tiny deterministic CV task: classify a bright square in noise.

    Replace this during Phase 1 when wiring the real task. Keep the spirit of the
    contract: return examples whose labels can be mapped to task metrics without
    parsing logs or display strings.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    data = []
    for idx in range(size):
        label = idx % 2
        image = rng.random((image_size, image_size, 3), dtype=np.float32) * 0.25
        if label:
            start = image_size // 3
            stop = start + image_size // 3
            image[start:stop, start:stop, :] += 0.75
        data.append((image.clip(0.0, 1.0), label))
    random.Random(seed).shuffle(data)
    return data


def prepare_external_repo() -> None:
    """Make a connected model/training repo importable when one is provided.

    Set `EXTERNAL_REPO_PATH` to the local checkout of the source repo when the
    model or training API is not already installed as an importable Python
    package. Keep all adaptation in this file; do not copy the external repo
    into this one.
    """

    if not EXTERNAL_REPO_PATH:
        return
    repo_path = str(Path(EXTERNAL_REPO_PATH).expanduser().resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def import_symbol(spec: str):
    """Import a `module:attribute` symbol from an installed package or repo."""

    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(f"expected import spec like 'package.module:symbol', got {spec!r}")
    module = __import__(module_name, fromlist=[attribute_name])
    return getattr(module, attribute_name)


def build_train_augmentation():
    """Albumentations augmentation hook for train-time images."""
    import albumentations as A

    transforms = []
    if AUGMENT_HFLIP:
        transforms.append(A.HorizontalFlip(p=0.5))
    transforms.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    return A.Compose(transforms)


def build_eval_augmentation():
    """Albumentations preprocessing hook for eval-time images."""
    import albumentations as A

    return A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


def build_model():
    """Build or import the task model.

    CV autoresearch usually starts from a preconfigured model, imported library
    model, or checkpoint. During Phase 1, replace this default with the user's
    actual model source, for example torchvision, timm, ultralytics, segment
    anything, transformers, an installed package factory, an external repo
    factory, or a local checkpoint wrapper.
    """
    import torch

    prepare_external_repo()
    if MODEL_IMPORT:
        factory = import_symbol(MODEL_IMPORT)
        model = factory()
        return load_checkpoint_if_configured(model)

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 1),
    )
    return load_checkpoint_if_configured(model)


def load_checkpoint_if_configured(model):
    """Load the configured checkpoint when the model exposes a PyTorch state."""

    import torch

    if MODEL_CHECKPOINT:
        state = torch.load(MODEL_CHECKPOINT, map_location="cpu")
        model.load_state_dict(state)
    return model


def make_dataloader(data, *, train: bool):
    """Wrap examples with Albumentations and a PyTorch DataLoader.

    Real task wiring should keep image decoding, task-specific preprocessing, and
    target formatting here so the Lightning module receives tensors with a clear
    contract.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    augmentation = build_train_augmentation() if train else build_eval_augmentation()

    class VisionDataset(Dataset):
        def __len__(self) -> int:
            return len(data)

        def __getitem__(self, index: int):
            image, label = data[index]
            augmented = augmentation(image=image)["image"]
            augmented = np.ascontiguousarray(augmented)
            tensor = torch.from_numpy(augmented.transpose(2, 0, 1)).float()
            return tensor, torch.tensor([float(label)], dtype=torch.float32)

    return DataLoader(VisionDataset(), batch_size=BATCH_SIZE, shuffle=train)


def make_lightning_module(model):
    """Wrap the imported/preconfigured model in the Lightning training contract.

    For real tasks, adapt `forward()`, `training_step()`, and the loss here to
    match the model output shape and target structure. Keep optimizer and trainer
    behavior in Lightning rather than adding a custom training loop.
    """

    import lightning.pytorch as pl
    import torch

    class VisionModule(pl.LightningModule):
        """Minimal binary-classification LightningModule for the fallback task."""

        def __init__(self, network):
            super().__init__()
            self.network = network
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

        def forward(self, images):
            return self.network(images).view(-1)

        def training_step(self, batch, batch_idx):
            images, labels = batch
            logits = self(images)
            loss = self.loss_fn(logits, labels.view(-1))
            self.log("train_loss", loss, prog_bar=False)
            return loss

        def configure_optimizers(self):
            return torch.optim.AdamW(
                self.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )

    return VisionModule(model)


def collect_predictions(lightning_module, dataloader) -> tuple[list[float], list[int], float]:
    """Run inference and return structured values for metric calculation.

    This boundary matters: metrics should consume predictions and labels, not
    scrape trainer logs or printed text.
    """

    lightning_module.eval()
    probs: list[float] = []
    labels_out: list[int] = []
    losses: list[float] = []
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    with torch.no_grad():
        for images, labels in dataloader:
            logits = lightning_module(images)
            labels_flat = labels.view(-1)
            batch_losses = loss_fn(logits, labels_flat)
            losses.extend(float(value) for value in batch_losses)
            probs.extend(float(value) for value in torch.sigmoid(logits))
            labels_out.extend(int(value) for value in labels_flat)
    mean_loss = sum(losses) / max(1, len(losses))
    return probs, labels_out, mean_loss


def collect_external_predictions(model, dataloader) -> tuple[list[float], list[int], float]:
    """Run prediction through an imported package or external repo adapter.

    Wire `EXTERNAL_PREDICT_IMPORT` to a function when the model source has its
    own inference API. The function should return `(probs, labels, loss)` or
    structured values that this adapter converts before metric calculation.
    """

    if not EXTERNAL_PREDICT_IMPORT:
        raise RuntimeError(
            "TRAINING_BACKEND='external' requires EXTERNAL_PREDICT_IMPORT to map "
            "external model outputs into probabilities, labels, and loss."
        )
    prepare_external_repo()
    predict_fn = import_symbol(EXTERNAL_PREDICT_IMPORT)
    probs, labels, loss = predict_fn(model=model, dataloader=dataloader, threshold=THRESHOLD)
    return list(probs), [int(label) for label in labels], float(loss)


def metrics_from_predictions(probs: list[float], labels: list[int], loss: float) -> dict[str, float]:
    """Convert model outputs into precision, recall, and companion metrics.

    During Phase 1, replace this binary threshold bridge with the task's real
    matching rule, such as class equality, IoU matching, mask overlap, keypoint
    distance, or text normalization.
    """

    tp = fp = tn = fn = 0
    for prob, label in zip(probs, labels):
        pred = int(prob >= THRESHOLD)
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "loss": loss,
    }
    require_numeric(metrics, "precision", "recall")
    return metrics


def evaluate(lightning_module, dataloader) -> dict[str, float]:
    """Evaluate the current model using the explicit prediction-to-metric bridge."""

    probs, labels, loss = collect_predictions(lightning_module, dataloader)
    return metrics_from_predictions(probs, labels, loss)


def evaluate_external(model, dataloader) -> dict[str, float]:
    """Evaluate a connected repo model through the explicit adapter contract."""

    probs, labels, loss = collect_external_predictions(model, dataloader)
    return metrics_from_predictions(probs, labels, loss)


def fit_with_lightning(model, train_loader, val_loader):
    """Train a model through the local LightningModule adapter."""

    import lightning.pytorch as pl

    pl.seed_everything(SEED, workers=True)
    module = make_lightning_module(model)
    # Phase 1 gate: metrics must be numeric before training changes can matter.
    pretrain_metrics = evaluate(module, val_loader)

    trainer = pl.Trainer(
        accelerator="cpu",
        deterministic=True,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        max_epochs=MAX_EPOCHS,
    )
    trainer.fit(module, train_loader)
    return module, pretrain_metrics, trainer


def fit_with_external_source(model, train_loader, val_loader):
    """Delegate training to an imported source while preserving repo metrics.

    Use this when the installed package or connected repo already owns the
    meaningful training entrypoint, such as `model.train(...)`, `trainer.fit(...)`,
    or a project-specific `train_fn(...)`. The external function may use, bypass,
    or wrap Lightning; the only hard requirement is that evaluation still writes
    numeric precision and recall through this file's metric bridge.
    """

    if not EXTERNAL_TRAIN_IMPORT:
        raise RuntimeError("TRAINING_BACKEND='external' requires EXTERNAL_TRAIN_IMPORT.")
    prepare_external_repo()
    pretrain_metrics = evaluate_external(model, val_loader)
    train_fn = import_symbol(EXTERNAL_TRAIN_IMPORT)
    trained = train_fn(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        output_dir=OUTPUT_DIR,
        max_epochs=MAX_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    return trained if trained is not None else model, pretrain_metrics


def save_model_artifacts(model_or_module, trainer=None) -> Path:
    """Save the best available checkpoint format for the selected backend."""

    import torch

    checkpoint = OUTPUT_DIR / (
        "checkpoint.ckpt" if TRAINING_BACKEND == "lightning" else "external_checkpoint.pt"
    )
    if trainer is not None:
        trainer.save_checkpoint(checkpoint)
    elif hasattr(model_or_module, "state_dict"):
        torch.save(model_or_module.state_dict(), checkpoint)
    else:
        write_json(checkpoint.with_suffix(".json"), {"checkpoint": "managed by external repo"})
        return checkpoint.with_suffix(".json")

    network = getattr(model_or_module, "network", model_or_module)
    if hasattr(network, "state_dict"):
        torch.save(network.state_dict(), OUTPUT_DIR / "model_state.pt")
    return checkpoint


def run() -> RunResult:
    """Execute setup verification, training, evaluation, and artifact logging."""

    require_runtime()

    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_data = make_dataset(TRAIN_SIZE, image_size=IMAGE_SIZE, seed=SEED)
    val_data = make_dataset(VAL_SIZE, image_size=IMAGE_SIZE, seed=SEED + 1)
    train_loader = make_dataloader(train_data, train=True)
    val_loader = make_dataloader(val_data, train=False)

    model = build_model()
    if TRAINING_BACKEND == "lightning":
        trained_model, pretrain_metrics, trainer = fit_with_lightning(model, train_loader, val_loader)
        metrics = evaluate(trained_model, val_loader)
        checkpoint = save_model_artifacts(trained_model, trainer)
    elif TRAINING_BACKEND == "external":
        trained_model, pretrain_metrics = fit_with_external_source(model, train_loader, val_loader)
        metrics = evaluate_external(trained_model, val_loader)
        checkpoint = save_model_artifacts(trained_model)
    else:
        raise ValueError(f"unknown TRAINING_BACKEND: {TRAINING_BACKEND!r}")

    write_json(OUTPUT_DIR / "pretrain_metrics.json", pretrain_metrics)
    write_json(OUTPUT_DIR / "metrics.json", metrics)
    write_json(
        OUTPUT_DIR / "run.json",
        {
            "run_name": RUN_NAME,
            "timestamp": int(time.time()),
            "knobs": experiment_knobs(),
            "metrics": metrics,
            "pretrain_metrics": pretrain_metrics,
            "checkpoint": str(checkpoint),
        },
    )
    return RunResult(
        run_name=RUN_NAME,
        output_dir=str(OUTPUT_DIR),
        checkpoint=str(checkpoint),
        **metrics,
    )


def experiment_knobs() -> dict[str, float | int | bool | str | None]:
    """Return the experiment-facing settings recorded with each run."""

    return {
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "train_size": TRAIN_SIZE,
        "val_size": VAL_SIZE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "threshold": THRESHOLD,
        "augment_hflip": AUGMENT_HFLIP,
        "model_source": MODEL_SOURCE,
        "model_checkpoint": MODEL_CHECKPOINT,
        "external_repo_path": EXTERNAL_REPO_PATH,
        "model_import": MODEL_IMPORT,
        "training_backend": TRAINING_BACKEND,
        "external_train_import": EXTERNAL_TRAIN_IMPORT,
        "external_predict_import": EXTERNAL_PREDICT_IMPORT,
    }


def require_numeric(metrics: dict[str, float], *names: str) -> None:
    """Enforce the program.md acceptance rule for required metric values."""

    missing = [name for name in names if name not in metrics]
    if missing:
        raise ValueError(f"missing required metric(s): {', '.join(missing)}")
    bad = [name for name in names if not math.isfinite(float(metrics[name]))]
    if bad:
        raise ValueError(f"non-finite required metric(s): {', '.join(bad)}")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print machine-readable result JSON.")
    return parser.parse_args()


def main(print_fn: Callable[[str], None] = print) -> None:
    args = parse_args()
    result = run()
    payload = asdict(result)
    if args.json:
        print_fn(json.dumps(payload, sort_keys=True))
    else:
        print_fn(
            "precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
            "accuracy={accuracy:.4f} loss={loss:.4f}".format(**payload)
        )
        print_fn(f"output: {result.output_dir}")
        print_fn(f"checkpoint: {result.checkpoint}")


if __name__ == "__main__":
    main()
