"""Evaluator: val inference + dynamic metric computation via Hydra."""

from __future__ import annotations

import torch
import torchmetrics
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset


def instantiate_metric(metric_cfg: DictConfig) -> torchmetrics.Metric:
    """Instantiate a torchmetrics Metric from a Hydra config.

    Args:
        metric_cfg: OmegaConf config with _target_ and constructor args.

    Returns:
        Instantiated torchmetrics.Metric.
    """
    import hydra

    return hydra.utils.instantiate(metric_cfg)


def run_val_inference(
    model: LightningModule,
    val_dataset: Dataset,
    batch_size: int,
    device: str,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model.forward() on full validation set.

    Sets model to eval mode and disables gradient computation.
    Returns all predictions and targets concatenated as tensors.

    Args:
        model: Model with forward() method (LightningModule or nn.Module).
        val_dataset: Validation dataset returning (image, label) pairs.
        batch_size: Inference batch size.
        device: Compute device string (e.g. "cpu", "cuda").
        num_workers: DataLoader worker count.

    Returns:
        Tuple of (all_predictions, all_targets) tensors on CPU.
    """
    loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    model.eval()
    model_device = torch.device(device)

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(model_device)
            preds = model(images)
            all_preds.append(preds.cpu())
            all_targets.append(
                labels.cpu() if isinstance(labels, torch.Tensor) else torch.tensor(labels)
            )

    return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)


def compute_primary_metric(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metric: torchmetrics.Metric,
) -> float:
    """Compute a single metric value over predictions and targets.

    Resets metric before computing to avoid state leakage between calls.

    Args:
        predictions: Model output tensor.
        targets: Ground truth tensor.
        metric: Instantiated torchmetrics.Metric (reset before use).

    Returns:
        Scalar metric value as Python float.
    """
    metric.reset()
    metric.update(predictions, targets)
    return metric.compute().item()
