"""AutoResearchModule: wraps plain nn.Module + TaskDef, injects hyperparams."""

from __future__ import annotations

from typing import Any

import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    StepLR,
)

from cv_autoresearch.task import TaskDef


class AutoResearchModule(LightningModule):
    """Wraps a plain nn.Module and TaskDef, injecting hyperparams from trial.

    The user provides a plain nn.Module (not a LightningModule).
    loss_fn is owned by the task, not the model — keeping architecture
    decoupled from training objective.
    Validation is handled externally in autoresearch.py — no validation_step here.
    """

    def __init__(self, task: TaskDef, hyperparams: dict[str, Any]) -> None:
        """Initialize the wrapper module.

        Args:
            task: TaskDef with model and loss_fn.
            hyperparams: Trial hyperparams from Optuna (learning_rate, optimizer_type, etc.).
        """
        super().__init__()
        self.save_hyperparameters(hyperparams)
        self.model = task.model
        self.loss_fn = task.loss_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the user's model.

        Args:
            x: Input tensor.

        Returns:
            Model output tensor.
        """
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Compute loss using the task-owned loss_fn.

        Args:
            batch: Training batch.
            batch_idx: Batch index.

        Returns:
            Loss tensor.
        """
        logits = self.model(batch[0])
        return self.loss_fn(logits, batch)

    def configure_optimizers(self) -> dict[str, Any]:
        """Build optimizer and LR scheduler from trial hyperparams.

        Returns:
            PL-compatible dict with 'optimizer' and 'lr_scheduler' keys.
        """
        hp = dict(self.hparams)
        optimizer = _build_optimizer(self.model.parameters(), hp)
        scheduler = _build_scheduler(optimizer, hp)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}


def _build_optimizer(params: Any, hp: dict[str, Any]) -> optim.Optimizer:
    """Build optimizer from hyperparams dict.

    Args:
        params: Model parameters to optimize.
        hp: Hyperparams dict with optimizer_type, learning_rate, etc.

    Returns:
        Configured optimizer.
    """
    opt_type = hp.get("optimizer_type", "adam").lower()
    lr = hp.get("learning_rate", 1e-3)
    wd = hp.get("weight_decay", 1e-4)

    if opt_type == "sgd":
        momentum = hp.get("momentum", 0.9)
        return optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
    elif opt_type == "adamw":
        beta1 = hp.get("beta1", 0.9)
        beta2 = hp.get("beta2", 0.999)
        return optim.AdamW(params, lr=lr, weight_decay=wd, betas=(beta1, beta2))
    else:  # adam (default)
        beta1 = hp.get("beta1", 0.9)
        beta2 = hp.get("beta2", 0.999)
        return optim.Adam(params, lr=lr, weight_decay=wd, betas=(beta1, beta2))


def _build_scheduler(optimizer: optim.Optimizer, hp: dict[str, Any]) -> Any:
    """Build LR scheduler from hyperparams dict.

    Args:
        optimizer: Optimizer to schedule.
        hp: Hyperparams dict with lr_scheduler, lr_scheduler_step_size, etc.

    Returns:
        Configured LR scheduler.
    """
    sched_type = hp.get("lr_scheduler", "cosine").lower()
    step_size = hp.get("lr_scheduler_step_size", 10)
    gamma = hp.get("lr_scheduler_gamma", 0.1)

    if sched_type == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_type == "onecycle":
        total_steps = hp.get("total_steps", 100)
        return OneCycleLR(optimizer, max_lr=hp.get("learning_rate", 1e-3), total_steps=total_steps)
    elif sched_type == "cosine_with_restarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0=step_size)
    else:  # cosine (default)
        return CosineAnnealingLR(optimizer, T_max=step_size)
