"""Thin PyTorch Lightning training module."""

from __future__ import annotations

from typing import Any

import torch
from pytorch_lightning import LightningModule

from cv_autoresearch.engine.losses import build_loss
from cv_autoresearch.engine.models import build_model


class VisionLightningModule(LightningModule):
    """Own model forward, loss, optimizer, scheduler, and logging."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config_ref = config
        self.model = build_model(config)
        self.loss_fn = build_loss(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        logits = self(inputs)
        loss = self._loss(logits, targets)
        self.log("train_loss", loss, prog_bar=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
        inputs, targets = batch
        logits = self(inputs)
        loss = self._loss(logits, targets)
        self.log("val_loss", loss, prog_bar=False, on_epoch=True)
        return {"logits": logits.detach().cpu(), "targets": targets.detach().cpu(), "val_loss": loss.detach()}

    def configure_optimizers(self):
        opt_cfg = self.config_ref.optimizer
        lr = float(opt_cfg.learning_rate)
        wd = float(opt_cfg.get("weight_decay", 0.0))
        name = str(opt_cfg.get("name", "adamw")).lower()
        if name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                weight_decay=wd,
                momentum=float(opt_cfg.get("momentum", 0.9)),
            )
        elif name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

        sched_cfg = self.config_ref.scheduler
        sched_name = str(sched_cfg.get("name", "none")).lower()
        if sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(sched_cfg.get("step_size", 5)),
                gamma=float(sched_cfg.get("gamma", 0.1)),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(self.config_ref.iteration.max_epochs)),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        return optimizer

    def predict_batch(self, batch: Any) -> tuple[list[int], list[int]]:
        inputs, targets = batch
        with torch.inference_mode():
            logits = self(inputs.to(self.device)).detach().cpu()
        preds = torch.argmax(logits, dim=1)
        return preds.tolist(), targets.long().tolist()

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets.long())
