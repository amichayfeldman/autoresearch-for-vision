"""PyTorch Lightning DataModule for training iterations."""

from __future__ import annotations

from typing import Any

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from cv_autoresearch.engine.data.synthetic import SyntheticPromptDataset
from cv_autoresearch.engine.data.transforms import TransformDataset, build_train_transform


class VisionDataModule(LightningDataModule):
    """Own datasets, transforms, batching, and dataloaders."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None) -> None:
        data_cfg = self.config.data
        train = SyntheticPromptDataset(
            size=int(data_cfg.get("train_size", 32)),
            num_classes=int(self.config.evaluation.get("num_classes", 2)),
            image_size=int(data_cfg.get("image_size", 16)),
            channels=int(data_cfg.get("channels", 3)),
        )
        val = SyntheticPromptDataset(
            size=int(data_cfg.get("val_size", 16)),
            num_classes=int(self.config.evaluation.get("num_classes", 2)),
            image_size=int(data_cfg.get("image_size", 16)),
            channels=int(data_cfg.get("channels", 3)),
        )
        self.train_dataset = TransformDataset(train, build_train_transform(self.config.get("augmentations")))
        self.val_dataset = val

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.config.data.batch_size),
            num_workers=int(self.config.data.get("num_workers", 0)),
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("validate")
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.config.data.batch_size),
            num_workers=int(self.config.data.get("num_workers", 0)),
            shuffle=False,
        )
