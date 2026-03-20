"""AutoResearchDataModule: injects Albumentations into training DataLoader."""

from __future__ import annotations

from typing import Any

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class AugmentedDataset(Dataset):
    """Applies Albumentations transform to each image from a dataset.

    Wraps an existing dataset and applies the given Albumentations
    transform to each image. Labels are passed through unchanged.
    """

    def __init__(self, dataset: Dataset, transform: Any) -> None:
        """Initialize the augmented dataset.

        Args:
            dataset: Base dataset returning (image, label) pairs.
            transform: Albumentations Compose transform to apply.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """Apply augmentation to image at index.

        Args:
            idx: Dataset index.

        Returns:
            Tuple of (augmented_image, label).
        """
        image, label = self.dataset[idx]
        augmented = self.transform(image=np.asarray(image))["image"]
        return augmented, label


class AutoResearchDataModule(LightningDataModule):
    """Injects Albumentations pipeline into the training DataLoader.

    The validation DataLoader is returned without augmentation to ensure
    evaluation uses the unmodified images.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        augmentation_pipeline: Any,
        batch_size: int,
        num_workers: int,
    ) -> None:
        """Initialize the data module.

        Args:
            train_dataset: Training dataset (images + labels).
            val_dataset: Validation dataset (no augmentation applied).
            augmentation_pipeline: Albumentations Compose transform for training.
            batch_size: DataLoader batch size.
            num_workers: DataLoader worker count.
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.augmentation_pipeline = augmentation_pipeline
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        """Return augmented training DataLoader."""
        augmented = AugmentedDataset(self.train_dataset, self.augmentation_pipeline)
        return DataLoader(
            augmented,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return raw validation DataLoader (no augmentation)."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
