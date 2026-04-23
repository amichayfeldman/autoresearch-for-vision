"""
cv-autoresearch example: CIFAR-10 classification.

Demonstrates the full cv-autoresearch pipeline on CIFAR-10:
1. Define a LightningModule (ResNet-18 backbone)
2. Load datasets (augmentation injected automatically by the system)
3. Configure and run autoresearch

Run:
    python examples/cifar10_classification.py
"""

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import pytorch_lightning as pl
from torch.utils.data import Dataset

from cv_autoresearch import run_autoresearch, SearchConfig


# --- 1. Define your model as a LightningModule ---


class CIFAR10Classifier(pl.LightningModule):
    """ResNet-18-based CIFAR-10 classifier.

    Provides training_step and forward. configure_optimizers is intentionally
    left as a no-op — AutoResearchModule overrides it per trial.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)
        self.backbone.fc = torch.nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        return F.cross_entropy(self(x), y)

    def configure_optimizers(self):
        pass  # Overridden by AutoResearchModule per trial


# --- 2. Define datasets (no augmentation here; system injects them) ---


def get_datasets() -> tuple[Dataset, Dataset]:
    """Return (train_dataset, val_dataset) for CIFAR-10.

    Applies only a basic ToTensor transform — the system injects
    Albumentations augmentations on top of the training set.
    """
    base_transform = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=base_transform
    )
    val_ds = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=base_transform
    )
    return train_ds, val_ds


# --- 3. Configure and run autoresearch ---


if __name__ == "__main__":
    train_ds, val_ds = get_datasets()

    config = SearchConfig(
        task_description=(
            "10-class image classification on CIFAR-10. "
            "The model outputs logits for 10 classes. "
            "Optimize top-1 accuracy on the validation set."
        ),
        hp_trials=30,
        aug_trials=20,
        epochs_per_trial=5,
        device="cuda",
    )

    result = run_autoresearch(CIFAR10Classifier(), train_ds, val_ds, config)

    print(f"\nBest {result['best_metric']['name']}: {result['best_metric']['value']:.4f}")
    print(f"Best hyperparams: {result['best_hyperparams']}")
    print(f"Best augmentations: {result['best_augmentations']}")
    print(f"\nTotal trials: {result['total_trials']}")
    print(f"Phase breakdown: {result['phase_breakdown']}")
