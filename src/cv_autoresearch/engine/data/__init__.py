"""Data modules and dataset factories."""

from cv_autoresearch.engine.data.datamodule import VisionDataModule
from cv_autoresearch.engine.data.synthetic import SyntheticPromptDataset

__all__ = [
    "SyntheticPromptDataset",
    "VisionDataModule",
]
