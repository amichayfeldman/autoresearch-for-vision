"""PredictionCallback: placeholder for future VLM integration."""

from __future__ import annotations

from pytorch_lightning.callbacks import Callback


class PredictionCallback(Callback):
    """Placeholder callback for future VLM prediction collection.

    Currently a no-op. Future versions will collect val predictions
    and targets for VLM analysis after each epoch.
    """
