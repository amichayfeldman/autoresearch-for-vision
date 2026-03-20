"""cv-autoresearch: AI-directed CV hyperparameter and augmentation search."""

from cv_autoresearch.config.schema import SearchConfig
from cv_autoresearch.engine.autoresearch import run_autoresearch

__all__ = ["SearchConfig", "run_autoresearch"]
