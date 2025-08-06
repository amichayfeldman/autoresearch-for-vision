"""CV Error Analyzer - Automated AI system for computer vision model error analysis."""

from .core.engine import ErrorAnalyzerEngine
from .levels.level_1.analysis import BasicIterativeAnalysis
from .validation.training_validator import TrainingValidator
from .adapters.clearml_adapter import ClearMLAdapter

__version__ = "0.1.0"
__all__ = [
    "ErrorAnalyzerEngine",
    "BasicIterativeAnalysis", 
    "TrainingValidator",
    "ClearMLAdapter",
]