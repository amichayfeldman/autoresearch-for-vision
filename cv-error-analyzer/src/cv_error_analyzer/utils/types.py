"""Type definitions for CV Error Analyzer."""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum


class RecommendationType(str, Enum):
    """Types of analysis recommendations."""
    AUGMENTATION = "augmentation"
    TRAINING = "training"
    ARCHITECTURE = "architecture"
    PREPROCESSING = "preprocessing"


class MetricType(str, Enum):
    """Types of validation metrics."""
    LOSS = "loss"
    MAP = "mAP"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"


@dataclass
class ErrorMetrics:
    """Error metrics for a single sample."""
    bbox_iou_error: float
    keypoint_oks_error: float
    classification_f1_error: float
    total_error: float
    num_origin: int
    num_augmented: int
    num_matched: int


@dataclass
class SampleResult:
    """Analysis result for a single sample."""
    file_path: str
    error_metrics: ErrorMetrics
    original_image: Optional[Any] = None
    augmented_image: Optional[Any] = None
    augmentation_name: Optional[str] = None
    rank: Optional[int] = None


@dataclass
class IterationResult:
    """Results from a single iteration of analysis."""
    iteration: int
    samples: List[SampleResult]
    metrics_summary: Dict[str, float]
    percentile_75_error: float
    improvement_from_previous: Optional[float] = None
    reached_saturation: bool = False


@dataclass
class Recommendation:
    """Analysis recommendation for model improvement."""
    recommendation_type: RecommendationType
    confidence: float
    parameters: Dict[str, Any]
    expected_improvement: Optional[float] = None
    description: str = ""


@dataclass
class ImprovementMetrics:
    """Metrics tracking improvement over iterations."""
    initial_error: float
    final_error: float
    total_improvement: float
    improvement_rate: float
    final_improvement_rate: float
    iterations_to_saturation: Optional[int] = None


@dataclass
class AnalysisResult:
    """Complete analysis results for a level."""
    level: int
    iterations: List[IterationResult]
    improvement_metrics: ImprovementMetrics
    final_recommendations: List[Recommendation]
    reached_saturation: bool
    metadata: Dict[str, Any]


@dataclass
class TrainingMetrics:
    """Training validation metrics."""
    loss: Dict[str, float]
    map_50: Optional[float] = None
    map_75: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    epochs_completed: int = 0
    training_time: float = 0.0


@dataclass
class ValidationResult:
    """Training validation results."""
    improved: bool
    improvement_percentage: float
    baseline_metrics: TrainingMetrics
    new_metrics: TrainingMetrics
    selected_metric: MetricType
    confidence: float


@dataclass
class TrainingResult:
    """Results from training execution."""
    iteration: int
    training_metrics: TrainingMetrics
    new_model_path: str
    training_config: Dict[str, Any]
    epochs_used: int
    early_stopped: bool
    improved: bool
    validation_result: Optional[ValidationResult] = None


@dataclass
class DatasetInfo:
    """ClearML dataset information."""
    dataset_path: str
    image_files: Dict[str, List[str]]
    annotation_files: Dict[str, Any]
    num_keypoints: int
    splits: List[str]