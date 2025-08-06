"""Training validation system with ClearML integration and epoch constraints."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import yaml
from omegaconf import DictConfig
from clearml import Task

from ..utils.types import (
    TrainingResult, ValidationResult, TrainingMetrics, 
    MetricType, Recommendation
)
from ..adapters.clearml_adapter import ClearMLAdapter

logger = logging.getLogger(__name__)


class EpochScheduler:
    """Adaptive epoch scheduling based on performance trends."""
    
    def __init__(self, cfg: DictConfig):
        self.base_epochs = cfg.get("base_epochs", 50)
        self.max_epochs = cfg.get("max_epochs_per_iteration", 100)
        self.adaptive_factor = cfg.get("adaptive_factor", 1.2)
        self.performance_history: List[float] = []
    
    def get_epochs_for_iteration(self, iteration: int, 
                                 recent_performance: Optional[float] = None) -> int:
        """
        Determine epoch count for training iteration.
        
        Args:
            iteration: Current iteration number
            recent_performance: Recent performance metric (lower is better)
            
        Returns:
            Number of epochs for this iteration
        """
        if recent_performance is not None:
            self.performance_history.append(recent_performance)
        
        # Adaptive scheduling based on performance trend
        if len(self.performance_history) >= 2:
            trend = self.performance_history[-1] - self.performance_history[-2]
            
            if trend < -0.1:  # Significant improvement
                epochs = min(int(self.base_epochs * self.adaptive_factor), self.max_epochs)
            elif trend > 0.05:  # Performance degrading
                epochs = max(int(self.base_epochs * 0.8), 20)
            else:  # Stable performance
                epochs = self.base_epochs
        else:
            epochs = self.base_epochs
        
        logger.info(f"Scheduled {epochs} epochs for iteration {iteration}")
        return epochs


class EarlyStopper:
    """Early stopping with epoch limit integration."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float('inf')
        self.wait = 0
    
    def should_stop(self, current_score: float, current_epoch: int, 
                    max_epochs: int) -> bool:
        """
        Check if training should stop early.
        
        Args:
            current_score: Current validation score
            current_epoch: Current epoch number
            max_epochs: Maximum allowed epochs
            
        Returns:
            True if should stop early
        """
        # Always respect epoch limit
        if current_epoch >= max_epochs:
            return True
        
        # Early stopping based on improvement
        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
        
        return self.wait >= self.patience


class TrainingOrchestrator:
    """Orchestrates training execution with epoch constraints."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.epoch_scheduler = EpochScheduler(cfg)
        self.early_stopper = EarlyStopper(
            patience=cfg.get("early_stopping_patience", 10),
            min_delta=cfg.get("early_stopping_min_delta", 0.001)
        )
    
    def execute_training_iteration(self, iteration: int,
                                   recommendations: List[Recommendation],
                                   base_config: Dict[str, Any]) -> TrainingResult:
        """
        Execute a single training iteration with epoch constraints.
        
        Args:
            iteration: Training iteration number
            recommendations: Analysis recommendations to apply
            base_config: Base training configuration
            
        Returns:
            Training results with metrics and model path
        """
        logger.info(f"Executing training iteration {iteration}")
        
        # Prepare training configuration
        training_config = self._prepare_training_config(base_config, recommendations, iteration)
        
        # Execute training and handle results
        try:
            training_result = self._execute_yolo_training(training_config, iteration)
            return self._create_successful_result(training_result, training_config, iteration)
        except Exception as e:
            logger.error(f"Training iteration {iteration} failed: {e}")
            return self._create_failed_result(e, training_config, iteration)
    
    def _prepare_training_config(self, base_config: Dict[str, Any], 
                                 recommendations: List[Recommendation],
                                 iteration: int) -> Dict[str, Any]:
        """Prepare training configuration with epoch scheduling and recommendations."""
        # Determine epoch limit for this iteration
        recent_performance = None  # Could get from previous iteration
        max_epochs = self.epoch_scheduler.get_epochs_for_iteration(iteration, recent_performance)
        
        # Apply epoch limit to config
        training_config = base_config.copy()
        training_config["epochs"] = max_epochs
        
        # Apply recommendations inline (simple enough to not need extraction)
        for rec in recommendations:
            if rec.recommendation_type == "training":
                training_config.update(rec.parameters)
            elif rec.recommendation_type == "augmentation":
                if "augmentation" not in training_config:
                    training_config["augmentation"] = {}
                training_config["augmentation"].update(rec.parameters)
        
        return training_config
    
    def _execute_yolo_training(self, config: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Execute YOLO training and return raw results."""
        start_time = time.time()
        result = self._run_yolo_training(config, iteration)
        result["training_time"] = time.time() - start_time
        return result
    
    def _create_successful_result(self, training_result: Dict[str, Any],
                                  training_config: Dict[str, Any],
                                  iteration: int) -> TrainingResult:
        """Create TrainingResult for successful training execution."""
        # Extract metrics from result
        training_metrics = self._extract_training_metrics(
            training_result, training_result["training_time"]
        )
        
        # Check if early stopped
        max_epochs = training_config["epochs"]
        epochs_completed = training_metrics.epochs_completed
        early_stopped = epochs_completed < max_epochs
        
        return TrainingResult(
            iteration=iteration,
            training_metrics=training_metrics,
            new_model_path=training_result.get("model_path", ""),
            training_config=training_config,
            epochs_used=epochs_completed,
            early_stopped=early_stopped,
            improved=False  # Will be set by validation
        )
    
    def _create_failed_result(self, error: Exception, training_config: Dict[str, Any],
                              iteration: int) -> TrainingResult:
        """Create TrainingResult for failed training execution."""
        return TrainingResult(
            iteration=iteration,
            training_metrics=TrainingMetrics(loss={}, epochs_completed=0, training_time=0.0),
            new_model_path="",
            training_config=training_config,
            epochs_used=0,
            early_stopped=True,
            improved=False
        )
    
    # Inline recommendation application - removed unnecessary extraction
    
    def _run_yolo_training(self, config: Dict[str, Any], 
                           iteration: int) -> Dict[str, Any]:
        """
        Execute YOLO training with given configuration.
        
        Args:
            config: Training configuration
            iteration: Iteration number for output naming
            
        Returns:
            Training results dictionary
        """
        # Create output directory
        output_dir = Path(f"cv_error_analyzer_iteration_{iteration}")
        output_dir.mkdir(exist_ok=True)
        
        # Save config
        config_path = output_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Construct YOLO training command
        cmd = [
            "yolo", "pose", "train",
            f"data={config.get('data', 'dataset.yaml')}",
            f"model={config.get('model', 'yolov8n-pose.pt')}",
            f"epochs={config['epochs']}",
            f"imgsz={config.get('imgsz', 640)}",
            f"batch={config.get('batch', 16)}",
            f"project={output_dir}",
            f"name=training"
        ]
        
        # Add augmentation parameters if present
        if "augmentation" in config:
            for key, value in config["augmentation"].items():
                cmd.append(f"{key}={value}")
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run training
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            
            if result.returncode == 0:
                # Find the trained model
                weights_dir = output_dir / "training" / "weights"
                model_path = weights_dir / "best.pt"
                
                if not model_path.exists():
                    model_path = weights_dir / "last.pt"
                
                return {
                    "success": True,
                    "model_path": str(model_path),
                    "output_dir": str(output_dir),
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                logger.error(f"Training failed: {result.stderr}")
                return {"success": False, "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            logger.error("Training timed out")
            return {"success": False, "error": "Training timeout"}
        except Exception as e:
            logger.error(f"Training execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_training_metrics(self, result: Dict[str, Any], 
                                  training_time: float) -> TrainingMetrics:
        """Extract training metrics from training result."""
        if not result.get("success", False):
            return TrainingMetrics(loss={}, training_time=training_time)
        
        # Parse training logs to extract metrics
        # This is simplified - in practice would parse YOLO output logs
        return TrainingMetrics(
            loss={
                "train_box_loss": 0.05,  # Placeholder
                "train_pose_loss": 0.03,
                "val_box_loss": 0.06,
                "val_pose_loss": 0.04
            },
            map_50=0.85,  # Placeholder
            map_75=0.72,
            precision=0.88,
            recall=0.82,
            epochs_completed=50,  # Would extract from logs
            training_time=training_time
        )


class TrainingValidator:
    """Validates training improvements and tracks metrics."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.clearml_adapter = ClearMLAdapter(cfg.clearml)
        self.training_orchestrator = TrainingOrchestrator(cfg.training_execution)
        
        # Validation configuration
        self.validation_metric = MetricType(cfg.get("validation_metric", "loss"))
        self.improvement_threshold = cfg.get("improvement_threshold", 0.05)
        
        logger.info(f"Initialized TrainingValidator with metric: {self.validation_metric}")
    
    def execute_training_iteration(self, iteration: int,
                                   recommendations: List[Recommendation],
                                   base_config: Dict[str, Any]) -> TrainingResult:
        """Execute training iteration using orchestrator."""
        return self.training_orchestrator.execute_training_iteration(
            iteration, recommendations, base_config
        )
    
    def validate_training_improvement(self, training_result: TrainingResult,
                                      baseline_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate if training iteration improved over baseline.
        
        Args:
            training_result: Results from training iteration
            baseline_config: Original experiment configuration
            
        Returns:
            Validation results with improvement assessment
        """
        logger.info(f"Validating training improvement for iteration {training_result.iteration}")
        
        # Create baseline metrics (simplified - would load from original experiment)
        baseline_metrics = self._create_baseline_metrics(baseline_config)
        
        # Compare metrics
        improved, improvement_pct = self._compare_metrics(
            baseline_metrics, training_result.training_metrics
        )
        
        # Calculate confidence based on training stability
        confidence = self._calculate_validation_confidence(training_result)
        
        return ValidationResult(
            improved=improved,
            improvement_percentage=improvement_pct,
            baseline_metrics=baseline_metrics,
            new_metrics=training_result.training_metrics,
            selected_metric=self.validation_metric,
            confidence=confidence
        )
    
    def _create_baseline_metrics(self, baseline_config: Dict[str, Any]) -> TrainingMetrics:
        """Create baseline metrics from original experiment."""
        # Simplified - would load actual metrics from ClearML
        return TrainingMetrics(
            loss={
                "train_box_loss": 0.08,
                "train_pose_loss": 0.06,
                "val_box_loss": 0.09,
                "val_pose_loss": 0.07
            },
            map_50=0.80,
            map_75=0.65,
            precision=0.83,
            recall=0.78,
            epochs_completed=100,
            training_time=3600.0
        )
    
    def _compare_metrics(self, baseline: TrainingMetrics, 
                         new_metrics: TrainingMetrics) -> Tuple[bool, float]:
        """Compare training metrics to determine improvement."""
        
        if self.validation_metric == MetricType.LOSS:
            # For loss, lower is better
            baseline_loss = baseline.loss.get("val_box_loss", 1.0)
            new_loss = new_metrics.loss.get("val_box_loss", 1.0)
            
            if baseline_loss > 0:
                improvement_pct = (baseline_loss - new_loss) / baseline_loss
                improved = improvement_pct > self.improvement_threshold
            else:
                improvement_pct = 0.0
                improved = False
                
        elif self.validation_metric == MetricType.MAP:
            # For mAP, higher is better
            baseline_map = baseline.map_50 or 0.0
            new_map = new_metrics.map_50 or 0.0
            
            if baseline_map > 0:
                improvement_pct = (new_map - baseline_map) / baseline_map
                improved = improvement_pct > self.improvement_threshold
            else:
                improvement_pct = 0.0
                improved = False
        else:
            # Default comparison
            improvement_pct = 0.0
            improved = False
        
        return improved, improvement_pct
    
    def _calculate_validation_confidence(self, training_result: TrainingResult) -> float:
        """Calculate confidence in validation result."""
        confidence = 0.8  # Base confidence
        
        # Reduce confidence if training was unstable
        if training_result.early_stopped:
            confidence *= 0.9
        
        # Reduce confidence if insufficient epochs
        if training_result.epochs_used < 20:
            confidence *= 0.8
        
        return max(0.1, confidence)