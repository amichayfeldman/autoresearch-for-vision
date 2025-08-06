"""Tests for training validation system."""

import pytest
from omegaconf import DictConfig
from unittest.mock import Mock, patch

from cv_error_analyzer.validation.training_validator import (
    EpochScheduler, EarlyStopper, TrainingValidator
)
from cv_error_analyzer.utils.types import (
    TrainingMetrics, TrainingResult, MetricType
)


@pytest.fixture
def scheduler_config():
    """Epoch scheduler configuration fixture."""
    return DictConfig({
        "base_epochs": 50,
        "max_epochs_per_iteration": 100,
        "adaptive_factor": 1.2
    })


@pytest.fixture
def validator_config():
    """Training validator configuration fixture."""
    return DictConfig({
        "validation_metric": "loss",
        "improvement_threshold": 0.05,
        "clearml": {"dataset_project": "test"},
        "training_execution": {
            "base_epochs": 50,
            "max_epochs_per_iteration": 100
        }
    })


class TestEpochScheduler:
    """Test epoch scheduling functionality."""
    
    def test_initialization(self, scheduler_config):
        """Test scheduler initialization."""
        scheduler = EpochScheduler(scheduler_config)
        
        assert scheduler.base_epochs == 50
        assert scheduler.max_epochs == 100
        assert scheduler.adaptive_factor == 1.2
        assert len(scheduler.performance_history) == 0
    
    def test_initial_epoch_scheduling(self, scheduler_config):
        """Test epoch scheduling for first iteration."""
        scheduler = EpochScheduler(scheduler_config)
        
        epochs = scheduler.get_epochs_for_iteration(1)
        assert epochs == 50  # Should return base_epochs
    
    def test_adaptive_scheduling_improvement(self, scheduler_config):
        """Test adaptive scheduling when performance improves."""
        scheduler = EpochScheduler(scheduler_config)
        
        # Add performance history showing improvement
        scheduler.performance_history = [1.0, 0.8]  # Improvement of 0.2
        
        epochs = scheduler.get_epochs_for_iteration(2, 0.6)  # Further improvement
        expected = min(int(50 * 1.2), 100)  # Should increase epochs
        assert epochs == expected
    
    def test_adaptive_scheduling_degradation(self, scheduler_config):
        """Test adaptive scheduling when performance degrades."""
        scheduler = EpochScheduler(scheduler_config)
        
        # Add performance history showing degradation
        scheduler.performance_history = [0.8, 1.0]  # Degradation of 0.2
        
        epochs = scheduler.get_epochs_for_iteration(2, 1.1)  # Further degradation
        expected = max(int(50 * 0.8), 20)  # Should decrease epochs
        assert epochs == expected


class TestEarlyStopper:
    """Test early stopping functionality."""
    
    def test_initialization(self):
        """Test early stopper initialization."""
        stopper = EarlyStopper(patience=5, min_delta=0.01)
        
        assert stopper.patience == 5
        assert stopper.min_delta == 0.01
        assert stopper.best_score == float('inf')
        assert stopper.wait == 0
    
    def test_epoch_limit_respected(self):
        """Test that epoch limits are always respected."""
        stopper = EarlyStopper(patience=10)
        
        # Should stop when max epochs reached regardless of improvement
        should_stop = stopper.should_stop(0.5, 50, 50)
        assert should_stop
    
    def test_early_stopping_on_improvement(self):
        """Test early stopping behavior with improvement."""
        stopper = EarlyStopper(patience=3, min_delta=0.01)
        
        # Significant improvement - should not stop
        should_stop = stopper.should_stop(0.8, 10, 50)  # First call
        assert not should_stop
        assert stopper.best_score == 0.8
        assert stopper.wait == 0
        
        # Another improvement - should not stop
        should_stop = stopper.should_stop(0.7, 11, 50)
        assert not should_stop
        assert stopper.best_score == 0.7
        assert stopper.wait == 0
    
    def test_early_stopping_on_plateau(self):
        """Test early stopping when performance plateaus."""
        stopper = EarlyStopper(patience=2, min_delta=0.01)
        
        # Initial score
        stopper.should_stop(0.8, 10, 50)
        
        # No significant improvement
        stopper.should_stop(0.799, 11, 50)  # wait = 1
        should_stop = stopper.should_stop(0.798, 12, 50)  # wait = 2
        assert should_stop  # Should stop due to patience exceeded


class TestTrainingValidator:
    """Test training validation functionality."""
    
    def test_initialization(self, validator_config):
        """Test validator initialization."""
        with patch('cv_error_analyzer.validation.training_validator.ClearMLAdapter'):
            validator = TrainingValidator(validator_config)
            
            assert validator.validation_metric == MetricType.LOSS
            assert validator.improvement_threshold == 0.05
    
    def test_loss_metric_comparison(self, validator_config):
        """Test loss metric comparison (lower is better)."""
        with patch('cv_error_analyzer.validation.training_validator.ClearMLAdapter'):
            validator = TrainingValidator(validator_config)
            
            baseline = TrainingMetrics(loss={"val_box_loss": 0.1})
            new_metrics = TrainingMetrics(loss={"val_box_loss": 0.08})
            
            improved, improvement_pct = validator._compare_metrics(baseline, new_metrics)
            
            assert improved  # 20% improvement > 5% threshold
            assert abs(improvement_pct - 0.2) < 1e-6  # (0.1 - 0.08) / 0.1
    
    def test_map_metric_comparison(self, validator_config):
        """Test mAP metric comparison (higher is better)."""
        validator_config.validation_metric = "mAP"
        
        with patch('cv_error_analyzer.validation.training_validator.ClearMLAdapter'):
            validator = TrainingValidator(validator_config)
            
            baseline = TrainingMetrics(loss={}, map_50=0.8)
            new_metrics = TrainingMetrics(loss={}, map_50=0.85)
            
            improved, improvement_pct = validator._compare_metrics(baseline, new_metrics)
            
            assert improved  # 6.25% improvement > 5% threshold
            assert abs(improvement_pct - 0.0625) < 1e-6  # (0.85 - 0.8) / 0.8
    
    def test_validation_confidence_calculation(self, validator_config):
        """Test validation confidence calculation."""
        with patch('cv_error_analyzer.validation.training_validator.ClearMLAdapter'):
            validator = TrainingValidator(validator_config)
            
            # Stable training result
            stable_result = TrainingResult(
                iteration=1,
                training_metrics=TrainingMetrics(loss={}),
                new_model_path="",
                training_config={},
                epochs_used=50,
                early_stopped=False,
                improved=True
            )
            
            confidence = validator._calculate_validation_confidence(stable_result)
            assert confidence == 0.8  # Base confidence
            
            # Unstable training result (early stopped, few epochs)
            unstable_result = TrainingResult(
                iteration=1,
                training_metrics=TrainingMetrics(loss={}),
                new_model_path="",
                training_config={},
                epochs_used=15,
                early_stopped=True,
                improved=True
            )
            
            confidence = validator._calculate_validation_confidence(unstable_result)
            expected = 0.8 * 0.9 * 0.8  # base * early_stop_penalty * low_epochs_penalty
            assert abs(confidence - expected) < 1e-6
    
    def test_baseline_metrics_creation(self, validator_config):
        """Test baseline metrics creation."""
        with patch('cv_error_analyzer.validation.training_validator.ClearMLAdapter'):
            validator = TrainingValidator(validator_config)
            
            baseline_config = {"train": {"epochs": 100}}
            metrics = validator._create_baseline_metrics(baseline_config)
            
            assert isinstance(metrics, TrainingMetrics)
            assert "train_box_loss" in metrics.loss
            assert "val_box_loss" in metrics.loss
            assert metrics.map_50 is not None
            assert metrics.epochs_completed > 0


if __name__ == "__main__":
    pytest.main([__file__])