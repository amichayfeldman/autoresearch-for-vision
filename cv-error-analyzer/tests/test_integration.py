"""Integration tests for CV Error Analyzer failure scenarios."""

import pytest
from pathlib import Path
from omegaconf import DictConfig
from unittest.mock import Mock, patch, MagicMock

from cv_error_analyzer.core.engine import ErrorAnalyzerEngine
from cv_error_analyzer.levels.level_1.analysis import BasicIterativeAnalysis
from cv_error_analyzer.utils.types import DatasetInfo


class TestConfigurationValidation:
    """Test configuration validation at startup."""
    
    def test_missing_required_config_fails(self):
        """Test that missing required config fields raise errors."""
        # Missing analysis_level.max_iterations
        incomplete_cfg = DictConfig({
            "analysis_level": {
                "saturation_threshold": 0.01,
                "improvement_threshold": 0.05
            },
            "training_validation": {"validation_metric": "loss"},
            "clearml": {"dataset_project": "test"}
        })
        
        with pytest.raises(ValueError, match="Required configuration field missing"):
            ErrorAnalyzerEngine(incomplete_cfg)
    
    def test_invalid_percentile_range_fails(self):
        """Test that invalid percentile values are rejected."""
        invalid_cfg = DictConfig({
            "analysis_level": {
                "max_iterations": 5,
                "saturation_threshold": 0.01,
                "improvement_threshold": 0.05,
                "error_percentile": 120  # Invalid - over 95
            },
            "training_validation": {"validation_metric": "loss"},
            "clearml": {"dataset_project": "test"}
        })
        
        with pytest.raises(ValueError, match="error_percentile must be between 50-95"):
            ErrorAnalyzerEngine(invalid_cfg)
    
    def test_invalid_max_iterations_fails(self):
        """Test that invalid iteration counts are rejected."""
        invalid_cfg = DictConfig({
            "analysis_level": {
                "max_iterations": 150,  # Invalid - over 100
                "saturation_threshold": 0.01,
                "improvement_threshold": 0.05
            },
            "training_validation": {"validation_metric": "loss"},
            "clearml": {"dataset_project": "test"}
        })
        
        with pytest.raises(ValueError, match="max_iterations must be between 1-100"):
            ErrorAnalyzerEngine(invalid_cfg)
    
    def test_valid_configuration_passes(self):
        """Test that valid configuration passes validation."""
        valid_cfg = DictConfig({
            "analysis_level": {
                "max_iterations": 10,
                "saturation_threshold": 0.01,
                "improvement_threshold": 0.05,
                "error_percentile": 80,
                "start_level": 1,
                "max_level": 1
            },
            "training_validation": {"validation_metric": "loss"},
            "clearml": {"dataset_project": "test"}
        })
        
        # Should not raise any exceptions
        with patch('cv_error_analyzer.adapters.clearml_adapter.ClearMLAdapter'):
            with patch('cv_error_analyzer.validation.training_validator.TrainingValidator'):
                engine = ErrorAnalyzerEngine(valid_cfg)
                assert engine is not None


class TestClearMLFailures:
    """Test ClearML integration failure scenarios."""
    
    @patch('cv_error_analyzer.adapters.clearml_adapter.Dataset')
    def test_dataset_loading_failure_handling(self, mock_dataset):
        """Test graceful handling of ClearML dataset loading failures."""
        from cv_error_analyzer.adapters.clearml_adapter import ClearMLAdapter
        
        # Mock ClearML dataset failure
        mock_dataset.get.side_effect = Exception("ClearML connection failed")
        
        cfg = DictConfig({"dataset_project": "test"})
        adapter = ClearMLAdapter(cfg)
        
        with pytest.raises(Exception, match="ClearML connection failed"):
            adapter.load_dataset("1.2")
    
    @patch('cv_error_analyzer.adapters.clearml_adapter.Task')
    def test_experiment_config_fallback(self, mock_task):
        """Test fallback behavior when experiment config loading fails."""
        from cv_error_analyzer.adapters.clearml_adapter import ClearMLAdapter
        
        # Mock Task.get_task failure
        mock_task.get_task.side_effect = Exception("Experiment not found")
        
        cfg = DictConfig({"dataset_project": "test"})
        adapter = ClearMLAdapter(cfg)
        
        # Should return default config instead of crashing
        result = adapter.load_experiment_config("/fake/model.pt")
        
        assert "train" in result
        assert result["train"]["imgsz"] == 640  # Default value


class TestAnalysisFailures:
    """Test analysis failure scenarios."""
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        cfg = DictConfig({
            "max_iterations": 5,
            "saturation_threshold": 0.01,
            "improvement_threshold": 0.05,
            "error_percentile": 75
        })
        
        analysis = BasicIterativeAnalysis(cfg)
        
        # Empty dataset info
        empty_dataset = DatasetInfo(
            dataset_path="/tmp",
            image_files={},
            annotation_files={},
            num_keypoints=17,
            splits=[]
        )
        
        with patch.object(analysis, '_load_ground_truth') as mock_gt:
            mock_gt.return_value = None
            
            samples = analysis._analyze_split(empty_dataset, "train", 640, 1)
            assert samples == []
    
    def test_model_prediction_failures(self):
        """Test handling of model prediction failures."""
        cfg = DictConfig({
            "max_iterations": 5,
            "saturation_threshold": 0.01,
            "improvement_threshold": 0.05,
            "error_percentile": 75
        })
        
        analysis = BasicIterativeAnalysis(cfg)
        analysis.model = Mock()
        
        # Mock model failure
        analysis.model.side_effect = Exception("CUDA out of memory")
        
        import numpy as np
        fake_image = np.zeros((640, 640, 3), dtype=np.uint8)
        fake_gt = {"bboxes": [[0, 0, 100, 100]], "keypoints": [[[10, 20, 1]]]}
        
        # Should handle model failures gracefully
        with pytest.raises(Exception, match="CUDA out of memory"):
            analysis._analyze_sample_data(fake_image, fake_gt, "test.jpg")


class TestTrainingFailures:
    """Test training execution failure scenarios."""
    
    @patch('cv_error_analyzer.validation.training_validator.subprocess')
    def test_training_timeout_handling(self, mock_subprocess):
        """Test handling of training timeouts."""
        from cv_error_analyzer.validation.training_validator import TrainingOrchestrator
        
        cfg = DictConfig({
            "base_epochs": 50,
            "max_epochs_per_iteration": 100,
            "early_stopping_patience": 10
        })
        
        orchestrator = TrainingOrchestrator(cfg)
        
        # Mock subprocess timeout
        from subprocess import TimeoutExpired
        mock_subprocess.run.side_effect = TimeoutExpired("yolo", 7200)
        
        result = orchestrator._run_yolo_training({"epochs": 50}, 1)
        
        assert result["success"] is False
        assert "timeout" in result["error"].lower()
    
    @patch('cv_error_analyzer.validation.training_validator.subprocess')
    def test_training_command_failure(self, mock_subprocess):
        """Test handling of training command failures."""
        from cv_error_analyzer.validation.training_validator import TrainingOrchestrator
        
        cfg = DictConfig({
            "base_epochs": 50,
            "max_epochs_per_iteration": 100
        })
        
        orchestrator = TrainingOrchestrator(cfg)
        
        # Mock subprocess failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Model file not found"
        mock_subprocess.run.return_value = mock_result
        
        result = orchestrator._run_yolo_training({"epochs": 50}, 1)
        
        assert result["success"] is False
        assert "Model file not found" in result["error"]


class TestPercentileConfiguration:
    """Test configurable percentile functionality."""
    
    def test_custom_percentile_calculation(self):
        """Test that custom percentile values are used correctly."""
        cfg = DictConfig({
            "max_iterations": 5,
            "saturation_threshold": 0.01,
            "improvement_threshold": 0.05,
            "error_percentile": 90  # Use 90th percentile instead of 75th
        })
        
        analysis = BasicIterativeAnalysis(cfg)
        assert analysis.error_percentile == 90
        
        # Mock some samples with known error distribution
        samples = []
        import numpy as np
        from cv_error_analyzer.utils.types import ErrorMetrics, SampleResult
        
        # Create samples with errors 0.1, 0.2, ..., 1.0
        for i in range(10):
            error_val = (i + 1) * 0.1
            metrics = ErrorMetrics(
                bbox_iou_error=error_val,
                keypoint_oks_error=error_val,
                classification_f1_error=error_val,
                total_error=error_val,
                num_origin=1,
                num_augmented=1,
                num_matched=1
            )
            samples.append(SampleResult(
                file_path=f"test_{i}.jpg",
                error_metrics=metrics
            ))
        
        summary = analysis._calculate_metrics_summary(samples)
        
        # 90th percentile of [0.1, 0.2, ..., 1.0] should be 0.9
        assert abs(summary["p90_total_error"] - 0.9) < 0.1
        assert "configured_percentile" in summary
        assert summary["configured_percentile"] == 90
    
    def test_percentile_backward_compatibility(self):
        """Test backward compatibility with old percentile_threshold config."""
        cfg = DictConfig({
            "max_iterations": 5,
            "saturation_threshold": 0.01,
            "improvement_threshold": 0.05,
            "percentile_threshold": 80  # Old config name
        })
        
        analysis = BasicIterativeAnalysis(cfg)
        assert analysis.error_percentile == 80  # Should fallback to old config


if __name__ == "__main__":
    pytest.main([__file__])