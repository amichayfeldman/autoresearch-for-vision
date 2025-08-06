"""Tests for basic level analysis."""

import pytest
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from unittest.mock import Mock, patch

from cv_error_analyzer.levels.level_1.analysis import BasicIterativeAnalysis
from cv_error_analyzer.utils.types import DatasetInfo, ErrorMetrics


@pytest.fixture
def basic_config():
    """Basic analysis configuration fixture."""
    return DictConfig({
        "max_iterations": 3,
        "saturation_threshold": 0.01,
        "improvement_threshold": 0.05,
        "percentile_threshold": 75,
        "min_samples_per_iteration": 10
    })


@pytest.fixture
def mock_dataset_info():
    """Mock dataset info fixture."""
    return DatasetInfo(
        dataset_path="/tmp/dataset",
        image_files={"train": ["image1.jpg", "image2.jpg"], "val": ["val1.jpg"]},
        annotation_files={},
        num_keypoints=17,
        splits=["train", "val"]
    )


class TestBasicIterativeAnalysis:
    """Test basic level analysis functionality."""
    
    def test_initialization(self, basic_config):
        """Test basic analysis initialization."""
        analysis = BasicIterativeAnalysis(basic_config)
        
        assert analysis.cfg == basic_config
        assert analysis.max_iterations == 3
        assert analysis.saturation_threshold == 0.01
        assert analysis.iterations_completed == 0
        assert len(analysis.iteration_results) == 0
    
    def test_bbox_iou_calculation(self, basic_config):
        """Test bounding box IoU calculation."""
        analysis = BasicIterativeAnalysis(basic_config)
        
        # Perfect overlap
        box_a = [0, 0, 100, 100]
        box_b = [0, 0, 100, 100]
        iou = analysis._calculate_bbox_iou(box_a, box_b)
        assert iou == 1.0
        
        # No overlap
        box_a = [0, 0, 50, 50]
        box_b = [100, 100, 150, 150]
        iou = analysis._calculate_bbox_iou(box_a, box_b)
        assert iou == 0.0
        
        # Partial overlap
        box_a = [0, 0, 100, 100]
        box_b = [50, 50, 150, 150]
        iou = analysis._calculate_bbox_iou(box_a, box_b)
        expected_iou = 2500 / (10000 + 10000 - 2500)  # intersection / union
        assert abs(iou - expected_iou) < 1e-6
    
    def test_keypoint_oks_calculation(self, basic_config):
        """Test keypoint OKS calculation."""
        analysis = BasicIterativeAnalysis(basic_config)
        
        # Identical keypoints
        kpts_a = [[10, 20, 1], [30, 40, 1]]
        kpts_b = [[10, 20, 1], [30, 40, 1]]
        oks = analysis._calculate_keypoint_oks(kpts_a, kpts_b, 10000)
        assert oks == 1.0
        
        # Empty keypoints
        oks = analysis._calculate_keypoint_oks([], [], 10000)
        assert oks == 0.0
        
        # Keypoints with different visibility
        kpts_a = [[10, 20, 1], [30, 40, 1]]
        kpts_b = [[10, 20, 0], [30, 40, 1]]  # First keypoint not visible
        oks = analysis._calculate_keypoint_oks(kpts_a, kpts_b, 10000)
        assert oks == 1.0  # Only visible keypoints considered
    
    def test_metrics_summary_calculation(self, basic_config):
        """Test metrics summary calculation."""
        analysis = BasicIterativeAnalysis(basic_config)
        
        # Create mock samples
        samples = []
        for i in range(5):
            error_metrics = ErrorMetrics(
                bbox_iou_error=0.1 + i * 0.1,
                keypoint_oks_error=0.2 + i * 0.1, 
                classification_f1_error=0.05,
                total_error=0.35 + i * 0.3,
                num_origin=1,
                num_augmented=1,
                num_matched=1
            )
            samples.append(Mock(error_metrics=error_metrics))
        
        summary = analysis._calculate_metrics_summary(samples)
        
        assert "mean_total_error" in summary
        assert "p75_total_error" in summary
        assert "num_samples" in summary
        assert summary["num_samples"] == 5
        
        # Check percentile calculation
        total_errors = [0.35, 0.65, 0.95, 1.25, 1.55]
        expected_p75 = np.percentile(total_errors, 75)
        assert abs(summary["p75_total_error"] - expected_p75) < 1e-6
    
    def test_saturation_detection(self, basic_config):
        """Test improvement saturation detection."""
        analysis = BasicIterativeAnalysis(basic_config)
        
        # Not enough iterations
        assert not analysis._check_saturation(1)
        
        # Add some iteration results with improvements
        analysis.iteration_results = [
            Mock(improvement_from_previous=0.1),
            Mock(improvement_from_previous=0.05),
            Mock(improvement_from_previous=0.005)  # Below threshold
        ]
        
        # Should detect saturation
        assert analysis._check_saturation(3)
    
    @patch('cv_error_analyzer.levels.level_1.analysis.YOLO')
    @patch('cv_error_analyzer.levels.level_1.analysis.cv2')
    def test_sample_analysis_error_handling(self, mock_cv2, mock_yolo, basic_config):
        """Test error handling in sample analysis."""
        analysis = BasicIterativeAnalysis(basic_config)
        analysis.model = Mock()
        
        # Mock cv2.imread to return None (file not found)
        mock_cv2.imread.return_value = None
        
        result = analysis._analyze_single_sample("nonexistent.jpg", 17, 640)
        assert result is None
    
    def test_improvement_metrics_calculation(self, basic_config):
        """Test improvement metrics calculation."""
        analysis = BasicIterativeAnalysis(basic_config)
        
        # Create mock iteration results
        analysis.iteration_results = [
            Mock(percentile_75_error=1.0, improvement_from_previous=None, reached_saturation=False),
            Mock(percentile_75_error=0.8, improvement_from_previous=0.2, reached_saturation=False),
            Mock(percentile_75_error=0.7, improvement_from_previous=0.125, reached_saturation=True)
        ]
        
        metrics = analysis._calculate_improvement_metrics()
        
        assert metrics.initial_error == 1.0
        assert metrics.final_error == 0.7
        assert abs(metrics.total_improvement - 0.3) < 1e-6  # (1.0 - 0.7) / 1.0
        assert metrics.iterations_to_saturation == 3
    
    def test_recommendation_generation(self, basic_config):
        """Test recommendation generation."""
        analysis = BasicIterativeAnalysis(basic_config)
        
        # Mock iteration results with high errors
        mock_summary = {
            "mean_bbox_error": 0.4,  # High bbox error
            "mean_keypoint_error": 0.5,  # High keypoint error
            "mean_classification_error": 0.1
        }
        analysis.iteration_results = [Mock(
            samples=[Mock()],  # Non-empty samples
            metrics_summary=mock_summary
        )]
        
        recommendations = analysis._generate_recommendations()
        
        assert len(recommendations) > 0
        
        # Check for augmentation recommendation due to high bbox error
        aug_recs = [r for r in recommendations if r.recommendation_type == "augmentation"]
        assert len(aug_recs) > 0
        
        # Check for training recommendation due to high keypoint error  
        training_recs = [r for r in recommendations if r.recommendation_type == "training"]
        assert len(training_recs) > 0


if __name__ == "__main__":
    pytest.main([__file__])