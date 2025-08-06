#!/usr/bin/env python3
"""
Core functionality validation script.
Tests the essential algorithms without external ML dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_types():
    """Test type definitions."""
    from cv_error_analyzer.utils.types import ErrorMetrics, AnalysisResult, RecommendationType
    
    # Test ErrorMetrics creation
    metrics = ErrorMetrics(
        bbox_iou_error=0.1,
        keypoint_oks_error=0.2,
        classification_f1_error=0.05,
        total_error=0.35,
        num_origin=1,
        num_augmented=1,
        num_matched=1
    )
    assert metrics.total_error == 0.35
    print("✅ ErrorMetrics working")
    
    # Test RecommendationType enum
    assert RecommendationType.AUGMENTATION == "augmentation"
    print("✅ RecommendationType enum working")

def test_basic_analysis():
    """Test basic analysis core algorithms."""
    from omegaconf import DictConfig
    from cv_error_analyzer.levels.level_1.analysis import BasicIterativeAnalysis
    
    cfg = DictConfig({
        'max_iterations': 5,
        'saturation_threshold': 0.01,
        'improvement_threshold': 0.05
    })
    
    analysis = BasicIterativeAnalysis(cfg)
    print("✅ BasicIterativeAnalysis instantiated")
    
    # Test IoU calculations
    iou_perfect = analysis._calculate_bbox_iou([0, 0, 100, 100], [0, 0, 100, 100])
    assert iou_perfect == 1.0, f"Perfect IoU should be 1.0, got {iou_perfect}"
    
    iou_none = analysis._calculate_bbox_iou([0, 0, 50, 50], [100, 100, 150, 150])
    assert iou_none == 0.0, f"No overlap IoU should be 0.0, got {iou_none}"
    
    iou_partial = analysis._calculate_bbox_iou([0, 0, 100, 100], [50, 50, 150, 150])
    expected = 2500 / (10000 + 10000 - 2500)  # intersection / union
    assert abs(iou_partial - expected) < 1e-6
    print("✅ IoU calculations correct")
    
    # Test OKS calculations
    oks_perfect = analysis._calculate_keypoint_oks([[10, 20, 1]], [[10, 20, 1]], 10000)
    assert oks_perfect == 1.0, f"Perfect OKS should be 1.0, got {oks_perfect}"
    
    oks_empty = analysis._calculate_keypoint_oks([], [], 10000)
    assert oks_empty == 0.0, f"Empty OKS should be 0.0, got {oks_empty}"
    print("✅ OKS calculations correct")
    
    # Test improvement metrics with empty results
    analysis.iteration_results = []
    metrics = analysis._calculate_improvement_metrics()
    assert metrics.initial_error == 0
    assert metrics.final_error == 0
    print("✅ Improvement metrics calculation working")
    
    # Test recommendation generation with empty results
    recommendations = analysis._generate_recommendations()
    assert len(recommendations) == 0
    print("✅ Recommendation generation working")

def test_epoch_scheduler():
    """Test epoch scheduling without ClearML dependency."""
    from omegaconf import DictConfig
    from cv_error_analyzer.validation.training_validator import EpochScheduler
    
    cfg = DictConfig({
        'base_epochs': 50,
        'max_epochs_per_iteration': 100,
        'adaptive_factor': 1.2
    })
    
    scheduler = EpochScheduler(cfg)
    
    # Test initial scheduling
    epochs = scheduler.get_epochs_for_iteration(1)
    assert epochs == 50, f"Initial epochs should be 50, got {epochs}"
    
    # Test adaptive scheduling
    scheduler.performance_history = [1.0, 0.8]  # Improvement
    epochs = scheduler.get_epochs_for_iteration(2, 0.6)
    expected_max = min(int(50 * 1.2), 100)
    assert epochs == expected_max
    print("✅ EpochScheduler working")

def test_early_stopper():
    """Test early stopping logic."""
    from cv_error_analyzer.validation.training_validator import EarlyStopper
    
    stopper = EarlyStopper(patience=3, min_delta=0.01)
    
    # Test epoch limit
    should_stop = stopper.should_stop(0.5, 50, 50)  # current_epoch >= max_epochs
    assert should_stop, "Should stop when epoch limit reached"
    
    # Test improvement tracking
    stopper = EarlyStopper(patience=2, min_delta=0.01)
    assert not stopper.should_stop(0.8, 10, 50)  # First call
    assert not stopper.should_stop(0.7, 11, 50)  # Improvement
    assert not stopper.should_stop(0.69, 12, 50)  # Small improvement
    assert stopper.should_stop(0.69, 13, 50)  # Patience exceeded
    print("✅ EarlyStopper working")

def main():
    """Run all validation tests."""
    print("🚀 Starting core functionality validation...")
    print("=" * 50)
    
    try:
        test_types()
        test_basic_analysis()
        test_epoch_scheduler()
        test_early_stopper()
        
        print("=" * 50)
        print("✅ ALL CORE FUNCTIONALITY VALIDATED SUCCESSFULLY!")
        print("📋 Summary:")
        print("  • Type definitions working")
        print("  • IoU and OKS calculations accurate")
        print("  • Improvement metrics calculation working")
        print("  • Epoch scheduling logic working")
        print("  • Early stopping logic working")
        print("  • Recommendation system initialized")
        
        return 0
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())