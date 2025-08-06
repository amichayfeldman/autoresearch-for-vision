"""Core Error Analyzer Engine with hierarchical level processing."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from omegaconf import DictConfig
import hydra
# from hydra import compose, initialize  # Not needed in engine

from ..levels.level_1.analysis import BasicIterativeAnalysis
from ..validation.training_validator import TrainingValidator
from ..adapters.clearml_adapter import ClearMLAdapter
from ..utils.types import AnalysisResult, TrainingResult, Recommendation

logger = logging.getLogger(__name__)

# Type aliases for Python 3.8 compatibility
AnalysisLevel = int
LevelName = str

class ErrorAnalyzerEngine:
    """
    Main engine orchestrating multi-level error analysis and training validation.
    
    Implements hierarchical analysis progression from basic (level 1) to advanced levels,
    with automatic iteration and training validation feedback loops.
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        # Validate configuration
        self._validate_configuration(cfg)
        
        self.clearml_adapter = ClearMLAdapter(cfg.clearml)
        self.training_validator = TrainingValidator(cfg.training_validation)
        
        # Level registry - expandable for future analysis levels
        self.analysis_levels: Dict[AnalysisLevel, Any] = {
            1: BasicIterativeAnalysis(cfg.analysis_level)
        }
        
        # Track analysis state
        self.current_level = cfg.analysis_level.start_level
        self.max_level = cfg.analysis_level.max_level
        self.level_results: Dict[AnalysisLevel, List[AnalysisResult]] = {}
        
        logger.info(f"Initialized ErrorAnalyzerEngine with levels {list(self.analysis_levels.keys())}")
    
    def _validate_configuration(self, cfg: DictConfig) -> None:
        """Validate critical configuration parameters."""
        required_fields = [
            "analysis_level.max_iterations",
            "analysis_level.saturation_threshold", 
            "analysis_level.improvement_threshold",
            "training_validation.validation_metric",
            "clearml.dataset_project"
        ]
        
        for field in required_fields:
            if not self._has_nested_key(cfg, field):
                raise ValueError(f"Required configuration field missing: {field}")
        
        # Validate percentile range
        error_percentile = cfg.analysis_level.get("error_percentile", 75)
        if not (50 <= error_percentile <= 95):
            raise ValueError(f"error_percentile must be between 50-95, got {error_percentile}")
        
        # Validate iteration limits
        max_iterations = cfg.analysis_level.max_iterations
        if not (1 <= max_iterations <= 100):
            raise ValueError(f"max_iterations must be between 1-100, got {max_iterations}")
        
        logger.info("✅ Configuration validation passed")
    
    def _has_nested_key(self, cfg: DictConfig, key_path: str) -> bool:
        """Check if nested configuration key exists."""
        keys = key_path.split(".")
        current = cfg
        for key in keys:
            if not hasattr(current, key) or getattr(current, key) is None:
                return False
            current = getattr(current, key)
        return True
    
    def analyze(self, model_path: str, dataset_version: str) -> Dict[str, Any]:
        """
        Main analysis entry point with level loop progression.
        
        Args:
            model_path: Path to trained YOLO model
            dataset_version: ClearML dataset version
            
        Returns:
            Complete analysis results with recommendations and training validation
        """
        logger.info(f"Starting analysis for model: {model_path}, dataset: {dataset_version}")
        
        # Load experiment configuration and dataset
        experiment_config = self.clearml_adapter.load_experiment_config(model_path)
        dataset_info = self.clearml_adapter.load_dataset(dataset_version)
        
        analysis_results = {}
        training_history = []
        
        # Level loop progression
        for level in range(self.current_level, self.max_level + 1):
            if level not in self.analysis_levels:
                logger.warning(f"Analysis level {level} not implemented, skipping")
                continue
            
            logger.info(f"Starting analysis level {level}")
            level_analyzer = self.analysis_levels[level]
            
            # Run analysis level
            level_result = level_analyzer.run_analysis(
                model_path=model_path,
                dataset_info=dataset_info,
                experiment_config=experiment_config
            )
            
            self.level_results[level] = level_result.iterations
            analysis_results[f"level_{level}"] = level_result
            
            # Check if we should progress to next level
            if not self._should_progress_to_next_level(level, level_result):
                logger.info(f"Stopping at level {level} - progression criteria not met")
                break
            
            # Execute training validation if recommendations available
            if level_result.final_recommendations:
                logger.info(f"Executing training validation for level {level}")
                training_result = self._execute_training_iteration(
                    level_result.final_recommendations,
                    experiment_config,
                    iteration=len(training_history) + 1
                )
                training_history.append(training_result)
                
                # Update model path for next level if training improved
                if training_result.improved:
                    model_path = training_result.new_model_path
                    logger.info(f"Updated model path to: {model_path}")
        
        return {
            "analysis_results": analysis_results,
            "training_history": training_history,
            "final_recommendations": self._compile_final_recommendations(),
            "metadata": {
                "levels_completed": list(self.level_results.keys()),
                "total_training_iterations": len(training_history),
                "final_model_path": model_path
            }
        }
    
    def _should_progress_to_next_level(self, current_level: AnalysisLevel, 
                                       result: AnalysisResult) -> bool:
        """
        Determine if analysis should progress to next level based on current results.
        
        Args:
            current_level: Current analysis level
            result: Results from current level analysis
            
        Returns:
            True if should progress to next level
        """
        # Level 1 specific progression criteria
        if current_level == 1:
            # Progress if we found substantial improvements or reached saturation
            if result.reached_saturation:
                logger.info("Level 1 reached saturation, progressing to level 2")
                return True
            
            # Progress if improvement rate is high enough
            if result.improvement_metrics.get("final_improvement_rate", 0) > 0.1:
                logger.info("Level 1 shows high improvement rate, progressing to level 2")
                return True
        
        # Default: don't progress (conservative approach)
        return False
    
    def _execute_training_iteration(self, recommendations: List[Recommendation],
                                    experiment_config: Dict[str, Any],
                                    iteration: int) -> TrainingResult:
        """
        Execute training iteration based on recommendations.
        
        Args:
            recommendations: Analysis recommendations to implement
            experiment_config: Original experiment configuration
            iteration: Training iteration number
            
        Returns:
            Training results with validation metrics
        """
        logger.info(f"Executing training iteration {iteration}")
        
        # Apply recommendations to create new training config (inlined)
        new_config = experiment_config.copy()
        for rec in recommendations:
            if rec.recommendation_type == "augmentation":
                if "augmentation" not in new_config:
                    new_config["augmentation"] = {}
                new_config["augmentation"].update(rec.parameters)
            elif rec.recommendation_type == "training":
                new_config.update(rec.parameters)
        
        # Execute training with epoch constraints
        training_result = self.training_validator.execute_training_iteration(
            iteration=iteration,
            recommendations=recommendations,
            base_config=new_config
        )
        
        # Validate improvements
        validation_result = self.training_validator.validate_training_improvement(
            training_result, experiment_config
        )
        
        training_result.validation_result = validation_result
        return training_result
    
    # Removed _apply_recommendations_to_config - inlined for single use
    
    def _compile_final_recommendations(self) -> List[Recommendation]:
        """Compile final recommendations from all analysis levels."""
        final_recs = []
        
        for level, results in self.level_results.items():
            if results and results[-1].final_recommendations:
                final_recs.extend(results[-1].final_recommendations)
        
        return final_recs


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """CLI entry point for error analysis."""
    engine = ErrorAnalyzerEngine(cfg)
    
    results = engine.analyze(
        model_path=cfg.model_path,
        dataset_version=cfg.dataset_version
    )
    
    logger.info("Analysis completed successfully")
    logger.info(f"Levels completed: {results['metadata']['levels_completed']}")
    logger.info(f"Training iterations: {results['metadata']['total_training_iterations']}")


if __name__ == "__main__":
    main()