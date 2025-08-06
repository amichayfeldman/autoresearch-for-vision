"""Command-line interface for CV Error Analyzer."""

import logging
import sys
from pathlib import Path
from omegaconf import DictConfig
import hydra
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

from .core.engine import ErrorAnalyzerEngine

logger = logging.getLogger(__name__)


def setup_logging(cfg: DictConfig) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format=cfg.logging.format
    )


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main CLI entry point for CV Error Analyzer.
    
    Args:
        cfg: Hydra configuration object
    """
    setup_logging(cfg)
    
    logger.info("Starting CV Error Analyzer")
    logger.info(f"Configuration: {cfg}")
    
    # Validate required parameters
    if not cfg.get("model_path"):
        logger.error("model_path is required")
        sys.exit(1)
    
    if not cfg.get("dataset_version"):
        logger.error("dataset_version is required") 
        sys.exit(1)
    
    if not Path(cfg.model_path).exists():
        logger.error(f"Model file not found: {cfg.model_path}")
        sys.exit(1)
    
    try:
        # Initialize and run analysis engine
        engine = ErrorAnalyzerEngine(cfg)
        
        results = engine.analyze(
            model_path=cfg.model_path,
            dataset_version=cfg.dataset_version
        )
        
        # Log final results
        metadata = results["metadata"]
        logger.info("="*60)
        logger.info("CV ERROR ANALYZER RESULTS")
        logger.info("="*60)
        logger.info(f"Levels completed: {metadata['levels_completed']}")
        logger.info(f"Training iterations: {metadata['total_training_iterations']}")
        logger.info(f"Final model: {metadata['final_model_path']}")
        
        # Log recommendations
        final_recs = results["final_recommendations"]
        if final_recs:
            logger.info(f"Final recommendations ({len(final_recs)}):")
            for i, rec in enumerate(final_recs, 1):
                logger.info(f"  {i}. {rec.recommendation_type}: {rec.description}")
        
        logger.info("Analysis completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


def run_basic_analysis(model_path: str, dataset_version: str) -> None:
    """
    Convenience function to run basic analysis programmatically.
    
    Args:
        model_path: Path to trained YOLO model
        dataset_version: ClearML dataset version
    """
    with initialize(config_path="../../config"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"model_path={model_path}",
                f"dataset_version={dataset_version}"
            ]
        )
        
        engine = ErrorAnalyzerEngine(cfg)
        return engine.analyze(model_path, dataset_version)


if __name__ == "__main__":
    main()