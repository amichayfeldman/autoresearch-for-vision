# CV Error Analyzer

Automated agentic AI system for computer vision model error analysis and improvement.

## Features

- **Hierarchical Analysis Levels**: Basic to advanced error analysis with automatic level progression
- **Iterative Training**: Automatic training execution with configurable epoch constraints
- **ClearML Integration**: Experiment tracking and metrics validation
- **Modular Design**: Plugin architecture for extensible analysis capabilities
- **Configuration Management**: Structured config with Hydra defaults lists

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
cv-error-analyzer model_path=/path/to/model.pt dataset_version=1.2
```

## Configuration

The system uses Hydra configuration management with hierarchical defaults:

```yaml
defaults:
  - analysis_level: basic
  - training_execution: epoch_constrained
  - iterative_improvement: default
```

## Usage

### Basic Analysis

Run error analysis with default configuration:

```bash
cv-error-analyzer model_path=/path/to/model.pt dataset_version=1.2
```

### Programmatic Usage

```python
from cv_error_analyzer import ErrorAnalyzerEngine
from omegaconf import DictConfig
import hydra

# Initialize with config
with hydra.initialize(config_path="config"):
    cfg = hydra.compose(config_name="config", 
                       overrides=["model_path=/path/to/model.pt", 
                                "dataset_version=1.2"])
    
    engine = ErrorAnalyzerEngine(cfg)
    results = engine.analyze(cfg.model_path, cfg.dataset_version)
    
    print(f"Analysis completed with {len(results['final_recommendations'])} recommendations")
```

### Custom Configuration

Adjust training parameters:

```bash
cv-error-analyzer \
  model_path=/path/to/model.pt \
  dataset_version=1.2 \
  training_execution.max_epochs_per_iteration=50 \
  iterative_improvement.max_iterations=10 \
  analysis_level.saturation_threshold=0.005
```

### Analysis Levels

- **Level 1 (Basic)**: Semantic augmentation analysis with iterative improvement
- **Future Levels**: Geometric augmentations, semantic segmentation integration (SAM2)

### Output Structure

```python
results = {
    "analysis_results": {
        "level_1": AnalysisResult(...)
    },
    "training_history": [TrainingResult(...)],
    "final_recommendations": [Recommendation(...)],
    "metadata": {
        "levels_completed": [1],
        "total_training_iterations": 3,
        "final_model_path": "/path/to/improved/model.pt"
    }
}
```