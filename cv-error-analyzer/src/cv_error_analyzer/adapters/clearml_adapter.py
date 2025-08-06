"""ClearML integration adapter for dataset and experiment management."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import yaml
import pandas as pd
from omegaconf import DictConfig
from clearml import Task, Dataset

from ..utils.types import DatasetInfo

logger = logging.getLogger(__name__)


class ClearMLAdapter:
    """
    Adapter for ClearML integration providing dataset loading and experiment tracking.
    
    Based on reference implementations from naive_error_analysis.py.
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.project_name = cfg.get("dataset_project", "yolo_pose_manual_annotations_dataset")
        logger.info(f"Initialized ClearMLAdapter for project: {self.project_name}")
    
    def load_experiment_config(self, model_path: str) -> Dict[str, Any]:
        """
        Load experiment configuration from model path.
        
        Based on naive_error_analysis.py:45-68
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dictionary containing experiment configuration
        """
        logger.info(f"Loading experiment config for model: {model_path}")
        
        # Extract experiment name from model path
        model_path = Path(model_path)
        experiment_name = model_path.parent.parent.name
        
        try:
            task = Task.get_task(
                project_name="YOLO-Pose",
                task_name=experiment_name
            )
            config = yaml.safe_load(task.get_configuration_object(name="OmegaConf"))
            
            logger.info(f"Successfully loaded config for experiment: {experiment_name}")
            return config
            
        except Exception as e:
            logger.warning(f"Could not load experiment config: {e}")
            # Return default config
            return {
                "train": {
                    "imgsz": 640,
                    "epochs": 100,
                    "batch": 16
                }
            }
    
    def load_dataset(self, version: str) -> DatasetInfo:
        """
        Load dataset from ClearML and parse image lists and annotation files.
        
        Based on naive_error_analysis.py:72-129
        
        Args:
            version: Dataset version to load
            
        Returns:
            DatasetInfo object with all dataset information
        """
        logger.info(f"Loading dataset {self.project_name} version {version}")
        
        try:
            dataset = Dataset.get(
                dataset_project=self.project_name,
                dataset_name=self.project_name,
                dataset_version=version
            )
            dataset_path = dataset.get_local_copy()
            logger.info(f"Dataset downloaded to: {dataset_path}")
            
        except Exception as e:
            logger.error(f"Failed to load ClearML dataset: {e}")
            raise
        
        # Load image lists and annotations for each split
        image_files = {}
        annotation_files = {}
        
        dataset_dir = Path(dataset_path)
        image_txt_files = list(dataset_dir.glob("*_images.txt"))
        
        for txt_file in image_txt_files:
            set_name = txt_file.name.replace("_images.txt", "")
            logger.info(f"Loading {set_name} split...")
            
            # Load image paths
            with open(txt_file, 'r') as f:
                image_paths = [line.strip() for line in f if line.strip()]
            image_files[set_name] = image_paths
            
            # Load corresponding annotations CSV
            annotation_file = dataset_dir / f"{set_name}_annotations.csv"
            if annotation_file.exists():
                annotations_df = pd.read_csv(annotation_file)
                annotation_files[set_name] = annotations_df
                logger.info(f"Loaded {len(annotations_df)} annotations for {set_name}")
            else:
                logger.warning(f"Annotation file not found: {annotation_file}")
                annotation_files[set_name] = pd.DataFrame()
        
        # Get keypoint information from dataset config
        dataset_yaml = dataset_dir / "dataset.yaml"
        num_keypoints = 17  # Default
        if dataset_yaml.exists():
            with open(dataset_yaml, 'r') as f:
                dataset_cfg = yaml.safe_load(f)
                num_keypoints = dataset_cfg.get("kpt_shape", [17])[0]
        
        splits = list(image_files.keys())
        logger.info(f"Loaded {len(splits)} splits: {splits}")
        
        return DatasetInfo(
            dataset_path=str(dataset_path),
            image_files=image_files,
            annotation_files=annotation_files,
            num_keypoints=num_keypoints,
            splits=splits
        )
    
    def create_training_task(self, base_config: Dict[str, Any], 
                             recommendations: List[Any],
                             iteration: int) -> Task:
        """
        Create a new ClearML task for training iteration.
        
        Args:
            base_config: Base experiment configuration
            recommendations: Analysis recommendations to apply
            iteration: Training iteration number
            
        Returns:
            ClearML Task object
        """
        task_name = f"cv_error_analyzer_iteration_{iteration}"
        
        task = Task.init(
            project_name="CV-Error-Analyzer",
            task_name=task_name,
            tags=["automated", "error-analysis", f"iteration-{iteration}"]
        )
        
        # Apply recommendations to config
        updated_config = self._apply_recommendations_to_config(base_config, recommendations)
        task.connect_configuration(updated_config)
        
        logger.info(f"Created training task: {task_name}")
        return task
    
    def get_training_metrics(self, task: Task) -> Dict[str, float]:
        """
        Retrieve training metrics from ClearML task.
        
        Args:
            task: ClearML Task object
            
        Returns:
            Dictionary of training metrics
        """
        try:
            # Get scalar metrics from task
            metrics = {}
            
            # Common YOLO training metrics
            metric_names = [
                "train/box_loss",
                "train/pose_loss", 
                "train/dfl_loss",
                "val/box_loss",
                "val/pose_loss",
                "val/dfl_loss",
                "metrics/mAP50",
                "metrics/mAP50-95"
            ]
            
            for metric_name in metric_names:
                try:
                    series = task.get_last_scalar_metrics()[metric_name]
                    if series:
                        metrics[metric_name.replace("/", "_")] = series.get("value", 0.0)
                except (KeyError, AttributeError):
                    continue
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Could not retrieve metrics from task: {e}")
            return {}
    
    def _apply_recommendations_to_config(self, base_config: Dict[str, Any],
                                         recommendations: List[Any]) -> Dict[str, Any]:
        """Apply recommendations to training configuration."""
        config = base_config.copy()
        
        for rec in recommendations:
            if hasattr(rec, 'recommendation_type') and hasattr(rec, 'parameters'):
                if rec.recommendation_type == "augmentation":
                    if "augmentation" not in config:
                        config["augmentation"] = {}
                    config["augmentation"].update(rec.parameters)
                
                elif rec.recommendation_type == "training":
                    config.update(rec.parameters)
        
        return config