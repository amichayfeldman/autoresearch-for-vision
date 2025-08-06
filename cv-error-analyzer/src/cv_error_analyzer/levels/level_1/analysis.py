"""Basic Level Analysis Implementation - Level 1."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import cv2
from tqdm import tqdm
from omegaconf import DictConfig
from ultralytics import YOLO
import albumentations as A
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional dependency

from ...utils.types import (
    AnalysisResult, IterationResult, SampleResult, ErrorMetrics,
    Recommendation, RecommendationType, ImprovementMetrics, DatasetInfo
)

logger = logging.getLogger(__name__)


class BasicIterativeAnalysis:
    """
    Basic Level Error Analysis (Level 1).
    
    Implements predefined iterative analysis with automatic improvement tracking,
    saturation detection, and recommendation generation based on reference implementations.
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.iterations_completed = 0
        self.iteration_results: List[IterationResult] = []
        self.model: Optional[YOLO] = None
        
        # Analysis parameters from config
        self.max_iterations = cfg.max_iterations
        self.saturation_threshold = cfg.saturation_threshold
        self.improvement_threshold = cfg.improvement_threshold
        self.error_percentile = cfg.get("error_percentile", cfg.get("percentile_threshold", 75))
        
        # Augmentation pipeline (semantic-only, following reference)
        self.augmentation = A.Compose([
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
                p=1.0
            )
        ])
        
        logger.info(f"Initialized BasicIterativeAnalysis with {self.max_iterations} max iterations")
    
    def run_analysis(self, model_path: str, dataset_info: DatasetInfo,
                     experiment_config: Dict[str, Any]) -> AnalysisResult:
        """
        Run complete basic level analysis with predefined iterations.
        
        Args:
            model_path: Path to YOLO model
            dataset_info: ClearML dataset information
            experiment_config: Experiment configuration
            
        Returns:
            Complete analysis results with recommendations
        """
        logger.info("Starting basic level iterative analysis")
        
        # Initialize analysis
        self.model = YOLO(model_path)
        imgsz = experiment_config["train"]["imgsz"]
        
        # Run iteration loop
        self._execute_iteration_loop(dataset_info, imgsz)
        
        # Compile final results
        return self._compile_analysis_results()
    
    def _execute_iteration_loop(self, dataset_info: DatasetInfo, imgsz: int) -> None:
        """Execute the main iteration loop with saturation checking."""
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"Running iteration {iteration}/{self.max_iterations}")
            
            iteration_result = self._run_single_iteration(
                dataset_info, imgsz, iteration
            )
            
            self.iteration_results.append(iteration_result)
            
            # Check for saturation
            if self._check_saturation(iteration):
                logger.info(f"Saturation detected at iteration {iteration}")
                iteration_result.reached_saturation = True
                break
    
    def _compile_analysis_results(self) -> AnalysisResult:
        """Compile final analysis results with metrics and recommendations."""
        # Calculate improvement metrics
        improvement_metrics = self._calculate_improvement_metrics()
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Create plots and track if successful
        plot_filename = self._create_improvement_plots()
        
        return AnalysisResult(
            level=1,
            iterations=self.iteration_results,
            improvement_metrics=improvement_metrics,
            final_recommendations=recommendations,
            reached_saturation=any(r.reached_saturation for r in self.iteration_results),
            metadata={
                "total_iterations": len(self.iteration_results),
                "analysis_config": dict(self.cfg),
                "error_percentile": self.error_percentile,
                "plots_generated": plot_filename is not None,
                "plot_filename": plot_filename
            }
        )
    
    def _run_single_iteration(self, dataset_info: DatasetInfo, imgsz: int,
                              iteration: int) -> IterationResult:
        """
        Run analysis for a single iteration across train and val sets.
        
        Args:
            dataset_info: Dataset information
            imgsz: Image size for model input
            iteration: Current iteration number
            
        Returns:
            Results for this iteration
        """
        all_samples = []
        
        # Process both train and val splits
        splits_to_analyze = ["train", "val"] if "val" in dataset_info.splits else ["train"]
        
        for split in splits_to_analyze:
            if split not in dataset_info.image_files:
                logger.warning(f"Split {split} not found in dataset")
                continue
            
            split_samples = self._analyze_split(
                dataset_info, split, imgsz, iteration
            )
            all_samples.extend(split_samples)
        
        # Sort by total error (descending)
        all_samples.sort(key=lambda x: x.error_metrics.total_error, reverse=True)
        
        # Add ranking
        for rank, sample in enumerate(all_samples, 1):
            sample.rank = rank
        
        # Calculate metrics summary using configurable percentile
        if all_samples:
            errors = [s.error_metrics.total_error for s in all_samples]
            percentile_error = float(np.percentile(errors, self.error_percentile))
        else:
            percentile_error = 0.0
        
        metrics_summary = self._calculate_metrics_summary(all_samples)
        
        # Calculate improvement from previous iteration
        improvement = None
        if len(self.iteration_results) > 0:
            prev_percentile = self.iteration_results[-1].percentile_75_error
            improvement = (prev_percentile - percentile_error) / prev_percentile if prev_percentile > 0 else 0.0
        
        return IterationResult(
            iteration=iteration,
            samples=all_samples,
            metrics_summary=metrics_summary,
            percentile_75_error=percentile_error,  # Now uses configurable percentile
            improvement_from_previous=improvement
        )
    
    def _analyze_split(self, dataset_info: DatasetInfo, split: str, imgsz: int,
                       iteration: int) -> List[SampleResult]:
        """Analyze samples in a single dataset split."""
        logger.info(f"Analyzing {split} split for iteration {iteration}")
        
        image_paths = dataset_info.image_files[split]
        samples = []
        
        for image_path in tqdm(image_paths, desc=f"Processing {split}"):
            try:
                sample_result = self._analyze_single_sample(
                    image_path, dataset_info.num_keypoints, imgsz
                )
                if sample_result:
                    samples.append(sample_result)
            except Exception as e:
                logger.warning(f"Error processing {image_path}: {e}")
                continue
        
        return samples
    
    def _analyze_single_sample(self, image_path: str, num_keypoints: int,
                               imgsz: int) -> Optional[SampleResult]:
        """
        Analyze a single image sample following reference implementation pattern.
        
        Based on naive_error_analysis.py:471-510
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (imgsz, imgsz))
        
        # Load ground truth
        ground_truth = self._load_ground_truth(image_path, num_keypoints, imgsz)
        if not ground_truth:
            return None
        
        # Use pure analysis function for better testability
        return self._analyze_sample_data(image, ground_truth, image_path)
    
    def _analyze_sample_data(self, image: np.ndarray, ground_truth: Dict[str, List],
                             image_path: str) -> SampleResult:
        """
        Pure function to analyze sample data without file I/O dependencies.
        More testable version of sample analysis.
        
        Args:
            image: Preprocessed image as numpy array
            ground_truth: Ground truth annotations
            image_path: File path for result metadata
            
        Returns:
            Sample analysis result
        """
        # Process ORIGINAL image
        pred_original = self.model(image, verbose=False)[0]
        
        # Process AUGMENTED image
        aug_image = self.augmentation(image=image)['image']
        pred_augmented = self.model(aug_image, verbose=False)[0]
        
        # Compare predictions and calculate errors
        consistency_errors = self._compare_prediction_errors(
            pred_augmented, pred_original, ground_truth
        )
        
        return SampleResult(
            file_path=image_path,
            error_metrics=ErrorMetrics(**consistency_errors),
            original_image=image,
            augmented_image=aug_image,
            augmentation_name="ColorJitter"
        )
    
    def _load_ground_truth(self, image_path: str, num_keypoints: int,
                           imgsz: int) -> Optional[Dict[str, List]]:
        """
        Load ground truth from YOLO label file.
        
        Based on naive_error_analysis.py:132-203
        """
        label_path = image_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
        if not Path(label_path).exists():
            return None
        
        try:
            with open(label_path, 'r') as f:
                lines = [line.strip().split() for line in f if line.strip()]
            
            if not lines:
                return None
            
            lines = [[float(x) for x in vals] for vals in lines]
            arr = np.array(lines, dtype=float)
            
            # Convert bbox from normalized to absolute coordinates
            xc, yc, bw, bh = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
            
            # Clip values to valid range
            xc = np.clip(xc, 0.0, 1.0)
            yc = np.clip(yc, 0.0, 1.0)
            bw = np.clip(bw, 0.0, 1.0)
            bh = np.clip(bh, 0.0, 1.0)
            
            x1 = xc * imgsz
            y1 = yc * imgsz
            x2 = (xc + bw) * imgsz
            y2 = (yc + bh) * imgsz
            
            bboxes = np.stack([x1, y1, x2, y2], axis=1).tolist()
            
            # Convert keypoints
            kps = arr[:, 5:5+num_keypoints*3].reshape(-1, num_keypoints, 3)
            kps[..., 0] *= imgsz  # x coordinates
            kps[..., 1] *= imgsz  # y coordinates
            keypoints = kps.tolist()
            
            return {'bboxes': bboxes, 'keypoints': keypoints}
            
        except Exception as e:
            logger.warning(f"Error loading ground truth for {image_path}: {e}")
            return None
    
    def _compare_prediction_errors(self, pred_aug, pred_orig, ground_truth) -> Dict[str, float]:
        """
        Compare predictions between augmented and original images.
        
        Based on naive_error_analysis.py:333-412
        """
        def extract_predictions(pred):
            bboxes, keypoints, classes = [], [], []
            if pred.boxes is not None:
                bboxes = pred.boxes.xyxy.cpu().numpy().tolist()
                classes = pred.boxes.cls.cpu().numpy().astype(int).tolist()
            if pred.keypoints is not None:
                kpts = pred.keypoints.data.cpu().numpy()
                keypoints = [k.tolist() for k in kpts]
            return bboxes, keypoints, classes
        
        bboxes_o, kpts_o, cls_o = extract_predictions(pred_orig)
        bboxes_a, kpts_a, cls_a = extract_predictions(pred_aug)
        
        if not bboxes_o or not bboxes_a:
            return {
                'bbox_iou_error': 1.0,
                'keypoint_oks_error': 1.0,
                'classification_f1_error': 1.0,
                'total_error': 3.0,
                'num_origin': len(bboxes_o),
                'num_augmented': len(bboxes_a),
                'num_matched': 0
            }
        
        # Simple matching for consistency analysis
        bbox_errors, kpt_errors, cls_errors = [], [], []
        matches = min(len(bboxes_o), len(bboxes_a))
        
        for i in range(matches):
            # BBox IoU error
            iou = self._calculate_bbox_iou(bboxes_o[i], bboxes_a[i])
            bbox_errors.append(1.0 - iou)
            
            # Keypoint OKS error
            if i < len(kpts_o) and i < len(kpts_a) and kpts_o[i] and kpts_a[i]:
                area = max(0, (bboxes_o[i][2] - bboxes_o[i][0]) * (bboxes_o[i][3] - bboxes_o[i][1]))
                oks = self._calculate_keypoint_oks(kpts_o[i], kpts_a[i], area)
                kpt_errors.append(1.0 - oks)
            else:
                kpt_errors.append(1.0)
            
            # Classification error
            if i < len(cls_o) and i < len(cls_a):
                cls_errors.append(0.0 if cls_o[i] == cls_a[i] else 1.0)
            else:
                cls_errors.append(1.0)
        
        return {
            'bbox_iou_error': float(np.mean(bbox_errors)) if bbox_errors else 1.0,
            'keypoint_oks_error': float(np.mean(kpt_errors)) if kpt_errors else 1.0,
            'classification_f1_error': float(np.mean(cls_errors)) if cls_errors else 1.0,
            'total_error': float(np.mean(bbox_errors + kpt_errors + cls_errors)) if bbox_errors else 3.0,
            'num_origin': len(bboxes_o),
            'num_augmented': len(bboxes_a),
            'num_matched': matches
        }
    
    def _calculate_bbox_iou(self, box_a: List[float], box_b: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])
        
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        box_a_area = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        box_b_area = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union_area = box_a_area + box_b_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _calculate_keypoint_oks(self, kpts_a: List[List[float]], kpts_b: List[List[float]],
                                bbox_area: float) -> float:
        """Calculate Object Keypoint Similarity (OKS)."""
        if not kpts_a or not kpts_b:
            return 0.0
        
        kpts_a = np.array(kpts_a)
        kpts_b = np.array(kpts_b)
        
        min_kpts = min(len(kpts_a), len(kpts_b))
        if min_kpts == 0:
            return 0.0
        
        kpts_a = kpts_a[:min_kpts]
        kpts_b = kpts_b[:min_kpts]
        
        # Standard COCO keypoint sigmas
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89
        ][:min_kpts]) / 10.0
        
        dx = kpts_a[:, 0] - kpts_b[:, 0]
        dy = kpts_a[:, 1] - kpts_b[:, 1]
        d_squared = dx**2 + dy**2
        
        scale = bbox_area if bbox_area > 0 else 640 * 640
        oks_per_keypoint = np.exp(-d_squared / (2 * scale * sigmas**2))
        
        # Consider only visible keypoints
        visible = kpts_b[:, 2] > 0
        if visible.sum() == 0:
            return 0.0
        
        return oks_per_keypoint[visible].mean()
    
    def _calculate_metrics_summary(self, samples: List[SampleResult]) -> Dict[str, float]:
        """Calculate summary statistics for iteration results."""
        if not samples:
            return {}
        
        total_errors = [s.error_metrics.total_error for s in samples]
        bbox_errors = [s.error_metrics.bbox_iou_error for s in samples]
        kpt_errors = [s.error_metrics.keypoint_oks_error for s in samples]
        cls_errors = [s.error_metrics.classification_f1_error for s in samples]
        
        return {
            "mean_total_error": float(np.mean(total_errors)),
            "std_total_error": float(np.std(total_errors)),
            "median_total_error": float(np.median(total_errors)),
            "p75_total_error": float(np.percentile(total_errors, 75)),
            f"p{self.error_percentile}_total_error": float(np.percentile(total_errors, self.error_percentile)),
            "p90_total_error": float(np.percentile(total_errors, 90)),
            "mean_bbox_error": float(np.mean(bbox_errors)),
            "mean_keypoint_error": float(np.mean(kpt_errors)),
            "mean_classification_error": float(np.mean(cls_errors)),
            "num_samples": len(samples),
            "configured_percentile": self.error_percentile
        }
    
    def _check_saturation(self, current_iteration: int) -> bool:
        """Check if improvement has saturated."""
        if len(self.iteration_results) < 3:
            return False
        
        # Check recent improvements
        recent_improvements = [
            r.improvement_from_previous for r in self.iteration_results[-2:]
            if r.improvement_from_previous is not None
        ]
        
        if not recent_improvements:
            return False
        
        # Saturation if recent improvements are below threshold
        avg_recent_improvement = np.mean(recent_improvements)
        return avg_recent_improvement < self.saturation_threshold
    
    def _calculate_improvement_metrics(self) -> ImprovementMetrics:
        """Calculate overall improvement metrics across iterations."""
        if not self.iteration_results:
            return ImprovementMetrics(0, 0, 0, 0, 0)
        
        initial_error = self.iteration_results[0].percentile_75_error
        final_error = self.iteration_results[-1].percentile_75_error
        
        total_improvement = (initial_error - final_error) / initial_error if initial_error > 0 else 0
        improvement_rate = total_improvement / len(self.iteration_results) if len(self.iteration_results) > 0 else 0
        
        # Final improvement rate (last iteration)
        final_improvement_rate = (
            self.iteration_results[-1].improvement_from_previous 
            if self.iteration_results[-1].improvement_from_previous is not None else 0
        )
        
        iterations_to_saturation = None
        for i, result in enumerate(self.iteration_results):
            if result.reached_saturation:
                iterations_to_saturation = i + 1
                break
        
        return ImprovementMetrics(
            initial_error=initial_error,
            final_error=final_error,
            total_improvement=total_improvement,
            improvement_rate=improvement_rate,
            final_improvement_rate=final_improvement_rate,
            iterations_to_saturation=iterations_to_saturation
        )
    
    def _generate_recommendations(self) -> List[Recommendation]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if not self.iteration_results:
            return recommendations
        
        # Analyze error patterns
        final_result = self.iteration_results[-1]
        if final_result.samples:
            # Augmentation recommendations based on consistency errors
            avg_metrics = final_result.metrics_summary
            
            if avg_metrics.get("mean_bbox_error", 0) > 0.3:
                recommendations.append(Recommendation(
                    recommendation_type=RecommendationType.AUGMENTATION,
                    confidence=0.8,
                    parameters={
                        "geometric_augmentations": [
                            "RandomRotate90", "HorizontalFlip", "ShiftScaleRotate"
                        ],
                        "strength": 0.3
                    },
                    description="High bbox inconsistency suggests need for geometric augmentations"
                ))
            
            if avg_metrics.get("mean_keypoint_error", 0) > 0.4:
                recommendations.append(Recommendation(
                    recommendation_type=RecommendationType.TRAINING,
                    confidence=0.7,
                    parameters={
                        "keypoint_loss_weight": 2.0,
                        "pose_loss_gain": 1.5
                    },
                    description="High keypoint error suggests increasing pose loss weight"
                ))
        
        return recommendations
    
    def _create_improvement_plots(self) -> Optional[str]:
        """
        Create improvement visualization plots and return filename if saved.
        
        Returns:
            Filename of saved plot or None if no plot created
        """
        if len(self.iteration_results) < 2:
            return None
        
        # Create plot figure
        plot_figure = self._generate_improvement_plot_figure()
        
        # Save plot and return filename
        filename = 'level_1_improvement_analysis.png'
        plot_figure.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(plot_figure)
        
        logger.info(f"Improvement plots saved as '{filename}'")
        return filename
    
    def _generate_improvement_plot_figure(self):
        """
        Generate improvement plot figure without side effects.
        Pure function that returns matplotlib figure object.
        
        Returns:
            matplotlib Figure object with improvement plots
        """
        # Extract data for plotting
        iterations = [r.iteration for r in self.iteration_results]
        percentile_errors = [r.percentile_75_error for r in self.iteration_results]
        improvements = [
            r.improvement_from_previous or 0 
            for r in self.iteration_results[1:]
        ]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Configured percentile error over iterations
        ax1.plot(iterations, percentile_errors, marker='o', linewidth=2, markersize=6)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel(f'{self.error_percentile}th Percentile Error')
        ax1.set_title('Error Reduction Over Iterations')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement rate per iteration
        if improvements:
            ax2.bar(iterations[1:], improvements, alpha=0.7, color='green')
            ax2.axhline(y=self.saturation_threshold, color='red', linestyle='--', 
                       label=f'Saturation Threshold ({self.saturation_threshold})')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Improvement Rate')
            ax2.set_title('Improvement Rate per Iteration')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig