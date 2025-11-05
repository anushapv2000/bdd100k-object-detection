"""
Model Evaluation Pipeline for BDD100k Object Detection.

This script evaluates a trained YOLOv8 model on the validation dataset,
computing comprehensive metrics including mAP, precision, recall, and confusion matrix.

Author: Bosch Assignment - Phase 3
Date: November 2025
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
from tqdm import tqdm
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from metrics import (
    compute_map, 
    compute_precision_recall_f1,
    compute_confusion_matrix,
    compute_ap
)


# BDD100k class names
BDD_CLASSES = [
    'person',
  'rider',
  'car',
  'truck',
  'bus',
  'train',
  'motorcycle',
  'bicycle',
  'traffic light',
  'traffic sign'
]


class BDD100kEvaluator:
    """Evaluator for BDD100k object detection task."""
    
    def __init__(
        self,
        model_path: str,
        data_root: str,
        labels_path: str,
        output_dir: str = "outputs/evaluation",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model weights
            data_root: Root directory of dataset
            labels_path: Path to labels JSON file
            output_dir: Directory to save results
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for matching
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.data_root = Path(data_root)
        self.labels_path = labels_path
        self.output_dir = Path(output_dir)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.num_classes = len(BDD_CLASSES)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "predictions").mkdir(exist_ok=True)
        
        # Load model
        print(f"\n{'='*60}")
        print(f"Loading model from: {model_path}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")
        
        self.model = self._load_model()
        
        # Load ground truth
        print(f"Loading ground truth from: {labels_path}")
        self.ground_truth = self._load_ground_truth()
        print(f"Loaded {len(self.ground_truth)} validation samples\n")
    
    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            model.to(self.device)
            return model
        except ImportError:
            print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR loading model: {e}")
            sys.exit(1)
    
    def _load_ground_truth(self) -> List[Dict]:
        """Load ground truth labels from JSON file."""
        with open(self.labels_path, 'r') as f:
            labels = json.load(f)
        
        # The validation labels JSON already contains only validation samples
        # No need to filter further
        return labels
    
    def _parse_gt_boxes(self, labels: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse ground truth boxes from label data.
        
        Returns:
            boxes: [N, 4] in format [x1, y1, x2, y2]
            labels: [N] class indices
        """
        boxes = []
        class_labels = []
        
        for label in labels:
            if label['category'] in BDD_CLASSES:
                class_id = BDD_CLASSES.index(label['category'])
                box2d = label['box2d']
                
                x1 = box2d['x1']
                y1 = box2d['y1']
                x2 = box2d['x2']
                y2 = box2d['y2']
                
                boxes.append([x1, y1, x2, y2])
                class_labels.append(class_id)
        
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        
        return np.array(boxes, dtype=np.float32), np.array(class_labels, dtype=np.int32)
    
    def run_inference(self, max_images: int = None) -> Tuple[List, List, List]:
        """
        Run inference on validation set.
        
        Args:
            max_images: Maximum number of images to process (None for all)
        
        Returns:
            pred_boxes, pred_scores, pred_labels: Lists of predictions per image
        """
        print(f"\n{'='*60}")
        print(f"Running Inference on Validation Set")
        print(f"{'='*60}\n")
        
        pred_boxes_all = []
        pred_scores_all = []
        pred_labels_all = []
        
        gt_data = self.ground_truth[:max_images] if max_images else self.ground_truth
        
        inference_times = []
        
        for item in tqdm(gt_data, desc="Processing images"):
            img_name = item['name']
            
            # Try multiple possible image paths
            possible_paths = [
                self.data_root / 'bdd100k' / 'images' / '100k' / 'val' / img_name,  # Original BDD100k structure
                self.data_root / 'images' / '100k' / 'val' / img_name,  # Alternative 1
                self.data_root / 'val' / 'images' / img_name,  # Alternative 2
                self.data_root / 'bdd100k_yolo_dataset' / 'val' / 'images' / img_name,  # YOLO format
                Path('../../data_analysis/data/bdd100k_yolo_dataset/val/images') / img_name,  # Relative YOLO path
            ]
            
            img_path = None
            for path in possible_paths:
                if path.exists():
                    img_path = path
                    break
            
            if img_path is None:
                print(f"Warning: Image not found: {img_name}")
                pred_boxes_all.append(np.zeros((0, 4), dtype=np.float32))
                pred_scores_all.append(np.zeros((0,), dtype=np.float32))
                pred_labels_all.append(np.zeros((0,), dtype=np.int32))
                continue
            
            # Run inference
            start_time = time.time()
            results = self.model.predict(
                str(img_path),
                conf=self.conf_threshold,
                iou=0.45,  # NMS IoU threshold
                verbose=False,
                device=self.device
            )
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Parse results
            result = results[0]
            boxes = result.boxes
            
            if len(boxes) > 0:
                pred_boxes = boxes.xyxy.cpu().numpy()  # [N, 4]
                pred_scores = boxes.conf.cpu().numpy()  # [N]
                pred_labels = boxes.cls.cpu().numpy().astype(np.int32)  # [N]
            else:
                pred_boxes = np.zeros((0, 4), dtype=np.float32)
                pred_scores = np.zeros((0,), dtype=np.float32)
                pred_labels = np.zeros((0,), dtype=np.int32)
            
            pred_boxes_all.append(pred_boxes)
            pred_scores_all.append(pred_scores)
            pred_labels_all.append(pred_labels)
        
        # Print inference statistics
        avg_inference_time = np.mean(inference_times) if len(inference_times) > 0 else 0
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"Inference Statistics:")
        print(f"  Total images: {len(gt_data)}")
        print(f"  Average inference time: {avg_inference_time*1000:.2f} ms")
        print(f"  FPS: {fps:.2f}")
        print(f"{'='*60}\n")
        
        return pred_boxes_all, pred_scores_all, pred_labels_all
    
    def evaluate(self, max_images: int = None) -> Dict:
        """
        Run full evaluation pipeline.
        
        Args:
            max_images: Maximum number of images (None for all)
        
        Returns:
            Dictionary with all evaluation metrics
        """
        # Run inference
        pred_boxes, pred_scores, pred_labels = self.run_inference(max_images)
        
        # Prepare ground truth
        gt_boxes = []
        gt_labels = []
        
        gt_data = self.ground_truth[:max_images] if max_images else self.ground_truth
        
        for item in gt_data:
            boxes, labels = self._parse_gt_boxes(item.get('labels', []))
            gt_boxes.append(boxes)
            gt_labels.append(labels)
        
        print(f"\n{'='*60}")
        print(f"Computing Metrics")
        print(f"{'='*60}\n")
        
        # 1. Compute mAP
        print("Computing mAP...")
        map_results = compute_map(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels,
            self.num_classes,
            iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        )
        
        # 2. Compute Precision, Recall, F1
        print("Computing precision, recall, F1...")
        prf_metrics = compute_precision_recall_f1(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels,
            self.num_classes,
            iou_threshold=self.iou_threshold,
            conf_threshold=self.conf_threshold
        )
        
        # 3. Compute Confusion Matrix
        print("Computing confusion matrix...")
        conf_matrix = compute_confusion_matrix(
            pred_labels, gt_labels, pred_boxes, gt_boxes, pred_scores,
            self.num_classes,
            iou_threshold=self.iou_threshold,
            conf_threshold=self.conf_threshold
        )
        
        # 4. Compute per-class AP with precision-recall curves
        print("Computing per-class AP with curves...")
        per_class_curves = {}
        for class_id in range(self.num_classes):
            ap, precision, recall = compute_ap(
                pred_boxes, pred_scores, pred_labels,
                gt_boxes, gt_labels,
                class_id,
                iou_threshold=0.5
            )
            per_class_curves[class_id] = {
                'ap': ap,
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'class_name': BDD_CLASSES[class_id]
            }
        
        # Compile results
        results = {
            'map': map_results,
            'precision_recall_f1': prf_metrics,
            'confusion_matrix': conf_matrix.tolist(),
            'per_class_curves': per_class_curves,
            'config': {
                'conf_threshold': self.conf_threshold,
                'iou_threshold': self.iou_threshold,
                'num_images': len(gt_data),
                'num_classes': self.num_classes,
                'class_names': BDD_CLASSES
            }
        }
        
        # Save results
        self._save_results(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _save_results(self, results: Dict):
        """Save evaluation results to files."""
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        # Convert results
        results_serializable = convert_to_native(results)
        
        # Save full results as JSON
        results_path = self.output_dir / "metrics" / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"\nSaved full results to: {results_path}")
        
        # Save summary metrics as CSV
        import pandas as pd
        
        # Per-class metrics
        per_class_data = []
        for class_id in range(self.num_classes):
            class_name = BDD_CLASSES[class_id]
            metrics = results['precision_recall_f1'][class_id]
            ap = results['per_class_curves'][class_id]['ap']
            
            per_class_data.append({
                'class_id': class_id,
                'class_name': class_name,
                'AP@0.5': ap,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1'],
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'fn': metrics['fn']
            })
        
        df = pd.DataFrame(per_class_data)
        csv_path = self.output_dir / "metrics" / "per_class_metrics.csv"
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"Saved per-class metrics to: {csv_path}")
        
        # Save confusion matrix
        conf_matrix_path = self.output_dir / "metrics" / "confusion_matrix.csv"
        conf_matrix = np.array(results['confusion_matrix'])
        labels = BDD_CLASSES + ['background']
        df_conf = pd.DataFrame(conf_matrix, columns=labels, index=labels)
        df_conf.to_csv(conf_matrix_path, float_format='%.0f')
        print(f"Saved confusion matrix to: {conf_matrix_path}")
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}\n")
        
        # Overall mAP
        map_50 = results['map']['mAP@0_50']
        map_50_95 = results['map'].get('mAP@0_5:0_95', 0)
        
        print(f"Overall Performance:")
        print(f"  mAP@0.5:      {map_50:.4f}")
        print(f"  mAP@0.5:0.95: {map_50_95:.4f}")
        
        # Per-class AP
        print(f"\nPer-Class Average Precision (AP@0.5):")
        print(f"{'Class':<20} {'AP':>8} {'Precision':>10} {'Recall':>10} {'F1':>8}")
        print(f"{'-'*60}")
        
        for class_id in range(self.num_classes):
            class_name = BDD_CLASSES[class_id]
            ap = results['per_class_curves'][class_id]['ap']
            metrics = results['precision_recall_f1'][class_id]
            
            print(f"{class_name:<20} {ap:>8.4f} {metrics['precision']:>10.4f} "
                  f"{metrics['recall']:>10.4f} {metrics['f1']:>8.4f}")
        
        # Mean across classes
        mean_ap = np.mean([results['per_class_curves'][i]['ap'] for i in range(self.num_classes)])
        mean_p = np.mean([results['precision_recall_f1'][i]['precision'] for i in range(self.num_classes)])
        mean_r = np.mean([results['precision_recall_f1'][i]['recall'] for i in range(self.num_classes)])
        mean_f1 = np.mean([results['precision_recall_f1'][i]['f1'] for i in range(self.num_classes)])
        
        print(f"{'-'*60}")
        print(f"{'Mean':<20} {mean_ap:>8.4f} {mean_p:>10.4f} {mean_r:>10.4f} {mean_f1:>8.4f}")
        
        print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BDD100k Object Detection Model")
    
    parser.add_argument(
        '--model',
        type=str,
        default='../yolov8m.pt',
        help='Path to model weights'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='../../data_analysis/data/bdd100k_images_100k',
        help='Root directory of dataset'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default='../../data_analysis/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json',
        help='Path to validation labels JSON'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../outputs/evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.25,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for matching'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device (cpu or cuda)'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum images to evaluate (for testing)'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = BDD100kEvaluator(
        model_path=args.model,
        data_root=args.data_root,
        labels_path=args.labels,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device
    )
    
    # Run evaluation
    results = evaluator.evaluate(max_images=args.max_images)
    
    print("\n✓ Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
