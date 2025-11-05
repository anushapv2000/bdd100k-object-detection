"""
Qualitative Visualization for BDD100k Object Detection.

This script creates visual comparisons of ground truth vs predictions:
- Side-by-side GT and prediction visualizations
- Overlay visualizations with TP/FP/FN marking
- Success and failure case examples
- Confidence score visualization

Author: Bosch Assignment - Phase 3
Date: November 2025
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from metrics import compute_iou


# BDD100k class names
BDD_CLASSES = [
    'bike', 'bus', 'car', 'motor', 'person', 
    'rider', 'traffic light', 'traffic sign', 'train', 'truck'
]

# Colors for visualization
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0),
    (0, 128, 255), (128, 255, 0)
]

# Colors for TP, FP, FN
TP_COLOR = (0, 255, 0)  # Green
FP_COLOR = (0, 0, 255)  # Red
FN_COLOR = (255, 0, 0)  # Blue


class QualitativeVisualizer:
    """Visualizer for qualitative prediction analysis."""
    
    def __init__(
        self,
        model_path: str,
        data_root: str,
        labels_path: str,
        output_dir: str = "outputs/evaluation/predictions",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize visualizer.
        
        Args:
            model_path: Path to trained model
            data_root: Root directory of dataset
            labels_path: Path to validation labels JSON
            output_dir: Directory to save visualizations
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for matching
            device: Device to run on
        """
        self.model_path = model_path
        self.data_root = Path(data_root)
        self.labels_path = labels_path
        self.output_dir = Path(output_dir)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Create output directories
        (self.output_dir / "success").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "failures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "comparisons").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "overlay").mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"\nLoading model from: {model_path}")
        self.model = self._load_model()
        
        # Load ground truth
        print(f"Loading ground truth from: {labels_path}")
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        self.ground_truth = [item for item in labels if 'val' in item.get('name', '')]
        print(f"Loaded {len(self.ground_truth)} validation samples\n")
    
    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            model.to(self.device)
            return model
        except Exception as e:
            print(f"ERROR loading model: {e}")
            sys.exit(1)
    
    def _parse_gt_boxes(self, labels: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Parse ground truth boxes."""
        boxes = []
        class_labels = []
        
        for label in labels:
            if label['category'] in BDD_CLASSES:
                class_id = BDD_CLASSES.index(label['category'])
                box2d = label['box2d']
                boxes.append([box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']])
                class_labels.append(class_id)
        
        if len(boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0,), dtype=np.int32)
        
        return np.array(boxes, dtype=np.float32), np.array(class_labels, dtype=np.int32)
    
    def draw_boxes(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray, 
                   scores: np.ndarray = None, title: str = "") -> np.ndarray:
        """Draw boxes on image."""
        img = image.copy()
        
        # Add title
        if title:
            cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 0, 0), 2, cv2.LINE_AA)
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            label_idx = int(labels[i])
            color = COLORS[label_idx % len(COLORS)]
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Create label text
            label_text = BDD_CLASSES[label_idx]
            if scores is not None:
                label_text += f" {scores[i]:.2f}"
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # Draw label text
            cv2.putText(img, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def draw_overlay(self, image: np.ndarray, gt_boxes: np.ndarray, gt_labels: np.ndarray,
                    pred_boxes: np.ndarray, pred_labels: np.ndarray, 
                    pred_scores: np.ndarray) -> np.ndarray:
        """Draw overlay with TP/FP/FN marking."""
        img = image.copy()
        
        # Match predictions to GT
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        pred_matched = np.zeros(len(pred_boxes), dtype=bool)
        
        # Match each prediction to best GT
        for pred_idx in range(len(pred_boxes)):
            pred_box = pred_boxes[pred_idx]
            pred_label = pred_labels[pred_idx]
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx in range(len(gt_boxes)):
                if gt_matched[gt_idx]:
                    continue
                if gt_labels[gt_idx] != pred_label:
                    continue
                
                iou = compute_iou(pred_box, gt_boxes[gt_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                # True Positive
                gt_matched[best_gt_idx] = True
                pred_matched[pred_idx] = True
        
        # Draw True Positives (green)
        for pred_idx in range(len(pred_boxes)):
            if pred_matched[pred_idx]:
                x1, y1, x2, y2 = pred_boxes[pred_idx].astype(int)
                label_idx = int(pred_labels[pred_idx])
                
                cv2.rectangle(img, (x1, y1), (x2, y2), TP_COLOR, 2)
                label_text = f"TP: {BDD_CLASSES[label_idx]} {pred_scores[pred_idx]:.2f}"
                
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(img, (x1, y1 - 15), (x1 + w, y1), TP_COLOR, -1)
                cv2.putText(img, label_text, (x1, y1 - 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw False Positives (red)
        for pred_idx in range(len(pred_boxes)):
            if not pred_matched[pred_idx]:
                x1, y1, x2, y2 = pred_boxes[pred_idx].astype(int)
                label_idx = int(pred_labels[pred_idx])
                
                cv2.rectangle(img, (x1, y1), (x2, y2), FP_COLOR, 2)
                label_text = f"FP: {BDD_CLASSES[label_idx]} {pred_scores[pred_idx]:.2f}"
                
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(img, (x1, y1 - 15), (x1 + w, y1), FP_COLOR, -1)
                cv2.putText(img, label_text, (x1, y1 - 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw False Negatives (blue)
        for gt_idx in range(len(gt_boxes)):
            if not gt_matched[gt_idx]:
                x1, y1, x2, y2 = gt_boxes[gt_idx].astype(int)
                label_idx = int(gt_labels[gt_idx])
                
                cv2.rectangle(img, (x1, y1), (x2, y2), FN_COLOR, 2)
                label_text = f"FN: {BDD_CLASSES[label_idx]}"
                
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(img, (x1, y1 - 15), (x1 + w, y1), FN_COLOR, -1)
                cv2.putText(img, label_text, (x1, y1 - 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add legend
        legend_y = 50
        cv2.putText(img, "TP (Green) | FP (Red) | FN (Blue)", 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img
    
    def create_comparison(self, image_path: Path, sample_idx: int):
        """Create side-by-side comparison of GT and predictions."""
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Get GT
        item = self.ground_truth[sample_idx]
        gt_boxes, gt_labels = self._parse_gt_boxes(item.get('labels', []))
        
        # Run inference
        results = self.model.predict(
            str(image_path),
            conf=self.conf_threshold,
            iou=0.45,
            verbose=False,
            device=self.device
        )
        
        result = results[0]
        boxes = result.boxes
        
        if len(boxes) > 0:
            pred_boxes = boxes.xyxy.cpu().numpy()
            pred_scores = boxes.conf.cpu().numpy()
            pred_labels = boxes.cls.cpu().numpy().astype(np.int32)
        else:
            pred_boxes = np.zeros((0, 4))
            pred_scores = np.zeros((0,))
            pred_labels = np.zeros((0,), dtype=np.int32)
        
        # Draw GT
        img_gt = self.draw_boxes(img, gt_boxes, gt_labels, title="Ground Truth")
        
        # Draw predictions
        img_pred = self.draw_boxes(img, pred_boxes, pred_labels, pred_scores, 
                                   title="Predictions")
        
        # Draw overlay
        img_overlay = self.draw_overlay(img, gt_boxes, gt_labels, 
                                        pred_boxes, pred_labels, pred_scores)
        
        # Concatenate side by side
        comparison = np.hstack([img_gt, img_pred])
        
        return comparison, img_overlay, len(gt_boxes), len(pred_boxes)
    
    def generate_visualizations(self, num_samples: int = 50):
        """Generate visualization samples."""
        print(f"\n{'='*60}")
        print(f"Generating Qualitative Visualizations")
        print(f"{'='*60}\n")
        
        # Sample random indices
        sample_indices = random.sample(range(len(self.ground_truth)), 
                                       min(num_samples, len(self.ground_truth)))
        
        success_count = 0
        failure_count = 0
        
        for idx in tqdm(sample_indices, desc="Creating visualizations"):
            item = self.ground_truth[idx]
            img_name = item['name']
            img_path = self.data_root / 'images' / '100k' / 'val' / img_name
            
            if not img_path.exists():
                img_path = self.data_root / 'val' / 'images' / img_name
            
            if not img_path.exists():
                continue
            
            result = self.create_comparison(img_path, idx)
            if result is None:
                continue
            
            comparison, overlay, num_gt, num_pred = result
            
            # Save comparison
            comparison_path = self.output_dir / "comparisons" / f"comparison_{idx:04d}.jpg"
            cv2.imwrite(str(comparison_path), comparison)
            
            # Save overlay
            overlay_path = self.output_dir / "overlay" / f"overlay_{idx:04d}.jpg"
            cv2.imwrite(str(overlay_path), overlay)
            
            # Categorize as success or failure
            if num_gt > 0 and num_pred > 0:
                success_path = self.output_dir / "success" / f"success_{success_count:04d}.jpg"
                cv2.imwrite(str(success_path), overlay)
                success_count += 1
            elif num_gt > 0 and num_pred == 0:
                failure_path = self.output_dir / "failures" / f"failure_{failure_count:04d}.jpg"
                cv2.imwrite(str(failure_path), overlay)
                failure_count += 1
        
        print(f"\n{'='*60}")
        print(f"Visualization Summary:")
        print(f"  Total samples: {len(sample_indices)}")
        print(f"  Success cases saved: {success_count}")
        print(f"  Failure cases saved: {failure_count}")
        print(f"  Output directory: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Create qualitative visualizations")
    
    parser.add_argument('--model', type=str, default='../yolov8m.pt',
                       help='Path to model weights')
    parser.add_argument('--data-root', type=str,
                       default='../../data_analysis/data/bdd100k_images_100k',
                       help='Root directory of dataset')
    parser.add_argument('--labels', type=str,
                       default='../../data_analysis/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json',
                       help='Path to validation labels JSON')
    parser.add_argument('--output-dir', type=str,
                       default='../outputs/evaluation/predictions',
                       help='Output directory')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to visualize')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Create visualizer
    visualizer = QualitativeVisualizer(
        model_path=args.model,
        data_root=args.data_root,
        labels_path=args.labels,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device
    )
    
    # Generate visualizations
    visualizer.generate_visualizations(num_samples=args.num_samples)
    
    print("✓ Done!")


if __name__ == "__main__":
    main()
