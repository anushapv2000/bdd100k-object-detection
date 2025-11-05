"""
Failure Analysis and Clustering for BDD100k Object Detection.

This script analyzes model failures and clusters them into meaningful categories:
- Small object detection failures
- Occlusion issues
- Class confusion patterns
- Crowded scene performance
- Lighting/weather condition impact

Author: Bosch Assignment - Phase 3
Date: November 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from metrics import compute_iou


# BDD100k class names
BDD_CLASSES = [
    'bike', 'bus', 'car', 'motor', 'person', 
    'rider', 'traffic light', 'traffic sign', 'train', 'truck'
]


class FailureAnalyzer:
    """Analyzer for model failure patterns."""
    
    def __init__(
        self,
        results_path: str,
        model_path: str,
        data_root: str,
        labels_path: str,
        output_dir: str = "outputs/evaluation/failure_analysis",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize failure analyzer.
        
        Args:
            results_path: Path to evaluation results JSON
            model_path: Path to model weights
            data_root: Root directory of dataset
            labels_path: Path to validation labels JSON
            output_dir: Directory to save analysis
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            device: Device to run on
        """
        self.results_path = results_path
        self.model_path = model_path
        self.data_root = Path(data_root)
        self.labels_path = labels_path
        self.output_dir = Path(output_dir)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Create output directories
        (self.output_dir / "small_objects").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "class_confusion").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "crowded_scenes").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "low_confidence").mkdir(parents=True, exist_ok=True)
        
        # Load results
        print(f"Loading evaluation results from: {results_path}")
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        # Load model
        print(f"Loading model from: {model_path}")
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
    
    def _compute_box_size(self, box: np.ndarray) -> float:
        """Compute box area in pixels."""
        return (box[2] - box[0]) * (box[3] - box[1])
    
    def analyze_small_objects(self, max_samples: int = 100) -> Dict:
        """Analyze performance on small objects."""
        print("\nAnalyzing small object detection...")
        
        small_threshold = 32 * 32  # COCO small object threshold
        
        failures = []
        small_gt_count = 0
        small_detected_count = 0
        
        for idx in tqdm(range(min(max_samples, len(self.ground_truth))), 
                       desc="Analyzing small objects"):
            item = self.ground_truth[idx]
            img_name = item['name']
            img_path = self.data_root / 'images' / '100k' / 'val' / img_name
            
            if not img_path.exists():
                img_path = self.data_root / 'val' / 'images' / img_name
            
            if not img_path.exists():
                continue
            
            # Get GT
            gt_boxes, gt_labels = self._parse_gt_boxes(item.get('labels', []))
            
            # Find small objects
            small_mask = np.array([self._compute_box_size(box) < small_threshold 
                                  for box in gt_boxes])
            
            if not small_mask.any():
                continue
            
            small_gt_count += small_mask.sum()
            
            # Run inference
            results = self.model.predict(str(img_path), conf=self.conf_threshold, 
                                        verbose=False, device=self.device)
            result = results[0]
            boxes = result.boxes
            
            if len(boxes) > 0:
                pred_boxes = boxes.xyxy.cpu().numpy()
                pred_scores = boxes.conf.cpu().numpy()
                pred_labels = boxes.cls.cpu().numpy().astype(np.int32)
            else:
                continue
            
            # Match small GT boxes to predictions
            small_gt_boxes = gt_boxes[small_mask]
            small_gt_labels = gt_labels[small_mask]
            
            for gt_box, gt_label in zip(small_gt_boxes, small_gt_labels):
                best_iou = 0
                for pred_box, pred_label in zip(pred_boxes, pred_labels):
                    if pred_label == gt_label:
                        iou = compute_iou(gt_box, pred_box)
                        best_iou = max(best_iou, iou)
                
                if best_iou >= self.iou_threshold:
                    small_detected_count += 1
                else:
                    failures.append({
                        'image': img_name,
                        'class': BDD_CLASSES[gt_label],
                        'box_area': self._compute_box_size(gt_box),
                        'best_iou': best_iou
                    })
        
        recall = small_detected_count / small_gt_count if small_gt_count > 0 else 0
        
        analysis = {
            'total_small_objects': int(small_gt_count),
            'detected_small_objects': int(small_detected_count),
            'recall': float(recall),
            'failures': failures[:50]  # Top 50 failures
        }
        
        print(f"  Small objects: {small_gt_count}")
        print(f"  Detected: {small_detected_count}")
        print(f"  Recall: {recall:.3f}")
        
        return analysis
    
    def analyze_class_confusion(self) -> Dict:
        """Analyze class confusion patterns."""
        print("\nAnalyzing class confusion...")
        
        conf_matrix = np.array(self.results['confusion_matrix'])
        
        # Find top confusions (excluding diagonal and background)
        confusions = []
        for i in range(len(BDD_CLASSES)):
            for j in range(len(BDD_CLASSES)):
                if i != j:
                    count = conf_matrix[i, j]
                    if count > 0:
                        confusions.append({
                            'gt_class': BDD_CLASSES[i],
                            'pred_class': BDD_CLASSES[j],
                            'count': int(count),
                            'gt_class_id': i,
                            'pred_class_id': j
                        })
        
        # Sort by count
        confusions = sorted(confusions, key=lambda x: x['count'], reverse=True)
        
        print(f"  Top 10 class confusions:")
        for conf in confusions[:10]:
            print(f"    {conf['gt_class']:15} -> {conf['pred_class']:15}: {conf['count']:4d}")
        
        return {'confusions': confusions[:20]}
    
    def analyze_crowded_scenes(self, max_samples: int = 100) -> Dict:
        """Analyze performance in crowded scenes."""
        print("\nAnalyzing crowded scene performance...")
        
        crowded_threshold = 20  # Scenes with 20+ objects
        
        scene_performance = []
        
        for idx in tqdm(range(min(max_samples, len(self.ground_truth))), 
                       desc="Analyzing crowded scenes"):
            item = self.ground_truth[idx]
            img_name = item['name']
            img_path = self.data_root / 'images' / '100k' / 'val' / img_name
            
            if not img_path.exists():
                img_path = self.data_root / 'val' / 'images' / img_name
            
            if not img_path.exists():
                continue
            
            # Get GT
            gt_boxes, gt_labels = self._parse_gt_boxes(item.get('labels', []))
            num_objects = len(gt_boxes)
            
            if num_objects < crowded_threshold:
                continue
            
            # Run inference
            results = self.model.predict(str(img_path), conf=self.conf_threshold, 
                                        verbose=False, device=self.device)
            result = results[0]
            boxes = result.boxes
            
            num_predictions = len(boxes) if len(boxes) > 0 else 0
            
            # Simple recall estimation
            recall = num_predictions / num_objects if num_objects > 0 else 0
            
            scene_performance.append({
                'image': img_name,
                'num_objects': int(num_objects),
                'num_predictions': int(num_predictions),
                'recall_estimate': float(recall)
            })
        
        # Sort by number of objects
        scene_performance = sorted(scene_performance, key=lambda x: x['num_objects'], 
                                  reverse=True)
        
        avg_recall = np.mean([s['recall_estimate'] for s in scene_performance]) if scene_performance else 0
        
        print(f"  Crowded scenes analyzed: {len(scene_performance)}")
        print(f"  Average recall estimate: {avg_recall:.3f}")
        
        return {
            'num_scenes': len(scene_performance),
            'avg_recall': float(avg_recall),
            'scenes': scene_performance[:30]
        }
    
    def analyze_low_confidence_detections(self, max_samples: int = 100) -> Dict:
        """Analyze low confidence detections."""
        print("\nAnalyzing low confidence detections...")
        
        low_conf_threshold = 0.5
        low_conf_detections = []
        
        for idx in tqdm(range(min(max_samples, len(self.ground_truth))), 
                       desc="Analyzing confidence"):
            item = self.ground_truth[idx]
            img_name = item['name']
            img_path = self.data_root / 'images' / '100k' / 'val' / img_name
            
            if not img_path.exists():
                img_path = self.data_root / 'val' / 'images' / img_name
            
            if not img_path.exists():
                continue
            
            # Run inference
            results = self.model.predict(str(img_path), conf=self.conf_threshold, 
                                        verbose=False, device=self.device)
            result = results[0]
            boxes = result.boxes
            
            if len(boxes) == 0:
                continue
            
            pred_scores = boxes.conf.cpu().numpy()
            pred_labels = boxes.cls.cpu().numpy().astype(np.int32)
            
            # Find low confidence predictions
            low_conf_mask = pred_scores < low_conf_threshold
            
            if low_conf_mask.any():
                for score, label in zip(pred_scores[low_conf_mask], 
                                       pred_labels[low_conf_mask]):
                    low_conf_detections.append({
                        'image': img_name,
                        'class': BDD_CLASSES[int(label)],
                        'confidence': float(score)
                    })
        
        # Group by class
        by_class = defaultdict(list)
        for det in low_conf_detections:
            by_class[det['class']].append(det['confidence'])
        
        class_stats = {}
        for class_name, scores in by_class.items():
            class_stats[class_name] = {
                'count': len(scores),
                'avg_confidence': float(np.mean(scores)),
                'min_confidence': float(np.min(scores)),
                'max_confidence': float(np.max(scores))
            }
        
        print(f"  Low confidence detections: {len(low_conf_detections)}")
        
        return {
            'total_low_conf': len(low_conf_detections),
            'by_class': class_stats,
            'samples': low_conf_detections[:50]
        }
    
    def generate_failure_report(self):
        """Generate comprehensive failure analysis report."""
        print(f"\n{'='*60}")
        print(f"Running Failure Analysis")
        print(f"{'='*60}\n")
        
        # Run analyses
        small_obj_analysis = self.analyze_small_objects(max_samples=200)
        class_confusion_analysis = self.analyze_class_confusion()
        crowded_scene_analysis = self.analyze_crowded_scenes(max_samples=200)
        low_conf_analysis = self.analyze_low_confidence_detections(max_samples=200)
        
        # Compile report
        report = {
            'small_objects': small_obj_analysis,
            'class_confusion': class_confusion_analysis,
            'crowded_scenes': crowded_scene_analysis,
            'low_confidence': low_conf_analysis
        }
        
        # Save report
        report_path = self.output_dir / "failure_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Failure Analysis Complete")
        print(f"Report saved to: {report_path}")
        print(f"{'='*60}\n")
        
        # Create visualizations
        self._create_failure_visualizations(report)
        
        return report
    
    def _create_failure_visualizations(self, report: Dict):
        """Create visualizations for failure analysis."""
        print("Creating failure analysis visualizations...")
        
        # 1. Small object performance
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Small Objects\nDetected', 'Small Objects\nMissed']
        detected = report['small_objects']['detected_small_objects']
        total = report['small_objects']['total_small_objects']
        missed = total - detected
        values = [detected, missed]
        colors = ['#06D6A0', '#EF476F']
        
        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value}\n({value/total*100:.1f}%)',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        recall = report['small_objects']['recall']
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(f'Small Object Detection Performance\nRecall: {recall:.3f}', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "small_object_performance.png", dpi=300)
        plt.close()
        
        # 2. Top class confusions
        confusions = report['class_confusion']['confusions'][:10]
        if confusions:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            labels = [f"{c['gt_class']} → {c['pred_class']}" for c in confusions]
            counts = [c['count'] for c in confusions]
            
            bars = ax.barh(labels, counts, color='#FF6B6B', edgecolor='black', linewidth=1)
            for bar, count in zip(bars, counts):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f' {count}',
                       ha='left', va='center', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Count', fontsize=12, fontweight='bold')
            ax.set_title('Top 10 Class Confusion Patterns', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "class_confusion_patterns.png", dpi=300)
            plt.close()
        
        print(f"  Saved visualizations to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze model failure patterns")
    
    parser.add_argument('--results', type=str,
                       default='../outputs/evaluation/metrics/evaluation_results.json',
                       help='Path to evaluation results JSON')
    parser.add_argument('--model', type=str, default='../yolov8m.pt',
                       help='Path to model weights')
    parser.add_argument('--data-root', type=str,
                       default='../../data_analysis/data/bdd100k_images_100k',
                       help='Root directory of dataset')
    parser.add_argument('--labels', type=str,
                       default='../../data_analysis/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json',
                       help='Path to validation labels JSON')
    parser.add_argument('--output-dir', type=str,
                       default='../outputs/evaluation/failure_analysis',
                       help='Output directory')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = FailureAnalyzer(
        results_path=args.results,
        model_path=args.model,
        data_root=args.data_root,
        labels_path=args.labels,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device
    )
    
    # Run analysis
    report = analyzer.generate_failure_report()
    
    print("✓ Done!")


if __name__ == "__main__":
    main()
