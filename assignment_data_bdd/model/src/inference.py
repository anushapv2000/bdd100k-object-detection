"""
Inference script for YOLOv8 on BDD100k validation set.

This module loads a trained/pre-trained YOLOv8 model and runs inference
on validation images, saving visualizations and predictions.

Author: Bosch Assignment - Model Task
Date: November 2025
"""

import torch
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from ultralytics import YOLO


class BDD100kInference:
    """
    Inference class for YOLOv8 on BDD100k dataset.
    """
    
    # BDD100k class names
    CLASSES = [
        'bike', 'bus', 'car', 'motor', 'person',
        'rider', 'traffic light', 'traffic sign', 'train', 'truck'
    ]
    
    # Colors for visualization (BGR format for OpenCV)
    COLORS = [
        (255, 0, 0),      # bike - blue
        (0, 255, 0),      # bus - green
        (0, 0, 255),      # car - red
        (255, 255, 0),    # motor - cyan
        (255, 0, 255),    # person - magenta
        (0, 255, 255),    # rider - yellow
        (128, 0, 128),    # traffic light - purple
        (255, 128, 0),    # traffic sign - orange
        (0, 128, 255),    # train - light blue
        (128, 255, 0),    # truck - lime
    ]
    
    def __init__(
        self,
        model_path: str = 'yolov8m.pt',
        device: str = 'auto',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize inference class.
        
        Args:
            model_path: Path to model weights
            device: Device to run on ('auto', 'cuda', 'cpu')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        print(f"✓ Model loaded successfully on {self.device}")
    
    def predict_image(
        self,
        image_path: str,
        visualize: bool = True,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            visualize: Whether to create visualization
            save_path: Path to save visualization (if None, auto-generated)
        
        Returns:
            Dictionary with predictions
        """
        # Run inference
        results = self.model.predict(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        # Extract detections
        boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        
        # Create prediction dictionary
        predictions = {
            'image': Path(image_path).name,
            'num_detections': len(boxes),
            'detections': []
        }
        
        for i in range(len(boxes)):
            predictions['detections'].append({
                'class_id': int(classes[i]),
                'class_name': self.CLASSES[classes[i]],
                'confidence': float(scores[i]),
                'bbox': boxes[i].tolist()  # [x1, y1, x2, y2]
            })
        
        # Visualize if requested
        if visualize:
            if save_path is None:
                save_path = Path('../outputs/inference_samples') / Path(image_path).name
            
            self._visualize_predictions(image_path, boxes, scores, classes, save_path)
        
        return predictions
    
    def predict_batch(
        self,
        image_paths: List[str],
        save_dir: str = '../outputs/inference_samples',
        save_visualizations: bool = True
    ) -> List[Dict]:
        """
        Run inference on multiple images.
        
        Args:
            image_paths: List of image paths
            save_dir: Directory to save results
            save_visualizations: Whether to save visualizations
        
        Returns:
            List of prediction dictionaries
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        all_predictions = []
        
        print(f"Running inference on {len(image_paths)} images...")
        for img_path in tqdm(image_paths):
            predictions = self.predict_image(
                img_path,
                visualize=save_visualizations,
                save_path=save_path / Path(img_path).name if save_visualizations else None
            )
            all_predictions.append(predictions)
        
        return all_predictions
    
    def predict_from_dataset(
        self,
        images_dir: str,
        labels_path: str,
        num_samples: int = 50,
        save_dir: str = '../outputs/inference_samples'
    ) -> List[Dict]:
        """
        Run inference on samples from BDD100k dataset.
        
        Args:
            images_dir: Path to images directory
            labels_path: Path to labels JSON
            num_samples: Number of samples to process
            save_dir: Directory to save results
        
        Returns:
            List of predictions
        """
        # Load labels
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        
        # Get sample images
        sample_images = [
            str(Path(images_dir) / item['name'])
            for item in labels_data[:num_samples]
        ]
        
        # Run inference
        predictions = self.predict_batch(sample_images, save_dir)
        
        print(f"\n✓ Inference completed on {len(predictions)} images")
        print(f"✓ Results saved to: {save_dir}")
        
        return predictions
    
    def _visualize_predictions(
        self,
        image_path: str,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        save_path: str
    ):
        """
        Visualize predictions on image.
        
        Args:
            image_path: Path to input image
            boxes: Bounding boxes [N, 4] in format [x1, y1, x2, y2]
            scores: Confidence scores [N]
            classes: Class IDs [N]
            save_path: Path to save visualization
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return
        
        # Draw each detection
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            score = scores[i]
            cls_id = classes[i]
            
            # Get color for this class
            color = self.COLORS[cls_id]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{self.CLASSES[cls_id]}: {score:.2f}"
            
            # Get label size
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                image,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                image,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Save visualization
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(save_path), image)
    
    def export_predictions_to_json(
        self,
        predictions: List[Dict],
        output_path: str
    ):
        """
        Export predictions to JSON file.
        
        Args:
            predictions: List of prediction dictionaries
            output_path: Path to save JSON
        """
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"✓ Predictions exported to: {output_path}")
    
    def get_statistics(self, predictions: List[Dict]) -> Dict:
        """
        Compute statistics from predictions.
        
        Args:
            predictions: List of prediction dictionaries
        
        Returns:
            Dictionary with statistics
        """
        total_detections = sum(p['num_detections'] for p in predictions)
        
        # Count per class
        class_counts = {cls: 0 for cls in self.CLASSES}
        confidence_scores = {cls: [] for cls in self.CLASSES}
        
        for pred in predictions:
            for det in pred['detections']:
                class_name = det['class_name']
                class_counts[class_name] += 1
                confidence_scores[class_name].append(det['confidence'])
        
        # Compute average confidence per class
        avg_confidence = {}
        for cls in self.CLASSES:
            if len(confidence_scores[cls]) > 0:
                avg_confidence[cls] = np.mean(confidence_scores[cls])
            else:
                avg_confidence[cls] = 0.0
        
        return {
            'num_images': len(predictions),
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(predictions),
            'class_counts': class_counts,
            'avg_confidence_per_class': avg_confidence
        }


def main():
    """
    Main function to run inference demo.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run YOLOv8 inference on BDD100k')
    parser.add_argument('--model', type=str, default='yolov8m.pt',
                        help='Path to model weights')
    parser.add_argument('--images-dir', type=str,
                        default='../../data/bdd100k_images_100k/bdd100k/images/100k/val/',
                        help='Path to images directory')
    parser.add_argument('--labels', type=str,
                        default='../../data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json',
                        help='Path to labels JSON')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of samples to process')
    parser.add_argument('--output-dir', type=str, default='../outputs/inference_samples',
                        help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("YOLOv8 Inference on BDD100k".center(70))
    print("=" * 70)
    print()
    
    # Initialize inference
    inference = BDD100kInference(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run inference
    predictions = inference.predict_from_dataset(
        images_dir=args.images_dir,
        labels_path=args.labels,
        num_samples=args.num_samples,
        save_dir=args.output_dir
    )
    
    # Get statistics
    stats = inference.get_statistics(predictions)
    
    print("\n" + "=" * 70)
    print("Inference Statistics".center(70))
    print("=" * 70)
    print(f"\nProcessed {stats['num_images']} images")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Average detections per image: {stats['avg_detections_per_image']:.2f}")
    print("\nDetections per class:")
    for cls, count in stats['class_counts'].items():
        avg_conf = stats['avg_confidence_per_class'][cls]
        print(f"  {cls:15s}: {count:5d} (avg conf: {avg_conf:.3f})")
    
    # Export predictions
    output_json = Path(args.output_dir) / 'predictions.json'
    inference.export_predictions_to_json(predictions, str(output_json))
    
    print("\n" + "=" * 70)
    print("Inference completed successfully!".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
