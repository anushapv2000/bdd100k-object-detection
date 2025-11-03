"""
Inference Module for BDD100K Object Detection

This module handles model inference, post-processing, and prediction
visualization for the trained BDD100K object detection model.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

try:
    import torch
    import numpy as np
    from ultralytics import YOLO
    from PIL import Image, ImageDraw, ImageFont
    import cv2
except ImportError:
    print("Warning: Some dependencies not available. Install requirements.txt")


class BDDInference:
    """
    Inference class for BDD100K object detection models.
    
    Handles loading trained models, running inference on images,
    and post-processing results with proper BDD class mapping.
    """
    
    def __init__(self, model_path: str, class_names: Optional[List[str]] = None,
                 device: str = "auto"):
        """
        Initialize inference engine.
        
        Args:
            model_path (str): Path to trained model weights
            class_names (List[str], optional): List of class names
            device (str): Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        
        # BDD100K default classes
        if class_names is None:
            self.class_names = [
                'traffic light', 'traffic sign', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle'
            ]
        else:
            self.class_names = class_names
        
        self.logger = logging.getLogger(__name__)
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained YOLO model."""
        try:
            self.model = YOLO(self.model_path)
            self.logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_single_image(self, image_path: str, 
                           confidence_threshold: float = 0.25,
                           iou_threshold: float = 0.45) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image_path (str): Path to input image
            confidence_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
            
        Returns:
            Dict: Detection results with boxes, scores, classes, and metadata
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Run inference
        start_time = time.time()
        results = self.model(
            image_path, 
            conf=confidence_threshold,
            iou=iou_threshold,
            verbose=False
        )
        inference_time = time.time() - start_time
        
        # Parse results
        result = results[0]  # Single image result
        
        # Extract detection information
        detections = {
            'image_path': image_path,
            'image_shape': result.orig_shape,
            'inference_time': inference_time,
            'detections': []
        }
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
                detection = {
                    'bbox': box.tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(score),
                    'class_id': int(cls_id),
                    'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else f'class_{cls_id}'
                }
                detections['detections'].append(detection)
        
        return detections
    
    def predict_batch(self, image_paths: List[str], 
                     confidence_threshold: float = 0.25,
                     iou_threshold: float = 0.45,
                     batch_size: int = 8) -> List[Dict]:
        """
        Run inference on a batch of images.
        
        Args:
            image_paths (List[str]): List of image paths
            confidence_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
            batch_size (int): Batch size for inference
            
        Returns:
            List[Dict]: List of detection results for each image
        """
        results = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Run batch inference
            batch_results = self.model(
                batch_paths,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            # Process each result in the batch
            for j, result in enumerate(batch_results):
                image_path = batch_paths[j]
                
                detections = {
                    'image_path': image_path,
                    'image_shape': result.orig_shape,
                    'detections': []
                }
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, score, cls_id in zip(boxes, scores, classes):
                        detection = {
                            'bbox': box.tolist(),
                            'confidence': float(score),
                            'class_id': int(cls_id),
                            'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else f'class_{cls_id}'
                        }
                        detections['detections'].append(detection)
                
                results.append(detections)
        
        return results
    
    def visualize_predictions(self, image_path: str, predictions: Dict,
                            output_path: Optional[str] = None,
                            show_confidence: bool = True,
                            box_thickness: int = 2) -> Union[str, np.ndarray]:
        """
        Visualize predictions on an image.
        
        Args:
            image_path (str): Path to input image
            predictions (Dict): Predictions from predict_single_image()
            output_path (str, optional): Path to save visualization
            show_confidence (bool): Whether to show confidence scores
            box_thickness (int): Thickness of bounding box lines
            
        Returns:
            Union[str, np.ndarray]: Output path if saved, or image array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Color palette for different classes
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 0), (128, 128, 128)
        ]
        
        # Draw detections
        for detection in predictions['detections']:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)
            
            # Prepare label text
            if show_confidence:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Save or return image
        if output_path:
            cv2.imwrite(output_path, image)
            return output_path
        else:
            return image
    
    def export_predictions_to_json(self, predictions: List[Dict], 
                                  output_path: str) -> None:
        """
        Export predictions to JSON file in BDD100K format.
        
        Args:
            predictions (List[Dict]): List of prediction results
            output_path (str): Path to save JSON file
        """
        bdd_format = []
        
        for pred in predictions:
            image_name = os.path.basename(pred['image_path'])
            
            labels = []
            for detection in pred['detections']:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                label = {
                    'id': len(labels),
                    'category': detection['class_name'],
                    'box2d': {
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    }
                }
                labels.append(label)
            
            bdd_item = {
                'name': image_name,
                'labels': labels
            }
            bdd_format.append(bdd_item)
        
        with open(output_path, 'w') as f:
            json.dump(bdd_format, f, indent=2)
        
        self.logger.info(f"Predictions exported to {output_path}")
    
    def benchmark_inference_speed(self, image_paths: List[str], 
                                num_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark inference speed on a set of images.
        
        Args:
            image_paths (List[str]): List of image paths for benchmarking
            num_runs (int): Number of runs for averaging
            
        Returns:
            Dict[str, float]: Benchmarking results
        """
        if len(image_paths) == 0:
            raise ValueError("No images provided for benchmarking")
        
        total_times = []
        
        # Warmup runs
        for _ in range(3):
            _ = self.model(image_paths[0], verbose=False)
        
        # Benchmark runs
        for run in range(num_runs):
            start_time = time.time()
            
            for image_path in image_paths:
                _ = self.model(image_path, verbose=False)
            
            total_time = time.time() - start_time
            total_times.append(total_time)
        
        # Calculate statistics
        avg_total_time = np.mean(total_times)
        avg_per_image = avg_total_time / len(image_paths)
        fps = 1.0 / avg_per_image
        
        benchmark_results = {
            'num_images': len(image_paths),
            'num_runs': num_runs,
            'avg_total_time_seconds': avg_total_time,
            'avg_time_per_image_seconds': avg_per_image,
            'avg_fps': fps,
            'min_time_seconds': min(total_times),
            'max_time_seconds': max(total_times)
        }
        
        return benchmark_results


def load_bdd_inference_model(model_path: str, **kwargs) -> BDDInference:
    """
    Convenience function to load a BDD inference model.
    
    Args:
        model_path (str): Path to model weights
        **kwargs: Additional arguments for BDDInference
        
    Returns:
        BDDInference: Initialized inference engine
    """
    return BDDInference(model_path, **kwargs)


if __name__ == "__main__":
    # Example usage
    model_path = "path/to/trained/model.pt"
    inference_engine = BDDInference(model_path)
    
    # Single image inference
    results = inference_engine.predict_single_image("path/to/test/image.jpg")
    print(f"Found {len(results['detections'])} objects")
    
    # Visualize results
    vis_image = inference_engine.visualize_predictions(
        "path/to/test/image.jpg", 
        results,
        output_path="output_visualization.jpg"
    )
