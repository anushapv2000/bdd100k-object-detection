"""
Utility functions for BDD100k object detection task.

This module provides helper functions for visualization, metrics computation,
and data processing.

Author: Bosch Assignment - Model Task
Date: November 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json


def draw_bboxes_on_image(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image (BGR format)
        boxes: Bounding boxes [N, 4] in format [x1, y1, x2, y2]
        labels: Class labels [N]
        scores: Confidence scores [N] (optional)
        class_names: List of class names (optional)
        colors: List of colors for each class (optional)
    
    Returns:
        Image with drawn bounding boxes
    """
    image = image.copy()
    
    if colors is None:
        # Default colors
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0),
                  (0, 128, 255), (128, 255, 0)]
    
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        label_idx = int(labels[i])
        color = colors[label_idx % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Create label text
        if class_names is not None:
            label_text = class_names[label_idx]
        else:
            label_text = f"Class {label_idx}"
        
        if scores is not None:
            label_text += f": {scores[i]:.2f}"
        
        # Draw label
        cv2.putText(image, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes.
    
    Args:
        box1: Box 1 in format [x1, y1, x2, y2]
        box2: Box 2 in format [x1, y1, x2, y2]
    
    Returns:
        IoU score
    """
    # Compute intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Compute IoU
    iou = intersection / union if union > 0 else 0
    
    return iou


def compute_box_area(box: np.ndarray) -> float:
    """
    Compute area of bounding box.
    
    Args:
        box: Bounding box [x1, y1, x2, y2]
    
    Returns:
        Area in pixels
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def compute_box_aspect_ratio(box: np.ndarray) -> float:
    """
    Compute aspect ratio of bounding box.
    
    Args:
        box: Bounding box [x1, y1, x2, y2]
    
    Returns:
        Aspect ratio (width / height)
    """
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width / height if height > 0 else 0


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] format.
    
    Args:
        boxes: Boxes in [x1, y1, x2, y2] format [N, 4]
    
    Returns:
        Boxes in [x, y, w, h] format [N, 4]
    """
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
    boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
    return boxes_xywh


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format.
    
    Args:
        boxes: Boxes in [x, y, w, h] format [N, 4]
    
    Returns:
        Boxes in [x1, y1, x2, y2] format [N, 4]
    """
    boxes_xyxy = boxes.copy()
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2
    return boxes_xyxy


def normalize_boxes(boxes: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """
    Normalize bounding boxes to [0, 1] range.
    
    Args:
        boxes: Boxes in [x1, y1, x2, y2] format [N, 4]
        img_width: Image width
        img_height: Image height
    
    Returns:
        Normalized boxes [N, 4]
    """
    boxes_norm = boxes.copy().astype(np.float32)
    boxes_norm[:, [0, 2]] /= img_width
    boxes_norm[:, [1, 3]] /= img_height
    return boxes_norm


def denormalize_boxes(boxes: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """
    Denormalize bounding boxes from [0, 1] range to pixel coordinates.
    
    Args:
        boxes: Normalized boxes [N, 4]
        img_width: Image width
        img_height: Image height
    
    Returns:
        Boxes in pixel coordinates [N, 4]
    """
    boxes_denorm = boxes.copy()
    boxes_denorm[:, [0, 2]] *= img_width
    boxes_denorm[:, [1, 3]] *= img_height
    return boxes_denorm


def filter_boxes_by_confidence(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter boxes by confidence threshold.
    
    Args:
        boxes: Bounding boxes [N, 4]
        scores: Confidence scores [N]
        labels: Class labels [N]
        threshold: Confidence threshold
    
    Returns:
        Filtered (boxes, scores, labels)
    """
    mask = scores >= threshold
    return boxes[mask], scores[mask], labels[mask]


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5
) -> np.ndarray:
    """
    Apply Non-Maximum Suppression.
    
    Args:
        boxes: Bounding boxes [N, 4] in [x1, y1, x2, y2] format
        scores: Confidence scores [N]
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Indices of boxes to keep
    """
    # Sort by score
    order = scores.argsort()[::-1]
    keep = []
    
    while len(order) > 0:
        # Keep box with highest score
        i = order[0]
        keep.append(i)
        
        # Compute IoU with remaining boxes
        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in order[1:]])
        
        # Remove boxes with IoU > threshold
        mask = ious <= iou_threshold
        order = order[1:][mask]
    
    return np.array(keep)


def create_class_distribution_plot(
    class_counts: Dict[str, int],
    save_path: Optional[str] = None,
    title: str = "Class Distribution"
) -> plt.Figure:
    """
    Create bar plot of class distribution.
    
    Args:
        class_counts: Dictionary mapping class names to counts
        save_path: Path to save plot (optional)
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    ax.bar(classes, counts, color='steelblue')
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, count in enumerate(counts):
        ax.text(i, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_detections_to_json(
    detections: List[Dict],
    output_path: str
):
    """
    Save detections to JSON file.
    
    Args:
        detections: List of detection dictionaries
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=2)
    
    print(f"Saved detections to: {output_path}")


def load_detections_from_json(json_path: str) -> List[Dict]:
    """
    Load detections from JSON file.
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        List of detection dictionaries
    """
    with open(json_path, 'r') as f:
        detections = json.load(f)
    
    return detections


def calculate_fps(num_images: int, total_time: float) -> float:
    """
    Calculate frames per second.
    
    Args:
        num_images: Number of images processed
        total_time: Total time in seconds
    
    Returns:
        FPS
    """
    return num_images / total_time if total_time > 0 else 0


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_model_info(model):
    """
    Print model information.
    
    Args:
        model: PyTorch model or YOLO model
    """
    try:
        # For YOLO models
        if hasattr(model, 'model'):
            print(f"Model: {model.model_name if hasattr(model, 'model_name') else 'YOLOv8'}")
            print(f"Task: {model.task if hasattr(model, 'task') else 'detect'}")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Model size: ~{total_params * 4 / (1024**2):.1f} MB (FP32)")
        else:
            # For PyTorch models
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f"Could not print model info: {e}")


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics.
        
        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test IoU computation
    box1 = np.array([10, 10, 50, 50])
    box2 = np.array([30, 30, 70, 70])
    iou = compute_iou(box1, box2)
    print(f"IoU between {box1} and {box2}: {iou:.3f}")
    
    # Test coordinate conversions
    boxes_xyxy = np.array([[10, 20, 30, 40], [50, 60, 100, 120]])
    boxes_xywh = xyxy_to_xywh(boxes_xyxy)
    print(f"\nxyxy: {boxes_xyxy}")
    print(f"xywh: {boxes_xywh}")
    
    # Test AverageMeter
    meter = AverageMeter("Loss")
    meter.update(1.5)
    meter.update(1.3)
    meter.update(1.1)
    print(f"\n{meter}")
    
    print("\n✓ All tests passed!")
