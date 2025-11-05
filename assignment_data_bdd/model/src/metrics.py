"""
Metrics computation for object detection evaluation.

This module provides implementations of standard object detection metrics:
- Average Precision (AP)
- mean Average Precision (mAP)
- Precision, Recall, F1-Score
- Confusion Matrix

Author: Bosch Assignment - Phase 3
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: Box 1 in format [x1, y1, x2, y2]
        box2: Box 2 in format [x1, y1, x2, y2]
    
    Returns:
        IoU score between 0 and 1
    """
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # IoU
    iou = intersection / union if union > 0 else 0
    return iou


def match_predictions_to_ground_truth(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match predictions to ground truth boxes.
    
    Args:
        pred_boxes: Predicted boxes [N, 4]
        pred_scores: Prediction confidence scores [N]
        pred_labels: Predicted class labels [N]
        gt_boxes: Ground truth boxes [M, 4]
        gt_labels: Ground truth class labels [M]
        iou_threshold: IoU threshold for matching
    
    Returns:
        - matches: Array of matched GT indices for each prediction (-1 if no match) [N]
        - ious: IoU values for each prediction [N]
        - tp: True positive mask [N]
    """
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)
    
    matches = np.full(num_preds, -1, dtype=np.int32)
    ious = np.zeros(num_preds, dtype=np.float32)
    tp = np.zeros(num_preds, dtype=bool)
    
    if num_gts == 0:
        return matches, ious, tp
    
    # Track which GT boxes have been matched
    gt_matched = np.zeros(num_gts, dtype=bool)
    
    # Sort predictions by confidence (highest first)
    sorted_indices = np.argsort(pred_scores)[::-1]
    
    for idx in sorted_indices:
        pred_box = pred_boxes[idx]
        pred_label = pred_labels[idx]
        
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching GT box with same class
        for gt_idx in range(num_gts):
            if gt_matched[gt_idx]:
                continue
            
            if gt_labels[gt_idx] != pred_label:
                continue
            
            iou = compute_iou(pred_box, gt_boxes[gt_idx])
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        ious[idx] = best_iou
        
        # Check if match is valid
        if best_iou >= iou_threshold and best_gt_idx != -1:
            matches[idx] = best_gt_idx
            gt_matched[best_gt_idx] = True
            tp[idx] = True
    
    return matches, ious, tp


def compute_ap(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    pred_labels: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    class_id: int,
    iou_threshold: float = 0.5
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Average Precision (AP) for a single class.
    
    Args:
        pred_boxes: List of predicted boxes per image
        pred_scores: List of prediction scores per image
        pred_labels: List of predicted labels per image
        gt_boxes: List of ground truth boxes per image
        gt_labels: List of ground truth labels per image
        class_id: Class ID to compute AP for
        iou_threshold: IoU threshold for TP
    
    Returns:
        - ap: Average Precision
        - precision: Precision values at different thresholds
        - recall: Recall values at different thresholds
    """
    # Collect all predictions and GT for this class
    all_pred_scores = []
    all_tp = []
    num_gt = 0
    
    for img_idx in range(len(gt_boxes)):
        # Ground truth for this class
        gt_mask = gt_labels[img_idx] == class_id
        img_gt_boxes = gt_boxes[img_idx][gt_mask]
        img_gt_labels = gt_labels[img_idx][gt_mask]
        num_gt += len(img_gt_boxes)
        
        # Predictions for this class
        pred_mask = pred_labels[img_idx] == class_id
        img_pred_boxes = pred_boxes[img_idx][pred_mask]
        img_pred_scores = pred_scores[img_idx][pred_mask]
        img_pred_labels = pred_labels[img_idx][pred_mask]
        
        if len(img_pred_boxes) == 0:
            continue
        
        # Match predictions to GT
        _, _, tp = match_predictions_to_ground_truth(
            img_pred_boxes, img_pred_scores, img_pred_labels,
            img_gt_boxes, img_gt_labels, iou_threshold
        )
        
        all_pred_scores.extend(img_pred_scores)
        all_tp.extend(tp)
    
    if num_gt == 0:
        return 0.0, np.array([]), np.array([])
    
    if len(all_pred_scores) == 0:
        return 0.0, np.array([1.0]), np.array([0.0])
    
    # Sort by confidence
    all_pred_scores = np.array(all_pred_scores)
    all_tp = np.array(all_tp)
    sorted_indices = np.argsort(all_pred_scores)[::-1]
    all_tp = all_tp[sorted_indices]
    
    # Compute cumulative TP and FP
    tp_cumsum = np.cumsum(all_tp)
    fp_cumsum = np.cumsum(~all_tp)
    
    # Compute precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / num_gt
    
    # Add sentinel values
    precision = np.concatenate([[1.0], precision, [0.0]])
    recall = np.concatenate([[0.0], recall, [1.0]])
    
    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max() / 11.0
    
    return ap, precision, recall


def compute_map(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    pred_labels: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    num_classes: int,
    iou_thresholds: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP) across all classes.
    
    Args:
        pred_boxes: List of predicted boxes per image
        pred_scores: List of prediction scores per image
        pred_labels: List of predicted labels per image
        gt_boxes: List of ground truth boxes per image
        gt_labels: List of ground truth labels per image
        num_classes: Number of classes
        iou_thresholds: List of IoU thresholds (default: [0.5])
    
    Returns:
        Dictionary with mAP metrics and per-class AP
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]
    
    results = {}
    
    for iou_thresh in iou_thresholds:
        aps = []
        per_class_ap = {}
        
        for class_id in range(num_classes):
            ap, _, _ = compute_ap(
                pred_boxes, pred_scores, pred_labels,
                gt_boxes, gt_labels, class_id, iou_thresh
            )
            aps.append(ap)
            per_class_ap[f"AP_class_{class_id}"] = ap
        
        map_value = np.mean(aps)
        
        # Store results
        iou_key = f"@{iou_thresh:.2f}".replace(".", "_")
        results[f"mAP{iou_key}"] = map_value
        results[f"per_class{iou_key}"] = per_class_ap
    
    # Compute mAP@0.5:0.95 if multiple thresholds
    if len(iou_thresholds) > 1:
        map_values = [results[f"mAP@{t:.2f}".replace(".", "_")] for t in iou_thresholds]
        results["mAP@0_5:0_95"] = np.mean(map_values)
    
    return results


def compute_precision_recall_f1(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    pred_labels: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25
) -> Dict[str, Dict]:
    """
    Compute precision, recall, and F1-score per class.
    
    Args:
        pred_boxes: List of predicted boxes per image
        pred_scores: List of prediction scores per image
        pred_labels: List of predicted labels per image
        gt_boxes: List of ground truth boxes per image
        gt_labels: List of ground truth labels per image
        num_classes: Number of classes
        iou_threshold: IoU threshold for TP
        conf_threshold: Confidence threshold for predictions
    
    Returns:
        Dictionary with per-class metrics
    """
    per_class_metrics = {}
    
    for class_id in range(num_classes):
        tp = 0
        fp = 0
        fn = 0
        
        for img_idx in range(len(gt_boxes)):
            # Ground truth for this class
            gt_mask = gt_labels[img_idx] == class_id
            img_gt_boxes = gt_boxes[img_idx][gt_mask]
            img_gt_labels = gt_labels[img_idx][gt_mask]
            
            # Predictions for this class (above confidence threshold)
            pred_mask = (pred_labels[img_idx] == class_id) & (pred_scores[img_idx] >= conf_threshold)
            img_pred_boxes = pred_boxes[img_idx][pred_mask]
            img_pred_scores = pred_scores[img_idx][pred_mask]
            img_pred_labels = pred_labels[img_idx][pred_mask]
            
            if len(img_pred_boxes) == 0 and len(img_gt_boxes) == 0:
                continue
            
            if len(img_pred_boxes) == 0:
                fn += len(img_gt_boxes)
                continue
            
            if len(img_gt_boxes) == 0:
                fp += len(img_pred_boxes)
                continue
            
            # Match predictions to GT
            matches, _, tp_mask = match_predictions_to_ground_truth(
                img_pred_boxes, img_pred_scores, img_pred_labels,
                img_gt_boxes, img_gt_labels, iou_threshold
            )
            
            tp += tp_mask.sum()
            fp += (~tp_mask).sum()
            fn += len(img_gt_boxes) - len(np.unique(matches[matches >= 0]))
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return per_class_metrics


def compute_confusion_matrix(
    pred_labels: List[np.ndarray],
    gt_labels: List[np.ndarray],
    pred_boxes: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25
) -> np.ndarray:
    """
    Compute confusion matrix for object detection.
    
    Args:
        pred_labels: List of predicted labels per image
        gt_labels: List of ground truth labels per image
        pred_boxes: List of predicted boxes per image
        gt_boxes: List of ground truth boxes per image
        pred_scores: List of prediction scores per image
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching
        conf_threshold: Confidence threshold
    
    Returns:
        Confusion matrix [num_classes+1, num_classes+1]
        Last row/col represents background (FP/FN)
    """
    # +1 for background class
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)
    
    for img_idx in range(len(gt_boxes)):
        img_gt_boxes = gt_boxes[img_idx]
        img_gt_labels = gt_labels[img_idx]
        
        # Filter predictions by confidence
        conf_mask = pred_scores[img_idx] >= conf_threshold
        img_pred_boxes = pred_boxes[img_idx][conf_mask]
        img_pred_scores = pred_scores[img_idx][conf_mask]
        img_pred_labels = pred_labels[img_idx][conf_mask]
        
        if len(img_pred_boxes) == 0 and len(img_gt_boxes) == 0:
            continue
        
        # Track matched GT boxes
        gt_matched = np.zeros(len(img_gt_boxes), dtype=bool)
        
        # Match each prediction
        for pred_idx in range(len(img_pred_boxes)):
            pred_box = img_pred_boxes[pred_idx]
            pred_label = int(img_pred_labels[pred_idx])
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx in range(len(img_gt_boxes)):
                if gt_matched[gt_idx]:
                    continue
                
                iou = compute_iou(pred_box, img_gt_boxes[gt_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                gt_label = int(img_gt_labels[best_gt_idx])
                conf_matrix[gt_label, pred_label] += 1
                gt_matched[best_gt_idx] = True
            else:
                # False positive (background as GT)
                conf_matrix[num_classes, pred_label] += 1
        
        # Unmatched GT boxes are false negatives
        for gt_idx in range(len(img_gt_boxes)):
            if not gt_matched[gt_idx]:
                gt_label = int(img_gt_labels[gt_idx])
                # False negative (background as prediction)
                conf_matrix[gt_label, num_classes] += 1
    
    return conf_matrix


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics computation...")
    
    # Create dummy data
    num_images = 10
    num_classes = 3
    
    pred_boxes = [np.random.rand(5, 4) * 100 for _ in range(num_images)]
    pred_scores = [np.random.rand(5) for _ in range(num_images)]
    pred_labels = [np.random.randint(0, num_classes, 5) for _ in range(num_images)]
    
    gt_boxes = [np.random.rand(5, 4) * 100 for _ in range(num_images)]
    gt_labels = [np.random.randint(0, num_classes, 5) for _ in range(num_images)]
    
    # Test mAP computation
    map_results = compute_map(pred_boxes, pred_scores, pred_labels, 
                              gt_boxes, gt_labels, num_classes)
    print(f"\nmAP@0.5: {map_results['mAP@0_50']:.4f}")
    
    # Test precision/recall/F1
    metrics = compute_precision_recall_f1(pred_boxes, pred_scores, pred_labels,
                                         gt_boxes, gt_labels, num_classes)
    print(f"\nPer-class metrics:")
    for class_id, m in metrics.items():
        print(f"Class {class_id}: P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}")
    
    # Test confusion matrix
    conf_matrix = compute_confusion_matrix(pred_labels, gt_labels, pred_boxes, 
                                          gt_boxes, pred_scores, num_classes)
    print(f"\nConfusion matrix shape: {conf_matrix.shape}")
    
    print("\n✓ All tests passed!")
