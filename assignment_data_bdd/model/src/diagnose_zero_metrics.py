"""
Diagnostic script to check model predictions and class mappings.

This will help identify why we're getting zero metrics.
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def diagnose_model_and_data():
    """Diagnose the model and data to find the issue."""
    
    print("=" * 80)
    print("PHASE 3 DIAGNOSTIC - Finding Zero Metrics Issue")
    print("=" * 80)
    print()
    
    # 1. Check model class names
    print("1. Checking trained model class names...")
    try:
        from ultralytics import YOLO
        
        model_path = "/Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/outputs/training_logs/yolov8m_bdd100k_subset_demo/weights/best.pt"
        model = YOLO(model_path)
        
        print(f"   Model path: {model_path}")
        print(f"   Model class names: {model.names}")
        print(f"   Number of classes: {len(model.names)}")
        print()
    except Exception as e:
        print(f"   ERROR: {e}")
        print()
    
    # 2. Check ground truth class names
    print("2. Checking ground truth class names...")
    try:
        labels_path = "/Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/data_analysis/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
        
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        
        # Get unique categories
        categories = set()
        for item in labels[:100]:  # Check first 100
            if 'labels' in item:
                for label in item['labels']:
                    if 'category' in label:
                        categories.add(label['category'])
        
        print(f"   Labels path: {labels_path}")
        print(f"   Unique categories in validation set: {sorted(categories)}")
        print()
    except Exception as e:
        print(f"   ERROR: {e}")
        print()
    
    # 3. Test a single prediction
    print("3. Testing single image prediction...")
    try:
        from ultralytics import YOLO
        import cv2
        
        model_path = "/Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/outputs/training_logs/yolov8m_bdd100k_subset_demo/weights/best.pt"
        model = YOLO(model_path)
        
        # Find a validation image
        labels_path = "/Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/data_analysis/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        
        # Get first image
        first_item = labels[0]
        img_name = first_item['name']
        img_path = f"/Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/data_analysis/data/bdd100k_images_100k/images/100k/val/{img_name}"
        
        print(f"   Test image: {img_name}")
        
        # Run prediction
        results = model.predict(img_path, verbose=False, conf=0.25)
        result = results[0]
        
        print(f"   Number of detections: {len(result.boxes)}")
        
        if len(result.boxes) > 0:
            # Show first 5 detections
            for i in range(min(5, len(result.boxes))):
                cls_id = int(result.boxes.cls[i].item())
                conf = result.boxes.conf[i].item()
                cls_name = model.names[cls_id]
                print(f"      Detection {i+1}: class_id={cls_id}, class_name={cls_name}, conf={conf:.3f}")
        else:
            print("      No detections found!")
        print()
        
        # Check ground truth for same image
        print(f"   Ground truth for {img_name}:")
        if 'labels' in first_item and len(first_item['labels']) > 0:
            gt_categories = [label['category'] for label in first_item['labels'][:5]]
            print(f"      Categories: {gt_categories}")
        else:
            print("      No ground truth labels!")
        print()
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # 4. Check class name mapping
    print("4. Checking class name mapping...")
    
    BDD_CLASSES = [
        'person', 'rider', 'car', 'truck', 'bus', 'train',
        'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
    ]
    
    print(f"   Expected BDD100k classes (evaluate.py): {BDD_CLASSES}")
    print()
    
    try:
        from ultralytics import YOLO
        model_path = "/Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/outputs/training_logs/yolov8m_bdd100k_subset_demo/weights/best.pt"
        model = YOLO(model_path)
        
        print(f"   Model class mapping:")
        for idx, name in model.names.items():
            expected = BDD_CLASSES[idx] if idx < len(BDD_CLASSES) else "N/A"
            match = "✓" if name == expected else "✗ MISMATCH!"
            print(f"      {idx}: {name:<20} (expected: {expected:<20}) {match}")
        print()
    except Exception as e:
        print(f"   ERROR: {e}")
        print()
    
    print("=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    print()
    print("LIKELY ISSUES:")
    print("1. Class name mismatch between model and evaluation script")
    print("2. Model predicting COCO classes instead of BDD100k classes")
    print("3. Empty predictions (no detections above confidence threshold)")
    print()
    print("SOLUTION:")
    print("Check the output above to see which class names don't match,")
    print("then we'll fix the evaluate.py script accordingly.")
    print()


if __name__ == "__main__":
    diagnose_model_and_data()
