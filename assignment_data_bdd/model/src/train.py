"""
Training Pipeline for YOLOv8 on BDD100k

This module demonstrates end-to-end training for 1 epoch on a subset of data.
For the assignment, this shows that the training pipeline works correctly.

Author: Bosch Assignment - Model Task
Date: November 2025
"""

import torch
import time
from pathlib import Path
from typing import Optional, Dict, Any
from ultralytics import YOLO
import sys


def train_one_epoch_demo(
    model_path: str = 'yolov8m.pt',
    data_yaml_path: str = 'configs/bdd100k.yaml',
    subset_size: int = 300,
    batch_size: int = 8,
    img_size: int = 640,
    device: str = 'auto',
    output_dir: str = '../outputs/training_logs',
    use_subset: bool = True
) -> Dict[str, Any]:
    """
    Train YOLOv8 for 1 epoch on a subset of BDD100k data.
    
    This demonstrates that the training pipeline works end-to-end
    without requiring days of training.
    
    Args:
        model_path: Path to pre-trained model weights (or model variant name)
        data_yaml_path: Path to dataset YAML configuration
        subset_size: Number of images to use for demo (default: 300)
        batch_size: Training batch size (default: 8)
        img_size: Input image size (default: 640)
        device: Device to train on ('auto', 'cuda', 'cpu')
        output_dir: Directory to save training outputs
        use_subset: Whether to use/create the 300-image subset (default: True)
    
    Returns:
        Dictionary containing training results and metrics
    """
    print("=" * 70)
    print("YOLOv8 Training Demo - 1 Epoch on Subset".center(70))
    print("=" * 70)
    print()
    
    # Convert relative paths to absolute paths
    script_dir = Path(__file__).resolve().parent  # /model/src/
    model_dir = script_dir.parent                  # /model/
    
    # If use_subset is True, use the subset_300 directory
    if use_subset:
        subset_dir = model_dir / 'subset_300'
        subset_yaml = subset_dir / 'data.yaml'
        
        # Check if subset exists, if not create it
        if not subset_yaml.exists():
            print(f"[INFO] Subset dataset not found at: {subset_dir}")
            print(f"[INFO] Creating {subset_size}-image subset dataset...")
            print()
            
            # Import and run the subset creation
            try:
                from create_subset_dataset import create_subset_dataset
                
                # Default source paths
                source_images = model_dir.parent / 'data_analysis' / 'data' / 'bdd100k_yolo_dataset' / 'train' / 'images'
                source_labels = model_dir.parent / 'data_analysis' / 'data' / 'bdd100k_yolo_dataset' / 'train' / 'labels'
                
                images_copied, labels_copied = create_subset_dataset(
                    source_images_dir=str(source_images),
                    source_labels_dir=str(source_labels),
                    output_dir=str(subset_dir),
                    num_images=subset_size,
                    seed=42
                )
                
                if images_copied == 0:
                    print(f"[ERROR] Failed to create subset dataset")
                    print(f"[INFO] Falling back to original data YAML path")
                else:
                    print(f"[INFO] Successfully created subset with {images_copied} images")
                    data_yaml_path = str(subset_yaml)
                    print()
            except Exception as e:
                print(f"[ERROR] Failed to create subset: {e}")
                print(f"[INFO] You can manually create the subset by running:")
                print(f"       cd {script_dir}")
                print(f"       python create_subset_dataset.py")
                print()
        else:
            print(f"[INFO] Using existing subset dataset: {subset_yaml}")
            data_yaml_path = str(subset_yaml)
            print()
    
    # Resolve data YAML path - check multiple possible locations
    if not Path(data_yaml_path).is_absolute():
        # First try: relative to model directory
        yaml_candidate1 = (model_dir / data_yaml_path).resolve()
        # Second try: relative to current working directory
        yaml_candidate2 = Path(data_yaml_path).resolve()
        
        if yaml_candidate1.exists():
            data_yaml_path = yaml_candidate1
        elif yaml_candidate2.exists():
            data_yaml_path = yaml_candidate2
        else:
            # Default to model directory path
            data_yaml_path = yaml_candidate1
    else:
        data_yaml_path = Path(data_yaml_path).resolve()
    
    # Resolve output directory
    if not Path(output_dir).is_absolute():
        # First try: relative to model directory
        output_candidate1 = (model_dir / output_dir).resolve()
        # Second try: relative to current working directory  
        output_candidate2 = Path(output_dir).resolve()
        
        # Use the one relative to model dir
        output_dir = output_candidate1
    else:
        output_dir = Path(output_dir).resolve()
    
    # Resolve model path (only if it's not just a model name like 'yolov8m.pt')
    if '/' in model_path or Path(model_path).suffix == '.pt':
        # Check if it's in the model directory
        if not Path(model_path).is_absolute():
            model_candidate1 = (model_dir / model_path).resolve()
            model_candidate2 = Path(model_path).resolve()
            
            if model_candidate1.exists():
                model_path = str(model_candidate1)
            elif model_candidate2.exists():
                model_path = str(model_candidate2)
            # else: let YOLO download it (for model names like 'yolov8m.pt')
        else:
            model_path = str(Path(model_path).resolve())
    
    # Convert paths to strings
    data_yaml_path = str(data_yaml_path)
    output_dir = str(output_dir)
    
    # Detect device
    if device == 'auto':
        # Check for MPS (Apple Silicon GPU), then CUDA, then CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    print(f"[INFO] Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Dataset: {data_yaml_path}")
    print(f"  Using subset: {use_subset} ({subset_size} images)")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Device: {device}")
    print(f"  Output dir: {output_dir}")
    print()
    
    # Check if data YAML exists
    if not Path(data_yaml_path).exists():
        print(f"[WARNING] Data YAML not found at: {data_yaml_path}")
        print("[WARNING] Please ensure the dataset configuration is set up correctly.")
        print()
        print("[INFO] To create the subset dataset manually, run:")
        print(f"       cd {script_dir}")
        print(f"       python create_subset_dataset.py --num-images {subset_size}")
        print()
        return {
            "success": False,
            "error": "Dataset YAML not found",
            "yaml_path": data_yaml_path
        }
    
    # Initialize YOLOv8 model
    print(f"[1/4] Loading YOLOv8 model...")
    start_time = time.time()
    
    try:
        model = YOLO(model_path)
        print(f"  ✓ Model loaded successfully in {time.time() - start_time:.2f}s")
        print(f"  ✓ Model type: {model.model_name if hasattr(model, 'model_name') else 'YOLOv8'}")
        print(f"  ✓ Device: {device}")
        print()
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        print()
        print("[ERROR] Please ensure YOLOv8 is installed:")
        print("  pip install ultralytics")
        return {"success": False, "error": str(e)}
    
    # Setup training configuration
    print(f"[2/4] Configuring training parameters...")
    train_config = {
        'data': data_yaml_path,
        'epochs': 1,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'workers': 4,
        'project': output_dir,
        'name': 'yolov8m_bdd100k_subset_demo',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,  # Lower LR for fine-tuning
        'verbose': True,
        'patience': 0,  # No early stopping for demo
        'save': True,
        'save_period': -1,  # Don't save intermediate checkpoints
        'val': True,
        'plots': True,
    }
    
    print("  Training configuration:")
    for key, value in train_config.items():
        if key != 'data':  # Don't print full path
            print(f"    {key}: {value}")
    print()
    
    # Train for 1 epoch
    print(f"[3/4] Starting training for 1 epoch on {subset_size} images...")
    print("-" * 70)
    
    try:
        results = model.train(**train_config)
        print("-" * 70)
        print(f"  ✓ Training completed successfully!")
        print()
        
        # Extract metrics
        if hasattr(results, 'results_dict'):
            print("[4/4] Training Results:")
            metrics = results.results_dict
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
        # Save model weights
        output_path = Path(output_dir) / 'yolov8m_bdd100k_subset_demo' / 'weights' / 'best.pt'
        if output_path.exists():
            print()
            print(f"  ✓ Model weights saved to: {output_path}")
        
        print()
        print("=" * 70)
        print("Training demo completed successfully!".center(70))
        print("=" * 70)
        
        return {
            "success": True,
            "model_loaded": True,
            "training_executed": True,
            "subset_used": use_subset,
            "subset_size": subset_size,
            "results": results,
            "output_dir": str(output_path.parent) if output_path.exists() else None
        }
        
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        print()
        print("[ERROR] Training encountered an error.")
        print(f"[ERROR] Details: {e}")
        return {
            "success": False,
            "error": str(e),
            "training_executed": False
        }


def train_full(
    model_path: str = 'yolov8m.pt',
    data_yaml_path: str = 'configs/bdd100k.yaml',
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = 'auto',
    output_dir: str = '../outputs/training_logs',
    resume: bool = False
) -> Dict[str, Any]:
    """
    Full training pipeline for YOLOv8 on BDD100k.
    
    This is for actual model training (not just demo).
    
    Args:
        model_path: Path to pre-trained model weights
        data_yaml_path: Path to dataset YAML configuration
        epochs: Number of training epochs
        batch_size: Training batch size
        img_size: Input image size
        device: Device to train on
        output_dir: Directory to save training outputs
        resume: Whether to resume from last checkpoint
    
    Returns:
        Dictionary containing training results
    """
    print("=" * 70)
    print(f"YOLOv8 Full Training - {epochs} Epochs".center(70))
    print("=" * 70)
    print()
    
    # Convert relative paths to absolute paths
    script_dir = Path(__file__).resolve().parent  # /model/src/
    model_dir = script_dir.parent                  # /model/
    
    # Resolve data YAML path - check multiple possible locations
    if not Path(data_yaml_path).is_absolute():
        # First try: relative to model directory
        yaml_candidate1 = (model_dir / data_yaml_path).resolve()
        # Second try: relative to current working directory
        yaml_candidate2 = Path(data_yaml_path).resolve()
        
        if yaml_candidate1.exists():
            data_yaml_path = yaml_candidate1
        elif yaml_candidate2.exists():
            data_yaml_path = yaml_candidate2
        else:
            # Default to model directory path
            data_yaml_path = yaml_candidate1
    else:
        data_yaml_path = Path(data_yaml_path).resolve()
    
    # Resolve output directory
    if not Path(output_dir).is_absolute():
        # First try: relative to model directory
        output_candidate1 = (model_dir / output_dir).resolve()
        # Second try: relative to current working directory  
        output_candidate2 = Path(output_dir).resolve()
        
        # Use the one relative to model dir
        output_dir = output_candidate1
    else:
        output_dir = Path(output_dir).resolve()
    
    # Resolve model path (only if it's not just a model name like 'yolov8m.pt')
    if '/' in model_path or Path(model_path).suffix == '.pt':
        # Check if it's in the model directory
        if not Path(model_path).is_absolute():
            model_candidate1 = (model_dir / model_path).resolve()
            model_candidate2 = Path(model_path).resolve()
            
            if model_candidate1.exists():
                model_path = str(model_candidate1)
            elif model_candidate2.exists():
                model_path = str(model_candidate2)
            # else: let YOLO download it (for model names like 'yolov8m.pt')
        else:
            model_path = str(Path(model_path).resolve())
    
    # Convert paths to strings
    data_yaml_path = str(data_yaml_path)
    output_dir = str(output_dir)
    
    # Auto-detect device
    if device == 'auto':
        # Check for MPS (Apple Silicon GPU), then CUDA, then CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    print(f"Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Data YAML: {data_yaml_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Device: {device}")
    print(f"  Resume: {resume}")
    print()
    
    # Load model
    model = YOLO(model_path)
    
    # Training configuration
    train_config = {
        'data': data_yaml_path,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'workers': 8,
        'project': output_dir,
        'name': 'yolov8m_bdd100k_full',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'verbose': True,
        'patience': 50,
        'save': True,
        'save_period': 10,
        'val': True,
        'plots': True,
        'resume': resume
    }
    
    print("Starting full training...")
    print("-" * 70)
    
    # Train
    results = model.train(**train_config)
    
    print("-" * 70)
    print("Training completed!")
    print()
    
    return {
        "success": True,
        "results": results,
        "epochs": epochs
    }


def validate_model(
    model_path: str,
    data_yaml_path: str,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = 'auto'
) -> Dict[str, Any]:
    """
    Validate trained model on validation set.
    
    Args:
        model_path: Path to trained model weights
        data_yaml_path: Path to dataset YAML configuration
        batch_size: Validation batch size
        img_size: Input image size
        device: Device to run validation on
    
    Returns:
        Dictionary containing validation metrics
    """
    print("=" * 70)
    print("Model Validation".center(70))
    print("=" * 70)
    print()
    
    if device == 'auto':
        # Check for MPS (Apple Silicon GPU), then CUDA, then CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    # Load model
    model = YOLO(model_path)
    
    print(f"Running validation...")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}")
    print()
    
    # Validate
    results = model.val(
        data=data_yaml_path,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        verbose=True
    )
    
    print()
    print("Validation Results:")
    if hasattr(results, 'results_dict'):
        for key, value in results.results_dict.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    
    return {
        "success": True,
        "results": results
    }


if __name__ == "__main__":
    """
    Main entry point for training script.
    
    Usage:
        # Demo training (1 epoch, 100 images)
        python train.py
        
        # Full training
        python train.py --full --epochs 50
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 on BDD100k')
    parser.add_argument('--model', type=str, default='yolov8m.pt', 
                        help='Model variant or path to weights')
    parser.add_argument('--data', type=str, default='../configs/bdd100k.yaml',
                        help='Path to dataset YAML')
    parser.add_argument('--full', action='store_true',default = True,
                        help='Run full training (not just demo)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/cpu)')
    parser.add_argument('--subset', type=int, default=100,
                        help='Subset size for demo')
    
    args = parser.parse_args()
    
    if args.full:
        # Full training
        results = train_full(
            model_path=args.model,
            data_yaml_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            device=args.device
        )
    else:
        # Demo training
        results = train_one_epoch_demo(
            model_path=args.model,
            data_yaml_path=args.data,
            subset_size=args.subset,
            batch_size=args.batch,
            img_size=args.imgsz,
            device=args.device
        )
    
    # Print summary
    print()
    if results.get('success'):
        print("✓ Script completed successfully!")
    else:
        print("✗ Script failed!")
        if 'error' in results:
            print(f"Error: {results['error']}")
