"""
Create a subset of BDD100k dataset in YOLO format for quick training demos.

This script:
1. Randomly selects 300 images from the training set
2. Copies images and their corresponding YOLO format labels
3. Creates proper YOLO directory structure

Author: Bosch Assignment - Model Task
Date: November 2025
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import json


def create_subset_dataset(
    source_images_dir: str,
    source_labels_dir: str,
    output_dir: str,
    num_images: int = 300,
    seed: int = 42
) -> Tuple[int, int]:
    """
    Create a subset of the dataset in YOLO format.
    
    Args:
        source_images_dir: Path to source images directory
        source_labels_dir: Path to source YOLO labels directory
        output_dir: Path to output subset directory
        num_images: Number of images to include in subset
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (images_copied, labels_copied)
    """
    print("=" * 70)
    print("Creating BDD100k Subset Dataset".center(70))
    print("=" * 70)
    print()
    
    # Set random seed
    random.seed(seed)
    
    # Convert to Path objects
    source_images_dir = Path(source_images_dir)
    source_labels_dir = Path(source_labels_dir)
    output_dir = Path(output_dir)
    
    print(f"Configuration:")
    print(f"  Source images: {source_images_dir}")
    print(f"  Source labels: {source_labels_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Subset size: {num_images} images")
    print(f"  Random seed: {seed}")
    print()
    
    # Check if source directories exist
    if not source_images_dir.exists():
        print(f"ERROR: Source images directory not found: {source_images_dir}")
        return 0, 0
    
    if not source_labels_dir.exists():
        print(f"ERROR: Source labels directory not found: {source_labels_dir}")
        return 0, 0
    
    # Create output directory structure
    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/4] Scanning source directory...")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png'}
    all_images = [
        f for f in source_images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    print(f"  Found {len(all_images)} images in source directory")
    
    if len(all_images) == 0:
        print("ERROR: No images found in source directory!")
        return 0, 0
    
    # Randomly select subset
    print(f"\n[2/4] Selecting {num_images} random images...")
    
    if len(all_images) < num_images:
        print(f"  WARNING: Only {len(all_images)} images available, using all of them")
        selected_images = all_images
    else:
        selected_images = random.sample(all_images, num_images)
    
    print(f"  Selected {len(selected_images)} images")
    
    # Copy images and labels
    print(f"\n[3/4] Copying images and labels...")
    
    images_copied = 0
    labels_copied = 0
    labels_missing = 0
    
    for img_path in selected_images:
        # Copy image
        dest_img_path = output_images_dir / img_path.name
        shutil.copy2(img_path, dest_img_path)
        images_copied += 1
        
        # Copy corresponding label
        label_name = img_path.stem + '.txt'
        source_label_path = source_labels_dir / label_name
        
        if source_label_path.exists():
            dest_label_path = output_labels_dir / label_name
            shutil.copy2(source_label_path, dest_label_path)
            labels_copied += 1
        else:
            # Create empty label file (no objects in image)
            dest_label_path = output_labels_dir / label_name
            dest_label_path.touch()
            labels_missing += 1
        
        # Progress indicator
        if (images_copied % 50) == 0:
            print(f"  Progress: {images_copied}/{len(selected_images)} images copied")
    
    print(f"  ✓ Copied {images_copied} images")
    print(f"  ✓ Copied {labels_copied} labels")
    if labels_missing > 0:
        print(f"  ⚠ Created {labels_missing} empty label files (no objects)")
    
    # Create data.yaml for this subset
    print(f"\n[4/4] Creating data.yaml configuration...")
    
    yaml_content = f"""# BDD100k Subset Dataset Configuration
# Auto-generated for training demo

# Dataset root (absolute path)
path: {output_dir.absolute()}

# Relative paths from 'path' above
train: .
val: .  # Using same data for validation in demo

# Class names (10 detection classes)
names:
  - person
  - rider
  - car
  - truck
  - bus
  - train
  - motorcycle
  - bicycle
  - traffic light
  - traffic sign

# Number of classes
nc: 10

# Image size
imgsz: 640

# Dataset info
dataset: BDD100k_Subset
task: detect
subset_size: {len(selected_images)}
description: Subset of BDD100k for quick training demos
"""
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"  ✓ Created {yaml_path}")
    
    # Create README
    readme_content = f"""# BDD100k Subset Dataset

This is a subset of the BDD100k dataset for quick training demos.

## Statistics
- **Images**: {images_copied}
- **Labels**: {labels_copied}
- **Empty labels**: {labels_missing}
- **Classes**: 10 (person, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign)

## Structure
```
{output_dir.name}/
├── data.yaml          # Dataset configuration
├── images/            # {images_copied} images
├── labels/            # {labels_copied + labels_missing} label files
└── README.md          # This file
```

## Usage

```bash
# Train YOLOv8 on this subset
cd model/src/
python train.py --data ../subset_300/data.yaml --epochs 1 --batch 8
```

## Created
- Date: {Path(__file__).stat().st_mtime}
- Random seed: {seed}
- Source: BDD100k training set
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"  ✓ Created {readme_path}")
    
    print()
    print("=" * 70)
    print("Subset Dataset Creation Complete!".center(70))
    print("=" * 70)
    print()
    print(f"Summary:")
    print(f"  Images copied: {images_copied}")
    print(f"  Labels copied: {labels_copied}")
    print(f"  Output directory: {output_dir.absolute()}")
    print(f"  Config file: {yaml_path.absolute()}")
    print()
    print(f"To use this subset for training:")
    print(f"  cd model/src/")
    print(f"  python train.py --data {yaml_path.relative_to(Path.cwd())} --epochs 1")
    print()
    
    return images_copied, labels_copied


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create a subset of BDD100k dataset for quick training'
    )
    parser.add_argument(
        '--source-images',
        type=str,
        default='../../data_analysis/data/bdd100k_yolo_dataset/train/images',
        help='Path to source images directory'
    )
    parser.add_argument(
        '--source-labels',
        type=str,
        default='../../data_analysis/data/bdd100k_yolo_dataset/train/labels',
        help='Path to source YOLO labels directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../subset_300',
        help='Output directory for subset'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=300,
        help='Number of images in subset'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).resolve().parent
    
    source_images = Path(args.source_images)
    if not source_images.is_absolute():
        source_images = (script_dir / source_images).resolve()
    
    source_labels = Path(args.source_labels)
    if not source_labels.is_absolute():
        source_labels = (script_dir / source_labels).resolve()
    
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = (script_dir.parent / args.output).resolve()
    
    # Create subset
    images_copied, labels_copied = create_subset_dataset(
        source_images_dir=str(source_images),
        source_labels_dir=str(source_labels),
        output_dir=str(output_dir),
        num_images=args.num_images,
        seed=args.seed
    )
    
    if images_copied > 0:
        print("✓ Subset creation successful!")
    else:
        print("✗ Subset creation failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
