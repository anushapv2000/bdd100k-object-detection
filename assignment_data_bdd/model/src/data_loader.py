"""
BDD100k Dataset Loader for YOLOv8

This module provides a custom PyTorch Dataset class for loading
BDD100k object detection data and converting it to YOLO format.

Author: Bosch Assignment - Model Task
Date: November 2025
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class BDD100kDataset(Dataset):
    """
    PyTorch Dataset for BDD100k object detection.
    
    Converts BDD100k JSON format to YOLO format:
    - Bounding boxes: [x_center, y_center, width, height] (normalized 0-1)
    - Class labels: integer indices
    
    Attributes:
        CLASSES (list): List of 10 BDD100k detection classes
        images_dir (Path): Directory containing images
        img_size (int): Target image size (square)
        samples (list): List of valid image samples with annotations
        class_to_idx (dict): Mapping from class name to index
    """
    
    # BDD100k class names (10 detection classes)
    CLASSES = [
        'bike', 'bus', 'car', 'motor', 'person',
        'rider', 'traffic light', 'traffic sign', 'train', 'truck'
    ]
    
    def __init__(
        self,
        images_dir: str,
        labels_path: str,
        img_size: int = 640,
        subset_size: Optional[int] = None
    ):
        """
        Initialize BDD100k dataset.
        
        Args:
            images_dir: Path to images directory
            labels_path: Path to labels JSON file
            img_size: Target image size (square) for resizing
            subset_size: If specified, only use first N samples (for quick testing)
        """
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        
        # Validate paths
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        if not Path(labels_path).exists():
            raise ValueError(f"Labels file not found: {labels_path}")
        
        # Load labels from JSON
        print(f"Loading labels from: {labels_path}")
        with open(labels_path, 'r') as f:
            self.labels_data = json.load(f)
        
        # Filter to only images with box2d annotations
        self.samples = []
        for item in self.labels_data:
            # Check if image has at least one box2d annotation
            has_boxes = any(
                'box2d' in label 
                for label in item.get('labels', [])
            )
            if has_boxes:
                self.samples.append(item)
        
        # Apply subset if specified (for quick testing)
        if subset_size is not None:
            self.samples = self.samples[:subset_size]
            print(f"Using subset of {subset_size} images")
        
        # Create class name to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}
        
        print(f"Dataset initialized:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Image size: {self.img_size}x{self.img_size}")
        print(f"  Classes: {len(self.CLASSES)}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
        
        Returns:
            image: Tensor of shape (3, img_size, img_size), normalized to [0, 1]
            target: Dictionary containing:
                - 'boxes': Tensor of shape (N, 4) in format [x_center, y_center, w, h]
                - 'labels': Tensor of shape (N,) with class indices
                - 'image_id': String image identifier
                - 'orig_size': Tuple (orig_height, orig_width)
        """
        sample = self.samples[idx]
        
        # Load image
        img_name = sample['name']
        img_path = self.images_dir / img_name
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
        
        orig_width, orig_height = image.size
        
        # Parse bounding boxes
        boxes = []
        labels = []
        
        for label in sample.get('labels', []):
            if 'box2d' not in label:
                continue
            
            category = label['category']
            
            # Skip if category not in our class list
            if category not in self.class_to_idx:
                continue
            
            box2d = label['box2d']
            x1, y1 = box2d['x1'], box2d['y1']
            x2, y2 = box2d['x2'], box2d['y2']
            
            # Validate box coordinates
            if x2 <= x1 or y2 <= y1:
                continue  # Skip invalid boxes
            
            # Convert to YOLO format: [x_center, y_center, width, height] (normalized)
            x_center = ((x1 + x2) / 2) / orig_width
            y_center = ((y1 + y2) / 2) / orig_height
            width = (x2 - x1) / orig_width
            height = (y2 - y1) / orig_height
            
            # Clip to [0, 1] to handle edge cases
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            width = np.clip(width, 0, 1)
            height = np.clip(height, 0, 1)
            
            boxes.append([x_center, y_center, width, height])
            labels.append(self.class_to_idx[category])
        
        # Resize image to target size (letterbox padding to maintain aspect ratio)
        image = self._letterbox_resize(image, self.img_size)
        
        # Convert image to tensor and normalize to [0, 1]
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
        # Convert boxes and labels to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            # No boxes in this image (shouldn't happen due to filtering)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': img_name,
            'orig_size': (orig_height, orig_width)
        }
        
        return image, target
    
    def _letterbox_resize(self, image: Image.Image, target_size: int) -> Image.Image:
        """
        Resize image with letterbox padding to maintain aspect ratio.
        
        Args:
            image: PIL Image
            target_size: Target size (square)
        
        Returns:
            Resized and padded image
        """
        orig_width, orig_height = image.size
        
        # Calculate scaling factor
        scale = min(target_size / orig_width, target_size / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Resize image
        image = image.resize((new_width, new_height), Image.BILINEAR)
        
        # Create new image with padding
        new_image = Image.new('RGB', (target_size, target_size), (114, 114, 114))
        
        # Calculate padding offsets (center the image)
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        
        # Paste resized image onto padded canvas
        new_image.paste(image, (paste_x, paste_y))
        
        return new_image
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get distribution of classes in the dataset.
        
        Returns:
            Dictionary mapping class names to counts
        """
        class_counts = {cls: 0 for cls in self.CLASSES}
        
        for sample in self.samples:
            for label in sample.get('labels', []):
                category = label.get('category')
                if category in class_counts:
                    class_counts[category] += 1
        
        return class_counts


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Custom collate function for DataLoader.
    
    Handles variable number of objects per image by keeping targets as a list.
    
    Args:
        batch: List of (image, target) tuples
    
    Returns:
        images: Stacked tensor of shape (B, 3, H, W)
        targets: List of target dictionaries (length B)
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack images into batch
    images = torch.stack(images, dim=0)
    
    return images, targets


def create_data_loaders(
    train_images_dir: str,
    train_labels_path: str,
    val_images_dir: str,
    val_labels_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 640,
    subset_size: Optional[int] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        train_images_dir: Path to training images directory
        train_labels_path: Path to training labels JSON file
        val_images_dir: Path to validation images directory
        val_labels_path: Path to validation labels JSON file
        batch_size: Batch size for both loaders
        num_workers: Number of data loading workers
        img_size: Target image size (square)
        subset_size: If specified, use subset for quick testing
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    print("Creating datasets...")
    
    # Create datasets
    train_dataset = BDD100kDataset(
        images_dir=train_images_dir,
        labels_path=train_labels_path,
        img_size=img_size,
        subset_size=subset_size
    )
    
    val_dataset = BDD100kDataset(
        images_dir=val_images_dir,
        labels_path=val_labels_path,
        img_size=img_size,
        subset_size=subset_size
    )
    
    print(f"\nTrain dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def test_data_loader(
    images_dir: str,
    labels_path: str,
    num_samples: int = 5
):
    """
    Test the data loader by loading a few samples and printing statistics.
    
    Args:
        images_dir: Path to images directory
        labels_path: Path to labels JSON file
        num_samples: Number of samples to test
    """
    print("=" * 60)
    print("Testing BDD100k Data Loader")
    print("=" * 60)
    
    # Create dataset
    dataset = BDD100kDataset(
        images_dir=images_dir,
        labels_path=labels_path,
        img_size=640,
        subset_size=num_samples
    )
    
    print(f"\nTesting {num_samples} samples...\n")
    
    for i in range(min(num_samples, len(dataset))):
        image, target = dataset[i]
        
        print(f"Sample {i+1}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Num objects: {len(target['labels'])}")
        print(f"  Classes: {target['labels'].tolist()}")
        print(f"  Boxes shape: {target['boxes'].shape}")
        print(f"  Image ID: {target['image_id']}")
        print(f"  Original size: {target['orig_size']}")
        print()
    
    # Print class distribution
    print("Class Distribution:")
    class_dist = dataset.get_class_distribution()
    for cls, count in class_dist.items():
        print(f"  {cls}: {count}")
    
    print("\n" + "=" * 60)
    print("Data loader test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Test with sample data
    TRAIN_IMAGES = "../../assignment_data_bdd/data/bdd100k_yolo_dataset/train/images/"
    TRAIN_LABELS = "../../assignment_data_bdd/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
    
    test_data_loader(TRAIN_IMAGES, TRAIN_LABELS, num_samples=5)
