# Training Strategy for YOLOv8 on BDD100k

## Overview

This document outlines the training strategy for fine-tuning YOLOv8m on the BDD100k dataset for object detection. The strategy is designed to maximize performance while being practical for the assignment's time constraints.

---

## 1. Training Approach

### 1.1 Transfer Learning Strategy

**Why Transfer Learning?**
- BDD100k is a domain-specific dataset (autonomous driving)
- Pre-trained ImageNet/COCO weights provide good initialization
- Significantly reduces training time (hours instead of days)
- Achieves better performance with limited data

**Approach**: Fine-tune pre-trained YOLOv8m model
1. Start with COCO pre-trained weights
2. Replace final classification layer (80 classes → 10 classes)
3. Fine-tune all layers with lower learning rate
4. Use BDD100k-specific data augmentations

---

## 2. Dataset Preparation

### 2.1 Data Format Conversion

**BDD100k Format → YOLO Format**

BDD100k JSON:
```json
{
  "name": "image_001.jpg",
  "labels": [
    {
      "category": "car",
      "box2d": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}
    }
  ]
}
```

YOLO Format (per image .txt file):
```
class_id x_center y_center width height
0 0.5 0.6 0.3 0.4  # normalized [0, 1]
```

### 2.2 Class Mapping

| BDD100k Class | Class ID | YOLO Index |
|---------------|----------|------------|
| bike | 0 | 0 |
| bus | 1 | 1 |
| car | 2 | 2 |
| motor | 3 | 3 |
| person | 4 | 4 |
| rider | 5 | 5 |
| traffic light | 6 | 6 |
| traffic sign | 7 | 7 |
| train | 8 | 8 |
| truck | 9 | 9 |

---

## 3. Hyperparameters

### 3.1 Core Training Parameters

```yaml
# Model
model: yolov8m.pt  # Pre-trained weights
imgsz: 640         # Input image size

# Training
epochs: 50         # Full training (for demo: 1 epoch)
batch: 16          # Batch size (adjust based on GPU memory)
workers: 8         # Number of data loading workers

# Optimizer
optimizer: AdamW   # Adam with weight decay
lr0: 0.001         # Initial learning rate (lower for fine-tuning)
lrf: 0.01          # Final learning rate (lr0 * lrf)
momentum: 0.937    # SGD momentum/Adam beta1
weight_decay: 0.0005  # Weight decay

# Scheduler
warmup_epochs: 3   # Warmup epochs
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# Loss weights
box: 7.5           # Box loss gain
cls: 0.5           # Class loss gain
dfl: 1.5           # Distribution focal loss gain
```

### 3.2 Learning Rate Strategy

**Cosine Annealing with Warmup**

```
LR
│     ┌─────────────────────────────┐
│    /                               \
│   /                                 \
│  /                                   \___
│ /                                        
└─────────────────────────────────────────→ Epoch
  0   3                               50
  └─┘ └───────────────────────────────┘
Warmup      Cosine Annealing
```

**Rationale**:
- **Warmup (0-3 epochs)**: Gradually increase LR to stabilize training
- **Cosine decay (3-50 epochs)**: Smooth decrease for convergence
- **Final LR**: 1% of initial (0.00001) for fine-grained optimization

---

## 4. Data Augmentation

### 4.1 Augmentation Pipeline

```yaml
# Geometric Augmentations
hsv_h: 0.015       # HSV-Hue augmentation
hsv_s: 0.7         # HSV-Saturation augmentation
hsv_v: 0.4         # HSV-Value augmentation
degrees: 0.0       # Rotation (+/- deg)
translate: 0.1     # Translation (+/- fraction)
scale: 0.5         # Scaling (+/- gain)
shear: 0.0         # Shear (+/- deg)
perspective: 0.0   # Perspective (+/- fraction)
flipud: 0.0        # Vertical flip (probability)
fliplr: 0.5        # Horizontal flip (probability)

# Advanced Augmentations
mosaic: 1.0        # Mosaic augmentation (probability)
mixup: 0.1         # MixUp augmentation (probability)
copy_paste: 0.0    # Copy-paste augmentation (probability)
```

### 4.2 Augmentation Strategies Explained

#### a) **Mosaic Augmentation**
- Combines 4 images into one
- Forces model to learn partial objects
- Improves small object detection
- Applied to 100% of training images

```
┌─────────┬─────────┐
│ Image 1 │ Image 2 │
├─────────┼─────────┤
│ Image 3 │ Image 4 │
└─────────┴─────────┘
```

#### b) **MixUp**
- Blends two images with alpha blending
- Regularization technique
- Applied to 10% of images

```
Output = α × Image1 + (1-α) × Image2
```

#### c) **HSV Augmentation**
- Simulates different lighting conditions
- Important for day/night/weather variations
- BDD100k-specific: helps with diverse conditions

#### d) **Horizontal Flip**
- 50% probability
- Natural for driving scenes (left/right symmetry)
- No vertical flip (cars don't appear upside down)

---

## 5. Training Loop Design

### 5.1 One Epoch Training Flow

```python
for epoch in range(num_epochs):
    # 1. Training Phase
    model.train()
    for batch_idx, (images, targets) in enumerate(train_loader):
        # Forward pass
        predictions = model(images)
        
        # Compute loss
        loss, loss_items = compute_loss(predictions, targets)
        # loss_items: [box_loss, cls_loss, dfl_loss]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log metrics
        if batch_idx % log_interval == 0:
            print(f"Loss: {loss.item():.4f}")
    
    # 2. Validation Phase
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            predictions = model(images)
            val_loss = compute_loss(predictions, targets)
    
    # 3. Update learning rate
    scheduler.step()
    
    # 4. Save checkpoint
    if epoch % save_interval == 0:
        save_checkpoint(model, optimizer, epoch)
```

### 5.2 Loss Computation

```python
def compute_loss(predictions, targets):
    """
    Compute total loss for YOLOv8.
    
    Args:
        predictions: Model output [batch, 8400, 15]
        targets: Ground truth [num_targets, 6]
                 Each row: [img_idx, class, x, y, w, h]
    
    Returns:
        total_loss: Sum of all losses
        loss_items: [box_loss, cls_loss, dfl_loss]
    """
    # 1. Assign targets to predictions
    matched_preds, matched_targets = task_aligned_assigner(
        predictions, targets
    )
    
    # 2. Classification loss (BCE)
    cls_loss = bce_loss(
        matched_preds['classes'],
        matched_targets['classes']
    )
    
    # 3. Box regression loss (CIoU)
    box_loss = ciou_loss(
        matched_preds['boxes'],
        matched_targets['boxes']
    )
    
    # 4. Distribution focal loss
    dfl_loss = distribution_focal_loss(
        matched_preds['box_distributions'],
        matched_targets['boxes']
    )
    
    # 5. Weighted sum
    total_loss = (
        0.5 * cls_loss +
        7.5 * box_loss +
        1.5 * dfl_loss
    )
    
    return total_loss, [box_loss, cls_loss, dfl_loss]
```

---

## 6. Hardware Requirements & Optimization

### 6.1 Recommended Hardware

**Minimum**:
- GPU: NVIDIA GTX 1660 Ti (6GB VRAM)
- RAM: 16 GB
- Storage: 50 GB SSD
- Batch size: 8

**Recommended**:
- GPU: NVIDIA RTX 3090 (24GB VRAM) or better
- RAM: 32 GB
- Storage: 100 GB NVMe SSD
- Batch size: 16-32

**For Demo (1 epoch)**:
- GPU: Any CUDA-capable GPU
- Can use CPU (slow, ~30min per epoch)
- Subset: 100 images, batch size 8

### 6.2 Training Time Estimates

| Setup | Full Dataset | Subset (100 imgs) |
|-------|-------------|-------------------|
| RTX 3090 (batch=16) | ~6 hours (50 epochs) | ~2 minutes |
| GTX 1660 Ti (batch=8) | ~15 hours | ~5 minutes |
| CPU (batch=4) | ~100 hours | ~30 minutes |

### 6.3 Memory Optimization

**Techniques to reduce memory usage**:

```yaml
# Mixed Precision Training (FP16)
amp: true  # Automatic Mixed Precision
# Reduces memory by ~40%

# Gradient Accumulation
accumulate: 2  # Accumulate gradients over N batches
# Effectively doubles batch size without extra memory

# Image Resolution
imgsz: 640  # Standard
imgsz: 512  # Lower memory (faster, slightly less accurate)

# Model Variant
model: yolov8n.pt  # Nano (3.2M params) - lowest memory
model: yolov8s.pt  # Small (11.2M params)
model: yolov8m.pt  # Medium (25.9M params) - recommended
```

---

## 7. Monitoring & Logging

### 7.1 Metrics to Track

**Training Metrics**:
- `train/box_loss`: Bounding box regression loss
- `train/cls_loss`: Classification loss
- `train/dfl_loss`: Distribution focal loss
- `train/total_loss`: Combined loss

**Validation Metrics**:
- `val/box_loss`: Box loss on validation set
- `val/cls_loss`: Classification loss on validation set
- `metrics/mAP50`: mAP at IoU threshold 0.5
- `metrics/mAP50-95`: mAP averaged over IoU 0.5 to 0.95
- `metrics/precision`: Precision
- `metrics/recall`: Recall

**Per-Class Metrics**:
- `metrics/mAP50(bike)`: mAP for bike class
- `metrics/mAP50(car)`: mAP for car class
- ... (for all 10 classes)

### 7.2 TensorBoard Logging

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('outputs/training_logs')

# Log scalars
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('mAP/val', mAP, epoch)

# Log images with predictions
writer.add_image('Predictions', pred_img, epoch)

# Log learning rate
writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
```

### 7.3 Checkpointing Strategy

```python
# Save checkpoint every N epochs
if epoch % save_interval == 0:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'mAP': mAP,
    }
    torch.save(checkpoint, f'weights/yolov8m_epoch{epoch}.pt')

# Save best model
if mAP > best_mAP:
    best_mAP = mAP
    torch.save(checkpoint, 'weights/yolov8m_best.pt')

# Save last model (for resuming)
torch.save(checkpoint, 'weights/yolov8m_last.pt')
```

---

## 8. Class Imbalance Handling

### 8.1 Problem

BDD100k has significant class imbalance:
- **car**: ~47% of all annotations
- **train**: <0.1% of all annotations

### 8.2 Solutions

#### a) **Focal Loss** (Built into YOLOv8)
```python
# Focuses on hard examples
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
```

#### b) **Class Weights** (Optional)
```python
# Weight rare classes higher
class_weights = {
    'car': 1.0,      # Common
    'person': 1.5,   # Common
    'train': 5.0,    # Rare - upweight
    'bus': 3.0,      # Rare
}
```

#### c) **Sampling Strategy**
```python
# Oversample images with rare classes
def get_sampler(dataset):
    weights = []
    for item in dataset:
        # Higher weight if contains rare classes
        weight = 1.0
        if 'train' in item.labels:
            weight = 5.0
        weights.append(weight)
    
    return WeightedRandomSampler(weights, len(dataset))
```

---

## 9. Validation Strategy

### 9.1 Validation Frequency

- **During Training**: Every epoch
- **Metrics Computed**: mAP, Precision, Recall, Loss
- **Subset Validation**: Can use 1000 images for faster iteration

### 9.2 Validation Protocol

```python
@torch.no_grad()
def validate(model, val_loader, conf_thresh=0.001, iou_thresh=0.6):
    """
    Run validation and compute metrics.
    
    Args:
        model: YOLOv8 model
        val_loader: Validation data loader
        conf_thresh: Confidence threshold for predictions
        iou_thresh: IoU threshold for NMS
    
    Returns:
        metrics: Dictionary with mAP, precision, recall
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    for images, targets in val_loader:
        # Run inference
        predictions = model(images, conf=conf_thresh, iou=iou_thresh)
        
        all_predictions.extend(predictions)
        all_targets.extend(targets)
    
    # Compute COCO metrics
    metrics = compute_coco_metrics(all_predictions, all_targets)
    
    return metrics
```

---

## 10. One Epoch Demo Strategy

### 10.1 Purpose

Demonstrate that the training pipeline works end-to-end without requiring days of training.

### 10.2 Configuration

```yaml
# Demo Configuration
subset_size: 100          # Use only 100 training images
batch_size: 8             # Small batch for faster iteration
epochs: 1                 # Just 1 epoch
workers: 4                # Data loading workers
imgsz: 640                # Standard size
device: 'cuda'            # Use GPU if available

# Expected Runtime
# - GPU: ~2 minutes
# - CPU: ~20-30 minutes
```

### 10.3 Expected Behavior

**What to Observe**:
1. ✅ Model loads pre-trained weights successfully
2. ✅ Data loader loads images and labels correctly
3. ✅ Forward pass produces predictions
4. ✅ Loss is computed and backpropagated
5. ✅ Loss decreases over batches (even in 1 epoch)
6. ✅ Model can be saved and loaded

**Example Output**:
```
Epoch 1/1:
Batch 1/12: Loss=1.234 (box=0.45, cls=0.67, dfl=0.114)
Batch 2/12: Loss=1.198 (box=0.44, cls=0.64, dfl=0.118)
Batch 3/12: Loss=1.165 (box=0.43, cls=0.62, dfl=0.115)
...
Batch 12/12: Loss=1.087 (box=0.39, cls=0.58, dfl=0.117)

✓ Training completed!
✓ Loss decreased: 1.234 → 1.087 (-11.9%)
✓ Model saved to: weights/yolov8m_bdd100k_1epoch.pt
```

---

## 11. Full Training Strategy (If Extended)

### 11.1 Multi-Stage Training

**Stage 1: Freeze Backbone (Epochs 1-10)**
```python
# Freeze backbone, train only head
for param in model.backbone.parameters():
    param.requires_grad = False

# Train head with higher LR
optimizer = AdamW(model.head.parameters(), lr=0.01)
```

**Stage 2: Unfreeze All (Epochs 11-50)**
```python
# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Lower LR for fine-tuning
optimizer = AdamW(model.parameters(), lr=0.001)
```

### 11.2 Progressive Image Resizing

**Stage 1**: Train with 512×512 (epochs 1-20)
**Stage 2**: Train with 640×640 (epochs 21-50)

**Benefits**:
- Faster initial training
- Better generalization
- +1-2% mAP improvement

---

## 12. Troubleshooting

### 12.1 Common Issues

| Issue | Solution |
|-------|----------|
| **OOM (Out of Memory)** | Reduce batch size, use FP16, lower image size |
| **NaN Loss** | Lower learning rate, check data normalization |
| **No Loss Decrease** | Check data loader, verify labels format |
| **Low mAP** | Train longer, increase data augmentation |
| **Overfitting** | Add dropout, increase weight decay |

### 12.2 Debugging Checklist

- [ ] Data loader returns correct shapes
- [ ] Labels are in correct format [0-1] normalized
- [ ] Class IDs are in correct range [0-9]
- [ ] Images are RGB (not BGR)
- [ ] Loss values are reasonable (not NaN/Inf)
- [ ] Learning rate is appropriate
- [ ] GPU is being utilized

---

## 13. Best Practices

### 13.1 Do's ✅

- ✅ Use pre-trained weights (transfer learning)
- ✅ Start with recommended hyperparameters
- ✅ Monitor training curves (loss, mAP)
- ✅ Validate frequently
- ✅ Save checkpoints regularly
- ✅ Use data augmentation
- ✅ Test on subset before full training

### 13.2 Don'ts ❌

- ❌ Train from scratch (waste of time)
- ❌ Use very high learning rates
- ❌ Train without validation
- ❌ Ignore class imbalance
- ❌ Skip warmup epochs
- ❌ Forget to normalize inputs

---

## 14. References

### Papers
- **AdamW**: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (ICLR 2019)
- **Cosine Annealing**: Loshchilov & Hutter, "SGDR" (ICLR 2017)
- **Mosaic**: Bochkovskiy et al., "YOLOv4" (2020)
- **MixUp**: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)

### Resources
- **Ultralytics Training Guide**: https://docs.ultralytics.com/modes/train/
- **BDD100k Training Tips**: https://doc.bdd100k.com/
- **PyTorch Best Practices**: https://pytorch.org/tutorials/

---

**Document Version**: 1.0  
**Last Updated**: November 2025  
**Author**: Model Task - Bosch Assignment
