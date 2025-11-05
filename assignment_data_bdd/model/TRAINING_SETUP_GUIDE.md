# Training Setup Guide for BDD100k Object Detection

## 📋 Prerequisites Checklist

Before you start training, make sure you have:

### ✅ **1. Dataset** (YOU HAVE THIS)
- ✅ Training images: 69,863 images at `data_analysis/data/bdd100k_images_100k/bdd100k/images/100k/train/`
- ✅ Validation images: 10,000 images at `data_analysis/data/bdd100k_images_100k/bdd100k/images/100k/val/`
- ✅ Training labels: `data_analysis/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json`
- ✅ Validation labels: `data_analysis/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json`

### ⬇️ **2. Pre-trained Model Weights** (AUTO-DOWNLOADED)
The model weights will be automatically downloaded on first run. No manual download needed!

**What happens:**
```bash
# When you run training for the first time:
python train.py --model yolov8m.pt

# YOLOv8 will automatically:
# 1. Check if yolov8m.pt exists locally
# 2. If not found, download from Ultralytics GitHub (~52 MB)
# 3. Save to: ~/.cache/ultralytics/ or current directory
# 4. Start training
```

**Manual download (optional):**
If you want to download manually first:
```bash
cd model/weights/
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

### 📦 **3. Python Dependencies** (INSTALL THESE)
```bash
cd model/
pip install -r requirements.txt
```

**Key packages:**
- `ultralytics>=8.0.0` - YOLOv8 framework
- `torch>=2.0.0` - PyTorch
- `opencv-python>=4.8.0` - Image processing
- `tqdm` - Progress bars

### 💻 **4. Hardware Requirements**

**Minimum (for demo):**
- CPU: Any modern CPU
- RAM: 8 GB
- Storage: 10 GB free space
- Time: ~30 minutes for 1 epoch demo

**Recommended (for full training):**
- GPU: NVIDIA with 8+ GB VRAM (RTX 3060 Ti, RTX 3070, etc.)
- RAM: 16 GB
- Storage: 20 GB free space
- Time: ~6 hours for 50 epochs

---

## 🚀 Quick Start Training

### **Option 1: 1-Epoch Demo (2-5 minutes)**
Test that everything works without waiting hours:

```bash
cd model/src/
python train.py \
    --model yolov8m.pt \
    --data ../configs/bdd100k.yaml \
    --epochs 1 \
    --batch 8 \
    --subset 100 \
    --device auto
```

**What this does:**
- Uses only **100 training images** (subset)
- Trains for **1 epoch** (~2-5 minutes)
- Automatically detects GPU/CPU
- Shows that training pipeline works

**Expected output:**
```
YOLOv8 Training Demo - 1 Epoch on Subset
========================================

[1/4] Loading YOLOv8 model...
Downloading yolov8m.pt... (if first time)
  ✓ Model loaded successfully

[2/4] Configuring training parameters...
  epochs: 1
  batch: 8
  subset: 100 images

[3/4] Starting training...
Epoch 1/1: 100%|████████| 13/13 [00:45<00:00]
  box_loss: 0.450
  cls_loss: 0.620
  dfl_loss: 0.115

✓ Training completed!
✓ Model saved to: outputs/training_logs/.../weights/best.pt
```

### **Option 2: Full Training (6-15 hours)**
For actual model training on full dataset:

```bash
cd model/src/
python train.py --full \
    --model yolov8m.pt \
    --data ../configs/bdd100k.yaml \
    --epochs 50 \
    --batch 16 \
    --device cuda
```

**What this does:**
- Uses **all 69,863 training images**
- Trains for **50 epochs**
- Batch size 16 (adjust if GPU memory issues)
- Uses GPU (cuda) for faster training

**Time estimates:**
- RTX 3090: ~6 hours
- RTX 3070: ~9 hours
- GTX 1660 Ti: ~15 hours
- CPU: ~100 hours (not recommended)

---

## 🔧 Training Parameters Explained

### **Basic Parameters**

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--model` | Pre-trained model | `yolov8m.pt` | `yolov8m.pt` |
| `--data` | Dataset YAML path | `../configs/bdd100k.yaml` | Keep default |
| `--epochs` | Training epochs | 1 | 50 for full training |
| `--batch` | Batch size | 8 | 16 (GPU), 4 (CPU) |
| `--imgsz` | Image size | 640 | 640 |
| `--device` | Device | `auto` | `cuda` or `cpu` |

### **Advanced Parameters**

```bash
python train.py --full \
    --model yolov8m.pt \
    --data ../configs/bdd100k.yaml \
    --epochs 50 \
    --batch 16 \
    --imgsz 640 \
    --device cuda \
    --workers 8 \           # Data loading workers
    --lr0 0.001 \          # Initial learning rate
    --patience 10          # Early stopping patience
```

---

## 📊 What Gets Created During Training

```
model/outputs/training_logs/yolov8m_bdd100k_full/
├── weights/
│   ├── best.pt           # Best model (highest mAP)
│   ├── last.pt           # Last epoch checkpoint
│   └── epoch_*.pt        # Intermediate checkpoints
├── results.png           # Training curves (loss, mAP)
├── confusion_matrix.png  # Confusion matrix
├── results.csv           # Metrics per epoch
└── events.out.tfevents   # TensorBoard logs
```

---

## 📈 Monitoring Training

### **1. Console Output**
Real-time metrics printed during training:
```
Epoch 1/50: 100%|████████| 4366/4366 [08:34<00:00, 8.49it/s]
      Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
        all       10000      50000      0.645      0.589      0.621      0.442
       bike       10000       1234      0.556      0.489      0.512      0.335
        car       10000      23456      0.789      0.756      0.812      0.598
```

### **2. TensorBoard (Visual)**
```bash
# In another terminal, while training is running:
cd model/outputs/training_logs/
tensorboard --logdir .

# Then open browser: http://localhost:6006
```

You'll see:
- Loss curves (box, cls, dfl)
- mAP over epochs
- Learning rate schedule
- Sample predictions with ground truth

---

## ⚠️ Common Issues & Solutions

### **Issue 1: "CUDA Out of Memory"**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**
```bash
# Option A: Reduce batch size
python train.py --batch 4  # or --batch 2

# Option B: Use smaller image size
python train.py --imgsz 512

# Option C: Use smaller model
python train.py --model yolov8s.pt  # or yolov8n.pt

# Option D: Use CPU (slow)
python train.py --device cpu
```

### **Issue 2: "No module named 'ultralytics'"**
```bash
pip install ultralytics torch torchvision
```

### **Issue 3: Training is too slow**
**On CPU:**
- Expected: ~1-2 minutes per epoch on subset
- On full dataset: Not recommended (days)

**Speed up:**
- Use GPU if available
- Reduce `--workers` if CPU bottleneck
- Use smaller model variant

### **Issue 4: "FileNotFoundError: images not found"**
Check paths in `configs/bdd100k.yaml`:
```yaml
path: /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/data_analysis/data
train: bdd100k_images_100k/bdd100k/images/100k/train
val: bdd100k_images_100k/bdd100k/images/100k/val
```

Verify images exist:
```bash
ls /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/data_analysis/data/bdd100k_images_100k/bdd100k/images/100k/train/ | head
```

---

## 🎯 Step-by-Step First Run

### **Step 1: Verify Setup**
```bash
cd model/

# Check Python version (need 3.8+)
python --version

# Check if CUDA available (optional but recommended)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt

# Verify installation
python -c "from ultralytics import YOLO; print('✓ YOLOv8 ready')"
```

### **Step 3: Test Data Loader**
```bash
cd src/
python data_loader.py
```

Expected output:
```
Testing BDD100k Data Loader
============================
Loading labels from: .../bdd100k_labels_images_train.json
Using subset of 5 images
Dataset initialized:
  Total samples: 5
  Image size: 640x640
  Classes: 10

Sample 1:
  Image shape: torch.Size([3, 640, 640])
  Num objects: 8
  Classes: [2, 2, 4, 6, 7, ...]
  
✓ Data loader test completed successfully!
```

### **Step 4: Run 1-Epoch Demo**
```bash
python train.py \
    --model yolov8m.pt \
    --epochs 1 \
    --batch 8 \
    --subset 100
```

**First run will:**
1. Download `yolov8m.pt` (~52 MB) - takes 30-60 seconds
2. Load 100 training images
3. Train for 1 epoch (~2-5 minutes)
4. Save checkpoint to `outputs/`

### **Step 5: Check Results**
```bash
ls outputs/training_logs/

# View training images with predictions
open outputs/training_logs/yolov8m_bdd100k_1epoch_demo/train_batch*.jpg
```

---

## 📊 Expected Training Metrics

### **After 1 Epoch (Demo on 100 images):**
- box_loss: ~1.2 → ~0.9
- cls_loss: ~1.5 → ~1.1
- mAP@0.5: N/A (not enough training)
- Time: 2-5 minutes

### **After 50 Epochs (Full training on 69,863 images):**
- box_loss: ~0.35
- cls_loss: ~0.45
- mAP@0.5: ~68%
- mAP@0.5:0.95: ~48%
- Time: 6-15 hours (depending on GPU)

### **Per-Class mAP (Expected after full training):**
```
Class          AP@0.5    Count
─────────────────────────────
car            0.78      ~33k
person         0.72      ~15k
truck          0.71      ~5k
bus            0.68      ~2k
bike           0.63      ~3k
rider          0.62      ~2k
traffic sign   0.58      ~12k
motor          0.52      ~800
traffic light  0.48      ~8k
train          0.35      ~150 (very rare!)
```

---

## 🔄 After Training: Next Steps

### **1. Evaluate Model**
```bash
cd src/
python inference.py \
    --model ../outputs/training_logs/.../weights/best.pt \
    --num-samples 100
```

### **2. Export for Deployment**
```python
from ultralytics import YOLO

model = YOLO('outputs/.../weights/best.pt')

# Export to ONNX
model.export(format='onnx')

# Export to TensorRT (requires CUDA)
model.export(format='engine')
```

### **3. Resume Training**
If training was interrupted:
```bash
python train.py --full \
    --model outputs/.../weights/last.pt \
    --resume True
```

---

## 💡 Tips for Best Results

### **1. Start Small, Scale Up**
```bash
# Day 1: Test with demo
python train.py --subset 100 --epochs 1

# Day 2: Small run
python train.py --full --epochs 5 --batch 8

# Day 3+: Full training
python train.py --full --epochs 50 --batch 16
```

### **2. Monitor While Training**
```bash
# Terminal 1: Training
python train.py --full

# Terminal 2: TensorBoard
cd ../outputs/training_logs/
tensorboard --logdir .

# Terminal 3: GPU monitoring (if CUDA)
watch -n 1 nvidia-smi
```

### **3. Save Checkpoints**
Training automatically saves:
- `best.pt` - Best mAP model (for inference)
- `last.pt` - Latest checkpoint (for resuming)
- `epoch_N.pt` - Every N epochs

### **4. Hyperparameter Tuning**
After first training run, adjust:
```bash
# If underfitting (low training accuracy):
--epochs 100 --lr0 0.01

# If overfitting (training >> validation):
--weight_decay 0.001 --dropout 0.2

# If class imbalance issues:
--cls_weight [1.0, 3.0, 1.0, ...] # Weight rare classes higher
```

---

## 📝 Training Checklist

Before you start, verify:

- [ ] Python 3.8+ installed
- [ ] `pip install -r requirements.txt` completed
- [ ] GPU drivers installed (if using CUDA)
- [ ] Dataset paths verified in `configs/bdd100k.yaml`
- [ ] At least 10 GB free disk space
- [ ] Data loader test passed (`python data_loader.py`)

**First training run:**
- [ ] Start with 1-epoch demo (`--subset 100 --epochs 1`)
- [ ] Verify loss decreases
- [ ] Check output images in `outputs/`
- [ ] TensorBoard shows training curves

**Full training:**
- [ ] Run with `--full` flag
- [ ] Monitor GPU usage (`nvidia-smi`)
- [ ] Check TensorBoard periodically
- [ ] Wait 6-15 hours for completion

---

## 🆘 Getting Help

**Check logs:**
```bash
# View last training log
cat outputs/training_logs/*/results.csv

# View errors
cat outputs/training_logs/*/events.out.tfevents
```

**Common questions:**
1. **How long will it take?** See time estimates above
2. **Can I stop and resume?** Yes, use `--resume True` with `last.pt`
3. **Do I need GPU?** No, but highly recommended (10x faster)
4. **What if mAP is low?** Train longer, adjust hyperparameters, check data

---

## 📚 Additional Resources

- **YOLOv8 Docs**: https://docs.ultralytics.com/modes/train/
- **BDD100k Info**: https://bdd-data.berkeley.edu/
- **Training Tips**: See `docs/training_strategy.md`
- **Architecture**: See `docs/architecture_explained.md`

---

**Last Updated**: November 2025  
**Status**: Ready for training ✅  
**Estimated Setup Time**: 10-15 minutes  
**Estimated Training Time**: 2-5 min (demo) or 6-15 hours (full)
