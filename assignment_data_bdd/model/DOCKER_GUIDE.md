# Docker Setup Guide for Model Training

## 📋 Overview

This guide explains how to use Docker for training the YOLOv8 model on BDD100k dataset. We provide **two separate Docker setups**:

1. **`Dockerfile`** - GPU-enabled (NVIDIA CUDA) for Linux with GPU
2. **`Dockerfile.cpu`** - CPU-only for macOS (or any system without GPU)

---

## 🎯 Why Separate Containers?

### **Data Analysis Container** (`data_analysis/Dockerfile`)
- **Purpose**: Visualization and EDA
- **Size**: ~500 MB
- **Dependencies**: pandas, matplotlib, dash, plotly
- **Port**: 8050 (Dash app)

### **Model Training Container** (`model/Dockerfile.cpu`)
- **Purpose**: Deep learning training
- **Size**: ~2-3 GB
- **Dependencies**: PyTorch, ultralytics, opencv
- **Port**: 6006 (TensorBoard)

**Result**: Clean separation, no conflicts, faster builds! ✅

---

## 🚀 Quick Start (macOS - CPU Training)

### **Step 1: Build the Container**

```bash
cd /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model

# Build the CPU-only training image
docker build -f Dockerfile.cpu -t bdd100k-training-cpu .
```

**Build time**: ~5-10 minutes (first time)  
**Image size**: ~2.5 GB

### **Step 2: Test Data Loader**

```bash
# Test that data loading works
docker run --rm \
  -v /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/data_analysis/data:/data \
  bdd100k-training-cpu \
  python3 src/data_loader.py
```

**Expected output**:
```
Testing BDD100k Data Loader
============================
Loading labels from: /data/...
Dataset initialized:
  Total samples: 5
  Image size: 640x640
  Classes: 10
✓ Data loader test completed successfully!
```

### **Step 3: Run 1-Epoch Training Demo**

```bash
docker run --rm \
  -v /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/data_analysis/data:/data \
  -v /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model/outputs:/workspace/outputs \
  bdd100k-training-cpu \
  python3 src/train.py \
    --model yolov8m.pt \
    --epochs 1 \
    --batch 4 \
    --subset 100 \
    --device cpu
```

**What happens:**
1. Downloads `yolov8m.pt` (52 MB) inside container
2. Loads 100 training images from mounted `/data` volume
3. Trains for 1 epoch (~20-30 minutes on CPU)
4. Saves results to `outputs/` (mounted, persists after container stops)

---

## 📊 Volume Mounts Explained

```bash
-v /path/on/host:/path/in/container
```

### **Required Mounts:**

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `.../data_analysis/data` | `/data` | Read dataset (images + labels) |
| `.../model/outputs` | `/workspace/outputs` | Save training results |

### **Optional Mounts:**

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `.../model/weights` | `/workspace/weights` | Pre-downloaded model weights |
| `.../model/configs` | `/workspace/configs` | Custom dataset configs |

---

## 🎛️ Docker Commands Reference

### **Build Commands**

```bash
# CPU-only (for macOS)
docker build -f Dockerfile.cpu -t bdd100k-training-cpu .

# GPU-enabled (for Linux with NVIDIA GPU)
docker build -f Dockerfile -t bdd100k-training-gpu .
```

### **Training Commands**

```bash
# 1-epoch demo (CPU)
docker run --rm \
  -v $(pwd)/../data_analysis/data:/data \
  -v $(pwd)/outputs:/workspace/outputs \
  bdd100k-training-cpu \
  python3 src/train.py --model yolov8m.pt --epochs 1 --batch 4 --subset 100 --device cpu

# Full training (CPU, 50 epochs) - WARNING: Takes ~100+ hours
docker run --rm \
  -v $(pwd)/../data_analysis/data:/data \
  -v $(pwd)/outputs:/workspace/outputs \
  bdd100k-training-cpu \
  python3 src/train.py --full --model yolov8m.pt --epochs 50 --batch 4 --device cpu

# With GPU (Linux only, requires nvidia-docker)
docker run --rm --gpus all \
  -v $(pwd)/../data_analysis/data:/data \
  -v $(pwd)/outputs:/workspace/outputs \
  bdd100k-training-gpu \
  python3 src/train.py --model yolov8m.pt --epochs 50 --batch 16 --device cuda
```

### **Inference Commands**

```bash
# Run inference on validation set
docker run --rm \
  -v $(pwd)/../data_analysis/data:/data \
  -v $(pwd)/outputs:/workspace/outputs \
  bdd100k-training-cpu \
  python3 src/inference.py \
    --model yolov8m.pt \
    --images-dir /data/bdd100k_images_100k/bdd100k/images/100k/val \
    --labels /data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
    --num-samples 50 \
    --output-dir /workspace/outputs/inference_samples
```

### **TensorBoard (Monitor Training)**

```bash
# Start TensorBoard in container
docker run --rm -p 6006:6006 \
  -v $(pwd)/outputs:/workspace/outputs \
  bdd100k-training-cpu \
  tensorboard --logdir /workspace/outputs/training_logs --host 0.0.0.0

# Then open browser: http://localhost:6006
```

---

## 🔧 Interactive Development

### **Start Container with Shell**

```bash
# Launch interactive shell
docker run -it --rm \
  -v $(pwd)/../data_analysis/data:/data \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/src:/workspace/src \
  bdd100k-training-cpu \
  bash

# Now inside container:
python3 src/data_loader.py
python3 src/train.py --help
python3 src/inference.py --model yolov8m.pt --num-samples 5
```

### **Mount Source Code for Live Editing**

```bash
docker run --rm \
  -v $(pwd)/../data_analysis/data:/data \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/src:/workspace/src \
  bdd100k-training-cpu \
  python3 src/train.py --epochs 1 --subset 50
```

Now you can edit `src/*.py` on your Mac and changes reflect immediately in container!

---

## 🐳 Docker Compose (Optional)

Create `docker-compose.yml` in `model/` directory:

```yaml
version: '3.8'

services:
  training:
    build:
      context: .
      dockerfile: Dockerfile.cpu
    image: bdd100k-training-cpu
    volumes:
      - ../data_analysis/data:/data:ro  # Read-only
      - ./outputs:/workspace/outputs
      - ./src:/workspace/src
    command: python3 src/train.py --help
    
  tensorboard:
    image: bdd100k-training-cpu
    ports:
      - "6006:6006"
    volumes:
      - ./outputs:/workspace/outputs
    command: tensorboard --logdir /workspace/outputs/training_logs --host 0.0.0.0
```

**Usage:**

```bash
# Build
docker-compose build

# Run training
docker-compose run training python3 src/train.py --epochs 1 --subset 100

# Start TensorBoard
docker-compose up tensorboard
```

---

## 📦 Comparing Container Sizes

```bash
# Check image sizes
docker images | grep bdd100k

# Expected sizes:
# bdd100k-training-cpu    ~2.5 GB
# bdd100k-training-gpu    ~6.5 GB
# data-analysis           ~0.5 GB
```

---

## ⚠️ Common Issues

### **Issue 1: Volume Mount Not Working**

```bash
# Error: FileNotFoundError: images not found

# Fix: Use absolute paths
docker run -v /Users/ayushsoral/.../data:/data ...  # ✅
docker run -v ~/Desktop/.../data:/data ...           # ❌ (~ doesn't expand)
docker run -v ./data:/data ...                       # ❌ (relative path)
```

### **Issue 2: Container Runs Out of Memory**

```bash
# Limit memory usage
docker run --memory="8g" --memory-swap="16g" \
  -v $(pwd)/../data_analysis/data:/data \
  bdd100k-training-cpu \
  python3 src/train.py --batch 2 --subset 50
```

### **Issue 3: Slow CPU Training**

```bash
# Options to speed up:
# 1. Use smaller model
docker run ... python3 src/train.py --model yolov8n.pt  # Nano variant

# 2. Reduce batch size (less computation per step)
docker run ... python3 src/train.py --batch 2

# 3. Use smaller image size
docker run ... python3 src/train.py --imgsz 512

# 4. Use smaller subset
docker run ... python3 src/train.py --subset 50
```

### **Issue 4: Port Already in Use (TensorBoard)**

```bash
# If port 6006 is busy, use different port
docker run -p 6007:6006 ...  # Access at localhost:6007
```

---

## 🔄 Workflow: Data Analysis → Training

### **Complete Workflow:**

```bash
# Step 1: Data Analysis (existing container)
cd /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/data_analysis
docker-compose up

# Browse dashboard at http://localhost:8050
# Explore data, understand distribution

# Step 2: Model Training (new container)
cd /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model

# Build training container
docker build -f Dockerfile.cpu -t bdd100k-training-cpu .

# Run training
docker run --rm \
  -v /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/data_analysis/data:/data \
  -v /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model/outputs:/workspace/outputs \
  bdd100k-training-cpu \
  python3 src/train.py --model yolov8m.pt --epochs 1 --batch 4 --subset 100 --device cpu

# Step 3: Monitor Training
# In another terminal:
docker run --rm -p 6006:6006 \
  -v /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model/outputs:/workspace/outputs \
  bdd100k-training-cpu \
  tensorboard --logdir /workspace/outputs/training_logs --host 0.0.0.0

# Browse TensorBoard at http://localhost:6006
```

**Both containers run independently!** ✅
- Data analysis on port 8050
- TensorBoard on port 6006
- No conflicts!

---

## 📊 Container Resource Usage

### **CPU Training (macOS)**
- **Memory**: 4-8 GB RAM
- **CPU**: Will use all available cores
- **Disk**: ~10 GB for outputs
- **Time**: 20-30 min per epoch (subset), 5-10 hours per epoch (full)

### **GPU Training (Linux)**
- **Memory**: 4-8 GB RAM
- **GPU Memory**: 6-16 GB VRAM
- **CPU**: 8+ cores recommended
- **Time**: 2-5 min per epoch (subset), 10-15 min per epoch (full)

---

## 🎯 Best Practices

### **1. Use `.dockerignore`**

Create `model/.dockerignore`:
```
outputs/
weights/*.pt
__pycache__/
*.pyc
.DS_Store
.git/
```

Reduces build context size from GB to MB!

### **2. Layer Caching**

```dockerfile
# Good: Copy requirements first
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/

# Bad: Copy everything, then install
COPY . .
RUN pip install -r requirements.txt
```

### **3. Clean Up Old Containers**

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove all unused data
docker system prune -a
```

---

## 📚 Additional Resources

- **Docker Docs**: https://docs.docker.com/
- **NVIDIA Docker**: https://github.com/NVIDIA/nvidia-docker
- **Docker Compose**: https://docs.docker.com/compose/

---

## ✅ Summary

**Your Setup:**
```
assignment_data_bdd/
├── data_analysis/
│   ├── Dockerfile          ← Visualization container (port 8050)
│   └── docker-compose.yml  ← Easy startup
│
└── model/
    ├── Dockerfile          ← GPU training (Linux)
    ├── Dockerfile.cpu      ← CPU training (macOS) ✅ Use this!
    └── DOCKER_GUIDE.md     ← This file
```

**Recommendation for macOS:**
1. ✅ Use `Dockerfile.cpu` for model training
2. ✅ Keep existing `data_analysis/Dockerfile` unchanged
3. ✅ Run both containers independently when needed
4. ✅ Share data via volume mounts

**No conflicts, clean separation!** 🎉

---

**Last Updated**: November 2025  
**Status**: Ready to use  
**Tested on**: macOS (CPU), Linux (GPU)
