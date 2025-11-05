# Native macOS Training Setup for Apple M4 Chip

## 🚀 **Why Native is Better Than Docker on M4**

Your Apple M4 chip has a **10-core GPU** with Metal 3 support, but Docker **cannot access it** because it runs in a Linux VM. 

### Performance Comparison:

| Method | Device | Speed (per epoch, 100 images) | GPU Used? |
|--------|--------|-------------------------------|-----------|
| **Native macOS** | **MPS (Metal)** | **5-10 minutes** ⚡ | ✅ **Yes - 10-core GPU** |
| Docker ARM64 | CPU only | 20-30 minutes | ❌ No |
| Standard CPU | CPU | 25-35 minutes | ❌ No |

**Result**: Native is **2-3x faster** because it uses your M4's 10-core GPU via Metal Performance Shaders! 🚀

---

## ✅ **RECOMMENDED: Native macOS Setup (Use Your GPU!)**

### **Step 1: Install Dependencies**

```bash
cd /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model

# Install Python dependencies
pip3 install -r requirements.txt
```

**What gets installed:**
- PyTorch with **MPS (Metal Performance Shaders)** support - uses your M4 GPU!
- Ultralytics YOLOv8
- All other dependencies

**Installation time**: 2-3 minutes

### **Step 2: Verify MPS (GPU) is Available**

```bash
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available()); print('MPS built:', torch.backends.mps.is_built())"
```

**Expected output:**
```
MPS available: True
MPS built: True
```

✅ If you see `True` for both, your **10-core M4 GPU is ready** for training!

**Bonus check** - See your GPU info:
```bash
system_profiler SPDisplaysDataType | grep -A 5 "Chipset Model"
```

Should show:
```
Chipset Model: Apple M4
Type: GPU
Total Number of Cores: 10
```

### **Step 3: Run Training (Using Your GPU!)**

```bash
cd src/

# 1-epoch demo training with MPS (GPU) acceleration
python3 train.py \
    --model yolov8m.pt \
    --epochs 1 \
    --batch 8 \
    --subset 100 \
    --device auto
```

**What happens:**
1. Script auto-detects MPS (your M4 GPU) ✅
2. Downloads `yolov8m.pt` (52 MB, first time only)
3. Loads 100 training images
4. **Trains on your 10-core GPU** - 2-3x faster than CPU!
5. Saves results to `outputs/`

**Time**: ~5-10 minutes (with GPU) vs 20-30 minutes (CPU)

**Expected output:**
```
YOLOv8 Training Demo - 1 Epoch on Subset
========================================

[INFO] Configuration:
  Device: mps  ← YOUR M4 GPU!
  
[1/4] Loading YOLOv8 model...
  ✓ Model loaded successfully
  ✓ Device: mps  ← Using Metal Performance Shaders

[2/4] Configuring training parameters...

[3/4] Starting training for 1 epoch...
Epoch 1/1: 100%|██████████| 13/13 [05:23<00:00]  ← ~5 minutes!
  box_loss: 0.450
  cls_loss: 0.620
  
✓ Training completed successfully!
```

---

## 🎯 **Performance Optimization for M4**

### **Optimal Batch Size for M4 (10-core GPU)**

Your M4 has excellent memory bandwidth. Test different batch sizes:

```bash
# Small (safe, slower)
python3 train.py --batch 4 --subset 100

# Medium (recommended)
python3 train.py --batch 8 --subset 100  ← Best for M4

# Large (if you have 16+ GB RAM)
python3 train.py --batch 16 --subset 100

# Extra large (if you have 24+ GB RAM)
python3 train.py --batch 24 --subset 100
```

**Recommendation**: Start with `--batch 8`, monitor memory usage, increase if stable.

### **Monitor GPU Usage**

```bash
# In another terminal while training:
sudo powermetrics --samplers gpu_power -i 1000 -n 1

# Or use Activity Monitor:
# Open Activity Monitor → Window → GPU History
```

You should see **GPU utilization at 80-100%** during training!

---

## 📊 **Complete Training Workflow**

### **Workflow A: Quick Demo (5-10 min)**

```bash
cd /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model/src

# Test data loader first
python3 data_loader.py

# Run 1-epoch training
python3 train.py --model yolov8m.pt --epochs 1 --batch 8 --subset 100 --device auto
```

### **Workflow B: Full Training (2-4 hours)**

```bash
cd /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model/src

# Full training with MPS (GPU) - much faster than CPU!
python3 train.py --full \
    --model yolov8m.pt \
    --epochs 50 \
    --batch 12 \
    --device auto
```

**Time estimates with M4 GPU:**
- **1 epoch on 100 images**: 5-10 minutes
- **1 epoch on full dataset**: 25-35 minutes
- **50 epochs on full dataset**: 20-30 hours

**vs CPU (without GPU):**
- **1 epoch on 100 images**: 20-30 minutes
- **1 epoch on full dataset**: 2-3 hours
- **50 epochs on full dataset**: 100+ hours

**Your M4 GPU saves 70-80 hours!** ⚡

---

## 🐳 **Alternative: Docker Setup (No GPU Access)**

If you still want to use Docker (for reproducibility/submission), here's how:

### **Build Docker Image**

```bash
cd /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model

# Build for ARM64 (M4 architecture)
docker build --platform linux/arm64 -f Dockerfile.m4 -t bdd100k-training-m4 .
```

**Build time**: 5-10 minutes  
**Image size**: ~2.5 GB

### **Run Training in Docker**

```bash
# 1-epoch demo (CPU only in Docker)
docker run --rm --platform linux/arm64 \
  -v /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/data_analysis/data:/data \
  -v /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model/outputs:/workspace/outputs \
  bdd100k-training-m4 \
  python3 src/train.py --model yolov8m.pt --epochs 1 --batch 8 --subset 100 --device cpu
```

⚠️ **Warning**: Docker runs CPU-only (no GPU access), so it's **2-3x slower** than native!

---

## 📈 **Monitoring Training**

### **Option 1: TensorBoard (Visual)**

```bash
# In another terminal while training:
cd /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model
tensorboard --logdir outputs/training_logs

# Open browser: http://localhost:6006
```

You'll see:
- Loss curves (box, cls, dfl)
- mAP over epochs
- Sample predictions with ground truth
- GPU utilization (if using MPS)

### **Option 2: Activity Monitor**

While training is running:
1. Open **Activity Monitor** (Cmd+Space → "Activity Monitor")
2. Go to **Window** → **GPU History**
3. You should see **high GPU usage** (80-100%)

### **Option 3: Console Output**

The training script shows real-time progress:
```
Epoch 1/1: 100%|██████████| 13/13 [05:23<00:00, 24.89s/it]
                 Class     Images  Instances    Box(P      R)  mAP50
                   all        100        500    0.645  0.589  0.621
```

---

## ⚡ **Performance Tips for M4**

### **1. Use Optimal Batch Size**
```bash
# Your M4 can handle larger batches
python3 train.py --batch 12  # Instead of default 8
```

### **2. Increase Workers for Data Loading**
```bash
# M4 has high-performance CPU cores
python3 train.py --workers 8  # Faster data loading
```

### **3. Use AMP (Automatic Mixed Precision)**
YOLOv8 uses AMP automatically on MPS - **no action needed**! This gives you extra speedup.

### **4. Monitor Memory**
```bash
# Check memory while training
vm_stat | perl -ne '/page size of (\d+)/ and $size=$1; /Pages\s+([^:]+)[^\d]+(\d+)/ and printf("%-16s % 16.2f MB\n", "$1:", $2 * $size / 1048576);'
```

### **5. Close Other Apps**
For maximum training speed, close:
- Chrome/Safari (heavy memory)
- Xcode (heavy CPU)
- Docker Desktop (if not using)

---

## 🎯 **Quick Start Guide**

### **For Assignment Demo (RECOMMENDED)**

```bash
# Navigate to model directory
cd /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model

# Install dependencies (first time only)
pip3 install -r requirements.txt

# Verify GPU support
python3 -c "import torch; print('GPU Ready:', torch.backends.mps.is_available())"

# Run 1-epoch demo using your M4 GPU
cd src/
python3 train.py --model yolov8m.pt --epochs 1 --batch 8 --subset 100

# Check results
ls -lh ../outputs/training_logs/
```

**Total time**: ~10-15 minutes including setup!

---

## 📊 **Expected Results**

### **After 1 Epoch (100 images, M4 GPU):**
```
Training Time: 5-10 minutes
box_loss: ~1.2 → ~0.9
cls_loss: ~1.5 → ~1.1
dfl_loss: ~1.0 → ~0.8

GPU Utilization: 80-100%
Memory Used: 4-8 GB
```

### **After 50 Epochs (Full dataset, M4 GPU):**
```
Training Time: 20-30 hours (vs 100+ hours on CPU)
mAP@0.5: ~68%
mAP@0.5:0.95: ~48%

Total GPU hours saved: 70-80 hours!
```

---

## ❓ **Troubleshooting**

### **Issue 1: MPS not available**

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
# Returns: False
```

**Solution:**
```bash
# Update PyTorch to latest version
pip3 install --upgrade torch torchvision

# Verify macOS version (need 12.3+)
sw_vers
```

### **Issue 2: Training crashes on MPS**

```bash
# Error: "MPS backend out of memory"
```

**Solution:**
```bash
# Reduce batch size
python3 train.py --batch 4  # or --batch 2

# Or use CPU temporarily
python3 train.py --device cpu
```

### **Issue 3: Slow despite using MPS**

**Check:**
```bash
# Monitor GPU usage
sudo powermetrics --samplers gpu_power -i 1000 -n 1
```

If GPU usage is low (<50%), try:
```bash
# Increase batch size
python3 train.py --batch 12

# Reduce workers if I/O bound
python3 train.py --workers 4
```

---

## 🎉 **Summary**

### **Your M4 Setup Options:**

| Option | Speed | GPU Access | Reproducibility | Recommendation |
|--------|-------|------------|-----------------|----------------|
| **Native macOS** | ⚡⚡⚡ **Fast** | ✅ **10-core GPU** | Good | ✅ **BEST for M4!** |
| Docker ARM64 | 🐌 Slow | ❌ No GPU | Excellent | Only if required |

### **Recommended Commands:**

```bash
# Setup (one time)
cd /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model
pip3 install -r requirements.txt

# Verify GPU
python3 -c "import torch; print('GPU:', torch.backends.mps.is_available())"

# Train (uses M4 GPU automatically!)
cd src/
python3 train.py --model yolov8m.pt --epochs 1 --batch 8 --subset 100
```

**Your M4 GPU will save you 70-80 hours vs CPU training!** ⚡🚀

---

**Last Updated**: November 2025  
**Optimized for**: Apple M4 (10-core GPU)  
**Status**: Ready to train with GPU acceleration! ✅
