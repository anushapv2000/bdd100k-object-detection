# Optimized YOLOv8 Training Container for Apple Silicon (M1/M2/M3/M4)
# This Dockerfile is specifically designed for ARM64 (Apple M-series chips)

# Use ARM64-compatible Python base image
FROM --platform=linux/arm64 python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgfortran5 \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create working directory
WORKDIR /workspace

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install PyTorch for Apple Silicon (ARM64)
# Note: PyTorch has ARM64 wheels that work well on Apple Silicon
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY docs/ ./docs/
COPY README.md TRAINING_SETUP_GUIDE.md ./

# Create output directories
RUN mkdir -p /workspace/outputs/training_logs \
             /workspace/outputs/inference_samples \
             /workspace/weights \
             /workspace/logs

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Expose TensorBoard port
EXPOSE 6006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; import ultralytics; print('OK')" || exit 1

# Default command
CMD ["python3", "src/train.py", "--help"]

# =============================================================================
# BUILD & RUN INSTRUCTIONS FOR APPLE SILICON (M4)
# =============================================================================
#
# BUILD (optimized for M4):
#   docker build --platform linux/arm64 -f Dockerfile.m4 -t bdd100k-training-m4 .
#
# QUICK TEST:
#   docker run --rm --platform linux/arm64 bdd100k-training-m4 python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
#
# TEST DATA LOADER:
#   docker run --rm --platform linux/arm64 \
#     -v /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/data_analysis/data:/data \
#     bdd100k-training-m4 \
#     python3 src/data_loader.py
#
# TRAIN (1-epoch demo):
#   docker run --rm --platform linux/arm64 \
#     -v /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/data_analysis/data:/data \
#     -v /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model/outputs:/workspace/outputs \
#     bdd100k-training-m4 \
#     python3 src/train.py --model yolov8m.pt --epochs 1 --batch 8 --subset 100 --device cpu
#
# INTERACTIVE MODE:
#   docker run -it --rm --platform linux/arm64 \
#     -v /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/data_analysis/data:/data \
#     -v /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model/outputs:/workspace/outputs \
#     bdd100k-training-m4 bash
#
# TENSORBOARD:
#   docker run --rm --platform linux/arm64 -p 6006:6006 \
#     -v /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model/outputs:/workspace/outputs \
#     bdd100k-training-m4 \
#     tensorboard --logdir /workspace/outputs/training_logs --host 0.0.0.0
#
# =============================================================================
# PERFORMANCE NOTES FOR APPLE M4
# =============================================================================
#
# Your M4 chip has a 10-core GPU, but Docker on macOS runs in a Linux VM
# and CANNOT directly access Apple's Metal GPU acceleration.
#
# Performance expectations:
# - CPU Training (in Docker): ~20-30 min per epoch (100 images)
# - Native macOS (with MPS): ~5-10 min per epoch (100 images) - 2-3x faster!
#
# RECOMMENDATION: For best performance on M4, run NATIVELY (without Docker)
# using PyTorch with MPS backend. See instructions below.
#
# =============================================================================
