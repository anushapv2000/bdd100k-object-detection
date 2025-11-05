# YOLOv8 Training Container for BDD100k Object Detection
# This Dockerfile creates a containerized training environment

# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy model code
COPY src/ ./src/
COPY configs/ ./configs/
COPY docs/ ./docs/

# Create directories for outputs
RUN mkdir -p /workspace/outputs/training_logs \
    /workspace/outputs/inference_samples \
    /workspace/weights

# Expose TensorBoard port
EXPOSE 6006

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command: show help
CMD ["python3", "src/train.py", "--help"]

# Usage:
# Build:   docker build -t bdd100k-training .
# Run:     docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/outputs:/workspace/outputs bdd100k-training
# Train:   docker run --gpus all -v $(pwd)/data:/data bdd100k-training python3 src/train.py --model yolov8m.pt --epochs 1
