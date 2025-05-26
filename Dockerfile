#!/usr/bin/env python3
"""
Docker configuration for the Local-Global Attention Network.

This Dockerfile creates a complete environment for training and inference
with all necessary dependencies.
"""

# Use NVIDIA PyTorch base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY examples/ ./examples/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY README.md .

# Create necessary directories
RUN mkdir -p /workspace/data \
    /workspace/outputs \
    /workspace/checkpoints \
    /workspace/logs

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for TensorBoard
EXPOSE 6006

# Default command
CMD ["python", "examples/train_model.py", "--help"]
