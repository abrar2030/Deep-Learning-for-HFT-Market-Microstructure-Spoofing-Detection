# Multi-stage Dockerfile for TEN-GNN Spoofing Detection
# Supports both CPU and GPU inference
# Build context must be the project root (docker build .)

# ============================================
# Shared base (CPU)
# ============================================
FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps before copying full source for better layer caching
COPY code/requirements.txt code/requirements-prod.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -r requirements-prod.txt

# Copy entire project
COPY . .

# Runtime directories
RUN mkdir -p /app/logs /app/checkpoints /app/data /app/alerts

# ============================================
# CPU stage
# ============================================
FROM base AS cpu

ENV DEVICE=cpu
EXPOSE 8000
CMD ["python", "infrastructure/api/server.py", "--device", "cpu"]

# ============================================
# GPU base (CUDA 11.8 + cuDNN 8 on Ubuntu 22.04)
# ============================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS gpu-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    DEVICE=cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

COPY code/requirements.txt code/requirements-prod.txt ./

# Install CUDA-enabled PyTorch first, then remaining requirements
# (grep -v prevents the CPU torch wheel from overwriting the CUDA build)
RUN pip install --upgrade pip \
    && pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu118 \
    && grep -vE "^torch(vision)?([>=<!]|$)" requirements.txt \
        | pip install -r /dev/stdin \
    && pip install -r requirements-prod.txt

COPY . .

RUN mkdir -p /app/logs /app/checkpoints /app/data /app/alerts

EXPOSE 8000
CMD ["python", "infrastructure/api/server.py", "--device", "cuda"]
