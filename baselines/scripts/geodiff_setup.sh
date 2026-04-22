#!/bin/bash
# Set up GeoDiff conda environment (Python 3.9, PyTorch 1.11, CUDA 11.3)
# Uses pip for PyTorch to avoid conda MKL linking issues on HPC clusters.
set -e

REPO="$(cd "$(dirname "$0")/../GeoDiff" 2>/dev/null && pwd)" || {
    echo "[GeoDiff] ERROR: GeoDiff repo not found. Run clone_baselines.sh first."
    exit 1
}

echo "[GeoDiff] Creating conda environment (Python 3.9)..."
conda env list | grep -q "^geodiff " || \
    conda create -n geodiff python=3.9 -y --override-channels -c conda-forge

echo "[GeoDiff] Installing PyTorch 1.11.0 + CUDA 11.3 via pip..."
conda run -n geodiff pip install \
    torch==1.11.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

echo "[GeoDiff] Installing RDKit and core deps..."
conda run -n geodiff conda install -y rdkit pyaml scipy tqdm -c conda-forge

echo "[GeoDiff] Installing numpy..."
conda run -n geodiff pip install numpy

echo "[GeoDiff] Installing PyTorch Geometric (pinned for PyTorch 1.11 compatibility)..."
conda run -n geodiff pip install \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html"
conda run -n geodiff pip install torch-geometric==2.0.4

echo "[GeoDiff] Installing remaining repo deps..."
conda run -n geodiff pip install easydict networkx

echo "[GeoDiff] Setup complete."
