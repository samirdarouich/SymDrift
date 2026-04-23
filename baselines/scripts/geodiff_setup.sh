#!/bin/bash
# Set up GeoDiff conda environment (Python 3.9, PyTorch 2.1, CUDA 12.1)
# CUDA 12.1 wheels run fine on CUDA 12.x systems (backward compatible).
set -e

REPO="$(cd "$(dirname "$0")/../GeoDiff" 2>/dev/null && pwd)" || {
    echo "[GeoDiff] ERROR: GeoDiff repo not found. Run clone_baselines.sh first."
    exit 1
}

echo "[GeoDiff] Creating conda environment (Python 3.9)..."
conda env list | grep -q "^geodiff " || \
    conda create -n geodiff python=3.9 -y --override-channels -c conda-forge

PYTHON=$(conda info --envs | awk '$1=="geodiff"{print $NF"/bin/python3"}')

echo "[GeoDiff] Installing PyTorch 2.1.0 + CUDA 12.1 via pip..."
$PYTHON -m pip install torch==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu121

echo "[GeoDiff] Installing numpy<2 and core deps..."
$PYTHON -m pip install "numpy<2"
$PYTHON -m pip install rdkit pyaml scipy tqdm easydict networkx

echo "[GeoDiff] Installing PyTorch Geometric (prebuilt wheels for torch 2.1 + cu121)..."
$PYTHON -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    --find-links "https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html"
$PYTHON -m pip install torch-geometric

echo "[GeoDiff] Setup complete."
