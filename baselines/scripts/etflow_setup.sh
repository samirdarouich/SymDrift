#!/bin/bash
# Set up ETFlow conda environment (Python 3.10, PyTorch 2.1, CUDA 12.1)
# Uses pip for PyTorch to avoid conda MKL linking issues on HPC clusters.
set -e

REPO="$(cd "$(dirname "$0")/../ETFlow" 2>/dev/null && pwd)" || {
    echo "[ETFlow] ERROR: ETFlow repo not found. Run clone_baselines.sh first."
    exit 1
}

echo "[ETFlow] Creating conda environment (Python 3.10)..."
conda env list | grep -q "^etflow " || \
    conda create -n etflow python=3.10 -y --override-channels -c conda-forge

echo "[ETFlow] Installing PyTorch 2.1.0 + CUDA 12.1 via pip..."
conda run -n etflow pip install \
    torch==2.1.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121

echo "[ETFlow] Installing PyG and dependencies..."
conda run -n etflow pip install \
    torch-scatter torch-sparse torch-cluster torch-geometric \
    -f "https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html"

echo "[ETFlow] Installing remaining deps..."
conda run -n etflow conda install -y rdkit datamol numpy=1.26.4 -c conda-forge
conda run -n etflow pip install lightning wandb

echo "[ETFlow] Installing repo..."
conda run -n etflow pip install -e "$REPO"

echo "[ETFlow] Setup complete."
