#!/bin/bash
# Set up ETFlow conda environment (Python 3.10, PyTorch 2.x, CUDA 12.1).
set -e

REPO="$(cd "$(dirname "$0")/../ETFlow" 2>/dev/null && pwd)" || {
    echo "[ETFlow] ERROR: ETFlow repo not found. Run clone_baselines.sh first."
    exit 1
}

echo "[ETFlow] Creating conda environment (Python 3.10)..."
conda create -n etflow python=3.10 -y \
    --override-channels -c conda-forge -c defaults

echo "[ETFlow] Installing PyTorch + CUDA 12.1..."
conda run -n etflow conda install -y \
    pytorch pytorch-cuda=12.1 \
    -c pytorch -c nvidia -c conda-forge

echo "[ETFlow] Installing PyG and dependencies..."
conda run -n etflow conda install -y \
    pyg pytorch-cluster pytorch-scatter pytorch-sparse \
    rdkit datamol numpy=1.26.4 \
    -c pyg -c conda-forge

echo "[ETFlow] Installing Lightning and other pip deps..."
conda run -n etflow pip install lightning wandb

echo "[ETFlow] Installing repo..."
conda run -n etflow pip install -e "$REPO"

echo "[ETFlow] Setup complete."
