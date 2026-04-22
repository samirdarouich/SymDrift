#!/bin/bash
# Set up GeoDiff conda environment (Python 3.7, PyTorch 1.8.1, CUDA 10.2)
set -e

REPO="$(cd "$(dirname "$0")/../GeoDiff" 2>/dev/null && pwd)" || {
    echo "[GeoDiff] ERROR: GeoDiff repo not found. Run clone_baselines.sh first."
    exit 1
}

echo "[GeoDiff] Creating conda environment from $REPO/env.yml ..."
conda env create -f "$REPO/env.yml" --name geodiff || {
    echo "[GeoDiff] Environment may already exist; attempting update..."
    conda env update -f "$REPO/env.yml" --name geodiff
}

echo "[GeoDiff] Switching to pretrain branch (required for pretrained checkpoints)..."
cd "$REPO"
git checkout pretrain 2>/dev/null || echo "[GeoDiff] Already on pretrain branch or detached HEAD."

echo "[GeoDiff] Installing pinned PyTorch Geometric..."
conda run -n geodiff conda install -y \
    pytorch-geometric=1.7.2=py37_torch_1.8.0_cu102 \
    -c rusty1s -c conda-forge

echo "[GeoDiff] Setup complete."
