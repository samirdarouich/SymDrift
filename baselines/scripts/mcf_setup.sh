#!/bin/bash
# Set up MCF conda environment (Python 3.10, PyTorch 2.1).
set -e

REPO="$(cd "$(dirname "$0")/../ml-mcf" 2>/dev/null && pwd)" || {
    echo "[MCF] ERROR: ml-mcf repo not found. Run clone_baselines.sh first."
    exit 1
}

echo "[MCF] Creating conda environment (Python 3.10)..."
conda create -n mcf python=3.10 -y \
    --override-channels -c conda-forge -c defaults

echo "[MCF] Installing PyTorch 2.1.0 + CUDA 12.1..."
conda run -n mcf conda install -y \
    pytorch=2.1.0 pytorch-cuda=12.1 \
    -c pytorch -c nvidia -c conda-forge

echo "[MCF] Installing requirements from $REPO/environment/requirements.txt ..."
conda run -n mcf pip install -r "$REPO/environment/requirements.txt"

echo "[MCF] Installing repo..."
conda run -n mcf pip install -e "$REPO" --no-deps 2>/dev/null || true

echo "[MCF] Setup complete."
