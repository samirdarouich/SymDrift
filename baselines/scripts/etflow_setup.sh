#!/bin/bash
# Set up ETFlow conda environment (Python >=3.8, PyTorch + CUDA 12.1).
set -e

REPO="$(cd "$(dirname "$0")/../ETFlow" 2>/dev/null && pwd)" || {
    echo "[ETFlow] ERROR: ETFlow repo not found. Run clone_baselines.sh first."
    exit 1
}

echo "[ETFlow] Creating conda environment from $REPO/env.yml ..."
conda env create -n etflow -f "$REPO/env.yml" || {
    echo "[ETFlow] Environment may already exist; attempting update..."
    conda env update -n etflow -f "$REPO/env.yml"
}

echo "[ETFlow] Installing repo in editable mode..."
conda run -n etflow pip install -e "$REPO"

echo "[ETFlow] Setup complete."
