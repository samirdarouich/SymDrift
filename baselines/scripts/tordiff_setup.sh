#!/bin/bash
# Set up Torsional Diffusion conda environment (Python 3.9).
set -e

REPO="$(cd "$(dirname "$0")/../torsional-diffusion" 2>/dev/null && pwd)" || {
    echo "[TorsionalDiff] ERROR: torsional-diffusion repo not found. Run clone_baselines.sh first."
    exit 1
}

echo "[TorsionalDiff] Creating conda environment from $REPO/environment.yml ..."
conda env create -f "$REPO/environment.yml" --name torsional_diffusion || {
    echo "[TorsionalDiff] Environment may already exist; attempting update..."
    conda env update -f "$REPO/environment.yml" --name torsional_diffusion
}

echo "[TorsionalDiff] Installing PyTorch + PyTorch Geometric (CUDA 11.3)..."
# Adjust cu113 -> cu102/cpu/etc. to match your system
conda run -n torsional_diffusion conda install -y \
    pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch

conda run -n torsional_diffusion pip install \
    torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f "https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html"

echo "[TorsionalDiff] Installing e3nn..."
conda run -n torsional_diffusion pip install e3nn

echo "[TorsionalDiff] Setup complete."
