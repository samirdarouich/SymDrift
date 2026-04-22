#!/bin/bash
# Set up GeoMol conda environment.
# Note: the Makefile's create_env.sh is interactive (asks CUDA version).
# This script automates it for CUDA 11.3 — adjust CUDA_VERSION below if needed.
set -e

REPO="$(cd "$(dirname "$0")/../GeoMol" 2>/dev/null && pwd)" || {
    echo "[GeoMol] ERROR: GeoMol repo not found. Run clone_baselines.sh first."
    exit 1
}

# Adjust to match your CUDA version: cu92 / cu101 / cu102 / cu111 / cpu
CUDA_VERSION="cu113"
TORCH_VERSION="1.11.0"

echo "[GeoMol] Creating conda environment from $REPO/devtools/environment.yml ..."
conda env create -f "$REPO/devtools/environment.yml" --name geomol || {
    echo "[GeoMol] Environment may already exist; attempting update..."
    conda env update -f "$REPO/devtools/environment.yml" --name geomol
}

echo "[GeoMol] Installing PyTorch $TORCH_VERSION + $CUDA_VERSION ..."
conda run -n geomol conda install -y pytorch=$TORCH_VERSION torchvision \
    cudatoolkit=11.3 -c pytorch

echo "[GeoMol] Installing PyTorch Geometric components..."
conda run -n geomol pip install \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html"
conda run -n geomol pip install torch-geometric

echo "[GeoMol] Setup complete."
