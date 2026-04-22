#!/bin/bash
# Set up GeoMol conda environment (Python 3.9, PyTorch 1.11, CUDA 11.3)
# Uses pip for PyTorch to avoid conda MKL linking issues on HPC clusters.
set -e

REPO="$(cd "$(dirname "$0")/../GeoMol" 2>/dev/null && pwd)" || {
    echo "[GeoMol] ERROR: GeoMol repo not found. Run clone_baselines.sh first."
    exit 1
}

echo "[GeoMol] Creating conda environment (Python 3.9)..."
conda create -n geomol python=3.9 -y --override-channels -c conda-forge

echo "[GeoMol] Installing PyTorch 1.11.0 + CUDA 11.3 via pip..."
conda run -n geomol pip install \
    torch==1.11.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

echo "[GeoMol] Installing RDKit and core deps..."
conda run -n geomol conda install -y rdkit networkx pot scikit-learn tqdm pyyaml -c conda-forge

echo "[GeoMol] Installing PyTorch Geometric..."
conda run -n geomol pip install \
    torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f "https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html"

echo "[GeoMol] Setup complete."
