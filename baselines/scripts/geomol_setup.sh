#!/bin/bash
# Set up GeoMol conda environment (Python 3.9, PyTorch 2.1, CUDA 12.1)
set -e

REPO="$(cd "$(dirname "$0")/../GeoMol" 2>/dev/null && pwd)" || {
    echo "[GeoMol] ERROR: GeoMol repo not found. Run clone_baselines.sh first."
    exit 1
}

echo "[GeoMol] Creating conda environment (Python 3.9)..."
conda env list | grep -q "^geomol " || \
    conda create -n geomol python=3.9 -y --override-channels -c conda-forge

PYTHON=$(conda info --envs | awk '$1=="geomol"{print $NF"/bin/python3"}')

echo "[GeoMol] Installing PyTorch (auto-selects CUDA version)..."
$PYTHON -m pip install torch

echo "[GeoMol] Installing numpy<2 and core deps..."
$PYTHON -m pip install "numpy<2"
$PYTHON -m pip install rdkit networkx "pot>=0.7.0" scikit-learn tqdm pyyaml

echo "[GeoMol] Installing PyTorch Geometric (prebuilt wheels for torch 2.1 + cu121)..."
$PYTHON -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    --find-links "https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html"
$PYTHON -m pip install torch-geometric

echo "[GeoMol] Setup complete."
