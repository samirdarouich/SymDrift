#!/bin/bash
# Set up Torsional Diffusion conda environment (Python 3.9, PyTorch 1.11, CUDA 11.3)
# Uses pip for PyTorch to avoid conda MKL linking issues on HPC clusters.
set -e

REPO="$(cd "$(dirname "$0")/../torsional-diffusion" 2>/dev/null && pwd)" || {
    echo "[TorsionalDiff] ERROR: torsional-diffusion repo not found. Run clone_baselines.sh first."
    exit 1
}

echo "[TorsionalDiff] Creating conda environment (Python 3.9)..."
conda env list | grep -q "^torsional_diffusion " || \
    conda create -n torsional_diffusion python=3.9 -y --override-channels -c conda-forge

echo "[TorsionalDiff] Installing PyTorch 1.11.0 + CUDA 11.3 via pip..."
conda run -n torsional_diffusion pip install \
    torch==1.11.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

echo "[TorsionalDiff] Installing RDKit and core deps..."
conda run -n torsional_diffusion conda install -y rdkit pyaml matplotlib scipy networkx tqdm -c conda-forge

echo "[TorsionalDiff] Installing PyTorch Geometric..."
conda run -n torsional_diffusion pip install \
    torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f "https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html"

echo "[TorsionalDiff] Installing e3nn..."
conda run -n torsional_diffusion pip install e3nn

echo "[TorsionalDiff] Setup complete."
