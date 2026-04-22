#!/bin/bash
# Set up Torsional Diffusion conda environment (Python 3.9, PyTorch 1.11).
set -e

REPO="$(cd "$(dirname "$0")/../torsional-diffusion" 2>/dev/null && pwd)" || {
    echo "[TorsionalDiff] ERROR: torsional-diffusion repo not found. Run clone_baselines.sh first."
    exit 1
}

TORCH_VERSION="1.11.0"
CU_TAG="cu113"
CUDATOOLKIT="11.3"

echo "[TorsionalDiff] Creating conda environment (Python 3.9)..."
conda create -n torsional_diffusion python=3.9 -y \
    --override-channels -c conda-forge -c defaults

echo "[TorsionalDiff] Installing PyTorch $TORCH_VERSION + CUDA $CUDATOOLKIT ..."
conda run -n torsional_diffusion conda install -y \
    pytorch=$TORCH_VERSION cudatoolkit=$CUDATOOLKIT \
    -c pytorch -c conda-forge

echo "[TorsionalDiff] Installing RDKit and core deps..."
conda run -n torsional_diffusion conda install -y \
    rdkit pyaml matplotlib scipy networkx tqdm \
    -c conda-forge

echo "[TorsionalDiff] Installing PyTorch Geometric..."
conda run -n torsional_diffusion pip install \
    torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f "https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CU_TAG}.html"

echo "[TorsionalDiff] Installing e3nn..."
conda run -n torsional_diffusion pip install e3nn

echo "[TorsionalDiff] Setup complete."
