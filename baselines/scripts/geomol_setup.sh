#!/bin/bash
# Set up GeoMol conda environment (Python 3.9, PyTorch 1.11, PyG).
set -e

REPO="$(cd "$(dirname "$0")/../GeoMol" 2>/dev/null && pwd)" || {
    echo "[GeoMol] ERROR: GeoMol repo not found. Run clone_baselines.sh first."
    exit 1
}

TORCH_VERSION="1.11.0"
CU_TAG="cu113"
CUDATOOLKIT="11.3"

echo "[GeoMol] Creating conda environment (Python 3.9)..."
conda create -n geomol python=3.9 -y \
    --override-channels -c conda-forge -c defaults

echo "[GeoMol] Installing PyTorch $TORCH_VERSION + CUDA $CUDATOOLKIT ..."
conda run -n geomol conda install -y \
    pytorch=$TORCH_VERSION cudatoolkit=$CUDATOOLKIT \
    -c pytorch -c conda-forge

echo "[GeoMol] Installing RDKit and core deps..."
conda run -n geomol conda install -y \
    rdkit networkx pot scikit-learn tqdm pyyaml \
    -c conda-forge

echo "[GeoMol] Installing PyTorch Geometric..."
conda run -n geomol pip install \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CU_TAG}.html"
conda run -n geomol pip install torch-geometric

echo "[GeoMol] Installing repo deps..."
conda run -n geomol pip install -e "$REPO" --no-deps 2>/dev/null || true

echo "[GeoMol] Setup complete."
