#!/bin/bash
# Set up GeoDiff conda environment (Python 3.7, PyTorch 1.8.1, CUDA 10.2)
set -e

REPO="$(cd "$(dirname "$0")/../GeoDiff" 2>/dev/null && pwd)" || {
    echo "[GeoDiff] ERROR: GeoDiff repo not found. Run clone_baselines.sh first."
    exit 1
}

# Detect CUDA version to pick the right PyTorch/PyG build
CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
if [[ "$CUDA_VER" == 5* ]] || [[ "$CUDA_VER" == 4* ]]; then
    CU_TAG="cu102"
    CUDATOOLKIT="10.2"
else
    CU_TAG="cu113"
    CUDATOOLKIT="11.3"
fi
echo "[GeoDiff] Using CUDA tag: $CU_TAG"

# Create base env with flexible channel priority to avoid strict repo conflicts
echo "[GeoDiff] Creating conda environment (Python 3.7)..."
conda create -n geodiff python=3.7 -y \
    --override-channels -c conda-forge -c defaults

echo "[GeoDiff] Installing PyTorch 1.8.1 + CUDA $CUDATOOLKIT ..."
conda run -n geodiff conda install -y \
    pytorch=1.8.1 cudatoolkit=$CUDATOOLKIT \
    -c pytorch -c conda-forge

echo "[GeoDiff] Installing RDKit and core deps..."
conda run -n geodiff conda install -y \
    rdkit=2020.09.1 numpy=1.19.2 scipy scikit-learn \
    networkx tqdm easydict pyyaml \
    -c conda-forge

echo "[GeoDiff] Installing PyTorch Geometric 1.7.2..."
conda run -n geodiff pip install \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "https://pytorch-geometric.com/whl/torch-1.8.1+${CU_TAG}.html"
conda run -n geodiff pip install torch-geometric==1.7.2

echo "[GeoDiff] Installing repo..."
conda run -n geodiff pip install -e "$REPO" --no-deps 2>/dev/null || true

echo "[GeoDiff] Setup complete."
