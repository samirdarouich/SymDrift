#!/bin/bash
# Set up Torsional Diffusion conda environment (Python 3.9, PyTorch 2.1, CUDA 12.1)
set -e

REPO="$(cd "$(dirname "$0")/../torsional-diffusion" 2>/dev/null && pwd)" || {
    echo "[TorsionalDiff] ERROR: torsional-diffusion repo not found. Run clone_baselines.sh first."
    exit 1
}

echo "[TorsionalDiff] Creating conda environment (Python 3.9)..."
conda env list | grep -q "^torsional_diffusion " || \
    conda create -n torsional_diffusion python=3.9 -y --override-channels -c conda-forge

PYTHON=$(conda info --envs | awk '$1=="torsional_diffusion"{print $NF"/bin/python3"}')

echo "[TorsionalDiff] Installing PyTorch (auto-selects CUDA version)..."
$PYTHON -m pip install torch

echo "[TorsionalDiff] Installing numpy<2 and core deps..."
$PYTHON -m pip install "numpy<2"
$PYTHON -m pip install rdkit pyaml matplotlib scipy networkx tqdm

echo "[TorsionalDiff] Installing PyTorch Geometric via conda pyg channel (handles ABI matching)..."
conda install -n torsional_diffusion -y pytorch-scatter pytorch-sparse pytorch-cluster \
    -c pyg -c pytorch -c nvidia
$PYTHON -m pip install torch-geometric

echo "[TorsionalDiff] Installing e3nn..."
$PYTHON -m pip install e3nn

echo "[TorsionalDiff] Setup complete."
