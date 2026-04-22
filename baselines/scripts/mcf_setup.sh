#!/bin/bash
# Set up MCF conda environment (Python 3.10, PyTorch 2.1, CUDA 12.1)
# Uses pip for PyTorch to avoid conda MKL linking issues on HPC clusters.
set -e

REPO="$(cd "$(dirname "$0")/../ml-mcf" 2>/dev/null && pwd)" || {
    echo "[MCF] ERROR: ml-mcf repo not found. Run clone_baselines.sh first."
    exit 1
}

echo "[MCF] Creating conda environment (Python 3.10)..."
conda env list | grep -q "^mcf " || \
    conda create -n mcf python=3.10 -y --override-channels -c conda-forge

echo "[MCF] Installing PyTorch 2.1.0 + CUDA 12.1 via pip..."
conda run -n mcf pip install \
    torch==2.1.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121

echo "[MCF] Installing mamba-ssm (needs --no-build-isolation to see torch)..."
conda run -n mcf pip install mamba-ssm==1.0.1 --no-build-isolation

echo "[MCF] Installing remaining requirements (excluding mamba-ssm)..."
grep -v "mamba.ssm" "$REPO/environment/requirements.txt" > /tmp/mcf_requirements.txt
conda run -n mcf pip install -r /tmp/mcf_requirements.txt

echo "[MCF] Installing repo..."
conda run -n mcf pip install -e "$REPO" --no-deps 2>/dev/null || true

echo "[MCF] Setup complete."
