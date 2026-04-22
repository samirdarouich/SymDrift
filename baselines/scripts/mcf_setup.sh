#!/bin/bash
# Set up MCF (Molecular Conformer Fields) environment.
# Requires Python 3.10 + PyTorch 2.1.0 (+ CUDA 12.2 recommended).
set -e

REPO="$(cd "$(dirname "$0")/../ml-mcf" 2>/dev/null && pwd)" || {
    echo "[MCF] ERROR: ml-mcf repo not found. Run clone_baselines.sh first."
    exit 1
}

echo "[MCF] Creating conda environment 'mcf' (Python 3.10)..."
conda create -n mcf python=3.10 -y || echo "[MCF] Environment already exists."

echo "[MCF] Installing system packages (requires sudo or Docker-free system)..."
# setup.sh normally calls apt-get; skip system packages if not on Linux or already present.
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get install -y ffmpeg libhdf5-dev libffi-dev xvfb libgl1-mesa-glx 2>/dev/null || true
else
    echo "[MCF] apt-get not available (macOS?). Skipping system packages."
fi

echo "[MCF] Installing Python requirements from $REPO/environment/requirements.txt ..."
conda run -n mcf pip install -r "$REPO/environment/requirements.txt"

echo "[MCF] Installing repo in editable mode..."
conda run -n mcf pip install -e "$REPO"

echo "[MCF] Setup complete."
