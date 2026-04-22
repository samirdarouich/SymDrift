#!/bin/bash
# Download MCF QM9 checkpoint and processed QM9 data from Apple CDN.
set -e

REPO="$(cd "$(dirname "$0")/../ml-mcf" 2>/dev/null && pwd)" || {
    echo "[MCF] ERROR: ml-mcf repo not found."
    exit 1
}

# --- Checkpoint ---
CKPT_DIR="$REPO/ckpts"
CKPT_FILE="$CKPT_DIR/mcf_qm9.ckpt"
mkdir -p "$CKPT_DIR"

if [ -f "$CKPT_FILE" ]; then
    echo "[MCF] Checkpoint already exists at $CKPT_FILE — skipping."
else
    echo "[MCF] Downloading checkpoints archive..."
    wget -q --show-progress \
        "https://docs-assets.developer.apple.com/ml-research/datasets/mcf/ckpts/mcf_ckpts.tar.gz" \
        -O "$REPO/mcf_ckpts.tar.gz"
    echo "[MCF] Extracting checkpoints..."
    tar -xzf "$REPO/mcf_ckpts.tar.gz" -C "$REPO"
    rm "$REPO/mcf_ckpts.tar.gz"
    echo "[MCF] Checkpoint extracted to $CKPT_DIR"
fi

# --- QM9 Data ---
DATA_DIR="$REPO/data"
mkdir -p "$DATA_DIR"

if [ -d "$DATA_DIR/processed_qm9" ]; then
    echo "[MCF] QM9 data already exists at $DATA_DIR/processed_qm9 — skipping."
else
    echo "[MCF] Downloading processed QM9 data..."
    wget -q --show-progress \
        "https://docs-assets.developer.apple.com/ml-research/datasets/mcf/qm9_new/processed_qm9.tar.gz" \
        -O "$DATA_DIR/processed_qm9.tar.gz"
    echo "[MCF] Extracting QM9 data..."
    tar -xzf "$DATA_DIR/processed_qm9.tar.gz" -C "$DATA_DIR"
    rm "$DATA_DIR/processed_qm9.tar.gz"
    echo "[MCF] QM9 data extracted to $DATA_DIR/processed_qm9"
fi

echo "[MCF] Download complete."
