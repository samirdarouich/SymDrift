#!/bin/bash
# Download GeoDiff QM9 pretrained checkpoint from Google Drive.
# Folder: https://drive.google.com/drive/folders/1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh
set -e

REPO="$(cd "$(dirname "$0")/../GeoDiff" 2>/dev/null && pwd)" || {
    echo "[GeoDiff] ERROR: GeoDiff repo not found."
    exit 1
}

CKPT_DIR="$REPO/logs/qm9_default/checkpoints"
mkdir -p "$CKPT_DIR"

if [ -f "$CKPT_DIR/1000000.pt" ]; then
    echo "[GeoDiff] Checkpoint already exists at $CKPT_DIR/1000000.pt — skipping."
    exit 0
fi

command -v gdown >/dev/null 2>&1 || {
    echo "[GeoDiff] gdown not found. Installing..."
    pip install -q gdown
}

echo "[GeoDiff] Downloading QM9 checkpoint from Google Drive..."
# Download the qm9_default folder contents
gdown --folder "https://drive.google.com/drive/folders/1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh" \
    -O "$REPO/logs" --remaining-ok

echo "[GeoDiff] Checkpoint downloaded to $CKPT_DIR"
