#!/bin/bash
# MCF: sample 50 QM9 test molecules and report wall-clock timing.
# Creates a temporary config overriding num_molecules=50.

REPO="$(cd "$(dirname "$0")/../ml-mcf" 2>/dev/null && pwd)" || {
    echo "[MCF] ERROR: ml-mcf repo not found."
    exit 1
}

CKPT="$REPO/ckpts/mcf_qm9.ckpt"
if [ ! -f "$CKPT" ]; then
    echo "[MCF] ERROR: Checkpoint not found at $CKPT. Run mcf_download.sh first."
    exit 1
fi

BASE_CONFIG="$REPO/configs/test_qm9.yaml"
if [ ! -f "$BASE_CONFIG" ]; then
    echo "[MCF] ERROR: Config not found at $BASE_CONFIG"
    exit 1
fi

# Write a temporary config that limits evaluation to 50 molecules
TMPDIR_LOCAL=$(mktemp -d)
TMP_CONFIG="$TMPDIR_LOCAL/test_qm9_50.yaml"

# Copy base config and override num_molecules
sed 's/num_molecules:.*/num_molecules: 50/' "$BASE_CONFIG" > "$TMP_CONFIG"
# If the key doesn't exist in the config, append it
grep -q "num_molecules" "$TMP_CONFIG" || echo "num_molecules: 50" >> "$TMP_CONFIG"

N_SAMPLES=50
echo "[MCF] Sampling $N_SAMPLES molecules..."

conda run -n mcf python3 - <<PYEOF
import subprocess, time, sys, os
os.chdir("$REPO")
cmd = [
    "python", "test_mcf.py",
    "--task_config", "$TMP_CONFIG",
]
start = time.perf_counter()
result = subprocess.run(cmd)
elapsed = time.perf_counter() - start
n = $N_SAMPLES
print(f"[MCF] Total: {elapsed:.2f}s | Avg/sample: {elapsed/n*1000:.1f}ms")
sys.exit(result.returncode)
PYEOF

rm -rf "$TMPDIR_LOCAL"
