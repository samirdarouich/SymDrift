#!/bin/bash
# Torsional Diffusion: sample 50 QM9 test molecules and report wall-clock timing.

REPO="$(cd "$(dirname "$0")/../torsional-diffusion" 2>/dev/null && pwd)" || {
    echo "[TorsionalDiff] ERROR: torsional-diffusion repo not found."
    exit 1
}

MODEL_DIR="$REPO/workdir/qm9_default"
if [ ! -f "$MODEL_DIR/best_model.pt" ]; then
    echo "[TorsionalDiff] ERROR: Checkpoint not found at $MODEL_DIR/best_model.pt. Run tordiff_download.sh first."
    exit 1
fi

TEST_CSV_FULL="$REPO/data/QM9/test_smiles.csv"
if [ ! -f "$TEST_CSV_FULL" ]; then
    echo "[TorsionalDiff] ERROR: Test CSV not found at $TEST_CSV_FULL"
    exit 1
fi

# Create a 50-molecule subset (header + 50 rows)
TMPDIR_LOCAL=$(mktemp -d)
TEST_CSV_50="$TMPDIR_LOCAL/test_50.csv"
{ head -1 "$TEST_CSV_FULL"; tail -n +2 "$TEST_CSV_FULL" | head -50; } > "$TEST_CSV_50"

N_SAMPLES=50
echo "[TorsionalDiff] Sampling $N_SAMPLES molecules (20 inference steps)..."

conda run -n torsional_diffusion python3 - <<PYEOF
import subprocess, time, sys, os
os.chdir("$REPO")
cmd = [
    "python", "generate_confs.py",
    "--test_csv", "$TEST_CSV_50",
    "--inference_steps", "20",
    "--model_dir", "$MODEL_DIR",
    "--out", "$TMPDIR_LOCAL/tordiff_out.pkl",
    "--batch_size", "32",
    "--no_energy",
]
start = time.perf_counter()
result = subprocess.run(cmd)
elapsed = time.perf_counter() - start
n = $N_SAMPLES
print(f"[TorsionalDiff] Total: {elapsed:.2f}s | Avg/sample: {elapsed/n*1000:.1f}ms")
sys.exit(result.returncode)
PYEOF

rm -rf "$TMPDIR_LOCAL"
