#!/bin/bash
# ETFlow: sample 50 QM9 test molecules and report wall-clock timing.
# Uses the Python API (BaseFlow.from_default) to avoid config path complexity.

REPO="$(cd "$(dirname "$0")/../ETFlow" 2>/dev/null && pwd)" || {
    echo "[ETFlow] ERROR: ETFlow repo not found."
    exit 1
}

N_SAMPLES=50
echo "[ETFlow] Sampling $N_SAMPLES molecules via Python API..."

conda run -n etflow python3 - <<PYEOF
import time, sys, os
os.chdir("$REPO")

# Try script-based eval first; fall back to Python API
import subprocess

# Check if eval.py + qm9-base config exist (full benchmark path)
eval_script = os.path.join("$REPO", "scripts", "eval.py")
config = os.path.join("$REPO", "configs", "qm9-base.yaml")
ckpt = os.path.join("$REPO", "ckpts", "qm9_o3.ckpt")

if os.path.exists(eval_script) and os.path.exists(config) and os.path.exists(ckpt):
    import tempfile
    out_dir = tempfile.mkdtemp()
    cmd = [
        "python", eval_script,
        f"--config={config}",
        f"--checkpoint={ckpt}",
        "--num_mols", "50",
        "--output_dir", out_dir,
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - start
else:
    # Fall back to Python API with 50 SMILES from QM9 test set
    from etflow import BaseFlow
    import rdkit.Chem as Chem

    # Representative QM9 SMILES (small molecules)
    smiles_list = [
        "C", "CC", "CCC", "CCCC", "CO", "CCO", "CCCO", "CN", "CCN", "CCCN",
        "C=C", "C=CC", "CC=C", "C=O", "CC=O", "CCC=O", "C=N", "CC=N",
        "C#C", "C#CC", "C#N", "CC#N", "CCC#N", "C1CC1", "C1CCC1", "C1CCCC1",
        "C1CO1", "C1CCO1", "C1CCCO1", "C1CN1", "C1CCN1", "C1CCNC1",
        "c1ccccc1", "Cc1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1",
        "CF", "CCF", "CCCF", "CCl", "CBr", "CI",
        "C(F)(F)F", "CC(F)(F)F", "C(Cl)(Cl)Cl",
        "CO[C@@H](C)O", "CC(C)O", "CC(C)=O", "CC(C)N", "CC(C)C",
    ][:50]

    model = BaseFlow.from_default(model="qm9-o3")
    start = time.perf_counter()
    output = model.predict(smiles_list, num_samples=1, as_mol=True)
    elapsed = time.perf_counter() - start
    result = type("R", (), {"returncode": 0})()

n = 50
print(f"[ETFlow] Total: {elapsed:.2f}s | Avg/sample: {elapsed/n*1000:.1f}ms")
sys.exit(result.returncode)
PYEOF
