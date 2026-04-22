#!/bin/bash
# Torsional Diffusion: benchmark average inference speed using random weights.
# Creates a temp model_dir with default QM9 params and random checkpoint.

REPO="$(cd "$(dirname "$0")/../torsional-diffusion" 2>/dev/null && pwd)" || {
    echo "[TorsionalDiff] ERROR: torsional-diffusion repo not found. Run clone_baselines.sh first."
    exit 1
}

conda run -n torsional_diffusion python3 - "$REPO" <<'PYEOF'
import sys, os, time, tempfile, torch, yaml
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])

from utils.utils import get_model
from utils.featurization import featurize_mol_from_smiles
from diffusion.sampling import sample
from rdkit import Chem
from rdkit.Chem import AllChem
import types

# QM9 model hyperparameters (from utils/parsing.py defaults, QM9-specific)
model_params = dict(
    in_node_features=44,   # QM9 (drugs=74)
    in_edge_features=4,
    sigma_embed_dim=32,
    radius_embed_dim=50,
    num_conv_layers=4,
    max_radius=5.0,
    scale_by_sigma=True,
    ns=32,
    nv=8,
    residual=True,
    batch_norm=True,
    use_second_order_repr=False,
    sigma_min=0.01 * 3.14159,
    sigma_max=3.14159,
    dataset="qm9",
)

# Write temp model_dir so generate_confs.py-style code can load it if needed
tmpdir = tempfile.mkdtemp()
with open(f"{tmpdir}/model_parameters.yml", "w") as f:
    yaml.dump(model_params, f)

args = types.SimpleNamespace(**model_params)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model(args).to(device)  # random weights
model.eval()

SMILES = [
    "CC", "CCC", "CO", "CCO", "CN", "CCN", "C=C", "C=O",
    "C1CC1", "C1CCC1", "C1CO1", "C1CCO1", "C1CN1", "C1CCN1",
    "c1ccccc1", "Cc1ccccc1", "CF", "CCF",
    "CC(C)O", "CC(C)=O", "CC(C)N", "CC(C)C", "CCC=O",
    "CCCC", "CCCN", "CCCO", "C=CC",
    "C1CCCC1", "C1CCCO1", "C1CCNC1",
    "CC(F)F", "CCCl", "CCBr",
    "NCC(=O)O", "CC(N)C(=O)O", "OCC(O)CO",
    "CC1CC1", "CCCCN", "CCCCO", "CC(O)CO",
    "C(F)(F)F", "CC(C)(C)C", "CCCCC", "CCCC=O", "CCCCC=O",
    "c1ccncc1", "c1ccoc1", "Cc1ccncc1", "CC(=O)OCC", "CCC(=O)O",
    "CC(C)CC", "CCN(CC)CC", "C1CCCCC1", "CC1CCCCC1", "c1ccc(C)cc1",
]

def get_conformers(smiles, dataset="qm9"):
    """Use torsional diffusion's featurization to get seed conformers."""
    mols_data = []
    for smi in smiles:
        try:
            mol, data = featurize_mol_from_smiles(smi, dataset=dataset)
            if mol is not None:
                mols_data.append((mol, data))
        except Exception:
            pass
    return mols_data

N_WARMUP = 5
N_BENCH = 50

print("[TorsionalDiff] Warming up...", flush=True)
warmup_data = get_conformers(SMILES[:N_WARMUP])
if warmup_data:
    with torch.no_grad():
        conformers_w = [d for _, d in warmup_data[:N_WARMUP]]
        sample(conformers_w, model, args.sigma_max, args.sigma_min,
               inference_steps=20, batch_size=32, ode=False, likelihood=None,
               pdb=None)

print(f"[TorsionalDiff] Benchmarking {N_BENCH} molecules...", flush=True)
bench_data = get_conformers(SMILES[:N_BENCH])
conformers_b = [d for _, d in bench_data]

if device == "cuda":
    torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    sample(conformers_b, model, args.sigma_max, args.sigma_min,
           inference_steps=20, batch_size=32, ode=False, likelihood=None,
           pdb=None)
if device == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start

n = len(conformers_b)
print(f"[TorsionalDiff] Avg inference speed: {elapsed/n*1000:.1f} ms/sample  (total {elapsed:.2f}s for {n} samples)")

import shutil; shutil.rmtree(tmpdir)
PYEOF
