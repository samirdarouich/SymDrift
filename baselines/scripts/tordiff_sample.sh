#!/bin/bash
# Torsional Diffusion: benchmark average inference speed using random weights.

REPO="$(cd "$(dirname "$0")/../torsional-diffusion" 2>/dev/null && pwd)" || {
    echo "[TorsionalDiff] ERROR: torsional-diffusion repo not found."
    exit 1
}

PYTHON=$(conda info --envs 2>/dev/null | awk '$1=="torsional_diffusion"{print $NF"/bin/python3"}')

if [ ! -x "$PYTHON" ]; then
    echo "[TorsionalDiff] ERROR: conda env 'torsional_diffusion' not found at $PYTHON. Run tordiff_setup.sh first."
    exit 1
fi

"$PYTHON" - "$REPO" <<'PYEOF'
import sys, os, time, types, torch, yaml
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])

from utils.utils import get_model
from utils.featurization import featurize_mol_from_smiles
from diffusion.sampling import sample

model_params = dict(
    in_node_features=44, in_edge_features=4,
    sigma_embed_dim=32, radius_embed_dim=50,
    num_conv_layers=4, max_radius=5.0,
    scale_by_sigma=True, ns=32, nv=8,
    residual=True, batch_norm=True,
    use_second_order_repr=False,
    sigma_min=0.01 * 3.14159, sigma_max=3.14159,
    dataset="qm9",
)
args = types.SimpleNamespace(**model_params)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model(args).to(device)
model.eval()

SMILES = [
    "CC","CCC","CO","CCO","CN","CCN","C=C","C=O",
    "C1CC1","C1CCC1","C1CO1","C1CCO1","C1CN1","C1CCN1",
    "c1ccccc1","Cc1ccccc1","CF","CCF",
    "CC(C)O","CC(C)=O","CC(C)N","CC(C)C","CCC=O",
    "CCCC","CCCN","CCCO","C=CC",
    "C1CCCC1","C1CCCO1","C1CCNC1",
    "CC(F)F","CCCl","CCBr",
    "NCC(=O)O","CC(N)C(=O)O","OCC(O)CO",
    "CC1CC1","CCCCN","CCCCO","CC(O)CO",
    "C(F)(F)F","CC(C)(C)C","CCCCC","CCCC=O","CCCCC=O",
    "c1ccncc1","c1ccoc1","Cc1ccncc1","CC(=O)OCC","CCC(=O)O",
    "CC(C)CC","CCN(CC)CC","C1CCCCC1","CC1CCCCC1","c1ccc(C)cc1",
]

def get_conformers(smiles):
    result = []
    for smi in smiles:
        try:
            mol, data = featurize_mol_from_smiles(smi, dataset="qm9")
            if mol is not None:
                result.append((mol, data))
        except Exception:
            pass
    return result

print("[TorsionalDiff] Warming up...", flush=True)
warmup = get_conformers(SMILES[:5])
if warmup:
    with torch.no_grad():
        sample([d for _, d in warmup], model, args.sigma_max, args.sigma_min,
               inference_steps=20, batch_size=32, ode=False, likelihood=None, pdb=None)

N = 50
print(f"[TorsionalDiff] Benchmarking {N} molecules...", flush=True)
bench = get_conformers(SMILES[:N])
conformers = [d for _, d in bench]

if device == "cuda": torch.cuda.synchronize()
start = time.perf_counter()
with torch.no_grad():
    sample(conformers, model, args.sigma_max, args.sigma_min,
           inference_steps=20, batch_size=32, ode=False, likelihood=None, pdb=None)
if device == "cuda": torch.cuda.synchronize()
elapsed = time.perf_counter() - start

n = len(conformers)
print(f"[TorsionalDiff] Avg inference speed: {elapsed/n*1000:.1f} ms/sample  (total {elapsed:.2f}s for {n} samples)")
PYEOF
