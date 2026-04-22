#!/bin/bash
# MCF: benchmark average inference speed using random weights (PerceiverIO backbone).

REPO="$(cd "$(dirname "$0")/../ml-mcf" 2>/dev/null && pwd)" || {
    echo "[MCF] ERROR: ml-mcf repo not found."
    exit 1
}

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/.conda")
PYTHON="$CONDA_BASE/envs/mcf/bin/python3"

if [ ! -x "$PYTHON" ]; then
    echo "[MCF] ERROR: conda env 'mcf' not found at $PYTHON. Run mcf_setup.sh first."
    exit 1
fi

"$PYTHON" - "$REPO" <<'PYEOF'
import sys, os, time, torch
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])

from omegaconf import OmegaConf
from models.architectures import PerceiverIO

device = "cuda" if torch.cuda.is_available() else "cpu"

arch = PerceiverIO(
    pos_embed_config=OmegaConf.create({
        "target": "models.pos_embed.PosEmbed",
        "params": {"embed_type": "trainable", "input_num_channels": 75, "output_num_channels": 128},
    }),
    num_latents=512, d_latents=512, d_model=1024,
    time_sinusoidal_dim=256, num_blocks=8,
    num_self_attends_per_block=2, num_self_attention_heads=4,
    num_cross_attention_heads=4, signal_num_channels=3,
    proj_dim=128, coord_num_channels=75,
    use_flash=False, pos_embed_apply="both",
).to(device)
arch.eval()

N_MOL, N_ATOMS, N_EIGS, COORD_DIM, SIG_DIM, N_STEPS = 50, 10, 32, 75, 3, 50

def run_sampling(n_mol, n_steps):
    cx = torch.randn(n_mol, N_EIGS, COORD_DIM, device=device)
    cy = torch.randn(n_mol, N_EIGS, SIG_DIM, device=device)
    qx = torch.randn(n_mol, N_ATOMS, COORD_DIM, device=device)
    qy = torch.randn(n_mol, N_ATOMS, SIG_DIM, device=device)
    for step in range(n_steps):
        t = torch.full((n_mol,), step / n_steps, device=device)
        with torch.no_grad():
            pred = arch(context_x=cx, context_y=cy, t=t, query_x=qx, query_y=qy)
        qy = qy - (1.0 / n_steps) * pred
    return qy

print("[MCF] Warming up...", flush=True)
run_sampling(5, N_STEPS)

N = 50
print(f"[MCF] Benchmarking {N} molecules ({N_STEPS} DDIM steps)...", flush=True)
if device == "cuda": torch.cuda.synchronize()
start = time.perf_counter()
run_sampling(N, N_STEPS)
if device == "cuda": torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"[MCF] Avg inference speed: {elapsed/N*1000:.1f} ms/sample  (total {elapsed:.2f}s for {N} samples)")
PYEOF
