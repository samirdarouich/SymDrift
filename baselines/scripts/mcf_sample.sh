#!/bin/bash
# MCF: benchmark average inference speed using random weights.
# Instantiates PerceiverIO backbone directly (bypasses Lightning + metric overhead).

REPO="$(cd "$(dirname "$0")/../ml-mcf" 2>/dev/null && pwd)" || {
    echo "[MCF] ERROR: ml-mcf repo not found. Run clone_baselines.sh first."
    exit 1
}

conda run -n mcf python3 - "$REPO" <<'PYEOF'
import sys, os, time, math, torch
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])

from omegaconf import OmegaConf
from models.architectures import PerceiverIO

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate backbone from qm9_pio.yaml params (random weights)
arch = PerceiverIO(
    pos_embed_config=OmegaConf.create({
        "target": "models.pos_embed.PosEmbed",
        "params": {"embed_type": "trainable", "input_num_channels": 75, "output_num_channels": 128},
    }),
    num_latents=512,
    d_latents=512,
    d_model=1024,
    time_sinusoidal_dim=256,
    num_blocks=8,
    num_self_attends_per_block=2,
    num_self_attention_heads=4,
    num_cross_attention_heads=4,
    signal_num_channels=3,
    proj_dim=128,
    coord_num_channels=75,
    use_flash=False,  # flash attention may not be available
    pos_embed_apply="both",
).to(device)
arch.eval()

# Simulate sampling loop for N molecules.
# Each QM9 molecule: ~10 atoms, 32 eigenfunctions as positional encoding.
N_MOL = 50
N_ATOMS = 10       # avg QM9 molecule
N_EIGS = 32        # eigenfunctions used as context
COORD_DIM = 75     # input_coord_num_channels
SIG_DIM = 3        # xyz signal
N_DDIM_STEPS = 50  # num_timesteps_ddim from eval config

def run_sampling(n_mol, n_steps, device):
    # context = known conformers (n_eigs eigenfunctions)
    context_x = torch.randn(n_mol, N_EIGS, COORD_DIM, device=device)
    context_y = torch.randn(n_mol, N_EIGS, SIG_DIM, device=device)
    # query = atoms we want to generate positions for
    query_x = torch.randn(n_mol, N_ATOMS, COORD_DIM, device=device)
    query_y = torch.randn(n_mol, N_ATOMS, SIG_DIM, device=device)

    # Simple DDIM-style loop (same structure as MCF.ddim_sample)
    for step in range(n_steps):
        t = torch.full((n_mol,), step / n_steps, device=device)
        with torch.no_grad():
            pred = arch(context_x=context_x, context_y=context_y,
                        t=t, query_x=query_x, query_y=query_y)
        # DDIM update (simplified)
        query_y = query_y - (1.0 / n_steps) * pred
    return query_y

N_WARMUP = 5
print("[MCF] Warming up...", flush=True)
run_sampling(N_WARMUP, N_DDIM_STEPS, device)

print(f"[MCF] Benchmarking {N_MOL} molecules ({N_DDIM_STEPS} DDIM steps each)...", flush=True)
if device == "cuda":
    torch.cuda.synchronize()
start = time.perf_counter()
run_sampling(N_MOL, N_DDIM_STEPS, device)
if device == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"[MCF] Avg inference speed: {elapsed/N_MOL*1000:.1f} ms/sample  (total {elapsed:.2f}s for {N_MOL} samples)")
PYEOF
