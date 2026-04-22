"""
Check that loading a checkpoint actually updates the model parameters
by comparing loaded weights against a freshly initialized MLP.
"""
import torch
from tspath.model import MLP

CKPT_PATH = "/home/vinhtong/tspath/runs/tspath_run_2026_02_23_09_41_14/checkpoints/last.ckpt"

# --- fresh model (random init) ---
torch.manual_seed(0)
fresh = MLP()
fresh_sd = {k: v.clone() for k, v in fresh.state_dict().items()}

# --- loaded model ---
ckpt = torch.load(CKPT_PATH, weights_only=False)
loaded = MLP()
loaded.load_state_dict(
    {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
)
loaded_sd = loaded.state_dict()

# --- compare ---
print(f"Checkpoint epoch : {ckpt.get('epoch', 'n/a')}")
print(f"Checkpoint step  : {ckpt.get('global_step', 'n/a')}")
print()
print(f"{'param':<35} {'shape':<20} {'same as fresh?':<16} {'loaded norm':>12}  {'fresh norm':>12}")
print("-" * 100)

all_updated = True
for name, loaded_param in loaded_sd.items():
    fresh_param = fresh_sd[name]
    same = torch.allclose(loaded_param, fresh_param)
    if same:
        all_updated = False
    print(
        f"{name:<35} {str(tuple(loaded_param.shape)):<20} {'YES (NOT updated)' if same else 'no (updated)':<16}"
        f" {loaded_param.norm().item():>12.4f}  {fresh_param.norm().item():>12.4f}"
    )

print()
if all_updated:
    print("✓ All parameters differ from random init — checkpoint loaded correctly.")
else:
    print("✗ Some parameters are identical to random init — checkpoint may not have loaded correctly.")
