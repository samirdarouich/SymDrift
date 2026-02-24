from tspath.datasets import ToyMoleculeDataset
import matplotlib.pyplot as plt
import torch

# The single canonical data sample: two antipodal points on the unit circle.
CANONICAL = torch.tensor([[0.0, -1.0], [0.0, 1.0]])   # (2, 2)

# Fixed node-type features: one-hot so EGNN can tell the two nodes apart.
# These are rotation-invariant scalars — they never change.
NODE_FEAT = torch.tensor([[1.0], [1.0]])    # (2, n_feat=1)

# Inter-point distance of the canonical configuration (used as regulariser).
CANONICAL_DIST = (CANONICAL[0] - CANONICAL[1]).norm().item()   # = 2.0

def sample_orbit(n: int, device: str = "cpu") -> torch.Tensor:
    import math
    """
    Sample n configurations from the SO(2) orbit of CANONICAL.

    For each sample, draw θ ~ Uniform[0, 2π) and apply the rotation
        R(θ) = [[cos θ, -sin θ],
                [sin θ,  cos θ]]
    to every point in CANONICAL.

    Returns: (n, 2, 2)  — n configurations of 2 points each.
    """
    theta = torch.rand(n, device=device) * (2 * math.pi)
    cos_t, sin_t = theta.cos(), theta.sin()

    # Build (n, 2, 2) rotation matrices.
    R = torch.stack([
        torch.stack([ cos_t, -sin_t], dim=1),   # row 0
        torch.stack([ sin_t,  cos_t], dim=1),   # row 1
    ], dim=1)                                    # (n, 2, 2)

    # Apply R to each of the 2 canonical points.
    # y[b, node, coord] = sum_e R[b, coord, e] * CANONICAL[node, e]
    canonical = CANONICAL.to(device)
    y = torch.einsum("bce,ne->bnc", R, canonical)   # (n, 2, 2)
    return y

y = sample_orbit(100)

dataset = ToyMoleculeDataset(num_samples=10, T=0, seed=42,r0=2.0)
dataset.positions.shape
pos = torch.stack([data.pos for data in dataset])
plt.scatter(pos[:,:, 0], pos[:,:, 1], alpha=0.5)
plt.scatter(y[:,:, 0], y[:,:, 1], alpha=0.5)
plt.axis("equal")
plt.show()