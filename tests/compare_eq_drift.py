from tspath.datasets import ToyMoleculeDataset
from tspath.generative.drifting import EquivariantDriftingField, DriftingField
import torch

def sample_prior(n: int, device: str = "cpu") -> torch.Tensor:
    """
    Sample n prior configurations: two independent 2D positions from N(0, I),
    zero-centred so the centroid of the 2 nodes is always at the origin.

    This matches the orbit samples, where the two antipodal points always sum
    to (0, 0).

    Returns: (n, 2, 2)  — n pairs of zero-centred 2D positions.
    """
    z = torch.randn(n, 2, 2, device=device)
    z = z - z.mean(dim=1, keepdim=True)   # subtract centroid per configuration
    z = torch.cat([z, torch.zeros(n, 2, 1, device=device)], dim=-1)  # add zero z-coord
    return z

def _so2_rotations(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Compute the all-pairs optimal SO(2) rotation matrices.

    For each pair (i, j), finds R[i,j] = argmin ||src[i] @ R.T - tgt[j]||_F.

    Derivation: maximise tr(R @ H[i,j]) over SO(2), where
        H[i,j] = src[i].T @ tgt[j]  (2x2 cross-covariance).
    Closed form: θ[i,j] = atan2(H[0,1]-H[1,0],  H[0,0]+H[1,1]).

    src : (B, N, 2)
    tgt : (B, N, 2)
    Returns R : (B, B, 2, 2)  — R[i,j] aligns src[i] to tgt[j].
    """
    # H[i,j,a,b] = Σ_n  src[i,n,a] * tgt[j,n,b]
    H = torch.einsum("ina,jnb->ijab", src, tgt)   # (B, B, 2, 2)

    a = H[..., 0, 0] + H[..., 1, 1]               # (B, B)
    b = H[..., 0, 1] - H[..., 1, 0]               # (B, B)
    norm = (a ** 2 + b ** 2).sqrt().clamp(min=1e-8)
    cos_t, sin_t = a / norm, b / norm              # (B, B)

    # Build (B, B, 2, 2) rotation matrices
    R = torch.stack([
        torch.stack([ cos_t, -sin_t], dim=-1),     # row 0
        torch.stack([ sin_t,  cos_t], dim=-1),     # row 1
    ], dim=-2)                                      # (B, B, 2, 2)
    return R

def so2_drifting_field(
    x:     torch.Tensor,   # (B, N, 2) — generated configs  (with grad)
    y_pos: torch.Tensor,   # (B, N, 2) — real configs
    y_neg: torch.Tensor,   # (B, N, 2) — negative configs   (x.detach())
    sigma: float,
    remove_self_repulsion: bool = True,
    align: bool = True,
    use_rmsd: bool = False,
    use_softmax: bool = False,
) -> torch.Tensor:
    """
    Drifting field, optionally with all-pairs SO(2) alignment.

    align=False: plain Euclidean field on flattened (B, N*2) configurations,
                 identical to the original drifting.py algorithm.
    align=True:  all-pairs SO(2) alignment.

    For every pair (x_i, y_j) the optimal rotation R_ij that maps x_i
    closest to y_j is found in closed form.  The RBF kernel uses the
    resulting Procrustes distance, and the drift direction is expressed
    back in x_i's original frame so it can be added directly to x_i.

    Positive drift contribution from y_j toward x_i  (in x_i's frame):
        y_j @ R_ij  −  x_i
    (rotating y_j back by R_ij undoes the alignment, bringing it into x_i's
    coordinate system).

    Returns v : (B, N, 2) — drift vectors in x's original frame.
    """
    B, N, _ = x.shape

    if not align:
        # Plain Euclidean field on flattened configurations — same as original.
        x_flat    = x.reshape(B, -1)
        yp_flat   = y_pos.reshape(B, -1)
        yn_flat   = y_neg.reshape(B, -1)

        diff_p = yp_flat[None] - x_flat[:, None]           # (B, B, N*2)
        diff_n = yn_flat[None] - x_flat[:, None]

        dist2_pos = torch.sqrt((diff_p ** 2).sum(dim=-1))
        dist2_neg = torch.sqrt((diff_n ** 2).sum(dim=-1))
        
        if use_rmsd:
            dist2_pos /= 2**0.5 # rmsd is norm divided by sqrt(num_points)
            dist2_neg /= 2**0.5

        k_pos = torch.exp(-dist2_pos / sigma)
        k_neg = torch.exp(-dist2_neg / sigma)
        if remove_self_repulsion:
            k_neg = k_neg.clone()
            k_neg.fill_diagonal_(0.0)

        w_pos = k_pos / (k_pos.sum(dim=1, keepdim=True) + 1e-8)
        w_neg = k_neg / (k_neg.sum(dim=1, keepdim=True) + 1e-8)
        
        if use_softmax:
            dist2_neg_ = dist2_neg.clone()
            dist2_neg_ += torch.eye(B, device=dist2_neg.device) * 1e6  # large penalty on self-repulsion
            w_pos = torch.softmax(-dist2_pos / sigma, dim=1)
            w_neg = torch.softmax(-dist2_neg_ / sigma, dim=1)
        
        # print("w_pos:", w_pos)
        # print("w_pos_:", w_pos_)
        # print("w_neg:", w_neg)
        # print("w_neg_:", w_neg_)

        drift_pos = (w_pos[..., None] * diff_p).sum(dim=1)
        drift_neg = (w_neg[..., None] * diff_n).sum(dim=1)
        v_flat = drift_pos - drift_neg
        
        return v_flat.reshape(B, N, 2), drift_pos, drift_neg, diff_p, diff_n, dist2_pos, dist2_neg

    # ── all-pairs rotation matrices ──────────────────────────────────────────
    # R_pos[i,j] aligns x[i] to y_pos[j]  (computed without gradient)
    R_pos = _so2_rotations(x.detach(), y_pos)        # (B, B, 2, 2)
    # R_neg[i,j] aligns x[i] to y_neg[j]
    R_neg = _so2_rotations(x.detach(), y_neg)             # (B, B, 2, 2)

    # ── Procrustes distances for the RBF kernel ───────────────────────────────
    # x_aligned[i,j] = x[i] @ R_pos[i,j].T  →  (B, B, N, 2)
    # [i,j,n,b] = Σ_a x[i,n,a] * R_pos[i,j,b,a]
    x_aln  = torch.einsum("ina,ijba->ijnb", x.detach(), R_pos)
    dist2_pos = torch.sqrt(((x_aln - y_pos[None]) ** 2).sum(dim=(-2, -1)))   # (B, B)

    xn_aln = torch.einsum("ina,ijba->ijnb", y_neg, R_neg)
    dist2_neg = torch.sqrt(((xn_aln - y_neg[None]) ** 2).sum(dim=(-2, -1)))  # (B, B)
    
    if use_rmsd:
        dist2_pos /= 2**0.5 # rmsd is norm divided by sqrt(num_points)
        dist2_neg /= 2**0.5

    # ── kernel weights ────────────────────────────────────────────────────────
    k_pos = torch.exp(-dist2_pos / sigma)              # (B, B)
    k_neg = torch.exp(-dist2_neg / sigma)              # (B, B)

    if remove_self_repulsion:
        k_neg = k_neg.clone()
        k_neg.fill_diagonal_(0.0)

    w_pos = k_pos / (k_pos.sum(dim=1, keepdim=True) + 1e-8)
    w_neg = k_neg / (k_neg.sum(dim=1, keepdim=True) + 1e-8)

    # ── drift directions in x_i's original frame ─────────────────────────────
    # y_j rotated back to x_i's frame:  y_j @ R_pos[i,j]
    # [i,j,n,b] = Σ_a y_pos[j,n,a] * R_pos[i,j,a,b]
    y_in_xi  = torch.einsum("jna,ijab->ijnb", y_pos, R_pos)       # (B, B, N, 2)
    yn_in_xi = torch.einsum("jna,ijab->ijnb", y_neg, R_neg)       # (B, B, N, 2)

    diff_pos = y_in_xi  - x[:, None]
    diff_neg = yn_in_xi - x[:, None]
    
    drift_pos = (w_pos[:, :, None, None] * diff_pos).sum(dim=1)
    drift_neg = (w_neg[:, :, None, None] * diff_neg).sum(dim=1)
    
    V = drift_pos - drift_neg   # (B, N, 2)
    return V, drift_pos, drift_neg, diff_pos, diff_neg, dist2_pos, dist2_neg

torch.manual_seed(42)

dataset = ToyMoleculeDataset(n_samples=5, T=800, seed=42)
eq_drifting_field = EquivariantDriftingField(temperature=0.15, aligned=True)
naive_drifting_field = DriftingField(temperature=0.15)
sigma = 0.15
samples = [dataset[i] for i in range(len(dataset))]
y_pos = torch.stack([sample.pos for sample in samples]).float()
x = sample_prior(y_pos.shape[0], device=y_pos.device)
y_neg = x.clone()

# Compare equivariant with vinh with alignment (use and rmsd)
print("Comparing equivariant with vinh with alignment (use and rmsd)")
V, drift_pos, drift_neg, diff_pos, diff_neg, w_pos, w_neg = eq_drifting_field(x.view(-1, 3), y_pos.view(-1, 3), y_neg.view(-1, 3), n_atoms=2)
V_vinh, drift_pos_vinh, drift_neg_vinh, diff_pos_vinh, diff_neg_vinh, w_pos_vinh, w_neg_vinh = so2_drifting_field(x[...,:2], y_pos[...,:2], y_neg[...,:2], sigma=sigma, remove_self_repulsion=True, align=True, use_rmsd=True)
print((V_vinh-V.reshape(-1, 2, 3)[...,:2]).max())

# Compare equivariant with vinh without alignment (use rmsd)
print("\nComparing equivariant with vinh without alignment (use rmsd)")
V___, *___ = eq_drifting_field(x.view(-1, 3), y_pos.view(-1, 3), y_neg.view(-1, 3), n_atoms=2, aligned=False)
V____, *____ = so2_drifting_field(x[...,:2], y_pos[...,:2], y_neg[...,:2], sigma=sigma, remove_self_repulsion=True, align=False, use_rmsd=True)
print((V___.reshape(-1, 2, 3)[...,:2] - V____).max())

# Compare naive with vinh without alignment (use normal distance not rmsd)
print("\nComparing naive with vinh without alignment (use normal distance not rmsd)")
print("use no softmax")
V_, *_ = naive_drifting_field(x.view(-1, 6), y_pos.view(-1, 6), y_neg.view(-1, 6))
V__, *__ = so2_drifting_field(x[...,:2], y_pos[...,:2], y_neg[...,:2], sigma=sigma, remove_self_repulsion=True, align=False, use_rmsd=False)
print((V_.reshape(-1, 2, 3)[...,:2] - V__).max())
print("use softmax")
V__, *__ = so2_drifting_field(x[...,:2], y_pos[...,:2], y_neg[...,:2], sigma=sigma, remove_self_repulsion=True, align=False, use_rmsd=False, use_softmax=True)
print((V_.reshape(-1, 2, 3)[...,:2] - V__).max())