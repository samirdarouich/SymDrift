import torch
from ase.build.rotate import minimize_rotation_and_translation
from ase.io import read
from tspath.generative.drifting import minimal_distance
from tspath.utils import get_shortest_path_fast_batched_x_1, get_rmsd_batch

def minimal_distance_scatter(x, y):
    """
    x: (N, n_atoms, 3)
    y: (M, n_atoms, 3)
    returns: (N, M) RMSD matrix
    """
    N, n_atoms, _ = x.shape
    M = y.shape[0]
    device = x.device

    # Create all N x M pairs
    # x_pairs: (N, M, n_atoms, 3)
    # y_pairs: (N, M, n_atoms, 3)
    x_pairs = x[:, None, :, :].expand(N, M, n_atoms, 3)
    y_pairs = y[None, :, :, :].expand(N, M, n_atoms, 3)

    # Flatten to one big batch
    x_flat = x_pairs.reshape(N * M * n_atoms, 3)
    y_flat = y_pairs.reshape(N * M * n_atoms, 3)

    # Build batch index for each molecule pair
    batch = torch.arange(N * M, device=device).repeat_interleave(n_atoms)

    # Get aligned y for all pairs at once
    y_aligned = get_shortest_path_fast_batched_x_1(x_flat, y_flat, batch)
    
    # Compute directional difference for all pairs at once
    diff = y_aligned - x_flat  # (N*M*n_atoms, 3)

    # Compute RMSD for all pairs at once
    rmsd = get_rmsd_batch(x_flat, y_aligned, batch).view(N, M)

    return rmsd, diff.view(N, M, n_atoms, 3)


def kabsch_single(X, Y):
    # X, Y: (N, 3)

    centroid_X = X.mean(dim=0, keepdim=True)
    centroid_Y = Y.mean(dim=0, keepdim=True)

    Xc = X - centroid_X
    Yc = Y - centroid_Y

    H = Xc.T @ Yc
    U, S, Vt = torch.linalg.svd(H)

    # Reflection correction
    R = Vt.T @ U.T
    det = torch.det(R)
    sign = torch.where(det < 0, -1.0, 1.0)
    Vt = Vt.clone()
    Vt[-1, :] *= sign
    R = Vt.T @ U.T

    X_aligned = Xc @ R.T + centroid_Y

    return R, X_aligned

def kabsch_batched(X, Y):
    """
    align y to x
    X, Y: (B, N, 3)
    """

    centroid_X = X.mean(dim=1, keepdim=True)
    centroid_Y = Y.mean(dim=1, keepdim=True)

    Xc = X - centroid_X
    Yc = Y - centroid_Y

    H = torch.matmul(Yc.transpose(1, 2), Xc)  # (B,3,3)

    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(1, 2)

    R = torch.matmul(V, U.transpose(1, 2))

    # Reflection correction (correct batched version)
    det = torch.det(R)
    sign = torch.ones_like(det)
    sign[det < 0] = -1.0

    # Flip last column of V
    V[:, :, -1] *= sign.unsqueeze(-1)

    R = torch.matmul(V, U.transpose(1, 2))

    # Apply rotation
    Y_rot = torch.matmul(Yc, R.transpose(1, 2))

    Y_aligned = Y_rot + centroid_X

    return Y_aligned, R

# Gold version: ASE
samples = read("/home/samirdarouich/projects/TS_physics/tspath/data/transition1x_eq/raw/t1x_eq_C5H8O.xyz", ":5")
rmsd_matrix = torch.zeros((len(samples), len(samples)))
for i in range(len(samples)):
    for j in range(i + 1, len(samples)):
        sample_i = samples[i].copy()
        sample_j = samples[j].copy()
        sample_i.positions = sample_i.positions - sample_i.positions.mean(axis=0)
        sample_j.positions = sample_j.positions - sample_j.positions.mean(axis=0)
        minimize_rotation_and_translation(sample_i, sample_j)
        rmsd = ((sample_i.positions - sample_j.positions)**2).sum(-1).mean()**0.5
        rmsd_matrix[i, j] = rmsd
        rmsd_matrix[j, i] = rmsd
        
x = torch.stack([torch.tensor(sample.positions) for sample in samples]).float()
y = x.clone()  # Identity mapping for testing

# BATCHED NORMAL VERSION
N = x.shape[0]
M = y.shape[0]
n_atoms = x.shape[1]
x_pairs = x[:, None, :, :].expand(N, M, n_atoms, 3)
y_pairs = y[None, :, :, :].expand(N, M, n_atoms, 3)

x_pairs_flat = x_pairs.reshape(-1, n_atoms, 3)
y_pairs_flat = y_pairs.reshape(-1, n_atoms, 3)

y_aligned, R__ = kabsch_batched(x_pairs_flat, y_pairs_flat)
rmsd = torch.sqrt(((y_aligned - x_pairs_flat) ** 2).sum(dim=-1).mean(dim=-1))
rmsd_matrix_kabsch = rmsd.view(N, M)

# VMAP VERSION
kabsch_batched_ = torch.vmap(kabsch_single, in_dims=(0, 0))
R, x_aligned = kabsch_batched_(x_pairs_flat, y_pairs_flat)
rmsd = torch.sqrt(((x_aligned - y_pairs_flat) ** 2).sum(dim=-1).mean(dim=-1))
rmsd_matrix_kabsch_ = rmsd.view(N, M)

# SCATTER VERSION
rmsd_batched_scatter, _ = minimal_distance_scatter(x, y)

# batched code version

rmsd_batched_code, _ = minimal_distance(x, y)

print("RMSD matrix from ASE:", rmsd_matrix)
print("RMSD matrix from Kabsch batched:", rmsd_matrix_kabsch)
print("RMSD matrix from Kabsch single with vmap:", rmsd_matrix_kabsch_)
print("RMSD matrix from minimal_distance:", rmsd_batched_scatter)
print("RMSD matrix from minimal_distance (code version):", rmsd_batched_code)