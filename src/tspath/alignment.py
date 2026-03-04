import torch
from torch_linear_assignment import batch_linear_assignment
from tspath.utils import get_composition, get_brute_force_permutations
from torch_scatter import scatter_mean

__all__ = [
    "permute_by_atom_type", 
    "permute_by_atom_type_parallel",
    "kabsch_batched_scatter",
    "kabsch_batched",
    "hungarian_batched",
    "hungarian_and_kabch_batched",
    "brute_force_and_kabch_batched",
    "get_rmsd_batched_scatter",
    "get_rmsd_batched",
]

def permute_by_atom_type(x, y, atomic_numbers):
    """
    x, y: (B, N, d)
    atomic_numbers: (B, N)  integer labels
    """
    B, N, d = x.shape
    device = x.device

    y_permuted = torch.zeros_like(y)

    for b in range(B):
        types = atomic_numbers[b]
        unique_types = torch.unique(types)

        perm_indices = torch.empty(N, dtype=torch.long, device=device)

        for t in unique_types:
            mask = (types == t)

            idx = mask.nonzero(as_tuple=False).squeeze(-1)

            x_t = x[b, idx]           # (n_t, d)
            y_t = y[b, idx]           # (n_t, d)

            cost = torch.cdist(x_t, y_t)
            assignment = batch_linear_assignment(cost.unsqueeze(0)).squeeze(0)  # Hungarian (n_t,)
            perm_indices[idx] = idx[assignment]

        y_permuted[b] = y[b, perm_indices]
    return y_permuted

def permute_by_atom_type_parallel(x, y, atomic_numbers):
    """
    x, y: (B, N, d)
    atomic_numbers: (B, N)  integer labels
    ! This assumes that the number of atoms of each type is the same across the batch.
    """
    B, N, d = x.shape
    device = x.device

    y_permuted = torch.zeros_like(y)

    unique_types = torch.unique(atomic_numbers)

    for t in unique_types:
        # mask: (B, N)
        mask = (atomic_numbers == t)

        # number of atoms of this type (assume constant per batch)
        n_t = mask.sum(dim=1)

        # if variable per batch → more complex handling required
        assert torch.all(n_t == n_t[0]), "Different counts per batch not supported in this simple version"
        n_t = n_t[0].item()

        # Gather atoms of type t
        idx = mask.nonzero(as_tuple=False)
        # idx: (B*n_t, 2) → (batch_idx, atom_idx)

        x_t = torch.zeros(B, n_t, d, device=device, dtype=x.dtype)
        y_t = torch.zeros(B, n_t, d, device=device, dtype=y.dtype)

        for b in range(B):
            atom_idx = idx[idx[:,0] == b][:,1]
            x_t[b] = x[b, atom_idx]
            y_t[b] = y[b, atom_idx]

        # Compute batched cost
        cost = torch.cdist(x_t, y_t)  # (B, n_t, n_t)

        # Solve Hungarian in batch
        assignment = batch_linear_assignment(cost)  # (B, n_t)

        # Scatter back
        for b in range(B):
            atom_idx = idx[idx[:,0] == b][:,1]
            y_permuted[b, atom_idx] = y_t[b, assignment[b]]
    return y_permuted



def kabsch_batched_scatter(x_0_N_3, x_1_N_3, batch):
    """
    Perform Kabsch alignment of two sets of points (x_0 and x_1) in a batched manner.
    Each point set is grouped by the 'batch' tensor, which indicates which points belong to the group
    x_1 is rotated to best align with x_0 for each group, and the aligned points are returned.
    """
    # x_0_N_3, x_1_N_3 are tensors of shape (N, 3)
    # batch is a 1D tensor of length N with group indices.
    device = x_0_N_3.device
    Nm = int(batch.max().item() + 1)

    # Compute counts and centers
    counts = torch.bincount(batch, minlength=Nm).to(x_0_N_3.dtype).clamp(min=1)
    
    # Compute group centroids
    centers_x0_Nm_3 = torch.zeros((Nm, 3), dtype=x_0_N_3.dtype, device=device)
    centers_x1_Nm_3 = torch.zeros((Nm, 3), dtype=x_1_N_3.dtype, device=device)
    centers_x0_Nm_3.index_add_(0, batch, x_0_N_3)
    centers_x1_Nm_3.index_add_(0, batch, x_1_N_3)
    centers_x0_Nm_3 = centers_x0_Nm_3 / counts.unsqueeze(1)
    centers_x1_Nm_3 = centers_x1_Nm_3 / counts.unsqueeze(1)

    # Center the points
    x0_centered_N_3 = x_0_N_3 - centers_x0_Nm_3[batch]
    x1_centered_N_3 = x_1_N_3 - centers_x1_Nm_3[batch]

    # Covariance Matrix construction
    prod_N_3_3 = x1_centered_N_3.unsqueeze(2) * x0_centered_N_3.unsqueeze(1)
    M_Nm_3_3 = torch.zeros((Nm, 3, 3), dtype=prod_N_3_3.dtype, device=device)
    M_Nm_3_3.index_add_(0, batch, prod_N_3_3)

    # Batched SVD
    U_Nm_3_3, _, Vt_Nm_3_3 = torch.linalg.svd(M_Nm_3_3)

    # 1. Compute determinant of the uncorrected rotation matrix UV^T
    # use the property: det(UV^T) = det(U) * det(V^T)
    R_temp = torch.bmm(U_Nm_3_3, Vt_Nm_3_3)
    det_Nm = torch.det(R_temp)
    
    # 2. Reflection Correction:
    # Instead of constructing a diagonal matrix D and doing R = U @ D @ Vt,
    # flip the sign of the last row of Vt where det < 0.
    mask_neg = det_Nm < 0
    if mask_neg.any():
        # Clone to avoid in-place modification issues if gradients are required later
        Vt_Nm_3_3 = Vt_Nm_3_3.clone() 
        Vt_Nm_3_3[mask_neg, 2, :] *= -1

    # 3. Final Rotation
    R_opt_Nm_3_3 = torch.bmm(U_Nm_3_3, Vt_Nm_3_3)

    # Apply rotation
    # (N, 1, 3) @ (N, 3, 3) -> (N, 1, 3)
    x_1_rotated_N_3 = torch.bmm(x1_centered_N_3.unsqueeze(1), R_opt_Nm_3_3[batch]).squeeze(1)

    return x_1_rotated_N_3 + centers_x0_Nm_3[batch]


def kabsch_batched(X, Y):
    """
    align y to x
    X, Y: (B, N, d)
    """

    centroid_X = X.mean(dim=1, keepdim=True)
    centroid_Y = Y.mean(dim=1, keepdim=True)

    Xc = X - centroid_X
    Yc = Y - centroid_Y

    H = torch.matmul(Yc.transpose(1, 2), Xc)  # (B,d,d)

    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(1, 2)

    R = torch.matmul(V, U.transpose(1, 2))

    # Reflection correction
    det = torch.det(R)
    mask = det < 0

    if mask.any():
        V[mask, :, -1] *= -1
        R = torch.matmul(V, U.transpose(1, 2))

    # Apply rotation
    Y_rot = torch.matmul(Yc, R.transpose(1, 2))

    Y_aligned = Y_rot + centroid_X

    return Y_aligned, R

def hungarian_batched(x, y, atomic_numbers=None):
    B, n_atoms, d = x.shape
    batch_indices = torch.arange(B)[:, None]
    
    # assume that all atoms are of the same species if atomic_numbers is None
    if atomic_numbers is None:
        atomic_numbers = torch.zeros((B, n_atoms), dtype=torch.long, device=x.device)
    
    # Get composition of each system in the batch and check if all systems have the same
    # composition.
    comp = get_composition(atomic_numbers.long())
    all_equal = torch.all(comp == comp[0], dim=1).all()
    
    if all_equal and (comp[0]>0).sum().item() == 1:
        # or if only one type of atom is present.
        cost = torch.cdist(x, y)
        assignment = batch_linear_assignment(cost)
        y_permuted = y[batch_indices, assignment]
    else:
        # Check if composition is the same across the batch, then do parallelized 
        # version if so, otherwise do looped version
        if all_equal:
            y_permuted = permute_by_atom_type_parallel(x, y, atomic_numbers)
        else:
            y_permuted = permute_by_atom_type(x, y, atomic_numbers)
    return y_permuted

def hungarian_and_kabch_batched(x, y, atomic_numbers=None, max_iter=3, tol=1e-2, verbose=False):
    """ Perform permutations and aligment of y to x using Hungarian and Kabsch algorithm
    in an iterative manner. 
    
    1) Compute optimal permutations according to current cost plan (cdist(x,y_iter))
    2) Align permuted y_iter to x.
    3) Start again from 1 until convergence achieved (mean rmsd change is below
        thresholdplan or maximum number of iterations are achieved)

    Parameters
    ----------
    x : array
        trial structures (B, n_atoms, d)
    y : array
        reference structures (B, n_atoms, d)
    atomic_numbers : array
        atomic numbers of each atom in target structure, used to only permute within
        same atomic number (B, n_atoms)
    max_iter : int
        maximum number of iterations to perform
    tol : float
        convergence threshold for mean change in RMSD between iterations
    verbose : bool
        whether to print convergence information at each iteration

    Returns
    -------
    y_permuted_aligned: array
        aligned and permuted reference structures of shape (B, n_atoms, d)
    """
    B, n_atoms, d = x.shape
    y_aligned = y.clone()
    rmsds = [get_rmsd_batched(x, y_aligned).max().item()]
    if verbose:
        print(f"Initial RMSD: {rmsds[-1]:.6f}")
    for i in range(max_iter):
        # find permutation that minimizes RMSD to x (if specified respect atomic numbers)
        y_permuted = hungarian_batched(x, y_aligned, atomic_numbers)
        y_new, _ = kabsch_batched(x, y_permuted)
        rmsd = get_rmsd_batched(x, y_new).max().item()
        y_aligned = y_new
        rmsds.append(rmsd)
        delta_rmsd = abs(rmsds[-1] - rmsds[-2])
        if verbose:
            print(f"Iteration {i}: RMSD: {rmsds[-1]:.6f}, delta RMSD = {delta_rmsd:.6f}")
        if delta_rmsd < tol or rmsd < tol:
            if verbose:
                print(f"Converged after {i} iterations with delta RMSD: {rmsds[-1]:.6f}")
            break
    return y_aligned

def brute_force_and_kabch_batched(x, y, atomic_numbers=None):
    """ Perform permutations and aligment of y to x using brute force permutation and 
    Kabsch algorithm.
    
    1) Get all possible permutations of atoms in y ()
    2) For each permutation, align to x using Kabsch and compute RMSD
    3) Select permutation with lowest RMSD.

    Parameters
    ----------
    x : array
        trial structures (B, n_atoms, d)
    y : array
        reference structures (B, n_atoms, d)
    atomic_numbers : array
        atomic numbers of each atom in target structure, used to only permute within
        same atomic number (B, n_atoms)
    max_iter : int
        maximum number of iterations to perform
    tol : float
        convergence threshold for mean change in RMSD between iterations
    verbose : bool
        whether to print convergence information at each iteration

    Returns
    -------
    y_permuted_aligned: array
        aligned and permuted reference structures of shape (B, n_atoms, d)
    """
    B, n_atoms, d = x.shape

    # --- 1) Get all possible permutations ---
    # (P*B, n_atoms, d)
    x_flat, y_flat, perms = get_brute_force_permutations(x, y, atomic_numbers)  
    P = perms.shape[0]
    
    # --- 2) Align all with Kabsch ---
    y_aligned_flat, _ = kabsch_batched(x_flat, y_flat)

    # --- 3) Compute RMSD ---
    rmsd = get_rmsd_batched(x_flat, y_aligned_flat)  # (P*B,)
    rmsd = rmsd.view(P, B)    
    
    # --- 4) Pick best permutation per batch ---
    best_idx = rmsd.argmin(dim=0)                    # (B,)
    best_perm = perms[best_idx]                      # (B,n)

    # --- 5) Gather best aligned structure ---
    y_aligned = y_aligned_flat.view(P, B, n_atoms, d)
    y_best_aligned = y_aligned[best_idx, torch.arange(B)]
    
    return y_best_aligned

def get_rmsd_batched_scatter(xi, xj, batch, align=False):
    if align:
        xj = kabsch_batched_scatter(xi, xj, batch)
    diff = (xi - xj)**2
    rmsd = scatter_mean(diff.sum(-1), batch, dim=0).sqrt()
    return rmsd

def get_rmsd_batched(x, y, align=False):
    """
    Compute RMSD between two batches of structures x and y, where x and y are of shape
    (B, N, d). The RMSD is computed for each pair of structures in the batch.
    RMSD(x,y) = sqrt(1/N * sum((x-y)^2))
    """
    assert x.shape == y.shape, "X and Y must have same shape"
    if align:
        y, _ = kabsch_batched(x, y)
    B, n_atoms, d = x.shape
    rmsd = (((x - y)**2).sum(dim=(-2,-1))/n_atoms).sqrt()
    return rmsd