import torch
from ase import Atoms

from typing import Union, Dict, Sequence, Optional, Tuple

import rich
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from rich.syntax import Syntax
from rich.tree import Tree
from torch_scatter import scatter_mean
from torch_linear_assignment import batch_linear_assignment
from itertools import permutations

__all__ = [
    "print_config", 
    "batch_center_systems", 
    "kabsch_batched_scatter",
    "kabsch_batched",
    "hungarian_and_kabch_batched",
    "brute_force_and_kabch_batched",
    "get_brute_force_permutations",
    "get_x_y_pairs",
    "get_rmsd_batch",
    "get_rmsd_batch_aligned",
    "batch_inputs_to_atoms",
]


def empty(*args, **kwargs):
    pass

def todict(config: Union[DictConfig, Dict]):
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))
    return config_dict

@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "run",
        "globals",
        "data",
        "model",
        "task",
        "trainer",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Config.
        fields (Sequence[str], optional): Determines which main fields from config will be printed
        and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = Tree(
        ":gear: Running with the following config:", style=style, guide_style=style
    )

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml"))

    rich.print(tree)
    
def batch_center_systems(systems: torch.Tensor, batch: torch.Tensor, dim: int = 0):
    """
    center batch of systems moleculewise to have zero center of geometry

    Args:
        systems (torch.tensor): batch of systems (molecules)
        batch (torch.tensor): the system id for each atom in the batch
        dim (int): dimension to scatter over
    """

    # Compute mean position per molecule
    mean = scatter_mean(systems, batch, dim=dim)

    # broadcast mean to the same shape as systems
    mean = mean.movedim(dim, 0)[batch].movedim(0, dim)

    return systems - mean

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

def get_rmsd_batched(x, y):
    """
    Compute RMSD between two batches of structures x and y, where x and y are of shape
    (B, N, d). The RMSD is computed for each pair of structures in the batch.
    RMSD(x,y) = sqrt(1/N * sum((x-y)^2))
    """
    assert x.shape == y.shape, "X and Y must have same shape"
    B, n_atoms, d = x.shape
    rmsd = (((x - y)**2).sum(dim=(-2,-1))/n_atoms).sqrt()
    return rmsd

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
    batch_indices = torch.arange(B)[:, None].to(x.device)  # (B, 1) for indexing
    y_aligned = y.clone()
    converged = False
    rmsds = [get_rmsd_batched(x, y_aligned).max().item()]
    if verbose:
        print(f"Initial RMSD: {rmsds[-1]:.6f}")
    for i in range(max_iter):
        cost = torch.cdist(x, y_aligned)
        assignment = batch_linear_assignment(cost)
        y_permuted = y_aligned[batch_indices, assignment]
        y_new, _ = kabsch_batched(x, y_permuted)
        rmsd = get_rmsd_batched(x, y_new).max().item()
        y_aligned = y_new
        rmsds.append(rmsd)
        delta_rmsd = abs(rmsds[-1] - rmsds[-2])
        if verbose:
            print(f"Iteration {i}: RMSD: {rmsds[-1]:.6f}, delta RMSD = {delta_rmsd:.6f}")
        if delta_rmsd < tol:
            if verbose:
                print(f"Converged after {i} iterations with delta RMSD: {rmsds[-1]:.6f}")
            converged = True
            break
    return y_aligned, assignment, converged

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
    x_flat, y_flat, perms = get_brute_force_permutations(x, y)  # (P*B, n_atoms, d)
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
    
    return y_best_aligned, best_perm

def get_brute_force_permutations(x, y):
    """
    Get all permutations of y and flatten into batch dimension for parallel processing.
    x, y: (B, n_atoms, d)

    Returns:
        x_flat, y_flat, P: (P*B, n_atoms, d) where P is the number of permutations (n!)
    """
    device = x.device
    B, n, d = x.shape
    
    # --- 1) Generate all permutations ---
    perms = torch.tensor(list(permutations(range(n))), device=device)
    P = perms.shape[0]  # n!

    # --- 2) Apply all permutations in parallel ---
    # Expand y to (P, B, n, d)
    y_exp = y.unsqueeze(0).expand(P, B, n, d)

    # Expand perms to (P, B, n)
    perms_exp = perms.unsqueeze(1).expand(P, B, n)

    # Apply permutation indexing to get y_perm of shape (P, B, n, d)
    # So the first entry in y_perm corresponds to the first permutation in perms, and so on.
    y_perm = torch.gather(
        y_exp,
        2,  # gather along atom dimension
        perms_exp.unsqueeze(-1).expand(P, B, n, d)
    )

    # (P,B,n,d)
    x_rep = x.unsqueeze(0).expand(P, B, n, d)     

    # --- 3) Flatten permutations into batch dimension ---
    x_flat = x_rep.reshape(P*B, n, d)
    y_flat = y_perm.reshape(P*B, n, d)
    return x_flat, y_flat, perms

def get_x_y_pairs(x, y, atomic_numbers=None):
    """
    x: (N, n_atoms, d), 
    y: (M, n_atoms, d)
    atomic_numbers: (n_atoms,) atomic numbers of each atom in target structure. Only 
    permute within same atomic number if provided.
    
    returns:
    x_flat, y_flat of shape (N*M, n_atoms, d) for all pairs
    """
    assert x.shape[1] == y.shape[1], "X and Y must have same number of atoms"
    N, n_atoms, d = x.shape
    M = y.shape[0]
    
    # Create all N x M pairs and flatten to (N*M, n_atoms, d)
    x_pairs = x[:, None, :, :].expand(N, M, n_atoms, d)
    y_pairs = y[None, :, :, :].expand(N, M, n_atoms, d)
    x_flat = x_pairs.reshape(N * M, n_atoms, d)
    y_flat = y_pairs.reshape(N * M, n_atoms, d)
    
    if atomic_numbers is not None:
        assert atomic_numbers.shape == (M*n_atoms,), "Atomic numbers should have shape (M*n_atoms,)"
        atomic_numbers_b = atomic_numbers.view(M, n_atoms).repeat(N, 1)  # (N*M, n_atoms)
        return x_flat, y_flat, atomic_numbers_b
    
    return x_flat, y_flat

def get_rmsd_batch(xi, xj, batch):
    diff = (xi - xj)**2
    rmsd = scatter_mean(diff.sum(-1), batch, dim=0).sqrt()
    return rmsd

def get_rmsd_batch_aligned(xi, xj, batch):
    xj_aligned = kabsch_batched_scatter(xi, xj, batch)
    rmsd = get_rmsd_batch(xi, xj_aligned, batch)
    return rmsd


def batch_inputs_to_atoms(batch, pos_key='pos', info_keys=[]):
    """
    Converts a batch of inputs to a list of ASE Atoms objects.

    Args:
        batch: The input batch in PyTorch geometric format.
        pos_key (str): The key in the batch that contains the positions.
    Returns:
        List[Atoms]: The list of ASE Atoms objects.
    """
    atoms_list = []

    for m in batch.batch.unique():
        mask = batch.batch == m
        R = batch[pos_key][mask].detach().cpu().numpy()
        Z = batch.x[mask].detach().cpu().numpy()
        info = {}
        for key in info_keys:
            if hasattr(batch, key):
                info[key] = batch[key][m].item()
        atoms = Atoms(positions=R, numbers=Z, info=info)
        atoms_list.append(atoms)
    return atoms_list


def sample_noise(
    shape: Tuple,
    batch: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Sample Gaussian noise based on input shape.
    Project to the zero center of geometry if invariant.

    Args:
        shape: shape of the noise.
        batch: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    """
    # sample noise
    noise = torch.randn(shape, device=device, dtype=dtype)

    # The invariance trick: project noise to the zero center of geometry.
    # system-wise center of geometry
    if batch is not None:
        noise = batch_center_systems(noise, batch, dim=-2)  # type: ignore
    # global center of geometry if one system passed.
    else:
        noise -= noise.mean(-2).unsqueeze(-2)

    return noise

def sample_noise_like(
    x: torch.Tensor,
    batch: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Sample Gaussian noise based on input x.

    Args:
        x: input tensor, e.g. to infer shape.
        batch: same as ``batch.batch`` to map each row of x to its system.
                Set to None if one system or no invariance needed.
    """
    return sample_noise(x.shape, batch, device=x.device, dtype=x.dtype)

def sample_isotropic_Gaussian(
    mean: torch.Tensor,
    std: torch.Tensor,
    batch: Optional[torch.Tensor],
    noise: Optional[torch.Tensor] = None,
):
    """
    Use the reparametrization trick to Sample from iso Gaussian distribution
    with given mean and std.

    Args:
        mean: mean of the Gaussian distribution.
        std: standard deviation of the Gaussian distribution.
        batch: same as ``proporties.batch`` to map each row of x to its system.
                Set to None if one system or no invariance needed.
        noise: the Gaussian noise. If None, a new noise is sampled.
    """
    # sample noise if not given.
    if noise is None:
        noise = sample_noise_like(mean, batch)

    # sample using the Gaussian reparametrization trick.
    sample = mean + std * noise

    return sample, noise

def sample_noise_like_2d(pos: torch.Tensor, batch: torch.Tensor):
    """
    Sample 2d Gaussian noise and add zero z-component. 
    Center the noise to have zero center of geometry.
    """
    z = torch.randn(pos.shape[0], 2, device=pos.device)
    z = batch_center_systems(z, batch)  # zero center of geometry
    z = torch.cat([z, torch.zeros(z.shape[0], 1, device=z.device)], dim=1)
    return z