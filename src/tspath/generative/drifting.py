import torch
from tspath.utils import kabsch_batched, hungarian_and_kabch_batched, brute_force_and_kabch_batched, get_x_y_pairs
from functools import partial

def naive_distance(x,y, **kwargs):
    """
    x: array
        (N, n_atoms, d)
    y: array
        (M, n_atoms, d)
    
    Returns
    -------
    rmsd: array
        RMSD matrix (N, M)
    diff_pos: array
        directional difference y-x of shape (N, M, n_atoms, d)
    """
    N, n_atoms, d = x.shape
    diff_pos = (y[None, :, :, :] - x[:, None, :, :]) # (N, M, n_atoms, 3)
    rmsd = torch.sqrt((diff_pos**2).sum(dim=(2, 3)) / n_atoms) # (N, M)
    return rmsd, diff_pos

def minimal_distance(x, y, **kwargs):
    """
    x: array
        (N, n_atoms, d)
    y: array
        (M, n_atoms, d)
    
    Returns
    -------
    rmsd: array
        RMSD matrix (N, M)
    diff_pos: array
        aligned directional difference y-x of shape (N, M, n_atoms, d)
    """
    assert x.shape[1] == y.shape[1], "X and Y must have same number of atoms"   
    N, n_atoms, d = x.shape
    M = y.shape[0]

    # Create all N x M pairs
    x_flat, y_flat = get_x_y_pairs(x, y)

    # Get aligned y for all pairs at once
    y_aligned, _ = kabsch_batched(x_flat, y_flat)
    
    # Compute directional difference for all pairs at once and reshape to (N, M, n_atoms, d)
    diff_pos = (y_aligned - x_flat).view(N, M, n_atoms, d)
    
    # Compute RMSD for all pairs at once (N, M)
    rmsd = torch.sqrt((diff_pos**2).sum(dim=(2, 3)) / n_atoms)

    return rmsd, diff_pos

def minimal_distance_permuted(x, y, atomic_numbers, brute_force_permutations=False):
    """ Compute EOT plan for batch of molecules. Reorder and permute y to match x.

    Parameters
    ----------
    x : array
        trial structures (N, n_atoms, d)
    y : array
        reference structures (M, n_atoms, d)
    atomic_numbers : array
        atomic numbers of each atom in target structure, used to only permute within
        same atomic number (M*n_atoms,)

    Returns
    -------
    rmsd: array
        RMSD matrix (N, M)
    diff_pos: array
        aligned and permuted directional difference y-x of shape (N, M, n_atoms, d)
    """
    assert x.shape[1] == y.shape[1], "X and Y must have same number of atoms"
    N, n_atoms, d = x.shape
    M = y.shape[0]
    
    # Create all N x M pairs
    x_flat, y_flat, atomic_numbers_flat = get_x_y_pairs(x, y, atomic_numbers)
    
    # Get aligned and permuted y for all pairs (use Hungarian algorithm to permute y 
    # and Kabsch to align)
    if brute_force_permutations:
        y_aligned_and_permuted, _ = brute_force_and_kabch_batched(
            x_flat, y_flat, atomic_numbers_flat
        )
    else:
        y_aligned_and_permuted, *_ = hungarian_and_kabch_batched(
            x_flat, y_flat, atomic_numbers=atomic_numbers_flat, max_iter=3
        )
    
    # Compute directional difference for all pairs at once and reshape to (N, M, n_atoms, d)
    diff_pos = (y_aligned_and_permuted - x_flat).view(N, M, n_atoms, d)
    
    # Compute RMSD for all pairs at once (N, M)
    rmsd = torch.sqrt((diff_pos**2).sum(dim=(2, 3)) / n_atoms)

    return rmsd, diff_pos


class DriftingField(torch.nn.Module):
    
    def __init__(self, temperature=1.0, mask_self=True, normalize_over_x=False, **kwargs):
        super().__init__()
        self.temperature = temperature

        self.mask_self = mask_self
        self.normalize_over_x = normalize_over_x
    
    def forward(self, x, y_pos, y_neg, temperature=None, **kwargs):
        """
        x: [N, D]
        y_pos: [N_pos, D]
        y_neg: [N_neg, D]
        temperature: float (optional, if provided overrides self.temperature)
        """
        if temperature is not None:
            self.temperature = temperature
            
        N = x.shape[0]
        N_pos = y_pos.shape[0]
        N_neg = y_neg.shape[0]
        device = x.device
        
        # 1. Compute pairwise L2 distances
        diff_pos = y_pos[None, :, :] - x[:, None, :]  # (N, N_pos, D)
        diff_neg = y_neg[None, :, :] - x[:, None, :]  # (N, N_neg, D)
        dist_pos = torch.sqrt((diff_pos**2).sum(dim=2))  # (N, N_pos)
        dist_neg = torch.sqrt((diff_neg**2).sum(dim=2))  # (N, N_neg)

        # 2. Mask self-distances (when y_neg contains x)
        if self.mask_self and N == N_neg:
            mask = torch.eye(N, device=device) * 1e6
            dist_neg = dist_neg + mask

        # 3. Compute logits
        logit_pos = -dist_pos / self.temperature  # (N, N_pos)
        logit_neg = -dist_neg / self.temperature  # (N, N_neg)

        # Compute kernel (normalize over y and optionally over x)
        w_pos = torch.softmax(logit_pos, dim=1)  # softmax over y (columns)
        w_neg = torch.softmax(logit_neg, dim=1)  # softmax over y (columns)
        if self.normalize_over_x:
            w_pos_ = torch.softmax(logit_pos, dim=0)  # softmax over x (rows)
            w_neg_ = torch.softmax(logit_neg, dim=0)  # softmax over x (rows)
            w_pos = torch.sqrt(w_pos * w_pos_)  # geometric mean
            w_neg = torch.sqrt(w_neg * w_neg_)  # geometric mean
 
        # 7. Compute drift as weighted average of differences (x is dim=0, y is dim=1).
        # Aim is compute the drift for each molecule in x as a weighted average of the
        # differences to all molecules in y.
        drift_pos = (w_pos[..., None] * diff_pos).sum(dim=1)
        drift_neg = (w_neg[..., None] * diff_neg).sum(dim=1)
        V = drift_pos - drift_neg
        
        return V, drift_pos, drift_neg, diff_pos, diff_neg, w_pos, w_neg

class EquivariantDriftingField(torch.nn.Module):
    def __init__(self, temperature=1.0, mask_self=True, normalize_over_x=False, aligned=True, permuted=True, brute_force_permutations=False):
        super().__init__()
        self.temperature = temperature
        self.mask_self = mask_self
        self.normalize_over_x = normalize_over_x
        self.aligned = aligned
        self.permuted = permuted
        self.brute_force_permutations = brute_force_permutations
        # Initialize distance function based on alignment and permutation settings
        self.get_distance_fn()

    def get_distance_fn(self):
        if self.aligned and self.permuted:
            self.distance_fn = partial(minimal_distance_permuted, brute_force_permutations=self.brute_force_permutations)
        elif self.aligned and not self.permuted:
            self.distance_fn = minimal_distance
        elif not self.aligned and not self.permuted:
            self.distance_fn = naive_distance
        else:
            raise ValueError("Permuted but not aligned doesn't make sense since permutation is only meaningful with alignment. Please set permuted=False if aligned=False.")
        
    def forward(self, x, y_pos, y_neg, n_atoms, atomic_numbers=None, temperature=None, aligned=None, permuted=None, brute_force_permutations=None):
        """
        x: (N*n_atoms, d)
        y_pos: (M*n_atoms, d)
        y_neg: (M*n_atoms, d)
        n_atoms: int
        atomic_numbers: (M*n_atoms,) atomic numbers of each atom in target structure,
            used to only permute within same atomic number (M*n_atoms,)
        temperature: float (optional, if provided overrides self.temperature)
        aligned: bool (optional, if provided overrides self.aligned and updates distance function)
        permuted: bool (optional, if provided overrides self.permuted and updates distance function)
        brute_force_permutations: bool (optional, if provided overrides self.brute_force_permutations and updates distance function)
        returns: (N*n_atoms, d) drifting field for each molecule in x
        """
        _, d = x.shape
        assert y_pos.shape[1] == d, f"Expected y_pos to have {d} dimensions, got {y_pos.shape[1]}"
        
        if temperature is not None:
            self.temperature = temperature
        if aligned is not None:
            self.aligned = aligned
            if permuted is not None:
                self.permuted = permuted
            if brute_force_permutations is not None:
                self.brute_force_permutations = brute_force_permutations
            self.get_distance_fn()
        
        # 0. Reshape to (N, n_atoms, d) and (N_pos/N_neg, n_atoms, d)
        x_ = x.view(-1, n_atoms, d)  # (N, n_atoms, d)
        y_pos_ = y_pos.view(-1, n_atoms, d)  # (M, n_atoms, d)
        y_neg_ = y_neg.view(-1, n_atoms, d)  # (M, n_atoms, d)
        
        N = x_.shape[0]
        N_pos = y_pos_.shape[0]
        N_neg = y_neg_.shape[0]
        device = x.device
        
        # 1. Compute alignment-aware pairwise L2 distances and returns (N, N_pos/N_neg) RMSD
        # matrix and aligned difference y-x of shape (N, N_pos/N_neg, n_atoms, d)
        dist_pos, diff_pos = self.distance_fn(x_, y_pos_, atomic_numbers=atomic_numbers)
        dist_neg, diff_neg = self.distance_fn(x_, y_neg_, atomic_numbers=atomic_numbers) 
        
        # 2. Mask self-distances (when y_neg contains x)
        if self.mask_self and N == N_neg:
            mask = torch.eye(N, device=device) * 1e6
            dist_neg = dist_neg + mask

        # 3. Compute logits
        logit_pos = -dist_pos / self.temperature  # (N, N_pos)
        logit_neg = -dist_neg / self.temperature  # (N, N_neg)

        # Compute kernel (normalize over y and optionally over x)
        w_pos = torch.softmax(logit_pos, dim=1)  # softmax over y (columns)
        w_neg = torch.softmax(logit_neg, dim=1)  # softmax over y (columns)
        if self.normalize_over_x:
            w_pos_ = torch.softmax(logit_pos, dim=0)  # softmax over x (rows)
            w_neg_ = torch.softmax(logit_neg, dim=0)  # softmax over x (rows)
            w_pos = torch.sqrt(w_pos * w_pos_)  # geometric mean
            w_neg = torch.sqrt(w_neg * w_neg_)  # geometric mean
 
        # 7. Compute drift as weighted average of differences (x is dim=0, y is dim=1).
        # Aim is compute the drift for each molecule in x as a weighted average of the
        # differences to all molecules in y.
        drift_pos = (w_pos[:, :, None, None] * diff_pos).sum(dim=1)
        drift_neg = (w_neg[:, :, None, None] * diff_neg).sum(dim=1)
        
        # 8. compute V and reshape (N*n_atoms, d)
        V = (drift_pos - drift_neg).view_as(x)
        
        return V, drift_pos, drift_neg, diff_pos, diff_neg, w_pos, w_neg