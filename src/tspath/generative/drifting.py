import torch
from tspath.utils import get_shortest_path_fast_batched_x_1, get_rmsd_batch

def minimal_distance(x, y):
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

def naive_distance(x,y):
    """
    x: (N, n_atoms, 3)
    y: (M, n_atoms, 3)
    returns: (N, M) RMSD matrix
    """
    diff_pos_unaligned = (y[None, :, :, :] - x[:, None, :, :])
    rmsd = torch.sqrt((diff_pos_unaligned**2).sum(dim=(2, 3)) / x.shape[2])
    return rmsd, diff_pos_unaligned


class DriftingField(torch.nn.Module):
    
    def __init__(self, temperature=1.0, mask_self=True, normalize_over_x=False):
        super().__init__()
        self.temperature = temperature

        self.mask_self = mask_self
        self.normalize_over_x = normalize_over_x
    
    def drifting_field(self, x, y_pos, y_neg, temperature=None):
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
        dist_pos = torch.cdist(x, y_pos) # [N, N_pos]
        dist_neg = torch.cdist(x, y_neg) # [N, N_neg]
        diff_pos = y_pos[None, :, :] - x[:, None, :]  # (N, N_pos, D)
        diff_neg = y_neg[None, :, :] - x[:, None, :]  # (N, N_neg, D)

        # 2. Mask self-distances (when y_neg contains x)
        if self.mask_self and N == N_neg:
            mask = torch.eye(N, device=device) * 1e6
            dist_neg = dist_neg + mask

        # 3. Compute logits
        logit_pos = -dist_pos / self.temperature  # (N, N_pos)
        logit_neg = -dist_neg / self.temperature  # (N, N_neg)
        
        # 4. Concat for normalization
        logit = torch.cat([logit_pos, logit_neg], dim=1)  # (N, N_pos + N_neg)
        
        # 5. Normalize along (BOTH) dimensions (key insight from paper)
        A_row = torch.softmax(logit, dim=1)   # softmax over y (columns)
        if self.normalize_over_x:
            A_col = torch.softmax(logit, dim=0)   # softmax over x (rows)
            A = torch.sqrt(A_row * A_col)         # geometric mean
        else:
            A = A_row
        
        # 6. Split back to pos and neg
        A_pos = A[:, :N_pos]  # (N, N_pos)
        A_neg = A[:, N_pos:]  # (N, N_neg)
        
        # 7. Compute weights (cross-weighting from paper eq. 17)
        W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # (N, N_pos)
        W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # (N, N_neg)
        
        # compute drift as weighted average of differences (x is dim=0, y is dim=1).
        # Aim is compute the drift for each point in x as a weighted average of the
        # differences to all points in y.
        drift_pos = torch.mm(W_pos,y_pos)
        drift_neg = torch.mm(W_neg,y_neg)
        V = drift_pos - drift_neg
        return V, drift_pos, drift_neg, diff_pos, diff_neg


class EquivariantDriftingField(torch.nn.Module):
    def __init__(self, temperature=1.0, mask_self=True, normalize_over_x=False, aligned=True):
        super().__init__()
        self.temperature = temperature

        self.mask_self = mask_self
        self.normalize_over_x = normalize_over_x
        self.aligned = aligned
        
        self.distance_fn = minimal_distance if aligned else naive_distance

    def forward(self, x, y_pos, y_neg, n_atoms, temperature=None, aligned=None):
        """
        x: (N*n_atoms, 3)
        y_pos: (M*n_atoms, 3)
        y_neg: (M*n_atoms, 3)
        n_atoms: int
        temperature: float (optional, if provided overrides self.temperature)
        returns: (N*n_atoms, 3) drifting field for each molecule in x
        """
        if temperature is not None:
            self.temperature = temperature
        if aligned is not None:
            self.aligned = aligned
            self.distance_fn = minimal_distance if aligned else naive_distance
            
        # 0. Reshape to (N, n_atoms, 3) and (N_pos/N_neg, n_atoms, 3)
        x_ = x.view(-1, n_atoms, 3)  # (N, n_atoms, 3)
        y_pos_ = y_pos.view(-1, n_atoms, 3)  # (M, n_atoms, 3)
        y_neg_ = y_neg.view(-1, n_atoms, 3)  # (M, n_atoms, 3)
        
        N = x_.shape[0]
        N_pos = y_pos_.shape[0]
        N_neg = y_neg_.shape[0]
        device = x.device
        
        # 1. Compute alignment-aware pairwise L2 distances and returns (N, N_pos/N_neg) RMSD
        # matrix and aligned difference y-x of shape (N, N_pos/N_neg, n_atoms, 3)
        dist_pos, diff_pos_aligned = self.distance_fn(x_, y_pos_)
        dist_neg, diff_neg_aligned = self.distance_fn(x_, y_neg_) 
        
        # 2. Mask self-distances (when y_neg contains x)
        if self.mask_self and N == N_neg:
            mask = torch.eye(N, device=device) * 1e6
            dist_neg = dist_neg + mask

        # 3. Compute logits
        logit_pos = -dist_pos / self.temperature  # (N, N_pos)
        logit_neg = -dist_neg / self.temperature  # (N, N_neg)

        # 4. Concat for normalization
        logit = torch.cat([logit_pos, logit_neg], dim=1)  # (N, N_pos + N_neg)

        # 5. Normalize along (BOTH) dimensions (key insight from paper)
        A_row = torch.softmax(logit, dim=1)  # softmax over y (columns)
        if self.normalize_over_x:
            A_col = torch.softmax(logit, dim=0)  # softmax over x (rows)
            A = torch.sqrt(A_row * A_col)  # geometric mean
        else:
            A = A_row

        # 6. Split back to pos and neg
        A_pos = A[:, :N_pos]  # (N, N_pos)
        A_neg = A[:, N_pos:]  # (N, N_neg)

        # 7. Compute drift as weighted average of differences (x is dim=0, y is dim=1).
        # Aim is compute the drift for each molecule in x as a weighted average of the
        # differences to all molecules in y.
        drift_pos = (A_pos[:, :, None, None] * diff_pos_aligned).sum(dim=1)
        drift_neg = (A_neg[:, :, None, None] * diff_neg_aligned).sum(dim=1)
        
        # 8. compute V and reshape (N*n_atoms, 3)
        V = (drift_pos - drift_neg).view_as(x)
        
        return V, drift_pos, drift_neg, diff_pos_aligned, diff_neg_aligned