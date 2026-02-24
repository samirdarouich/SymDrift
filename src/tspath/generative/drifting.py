import torch

def get_x_y_pairs(x, y):
    """
    x: (N, n_atoms, 3), 
    y: (M, n_atoms, 3) 
    returns:
    x_flat, y_flat of shape (N*M, n_atoms, 3) for all pairs
    """
    N, n_atoms, _ = x.shape
    M = y.shape[0]
    x_pairs = x[:, None, :, :].expand(N, M, n_atoms, 3)
    y_pairs = y[None, :, :, :].expand(N, M, n_atoms, 3)
    x_flat = x_pairs.reshape(N * M, n_atoms, 3)
    y_flat = y_pairs.reshape(N * M, n_atoms, 3)
    return x_flat, y_flat

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

    # Reflection correction
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
    
def minimal_distance(x, y):
    """
    x: (N, n_atoms, 3)
    y: (M, n_atoms, 3)
    returns: (N, M) RMSD matrix
    """
    N, n_atoms, _ = x.shape
    M = y.shape[0]

    # Create all N x M pairs
    x_flat, y_flat = get_x_y_pairs(x, y)

    # Get aligned y for all pairs at once
    y_aligned, _ = kabsch_batched(x_flat, y_flat)
    
    # Compute directional difference for all pairs at once and reshape to (N, M, n_atoms, 3)
    diff_pos = (y_aligned - x_flat).view(N, M, n_atoms, 3)
    
    # Compute RMSD for all pairs at once (N, M)
    rmsd = torch.sqrt((diff_pos**2).sum(dim=(2, 3)) / n_atoms)

    return rmsd, diff_pos

def naive_distance(x,y):
    """
    x: (N, n_atoms, 3)
    y: (M, n_atoms, 3)
    returns: (N, M) RMSD matrix
    """
    N, n_atoms, _ = x.shape
    diff_pos = (y[None, :, :, :] - x[:, None, :, :]) # (N, M, n_atoms, 3)
    rmsd = torch.sqrt((diff_pos**2).sum(dim=(2, 3)) / n_atoms) # (N, M)
    return rmsd, diff_pos


class DriftingField(torch.nn.Module):
    
    def __init__(self, temperature=1.0, mask_self=True, normalize_over_x=False):
        super().__init__()
        self.temperature = temperature

        self.mask_self = mask_self
        self.normalize_over_x = normalize_over_x
    
    def forward(self, x, y_pos, y_neg, temperature=None):
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
        
        return V, drift_pos, drift_neg, diff_pos, diff_neg, dist_pos, dist_neg


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
        dist_pos, diff_pos = self.distance_fn(x_, y_pos_)
        dist_neg, diff_neg = self.distance_fn(x_, y_neg_) 
        
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
        
        # 8. compute V and reshape (N*n_atoms, 3)
        V = (drift_pos - drift_neg).view_as(x)
        
        return V, drift_pos, drift_neg, diff_pos, diff_neg, dist_pos, dist_neg