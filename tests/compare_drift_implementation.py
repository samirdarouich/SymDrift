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

def kernel_version_1(x, y_pos, y_neg, temperature):
    N = x.shape[0]
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]
    device = x.device
    
    dist_pos = torch.cdist(x, y_pos) # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg) # [N, N_neg]

    # 2. Mask self-distances (when y_neg contains x)
    mask = torch.eye(N, device=device) * 1e6
    dist_neg = dist_neg + mask

    # 3. Compute logits
    logit_pos = -dist_pos / temperature  # (N, N_pos)
    logit_neg = -dist_neg / temperature  # (N, N_neg)

    # 4. Concat for normalization
    logit = torch.cat([logit_pos, logit_neg], dim=1)  # (N, N_pos + N_neg)

    # 5. Normalize along (BOTH) dimensions (key insight from paper)
    A_row = torch.softmax(logit, dim=-1)   # softmax over y (columns)
    A_col = torch.softmax(logit, dim=-2)   # softmax over x (rows)
    A = torch.sqrt(A_row * A_col)         # geometric mean

    # 6. Split back to pos and neg
    A_pos = A[:, :N_pos]  # (N, N_pos)
    A_neg = A[:, N_pos:]  # (N, N_neg)

    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # (N, N_pos)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # (N, N_neg)
    
    drift_pos = torch.mm(W_pos, y_pos)  # (N, D)
    drift_neg = torch.mm(W_neg, y_neg)  # (N, D)
    
    return W_pos, W_neg, drift_pos, drift_neg, dist_pos, dist_neg

def kernel_version_1_2(x, y_pos, y_neg, temperature):
    N = x.shape[0]
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]
    device = x.device
    
    diff_p = y_pos[None] - x[:, None]           # (B, B, N*2)
    diff_n = y_neg[None] - x[:, None]

    dist_pos = (diff_p ** 2).sum(dim=-1).sqrt()
    dist_neg = (diff_n ** 2).sum(dim=-1).sqrt()
    
    # 2. Mask self-distances (when y_neg contains x)
    mask = torch.eye(N, device=device) * 1e6
    dist_neg = dist_neg + mask

    # 3. Compute logits
    logit_pos = -dist_pos / temperature  # (N, N_pos)
    logit_neg = -dist_neg / temperature  # (N, N_neg)

    # 6. Split back to pos and neg
    W_pos = torch.softmax(logit_pos, dim=1)  # (N, N_pos)
    W_neg = torch.softmax(logit_neg, dim=1)  # (N, N_neg)
    
    drift_pos = (W_pos[..., None] * diff_p).sum(dim=1)
    drift_neg = (W_neg[..., None] * diff_n).sum(dim=1)
    
    return W_pos, W_neg, drift_pos, drift_neg, dist_pos, dist_neg

def kernel_version_2(x, y_pos, y_neg, temperature):

    diff_p = y_pos[None] - x[:, None]           # (B, B, N*2)
    diff_n = y_neg[None] - x[:, None]

    dist2_pos = (diff_p ** 2).sum(dim=-1).sqrt()
    dist2_neg = (diff_n ** 2).sum(dim=-1).sqrt()

    k_pos = torch.exp(-dist2_pos / temperature)
    k_neg = torch.exp(-dist2_neg / temperature)

    k_neg = k_neg.clone()
    k_neg.fill_diagonal_(0.0)

    w_pos = k_pos / (k_pos.sum(dim=1, keepdim=True) + 1e-8)
    w_neg = k_neg / (k_neg.sum(dim=1, keepdim=True) + 1e-8)
    
    drift_pos = (w_pos[..., None] * diff_p).sum(dim=1)
    drift_neg = (w_neg[..., None] * diff_n).sum(dim=1)
    
    return w_pos, w_neg, drift_pos, drift_neg, dist2_pos, dist2_neg

x = sample_prior(5).view(5, -1)
y_pos = sample_prior(5).view(5, -1)
y_neg = x.clone().view(5, -1)
temperature = 0.15

w_pos_1, w_neg_1, drift_pos_1, drift_neg_1, dist_pos_1, dist_neg_1 = kernel_version_1(x, y_pos, y_neg, temperature)
w_pos_1_2, w_neg_1_2, drift_pos_1_2, drift_neg_1_2, dist_pos_1_2, dist_neg_1_2 = kernel_version_1_2(x, y_pos, y_neg, temperature)
w_pos_2, w_neg_2, drift_pos_2, drift_neg_2, dist_pos_2, dist_neg_2 = kernel_version_2(x, y_pos, y_neg, temperature)

assert (dist_pos_1-dist_pos_1_2).max() < 1e-6, "something wrong with distance"
assert (dist_pos_1-dist_pos_2).max() < 1e-6, "something wrong with distance"
assert (dist_pos_1_2-dist_pos_2).max() < 1e-6, "something wrong with distance"

print("Drift pos version 1:", drift_pos_1)
print("Drift pos version 1.2:", drift_pos_1_2)
print("Drift pos version 2:", drift_pos_2)
print("\n--------------------------------\n")
print("Drift neg version 1:", drift_neg_1)
print("Drift neg version 1.2:", drift_neg_1_2)
print("Drift neg version 2:", drift_neg_2)
        