import torch
import time
from torch_linear_assignment import batch_linear_assignment
from tspath.utils import kabsch_batched

def permute_by_atom_type(x, y, atom_types):
    """
    x, y: (B, N, d)
    atom_types: (B, N)  integer labels
    """
    B, N, d = x.shape
    device = x.device

    y_permuted = torch.zeros_like(y)

    for b in range(B):
        types = atom_types[b]
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

    y_permuted_aligned, R = kabsch_batched(x, y_permuted)
    return y_permuted_aligned, R

def permute_by_atom_type_parallel(x, y, atom_types):
    B, N, d = x.shape
    device = x.device

    y_permuted = torch.zeros_like(y)

    unique_types = torch.unique(atom_types)

    for t in unique_types:
        # mask: (B, N)
        mask = (atom_types == t)

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

    y_permuted_aligned, R = kabsch_batched(x, y_permuted)
    return y_permuted_aligned, R

# -----------------------------
# Generate synthetic test data
# -----------------------------
def generate_test_data(B=32, N=12, d=3, n_types=3, device="cpu", seed=42):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.randn(B, N, d, device=device, generator=generator)
    y = torch.randn(B, N, d, device=device, generator=generator)

    # random atom types
    atom_types = torch.randint(0, n_types, (N,), device=device, generator=generator)
    atom_types = atom_types.repeat(B, 1)  # (B, N)

    return x, y, atom_types


# -----------------------------
# Benchmark function
# -----------------------------
def benchmark(fn, n_runs=10, device="cpu"):
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()

    for _ in range(n_runs):
        x, y, atom_types = generate_test_data(device=device)
        fn(x, y, atom_types)

    torch.cuda.synchronize() if device == "cuda" else None
    end = time.time()

    return (end - start) / n_runs


# -----------------------------
# Run comparison
# -----------------------------
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on: {device}")

    # warmup
    x, y, atom_types = generate_test_data(B=64*64, N=20, device=device)
    permute_by_atom_type(x, y, atom_types)
    permute_by_atom_type_parallel(x, y, atom_types)

    # actual timing
    t1 = benchmark(permute_by_atom_type, device=device)
    t2 = benchmark(permute_by_atom_type_parallel, device=device)

    print(f"Original version avg time:  {t1:.6f} sec")
    print(f"Parallel version avg time:  {t2:.6f} sec")
    print(f"Speedup: {t1 / t2:.2f}x")