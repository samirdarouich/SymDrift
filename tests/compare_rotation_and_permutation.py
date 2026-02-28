import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch
from tspath.datasets import ToyMoleculeDataset
from tspath.generative.drifting import get_x_y_pairs
from tspath.utils import sample_noise_like_2d
import matplotlib.pyplot as plt
import torch
from tspath.datasets import ToyMoleculeDataset
from tspath.utils import get_x_y_pairs, hungarian_and_kabch_batched, kabsch_batched, brute_force_and_kabch_batched

import matplotlib.pyplot as plt

def plot_aligned_permuted(x, y):
    plt.scatter(x[:, 0], x[:, 1], alpha=0.5, color="red", label="Samples")
    plt.scatter(y[:, 0], y[:, 1], alpha=0.5, color="blue", label="References")
    for i, (px, py) in enumerate(x[:, :2]):
        plt.text(px.item(), py.item(), str(i+1), fontsize=10, ha="center", va="bottom")
        plt.text(y[i, 0].item(), y[i, 1].item(), str(i+1), fontsize=10, ha="center", va="top")
    plt.legend()
    plt.show()

def kabsch_numpy(p, q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.

    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert p.shape == q.shape, "Matrix dimensions must match"

    # Compute centroids
    assert np.allclose(np.mean(p, axis=0), np.zeros(3),atol=1e-5), "Centroid of P is not zero"
    assert np.allclose(np.mean(q, axis=0), np.zeros(3),atol=1e-5), "Centroid of Q is not zero"

    # Compute the covariance matrix
    H = np.dot(p.T, q)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Validate right-handed coordinate system
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0

    # Optimal rotation
    R = np.dot(Vt.T, U.T)

    # RMSD
    rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / p.shape[0])

    return R, rmsd

def compute_optimal_permutation(p_atoms, p_centroid, q_atoms, q_centroid):
    """
    Assigns atoms from structure p to structure q using the Hungarian algorithm.
    Parameters
    ----------
    p_atoms : array
        atomic numbers of trial structure
    p_centroid : array
        centered positions of trial structure
    q_atoms : array
        atomic numbers of reference structure
    q_centroid : array
        centered positions of reference structure
    """
    assert p_atoms.shape == q_atoms.shape, "Matrix dimensions must match"

    # Compute centroids
    assert np.allclose(np.mean(p_centroid, axis=0), np.zeros(3),atol=1e-5), "Centroid of P is not zero"
    assert np.allclose(np.mean(q_centroid, axis=0), np.zeros(3),atol=1e-5), "Centroid of Q is not zero"
    
    # generate full view from q shape to fill in atom view on the fly
    perm_inds = np.zeros(len(p_atoms), dtype=np.int64)

    # Find unique atoms
    species = np.unique(p_atoms)

    for specie in species:
        p_atom_inds = np.where(p_atoms == specie)[0]
        q_atom_inds = np.where(q_atoms == specie)[0]
        A = q_centroid[q_atom_inds]
        B = p_centroid[p_atom_inds]

        # Perform Hungarian analysis on distance matrix between atoms of 1st
        # structure and trial structure
        distances = cdist(A, B, "euclidean")
        
        # Solve the optimal assignment problem (Hungarian algorithm) and return the 
        # indices of how to permute B to match A.
        _a_inds, b_inds = linear_sum_assignment(distances)

        perm_inds[q_atom_inds] = p_atom_inds[b_inds]
    return perm_inds

def find_permutation_and_alignment(p_atoms, p_positions, q_atoms, q_positions):
    """
    Assigns atoms from structure p to structure q using the Hungarian algorithm and 
    align them using Kabsch.
    
    Parameters
    ----------
    p_atoms : array
        atomic numbers of trial structure
    p_positions : array
        positions of trial structure
    q_atoms : array
        atomic numbers of reference structure
    q_positions : array
        positions of reference structure
    """
    
    # center both structures to their centroids
    p_centroid = p_positions.mean(axis=0)
    q_centroid = q_positions.mean(axis=0)
    p_centered = p_positions - p_centroid
    q_centered = q_positions - q_centroid
    
    # compute optimal permutation
    perm_inds = compute_optimal_permutation(
        p_atoms, p_centered, q_atoms, q_centered
    )
    
    # compute optimal alignment of permuted structure
    R, rmsd = kabsch_numpy(p_centered[perm_inds], q_centered)
    p_positions = np.dot(p_centered[perm_inds], R.T) + q_centroid
    return p_positions, perm_inds
    
def sample_eot_plan(x0, x1, atomic_numbers):
    """ Compute EOT plan for batch of molecules. Reorder and permute x1 to match x0.

    Parameters
    ----------
    x0 : array
        trial structures
    x1 : array
        reference structures
    atomic_numbers : array
        atomic numbers of each atom
    batch : array
        batch indices for each atom

    Returns
    -------
    x1_permuted : array
        permuted reference structures
    """
    assert x0.shape == x1.shape, "X0 and X1 dimensions must match"
    x1_permuted = x1.clone()
    B = x0.shape[0]
    
    for b in range(B):
        x0_b = x0[b].cpu().numpy()
        x1_b = x1[b].cpu().numpy()
        atomic_numbers_b = atomic_numbers[b].cpu().numpy()
        
        x1_b_permuted, perm_inds = find_permutation_and_alignment(
            p_atoms=atomic_numbers_b, p_positions=x1_b, 
            q_atoms=atomic_numbers_b, q_positions=x0_b
        )
        x1_b_permuted = torch.from_numpy(x1_b_permuted).to(x1)
        x1_permuted[b] = x1_b_permuted

    return x1_permuted

def minimal_distance_permuted(x, y, atomic_numbers):
    """ Compute EOT plan for batch of molecules. Reorder and permute y to match x.

    Parameters
    ----------
    x : array
        trial structures (N, n_atoms, d)
    y : array
        reference structures (M, n_atoms, 3)
    atomic_numbers : array
        atomic numbers of each atom in target structure, used to only permute within
        same atomic number (M*n_atoms,)

    Returns
    -------
    rmsd: array
        RMSD matrix (N, M)
    diff_pos: array
        aligned and permuted directional difference y-x of shape (N, M, n_atoms, 3)
    """
    assert x.shape[1] == y.shape[1], "X and Y must have same number of atoms"
    N, n_atoms, d = x.shape
    M = y.shape[0]
    
    # Create all N x M pairs
    x_flat, y_flat, atomic_numbers_flat = get_x_y_pairs(x, y, atomic_numbers)
    
    # This is only doable on CPU for now since it involves scipy linear_sum_assignment,
    # but we can speed up by parallelizing over batches.
    B, n_atoms, _ = x_flat.shape
    y_aligned_and_permuted = y_flat.clone()
    for b in range(B):
        x_b = x_flat[b].cpu().numpy()
        y_b = y_flat[b].cpu().numpy()
        atomic_numbers_b = atomic_numbers_flat[b].cpu().numpy()
        
        y_b_permuted, perm_inds = find_permutation_and_alignment(
            p_atoms=atomic_numbers_b, p_positions=y_b, 
            q_atoms=atomic_numbers_b, q_positions=x_b
        )
        y_b_permuted = torch.from_numpy(y_b_permuted).to(y_flat)
        y_aligned_and_permuted[b] = y_b_permuted

    # Compute directional difference for all pairs at once and reshape to (N, M, n_atoms, 3)
    diff_pos = (y_aligned_and_permuted - x_flat).view(N, M, n_atoms, 3)
    
    # Compute RMSD for all pairs at once (N, M)
    rmsd = torch.sqrt((diff_pos**2).sum(dim=(2, 3)) / n_atoms)

    return rmsd, diff_pos

dataset = ToyMoleculeDataset("ccc", n_samples=5, T=800, seed=42, augment_with_permutations=True)
atomic_numbers = torch.stack([data.x for data in dataset], dim=0)
positions = torch.stack([data.pos for data in dataset], dim=0)
z = torch.randn((*positions.shape[:-1],2))
z -= z.mean(dim=1, keepdim=True)
z = torch.cat([z, torch.zeros_like(z[:,:,:1])], dim=-1)

# this is using scipy's linear_sum_assignment and kabsch on CPU, so it's a bit slow but should work for small batches and number of atoms
x, y = get_x_y_pairs(z, positions)
y_aligned = sample_eot_plan(x, y, atomic_numbers.repeat(positions.shape[0], 1))


# Augment with rotations and test if kabsch can align the samples to the reference
augment_with_rotations = True
augment_with_permutations = False
dataset = ToyMoleculeDataset(
    name="ccc",
    n_samples=1000,
    T=0,
    seed=42,
    r0=2.0,
    theta0=120.0,
    augment_with_rotations=augment_with_rotations,
    augment_with_permutations=augment_with_permutations,
)
# only take 2d
pos_dataset = torch.stack([data.pos[:,:2] for data in dataset], dim=0)


x = pos_dataset[0].clone().unsqueeze(0)
y = pos_dataset[1:].clone()

x_flat, y_flat = get_x_y_pairs(x, y)
y_aligned, _ = kabsch_batched(x_flat, y_flat)

if (x_flat - y_aligned).abs().max() > 1e-5:
    print("!!! Kabsch alignment failed to align y to x !!!")
else:
    print("Kabsch alignment succeeded in aligning y to x")

# Augment with permutations and test if Hungarian + kabsch can align the samples to the reference
augment_with_rotations = False
augment_with_permutations = True
dataset = ToyMoleculeDataset(
    name="ccc",
    n_samples=5,
    T=0,
    seed=42,
    r0=2.0,
    theta0=120.0,
    augment_with_rotations=augment_with_rotations,
    augment_with_permutations=augment_with_permutations,
)
# only take 2d
pos_dataset = torch.stack([data.pos[:,:2] for data in dataset], dim=0)


x = pos_dataset[0].clone().unsqueeze(0)
y = pos_dataset[1:].clone()

x_flat, y_flat = get_x_y_pairs(x, y)
y_aligned, _, converged = hungarian_and_kabch_batched(x_flat, y_flat)

if (x_flat - y_aligned).abs().max() > 1e-5:
    print("!!! Hungarian and Kabsch alignment failed to align y to x !!!")
else:
    print("Hungarian and Kabsch alignment succeeded in aligning y to x")


augment_with_rotations = True
augment_with_permutations = True
dataset = ToyMoleculeDataset(
    name="ccc",
    n_samples=5,
    T=0,
    seed=42,
    r0=2.0,
    theta0=120.0,
    augment_with_rotations=augment_with_rotations,
    augment_with_permutations=augment_with_permutations,
)
# only take 2d
pos_dataset = torch.stack([data.pos[:,:2] for data in dataset], dim=0)


x = pos_dataset[0].clone().unsqueeze(0)
y = pos_dataset[1:].clone()

x_flat, y_flat = get_x_y_pairs(x, y)
y_aligned, _, converged = hungarian_and_kabch_batched(x_flat, y_flat, verbose=False)

if (x_flat - y_aligned).abs().max() > 1e-5:
    print("!!! Hungarian and Kabsch alignment failed to align y to x !!!")
else:
    print("Hungarian and Kabsch alignment succeeded in aligning y to x")
    
    
y_aligned, _ = brute_force_and_kabch_batched(x_flat, y_flat)

if (x_flat - y_aligned).abs().max() > 1e-5:
    print("!!! Brute force and Kabsch alignment failed to align y to x !!!")
else:
    print("Brute force and Kabsch alignment succeeded in aligning y to x")
    