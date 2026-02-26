from collections import defaultdict
import math
import numpy as np
import torch
from sklearn.datasets import make_circles
import logging
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

def make_spiral(n_samples, noise=0.01, start=0.0, stop=2 * math.pi, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(start, stop, n_samples)
    t += np.random.randn(n_samples) * 0.02

    r = 0.5 + 1.5 * (t - t.min()) / (t.max() - t.min() + 1e-8)
    x = r * np.cos(t) + np.random.randn(n_samples) * noise
    y = r * np.sin(t) + np.random.randn(n_samples) * noise

    return np.stack([x, y], axis=1).astype(np.float32)


def make_multi_spiral(n_samples_per_arm, n_arms=2, noise=0.01, seed=None):
    if seed is not None:
        np.random.seed(seed)
    points = []
    for i in range(n_arms):
        start = i * (2 * math.pi / n_arms)
        stop = start + 2 * math.pi
        arm = make_spiral(
            n_samples_per_arm,
            noise=noise,
            start=start,
            stop=stop,
        )
        points.append(arm)
    return np.concatenate(points, axis=0)


def make_checkerboard(n, noise=0.05, grid=4, seed=None):
    """Create a checkerboard pattern dataset."""
    if seed is not None:
        np.random.seed(seed)
    pts = []
    cell = 4.0 / grid
    for i in range(grid):
        for j in range(grid):
            if (i + j) % 2 == 0:
                pts.append(
                    np.random.uniform(
                        [-2 + i * cell, -2 + j * cell],
                        [-2 + (i + 1) * cell, -2 + (j + 1) * cell],
                        (n // (grid**2 // 2), 2),
                    )
                )
    c = np.vstack(pts)
    np.random.shuffle(c)
    return (c[:n] + np.random.randn(n, 2) * noise).astype(np.float32)


def make_pinwheel(n, n_arms=5, noise=0.05, seed=None):
    """Create a pinwheel pattern dataset."""
    if seed is not None:
        np.random.seed(seed)
    per_arm = n // n_arms
    pts = []
    for k in range(n_arms):
        r = np.linspace(0.1, 2.0, per_arm) + np.random.randn(per_arm) * 0.02
        theta = np.linspace(0, 0.8 * np.pi, per_arm) + k * 2 * np.pi / n_arms
        pts.append(
            np.column_stack(
                [
                    r * np.cos(theta) + np.random.randn(per_arm) * noise,
                    r * np.sin(theta) + np.random.randn(per_arm) * noise,
                ]
            )
        )
    c = np.vstack(pts)
    np.random.shuffle(c)
    return c[:n].astype(np.float32)


def make_rings(n, noise=0.03, seed=None):
    """Create concentric rings dataset."""
    X, _ = make_circles(n, noise=noise, factor=0.5, random_state=seed)
    return (X * 2).astype(np.float32)


def make_swissroll2d(n, noise=0.05, seed=None):
    """Create a 2D Swiss roll dataset."""
    if seed is not None:
        np.random.seed(seed)
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
    return (
        np.column_stack([t * np.cos(t) / 6, t * np.sin(t) / 6]).astype(np.float32)
        + np.random.randn(n, 2).astype(np.float32) * noise
    )

def generate_cc_harmonic_2d(
    n_samples=1000,
    r0=1.54,
    bond_k=8.0,
    T=300.0,
    augment_with_rotations=False,
    seed=None
):
    """
    Returns:
        positions: (n_samples, 2, 3)
        bond_lengths: (n_samples,)
        energies: (n_samples,)
    """
    kB=8.617333e-5  # eV/K
    if seed is not None:
        torch.random.manual_seed(seed)
        
    # Bond length std from Boltzmann
    sigma = math.sqrt(kB * T / bond_k)

    # Sample bond lengths
    r = torch.normal(mean=r0, std=sigma, size=(n_samples,))

    # Sample random 2D angles (rotate if wanted)
    if augment_with_rotations:
        theta = 2 * math.pi * torch.rand(n_samples)
    else:
        theta = 2 * math.pi * torch.ones(n_samples)

    # Create positions
    positions = torch.zeros(n_samples, 2, 3)

    # Second carbon in 2D plane 
    positions[:, 1, 0] = r * torch.cos(theta)
    positions[:, 1, 1] = r * torch.sin(theta)

    return positions

def generate_ccc_harmonic_2d(
    n_samples=1000,
    r0=1.54,
    theta0=120.0,
    bond_k=8.0,
    angle_k=1.5,
    T=300.0,
    augment_with_rotations=False,
    augment_with_permutations=False,
    seed=None,
):
    """
    Returns:
        positions: (n_samples, 3, 3)
        bond_lengths: (n_samples,)
        energies: (n_samples,)
    """
    kB = 8.617333e-5  # eV/K
    if seed is not None:
        torch.random.manual_seed(seed)

    r1 = torch.normal(r0, math.sqrt(kB * T / bond_k), size=(n_samples,))
    r2 = torch.normal(r0, math.sqrt(kB * T / bond_k), size=(n_samples,))

    # Sample angle and convert to radians
    theta = torch.normal(
        theta0 * math.pi / 180.0, math.sqrt(kB * T / angle_k), size=(n_samples,)
    )

    # Geometry C2 is middle atom and the angle is between C1-C2-C3
    positions = torch.zeros(n_samples, 3, 3)
    positions[:, 0, 0] = -r1
    positions[:, 2, 0] = -r2 * torch.cos(theta)
    positions[:, 2, 1] = r2 * torch.sin(theta)
    
    # Apply random rotations to each sample (sample unfiform angle between 0 and 2pi)
    if augment_with_rotations:
        angles = torch.rand(n_samples) * 2 * math.pi
        cos_a, sin_a = angles.cos(), angles.sin()
        R = torch.stack(
            [
                torch.stack([cos_a, -sin_a], dim=1),  # row 0
                torch.stack([sin_a, cos_a], dim=1),  # row 1
            ],
            dim=1,
        )  # (n_samples, 2, 2)
        positions[:, :, :2] = torch.matmul(positions[:, :, :2], R.transpose(1, 2))

    # Apply random permutations of the atoms
    if augment_with_permutations:
        perms = torch.stack([torch.randperm(3) for _ in range(n_samples)])
        batch_indices = torch.arange(n_samples)[:, None]
        positions = positions[batch_indices, perms]
    
    return positions
    
def get_dataset(dataset_name="spiral", n_samples=10000, **kwargs):
    """
    Get a dataset by name.

    Args:
        dataset_name: one of ["spiral", "checkerboard", "pinwheel", "rings", "swissroll", "cc"]
        n_samples: number of samples to generate
        **kwargs: additional arguments for the dataset function

    Returns:
        numpy array of shape (n_samples, 2)
    """
    if dataset_name == "spiral":
        n_arms = kwargs.get("n_arms", 2)
        noise = kwargs.get("noise", 0.001)
        seed = kwargs.get("seed", None)
        return make_multi_spiral(n_samples // n_arms, n_arms=n_arms, noise=noise, seed=seed)
    elif dataset_name == "checkerboard":
        noise = kwargs.get("noise", 0.05)
        grid = kwargs.get("grid", 4)
        seed = kwargs.get("seed", None)
        return make_checkerboard(n_samples, noise=noise, grid=grid, seed=seed)
    elif dataset_name == "pinwheel":
        n_arms = kwargs.get("n_arms", 5)
        noise = kwargs.get("noise", 0.05)
        seed = kwargs.get("seed", None)
        return make_pinwheel(n_samples, n_arms=n_arms, noise=noise, seed=seed)
    elif dataset_name == "rings":
        noise = kwargs.get("noise", 0.03)
        seed = kwargs.get("seed", None)
        return make_rings(n_samples, noise=noise, seed=seed)
    elif dataset_name == "swissroll":
        noise = kwargs.get("noise", 0.05)
        seed = kwargs.get("seed", None)
        return make_swissroll2d(n_samples, noise=noise, seed=seed)
    elif dataset_name == "cc":
        r0 = kwargs.get("r0", 1.54)
        bond_k = kwargs.get("bond_k", 8.0)
        T = kwargs.get("T", 300.0)
        augment_with_rotations = kwargs.get("augment_with_rotations", False)
        seed = kwargs.get("seed", None)
        positions = generate_cc_harmonic_2d(
            n_samples, r0, bond_k, T, augment_with_rotations=augment_with_rotations, 
            seed=seed
        )
        return positions.numpy()
    elif dataset_name == "ccc":
        r0 = kwargs.get("r0", 1.54)
        bond_k = kwargs.get("bond_k", 8.0)
        theta0 = kwargs.get("theta0", 120.0)
        angle_k = kwargs.get("angle_k", 1.5)
        T = kwargs.get("T", 300.0)
        augment_with_rotations = kwargs.get("augment_with_rotations", False)
        augment_with_permutations = kwargs.get("augment_with_permutations", False)
        seed = kwargs.get("seed", None)
        positions = generate_ccc_harmonic_2d(
            n_samples, r0, theta0, bond_k, angle_k, T, 
            augment_with_rotations=augment_with_rotations, 
            augment_with_permutations=augment_with_permutations,
            seed=seed
        )
        return positions.numpy()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, name="spiral", n_samples=10000, **kwargs):
        self.data = get_dataset(name, n_samples=n_samples, **kwargs)
        
        logger.info(f"Loaded dataset '{name}' with {n_samples} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)
    

class ToyMoleculeDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        name="cc", 
        n_samples=1000, 
        T=500.0, 
        augment_with_rotations=True, 
        augment_with_permutations=True,
        **kwargs
    ):
        self.positions = get_dataset(
            name, n_samples=n_samples, T=T, 
            augment_with_rotations=augment_with_rotations,
            augment_with_permutations=augment_with_permutations,
            **kwargs
        )
        logger.info(f"Loaded dataset '{name}' with {n_samples} samples.")
        if augment_with_rotations:
            logger.info("Augmenting dataset with random rotations.")
        if augment_with_permutations:
            logger.info("Augmenting dataset with random permutations.")

        self.comp_to_indices = defaultdict(list)
        for idx in range(len(self)):
            self.comp_to_indices[self[idx].formula.item()].append(idx)
        self.compositions = sorted(list(self.comp_to_indices.keys()))
        
    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        pos = torch.tensor(self.positions[idx], dtype=torch.float)
        x = torch.tensor([6]*pos.shape[0], dtype=torch.float)  # Carbon atomic numbers
        num_atoms = torch.tensor(pos.shape[0], dtype=torch.long)
        data = Data(
            x=x, pos=pos, num_atoms=num_atoms, formula=torch.tensor(0, dtype=torch.long)
        )
        data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)  # Center the molecule
        return data