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
    k=8.0,              # softer than realistic
    T=300.0,
    kB=8.617333e-5,
    seed=None
):
    """
    Returns:
        positions: (n_samples, 2, 3)
        bond_lengths: (n_samples,)
        energies: (n_samples,)
    """
    if seed is not None:
        torch.random.manual_seed(seed)
        
    # Bond length std from Boltzmann
    sigma = math.sqrt(kB * T / k)

    # Sample bond lengths
    r = torch.normal(mean=r0, std=sigma, size=(n_samples,))

    # Sample random 2D angles
    theta = 2 * math.pi * torch.rand(n_samples)

    # Create positions
    positions = torch.zeros(n_samples, 2, 3)

    # Second carbon in 2D plane
    positions[:, 1, 0] = r * torch.cos(theta)
    positions[:, 1, 1] = r * torch.sin(theta)

    return positions, r
    
    
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
        k = kwargs.get("k", 8.0)
        T = kwargs.get("T", 300.0)
        seed = kwargs.get("seed", None)
        positions, bond_lengths = generate_cc_harmonic_2d(n_samples, r0, k, T, seed=seed)
        return positions.numpy(), bond_lengths.numpy()
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
    def __init__(self, name="cc", n_samples=1000, r0=1.54, k=8.0, T=500.0, **kwargs):
        self.positions, self.bond_lengths = get_dataset(
            name, n_samples=n_samples, r0=r0, k=k, T=T, **kwargs
        )
        
        logger.info(f"Loaded dataset '{name}' with {n_samples} samples.")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        data = Data(
            x = torch.tensor([6, 6], dtype=torch.float),  # Carbon atomic numbers
            pos=torch.tensor(self.positions[idx], dtype=torch.float),
            bond_length=torch.tensor([self.bond_lengths[idx]], dtype=torch.float),
            num_atoms=torch.tensor(2, dtype=torch.long)
        )
        data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)  # Center the molecule
        return data