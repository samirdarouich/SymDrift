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
import itertools

__all__ = [
    "print_config", 
    "batch_center_systems", 
    "get_composition",
    "get_elementwise_permutations",
    "get_brute_force_permutations",
    "get_x_y_pairs",
    "batch_inputs_to_atoms",
    "sample_noise",
    "sample_noise_like",
    "sample_noise_like_2d",
    "sample_isotropic_Gaussian",
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

def get_composition(atomic_numbers):
    B, N = atomic_numbers.shape
    # (B, N, n_types)
    n_types = int(atomic_numbers.max().item() + 1)
    one_hot = torch.nn.functional.one_hot(atomic_numbers, num_classes=n_types)
    # (B, n_types)
    comp = one_hot.sum(dim=1)
    return comp

def get_elementwise_permutations(atom_types):
    """
    atom_types: (n,)
    Returns tensor of shape (P, n)
    """
    device = atom_types.device

    # preserve first appearance order to get consistent permutations
    unique_elements = []
    for a in atom_types.tolist():
        if a not in unique_elements:
            unique_elements.append(a)

    element_indices = [
        torch.where(atom_types == elem)[0].tolist()
        for elem in unique_elements
    ]

    element_perms = [
        list(itertools.permutations(indices))
        for indices in element_indices
    ]

    all_perms = []

    for prod in itertools.product(*element_perms):
        perm = list(itertools.chain(*prod))
        all_perms.append(perm)

    return torch.tensor(all_perms, device=device)

def get_brute_force_permutations(x, y, atomic_numbers=None):
    """
    Get all permutations of y and flatten into batch dimension for parallel processing.
    x, y: (B, n_atoms, d)

    Returns:
        x_flat, y_flat, P: (P*B, n_atoms, d) where P is the number of permutations (n!)
    """
    device = x.device
    B, n, d = x.shape
    
    # --- 1) Generate all permutations ---
    if atomic_numbers is not None:
        comp = get_composition(atomic_numbers.long())
        all_equal = torch.all(comp == comp[0], dim=1).all()
        assert all_equal, "Different composition across batch not supported in this simple version"
        perms = get_elementwise_permutations(atomic_numbers[0]) 
    else:
        perms = torch.tensor(list(itertools.permutations(range(n))), device=device)

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
    
    atomic_numbers_b = None
    if atomic_numbers is not None:
        assert atomic_numbers.shape == (M*n_atoms,), f"Atomic numbers should have shape (M*n_atoms,) not {atomic_numbers.shape}"
        atomic_numbers_b = atomic_numbers.view(M, n_atoms).repeat(N, 1)  # (N*M, n_atoms)
    
    return x_flat, y_flat, atomic_numbers_b


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

def sample_noise_like_2d(pos: torch.Tensor, batch: torch.Tensor):
    """
    Sample 2d Gaussian noise and add zero z-component. 
    Center the noise to have zero center of geometry.
    """
    z = torch.randn(pos.shape[0], 2, device=pos.device)
    z = batch_center_systems(z, batch)  # zero center of geometry
    z = torch.cat([z, torch.zeros(z.shape[0], 1, device=z.device)], dim=1)
    return z

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