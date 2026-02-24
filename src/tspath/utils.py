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

__all__ = [
    "print_config", 
    "batch_center_systems", 
    "get_shortest_path_fast_batched_x_1",
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
        f":gear: Running with the following config:", style=style, guide_style=style
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

def get_shortest_path_fast_batched_x_1(x_0_N_3, x_1_N_3, batch):
    # x_0_N_3, x_1_N_3 are tensors of shape (N, 3)
    # batch is a 1D tensor of length N with group indices.
    device = x_0_N_3.device
    Nm = int(batch.max().item() + 1)  # number of molecules in the batch

    # Compute group centroids via index_add.
    centers_x0_Nm_3 = scatter_mean(x_0_N_3, batch, dim=0)
    centers_x1_Nm_3 = scatter_mean(x_1_N_3, batch, dim=0)
    
    # Center the points.
    x0_centered_N_3 = x_0_N_3 - centers_x0_Nm_3[batch]
    x1_centered_N_3 = x_1_N_3 - centers_x1_Nm_3[batch]

    # For each point, compute the outer product: shape (N, 3, 3)
    prod_N_3_3 = x1_centered_N_3.unsqueeze(2) * x0_centered_N_3.unsqueeze(1)
    # Sum the outer products per group to form (B, 3, 3) covariance matrices.
    M_Nm_3_3 = torch.zeros((Nm, 3, 3), device=device)
    M_Nm_3_3 = M_Nm_3_3.index_add(0, batch, prod_N_3_3)

    # Compute the batched SVD
    U_Nm_3_3, S_Nm_3, Vt_Nm_3_3 = torch.linalg.svd(M_Nm_3_3)

    # Reflection correction per group.
    det_Nm = torch.det(torch.bmm(U_Nm_3_3, Vt_Nm_3_3))
    # (3, 3) -> (1, 3, 3) -> (Nm, 3, 3). That is, repeat the (3,3)-identity matrix Nm times.
    D_Nm_3_3 = torch.eye(3, device=device).unsqueeze(0).repeat(Nm, 1, 1)
    # Change the 2,2 element of the identity matrix to -1 if det < 0.
    D_Nm_3_3[det_Nm < 0, 2, 2] = -1
    # Apply the reflection correction.
    R_opt_Nm_3_3 = torch.bmm(U_Nm_3_3, torch.bmm(D_Nm_3_3, Vt_Nm_3_3))

    # Apply the optimal rotation:
    # For each point i (belonging to group j), we set:
    #   x1_aligned[i] = (x1[i] - centers_x1[j]) @ R_opt[j] + centers_x0[j]
    x_1_rotated_N_3 = torch.bmm(
        x1_centered_N_3.unsqueeze(1), # (N,3) -> (N,1,3)
        R_opt_Nm_3_3[batch]   # (Nm,3,3) -> (N,3,3)
    ).squeeze(1) # (N,1,3) -> (N,3)
    x1_aligned_N_3 = x_1_rotated_N_3 + centers_x0_Nm_3[batch]

    return x1_aligned_N_3

def get_rmsd_batch(xi, xj, batch):
    diff = (xi - xj)**2
    rmsd = scatter_mean(diff.sum(-1), batch, dim=0).sqrt()
    return rmsd

def get_rmsd_batch_aligned(xi, xj, batch):
    xj_aligned = get_shortest_path_fast_batched_x_1(xi, xj, batch)
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