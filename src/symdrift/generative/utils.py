from typing import Optional, Tuple

import torch
from torch_scatter import scatter_mean

__all__ = [
    "batch_center_systems",
    "sample_noise",
    "sample_noise_like",
    "sample_noise_like_2d",
    "sample_isotropic_Gaussian",
]

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
