"""Implementation of the Gaussian moment descriptor to encode local atomic environment from Zaverkin et al. (https://doi.org/10.1021/acs.jctc.0c00347)"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import einops
import numpy as np
from torch_scatter import scatter_add

GM_DIM = {3: 94, 4: 198, 5: 360, 6: 593, 7: 910}


def uniform_range(minval: float, maxval: float):
    """
    Builds an initializer that returns real uniformly-distributed random arrays
    in a specified range [minval, maxval].

    Args:
        minval (float): Minimum value of the uniform distribution.
        maxval (float): Maximum value of the uniform distribution.

    Returns:
        Callable: A function that, when called with a shape, returns a tensor initialized
        with uniform values in the specified range.
    """

    def init(shape):
        """
        Initializes a tensor with uniform values in the range [minval, maxval].

        Args:
            shape (tuple): Shape of the tensor to be initialized.

        Returns:
            torch.Tensor: Tensor filled with uniform random values in the specified range.
        """
        return torch.empty(shape).uniform_(minval, maxval)

    return init

def tril_2d_indices(n_radial: int) -> Tensor:
    """
    Generates 2D lower-triangular indices up to the given radial dimension.

    The function returns indices where the row index `i` is always less than or
    equal to the column index `j`. This is used to efficiently extract or process
    pairwise combinations from a matrix.

    Args:
        n_radial (int):
          The size of the radial dimension, defining the range of indices.

    Returns:
        torch.Tensor:
          A tensor of shape (num_indices, 2) containing 2D lower-triangular indices
          where each row is a pair (i, j) with i <= j.
    """
    tril_idxs = []
    for i in range(n_radial):
        tril_idxs.append([i, i])
        for j in range(i + 1, n_radial):
            tril_idxs.append([i, j])
    tril_idxs = torch.tensor(tril_idxs)
    return tril_idxs


def tril_3d_indices(n_radial: int) -> Tensor:
    """
    Generates 3D lower-triangular indices up to the given radial dimension.

    The function returns indices where the first index `i` is less than or equal to
    the second index `j`, and the second index `j` is less than or equal to the third
    index `k`. This is useful for extracting or processing triplet combinations from a
    tensor.

    Args:
        n_radial (int):
          The size of the radial dimension, defining the range of indices.

    Returns:
        torch.Tensor:
          A tensor of shape (num_indices, 3) containing 3D lower-triangular indices
          where each row is a triplet (i, j, k) with i <= j <= k.
    """
    tril_idxs = []
    for i in range(n_radial):
        tril_idxs.append([i, i, i])
        for j in range(n_radial):
            if j != i:
                tril_idxs.append([i, j, j])
        for j in range(i + 1, n_radial):
            for k in range(j + 1, n_radial):
                tril_idxs.append([i, j, k])
    tril_idxs = torch.tensor(tril_idxs)
    return tril_idxs


def geometric_moments(
    radial_function: Tensor, dn: Tensor, idx_i: Tensor,
):
    """
    Compute geometric moments based on radial functions and distance vectors.

    This function calculates the geometric moments (zero, first, second, and third)
    of a set of radial functions with respect to distance vectors between neighbors. The
    moments are computed by summing the contributions to each moment across all
    neighbors and then aggregating them for each atom.

    Args:
        radial_function (torch.Tensor):
          A tensor of shape `(n_neighbors, n_radial)` representing the radial functions
          for each neighbor.
        dn (torch.Tensor):
          A tensor of shape `(n_neighbors, 3)` representing the normalized distance
          vectors between the central atom and its neighbors.
        idx_i (torch.Tensor):
          A tensor of shape `(n_neighbors,)` containing indices for scattering the
          computed moments to the correct atoms.

    Returns:
        List[torch.Tensor]:
          A list containing tensors for the zero, first, second, and third moments.
          Each tensor has shape `(n_atoms, n_radial, (3)^moment_number)`, where
          `moment_number` ranges from 0 to 3.
    """
    # dn shape: neighbors x 3
    # radial_function shape: n_neighbors x n_radial

    # s = spatial dim = 3
    xyz = einops.repeat(dn, "nbrs s -> nbrs 1 s")
    xyz2 = einops.repeat(dn, "nbrs s -> nbrs 1 1 s")
    xyz3 = einops.repeat(dn, "nbrs s -> nbrs 1 1 1 s")

    # shape: n_neighbors x n_radial x (3)^(moment_number)
    # s_i = spatial = 3
    zero_moment = radial_function
    first_moment = einops.repeat(zero_moment, "n r -> n r 1") * xyz
    second_moment = einops.repeat(first_moment, "n r s1 -> n r s1 1") * xyz2
    third_moment = einops.repeat(second_moment, "n r s1 s2 -> n r s1 s2 1") * xyz3

    # shape: n_atoms x n_radial x (3)^(moment_number)
    zero_moment = scatter_add(zero_moment, idx_i, dim=0,)
    first_moment = scatter_add(first_moment, idx_i, dim=0,)
    second_moment = scatter_add(second_moment, idx_i, dim=0,)
    third_moment = scatter_add(third_moment, idx_i, dim=0,)

    moments = [zero_moment, first_moment, second_moment, third_moment]

    return moments


def mask_by_neighbor(arr: Tensor, idx: Tensor):
    """
    Apply a mask to an array based on neighbor relationships between sending and
    receiving nodes. This masks out self interactions between sending and receiving
    nodes.

    Args:
        arr (Tensor):
            The input array to be masked. The shape of `arr` can be either 2D or 4D.
        idx (Tensor):
            An array of shape (2, N) where:
            - `idx[0]` contains the indices of sending nodes.
            - `idx[1]` contains the indices of receiving nodes.

    Returns:
        Tensor:
            The masked array. The shape of the returned array matches `arr`,
            with elements masked according to the computed mask.

    Notes:
        - For 2D arrays, the mask is reshaped to have a shape of (N, 1) before
          applying it to `arr`.
        - For 4D arrays, the mask is reshaped to have a shape of (N, 1, 1, 1)
          before applying it to `arr`.
        - The mask is a binary array where a value of 1 indicates that the sending
          and receiving nodes are different, meaning they are considered neighbors.
    """
    mask = ((idx[0] - idx[1]) != 0).to(arr.dtype)
    if len(arr.shape) == 2:
        mask = mask[..., None]
    elif len(arr.shape) == 4:
        mask = mask[:, None, None, None]
    return arr * mask



class GaussianBasis(nn.Module):
    def __init__(self, n_basis: int = 7, r_min: float = 0.5, r_max: float = 6.0):
        super().__init__()
        self.n_basis = n_basis
        self.r_min = r_min
        self.r_max = r_max
        self.dim = n_basis

        # Compute the constants used in the Gaussian basis
        self.betta = self.n_basis**2 / self.r_max**2
        self.rad_norm = (2.0 * self.betta / np.pi) ** 0.25

        # Create shifts
        shifts = self.r_min + (self.r_max - self.r_min) / self.n_basis * np.arange(
            self.n_basis
        )

        # Convert to tensor and reshape to 1 x n_basis (broadcast-compatible with dr)
        self.shifts = torch.tensor(shifts).view(1, -1)

    def forward(self, dr: Tensor):
        # Reshape dr to be neighbors x 1 (add an extra dimension for broadcasting)
        dr = dr.view(-1, 1)

        # Compute the distance between shifts and dr
        distances = self.shifts.to(dr.device) - dr

        # Apply the Gaussian basis function
        basis = torch.exp(-self.betta * (distances**2))
        basis = self.rad_norm * basis
        return basis.to(dr.dtype)
    
class RadialFunction(nn.Module):
    def __init__(
        self,
        n_radial: int = 5,
        basis_fn: nn.Module = GaussianBasis(),
        n_species: int = 119,
        emb_init: Optional[str] = "uniform",
        use_embed_norm: bool = True,
        one_sided_dist: bool = False,
    ):
        """
        A module for computing radial functions used in molecular models.

        The radial function calculates the distance-dependent embedding for atomic pairs,
        using a Gaussian basis and optional type embeddings for species pairs.

        Args:
            n_radial (int, optional):
              Number of radial basis functions to use. Defaults to 5.
            basis_fn (nn.Module, optional):
              The basis function, typically a Gaussian basis.
            n_species (int, optional):
              Number of unique atomic species. Defaults to 119.
            emb_init (Optional[str]):
              Initialization type for embeddings ('uniform' supported).
            use_embed_norm (bool), optional:
              Whether to apply normalization to the embeddings.
            one_sided_dist (bool, optional):
              If True, distances are considered one-sided (non-negative).
        """
        super().__init__()
        self.basis_fn = basis_fn
        self.n_radial = n_radial
        self.n_species = n_species
        self.emb_init = emb_init
        self.use_embed_norm = use_embed_norm
        self.one_sided_dist = one_sided_dist

        self.r_max = self.basis_fn.r_max
        self.embed_norm = torch.tensor(1.0 / np.sqrt(self.basis_fn.n_basis))

        if self.one_sided_dist:
            self.lower_bound = 0.0
        else:
            self.lower_bound = -1.0

        if self.emb_init is not None:
            self._n_radial = self.n_radial
            if self.emb_init == "uniform":
                emb_initializer = uniform_range(self.lower_bound, 1.0)
                self.embeddings = nn.Parameter(
                    emb_initializer(
                        (
                            self.n_species,
                            self.n_species,
                            self.n_radial,
                            self.basis_fn.n_basis,
                        )
                    )
                )
            else:
                raise ValueError(
                    "Currently only uniformly initialized embeddings are supported."
                )
        else:
            self._n_radial = self.basis_fn.n_basis

    def forward(self, dr, Z_i=None, Z_j=None):
        """
        Forward pass to compute the radial function based on pairwise distances and species.

        Args:
            dr (torch.Tensor):
              Distance matrix (n_neighbors, 1).
            Z_i (torch.Tensor):
              Atomic species for atoms i.
            Z_j (torch.Tensor):
              Atomic species for atoms j.

        Returns:
            torch.Tensor:
              The computed radial function for each neighbor (n_neighbors x n_radial).
        """
        # Basis function evaluation
        basis = self.basis_fn(dr)

        if self.emb_init is None:
            radial_function = basis
        else:
            assert Z_i is not None and Z_j is not None, "Atomic species Z_i and Z_j must be provided when using embeddings."
            species_pair_coeffs = self.embeddings[Z_i, Z_j, ...]
            if self.use_embed_norm:
                species_pair_coeffs = self.embed_norm * species_pair_coeffs

            # Compute the radial function with the embedding coefficients
            radial_function = einops.einsum(
                species_pair_coeffs,
                basis,
                "nbrs radial basis, nbrs basis -> nbrs radial",
            )

        # Apply a cutoff function to the radial function
        # dr_clipped = torch.clamp(dr, max=self.r_max)
        # cos_cutoff = 0.5 * (torch.cos(np.pi * dr_clipped / self.r_max) + 1.0)

        # radial_function = radial_function * cos_cutoff

        return radial_function

class GaussianMomentDescriptor(nn.Module):
    def __init__(
        self,
        n_contr: int = 8,
        n_basis: int = 7,
        max_radius: float = 6.0,
        n_radial: int = 5,
        use_atom_type_embeddings: bool = False,
        reduced_dim: Optional[int] = None,
    ):
        """
        Initializes the GaussianMomentDescriptor with given radial function and number of contractions.

        Args:
            n_radial (int, optional):
              Number of radial basis functions to use. Defaults to 5. Only relevant if 
              use_atom_type_embeddings is True.
            n_contr (int, optional):
              Number of contractions to compute (up to 8). Defaults to 8.
            max_radius (float, optional):
              Maximum radius for the radial basis functions. Defaults to 6.0.
            use_atom_type_embeddings (bool, optional):
              Whether to use atom type embeddings in the radial function. Defaults to False.
            reduced_dim (Optional[int], optional):
              If specified, reduces the output dimension of the descriptor to this value using a linear layer. Defaults to None (no reduction).
        """
        super().__init__()
        self.n_contr = n_contr
        self.radial_fn = RadialFunction(
            n_radial=n_radial,
            basis_fn=GaussianBasis(n_basis=n_basis, r_max=max_radius),
            emb_init="uniform" if use_atom_type_embeddings else None
        )
        self.r_max = max_radius
        self.n_radial = self.radial_fn._n_radial
        self.triang_idxs_2d = tril_2d_indices(self.n_radial)
        self.triang_idxs_3d = tril_3d_indices(self.n_radial)
        self.reduced_dim = reduced_dim

        if reduced_dim is not None:
            self.dim = GM_DIM[self.n_radial]
            self.reduction_layer = nn.Linear(self.dim, reduced_dim, bias=False)
            self.dim = reduced_dim

    def forward(
        self,
        positions: Tensor,
        edge_index: Tensor,
        Z: Optional[Tensor] = None,
    ):
        """
        Computes the Gaussian moments for given coordinates, edge indices, and atomic numbers.

        Args:
            positions (Tensor):
              Tensor of shape (n_atoms, 3) containing the positions of the atoms.
            edge_index (Tensor):
              Tensor of shape (2, n_edges) containing the indices of neighboring atoms.
            Z (Tensor, optional):
              Tensor of shape (n_atoms) containing the atomic numbers of the atoms.

        Returns:
            Tensor:
              A tensor containing the concatenated Gaussian moments up to the specified
              number of contractions. The shape depends on the number of contractions
              and the radial basis functions.
        """
        jj, ii = edge_index[0], edge_index[1]
        rij = positions[jj] - positions[ii]
        distances = torch.norm(rij, dim=-1, keepdim=True)
        coord_diff = rij / (distances + 1e-8)

        # Radial function
        if Z is not None:
            Z_i, Z_j = Z[ii].long(), Z[jj].long()
        else:
            Z_i, Z_j = None, None
        radial_function = self.radial_fn(distances, Z_i, Z_j)

        # Compute geometric moments
        moments = geometric_moments(radial_function, coord_diff, ii)

        # Perform contractions
        contr_0 = moments[0]
        contr_1 = torch.einsum("ari, asi -> rsa", moments[1], moments[1])
        contr_2 = torch.einsum("arij, asij -> rsa", moments[2], moments[2])
        contr_3 = torch.einsum("arijk, asijk -> rsa", moments[3], moments[3])
        contr_4 = torch.einsum(
            "arij, asik, atjk -> rsta", moments[2], moments[2], moments[2]
        )
        contr_5 = torch.einsum(
            "ari, asj, atij -> rsta", moments[1], moments[1], moments[2]
        )
        contr_6 = torch.einsum(
            "arijk, asijl, atkl -> rsta", moments[3], moments[3], moments[2]
        )
        contr_7 = torch.einsum(
            "arijk, asij, atk -> rsta", moments[3], moments[2], moments[1]
        )

        n_symm01_features = self.triang_idxs_2d.shape[0] * self.n_radial

        tril_2_i, tril_2_j = self.triang_idxs_2d[:, 0], self.triang_idxs_2d[:, 1]
        tril_3_i, tril_3_j, tril_3_k = (
            self.triang_idxs_3d[:, 0],
            self.triang_idxs_3d[:, 1],
            self.triang_idxs_3d[:, 2],
        )

        # Apply triangular indices
        contr_1 = contr_1[tril_2_i, tril_2_j]
        contr_2 = contr_2[tril_2_i, tril_2_j]
        contr_3 = contr_3[tril_2_i, tril_2_j]
        contr_4 = contr_4[tril_3_i, tril_3_j, tril_3_k]
        contr_5 = contr_5[tril_2_i, tril_2_j]
        contr_6 = contr_6[tril_2_i, tril_2_j]

        # Reshape the higher-order contractions
        contr_5 = torch.reshape(contr_5, [n_symm01_features, -1])
        contr_6 = torch.reshape(contr_6, [n_symm01_features, -1])
        contr_7 = torch.reshape(contr_7, [self.n_radial**3, -1])

        # Transpose to match the desired output shape
        contr_1 = torch.transpose(contr_1, 0, 1)
        contr_2 = torch.transpose(contr_2, 0, 1)
        contr_3 = torch.transpose(contr_3, 0, 1)
        contr_4 = torch.transpose(contr_4, 0, 1)
        contr_5 = torch.transpose(contr_5, 0, 1)
        contr_6 = torch.transpose(contr_6, 0, 1)
        contr_7 = torch.transpose(contr_7, 0, 1)

        gaussian_moments = [
            contr_0,
            contr_1,
            contr_2,
            contr_3,
            contr_4,
            contr_5,
            contr_6,
            contr_7,
        ]

        # Concatenate the relevant Gaussian moments up to self.n_contr
        gaussian_moments = torch.cat(gaussian_moments[: self.n_contr], dim=-1)

        # Reduce the dimension
        if self.reduced_dim is not None:
            gaussian_moments = self.reduction_layer(gaussian_moments)

        return gaussian_moments