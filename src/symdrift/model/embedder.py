"""Implementation of the Gaussian moment descriptor to encode local atomic environment from Zaverkin et al. (https://doi.org/10.1021/acs.jctc.0c00347)"""

from typing import Optional

import einops
import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn import radius_graph
from torch_scatter import scatter, scatter_add

from symdrift.model.utils import get_fully_connected_triu_edges

__all__ = ["GaussianMomentEmbedder", "DistanceEmbedder"]


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
    radial_function: Tensor,
    dn: Tensor,
    idx_i: Tensor,
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
    zero_moment = scatter_add(
        zero_moment,
        idx_i,
        dim=0,
    )
    first_moment = scatter_add(
        first_moment,
        idx_i,
        dim=0,
    )
    second_moment = scatter_add(
        second_moment,
        idx_i,
        dim=0,
    )
    third_moment = scatter_add(
        third_moment,
        idx_i,
        dim=0,
    )

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


class GaussianBasis:
    def __init__(self, n_basis: int = 7, r_min: float = 0.5, r_max: float = 6.0):
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

    def __call__(self, dr: Tensor):
        # Reshape dr to be neighbors x 1 (add an extra dimension for broadcasting)
        dr = dr.view(-1, 1)

        # Compute the distance between shifts and dr
        distances = self.shifts.to(dr.device) - dr

        # Apply the Gaussian basis function
        basis = torch.exp(-self.betta * (distances**2))
        basis = self.rad_norm * basis
        return basis.to(dr.dtype)


class GaussianMomentEmbedder:
    def __init__(
        self,
        n_contr: int = 8,
        n_basis: int = 7,
        max_radius: float = 15.0,
        max_num_neighbors: int = 500,
        aggregation: Optional[str] = "mean",
    ):
        """
        Initializes the GaussianMomentDescriptor with given radial function and number of contractions.

        Args:
            n_contr (int, optional):
              Number of contractions to compute (up to 8). Defaults to 8.
            n_basis (int, optional):
              Number of radial basis functions to use. Defaults to 7. Only relevant if
              use_atom_type_embeddings is True.
            max_radius (float, optional):
              Maximum radius for the radial basis functions. Defaults to 15.0.
            max_num_neighbors (int, optional):
              Maximum number of neighbors to consider for each atom when constructing
              the graph. Defaults to 500.
            aggregation (str, optional):
                Method for aggregating the moments (e.g., 'mean', 'add').
        """
        self.n_contr = n_contr
        self.radial_fn = GaussianBasis(n_basis=n_basis, r_max=max_radius)
        self.n_radial = n_basis
        self.r_max = max_radius
        self.max_num_neighbors = max_num_neighbors
        self.triang_idxs_2d = tril_2d_indices(self.n_radial)
        self.triang_idxs_3d = tril_3d_indices(self.n_radial)
        self.aggregation = aggregation

    def __repr__(self):
        return (
            f"GaussianMomentEmbedder(n_contr={self.n_contr}, "
            f"n_basis={self.n_radial}, "
            f"r_max={self.r_max}, "
            f"max_num_neighbors={self.max_num_neighbors}, "
            f"aggregation={self.aggregation})"
        )

    def __call__(
        self,
        positions: Tensor,
        batch: Optional[Tensor] = None,
        Z: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        **kwargs,
    ):
        """
        Computes the Gaussian moments for given coordinates, edge indices, and atomic numbers.

        Args:
            positions (Tensor):
              Tensor of shape (n_atoms, 3) containing the positions of the atoms.
            batch (Tensor):
                Tensor of shape (n_atoms,) containing the batch indices for each atom, if applicable.
            Z (Tensor, optional):
              Tensor of shape (n_atoms) containing the atomic numbers of the atoms.
            edge_index (Tensor, optional):
              Tensor of shape (2, n_edges) containing the indices of neighboring atoms.

        Returns:
            Tensor:
              A tensor containing the concatenated Gaussian moments up to the specified
              number of contractions. The shape depends on the number of contractions
              and the radial basis functions.
        """
        if batch is None:
            batch = torch.zeros(
                positions.shape[0], dtype=torch.long, device=positions.device
            )
        if edge_index is None:
            edge_index = radius_graph(
                positions,
                r=self.r_max,
                batch=batch,
                max_num_neighbors=self.max_num_neighbors,
            )

        jj, ii = edge_index[0], edge_index[1]
        rij = positions[jj] - positions[ii]
        distances = torch.norm(rij, dim=-1, keepdim=True)
        coord_diff = rij / (distances + 1e-8)

        # Radial function
        radial_function = self.radial_fn(distances)

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

        # Aggregate over the batch dimension if wanted
        if self.aggregation is None:
            return gaussian_moments
        else:
            # aggregate the moments for each graph in the batch using the specified
            # aggregation method and flatten them. Createa a mask to identify which
            # entries in the aggregated tensor correspond to which batch.
            B = batch.max().item() + 1
            aggregated = scatter(
                gaussian_moments, batch, dim=0, reduce=self.aggregation
            ).view(-1)
            mask = torch.arange(B).repeat_interleave(gaussian_moments.shape[1])
            return aggregated, mask


class DistanceEmbedder:
    def __init__(
        self,
        invariant=True,
        r_max: Optional[float] = None,
        max_num_neighbors=500,
    ):
        self.invariant = invariant
        if r_max is None:
            r_max = float("inf")
        self.r_max = r_max
        self.max_num_neighbors = max_num_neighbors

    def __repr__(self):
        return (
            f"DistanceEmbedder(invariant={self.invariant}, "
            f"r_max={self.r_max}, "
            f"max_num_neighbors={self.max_num_neighbors})"
        )

    def _compute_atom_type_orbit_ids(
        self, row: Tensor, col: Tensor, Z: Tensor
    ) -> Tensor:
        """Compute orbit IDs by treating every atom of the same type as interchangeable.
        """
        Zi, Zj = Z[row], Z[col]
        Zmax_val = Z.max() + 1
        orbit_ids = torch.minimum(Zi, Zj) * Zmax_val + torch.maximum(Zi, Zj)
        return orbit_ids

    def _broadcast_orbit_ids(
        self,
        orbit_ids: Tensor,
        edge_batch: Tensor,
    ) -> Tensor:
        """Use the orbit IDs provided per batch edge and broadcast them along the edge
        batch. This will make sure that no edges that belong to different batches will
        be treated as interchangeable when sorting.
        """
        group_id = edge_batch * (orbit_ids.max() + 1) + orbit_ids
        return group_id

    def __call__(
        self,
        positions: Tensor,
        batch: Optional[Tensor] = None,
        Z: Optional[Tensor] = None,
        orbit_ids: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        invariant: Optional[bool] = None,
        **kwargs,
    ):
        """
        Embed the positions of atoms using a distance-based embedding.

        Args:
            positions (Tensor):
                Tensor of shape (B*n_atoms, 3) containing the positions of the atoms.
            batch (Tensor, optional):
                Tensor of shape (B*n_atoms,) containing the batch indices for each atom.
            Z (Tensor, optional):
                Tensor of shape (B*n_atoms) containing the atomic numbers of the atoms.
                This will be used to compute the invariant embedding by treating atoms 
                of the same type as interchangeable. Needed if `invariant` is True to 
                compute the invariant embedding.
            orbit_ids (Tensor, optional):
                Flat [sum(n_pairs_i)] tensor of precomputed upper-triangle orbit IDs
                assigning each pair of atoms to an orbit defined by the automorphisms.
                This will treat pairs of atoms that are considered interchangeable under
                graph automorphisms as interchangeable, which is a more fine-grained 
                notion of invariance than just treating atoms of the same type as 
                interchangeable. Needed if `invariant` is True and `use_automorphisms` 
                is True to compute the invariant embedding.
            edge_index (Tensor, optional):
              Tensor of shape (2, n_edges) containing the indices of neighboring atoms.
            invariant (bool, optional):
                If True, the embedding will be invariant to permutations of atoms.
                If False, the embedding will be based on the full distance matrix.
                If None, it will use the class attribute `self.invariant`.

        Output:
            dist (Tensor): 
                Tensor of shape (n_edges,) containing the distances for each edge.
            edge_batch (Tensor): 
                Tensor of shape (n_edges,) containing the batch index for each edge.
        """
        if invariant is not None:
            # if desired overwrite invariant attribute with forward argument
            self.invariant = invariant
        if batch is None:
            batch = torch.zeros(
                positions.shape[0], dtype=torch.long, device=positions.device
            )
        if edge_index is not None:
            row, col = edge_index
        elif self.r_max < float("inf"):
            assert orbit_ids is None, (
                "Precomputed orbit_ids should not be provided when using radius "
                "graph with finite r_max since the edges won't align with the fully "
                "connected upper triangular format used to compute the orbit_ids."
            )
            row, col = radius_graph(
                positions,
                r=self.r_max,
                batch=batch,
                max_num_neighbors=self.max_num_neighbors,
            )
            # mask out all symmetric entries (keep only one of (i,j) and (j,i))
            mask = row < col
            row, col = row[mask], col[mask]
        else:
            row, col = get_fully_connected_triu_edges(batch)

        # compute distances for the edges (assuming fully connected graph, reconstructs
        # the full distance matrix)
        dist = (positions[row] - positions[col]).norm(dim=-1)

        # get batch indices for the edges
        edge_batch = batch[row]

        if not self.invariant:
            return dist, edge_batch

        # Use the orbit IDs provided per batch edge and broadcast them along the edge batch. 
        # Edges that belong to the same group will be treated as interchangeable when 
        # sorting, which ensures the embedding is invariant to permutations of atoms 
        # that are considered interchangeable under the provided orbit IDs.
        if orbit_ids is not None:
            group_id = self._broadcast_orbit_ids(orbit_ids, edge_batch)
        elif Z is not None:
            orbit_ids = self._compute_atom_type_orbit_ids(row, col, Z)
            group_id = self._broadcast_orbit_ids(orbit_ids, edge_batch)
        else:
            raise ValueError(
                "To compute an invariant embedding, either orbit_ids, "
                "or atomic numbers Z must be provided."
            )

        # sort edges by distance that lie within the same group type to make it 
        # permutation invariant.
        max_dist = dist.max().detach() + 1.0
        key = group_id * max_dist + dist

        # apply permutation
        perm = torch.argsort(key)
        dist_sorted = dist[perm]
        edge_batch_sorted = edge_batch[perm]
        assert (edge_batch_sorted[1:] >= edge_batch_sorted[:-1]).all()

        return dist_sorted, edge_batch_sorted
