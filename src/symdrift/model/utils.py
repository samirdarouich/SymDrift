from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.nn import radius_graph

__all__ = [
    "signed_volume",
    "extend_graph_order_radius",
    "extend_bond_index",
    "get_fully_connected_triu_edges",
]


def signed_volume(local_coords):
    """
    Compute signed volume given ordered neighbor local coordinates
    From GeoMol

    :param local_coords: (n_tetrahedral_chiral_centers, 4, n_generated_confs, 3)
    :return: signed volume of each tetrahedral center
    (n_tetrahedral_chiral_centers, n_generated_confs)
    """
    v1 = local_coords[:, 0] - local_coords[:, 3]
    v2 = local_coords[:, 1] - local_coords[:, 3]
    v3 = local_coords[:, 2] - local_coords[:, 3]
    cp = v2.cross(v3, dim=-1)
    vol = torch.sum(v1 * cp, dim=-1)
    return torch.sign(vol)


def extend_graph_order_radius(
    pos: torch.Tensor,
    batch: torch.Tensor,
    edge_index: Optional[torch.Tensor],
    edge_type: Optional[torch.Tensor],
    cutoff: float = 10.0,
    max_neighbors: int = 32,
    unspecified_type_number: int = 0,
):
    if edge_index is not None:
        assert edge_type.dim() == 1
        N = pos.size(0)

        bgraph_adj = torch.sparse_coo_tensor(edge_index, edge_type, torch.Size([N, N]))

        rgraph_edge_index = radius_graph(
            pos, r=cutoff, batch=batch, max_num_neighbors=max_neighbors
        )  # (2, E_r)

        rgraph_adj = torch.sparse_coo_tensor(
            rgraph_edge_index,
            torch.ones(rgraph_edge_index.size(1)).long().to(pos.device)
            * unspecified_type_number,
            torch.Size([N, N]),
        )

        composed_adj = (bgraph_adj + rgraph_adj).coalesce()  # Sparse (N, N, T)

        new_edge_index = composed_adj.indices()
        new_edge_type = composed_adj.values().long()
    else:
        # If no initial edge_index is provided, we just create a radius graph
        new_edge_index = radius_graph(
            pos, r=cutoff, batch=batch, max_num_neighbors=max_neighbors
        )  # (2, E_r)
        new_edge_type = (
            torch.ones(new_edge_index.size(1)).long().to(pos.device)
            * unspecified_type_number
        )

    return new_edge_index, new_edge_type


def extend_bond_index(
    pos: torch.Tensor,
    batch: torch.Tensor,
    bond_index: Optional[torch.Tensor] = None,
    bond_attr: Optional[torch.Tensor] = None,
    one_hot: bool = False,
    one_hot_types: int = 5,
    cutoff: float = 10.0,
    max_neighbors: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bond_type = None
    if bond_attr is None:
        if bond_index is not None:
            # all molecular graph edges are type 1, radius based become 0
            bond_type = torch.ones(
                bond_index.shape[1], dtype=torch.long, device=pos.device
            )
    else:
        bond_type = bond_attr.view(-1).long() + 1  # we reserve 0 for radius based edges
        assert bond_type.shape[0] == bond_index.shape[1], (
            "Edge type should have same shape as number of edges."
        )

    edge_index, edge_type = extend_graph_order_radius(
        pos=pos,
        edge_index=bond_index,
        edge_type=bond_type,
        batch=batch,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        unspecified_type_number=0,
    )

    if bond_index is not None:
        assert bond_index.shape[1] == (edge_type > 0).sum().item(), (
            "Edge Type should be greater than 0 when edge is a molecular bond."
        )

    # make one_hot if provided
    if one_hot:
        # +1 to account for radius based edges
        edge_type = torch.nn.functional.one_hot(
            edge_type, num_classes=one_hot_types + 1
        ).float()

    return edge_index, edge_type


def get_fully_connected_triu_edges(batch: Tensor) -> tuple[Tensor, Tensor]:
    """
    Return upper-triangle edge indices for a batch of fully-connected graphs.

    Produces edges in canonical row-major (triu_indices) order within each
    graph.

    The key insight: atoms within each graph occupy a contiguous block of
    global indices, so for atom k (local index l in graph g) every
    upper-triangle neighbour is simply k+1, k+2, ..., k+(n_g-l-1).

    Args:
        batch: node-to-graph assignment, shape [n_nodes].

    Returns:
        (row, col): each shape [total_pairs], dtype=long, on same device as batch.
    """
    device = batch.device
    n_nodes = batch.shape[0]
    num_atoms = batch.bincount()  # [n_graphs]

    node_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=device),
            num_atoms.cumsum(0)[:-1],
        ]
    )  # [n_graphs]

    # Local index of each node within its graph
    local_idx = torch.arange(n_nodes, device=device) - node_offsets[batch]  # [n_nodes]

    # Number of upper-triangle edges originating from each node
    n_edges_from_node = (num_atoms[batch] - local_idx - 1).clamp(min=0)  # [n_nodes]

    total_pairs = n_edges_from_node.sum()

    # Row: repeat each global node index by its edge count
    row = torch.repeat_interleave(
        torch.arange(n_nodes, device=device), n_edges_from_node
    )  # [total_pairs]

    # Intra-run offset: 0, 1, ..., n_edges_from_node[k]-1 for node k
    run_start = torch.repeat_interleave(
        torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                n_edges_from_node.cumsum(0)[:-1],
            ]
        ),
        n_edges_from_node,
    )  # [total_pairs]
    col = (
        row + (torch.arange(total_pairs, device=device) - run_start) + 1
    )  # [total_pairs]

    return row, col
