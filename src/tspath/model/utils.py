import torch
from typing import Optional
import torch.nn as nn
from torch_geometric.nn import radius_graph
from typing import Tuple

def map_shortest_hops_safe(old_edge_index, shortest_hops, new_edge_index, N, fill_value=-1):
    # Step 1: compute unique edge hashes
    old_hash = old_edge_index[0] * N + old_edge_index[1]
    new_hash = new_edge_index[0] * N + new_edge_index[1]

    # Step 2: Sort the old hashes to allow binary search
    sorted_old_hash, sort_idx = torch.sort(old_hash)
    sorted_hops = shortest_hops[sort_idx]

    # Step 3: Find where new hashes fit in the sorted old hashes
    idx = torch.searchsorted(sorted_old_hash, new_hash)

    # Step 4: Validate matches (searchsorted might return out-of-bounds or non-exact matches)
    idx_clamped = idx.clamp(max=len(sorted_old_hash) - 1)
    is_match = sorted_old_hash[idx_clamped] == new_hash

    # Step 5: Fill new hops, defaulting to fill_value where no match was found
    new_shortest_hops = torch.full_like(new_hash, fill_value, dtype=shortest_hops.dtype)
    new_shortest_hops[is_match] = sorted_hops[idx_clamped[is_match]]

    return new_shortest_hops

def extend_graph_order_radius(
    pos: torch.Tensor,
    batch: torch.Tensor,
    edge_index: Optional[torch.Tensor],
    edge_type: Optional[torch.Tensor],
    cutoff: float = 10.0,
    max_neighbors: int = 32,
    shortest_hops: torch.Tensor = None,
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
        new_edge_type = torch.ones(
            new_edge_index.size(1)
        ).long().to(pos.device) * unspecified_type_number
        
    new_shortest_hops = None
    if shortest_hops is not None:
        # new_shortest_hops = map_shortest_hops_safe(
        #     old_edge_index=fully_connected_edges_batched(batch),
        #     shortest_hops=shortest_hops,
        #     new_edge_index=new_edge_index,
        #     N=N,
        #     fill_value=-1,  # or some other value indicating "not found"
        # )
        new_shortest_hops = None


    return new_edge_index, new_edge_type, new_shortest_hops


def extend_bond_index(
    pos: torch.Tensor,
    batch: torch.Tensor,
    bond_index: Optional[torch.Tensor],
    bond_attr: Optional[torch.Tensor],
    one_hot: bool = False,
    one_hot_types: int = 5,
    cutoff: float = 10.0,
    max_neighbors: int = 32,
    shortest_hops: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bond_type = None
    if bond_attr is None:
        if bond_index is not None:
            # all molecular graph edges are type 1, radius based become 0
            bond_type = torch.ones(bond_index.shape[1], dtype=torch.long, device=pos.device)
    else:
        bond_type = bond_attr.view(-1).long() + 1  # we reserve 0 for radius based edges
        assert bond_type.shape[0] == bond_index.shape[1], (
            "Edge type should have same shape as number of edges."
        )

    edge_index, edge_type, shortest_hops = extend_graph_order_radius(
        pos=pos,
        edge_index=bond_index,
        edge_type=bond_type,
        batch=batch,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        shortest_hops=shortest_hops,
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

    return edge_index, edge_type, shortest_hops