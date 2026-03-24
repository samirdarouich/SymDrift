import math

import torch
import torch.nn as nn
from tspath.model.dit.utils import MLP, get_index_embedding


class NodeAttributeEmbedding(nn.Module):
    def __init__(self, in_dim, num_features, activation_fn):
        super().__init__()
        self.num_features = num_features
        self.activation_fn = activation_fn

        self.mlp = MLP(
            in_dim=in_dim,
            hidden_dim=num_features,
            out_dim=self.num_features,
            num_layers=2,
            activation_fn=self.activation_fn,
        )
        self.norm = nn.LayerNorm(num_features)

    def forward(self, data):
        """
        data.node_attr : (num_nodes, node_attr_dim)
        """
        node_attr = data.node_attr
        h = self.mlp(node_attr)
        h = self.norm(h)
        return h  # (N,num_features)


class EdgeAttributeEmbedding(nn.Module):
    def __init__(self, in_dim, num_features, activation_fn):
        super().__init__()
        self.num_features = num_features
        self.activation_fn = activation_fn

        self.mlp = MLP(
            in_dim=in_dim,
            hidden_dim=num_features,
            out_dim=self.num_features,
            num_layers=2,
            activation_fn=self.activation_fn,
        )
        self.norm = nn.LayerNorm(num_features)

    def forward(self, data):
        """
        data.edge_features : (num_edges, edge_attr_dim)
        """
        edge_attr = data.edge_attr
        e = self.mlp(edge_attr)
        e = self.norm(e)
        return e  # (E,F)


class DiTNodeEmbed(nn.Module):
    def __init__(
        self,
        num_features,
        activation_fn="silu",
        positional_encoding_bool=True,
        positional_embedding_bool=True,
    ):
        super().__init__()

        self.num_features = num_features
        self.positional_encoding_bool = positional_encoding_bool
        self.positional_embedding_bool = positional_embedding_bool

        self.atom_embed = nn.Embedding(119, num_features)

        if positional_embedding_bool:
            self.pos_mlp = MLP(
                in_dim=3,
                hidden_dim=num_features,
                out_dim=num_features,
                num_layers=2,
                activation_fn=activation_fn,
                use_bias=False,
            )

    def forward(self, data):

        atomic_numbers = data.x.long()  # (N)

        h = self.atom_embed(atomic_numbers)  # (N,F)

        if self.positional_encoding_bool:
            n_node = data.num_atoms
            num_nodes = len(data.pos)
            offsets = torch.repeat_interleave(data.ptr[:-1], n_node)
            indices = torch.arange(num_nodes, device=data.pos.device) - offsets
            h = h + get_index_embedding(indices, self.num_features)

        # Absolute positional embedding (does not adhere any symmetry)
        if self.positional_embedding_bool:
            positions = data.pos
            p = self.pos_mlp(positions)
            h = h + p

        return h  # (N,F)


class DiTEdgeEmbed(nn.Module):
    def __init__(
        self,
        num_features,
        activation_fn: str = "silu",
        embed_distances_bool: bool = True,
        embed_shortest_hops_bool: bool = True,
        radial_basis_bool=True,
        num_radial_basis=8,
        max_frequency=2 * math.pi,
    ):
        super().__init__()

        self.num_features = num_features
        self.embed_distances_bool = embed_distances_bool
        self.embed_shortest_hops_bool = embed_shortest_hops_bool
        self.radial_basis_bool = radial_basis_bool

        self.num_radial_basis = num_radial_basis
        self.max_frequency = max_frequency

        if radial_basis_bool:
            if num_radial_basis is None or max_frequency is None:
                raise ValueError("num_radial_basis and max_frequency required")

        if embed_distances_bool:
            self.dist_mlp = MLP(
                in_dim=3 if not radial_basis_bool else 3 * num_radial_basis,
                hidden_dim=num_features,
                out_dim=num_features,
                num_layers=2,
                activation_fn=activation_fn,
                use_bias=False,
            )
        
        
        if embed_shortest_hops_bool:
            self.shortest_hop_embedding = nn.Embedding(
                num_embeddings=512,
                embedding_dim=self.num_features,
            )

            self.shortest_hop_mlp = MLP(
                in_dim=self.num_features,
                hidden_dim=num_features,
                out_dim=num_features,
                num_layers=2,
                activation_fn=activation_fn,
                use_bias=True,  # we need a bias here s.t. the output is non-zero in case of CFG
            )

    def forward(self, data):

        senders = data.edge_index[0]
        receivers = data.edge_index[1]
        positions = data.pos
        num_edges = len(senders)

        # embed distances using Fourier features (adhere to translational but not
        # rotational symmetry)
        if self.embed_distances_bool:
            displacements = positions[senders] - positions[receivers]
            if self.radial_basis_bool:
                # simple Fourier basis
                freq = (
                    torch.arange(self.num_radial_basis, device=displacements.device)
                    * torch.pi
                    / self.max_frequency
                )
                displacements = displacements.unsqueeze(-1) * freq
                displacements = torch.sin(displacements)
                displacements = displacements.reshape(num_edges, -1)
            re = self.dist_mlp(displacements)
        else:
            re = torch.zeros(num_edges, self.num_features, device=positions.device)

        if self.embed_shortest_hops_bool:
            shortest_hops = data.shortest_hops
            sh_emb = self.shortest_hop_embedding(shortest_hops)
            sh_emb = self.shortest_hop_mlp(sh_emb)
            re = re + sh_emb

        return re  # (E,F)
