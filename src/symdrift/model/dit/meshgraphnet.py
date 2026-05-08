import torch
import torch.nn as nn
from torch_scatter import scatter_add
from symdrift.model.dit.embeddings import EdgeAttributeEmbedding, NodeAttributeEmbedding
from symdrift.model.dit.utils import MLP


class MeshGraphNetLayer(nn.Module):
    def __init__(self, num_node_features, num_edge_features, activation_fn="silu"):
        super().__init__()

        # edge update
        self.edge_mlp = MLP(
            in_dim=2 * num_node_features + num_edge_features,
            hidden_dim=num_edge_features,
            out_dim=num_edge_features,
            num_layers=2,
            activation_fn=activation_fn,
            use_bias=True,
        )
        self.layer_norm_edge = nn.LayerNorm(2 * num_node_features + num_edge_features)

        # node update
        self.node_mlp = MLP(
            in_dim=num_node_features + num_edge_features,
            hidden_dim=num_node_features,
            out_dim=num_node_features,
            num_layers=2,
            activation_fn=activation_fn,
            use_bias=True,
        )
        self.layer_norm_node = nn.LayerNorm(num_node_features + num_edge_features)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index  # source, target

        # ---- edge update ----
        edge_input = self.layer_norm_edge(
            torch.cat([x[row], x[col], edge_attr], dim=-1)
        )
        edge_attr = self.edge_mlp(edge_input)

        # ---- aggregate messages ----
        agg = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))  # (num_nodes, num_edge_features)

        # ---- node update ----
        node_input = self.layer_norm_node(torch.cat([x, agg], dim=-1))
        x = self.node_mlp(node_input)

        return x, edge_attr


# --- Encoder ---
class MeshGraphNetEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, num_layers, hidden_dim, activation_fn):
        super().__init__()

        self.node_embedding = NodeAttributeEmbedding(
            node_dim, hidden_dim, activation_fn
        )
        self.edge_embedding = EdgeAttributeEmbedding(
            edge_dim, hidden_dim, activation_fn
        )

        self.layers = nn.ModuleList(
            [
                MeshGraphNetLayer(
                    num_node_features=hidden_dim,
                    num_edge_features=hidden_dim,
                    activation_fn=activation_fn,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, data):
        """
        data.x          : node features (num_nodes, node_attr_dim)
        data.edge_index : (2, num_edges)
        data.edge_attr  : (num_edges, edge_attr_dim)
        """

        x = self.node_embedding(data)
        edge_attr = self.edge_embedding(data)

        # Use graph bonds for message passing
        for layer in self.layers:
            x, edge_attr = layer(x, data.bonded_edge_index, edge_attr)

        return x  # (num_atoms,  num_features)
