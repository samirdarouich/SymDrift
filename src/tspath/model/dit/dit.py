import math
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from tspath.model.dit.embeddings import DiTEdgeEmbed, DiTNodeEmbed
from tspath.model.dit.meshgraphnet import MeshGraphNetEncoder
from tspath.model.dit.utils import MLP, SimpleReadout, get_activation_fn, modulate_adaLN
from tspath.model.utils import extend_bond_index, signed_volume
from tspath.utils import batch_center_systems

class DiTLayer(nn.Module):
    def __init__(
        self,
        num_features,
        num_heads,
        num_features_mlp,
        num_features_condition=None,
        activation_fn="silu",
        activation_fn_mlp="gelu",
        edge_dim=None,
        act_dense_correct_bool: bool = False,
    ):
        super().__init__()

        self.num_features = num_features
        if num_features_condition is None:
            num_features_condition = num_features
        self.activation = get_activation_fn(activation_fn)
        self.act_dense_correct_bool = act_dense_correct_bool

        # AdaLN parameter generator
        self.adaLN_linear = nn.Linear(num_features_condition, 6 * num_features)
        nn.init.zeros_(self.adaLN_linear.weight)
        nn.init.zeros_(self.adaLN_linear.bias)

        # normalizations
        self.norm_cond = nn.LayerNorm(num_features_condition)
        self.norm1 = nn.LayerNorm(num_features, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(num_features, elementwise_affine=False)

        # graph transformer attention
        self.attn = TransformerConv(
            in_channels=num_features,
            out_channels=num_features // num_heads,
            heads=num_heads,
            edge_dim=edge_dim,
            concat=True,
        )

        # MLP
        self.mlp = MLP(
            num_features,
            num_features_mlp,
            num_features,
            num_layers=2,
            activation_fn=activation_fn_mlp,
        )

    def forward(
        self,
        x_nodes,
        edge_index,
        x_cond=None,
        edge_attr=None,
    ):

        if x_cond is None:
            x_cond = torch.zeros_like(x_nodes)

        # conditioning vector
        c = x_cond
        c = self.norm_cond(c)

        if self.act_dense_correct_bool:
            adaln_input = self.activation(self.adaLN_linear(c))
        else:
            adaln_input = self.adaLN_linear(c)

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = torch.chunk(
            adaln_input, 6, dim=-1
        )

        # ---- Attention block ----
        x = modulate_adaLN(self.norm1(x_nodes), gamma1, beta1)

        attn_out = self.attn(x, edge_index, edge_attr)

        x_nodes = x_nodes + attn_out * alpha1

        # ---- MLP block ----
        x = modulate_adaLN(self.norm2(x_nodes), gamma2, beta2)

        mlp_out = self.mlp(x)

        x_nodes = x_nodes + mlp_out * alpha2

        return x_nodes


class DiT(nn.Module):
    def __init__(
        self,
        sphere_channels=128,
        num_layers=6,
        num_heads=8,
        sphere_channels_mlp=None,
        mgn_num_node_features=10,
        mgn_num_edge_features=1,
        mgn_num_features=256,
        mgn_num_layers=2,
        mgn_activation_fn="silu",
        max_radius=6.0,
        max_neighbors=32,
        activation_fn="silu",
        activation_fn_mlp="gelu",
        absolute_positional_embedding_bool=False,  # breaks translation and rotation equivariance
        positional_encoding_bool=False,  # breaks permutation invariance
        relative_positional_embedding_bool=False,  # breaks rotation equivariance
        embed_shortest_hops_bool=False,
        act_dense_correct_bool=True,
        radial_basis_bool=False,
        num_distance_basis=8,
        max_frequency=2 * math.pi,
        parity_switch: Optional[str] = None,
    ):
        super().__init__()
        self.cutoff = max_radius
        self.max_neighbors = max_neighbors
        self.parity_switch = parity_switch

        if sphere_channels_mlp is None:
            sphere_channels_mlp = 4 * sphere_channels

        if mgn_num_layers > 0:
            self.enconder_cond = MeshGraphNetEncoder(
                node_dim=mgn_num_node_features,
                edge_dim=mgn_num_edge_features,
                hidden_dim=mgn_num_features,
                num_layers=mgn_num_layers,
                activation_fn=mgn_activation_fn,
            )
            num_features_condition = mgn_num_features
        else:
            self.enconder_cond = lambda data: None
            num_features_condition = sphere_channels

        self.node_embed = DiTNodeEmbed(
            num_features=sphere_channels,
            activation_fn=activation_fn,
            positional_encoding_bool=positional_encoding_bool,
            positional_embedding_bool=absolute_positional_embedding_bool,
        )

        self.edge_embed = DiTEdgeEmbed(
            num_features=sphere_channels,
            activation_fn=activation_fn,
            embed_distances_bool=relative_positional_embedding_bool,
            embed_shortest_hops_bool=embed_shortest_hops_bool,
            radial_basis_bool=radial_basis_bool,
            num_radial_basis=num_distance_basis,
            max_frequency=max_frequency,
        )

        self.layers = nn.ModuleList(
            [
                DiTLayer(
                    num_features=sphere_channels,
                    num_heads=num_heads,
                    num_features_mlp=sphere_channels_mlp,
                    num_features_condition=num_features_condition,
                    activation_fn=activation_fn_mlp,
                    edge_dim=sphere_channels,
                    act_dense_correct_bool=act_dense_correct_bool,
                )
                for _ in range(num_layers)
            ]
        )

        self.readout = SimpleReadout(
            num_features=sphere_channels, 
            num_features_condition=num_features_condition,
            activation_fn=activation_fn_mlp,
        )

    def forward(self, data):

        # Get conditioning vector from MGN encoder
        features_cond = self.enconder_cond(data)

        # combine bond and radius graph edges
        edge_index, _, new_shortest_hop = extend_bond_index(
            pos=data.pos,
            batch=data.batch,
            bond_index=data.get("bonded_edge_index", None),
            bond_attr=data.get("edge_attr", None),
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            shortest_hops=data.get("shortest_hops", None),
        )
        data.edge_index = edge_index
        data.shortest_hops = new_shortest_hop

        h = self.node_embed(data)
        edge_attr = self.edge_embed(data)

        for layer in self.layers:
            h = layer(
                x_nodes=h,
                edge_index=data.edge_index,
                x_cond=features_cond,
                edge_attr=edge_attr,
            )

        out = self.readout(features_nodes=h, features_cond=features_cond)
        out = batch_center_systems(out, data.batch)

        # Switch parity of positions for chiral molecules during inference if wanted.
        if self.parity_switch and not self.training:
            out = self.switch_parity_of_pos(
                out,
                data.chiral_index,
                data.chiral_nbr_index,
                data.chiral_tag,
                data.batch,
            )
        return out

    def switch_parity_of_pos(
        self, pos, chiral_index, chiral_nbr_index, chiral_tag, batch
    ):
        assert all(
            [
                key is not None
                for key in [chiral_index, chiral_nbr_index, chiral_tag, batch]
            ]
        )
        num_graphs = batch.max().item() + 1
        sv = signed_volume(
            pos[chiral_nbr_index.view(chiral_index.shape[1], 4)].unsqueeze(2)
        ).squeeze()
        ct = chiral_tag
        z_flip = sv * ct

        graph_diag = torch.ones(num_graphs, device=pos.device)
        graph_diag[batch[chiral_index][:, (z_flip == -1.0)].squeeze()] = -1.0
        node_factor = graph_diag[batch].unsqueeze(1)

        return pos * node_factor
