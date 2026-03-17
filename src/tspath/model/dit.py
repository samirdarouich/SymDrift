import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, radius_graph
from tspath.utils import batch_center_systems


def get_index_embedding(indices, emb_dim, max_len=256):
    """
    indices: (N,)
    returns: (N, emb_dim)
    """

    device = indices.device

    K = torch.arange(emb_dim // 2, device=device)

    indices = indices.unsqueeze(-1).float()

    denom = max_len ** (2 * K / emb_dim)

    sin = torch.sin(indices * math.pi / denom)
    cos = torch.cos(indices * math.pi / denom)

    pos_embedding = torch.cat([sin, cos], dim=-1)

    return pos_embedding

def modulate_adaLN(x, scale, shift):
    return x * (1 + scale) + shift


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation="gelu"):
        super().__init__()

        act = nn.GELU() if activation == "gelu" else nn.SiLU()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class DiTLayer(nn.Module):

    def __init__(
        self,
        num_features,
        num_heads,
        num_features_mlp,
        activation_fn="silu",
        activation_fn_mlp="gelu",
        edge_dim=None
    ):
        super().__init__()

        self.num_features = num_features

        self.activation = F.silu if activation_fn == "silu" else F.gelu

        # AdaLN parameter generator
        self.adaLN_linear = nn.Linear(num_features, 6 * num_features)
        nn.init.zeros_(self.adaLN_linear.weight)
        nn.init.zeros_(self.adaLN_linear.bias)

        # normalizations
        self.norm_cond = nn.LayerNorm(num_features)
        self.norm1 = nn.LayerNorm(num_features, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(num_features, elementwise_affine=False)

        # graph transformer attention
        self.attn = TransformerConv(
            in_channels=num_features,
            out_channels=num_features // num_heads,
            heads=num_heads,
            edge_dim=edge_dim,
            concat=True
        )

        # MLP
        self.mlp = MLP(num_features, num_features_mlp, num_features, activation_fn_mlp)

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

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = torch.chunk(
            self.activation(self.adaLN_linear(c)),
            6,
            dim=-1
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
    
    
class DiTNodeEmbed(nn.Module):

    def __init__(
        self,
        num_features,
        activation_fn="silu",
        self_conditioning_bool=False,
        positional_encoding_bool=True,
        positional_embedding_bool=True
    ):
        super().__init__()

        self.num_features = num_features
        self.self_conditioning_bool = self_conditioning_bool
        self.positional_encoding_bool = positional_encoding_bool
        self.positional_embedding_bool = positional_embedding_bool

        self.atom_embed = nn.Embedding(119, num_features)

        if self_conditioning_bool:
            self.self_cond_mlp = MLP(
                in_dim=3,
                hidden_dim=num_features,
                out_dim=num_features,
                activation=activation_fn
            )

        if positional_embedding_bool:
            self.pos_mlp = MLP(
                in_dim=3,
                hidden_dim=num_features,
                out_dim=num_features,
                activation=activation_fn
            )

    def forward(self, data):

        atomic_numbers = data.x.long()  # (N)

        h = self.atom_embed(atomic_numbers)                # (N,F)

        if self.self_conditioning_bool:

            self_cond = data.self_cond        # (N,3)
            sc = self.self_cond_mlp(self_cond)
            h = h + sc

        if self.positional_encoding_bool:

            n_node = data.num_atoms
            num_nodes = len(data.pos)
            offsets = torch.repeat_interleave(data.ptr[:-1], n_node)

            indices = torch.arange(num_nodes, device=data.pos.device) - offsets
            h = h + get_index_embedding(indices, self.num_features)

        if self.positional_embedding_bool:
            positions = data.pos
            p = self.pos_mlp(positions)
            h = h + p

        return h               # (N,F)    

class DiTEdgeEmbed(nn.Module):

    def __init__(
        self,
        num_features,
        activation_fn="silu",
        embed_distances_bool=True,
        radial_basis_bool=True,
        num_radial_basis=8,
        max_frequency=2*math.pi,
    ):
        super().__init__()

        self.num_features = num_features
        self.embed_distances_bool = embed_distances_bool
        self.radial_basis_bool = radial_basis_bool

        self.num_radial_basis = num_radial_basis
        self.max_frequency = max_frequency

        if radial_basis_bool:
            if num_radial_basis is None or max_frequency is None:
                raise ValueError("num_radial_basis and max_frequency required")

        self.dist_mlp = MLP(
            in_dim=3 if not radial_basis_bool else 3 * num_radial_basis,
            hidden_dim=num_features,
            out_dim=num_features,
            activation=activation_fn
        )

    def forward(self, data):

        senders = data.edge_index[0]
        receivers = data.edge_index[1]
        positions = data.pos
        num_edges = len(senders)

        if self.embed_distances_bool:

            displacements = positions[senders] - positions[receivers]

            if self.radial_basis_bool:
                # simple Fourier basis
                freq = torch.arange(
                    self.num_radial_basis,
                    device=displacements.device
                ) * torch.pi / self.max_frequency

                displacements = displacements.unsqueeze(-1) * freq
                displacements = torch.sin(displacements)
                displacements = displacements.reshape(num_edges, -1)

            re = self.dist_mlp(displacements)

        else:
            re = torch.zeros(
                num_edges,
                self.num_features,
                device=positions.device
            )


        return re  # (E,F)
    
class SimpleReadout(nn.Module):

    def __init__(self, num_features, activation_fn="silu"):
        super().__init__()

        self.num_features = num_features

        self.activation = F.silu if activation_fn == "silu" else F.gelu

        # conditioning → shift/scale
        self.adaLN_linear = nn.Linear(num_features, 2 * num_features)
        nn.init.zeros_(self.adaLN_linear.weight)
        nn.init.zeros_(self.adaLN_linear.bias)

        self.norm = nn.LayerNorm(num_features, elementwise_affine=False)

        # output projections
        self.out_proj = nn.Linear(num_features, 3)

    def forward(
        self,
        features_nodes,
        features_cond=None,
    ):

        # features_nodes: (num_nodes, num_features)
        assert features_nodes.ndim == 2

        features_nodes = features_nodes  # (N, F)

        if features_cond is None:
            features_nodes_cond = torch.zeros_like(features_nodes)

        # conditioning vector
        c = features_nodes_cond

        shift, scale = torch.chunk(
            self.activation(self.adaLN_linear(c)),
            2,
            dim=-1
        )

        y = modulate_adaLN(
            x=self.norm(features_nodes),
            shift=shift,
            scale=scale
        )

        out = self.out_proj(y)

        return out
    
class DiT(nn.Module):
    def __init__(
        self,
        sphere_channels=128,
        num_layers=6,
        num_heads=8,
        sphere_channels_mlp=256,
        max_radius=6.0,
        max_neighbors=32,
        activation_fn="silu",
        activation_fn_mlp="gelu",
        self_conditioning_bool=False,
        positional_encoding_bool=True,
        positional_embedding_bool=True,
        embed_distances_bool=True,
        radial_basis_bool=True,
        num_radial_basis=8,
        max_frequency=2 * math.pi
    ):
        super().__init__()
        self.cutoff = max_radius
        self.max_neighbors = max_neighbors

        self.node_embed = DiTNodeEmbed(
            num_features=sphere_channels,
            activation_fn=activation_fn,
            self_conditioning_bool=self_conditioning_bool,
            positional_encoding_bool=positional_encoding_bool,
            positional_embedding_bool=positional_embedding_bool
        )

        self.edge_embed = DiTEdgeEmbed(
            num_features=sphere_channels,
            activation_fn=activation_fn,
            embed_distances_bool=embed_distances_bool,
            radial_basis_bool=radial_basis_bool,
            num_radial_basis=num_radial_basis,
            max_frequency=max_frequency
        )

        self.layers = nn.ModuleList([
            DiTLayer(
                num_features=sphere_channels,
                num_heads=num_heads,
                num_features_mlp=sphere_channels_mlp,
                activation_fn=activation_fn_mlp,
                edge_dim=sphere_channels
            )
            for _ in range(num_layers)
        ])

        self.readout = SimpleReadout(
            num_features=sphere_channels,
            activation_fn=activation_fn_mlp
        )
        
    def forward(self, data):
        # compute graph connectivity
        idx_j, idx_i = radius_graph(
            x=data.pos,
            r=self.cutoff,
            batch=data.batch,
            max_num_neighbors=self.max_neighbors
        )
        
        data.edge_index = torch.stack([idx_j, idx_i], dim=0)
        
        h = self.node_embed(data)
        edge_attr = self.edge_embed(data)
        
        for layer in self.layers:
            h = layer(
                x_nodes=h,
                edge_index=data.edge_index,
                x_cond=None,
                edge_attr=edge_attr
            )
            
        out = self.readout(
            features_nodes=h,
            features_cond=None
        )
        
        out = batch_center_systems(out, data.batch)
        return out