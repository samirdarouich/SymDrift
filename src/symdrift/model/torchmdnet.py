import math
import warnings
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from symdrift.model.utils import extend_bond_index, signed_volume
from symdrift.generative import batch_center_systems


class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff_lower, cutoff_upper, max_z=100):
        super().__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(self.cutoff_lower, self.cutoff_upper, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
        vector_output=False,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.vector_output = vector_output
        self.layer_norm = layer_norm

        proj_out_channels = out_channels
        if self.vector_output:
            proj_out_channels = 1

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, proj_out_channels, bias=False)

        act_class = act_class_mapping[activation]

        if vector_output:
            self.update_net = nn.Sequential(
                nn.Linear(hidden_channels * 2, intermediate_channels),
                act_class(),
                nn.Linear(intermediate_channels, out_channels + 1),
            )
        else:
            self.update_net = nn.Sequential(
                nn.Linear(hidden_channels * 2, intermediate_channels),
                act_class(),
                nn.Linear(intermediate_channels, out_channels * 2),
            )
        if layer_norm:
            # add a layer norm after first linear layer
            self.update_net = nn.Sequential(
                self.update_net[0],
                nn.LayerNorm(intermediate_channels),
                self.update_net[1],
                self.update_net[2],
            )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[-1].weight)
        self.update_net[-1].bias.data.fill_(0)

    def forward(self, x, v):
        vec1_buffer = self.vec1_proj(v)

        # detach zero-entries to avoid NaN gradients during force loss backpropagation
        vec1 = torch.zeros(
            vec1_buffer.size(0), vec1_buffer.size(2), device=vec1_buffer.device
        )

        # mask = (vec1_buffer != 0).view(vec1_buffer.size(0), -1).any(dim=1)
        mask = (vec1_buffer != 0).view(vec1_buffer.size(0), -1).all(dim=1)
        if not mask.all():
            warnings.warn(
                (
                    f"Skipping gradients for {(~mask).sum()} atoms due to "
                    "vector features being zero. This is likely due to atoms "
                    "being outside the cutoff radius of any other atom. "
                    "These atoms will not interact with any other atom "
                    "unless you change the cutoff."
                )
            )
        vec1[mask] = torch.norm(vec1_buffer[mask], dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)

        if self.vector_output:
            out = self.update_net(x)
            x, v = out[:, : self.out_channels], out[:, self.out_channels :]
        else:
            x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)

        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v


rbf_class_mapping = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}

act_class_mapping = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.0):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors_feat: Tensor):
        """Coordinate features Normalization"""
        # shape of coors_feat: (num_atoms, 3, hidden_channels)
        norm = coors_feat.norm(dim=1, keepdim=True)
        normed_coors = coors_feat / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class OutputModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, allow_prior_model, reduce_op):
        super().__init__()
        self.allow_prior_model = allow_prior_model
        self.reduce_op = reduce_op

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        return

    def reduce(self, x, batch):
        return scatter(x, batch, dim=0, reduce=self.reduce_op)

    def post_reduce(self, x):
        return x


class Scalar(OutputModel):
    def __init__(
        self,
        hidden_channels,
        activation="silu",
        allow_prior_model=True,
        reduce_op="sum",
    ):
        super().__init__(allow_prior_model=allow_prior_model, reduce_op=reduce_op)
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        return self.output_network(x)


class EquivariantVectorOutput(OutputModel):
    def __init__(
        self,
        hidden_channels,
        activation="silu",
        reduce_op="sum",
        layer_norm: bool = False,
    ):
        super(EquivariantVectorOutput, self).__init__(
            allow_prior_model=False, reduce_op="sum"
        )

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                    layer_norm=layer_norm,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2,
                    hidden_channels,
                    activation=activation,
                    vector_output=True,
                    layer_norm=layer_norm,
                ),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)

        v = v.squeeze() + pos

        return x, v


class EquivariantVectorAndScalarOutput(OutputModel):
    def __init__(self, hidden_channels, activation="silu", reduce_op="sum"):
        super(EquivariantVectorAndScalarOutput, self).__init__(
            allow_prior_model=False, reduce_op="sum"
        )

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2,
                    hidden_channels,
                    activation=activation,
                    vector_output=True,
                ),
            ]
        )
        self.scalar_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class_mapping[activation](),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        x = self.scalar_network(x)
        v = v.squeeze() + pos
        return x, v


class EquivariantMultiHeadAttention(MessagePassing):
    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        distance_influence: str,
        num_heads: int,
        activation: str,
        attn_activation: str,
        cutoff_lower: float,
        cutoff_upper: float,
        node_attr_dim: int = 0,
        qk_norm: bool = False,
        norm_coors: bool = False,
        norm_coors_scale_init: float = 1e-2,
        so3_equivariant: bool = False,
        use_time_embedding: bool = False,
    ):
        super(EquivariantMultiHeadAttention, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.so3_equivariant = so3_equivariant
        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.node_attr_dim = node_attr_dim
        self.norm_coors = norm_coors  # boolean
        self.coors_norm = (
            CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()
        )
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.qk_norm = qk_norm
        self.use_time_embedding = use_time_embedding

        input_channels = (
            hidden_channels
            + (1 if use_time_embedding else 0)
            + (hidden_channels if node_attr_dim > 0 else 0)
        )
        self.mixing_mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        if qk_norm:
            # add layer norm to q and k projections
            # based on https://arxiv.org/pdf/2302.05442.pdf
            self.q_proj = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )
            self.k_proj = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )
        else:
            self.q_proj = nn.Linear(hidden_channels, hidden_channels)
            self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(
            hidden_channels, hidden_channels * (3 + int(so3_equivariant))
        )
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        # projection linear layers for edge attributes
        self.dk_proj = nn.Linear(num_rbf, hidden_channels)
        self.dv_proj = nn.Linear(num_rbf, hidden_channels * (3 + int(so3_equivariant)))

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        if self.qk_norm:
            self.q_proj[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.q_proj[0].weight)
            self.k_proj[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.k_proj[0].weight)
        else:
            self.q_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.q_proj.weight)
            self.k_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.k_proj.weight)

        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij, node_attr, t: Optional[Tensor] = None):

        # Mix x with node_attr (and raw time t, if enabled)
        if self.use_time_embedding:
            assert t is not None, "Time embedding is enabled but time tensor t is None."
            x = self.mixing_mlp(torch.cat([x, t, node_attr], dim=1))
        else:
            x = self.mixing_mlp(torch.cat([x, node_attr], dim=1))

        # Input features: (num_atoms, hidden_channels)
        x = self.layernorm(x)
        # key/query features: (num_atoms, num_heads, head_dim)
        # where head_dim * num_heads == hidden_channels
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        # value features: (num_atoms, num_heads, 3 * head_dim)
        v = self.v_proj(x).reshape(
            -1, self.num_heads, self.head_dim * (3 + int(self.so3_equivariant))
        )

        # vec features: (num_atoms, 3, hidden_channels) (all invariant)
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        # transform edge attributes (relative distances and user provided edge attributes)
        # into dk and dv vectors with shape (num_edges, num_heads, head_dim)
        # and (num_edges, num_heads, 3 * head_dim) respectively
        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(
            -1, self.num_heads, self.head_dim * (3 + int(self.so3_equivariant))
        )

        # Message Passing Propagate
        x, vec = self.propagate(
            edge_index,  # (2, edges)
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        # new shape: (num_atoms, hidden_channels)
        x = x.reshape(-1, self.hidden_channels)
        # new shape: (num_atoms, 3, hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)
        # normalize the vec if norm_coors is True
        vec = self.coors_norm(vec)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dvec = vec3 * o1.unsqueeze(1) + vec
        dx = vec_dot * o2 + o3
        return dx, dvec

    def message(
        self,
        q_i: Tensor,  # (num_edges, num_heads, head_dim)
        k_j: Tensor,  # (num_edges, num_heads, head_dim)
        v_j: Tensor,  # (num_edges, num_heads, head_dim * 3)
        vec_j: Tensor,  # (num_edges, 3, num_heads, head_dim)
        dk: Tensor,  # (num_edges, num_heads, head_dim)
        dv: Tensor,  # (num_edges, num_heads, head_dim * 3)
        r_ij: Tensor,  # (num_edges,) edge distances
        d_ij: Tensor,  # (num_edges, 3) edge vectors (unit vectors)
    ):
        # dot product attention, a score for each edge
        attn = (q_i * k_j * dk).sum(dim=-1)  # (num_edges, num_heads)

        # apply attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        # value pathway
        v_j = v_j * dv  # multiply with edge attr features

        if self.so3_equivariant:
            x, vec1, vec2, vec3 = torch.split(v_j, self.head_dim, dim=2)
        else:
            x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)
            vec3 = None

        # update scalar features
        x = x * attn.unsqueeze(2)  # (num_edges, num_heads, head_dim)
        # update vector features (num_edges, 3, num_heads, head_dim)
        if self.so3_equivariant:
            vec = (
                vec_j * vec1.unsqueeze(1)
                + vec2.unsqueeze(1) * d_ij.unsqueeze(2).unsqueeze(3)
                + vec3.unsqueeze(1)
                * torch.cross(d_ij.unsqueeze(2).unsqueeze(3), vec_j, dim=1)
            )
        else:
            vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(
                2
            ).unsqueeze(3)
        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        # scatter edge-level features (for x and vec) to node-level
        # x shape: (num_atoms, num_heads, head_dim)
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        # vec shape: (num_atoms, 3, num_heads, head_dim)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class TorchMD_ET_dynamics(nn.Module):
    r"""The TorchMD equivariant Transformer architecture.

    Parameters
    ----------
    hidden_channels (int, optional): Hidden embedding size.
        (default: :obj:`128`)
    num_layers (int, optional): The number of attention layers.
        (default: :obj:`6`)
    num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
        (default: :obj:`50`)
    rbf_type (string, optional): The type of radial basis function to use.
        (default: :obj:`"expnorm"`)
    trainable_rbf (bool, optional): Whether to train RBF parameters with
        backpropagation. (default: :obj:`True`)
    activation (string, optional): The type of activation function to use.
        (default: :obj:`"silu"`)
    attn_activation (string, optional): The type of activation function to use
        inside the attention mechanism. (default: :obj:`"silu"`)
    neighbor_embedding (bool, optional): Whether to perform an initial neighbor
        embedding step. (default: :obj:`True`)
    num_heads (int, optional): Number of attention heads.
        (default: :obj:`8`)
    distance_influence (string, optional): Where distance information is used inside
        the attention mechanism. (default: :obj:`"both"`)
    cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
        (default: :obj:`0.0`)
    cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
        (default: :obj:`5.0`)
    max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
        (default: :obj:`100`)
    qk_norm (bool, optional):
        Applies layer norm to q and k projections. Supposed to
        stabilize the training based on
        https://arxiv.org/pdf/2302.05442.pdf. (default: :obj:`False`)
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_rbf: int = 50,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = True,
        activation: str = "silu",
        attn_activation: str = "silu",
        neighbor_embedding: bool = True,
        num_heads: int = 8,
        distance_influence: str = "both",
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 10.0,
        max_z: int = 100,
        node_attr_dim: int = 0,
        edge_attr_dim: int = 0,
        qk_norm: bool = False,
        norm_coors: bool = False,
        norm_coors_scale_init: float = 1e-2,
        clip_during_norm: bool = False,
        so3_equivariant: bool = False,
        use_time_embedding: bool = False,
    ):
        super(TorchMD_ET_dynamics, self).__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f"Choose from {', '.join(rbf_class_mapping.keys())}."
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f"Choose from {', '.join(act_class_mapping.keys())}."
        )
        assert attn_activation in act_class_mapping, (
            f'Unknown attention activation function "{attn_activation}". '
            f"Choose from {', '.join(act_class_mapping.keys())}."
        )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.node_attr_dim = node_attr_dim
        self.edge_attr_dim = edge_attr_dim
        self.clip_during_norm = clip_during_norm
        self.use_time_embedding = use_time_embedding

        act_class = act_class_mapping[activation]

        self.embedding = nn.Embedding(self.max_z, self.hidden_channels)

        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels,
                num_rbf + edge_attr_dim,
                cutoff_lower,
                cutoff_upper,
                self.max_z,
            )
            if neighbor_embedding
            else None
        )

        if self.node_attr_dim > 0:
            self.node_mlp = nn.Sequential(
                nn.Linear(node_attr_dim, hidden_channels),
                act_class(),
                nn.LayerNorm(hidden_channels),
                nn.Linear(hidden_channels, hidden_channels),
            )

        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = EquivariantMultiHeadAttention(
                hidden_channels,
                num_rbf + edge_attr_dim,
                distance_influence,
                num_heads,
                act_class,
                attn_activation,
                cutoff_lower,
                cutoff_upper,
                node_attr_dim=node_attr_dim,
                qk_norm=qk_norm,
                norm_coors=norm_coors,
                norm_coors_scale_init=norm_coors_scale_init,
                so3_equivariant=so3_equivariant,
                use_time_embedding=use_time_embedding,
            )  # .jittable() TODO: Removing for now
            self.attention_layers.append(layer)

        self.out_norm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        edge_index,
        node_attr: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        # embed atomic numbers using an embedding layer
        if z.dim() > 1:
            z = z.squeeze()  # (num_atoms,)
        x = self.embedding(z)  # (num_atoms, hidden_channels)

        # append time to node features
        if self.node_attr_dim > 0:
            node_attr = self.node_mlp(node_attr)
        else:
            node_attr = None

        # compute distances
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        edge_weight = (edge_vec**2).sum(dim=-1, keepdim=False)

        # update edge_attributes with user input if they are given
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)  # (num_edges, 1)
            # (num_edges, num_rbf + edge_attr_dim)
            edge_attr = torch.cat(
                [self.distance_expansion(edge_weight), edge_attr], dim=-1
            )
        else:
            edge_attr = self.distance_expansion(edge_weight)

        mask = edge_index[0] == edge_index[1]
        masked_edge_weight = edge_weight.masked_fill(mask, 1).unsqueeze(1)

        if self.clip_during_norm:
            # clip edge_weight to avoid exploding values if two nodes are close
            masked_edge_weight = masked_edge_weight.clamp(min=1.0e-2)

        edge_vec = edge_vec / masked_edge_weight

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        # vec here is invariant values, we are not modifying the vectors.
        # (num_atoms, 3, hidden_channels)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
        for attn in self.attention_layers:
            dx, dvec = attn(
                x,
                vec,
                edge_index,
                edge_weight,
                edge_attr,
                edge_vec,
                node_attr=node_attr,
                t=t,
            )
            x = x + dx
            vec = vec + dvec
        x = self.out_norm(x)  # apply layer norm in the end.

        return x, vec, z, pos

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper})"
        )


class TorchMDDynamics(nn.Module):
    """
    TorchMDDynamics Model for DDPM training.

    Parameters
    ----------
    hidden_channels (int, optional):
        Hidden embedding size. (default: :obj:`128`)
    num_layers (int, optional):
        The number of attention layers. (default: :obj:`8`)
    num_rbf (int, optional):
        The number of radial basis functions :math:`\mu`.
        (default: :obj:`64`)
    rbf_type (string, optional):
        The type of radial basis function to use.
        (default: :obj:`"expnorm"`)
    trainable_rbf (bool, optional):
        Whether to train RBF parameters with backpropagation.
        (default: :obj:`False`)
    activation (string, optional):
        The type of activation function to use. (default: :obj:`"silu"`)
    neighbor_embedding (bool, optional):
        Whether to perform an initial neighbor embedding step.
        (default: :obj:`True`)
    cutoff_lower (float, optional):
        Lower cutoff distance for interatomic interactions.
        (default: :obj:`0.0`)
    cutoff_upper (float, optional):
        Upper cutoff distance for interatomic interactions.
        (default: :obj:`5.0`)
    max_z (int, optional):
        Maximum atomic number. Used for initializing embeddings.
        (default: :obj:`100`)
    node_attr_dim (int, optional):
        Dimension of additional input node  features (non-atomic numbers).
    attn_activation (string, optional):
        The type of activation function to use inside the attention
        mechanism. (default: :obj:`"silu"`)
    num_heads (int, optional):
        Number of attention heads. (default: :obj:`8`)
    distance_influence (string, optional):
        Where distance information is used inside the attention
        mechanism. (default: :obj:`"both"`)
    qk_norm (bool, optional):
        Applies layer norm to q and k projections. Supposed to
        stabilize the training based on
        https://arxiv.org/pdf/2302.05442.pdf. (default: :obj:`False`)
    """

    def __init__(
        self,
        sphere_channels: int = 128,
        num_layers: int = 8,
        num_distance_basis: int = 64,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = True,
        activation: str = "silu",
        neighbor_embedding: int = True,
        cutoff_lower: float = 0.0,
        max_radius: float = 10.0,
        max_z: int = 100,
        node_attr_dim: int = 10,
        edge_attr_dim: int = 1,
        attn_activation: str = "silu",
        num_heads: int = 8,
        distance_influence: str = "both",
        reduce_op: str = "sum",
        qk_norm: bool = True,
        output_layer_norm: bool = True,
        clip_during_norm: bool = True,
        so3_equivariant: bool = False,
        max_neighbors: int = 32,
        # make edge_type one_hot
        edge_one_hot: bool = False,
        edge_one_hot_types: int = 5,
        parity_switch=False,
        use_time_embedding: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.cutoff = max_radius
        self.max_neighbors = max_neighbors
        self.edge_one_hot = edge_one_hot
        self.edge_one_hot_types = edge_one_hot_types
        self.parity_switch = parity_switch
        self.use_time_embedding = use_time_embedding
        self.representation_model = TorchMD_ET_dynamics(
            hidden_channels=sphere_channels,
            num_layers=num_layers,
            num_rbf=num_distance_basis,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            neighbor_embedding=neighbor_embedding,
            cutoff_lower=cutoff_lower,
            cutoff_upper=max_radius,
            max_z=max_z,
            attn_activation=attn_activation,
            num_heads=num_heads,
            distance_influence=distance_influence,
            node_attr_dim=node_attr_dim,
            edge_attr_dim=edge_attr_dim,
            qk_norm=qk_norm,
            clip_during_norm=clip_during_norm,
            so3_equivariant=so3_equivariant,
            use_time_embedding=use_time_embedding,
        )
        self.output_model = EquivariantVectorOutput(
            hidden_channels=sphere_channels,
            activation=activation,
            reduce_op=reduce_op,
            layer_norm=output_layer_norm,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()

    def forward(self, data) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass over torchmd-net model.

        Parameters
        ----------
        data object containing:
        x: torch.Tensor
            Atomic numbers, shape (num_atoms,)
        pos: torch.Tensor
            Atomic positions, shape (num_atoms, 3)
        edge_index: torch.Tensor
            Edge index, shape (2, num_edges)
        batch: torch.Tensor, optional
            Batch vector representing which atoms belong to which molecule,
            shape (num_atoms,). If not given, all atoms are assumed to belong
            to the same molecule.
        edge_attr: torch.Tensor, optional
            Edge attributes, shape (num_edges, edge_attr_dim)
        node_attr: torch.Tensor, optional
            Node attributes, shape (num_atoms, node_attr_dim)
        """

        edge_index, edge_type = extend_bond_index(
            pos=data.pos,
            batch=data.batch,
            bond_index=data.get("bonded_edge_index", None),
            bond_attr=data.get("edge_attr", None),
            one_hot=self.edge_one_hot,
            one_hot_types=self.edge_one_hot_types,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
        )

        # If wanted add time embedding to node features.
        t = None
        if self.use_time_embedding:
            t = data.get("t", None)
            if t is None:
                t = torch.zeros(
                    data.pos.size(0), 
                    1, 
                    device=data.pos.device, 
                    dtype=data.pos.dtype
                )
            elif t.dim() == 1:
                t = t.unsqueeze(-1)

        # run the potentially wrapped representation model
        x, v, z, pos = self.representation_model(
            z=data.x.long(),
            pos=data.pos,
            node_attr=data.get("node_attr", None),
            edge_index=edge_index,
            edge_attr=edge_type,
            t=t,
        )

        # latent representation
        _, v = self.output_model.pre_reduce(x, v, z, pos, data.batch)
        v = batch_center_systems(v - pos, data.batch)

        # Switch parity of positions for chiral molecules during inference if wanted.
        if self.parity_switch and not self.training:
            v = self.switch_parity_of_pos(
                v, data.chiral_index, data.chiral_nbr_index, data.chiral_tag, data.batch
            )
        return v

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
