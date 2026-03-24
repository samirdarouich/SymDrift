import math
from typing import Union

import torch
from torch import device, einsum, nn
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from tspath.model.utils import extend_bond_index
from tspath.utils import batch_center_systems

__all__ = [
    "GVP",
    "GVPDropout",
    "GVPLayerNorm",
    "GVPConv",
    "NodePositionUpdate",
    "EdgeUpdate",
    "GVPModel",
]


# helper functions
def exists(val):
    return val is not None


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


# the classes GVP, GVPDropout, and GVPLayerNorm are taken from lucidrains' geometric-vector-perceptron repository
# https://github.com/lucidrains/geometric-vector-perceptron/tree/main
# some adaptations have been made to these classes to make them more consistent with the original GVP paper/implementation
# specifically, using _norm_no_nan instead of torch's built in norm function, and the weight intialiation scheme for Wh and Wu


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    device = D.device
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


class GVP(nn.Module):
    def __init__(
        self,
        dim_vectors_in,
        dim_vectors_out,
        dim_feats_in,
        dim_feats_out,
        n_cp_feats=0,  # number of cross-product features added to hidden vector features
        hidden_vectors=None,
        feats_activation=nn.SiLU(),
        vectors_activation=nn.Sigmoid(),
        vector_gating=True,
        xavier_init=False,
    ):
        super().__init__()
        self.dim_vectors_in = dim_vectors_in
        self.dim_feats_in = dim_feats_in
        self.n_cp_feats = n_cp_feats

        self.dim_vectors_out = dim_vectors_out
        dim_h = (
            max(dim_vectors_in, dim_vectors_out)
            if hidden_vectors is None
            else hidden_vectors
        )

        # create Wh matrix
        wh_k = 1 / math.sqrt(dim_vectors_in)
        self.Wh = torch.zeros(dim_vectors_in, dim_h, dtype=torch.float32).uniform_(
            -wh_k, wh_k
        )
        self.Wh = nn.Parameter(self.Wh)

        # create Wcp matrix if we are using cross-product features
        if n_cp_feats > 0:
            wcp_k = 1 / math.sqrt(dim_vectors_in)
            self.Wcp = torch.zeros(
                dim_vectors_in, n_cp_feats * 2, dtype=torch.float32
            ).uniform_(-wcp_k, wcp_k)
            self.Wcp = nn.Parameter(self.Wcp)

        # create Wu matrix
        if (
            n_cp_feats > 0
        ):  # the number of vector features going into Wu is increased by n_cp_feats if we are using cross-product features
            wu_in_dim = dim_h + n_cp_feats
        else:
            wu_in_dim = dim_h
        wu_k = 1 / math.sqrt(wu_in_dim)
        self.Wu = torch.zeros(wu_in_dim, dim_vectors_out, dtype=torch.float32).uniform_(
            -wu_k, wu_k
        )
        self.Wu = nn.Parameter(self.Wu)

        self.vectors_activation = vectors_activation

        self.to_feats_out = nn.Sequential(
            nn.Linear(dim_h + n_cp_feats + dim_feats_in, dim_feats_out),
            feats_activation,
        )

        # branching logic to use old GVP, or GVP with vector gating
        if vector_gating:
            self.scalar_to_vector_gates = nn.Linear(dim_feats_out, dim_vectors_out)
            if xavier_init:
                nn.init.xavier_uniform_(self.scalar_to_vector_gates.weight, gain=1)
                nn.init.constant_(self.scalar_to_vector_gates.bias, 0)
        else:
            self.scalar_to_vector_gates = None

        # self.scalar_to_vector_gates = nn.Linear(dim_feats_out, dim_vectors_out) if vector_gating else None

    def forward(self, data):
        feats, vectors = data
        b, n, _, v, c = *feats.shape, *vectors.shape

        # feats has shape (batch_size, n_feats)
        # vectors has shape (batch_size, n_vectors, 3)

        assert c == 3 and v == self.dim_vectors_in, "vectors have wrong dimensions"
        assert n == self.dim_feats_in, "scalar features have wrong dimensions"

        Vh = einsum(
            "b v c, v h -> b h c", vectors, self.Wh
        )  # has shape (batch_size, dim_h, 3)

        # if we are including cross-product features, compute them here
        if self.n_cp_feats > 0:
            # convert dim_vectors_in vectors to n_cp_feats*2 vectors
            Vcp = einsum(
                "b v c, v p -> b p c", vectors, self.Wcp
            )  # has shape (batch_size, n_cp_feats*2, 3)
            # split the n_cp_feats*2 vectors into two sets of n_cp_feats vectors
            cp_src, cp_dst = torch.split(
                Vcp, self.n_cp_feats, dim=1
            )  # each has shape (batch_size, n_cp_feats, 3)
            # take the cross product of the two sets of vectors
            cp = torch.linalg.cross(
                cp_src, cp_dst, dim=-1
            )  # has shape (batch_size, n_cp_feats, 3)

            # add the cross product features to the hidden vector features
            Vh = torch.cat(
                (Vh, cp), dim=1
            )  # has shape (batch_size, dim_h + n_cp_feats, 3)

        Vu = einsum(
            "b h c, h u -> b u c", Vh, self.Wu
        )  # has shape (batch_size, dim_vectors_out, 3)

        sh = _norm_no_nan(Vh)

        s = torch.cat((feats, sh), dim=1)

        feats_out = self.to_feats_out(s)

        if exists(self.scalar_to_vector_gates):
            gating = self.scalar_to_vector_gates(feats_out)
            gating = gating.unsqueeze(dim=-1)
        else:
            gating = _norm_no_nan(Vu)

        vectors_out = self.vectors_activation(gating) * Vu

        # if torch.isnan(feats_out).any() or torch.isnan(vectors_out).any():
        #     raise ValueError("NaNs in GVP forward pass")

        return (feats_out, vectors_out)


class _VDropout(nn.Module):
    """
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    """

    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class GVPDropout(nn.Module):
    """Separate dropout for scalars and vectors."""

    def __init__(self, rate):
        super().__init__()
        self.vector_dropout = _VDropout(rate)
        self.feat_dropout = nn.Dropout(rate)

    def forward(self, feats, vectors):
        return self.feat_dropout(feats), self.vector_dropout(vectors)


class GVPLayerNorm(nn.Module):
    """Normal layer norm for scalars, nontrainable norm for vectors."""

    def __init__(self, feats_h_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.feat_norm = nn.LayerNorm(feats_h_size)

    def forward(self, feats, vectors):

        normed_feats = self.feat_norm(feats)

        vn = _norm_no_nan(vectors, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True) + self.eps) + self.eps
        normed_vectors = vectors / vn
        return normed_feats, normed_vectors


class GVPConv(nn.Module):
    """GVP graph convolution on a homogenous graph."""

    def __init__(
        self,
        scalar_size: int = 128,
        vector_size: int = 16,
        n_cp_feats: int = 0,
        scalar_activation=nn.SiLU,
        vector_activation=nn.Sigmoid,
        n_message_gvps: int = 1,
        n_update_gvps: int = 1,
        use_dst_feats: bool = False,
        rbf_dmax: float = 20,
        rbf_dim: int = 16,
        edge_feat_size: int = 0,
        message_norm: Union[float, str] = 10,
        dropout: float = 0.0,
    ):

        super().__init__()

        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.n_cp_feats = n_cp_feats
        self.scalar_activation = scalar_activation
        self.vector_activation = vector_activation
        self.n_message_gvps = n_message_gvps
        self.n_update_gvps = n_update_gvps
        self.edge_feat_size = edge_feat_size
        self.use_dst_feats = use_dst_feats
        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim
        self.dropout_rate = dropout
        self.message_norm = message_norm

        # create message passing function
        message_gvps = []
        for i in range(n_message_gvps):
            dim_vectors_in = vector_size
            dim_feats_in = scalar_size

            # on the first layer, there is an extra edge vector for the displacement vector between the two node positions
            if i == 0:
                dim_vectors_in += 1
                dim_feats_in += rbf_dim + edge_feat_size

            # if this is the first layer and we are using destination node features to compute messages, add them to the input dimensions
            if use_dst_feats and i == 0:
                dim_vectors_in += vector_size
                dim_feats_in += scalar_size

            message_gvps.append(
                GVP(
                    dim_vectors_in=dim_vectors_in,
                    dim_vectors_out=vector_size,
                    n_cp_feats=n_cp_feats,
                    dim_feats_in=dim_feats_in,
                    dim_feats_out=scalar_size,
                    feats_activation=scalar_activation(),
                    vectors_activation=vector_activation(),
                    vector_gating=True,
                )
            )
        self.edge_message = nn.Sequential(*message_gvps)

        # create update function
        update_gvps = []
        for i in range(n_update_gvps):
            update_gvps.append(
                GVP(
                    dim_vectors_in=vector_size,
                    dim_vectors_out=vector_size,
                    n_cp_feats=n_cp_feats,
                    dim_feats_in=scalar_size,
                    dim_feats_out=scalar_size,
                    feats_activation=scalar_activation(),
                    vectors_activation=vector_activation(),
                    vector_gating=True,
                )
            )
        self.node_update = nn.Sequential(*update_gvps)

        self.dropout = GVPDropout(self.dropout_rate)
        self.message_layer_norm = GVPLayerNorm(self.scalar_size)
        self.update_layer_norm = GVPLayerNorm(self.scalar_size)

        if isinstance(self.message_norm, str) and self.message_norm not in [
            "mean",
            "sum",
        ]:
            raise ValueError(
                f"message_norm must be either 'mean', 'sum', or a number, got {self.message_norm}"
            )
        else:
            assert isinstance(self.message_norm, (float, int)), (
                "message_norm must be either 'mean', 'sum', or a number"
            )

    def forward(
        self,
        scalar_feats: torch.Tensor,
        coord_feats: torch.Tensor,
        vec_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor = None,
        x_diff: torch.Tensor = None,
        d: torch.Tensor = None,
    ):
        # vec_feat has shape (n_nodes, n_vectors, 3)
        src, dst = edge_index

        # normalize x_diff and compute rbf embedding of edge distance
        if x_diff is None:
            # relative displacement
            x_diff = coord_feats[src] - coord_feats[dst]
            dij = _norm_no_nan(x_diff, keepdims=True) + 1e-8
            x_diff = x_diff / dij
            d = _rbf(dij.squeeze(-1), D_max=self.rbf_dmax, D_count=self.rbf_dim)

        scalar_msg, vec_msg = self.message(
            h=scalar_feats,
            v=vec_feats,
            x_diff=x_diff,
            d=d,
            edge_index=edge_index,
            a=edge_feats,
        )

        # aggregate
        if isinstance(self.message_norm, (float, int)):
            scalar_msg = scalar_msg / self.message_norm
            vec_msg = vec_msg / self.message_norm
        else:
            scalar_msg = scatter(scalar_msg, dst, dim=0, reduce=self.message_norm)
            vec_msg = scatter(vec_msg, dst, dim=0, reduce=self.message_norm)
            

        # dropout scalar and vector messages
        scalar_msg, vec_msg = self.dropout(scalar_msg, vec_msg)

        # update scalar and vector features, apply layernorm
        scalar_feat = scalar_feats + scalar_msg
        vec_feat = vec_feats + vec_msg
        scalar_feat, vec_feat = self.message_layer_norm(scalar_feat, vec_feat)

        # apply node update function, apply dropout to residuals, apply layernorm
        scalar_residual, vec_residual = self.node_update((scalar_feat, vec_feat))
        scalar_residual, vec_residual = self.dropout(scalar_residual, vec_residual)
        scalar_feat = scalar_feat + scalar_residual
        vec_feat = vec_feat + vec_residual
        scalar_feat, vec_feat = self.update_layer_norm(scalar_feat, vec_feat)

        return scalar_feat, vec_feat

    def message(self, h, v, x_diff, d, edge_index, a=None):

        src, dst = edge_index

        # concatenate x_diff and v on every edge to produce vector features
        vec_feats = [x_diff.unsqueeze(1), v[src]]
        if self.use_dst_feats:
            vec_feats.append(v[dst])
        vec_feats = torch.cat(vec_feats, dim=1)

        # create scalar features
        scalar_feats = [h[src], d]
        if self.edge_feat_size > 0:
            scalar_feats.append(a)

        if self.use_dst_feats:
            scalar_feats.append(h[dst])

        scalar_feats = torch.cat(scalar_feats, dim=1)

        scalar_message, vector_message = self.edge_message((scalar_feats, vec_feats))

        return scalar_message, vector_message


class NodePositionUpdate(nn.Module):
    def __init__(self, n_scalars, n_vec_channels, n_gvps: int = 3, n_cp_feats: int = 0):
        super().__init__()

        self.gvps = []
        for i in range(n_gvps):
            if i == n_gvps - 1:
                vectors_out = 1
                vectors_activation = nn.Identity()
            else:
                vectors_out = n_vec_channels
                vectors_activation = nn.Sigmoid()

            self.gvps.append(
                GVP(
                    dim_feats_in=n_scalars,
                    dim_feats_out=n_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=vectors_out,
                    n_cp_feats=n_cp_feats,
                    vectors_activation=vectors_activation,
                )
            )
        self.gvps = nn.Sequential(*self.gvps)

    def forward(
        self, scalars: torch.Tensor, positions: torch.Tensor, vectors: torch.Tensor
    ):
        _, vector_updates = self.gvps((scalars, vectors))
        return positions + vector_updates.squeeze(1)


class EdgeUpdate(nn.Module):
    def __init__(
        self, n_node_scalars, n_edge_feats, update_edge_w_distance=False, rbf_dim=16
    ):
        super().__init__()

        self.update_edge_w_distance = update_edge_w_distance

        input_dim = n_node_scalars * 2 + n_edge_feats
        if update_edge_w_distance:
            input_dim += rbf_dim

        self.edge_update_fn = nn.Sequential(
            nn.Linear(input_dim, n_edge_feats),
            nn.SiLU(),
            nn.Linear(n_edge_feats, n_edge_feats),
            nn.SiLU(),
        )

        self.edge_norm = nn.LayerNorm(n_edge_feats)

    def forward(self, node_scalars, edge_feats, d, edge_index):

        # get indicies of source and destination nodes
        src_idxs, dst_idxs = edge_index

        mlp_inputs = [
            node_scalars[src_idxs],
            node_scalars[dst_idxs],
            edge_feats,
        ]

        if self.update_edge_w_distance:
            mlp_inputs.append(d)

        edge_feats = self.edge_norm(
            edge_feats + self.edge_update_fn(torch.cat(mlp_inputs, dim=-1))
        )
        return edge_feats


class GVPModel(nn.Module):
    def __init__(
        self,
        sphere_channels: int = 256,
        n_hidden_edge_feats: int = 128,
        num_layers: int = 8,
        max_radius: float = 5.0,
        num_distance_basis: int = 32,
        n_bond_types: int = 5,
        n_vec_channels: int = 16,
        n_cp_feats: int = 4,
        n_recycles: int = 1,
        convs_per_update: int = 2,
        n_message_gvps: int = 3,
        n_update_gvps: int = 3,
        separate_mol_updaters: bool = False,
        message_norm: Union[float, str] = 100,
        update_edge_w_distance: bool = False,
        max_neighbors: int = 500,
    ):
        super().__init__()

        self.n_bond_types = n_bond_types
        self.n_hidden_scalars = sphere_channels
        self.n_hidden_edge_feats = n_hidden_edge_feats
        self.n_vec_channels = n_vec_channels
        self.message_norm = message_norm
        self.n_recycles = n_recycles
        self.separate_mol_updaters: bool = separate_mol_updaters
        self.convs_per_update = convs_per_update
        self.n_molecule_updates = num_layers

        self.rbf_dmax = max_radius
        self.rbf_dim = num_distance_basis
        self.max_neighbors = max_neighbors

        assert n_vec_channels >= 3, "n_vec_channels must be >= 3"

        self.scalar_embedding = nn.Sequential(
            nn.Embedding(100, sphere_channels),
            nn.SiLU(),
            nn.Linear(sphere_channels, sphere_channels),
            nn.SiLU(),
            nn.LayerNorm(sphere_channels),
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(n_bond_types, n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_edge_feats),
        )

        self.conv_layers = []
        for conv_idx in range(convs_per_update * num_layers):
            self.conv_layers.append(
                GVPConv(
                    scalar_size=sphere_channels,
                    vector_size=n_vec_channels,
                    n_cp_feats=n_cp_feats,
                    edge_feat_size=n_hidden_edge_feats,
                    n_message_gvps=n_message_gvps,
                    n_update_gvps=n_update_gvps,
                    message_norm=message_norm,
                    rbf_dmax=self.rbf_dmax,
                    rbf_dim=self.rbf_dim,
                )
            )
        self.conv_layers = nn.ModuleList(self.conv_layers)

        # create molecule update layers
        self.node_position_updaters = nn.ModuleList([])
        self.edge_updaters = nn.ModuleList([])
        if self.separate_mol_updaters:
            n_updaters = num_layers
        else:
            n_updaters = 1
        for _ in range(n_updaters):
            self.node_position_updaters.append(
                NodePositionUpdate(
                    sphere_channels, n_vec_channels, n_gvps=3, n_cp_feats=n_cp_feats
                )
            )
            self.edge_updaters.append(
                EdgeUpdate(
                    sphere_channels,
                    n_hidden_edge_feats,
                    update_edge_w_distance=update_edge_w_distance,
                    rbf_dim=self.rbf_dim,
                )
            )

        # self.node_output_head = nn.Sequential(
        #     nn.Linear(n_hidden_scalars, n_hidden_scalars),
        #     nn.SiLU(),
        #     nn.Linear(n_hidden_scalars, n_atom_types + n_charges),
        # )

        # self.to_edge_logits = nn.Sequential(
        #     nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
        #     nn.SiLU(),
        #     nn.Linear(n_hidden_edge_feats, n_bond_types),
        # )

    def precompute_distances(self, pos, edge_index):
        """Precompute the pairwise distances between all nodes in the graph."""

        src, dst = edge_index

        x_diff = pos[src] - pos[dst]
        dij = _norm_no_nan(x_diff, keepdims=True) + 1e-8
        x_diff = x_diff / dij
        d = _rbf(dij.squeeze(-1), D_max=self.rbf_dmax, D_count=self.rbf_dim)

        return x_diff, d

    def forward(
        self,
        data,
    ):

        edge_index, edge_type, _ = extend_bond_index(
            pos=data.pos,
            batch=data.batch,
            bond_index=data.get("bonded_edge_index", None),
            bond_attr=data.get("edge_attr", None),
            one_hot=self.edge_one_hot,
            one_hot_types=self.edge_one_hot_types,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
        )

        data.edge_index = edge_index

        # Get scalar embedding
        node_scalar_features = self.scalar_embedding(data.x.long())

        node_positions = data.pos
        num_nodes = node_positions.shape[0]

        # initialize the vector features for every node to be zeros
        node_vec_features = torch.zeros(
            (num_nodes, self.n_vec_channels, 3), device=data.x.device
        )

        # Initialize edge features
        edge_features = self.edge_embedding(edge_type)

        x_diff, d = self.precompute_distances(node_positions, data.edge_index)
        for recycle_idx in range(self.n_recycles):
            for conv_idx, conv in enumerate(self.conv_layers):
                # perform a single convolution which updates node scalar and vector features (but not positions)
                node_scalar_features, node_vec_features = conv(
                    scalar_feats=node_scalar_features,
                    coord_feats=node_positions,
                    vec_feats=node_vec_features,
                    edge_feats=edge_features,
                    edge_index=data.edge_index,
                    x_diff=x_diff,
                    d=d,
                )

                # every convs_per_update convolutions, update the node positions and edge features
                if conv_idx != 0 and (conv_idx + 1) % self.convs_per_update == 0:
                    if self.separate_mol_updaters:
                        updater_idx = conv_idx // self.convs_per_update
                    else:
                        updater_idx = 0

                    node_positions = self.node_position_updaters[updater_idx](
                        node_scalar_features, node_positions, node_vec_features
                    )

                    x_diff, d = self.precompute_distances(
                        node_positions, data.edge_index
                    )

                    edge_features = self.edge_updaters[updater_idx](
                        node_scalar_features,
                        edge_features,
                        d=d,
                        edge_index=data.edge_index,
                    )

        # predict final charges and atom type logits
        # node_scalar_features = self.node_output_head(node_scalar_features)
        # atom_type_logits = node_scalar_features[:, : self.n_atom_types]
        # if not self.exclude_charges:
        #     atom_charge_logits = node_scalar_features[:, self.n_atom_types :]

        # predict the final edge logits
        # ue_feats = edge_features[upper_edge_mask]
        # le_feats = edge_features[~upper_edge_mask]
        # edge_logits = self.to_edge_logits(ue_feats + le_feats)

        # project node positions back into zero-COM subspace
        node_positions = batch_center_systems(node_positions, data.batch, dim=0)

        return node_positions
