# taken from: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/egnn/egnn_new.py
import torch
from torch import nn
from torch_geometric.nn import radius_graph

from symdrift.model.painn import GaussianRBF, TimestepEmbedder
from symdrift.generative import batch_center_systems

__all__ = ["EGNN"]


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(
    data, segment_ids, num_segments, normalization_factor, aggregation_method: str
):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == "sum":
        result = result / normalization_factor

    if aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


class GCL(nn.Module):
    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.SiLU(),
        attention=False,
    ):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(
            edge_attr,
            row,
            num_segments=x.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(
        self,
        h,
        edge_index,
        edge_attr=None,
        node_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(
        self,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=1,
        act_fn=nn.SiLU(),
        tanh=False,
        coords_range=10.0,
    ):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
        )
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = (
                coord_diff
                * torch.tanh(self.coord_mlp(input_tensor))
                * self.coords_range
            )
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(
            trans,
            row,
            num_segments=coord.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        coord = coord + agg
        return coord

    def forward(
        self,
        h,
        coord,
        edge_index,
        coord_diff,
        edge_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(
        self,
        hidden_nf,
        edge_feat_nf=2 * 64,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=2,
        attention=True,
        norm_diff=True,
        tanh=False,
        coords_range=15,
        norm_constant=1,
        num_distance_basis=64,
        max_radius=5.0,
        normalization_factor=100,
        aggregation_method="sum",
    ):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant

        if num_distance_basis > 0:
            self.radial_basis = GaussianRBF(
                n_rbf=num_distance_basis, cutoff=max_radius, trainable=False
            )
        else:
            self.radial_basis = None

        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=edge_feat_nf,
                    act_fn=act_fn,
                    attention=attention,
                    normalization_factor=self.normalization_factor,
                    aggregation_method=self.aggregation_method,
                ),
            )
        self.add_module(
            "gcl_equiv",
            EquivariantUpdate(
                hidden_nf,
                edges_in_d=edge_feat_nf,
                act_fn=nn.SiLU(),
                tanh=tanh,
                coords_range=self.coords_range_layer,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
            ),
        )
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        # Embedding of distances
        if self.radial_basis is not None:
            distances = self.radial_basis(distances).squeeze(1)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](
                h,
                edge_index,
                edge_attr=edge_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
        x = self._modules["gcl_equiv"](
            h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask
        )

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(nn.Module):
    def __init__(
        self,
        sphere_channels=128,
        in_edge_nf=2,
        max_radius=5.0,
        device="cpu",
        act_fn=nn.SiLU(),
        num_layers=3,
        attention=False,
        norm_diff=True,
        tanh=False,
        coords_range=15,
        norm_constant=1,
        inv_sublayers=2,
        num_distance_basis=64,
        normalization_factor=100,
        aggregation_method="sum",
        max_neighbors=32,
        out_node_nf=None,
        use_noise_schedule_sigma_encoding=False,
        **kwargs,
    ):
        super(EGNN, self).__init__()
        self.hidden_nf = sphere_channels
        self.cutoff = max_radius
        self.max_neighbors = max_neighbors
        self.device = device
        self.n_layers = num_layers
        self.coords_range_layer = float(coords_range / num_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.use_noise_schedule_sigma_encoding = use_noise_schedule_sigma_encoding

        if num_distance_basis > 0:
            self.radial_basis = GaussianRBF(
                n_rbf=num_distance_basis, cutoff=self.cutoff, trainable=False
            )
            edge_feat_nf = num_distance_basis * in_edge_nf
        else:
            edge_feat_nf = in_edge_nf
            self.radial_basis = None

        self.embedding = nn.Embedding(100, sphere_channels)
        for i in range(0, num_layers):
            self.add_module(
                "e_block_%d" % i,
                EquivariantBlock(
                    sphere_channels,
                    edge_feat_nf=edge_feat_nf,
                    device=device,
                    act_fn=act_fn,
                    n_layers=inv_sublayers,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    coords_range=coords_range,
                    norm_constant=norm_constant,
                    max_radius=max_radius,
                    num_distance_basis=num_distance_basis,
                    normalization_factor=self.normalization_factor,
                    aggregation_method=self.aggregation_method,
                ),
            )
        self.h_out_mlp = None
        self.out_node_nf = out_node_nf
        if out_node_nf is not None:
            self.h_out_mlp = nn.Sequential(
                nn.Linear(sphere_channels, sphere_channels),
                act_fn,
                nn.Linear(sphere_channels, out_node_nf),
            )
            
        if self.use_noise_schedule_sigma_encoding:
            self.noise_schedule_sigma_embedding = TimestepEmbedder(
                hidden_size=self.hidden_nf,
                frequency_embedding_size=256,
            )
        
        self.to(self.device)

    def forward(self, data, **kwargs):

        node_mask = getattr(data, "node_mask", None)
        edge_mask = getattr(data, "edge_mask", None)
        atomic_numbers = data.x.long()
        x = data.pos

        # compute graph connectivity
        idx_j, idx_i = radius_graph(
            x=data.pos,
            r=self.cutoff,
            batch=data.batch,
            max_num_neighbors=self.max_neighbors,
        )
        data.edge_index = torch.stack([idx_j, idx_i], dim=0)

        # EGNN definition works with edge index where the receiver is the first index,
        # but radius_graph returns sender-receiver order, so we swap them here
        edge_index = torch.stack([idx_i, idx_j], dim=0)
        
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)

        # Embedding of distances
        if self.radial_basis is not None:
            distances = self.radial_basis(distances).squeeze(1)

        h = self.embedding(atomic_numbers)
        
        # noise schedule sigma encoding
        if self.use_noise_schedule_sigma_encoding:
            noise_schedule_sigma_enbedding = self.noise_schedule_sigma_embedding(data.t)
            h = h + noise_schedule_sigma_enbedding
            
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h,
                x,
                edge_index,
                node_mask=node_mask,
                edge_mask=edge_mask,
                edge_attr=distances,
            )

        x = batch_center_systems(x, data.batch, dim=0)

        # in case we want to output node features in addition to coordinates
        if self.h_out_mlp is not None:
            h = self.h_out_mlp(h)
            return h, x

        return x
