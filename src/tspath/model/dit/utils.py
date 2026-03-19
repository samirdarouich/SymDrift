import math

import torch
import torch.nn as nn


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

def get_activation_fn(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu":
        return nn.SiLU()
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation {name}")

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int = None,
        num_layers: int = 2,
        activation_fn: str = "identity",
        use_bias: bool = True,
        output_is_zero_at_init: bool = False,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.num_layers = num_layers
        self.activation_fn = activation_fn
        self.use_bias = use_bias
        self.output_is_zero_at_init = output_is_zero_at_init

        self.layers = nn.ModuleList()
        self.act = get_activation_fn(activation_fn)

        self._build()

    def _build(self):
        dims = [self.hidden_dim] * (self.num_layers - 1) + [self.out_dim]
        prev_dim = self.in_dim
        for n in range(self.num_layers):
            layer = nn.Linear(prev_dim, dims[n], bias=self.use_bias)

            # --- initialization ---
            if self.output_is_zero_at_init and n == self.num_layers - 1:
                nn.init.zeros_(layer.weight)
                if self.use_bias:
                    nn.init.zeros_(layer.bias)
            else:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
                if self.use_bias:
                    nn.init.zeros_(layer.bias)

            self.layers.append(layer)
            prev_dim = dims[n]

    def forward(self, x):
        for n, layer in enumerate(self.layers):
            x = layer(x)
            # no activation on last layer
            if n < self.num_layers - 1:
                x = self.act(x)
        return x

class SimpleReadout(nn.Module):
    def __init__(self, num_features, activation_fn="silu"):
        super().__init__()

        self.num_features = num_features
        self.activation = get_activation_fn(activation_fn)

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
        if features_cond is None:
            features_cond = torch.zeros_like(features_nodes)

        # conditioning vector
        c = features_cond
        shift, scale = torch.chunk(self.activation(self.adaLN_linear(c)), 2, dim=-1)

        y = modulate_adaLN(x=self.norm(features_nodes), shift=shift, scale=scale)

        out = self.out_proj(y)

        return out
