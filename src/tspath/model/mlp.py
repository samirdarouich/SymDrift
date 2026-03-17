import torch
import torch.nn as nn
from torch_geometric.data import Data
from tspath.utils import batch_center_systems

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=None, num_layers=3, activation_fn=nn.ReLU, **kwargs):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, data):
        if isinstance(data, Data):
            # Treat molecules as flat vectors of positions
            B_times_n_atoms, _ = data.pos.shape
            B = data.batch.max().item() + 1
            n_atoms = B_times_n_atoms // B
            d_net = self.input_dim // (n_atoms + 1) # +1 for atomic number feature
            x = torch.cat(
                [
                    data.pos[:, :d_net].reshape(B, -1), 
                    data.x.float().view(B, -1)
                ], 
                dim=-1
            )
            # Shape (B, n_atoms * d_net) -> (B*n_atoms, output_dim)
            out = self.net(x).view(B_times_n_atoms, -1)
            out = batch_center_systems(out, data.batch)
            # add 3d dimension if output_dim is 2 (for compatibility with drifting field)
            if d_net== 2:
                out = torch.cat([out, torch.zeros_like(out[..., :1])], dim=-1)
            return out
        else:
            return self.net(data)   
        
        
