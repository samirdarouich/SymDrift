import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, noise_dim=2, hidden=64, out_dim=2, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.noise_dim = noise_dim
        self.out_dim = out_dim

    def forward(self, e):
        return self.net(e)