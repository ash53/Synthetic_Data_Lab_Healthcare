import torch, torch.nn as nn

class MLPGenerator(nn.Module):
    def __init__(self, latent_dim=64, out_dim=16, hidden=128, cond_dim=0):
        super().__init__()
        in_dim = latent_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, z, c=None):
        if c is not None:
            z = torch.cat([z, c], dim=1)
        return self.net(z)
