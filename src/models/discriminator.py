import torch, torch.nn as nn

class MLPDiscriminator(nn.Module):
    def __init__(self, in_dim=16, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)
