import torch
import torch.nn as nn


class SurrogateModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)