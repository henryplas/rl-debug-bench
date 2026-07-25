"""Critic network: maps observations to a scalar state-value estimate."""

import torch.nn as nn

from policy import init_orthogonal_layer


class ValueNetwork(nn.Module):
    def __init__(self, observation_dim):
        super().__init__()
        self.body = nn.Sequential(
            init_orthogonal_layer(nn.Linear(observation_dim, 64)),
            nn.Tanh(),
            init_orthogonal_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            init_orthogonal_layer(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, observations):
        return self.body(observations)
