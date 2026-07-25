"""Actor network: maps observations to a discrete action distribution."""

import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical


def init_orthogonal_layer(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyNetwork(nn.Module):
    """A small MLP producing action logits for a discrete action space."""

    def __init__(self, observation_dim, num_actions):
        super().__init__()
        self.body = nn.Sequential(
            init_orthogonal_layer(nn.Linear(observation_dim, 64)),
            nn.Tanh(),
            init_orthogonal_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            init_orthogonal_layer(nn.Linear(64, num_actions), std=0.01),
        )

    def act(self, observations, action=None):
        """Return (action, log_prob, entropy) for the given observations.

        If action is None, sample a fresh one from the current policy (used
        during rollout collection). Otherwise, evaluate the log-prob/entropy
        of the given action under the current policy (used during the PPO
        update, where the action comes from the rollout buffer rather than
        being freshly sampled).
        """
        logits = self.body(observations)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy()
