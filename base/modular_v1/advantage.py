"""Generalized Advantage Estimation (GAE)."""

import torch


def compute_gae(rewards, values, episode_done, bootstrap_value, bootstrap_done, gamma, gae_lambda):
    """Compute (advantages, returns) for one rollout via GAE(lambda).

    rewards, values, episode_done: [num_steps, num_envs].
    bootstrap_value, bootstrap_done: the value estimate and done flag for the
    observation immediately after the rollout, broadcastable to [num_envs].
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    running_gae = 0.0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            continuation_mask = 1.0 - bootstrap_done
            next_value = bootstrap_value
        else:
            continuation_mask = 1.0 - episode_done[t + 1]
            next_value = values[t + 1]
        td_residual = rewards[t] + gamma * next_value * continuation_mask - values[t]
        advantages[t] = running_gae = td_residual + gamma * gae_lambda * continuation_mask * running_gae
    returns = advantages + values
    return advantages, returns
