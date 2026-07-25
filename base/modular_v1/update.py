"""PPO clipped-surrogate update: one minibatch's policy + value loss and
optimizer step."""

import torch
import torch.nn as nn


def ppo_minibatch_step(
    policy, value_fn, optimizer, config,
    minibatch_observations, minibatch_actions, minibatch_behavior_log_probs,
    minibatch_advantages, minibatch_returns, minibatch_values,
):
    """Run one PPO minibatch update, returning a dict of scalars for logging."""
    _, current_log_prob, entropy = policy.act(minibatch_observations, minibatch_actions)
    current_value = value_fn(minibatch_observations).view(-1)

    log_probability_ratio = current_log_prob - minibatch_behavior_log_probs
    probability_ratio = log_probability_ratio.exp()

    with torch.no_grad():
        # see http://joschu.net/blog/kl-approx.html
        naive_kl_estimate = (-log_probability_ratio).mean()
        kl_estimate = ((probability_ratio - 1) - log_probability_ratio).mean()
        clip_fraction = ((probability_ratio - 1.0).abs() > config.clip_coef).float().mean().item()

    advantages = minibatch_advantages
    if config.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    unclipped_objective = -advantages * probability_ratio
    clipped_objective = -advantages * torch.clamp(probability_ratio, 1 - config.clip_coef, 1 + config.clip_coef)
    policy_loss = torch.max(unclipped_objective, clipped_objective).mean()

    if config.clip_vloss:
        unclipped_value_loss = (current_value - minibatch_returns) ** 2
        clipped_value = minibatch_values + torch.clamp(
            current_value - minibatch_values, -config.clip_coef, config.clip_coef,
        )
        clipped_value_loss = (clipped_value - minibatch_returns) ** 2
        value_loss = 0.5 * torch.max(unclipped_value_loss, clipped_value_loss).mean()
    else:
        value_loss = 0.5 * ((current_value - minibatch_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    total_loss = policy_loss - config.ent_coef * entropy_loss + value_loss * config.vf_coef

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(
        list(policy.parameters()) + list(value_fn.parameters()), config.max_grad_norm,
    )
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "naive_kl_estimate": naive_kl_estimate.item(),
        "kl_estimate": kl_estimate.item(),
        "clip_fraction": clip_fraction,
    }
