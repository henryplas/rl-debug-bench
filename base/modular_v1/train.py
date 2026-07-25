"""Main training entrypoint: wires config, policy, value, rollout, advantage,
and update into the PPO training loop."""

import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from advantage import compute_gae
from config import TrainConfig
from policy import PolicyNetwork
from rollout import RolloutBuffer, build_vector_env, collect_rollout
from update import ppo_minibatch_step
from value import ValueNetwork


def main():
    config = tyro.cli(TrainConfig)
    config.batch_size = int(config.num_envs * config.num_steps)
    config.minibatch_size = int(config.batch_size // config.num_minibatches)
    config.num_iterations = config.total_timesteps // config.batch_size

    run_name = f"{config.env_id}__{config.exp_name}__{config.seed}__{int(time.time())}"
    logger = SummaryWriter(f"runs/{run_name}")
    logger.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join(f"|{key}|{value}|" for key, value in vars(config).items())),
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")

    vector_env = build_vector_env(config.env_id, config.num_envs, config.capture_video, run_name)
    assert isinstance(vector_env.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    observation_dim = int(np.array(vector_env.single_observation_space.shape).prod())
    num_actions = vector_env.single_action_space.n
    policy = PolicyNetwork(observation_dim, num_actions).to(device)
    value_fn = ValueNetwork(observation_dim).to(device)
    optimizer = optim.Adam(
        list(policy.parameters()) + list(value_fn.parameters()), lr=config.learning_rate, eps=1e-5,
    )

    buffer = RolloutBuffer(
        config.num_steps, config.num_envs,
        vector_env.single_observation_space.shape, vector_env.single_action_space.shape,
        device,
    )

    total_env_steps = 0
    start_time = time.time()
    current_obs, _ = vector_env.reset(seed=config.seed)
    current_obs = torch.Tensor(current_obs).to(device)
    current_done = torch.zeros(config.num_envs).to(device)

    for update_idx in range(1, config.num_iterations + 1):
        if config.anneal_lr:
            decay_fraction = 1.0 - (update_idx - 1.0) / config.num_iterations
            optimizer.param_groups[0]["lr"] = decay_fraction * config.learning_rate

        current_obs, current_done, total_env_steps = collect_rollout(
            vector_env, policy, value_fn, buffer, current_obs, current_done, total_env_steps, logger,
        )

        with torch.no_grad():
            bootstrap_value = value_fn(current_obs).reshape(1, -1)
        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.episode_done,
            bootstrap_value, current_done, config.gamma, config.gae_lambda,
        )

        flat_observations = buffer.observations.reshape((-1,) + vector_env.single_observation_space.shape)
        flat_behavior_log_probs = buffer.behavior_log_probs.reshape(-1)
        flat_actions = buffer.actions.reshape((-1,) + vector_env.single_action_space.shape)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_values = buffer.values.reshape(-1)

        batch_indices = np.arange(config.batch_size)
        clip_fractions = []
        step_metrics = None
        for epoch in range(config.update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, config.batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                minibatch_indices = batch_indices[start:end]
                step_metrics = ppo_minibatch_step(
                    policy, value_fn, optimizer, config,
                    flat_observations[minibatch_indices],
                    flat_actions.long()[minibatch_indices],
                    flat_behavior_log_probs[minibatch_indices],
                    flat_advantages[minibatch_indices],
                    flat_returns[minibatch_indices],
                    flat_values[minibatch_indices],
                )
                clip_fractions.append(step_metrics["clip_fraction"])

            if config.target_kl is not None and step_metrics["kl_estimate"] > config.target_kl:
                break

        predicted = flat_values.cpu().numpy()
        actual = flat_returns.cpu().numpy()
        return_variance = np.var(actual)
        explained_variance = np.nan if return_variance == 0 else 1 - np.var(actual - predicted) / return_variance

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], total_env_steps)
        logger.add_scalar("losses/value_loss", step_metrics["value_loss"], total_env_steps)
        logger.add_scalar("losses/policy_loss", step_metrics["policy_loss"], total_env_steps)
        logger.add_scalar("losses/entropy", step_metrics["entropy_loss"], total_env_steps)
        logger.add_scalar("losses/old_approx_kl", step_metrics["naive_kl_estimate"], total_env_steps)
        logger.add_scalar("losses/approx_kl", step_metrics["kl_estimate"], total_env_steps)
        logger.add_scalar("losses/clipfrac", np.mean(clip_fractions), total_env_steps)
        logger.add_scalar("losses/explained_variance", explained_variance, total_env_steps)
        print("SPS:", int(total_env_steps / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(total_env_steps / (time.time() - start_time)), total_env_steps)

    vector_env.close()
    logger.close()


if __name__ == "__main__":
    main()
