"""Environment setup and rollout collection: interacting with the vectorized
environment to fill one batch of trajectory data."""

import gymnasium as gym
import numpy as np
import torch


def build_vector_env(env_id, num_envs, capture_video, run_name):
    def make_single(idx):
        def thunk():
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        return thunk

    return gym.vector.SyncVectorEnv([make_single(i) for i in range(num_envs)])


class RolloutBuffer:
    """Fixed-size storage for one rollout (num_steps x num_envs) of
    trajectory data."""

    def __init__(self, num_steps, num_envs, observation_shape, action_shape, device):
        self.observations = torch.zeros((num_steps, num_envs) + observation_shape, device=device)
        self.actions = torch.zeros((num_steps, num_envs) + action_shape, device=device)
        self.behavior_log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.episode_done = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)


def collect_rollout(vector_env, policy, value_fn, buffer, current_obs, current_done, total_env_steps, logger):
    """Fill buffer with num_steps of environment interaction, starting from
    current_obs/current_done. Returns (current_obs, current_done,
    total_env_steps) to continue from on the next call."""
    device = buffer.observations.device
    num_steps = buffer.observations.shape[0]

    for step in range(num_steps):
        total_env_steps += vector_env.num_envs
        buffer.observations[step] = current_obs
        buffer.episode_done[step] = current_done

        with torch.no_grad():
            action, log_prob, _ = policy.act(current_obs)
            value = value_fn(current_obs)
            buffer.values[step] = value.flatten()
        buffer.actions[step] = action
        buffer.behavior_log_probs[step] = log_prob

        next_obs, reward, terminations, truncations, infos = vector_env.step(action.cpu().numpy())
        episode_ended = np.logical_or(terminations, truncations)
        buffer.rewards[step] = torch.tensor(reward).to(device).view(-1)
        current_obs = torch.Tensor(next_obs).to(device)
        current_done = torch.Tensor(episode_ended).to(device)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"total_env_steps={total_env_steps}, episodic_return={info['episode']['r']}")
                    logger.add_scalar("charts/episodic_return", info["episode"]["r"], total_env_steps)
                    logger.add_scalar("charts/episodic_length", info["episode"]["l"], total_env_steps)

    return current_obs, current_done, total_env_steps
