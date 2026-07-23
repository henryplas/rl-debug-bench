"""Normalized post-fix performance (README.md section 8.1).

outcome = clip((achieved - broken_baseline) / (clean_baseline - broken_baseline), 0, 1)

Re-runs the agent's submitted file inside a fresh, network-disabled container
(the same sandboxing an episode itself uses), at 3 fresh seeds and the same
hyperparameters/total_timesteps used to build the instance's calibration
baseline (calibration/baselines.json) -- outcome and the baselines it's
normalized against are only comparable if "final performance" means the same
thing in both places, hence sharing harness.metrics.final_window_scalar_mean.
"""

import glob
import json
import os

from harness.container import EpisodeContainer
from harness.metrics import final_window_scalar_mean
from harness.tools import NUM_ENVS, NUM_STEPS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_PATH = os.path.join(REPO_ROOT, "calibration", "baselines.json")

NUM_SCORING_SEEDS = 3
# Distinct from calibration's own seed namespace (instance_seed*1000+k for
# k in 0..2), so scoring always re-runs on genuinely fresh seeds.
SEED_OFFSET = 500


def _load_baseline(instance_id):
    with open(BASELINES_PATH) as f:
        baselines = json.load(f)
    if instance_id not in baselines:
        raise KeyError(f"no calibration baseline for instance_id={instance_id!r}")
    return baselines[instance_id]


def _run_once(container, seed, total_timesteps, exp_name):
    argv = [
        "python", "ppo_cartpole.py",
        "--seed", str(seed),
        "--no-cuda", "--no-track", "--no-capture-video",
        "--total-timesteps", str(total_timesteps),
        "--num-envs", str(NUM_ENVS), "--num-steps", str(NUM_STEPS),
        "--num-minibatches", "4", "--update-epochs", "4",
        "--exp-name", exp_name,
    ]
    exit_code, stdout, stderr = container.exec(argv, timeout_s=600)
    if exit_code != 0:
        raise RuntimeError(f"scoring run failed (seed={seed}): {(stdout + stderr)[-2000:]}")

    pattern = os.path.join(container.host_workspace, "runs", f"*__{exp_name}__*", "events.out.tfevents.*")
    event_files = glob.glob(pattern)
    if not event_files:
        raise RuntimeError(f"no tensorboard event file produced for seed={seed}")
    return final_window_scalar_mean(event_files[0], "charts/episodic_return", total_timesteps)


def score_outcome(source_path, instance_id, num_seeds=NUM_SCORING_SEEDS):
    baseline = _load_baseline(instance_id)
    total_timesteps = baseline["total_timesteps"]
    clean_baseline = baseline["clean_baseline"]
    broken_baseline = baseline["broken_baseline"]

    runs = []
    with EpisodeContainer(source_path=source_path) as container:
        for k in range(num_seeds):
            seed = SEED_OFFSET + k
            runs.append(_run_once(container, seed, total_timesteps, exp_name=f"score{k}"))

    achieved = sum(runs) / len(runs)
    denom = clean_baseline - broken_baseline
    raw = (achieved - broken_baseline) / denom if denom != 0 else 0.0
    outcome = max(0.0, min(1.0, raw))

    return {
        "outcome": outcome,
        "achieved": achieved,
        "runs": runs,
        "clean_baseline": clean_baseline,
        "broken_baseline": broken_baseline,
    }
