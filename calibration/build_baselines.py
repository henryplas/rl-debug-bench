"""Build calibration baselines for every instance in bugs/registry.yaml.

See README.md section 6. For each instance (bug_id + instance_seed):
  1. Run the pristine base repo, 3 seeds, full iteration budget -> clean_baseline.
  2. Apply the bug patch, run 3 seeds, same budget -> broken_baseline.
  3. Compute seed standard deviation for both.
  4. Reject the instance if clean_baseline - broken_baseline < 3 * max(std_clean, std_broken).

Writes calibration/baselines.json and prints a summary table.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_SCRIPT_NAME = "ppo_cartpole.py"
BASE_SCRIPT = os.path.join(REPO_ROOT, "base", BASE_SCRIPT_NAME)
REGISTRY_PATH = os.path.join(REPO_ROOT, "bugs", "registry.yaml")
BASELINES_PATH = os.path.join(REPO_ROOT, "calibration", "baselines.json")

# "Final performance" = mean episodic_return over the last fraction of training,
# smoothing over per-episode noise instead of reading a single final episode.
FINAL_WINDOW_FRAC = 0.2


def load_registry():
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def build_patched_script(patch_relpath, workdir):
    """Apply a bug patch to a copy of the pristine base script; return its path."""
    dest_dir = os.path.join(workdir, "base")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, BASE_SCRIPT_NAME)
    shutil.copy(BASE_SCRIPT, dest)
    os.chmod(dest, 0o644)  # match the mode git tracked the patch against
    patch_path = os.path.join(REPO_ROOT, "bugs", patch_relpath)
    subprocess.run(["git", "apply", patch_path], cwd=workdir, check=True)
    return dest


def run_training(script_path, seed, total_timesteps):
    """Run the PPO script to completion in an isolated cwd; return the final-window
    mean episodic return."""
    with tempfile.TemporaryDirectory() as run_dir:
        env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
        cmd = [
            sys.executable, script_path,
            "--seed", str(seed),
            "--no-cuda", "--no-track", "--no-capture-video",
            "--total-timesteps", str(total_timesteps),
            "--num-envs", "4", "--num-steps", "128",
            "--num-minibatches", "4", "--update-epochs", "4",
            "--exp-name", "calib",
        ]
        result = subprocess.run(cmd, cwd=run_dir, env=env, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"training run failed (seed={seed}): {result.stderr[-2000:]}")

        event_files = glob.glob(os.path.join(run_dir, "runs", "*", "events.out.tfevents.*"))
        if not event_files:
            raise RuntimeError(f"no tensorboard event file produced for seed={seed}")
        ea = EventAccumulator(event_files[0])
        ea.Reload()
        scalars = ea.Scalars("charts/episodic_return")
        if not scalars:
            raise RuntimeError(f"no episodic_return scalars recorded for seed={seed}")

        cutoff = (1.0 - FINAL_WINDOW_FRAC) * total_timesteps
        window = [s.value for s in scalars if s.step >= cutoff]
        if not window:
            window = [scalars[-1].value]
        return sum(window) / len(window)


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, variance ** 0.5


def derive_seed(instance_seed, k):
    # Deterministic, distinct seed per calibration replicate. Unrelated to whatever
    # single episode seed the harness later uses to instantiate the task itself.
    return instance_seed * 1000 + k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=40000)
    parser.add_argument("--calibration-seeds", type=int, default=3)
    args = parser.parse_args()

    registry = load_registry()
    results = {}
    rows = []

    for entry in registry:
        bug_id = entry["bug_id"]
        with tempfile.TemporaryDirectory() as patch_workdir:
            broken_script = build_patched_script(entry["patch"], patch_workdir)

            for instance_seed in entry["instance_seeds"]:
                instance_id = f"{bug_id}__seed{instance_seed}"
                clean_runs = []
                broken_runs = []
                for k in range(args.calibration_seeds):
                    seed = derive_seed(instance_seed, k)
                    clean_runs.append(run_training(BASE_SCRIPT, seed, args.total_timesteps))
                    broken_runs.append(run_training(broken_script, seed, args.total_timesteps))

                clean_mean, clean_std = mean_std(clean_runs)
                broken_mean, broken_std = mean_std(broken_runs)
                threshold = 3 * max(clean_std, broken_std)
                margin = clean_mean - broken_mean
                accepted = margin >= threshold

                results[instance_id] = {
                    "bug_id": bug_id,
                    "instance_seed": instance_seed,
                    "total_timesteps": args.total_timesteps,
                    "clean_baseline": clean_mean,
                    "broken_baseline": broken_mean,
                    "std_clean": clean_std,
                    "std_broken": broken_std,
                    "clean_runs": clean_runs,
                    "broken_runs": broken_runs,
                    "margin": margin,
                    "threshold": threshold,
                    "accepted": accepted,
                }
                rows.append((instance_id, clean_mean, broken_mean, margin, threshold, accepted))
                status = "ACCEPTED" if accepted else "REJECTED"
                print(
                    f"{instance_id}: clean={clean_mean:.1f} broken={broken_mean:.1f} "
                    f"margin={margin:.1f} threshold={threshold:.1f} -> {status}"
                )

    os.makedirs(os.path.dirname(BASELINES_PATH), exist_ok=True)
    with open(BASELINES_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print()
    header = f"{'instance_id':<28}{'clean':>10}{'broken':>10}{'margin':>10}{'threshold':>12}  status"
    print(header)
    for instance_id, clean_mean, broken_mean, margin, threshold, accepted in rows:
        status = "ACCEPTED" if accepted else "REJECTED"
        print(f"{instance_id:<28}{clean_mean:>10.1f}{broken_mean:>10.1f}{margin:>10.1f}{threshold:>12.1f}  {status}")

    n_accepted = sum(1 for r in rows if r[-1])
    print(f"\n{n_accepted}/{len(rows)} instances accepted.")
    if n_accepted < len(rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
