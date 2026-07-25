"""Build calibration baselines for every instance in bugs/registry.yaml.

See tasks/tasks-list.md section 6. For each instance (bug_id + instance_seed):
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # so `python calibration/build_baselines.py` can `import harness`

from harness import bases  # noqa: E402
from harness.container import build_workspace  # noqa: E402
from harness.metrics import final_window_scalar_mean  # noqa: E402
from harness.tools import NUM_ENVS, NUM_STEPS  # noqa: E402

REGISTRY_PATH = os.path.join(REPO_ROOT, "bugs", "registry.yaml")
BASELINES_PATH = os.path.join(REPO_ROOT, "calibration", "baselines.json")

# "Final performance" = mean episodic_return over the last fraction of training,
# smoothing over per-episode noise instead of reading a single final episode.
FINAL_WINDOW_FRAC = 0.2


def load_registry():
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def run_training(base_id, script_dir, seed, total_timesteps):
    """Run base_id's entrypoint (living in script_dir) to completion in an
    isolated cwd; return the final-window mean episodic return."""
    entrypoint = os.path.join(script_dir, bases.base_entrypoint(base_id))
    with tempfile.TemporaryDirectory() as run_dir:
        env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
        cmd = [
            sys.executable, entrypoint,
            "--seed", str(seed),
            "--no-cuda", "--no-track", "--no-capture-video",
            "--total-timesteps", str(total_timesteps),
            "--num-envs", str(NUM_ENVS), "--num-steps", str(NUM_STEPS),
            "--num-minibatches", "4", "--update-epochs", "4",
            "--exp-name", "calib",
        ]
        result = subprocess.run(cmd, cwd=run_dir, env=env, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"training run failed (seed={seed}): {result.stderr[-2000:]}")

        event_files = glob.glob(os.path.join(run_dir, "runs", "*", "events.out.tfevents.*"))
        if not event_files:
            raise RuntimeError(f"no tensorboard event file produced for seed={seed}")

        return final_window_scalar_mean(
            event_files[0], "charts/episodic_return", total_timesteps, window_frac=FINAL_WINDOW_FRAC
        )


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
    parser.add_argument(
        "--bug-id", action="append", dest="bug_ids",
        help="Only (re)calibrate this bug_id; repeatable. Default: every bug in the registry.",
    )
    args = parser.parse_args()

    registry = load_registry()
    if args.bug_ids:
        known = {e["bug_id"] for e in registry}
        unknown = set(args.bug_ids) - known
        if unknown:
            sys.exit(f"unknown --bug-id: {sorted(unknown)}")
        registry = [e for e in registry if e["bug_id"] in args.bug_ids]

    if os.path.exists(BASELINES_PATH):
        with open(BASELINES_PATH) as f:
            results = json.load(f)
    else:
        results = {}
    rows = []

    for entry in registry:
        bug_id = entry["bug_id"]
        base_id = entry.get("base_id", "legacy_cleanrl")
        pristine_dir = bases.base_dir(base_id)
        broken_dir = build_workspace(base_id, patch_relpath=entry["patch"])
        try:
            for instance_seed in entry["instance_seeds"]:
                instance_id = f"{bug_id}__seed{instance_seed}"
                clean_runs = []
                broken_runs = []
                for k in range(args.calibration_seeds):
                    seed = derive_seed(instance_seed, k)
                    clean_runs.append(run_training(base_id, pristine_dir, seed, args.total_timesteps))
                    broken_runs.append(run_training(base_id, broken_dir, seed, args.total_timesteps))

                clean_mean, clean_std = mean_std(clean_runs)
                broken_mean, broken_std = mean_std(broken_runs)
                threshold = 3 * max(clean_std, broken_std)
                margin = clean_mean - broken_mean
                accepted = margin >= threshold

                results[instance_id] = {
                    "bug_id": bug_id,
                    "base_id": base_id,
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
        finally:
            shutil.rmtree(broken_dir, ignore_errors=True)

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
