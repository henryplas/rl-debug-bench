"""Base repo must train cleanly and produce identical results for a fixed seed.

This backs README.md section 3, rule 2 (determinism is mandatory) and the
build-order step 1 acceptance check. It runs the pristine, unmodified
base/ppo_cartpole.py end to end at a tiny iteration budget so it stays fast.
"""

import glob
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_SCRIPT = os.path.join(REPO_ROOT, "base", "ppo_cartpole.py")

RUN_ARGS = [
    "--seed", "0",
    "--no-cuda",
    "--no-track",
    "--no-capture-video",
    "--total-timesteps", "32",
    "--num-envs", "2",
    "--num-steps", "8",
    "--num-minibatches", "2",
    "--update-epochs", "2",
    "--exp-name", "detrun",
]

SCALAR_TAGS = [
    "losses/policy_loss",
    "losses/value_loss",
    "losses/entropy",
    "losses/approx_kl",
    "losses/explained_variance",
]


def _run(cwd):
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    result = subprocess.run(
        [sys.executable, BASE_SCRIPT, *RUN_ARGS],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result


def _read_scalars(cwd, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = glob.glob(os.path.join(cwd, "runs", "*", "events.out.tfevents.*"))
    assert event_files, f"no tensorboard event file found under {cwd}/runs"
    ea = EventAccumulator(event_files[0])
    ea.Reload()
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def test_base_trains_without_error(tmp_path):
    _run(cwd=tmp_path)


def test_base_is_deterministic_for_fixed_seed(tmp_path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()

    _run(cwd=run_a)
    _run(cwd=run_b)

    for tag in SCALAR_TAGS:
        scalars_a = _read_scalars(run_a, tag)
        scalars_b = _read_scalars(run_b, tag)
        assert scalars_a, f"no scalars recorded for {tag}"
        assert scalars_a == scalars_b, f"{tag} differs between identically-seeded runs"
