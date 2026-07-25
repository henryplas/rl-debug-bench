"""Modular base (Lever 1, tasks/hardness-v1.md) must train cleanly and
deterministically, same as the legacy base (tests/test_determinism.py).

train.py imports its sibling modules (config, policy, value, rollout,
advantage, update) as top-level modules resolved via the script's own
directory on sys.path, so it can run with any cwd -- isolating cwd per run
here only isolates each run's tensorboard output.
"""

import glob
import os
import subprocess
import sys

TRAIN_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "base", "modular_v1", "train.py"
)

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
        [sys.executable, TRAIN_SCRIPT, *RUN_ARGS],
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


def test_modular_base_trains_without_error(tmp_path):
    _run(cwd=tmp_path)


def test_modular_base_is_deterministic_for_fixed_seed(tmp_path):
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
