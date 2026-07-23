"""Every bug patch must apply cleanly to the pristine base file (README.md 5.3)."""

import ast
import os
import subprocess
import sys

import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUGS_DIR = os.path.join(REPO_ROOT, "bugs")
BASE_SCRIPT = os.path.join(REPO_ROOT, "base", "ppo_cartpole.py")


def _registry():
    with open(os.path.join(BUGS_DIR, "registry.yaml")) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("entry", _registry(), ids=lambda e: e["bug_id"])
def test_patch_applies_cleanly(entry, tmp_path):
    workdir = tmp_path / entry["bug_id"]
    (workdir / "base").mkdir(parents=True)
    with open(BASE_SCRIPT) as f:
        pristine = f.read()
    (workdir / "base" / "ppo_cartpole.py").write_text(pristine)

    patch_path = os.path.join(BUGS_DIR, entry["patch"])
    result = subprocess.run(
        ["git", "apply", "--verbose", patch_path],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{entry['patch']} failed to apply: {result.stderr}"

    patched = (workdir / "base" / "ppo_cartpole.py").read_text()
    assert patched != pristine, f"{entry['patch']} applied but changed nothing"
    ast.parse(patched)  # patched file must still be valid Python


@pytest.mark.parametrize("entry", _registry(), ids=lambda e: e["bug_id"])
def test_ground_truth_lines_are_within_base_file(entry):
    with open(BASE_SCRIPT) as f:
        num_lines = sum(1 for _ in f)
    for line in entry["ground_truth_lines"]:
        assert 1 <= line <= num_lines, (
            f"{entry['bug_id']} ground_truth_lines has {line}, "
            f"but base/ppo_cartpole.py only has {num_lines} lines"
        )
