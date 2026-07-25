"""Every bug patch must apply cleanly to its base_id's pristine files (tasks/tasks-list.md 5.3)."""

import ast
import os
import shutil
import sys

import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from harness import bases  # noqa: E402
from harness.container import build_workspace  # noqa: E402

BUGS_DIR = os.path.join(REPO_ROOT, "bugs")


def _registry():
    with open(os.path.join(BUGS_DIR, "registry.yaml")) as f:
        return yaml.safe_load(f)


def _entry_base_id(entry):
    return entry.get("base_id", "legacy_cleanrl")


@pytest.mark.parametrize("entry", _registry(), ids=lambda e: e["bug_id"])
def test_patch_applies_cleanly(entry):
    base_id = _entry_base_id(entry)
    workdir = build_workspace(base_id, patch_relpath=entry["patch"])
    try:
        ground_truth_file = entry.get("file", bases.base_entrypoint(base_id))
        pristine_path = os.path.join(bases.base_dir(base_id), ground_truth_file)
        patched_path = os.path.join(workdir, ground_truth_file)

        with open(pristine_path) as f:
            pristine = f.read()
        with open(patched_path) as f:
            patched = f.read()

        assert patched != pristine, f"{entry['patch']} applied but {ground_truth_file} is unchanged"
        ast.parse(patched)  # patched file must still be valid Python
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@pytest.mark.parametrize("entry", _registry(), ids=lambda e: e["bug_id"])
def test_ground_truth_lines_are_within_base_file(entry):
    base_id = _entry_base_id(entry)
    ground_truth_file = entry.get("file", bases.base_entrypoint(base_id))
    with open(os.path.join(bases.base_dir(base_id), ground_truth_file)) as f:
        num_lines = sum(1 for _ in f)
    for line in entry["ground_truth_lines"]:
        assert 1 <= line <= num_lines, (
            f"{entry['bug_id']} ground_truth_lines has {line}, "
            f"but {base_id}/{ground_truth_file} only has {num_lines} lines"
        )
