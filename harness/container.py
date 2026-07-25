"""Docker lifecycle: build the episode workspace, start/stop the container.

See tasks/tasks-list.md section 7.1. The workspace is a host-side directory
bind-mounted read-write into the container at /workspace; since it's a bind
mount (not a copy), host-side Python file I/O and in-container execution both
see the same files, so harness/tools.py can do plain file ops on the host
path while run_training executes inside the pinned, network-disabled
container.

A workspace holds every file for one base_id (harness/bases.py) -- one script
for base/legacy_cleanrl, several modules for base/modular_v1 -- not just a
single script, so both bases go through the same code path.
"""

import os
import shutil
import subprocess
import tempfile
import uuid

import docker

from harness import bases

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUGS_DIR = os.path.join(REPO_ROOT, "bugs")
IMAGE_NAME = "rl-debug-bench:v0"
WORKSPACE_MOUNT = "/workspace"


def build_workspace(base_id, patch_relpath=None, source_dir=None):
    """Create a host-side temp dir containing every .py file for base_id.
    Exactly one of:
    - patch_relpath: apply this bug patch (may touch any file in the base)
      to a fresh copy of the pristine base.
    - source_dir: copy .py files from this directory verbatim instead of the
      pristine base (used by scoring to re-run an agent's submission,
      sandboxed the same way an episode's own workspace is built).
    Returns the dir."""
    if patch_relpath is not None and source_dir is not None:
        raise ValueError("pass at most one of patch_relpath, source_dir")

    host_dir = tempfile.mkdtemp(prefix="rl-debug-bench-ws-")
    src_dir = source_dir if source_dir is not None else bases.base_dir(base_id)
    filenames = sorted(f for f in os.listdir(src_dir) if f.endswith(".py"))
    for fname in filenames:
        dest = os.path.join(host_dir, fname)
        shutil.copy(os.path.join(src_dir, fname), dest)
        os.chmod(dest, 0o644)

    if patch_relpath is not None:
        # The patch is written against base/<base_dirname>/...; apply it against
        # a tree with that layout, then flatten the files back to the workspace
        # root. A patch may touch any subset of the base's files.
        nested_dir = os.path.join(host_dir, "base", bases.base_dirname(base_id))
        os.makedirs(nested_dir, exist_ok=True)
        for fname in filenames:
            shutil.move(os.path.join(host_dir, fname), os.path.join(nested_dir, fname))
        patch_path = os.path.join(BUGS_DIR, patch_relpath)
        subprocess.run(["git", "apply", patch_path], cwd=host_dir, check=True)
        for fname in os.listdir(nested_dir):
            shutil.move(os.path.join(nested_dir, fname), os.path.join(host_dir, fname))
        shutil.rmtree(os.path.join(host_dir, "base"))

    return host_dir


class EpisodeContainer:
    """One Docker container backing a single episode's workspace."""

    def __init__(self, base_id="legacy_cleanrl", patch_relpath=None, source_dir=None):
        self.base_id = base_id
        self.host_workspace = build_workspace(base_id, patch_relpath=patch_relpath, source_dir=source_dir)
        self.client = docker.from_env()
        self.container = self.client.containers.run(
            IMAGE_NAME,
            command=["sleep", "infinity"],
            volumes={self.host_workspace: {"bind": WORKSPACE_MOUNT, "mode": "rw"}},
            working_dir=WORKSPACE_MOUNT,
            network_mode="none",  # ground rule 4: no network access inside the agent container
            detach=True,
            name=f"rl-debug-bench-{uuid.uuid4().hex[:8]}",
        )

    def exec(self, argv, timeout_s):
        """Run argv inside the container, workdir /workspace, hard-capped by
        timeout_s via the coreutils `timeout` binary. Returns (exit_code, stdout,
        stderr); exit_code 124 means the command was killed for running over
        timeout_s."""
        cmd = ["timeout", str(max(1, int(timeout_s)))] + list(argv)
        exit_code, (stdout, stderr) = self.container.exec_run(
            cmd, workdir=WORKSPACE_MOUNT, demux=True
        )
        return (
            exit_code,
            (stdout or b"").decode("utf-8", errors="replace"),
            (stderr or b"").decode("utf-8", errors="replace"),
        )

    def teardown(self):
        try:
            self.container.stop(timeout=5)
        finally:
            self.container.remove(force=True)
            shutil.rmtree(self.host_workspace, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.teardown()
