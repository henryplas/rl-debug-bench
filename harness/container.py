"""Docker lifecycle: build the episode workspace, start/stop the container.

See README.md section 7.1. The workspace is a host-side directory bind-mounted
read-write into the container at /workspace; since it's a bind mount (not a
copy), host-side Python file I/O and in-container execution both see the same
files, so harness/tools.py can do plain file ops on the host path while
run_training executes inside the pinned, network-disabled container.
"""

import os
import shutil
import subprocess
import tempfile
import uuid

import docker

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_SCRIPT = os.path.join(REPO_ROOT, "base", "ppo_cartpole.py")
BUGS_DIR = os.path.join(REPO_ROOT, "bugs")
IMAGE_NAME = "rl-debug-bench:v0"
WORKSPACE_MOUNT = "/workspace"


def _build_workspace(patch_relpath):
    """Create a host-side temp dir containing ppo_cartpole.py, with the given
    bug patch applied (pristine if patch_relpath is None). Returns the dir."""
    host_dir = tempfile.mkdtemp(prefix="rl-debug-bench-ws-")
    dest = os.path.join(host_dir, "ppo_cartpole.py")
    shutil.copy(BASE_SCRIPT, dest)
    os.chmod(dest, 0o644)

    if patch_relpath is not None:
        # The patch is written against base/ppo_cartpole.py; apply it against a
        # tree with that layout, then flatten ppo_cartpole.py back to the
        # workspace root.
        nested_dir = os.path.join(host_dir, "base")
        os.makedirs(nested_dir, exist_ok=True)
        shutil.move(dest, os.path.join(nested_dir, "ppo_cartpole.py"))
        patch_path = os.path.join(BUGS_DIR, patch_relpath)
        subprocess.run(["git", "apply", patch_path], cwd=host_dir, check=True)
        shutil.move(os.path.join(nested_dir, "ppo_cartpole.py"), dest)
        shutil.rmtree(nested_dir)

    return host_dir


class EpisodeContainer:
    """One Docker container backing a single episode's workspace."""

    def __init__(self, patch_relpath=None):
        self.host_workspace = _build_workspace(patch_relpath)
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
