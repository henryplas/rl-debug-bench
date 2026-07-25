"""Tool implementations exposed to the agent (README.md section 7.2).

list_files/read_file/edit_file operate directly on the host-side workspace
path, since it's bind-mounted into the container (not copied) — the agent's
in-container view and this host view are the same files. run_training and
get_metrics involve the pinned image's Python/torch environment or a metrics
store keyed by run_id, so they go through the container / MetricsStore.
"""

import os
import time

from harness import bases
from harness.metrics import MetricsStore
from scoring.components import COMPONENTS

NUM_ENVS = 4
NUM_STEPS = 128
BATCH_SIZE = NUM_ENVS * NUM_STEPS
STDOUT_TAIL_CHARS = 4000
DEFAULT_RUN_TIMEOUT_S = 300

# Common tool schema, provider-agnostic. harness/models.py adapters translate
# this into each provider's wire format.
TOOL_SCHEMAS_BASE = [
    {
        "name": "list_files",
        "description": "List files in a directory of the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path, relative to the workspace root."}},
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the workspace, with line numbers. Optionally slice by line range (1-indexed, inclusive).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start": {"type": "integer", "description": "First line to include, 1-indexed."},
                "end": {"type": "integer", "description": "Last line to include, 1-indexed."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace an exact, unique occurrence of old_str with new_str in a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "run_training",
        "description": "Run PPO training from scratch for the given number of update iterations. Returns a run_id and the tail of stdout.",
        "input_schema": {
            "type": "object",
            "properties": {"iterations": {"type": "integer", "description": "Number of PPO update iterations to run."}},
            "required": ["iterations"],
        },
    },
    {
        "name": "submit",
        "description": "End the episode. Call this once you believe the bug is fixed.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

TOOL_SCHEMAS_ARM_B_EXTRA = [
    {
        "name": "get_metrics",
        "description": "Fetch recorded metric arrays for a run_id, arm B only.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "keys": {"type": "array", "items": {"type": "string"}},
                "start": {"type": "integer", "description": "First step to include."},
                "end": {"type": "integer", "description": "Last step to include."},
            },
            "required": ["run_id", "keys"],
        },
    },
    {
        "name": "list_metric_keys",
        "description": "List available metric keys for a run_id, arm B only.",
        "input_schema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
        },
    },
]

# Arm D (tasks/weekend-sprint.md task 2): diagnosis-only, no file access at
# all. get_metrics/list_metric_keys are the same as arm B's; run_training is
# the same as arm A's; submit is entirely different (a structured diagnosis,
# not a no-arg "I'm done").
TOOL_SCHEMAS_ARM_D_EXTRA = [
    {
        "name": "get_metrics",
        "description": "Fetch recorded metric arrays for a run_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "keys": {"type": "array", "items": {"type": "string"}},
                "start": {"type": "integer", "description": "First step to include."},
                "end": {"type": "integer", "description": "Last step to include."},
            },
            "required": ["run_id", "keys"],
        },
    },
    {
        "name": "list_metric_keys",
        "description": "List available metric keys for a run_id.",
        "input_schema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
        },
    },
]

SUBMIT_SCHEMA_DIAGNOSIS = {
    "name": "submit",
    "description": "End the episode with a structured diagnosis. You have no file access; diagnose from training runs and metrics only.",
    "input_schema": {
        "type": "object",
        "properties": {
            "component": {"type": "string", "enum": COMPONENTS, "description": "Which component the bug lives in."},
            "failure_mode": {"type": "string", "description": "Free-text description of the failure mechanism."},
            "evidence_metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Metric keys that support this diagnosis.",
            },
        },
        "required": ["component", "failure_mode", "evidence_metrics"],
    },
}


def tool_schemas_for_arm(arm):
    if arm not in ("A", "B", "D"):
        raise ValueError(f"unknown arm: {arm!r}")
    if arm == "D":
        schemas = [s for s in TOOL_SCHEMAS_BASE if s["name"] == "run_training"]
        schemas += TOOL_SCHEMAS_ARM_D_EXTRA
        schemas += [SUBMIT_SCHEMA_DIAGNOSIS]
        return schemas
    schemas = list(TOOL_SCHEMAS_BASE)
    if arm == "B":
        schemas += TOOL_SCHEMAS_ARM_B_EXTRA
    return schemas


class WorkspaceEscapeError(ValueError):
    pass


class ToolBox:
    """Binds the tool implementations to one episode's container and arm."""

    def __init__(self, container, arm, episode_seed):
        self.container = container
        self.arm = arm
        self.episode_seed = episode_seed
        self.metrics = MetricsStore(container.host_workspace)
        self._run_counter = 0
        self.diagnosis = None

    def _resolve(self, path):
        path = path.lstrip("/")
        if path.startswith("workspace/"):
            path = path[len("workspace/"):]
        if path in ("", "."):
            path = "."
        resolved = os.path.normpath(os.path.join(self.container.host_workspace, path))
        root = self.container.host_workspace
        if not (resolved == root or resolved.startswith(root + os.sep)):
            raise WorkspaceEscapeError(f"path escapes workspace: {path!r}")
        return resolved

    def list_files(self, path="."):
        if self.arm == "D":
            return "error: list_files is not available on arm D (no file access)"
        try:
            target = self._resolve(path)
        except WorkspaceEscapeError as e:
            return f"error: {e}"
        if not os.path.exists(target):
            return f"error: no such path: {path!r}"
        if os.path.isfile(target):
            return os.path.basename(target)
        entries = sorted(os.listdir(target))
        return "\n".join(entries) if entries else "(empty directory)"

    def read_file(self, path, start=None, end=None):
        if self.arm == "D":
            return "error: read_file is not available on arm D (no file access)"
        try:
            target = self._resolve(path)
        except WorkspaceEscapeError as e:
            return f"error: {e}"
        if not os.path.isfile(target):
            return f"error: no such file: {path!r}"
        with open(target) as f:
            lines = f.readlines()

        lo = 1 if start is None else max(1, int(start))
        hi = len(lines) if end is None else min(len(lines), int(end))
        if lo > hi:
            return f"error: empty range start={start} end={end} for a {len(lines)}-line file"

        out = []
        for i in range(lo, hi + 1):
            out.append(f"{i}\t{lines[i - 1].rstrip(chr(10))}")
        return "\n".join(out)

    def edit_file(self, path, old_str, new_str):
        if self.arm == "D":
            return "error: edit_file is not available on arm D (no file access)"
        try:
            target = self._resolve(path)
        except WorkspaceEscapeError as e:
            return f"error: {e}"
        if not os.path.isfile(target):
            return f"error: no such file: {path!r}"
        with open(target) as f:
            content = f.read()

        count = content.count(old_str)
        if count == 0:
            return "error: old_str not found in file"
        if count > 1:
            return f"error: old_str is not unique ({count} occurrences); include more context"

        new_content = content.replace(old_str, new_str, 1)
        with open(target, "w") as f:
            f.write(new_content)
        return "ok: edit applied"

    def run_training(self, iterations, max_wall_s=None):
        iterations = max(1, int(iterations))
        total_timesteps = iterations * BATCH_SIZE
        run_id = f"run{self._run_counter}"
        seed = self.episode_seed * 1000 + self._run_counter
        self._run_counter += 1

        timeout_s = DEFAULT_RUN_TIMEOUT_S
        if max_wall_s is not None:
            timeout_s = max(1, min(timeout_s, int(max_wall_s)))

        argv = [
            "python", bases.base_entrypoint(self.container.base_id),
            "--seed", str(seed),
            "--no-cuda", "--no-track", "--no-capture-video",
            "--total-timesteps", str(total_timesteps),
            "--num-envs", str(NUM_ENVS), "--num-steps", str(NUM_STEPS),
            "--num-minibatches", "4", "--update-epochs", "4",
            "--exp-name", run_id,
        ]
        start = time.time()
        exit_code, stdout, stderr = self.container.exec(argv, timeout_s=timeout_s)
        elapsed = time.time() - start

        if exit_code == 124:
            return run_id, f"error: run_training timed out after {timeout_s}s"
        if exit_code != 0:
            tail = (stdout + stderr)[-STDOUT_TAIL_CHARS:]
            return run_id, f"error: training exited with code {exit_code} after {elapsed:.1f}s:\n{tail}"

        tail = stdout[-STDOUT_TAIL_CHARS:]
        return run_id, tail

    def list_metric_keys(self, run_id):
        if self.arm not in ("B", "D"):
            return "error: list_metric_keys is only available on arm B/D"
        return self.metrics.list_metric_keys(run_id)

    def get_metrics(self, run_id, keys, start=None, end=None):
        if self.arm not in ("B", "D"):
            return "error: get_metrics is only available on arm B/D"
        return self.metrics.get_metrics(run_id, keys, start=start, end=end)

    def submit(self, component=None, failure_mode=None, evidence_metrics=None):
        if component is not None:
            self.diagnosis = {
                "component": component,
                "failure_mode": failure_mode,
                "evidence_metrics": evidence_metrics,
            }
        return "submitted"

    def dispatch(self, name, tool_input, max_wall_s=None):
        """Call the named tool with tool_input (a dict), returning a string
        result. Unknown tool names, and malformed arguments to a known tool
        (wrong type, missing/extra keys), return an error string rather than
        raising, so a single bad tool call doesn't crash the episode loop."""
        try:
            if name == "list_files":
                return self.list_files(**tool_input)
            if name == "read_file":
                return self.read_file(**tool_input)
            if name == "edit_file":
                return self.edit_file(**tool_input)
            if name == "run_training":
                run_id, tail = self.run_training(max_wall_s=max_wall_s, **tool_input)
                return f"run_id: {run_id}\n{tail}"
            if name == "list_metric_keys":
                return self.list_metric_keys(**tool_input)
            if name == "get_metrics":
                return self.get_metrics(**tool_input)
            if name == "submit":
                return self.submit(**tool_input)
            return f"error: unknown tool {name!r}"
        except (TypeError, ValueError, KeyError) as e:
            return f"error: invalid arguments for {name}: {type(e).__name__}: {e}"
