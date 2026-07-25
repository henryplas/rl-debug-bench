"""Registry of available training-code bases (tasks/hardness-v1.md Lever 1).

Each base is a directory of .py files plus an entrypoint script to run for
training. Everything that touches a base by name (container workspace
construction, run_training's argv, calibration, scoring) resolves through
here instead of hardcoding a path, so adding a base means adding one entry.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_ROOT = os.path.join(REPO_ROOT, "base")

BASES = {
    "legacy_cleanrl": {
        "dir": os.path.join(BASE_ROOT, "legacy_cleanrl"),
        "entrypoint": "ppo_cartpole.py",
    },
    "modular_v1": {
        "dir": os.path.join(BASE_ROOT, "modular_v1"),
        "entrypoint": "train.py",
    },
}


def base_dir(base_id):
    return BASES[base_id]["dir"]


def base_dirname(base_id):
    """The base's own directory name, e.g. 'legacy_cleanrl' -- this is the
    path component bug patches are written against (a/base/<dirname>/...)."""
    return os.path.basename(base_dir(base_id))


def base_entrypoint(base_id):
    return BASES[base_id]["entrypoint"]


def base_files(base_id):
    """Relative filenames making up this base's pristine source."""
    d = base_dir(base_id)
    return sorted(f for f in os.listdir(d) if f.endswith(".py"))
