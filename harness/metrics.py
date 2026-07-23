"""Metrics store and arm B query API (README.md section 7.2).

run_training always writes a tensorboard event file under the workspace's
runs/<run_id>__.../ directory regardless of arm — the arm only controls
whether get_metrics/list_metric_keys are exposed to the agent as tools, per
section 7.2 ("The arm is a config flag, not a separate code path.").
"""

import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Recorded every iteration; see README.md section 7.2.
METRIC_KEYS = [
    "charts/episodic_return",
    "losses/policy_loss",
    "losses/value_loss",
    "losses/entropy",
    "losses/approx_kl",
    "losses/clipfrac",
    "losses/explained_variance",
    "charts/learning_rate",
]


class MetricsStore:
    def __init__(self, host_workspace):
        self.host_workspace = host_workspace

    def _event_file(self, run_id):
        pattern = os.path.join(self.host_workspace, "runs", f"*__{run_id}__*", "events.out.tfevents.*")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"no run recorded for run_id={run_id!r}")
        return matches[0]

    def _accumulator(self, run_id):
        ea = EventAccumulator(self._event_file(run_id))
        ea.Reload()
        return ea

    def list_metric_keys(self, run_id):
        ea = self._accumulator(run_id)
        return sorted(ea.Tags().get("scalars", []))

    def get_metrics(self, run_id, keys, start=None, end=None):
        ea = self._accumulator(run_id)
        available = set(ea.Tags().get("scalars", []))
        result = {}
        for key in keys:
            if key not in available:
                result[key] = {"error": f"unknown metric key {key!r}"}
                continue
            scalars = ea.Scalars(key)
            if start is not None:
                scalars = [s for s in scalars if s.step >= start]
            if end is not None:
                scalars = [s for s in scalars if s.step <= end]
            result[key] = [{"step": s.step, "value": s.value} for s in scalars]
        return result
