"""Fixed component label space for arm D diagnosis (tasks/weekend-sprint.md
task 2). A registry entry's `component` field is the ground truth; arm D
scoring is an exact match against it, nothing fuzzier."""

COMPONENTS = [
    "advantage_estimation",
    "policy_update",
    "value_function",
    "reward",
    "observation",
    "optimizer",
    "rollout_collection",
    "environment_dynamics",
]
