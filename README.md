# rl-debug-bench

An RL environment that evaluates language models on diagnosing and fixing silent
bugs in reinforcement learning training code.

**Status: v0 spec. Nothing is built yet. This document is the build order.**

---

## 1. Motivation

Bugs in RL training code fail silently. There is no stack trace, no exception, no
failing test. The run completes, the loss decreases, and the policy learns
nothing. A human diagnoses this by inspecting training curves and reasoning about
what should be true of the gradients, then checking whether it is.

Existing code benchmarks (SWE-bench and similar) test bugs that announce
themselves through failing tests or tracebacks. Silent numerical failures in
training loops are a distinct category and are not covered.

This benchmark measures whether frontier models can do that diagnosis, and it
provides a training environment with a programmatic, dense reward for improving
at it.

### Secondary research question

Human experts diagnose these bugs from plots. This benchmark exposes the training
run through three interchangeable observation modalities so the effect can be
measured:

- **A (logs)**: raw per-iteration scalars printed as text
- **B (metrics API)**: structured arrays the agent can query and compute over
- **C (plots)**: rendered PNG figures passed to a vision model

Arm B is the primary interface. A is the floor. C is an ablation. Do not build a
version where plots are the only channel.

---

## 2. Scope of v0

v0 is deliberately small. Ship this before extending anything.

In scope:

- 3 bug types, 5 instances each (15 tasks total)
- Single base repo (CleanRL PPO), single inner environment (CartPole-v1)
- Observation arms A and B only
- Docker harness with a fixed tool set
- Two-component reward (outcome and localization)
- Baseline eval of 1 to 2 models, 3 seeds each
- Public repo, MIT license, one-command reproduction

Explicitly out of scope for v0:

- Arm C (plot rendering)
- Multi-bug instances
- Continuous-control inner tasks
- Training a model on the environment (that is v2)
- Any LLM in the reward path

---

## 3. Ground rules for the implementation

1. **No LLM judges anywhere in the reward.** Every reward component must be
   computable by a deterministic script.
2. **Determinism is mandatory.** Same instance plus same seed must produce the
   same score. Seed Python, NumPy, and Torch. Set
   `torch.use_deterministic_algorithms(True)` and pin `CUBLAS_WORKSPACE_CONFIG`.
3. **The agent cannot touch the scorer.** Scoring code lives outside the agent's
   writable directory. Hash-check it after every episode and fail the episode if
   it changed.
4. **No network access inside the agent container.**
5. **A bug that does not degrade performance is not a task.** Every bug must be
   empirically validated (see section 6).
6. **Log every trajectory in full.** Transcripts are a primary output, not a
   byproduct.

---

## 4. Repository layout

```
rl-debug-bench/
  README.md
  LICENSE                      # MIT
  pyproject.toml
  Dockerfile
  Makefile

  base/
    ppo_cartpole.py            # vendored CleanRL single-file PPO, MIT, unmodified
    requirements.txt           # pinned

  bugs/
    registry.yaml              # canonical bug bank index
    patches/
      dead_surrogate_v1.diff
      adv_broadcast_v1.diff
      gae_bootstrap_v1.diff
      ...

  harness/
    __init__.py
    container.py               # docker lifecycle, workspace mount, teardown
    tools.py                   # tool implementations exposed to the agent
    episode.py                 # main agent loop, turn cap, wall clock cap
    metrics.py                 # metrics store, arm B query API
    render.py                  # arm C plot rendering (stub in v0)
    models.py                  # provider adapters (Anthropic, OpenAI, ...)

  scoring/
    outcome.py                 # normalized post-fix performance
    localization.py            # diff overlap against ground truth lines
    integrity.py               # scorer tamper detection
    score_episode.py           # entrypoint, emits results JSON

  calibration/
    build_baselines.py         # clean and broken baselines per instance
    baselines.json             # generated, committed

  eval/
    run_eval.py                # sweeps model x instance x seed
    analyze.py                 # tables and figures from results
    results/                   # committed JSON results
    transcripts/               # committed episode logs

  tests/
    test_determinism.py
    test_patches_apply.py
    test_scoring.py
    test_integrity.py
```

---

## 5. Bug bank

### 5.1 Registry schema

`bugs/registry.yaml`, one entry per bug type:

```yaml
- bug_id: dead_surrogate_v1
  category: silent_gradient
  patch: patches/dead_surrogate_v1.diff
  symptom: "Episodic return is flat after 500 iterations. No errors."
  difficulty: hard
  detectable_from: [metrics, plots]      # not diagnosable from logs alone
  ground_truth_lines: [312, 313, 314]    # lines in base/ppo_cartpole.py
  ground_truth_description: >
    The ratio in the PPO surrogate objective is computed from log-probs
    recorded under the current policy rather than the rollout policy, so the
    ratio is identically 1.0 and the policy gradient term vanishes.
  instances: 5
  instance_seeds: [0, 1, 2, 3, 4]
```

### 5.2 The three v0 bugs

Build exactly these first.

**1. `dead_surrogate_v1`** (hard, silent)
The old log-probs used in the PPO ratio are recomputed from the current policy
instead of being taken from the rollout buffer. The ratio is always 1.0, the
clipped surrogate has zero gradient, and only the value function and entropy
terms train. Detectable by comparing pre- and post-update log-probs, or by
noticing the policy loss is pinned near a constant.

**2. `adv_broadcast_v1`** (medium, silent)
Advantages keep shape `[N, 1]` where the log-prob ratio is `[N]`. The product
broadcasts to `[N, N]`, so every sample's advantage is applied against every
sample's ratio. Training is not fully dead but is badly degraded and
high-variance. Detectable from an anomalous loss magnitude or by printing shapes.

**3. `obs_norm_frozen_v1`** (medium, silent)
The running observation normalizer's update is disabled after initialization, so
statistics are frozen at whatever the first batch produced. Learning is slow and
plateaus early. Detectable by inspecting the normalizer state across iterations.

Each of the 5 instances per bug varies the instance seed and cosmetic details
(variable naming in the surrounding lines, iteration budget, the wording of the
reported symptom) so the tasks are not trivially memorizable.

### 5.3 Adding a bug later

A new bug type is: a patch that applies cleanly to the pristine base file, a
registry entry with ground truth lines, and a passing calibration run. Nothing
else.

---

## 6. Calibration (build this before the harness)

`calibration/build_baselines.py` must, for every instance:

1. Run the pristine base repo, 3 seeds, full iteration budget. Record mean final
   performance as `clean_baseline`.
2. Apply the bug patch. Run 3 seeds, same budget. Record mean final performance
   as `broken_baseline`.
3. Compute seed standard deviation for both.
4. **Reject the instance** if
   `clean_baseline - broken_baseline < 3 * max(std_clean, std_broken)`.
   The signal must exceed seed noise or the task is unscoreable.

Write results to `calibration/baselines.json` and commit it. Print a summary
table of accepted and rejected instances.

Use CartPole-v1 with an iteration budget that trains cleanly in under 2 minutes
on one GPU. That budget sets the cost of the entire project, so tune it here.

---

## 7. Harness

### 7.1 Episode structure

1. Create a container from the pinned image.
2. Copy the base repo plus applied bug patch into `/workspace` (agent-writable).
3. Mount scoring code read-only outside `/workspace`.
4. Present the agent with the system prompt and the instance symptom string.
5. Loop until the agent emits `submit`, or the turn cap or wall clock cap is hit.
6. Run scoring. Tear down. Write the results JSON and the full transcript.

Caps for v0: 30 turns, 20 minutes wall clock.

### 7.2 Tools exposed to the agent

```
list_files(path)                     -> string
read_file(path, start=None, end=None) -> string with line numbers
edit_file(path, old_str, new_str)     -> confirmation or error
run_training(iterations)              -> run_id, plus arm-A tail of stdout
get_metrics(keys, run_id, start, end) -> arm B only. JSON arrays.
list_metric_keys(run_id)              -> arm B only.
render_plot(keys, run_id)             -> arm C only. Returns a PNG. Stub in v0.
submit()                              -> ends the episode
```

Arm A gets `run_training` output as printed text only, with no `get_metrics`.
Arm B gets both. The arm is a config flag, not a separate code path.

Metrics recorded every iteration and available to arm B: `episodic_return`,
`policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clipfrac`,
`explained_variance`, `grad_norm`, `learning_rate`.

### 7.3 System prompt for the agent

Keep it short and fixed across all models. Something like: the workspace contains
an RL training script with a bug; the reported symptom is X; use the available
tools to diagnose and fix it; call submit when done. Do not hint at the bug
category or the file region.

---

## 8. Scoring

Emit both components separately. Never collapse them into a single headline
number.

### 8.1 Outcome reward

Re-run the agent's patched code, 3 fresh seeds, full budget:

```
outcome = clip((achieved - broken_baseline) / (clean_baseline - broken_baseline), 0, 1)
```

### 8.2 Localization reward

Diff the agent's final file against the pristine base. Let `changed` be the set of
modified line numbers mapped back to base-file coordinates, and `truth` be
`ground_truth_lines`.

```
localization = |changed ∩ truth| / |changed ∪ truth|
```

Report the binary form (`localization > 0`) as well.

### 8.3 Integrity check

Hash `scoring/` and `calibration/baselines.json` before and after the episode. Any
mismatch marks the episode `INVALID` and it is excluded from aggregate scores but
retained in the transcript set. Also flag episodes where the agent modified the
evaluation loop, altered the reward function of the inner environment, or
hardcoded a return value. These are reported as a `hack_attempt` count, which is a
result in its own right.

### 8.4 Results JSON

```json
{
  "instance_id": "dead_surrogate_v1__seed3",
  "model": "...",
  "arm": "B",
  "episode_seed": 0,
  "outcome": 0.94,
  "localization": 0.6,
  "localization_binary": true,
  "turns_used": 17,
  "wall_clock_s": 412,
  "tool_calls": {"read_file": 9, "run_training": 3, "get_metrics": 6},
  "status": "OK",
  "hack_attempt": false
}
```

---

## 9. Evaluation protocol

`eval/run_eval.py` sweeps model x instance x episode_seed, 3 episode seeds per
cell, and writes one JSON per episode.

`eval/analyze.py` must produce:

1. Outcome and localization per model, aggregate, with standard error over seeds
2. **Per-category breakdown.** Aggregate numbers are not the interesting result.
   The claim lives in the difference between categories.
3. Arm A versus arm B at equal turn budget
4. The outcome-minus-localization gap per model, that is, how often a model
   restores performance without identifying the cause
5. Turn count and token cost distributions
6. Hack attempt counts

Standard error on a proportion with n=45 episodes is roughly 7 points, so do not
report differences under about 15 points as real at v0 scale. State this in the
results section.

**Target difficulty: 30 to 50 percent outcome on the hardest category.** If the
first model tested clears 80 percent overall, the bank is too easy and needs
harder bugs before anything else is built. Test this after the first bug type is
implemented, not after all three.

---

## 10. Build order

Do these in sequence. Do not parallelize past step 4.

1. Repo skeleton, pinned Dockerfile, vendored CleanRL base, MIT license, tests
   that the base trains cleanly and deterministically
2. `dead_surrogate_v1` patch plus registry entry
3. `calibration/build_baselines.py`, run it, confirm the clean/broken gap clears
   the noise threshold
4. Harness with arm A only, tools, episode loop, transcript logging
5. Scoring: outcome, localization, integrity
6. **Smoke eval: one model, one bug, 3 seeds.** Check the difficulty target here
   before building more
7. Arm B metrics store and query API
8. Remaining two bug types plus calibration
9. Full eval sweep, analysis, figures
10. README rewrite from spec into documentation, publish

---

## 11. Acceptance criteria for v0

- `make repro` runs the full eval from a clean checkout in a single command
- Every instance in `registry.yaml` has a committed calibration entry
- `pytest` passes, including a determinism test that runs the same instance
  twice and asserts identical scores
- At least one model evaluated on all 15 instances, 3 seeds, both arms
- `eval/results/` and `eval/transcripts/` committed
- README replaced with real documentation and a results table

---

## 12. Notes for whoever builds this

- The base repo must stay pristine and unmodified. All bugs are patches. This is
  what lets the bank grow without regenerating anything.
- Do not write the bug patches by hand into the base file. Generate them as
  diffs and apply at instance construction time.
- Wall clock per episode is dominated by `run_training` calls. If the inner task
  takes more than 2 minutes, the full sweep becomes unaffordable. Fix that at
  calibration time.
- Resist adding a fourth bug type before step 6. The difficulty check is the
  point of no return, and everything built before it is at risk of being thrown
  away.
