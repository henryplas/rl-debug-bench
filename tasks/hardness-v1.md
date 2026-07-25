# rl-debug-bench: hardness spec (v1)

Addendum to `tasks/tasks-list.md`. Read that first. This document covers only the
changes needed to make the benchmark discriminative, and supersedes the v0
sections it contradicts.

**Precondition:** v0 build steps 1 through 5 are complete. Step 6 (smoke eval)
has failed its difficulty target twice.

---

## 1. Diagnosis

Two smoke evals of `dead_surrogate_v1` against a Sonnet-class model were solved
at or near ceiling, including after the patch was rewritten to remove the
literal self-subtraction code smell.

The cause is not that the bug is common. It is that **the correct reference
implementation is memorized**. CleanRL's single-file PPO on CartPole is one of
the most widely reproduced pieces of RL code in existence. The model is not
diagnosing from symptoms; it is diffing the workspace against a memorized
reference. Three properties of the v0 design make this possible:

1. The base is a canonical, widely copied file.
2. The base is ~400 lines, so the entire search space fits in a single read.
3. Every bug is a single wrong line, so a diff against memory localizes it.

Every lever below attacks one or more of these.

---

## 2. What "harder" must NOT mean

Do not aim for uniform difficulty. A benchmark where every task scores 0 is as
uninformative as one where every task scores 1.

**The deliverable is a difficulty gradient.** Keep the current easy bugs as the
floor. The target distribution across categories is roughly:

| Tier | Target solve rate | Purpose |
| --- | --- | --- |
| Easy (memorized-reference, single wrong line) | 80-100% | Floor, calibration |
| Medium (unmemorized base, single wrong line) | 40-70% | Main body |
| Hard (omission / interaction / statistical) | 10-40% | Headroom |

Every bug entry in `bugs/registry.yaml` gains a required `tier` field. Any tier
that comes back empirically outside its band is retuned or reclassified, not
deleted.

---

## 3. Hardness levers, in build order

Implement in this order. Each is independently useful, so ship and measure after
each rather than batching.

### Lever 1: de-memorize the base (highest value per hour)

Replace the vendored single-file CleanRL PPO with a multi-module reimplementation
of the same algorithm.

Requirements:
- Same algorithm, same hyperparameters, same learning curve. Verify by asserting
  the new base's clean baseline is within seed noise of the old base's.
- Split across at least 6 modules, e.g. `rollout.py`, `advantage.py`,
  `policy.py`, `value.py`, `update.py`, `config.py`, `train.py`.
- Different naming conventions from CleanRL. Do not keep `b_logprobs`,
  `mb_advantages`, `clipfracs`, or other recognizable CleanRL identifiers.
- Idiomatic and readable. The goal is to remove the memorized diff target, not
  to obfuscate. Deliberately confusing code is not a valid difficulty source and
  will not survive review.

Keep the old single-file base in the repo as `base/legacy_cleanrl/` and keep its
bug instances as the Easy tier. The contrast between the two bases with the same
bug injected is a direct measurement of the memorization effect, and it is a
publishable result on its own. Preserve it deliberately.

**Add a registry field `base_id`** so instances can reference either base.

### Lever 2: exceed the single-read search space

Grow the codebase so the model cannot read all of it within the turn budget.

- Target 2500 to 4000 lines across the training package.
- Add genuinely-used surrounding machinery: env wrappers, a checkpointing module,
  a metrics logger, a config loader, a small callback system, a replay/rollout
  buffer abstraction, evaluation utilities.
- Do not pad with dead code. Every module must be reachable and used, or the
  model will learn to ignore whole directories.
- Bugs may be injected anywhere in the package, not only in the update step.

This forces hypothesis-driven search from the symptom rather than exhaustive
reading, which is the capability the benchmark is supposed to measure.

Add registry field `file` so the ground truth records which module holds the bug.

### Lever 3: bug classes with no local code smell

Three new categories. Add `bug_class` to the registry with values
`wrong_line`, `omission`, `interaction`, `statistical`.

**3a. Omission.** The bug is a missing operation, not a wrong one. There is
nothing on screen to look at.
- Advantage normalization removed from the minibatch loop
- Gradient clipping removed
- The entropy term dropped from the total loss
- Value bootstrapping on truncation (as opposed to termination) not handled
- Observation normalizer never updated after construction

**3b. Interaction.** Each line is correct in isolation and wrong only in
combination with a config value or another module. No single-file inspection
finds it.
- Reward normalization applied both in a wrapper and again in the update path
- A learning rate schedule that assumes a total-timesteps value the config
  overrides, so it anneals to zero early
- Frame-stacking wrapper ordering that silently breaks the observation layout
  only when `num_envs > 1`
- A minibatch count that does not divide the rollout length, silently dropping
  the tail of every batch
- Gradient accumulation combined with a per-minibatch optimizer step, so the
  effective learning rate is off by the accumulation factor

**3c. Statistical.** Only visible in the numbers across iterations.
- All parallel envs seeded identically, so `num_envs` gives no sample diversity
- An off-by-one in a rolling-return window that biases the reported metric but
  not the true one
- Sampling with replacement where minibatches should partition the batch
- A stale-by-one-iteration value baseline used in advantage computation

For every one of these, the registry `detectable_from` field must be honest. If a
bug genuinely cannot be found from arm A (logs only), record that. The arm A
versus arm B comparison depends on this field being accurate.

### Lever 4: diagnosis-only hard mode

A new arm, `D`. The agent gets `get_metrics` and `list_metric_keys` and
`run_training`, but **no file access at all**. It must produce a structured
diagnosis rather than a patch.

Output schema the agent must emit at `submit`:

```json
{
  "component": "advantage_estimation",
  "failure_mode": "advantages not normalized within minibatch",
  "evidence_metrics": ["policy_loss", "grad_norm", "approx_kl"]
}
```

Scoring: `component` exact match against the registry, `failure_mode` graded by
string overlap against a list of accepted phrasings stored in the registry, and
`evidence_metrics` scored as set overlap against the registry's
`diagnostic_metrics` field.

This arm is cheap to add on top of the existing harness and directly isolates
whether models can reason from training dynamics rather than from code.

**Constraint:** the `failure_mode` grader must not call an LLM. Store an explicit
accepted-phrasings list per bug and match on normalized token overlap. If that
proves too brittle, drop `failure_mode` from scoring and keep `component` and
`evidence_metrics` only.

---

## 4. Registry schema changes

Extend each entry in `bugs/registry.yaml`:

```yaml
- bug_id: adv_norm_omitted_v1
  base_id: modular_v1            # NEW: modular_v1 | legacy_cleanrl
  tier: medium                   # NEW: easy | medium | hard
  bug_class: omission            # NEW: wrong_line | omission | interaction | statistical
  file: update.py                # NEW: which module holds the ground truth
  category: normalization
  patch: patches/adv_norm_omitted_v1.diff
  symptom: "Return plateaus around 120 and training is high variance."
  detectable_from: [metrics]
  ground_truth_lines: [88, 89]
  diagnostic_metrics: [policy_loss, grad_norm, approx_kl]   # NEW: for arm D
  accepted_failure_phrasings:                               # NEW: for arm D
    - "advantages are not normalized"
    - "missing advantage normalization"
    - "advantage standardization removed"
  instances: 5
  instance_seeds: [0, 1, 2, 3, 4]
```

All existing entries must be backfilled with the new fields before any new bug is
added.

---

## 5. Build order

1. Lever 1. Reimplement the base as a multi-module package. Assert clean-baseline
   equivalence with the legacy base.
2. Port `dead_surrogate_v1` to the new base unchanged. Re-run calibration.
3. **Checkpoint: smoke eval the same bug on both bases, 3 seeds each.** The
   legacy-versus-modular delta is the memorization measurement. Record it.
4. Registry schema migration (section 4), backfill existing entries.
5. Lever 3a. Two omission bugs. Calibrate. Smoke eval.
6. Lever 2. Grow the package to target size. Recalibrate everything.
7. Lever 3b and 3c. Two interaction bugs, two statistical bugs.
8. Lever 4. Arm D harness and scoring.
9. Full sweep across base x tier x arm.

Do not proceed past step 3 without recording the result. If the modular base
alone drops the solve rate into the 40-70% band, that is the single most
important finding in the project so far and it changes what is worth building
next.

---

## 6. Acceptance criteria

- At least one bug instance in each of the three tiers, empirically landing in
  its target band on at least one frontier model
- At least one bug in each of the four `bug_class` values
- The same bug injected into both bases, evaluated on both, with the delta
  reported
- Arm D implemented with a fully programmatic grader
- All registry entries carry the section 4 fields and a passing calibration
  record
- `make repro` still runs the full sweep from a clean checkout

---

## 7. Notes

- The easy tier is not a failure to be fixed. It is the control. Do not delete
  the legacy base or its instances.
- Resist making the modular base weird in order to make it hard. Difficulty must
  come from the bug and the search space, not from unreadable code. A reviewer
  will check this.
- Every new bug still passes the v0 calibration gate: the clean-versus-broken gap
  must exceed 3x the larger seed standard deviation, or the instance is rejected.
- Log everything. The transcripts showing how a model searches a 3000-line
  package from a one-line symptom are likely the most interesting qualitative
  material the project will produce.
