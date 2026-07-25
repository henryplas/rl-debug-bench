# rl-debug-bench: weekend sprint

A scoped, time-boxed task list. Supersedes `tasks/hardness-v1.md` and
`tasks/roadmap.md` for the duration of this sprint. Those documents remain the
plan afterward.

**Goal: produce one committed, defensible result.** Not more infrastructure, not
more bug types, not a new base. The deliverable is a results table and a finding.

**Explicitly out of scope this sprint.** Do not start any of these, even if
blocked on something else:

- The drone base (`drone_v1`)
- The modular CleanRL rewrite (hardness-v1 lever 1)
- Growing the codebase past single-read size (lever 2)
- Wall clock optimization work
- Arm C (plot rendering)
- Multi-bug instances
- Anything in `roadmap.md` phase 2 or later

If a task here is blocked, stop and report rather than substituting work from the
out-of-scope list.

---

## Context: two findings so far

Both say the same thing. The substrate is too easy, not the bugs.

**Finding 1, memorization.** `dead_surrogate_v1` was solved at ceiling by a
`claude-sonnet-4-5` smoke eval, twice, including after the patch was rewritten to
remove the visible code smell while preserving the identical mechanism
(calibration byte-identical). Diagnosis: CleanRL single-file PPO is one of the
most reproduced pieces of RL code in existence, it fits in one `read_file` call,
and every bug is a single wrong line. The model is diffing against a memorized
reference rather than diagnosing from symptoms.

**Finding 2, non-load-bearing omissions.** All four omission-bug candidates
(gradient clipping, entropy bonus, advantage normalization) *improved* rather
than degraded training at the 40k-timestep CartPole-v1 budget. This is the
expected result, not a fluke. Those mechanisms are variance reduction and
stability machinery that pays off on hard tasks and costs on easy ones. CartPole
has two actions, uniform +1 reward, no exploration problem worth an entropy
bonus, and advantages that are already well scaled. Removing them cannot degrade
a task where they are not doing work.

Do not retry the omission candidates at a longer budget. CartPole caps at 500
return, so both arms saturate and you get a ceiling effect instead of separation.

### New standing design rule

Add to the invariants in the README:

> **A bug class is only valid on a base where the affected mechanism is
> load-bearing.** Before writing an omission or interaction patch, ablate the
> mechanism on the clean base and confirm it measurably degrades the clean
> baseline. If it does not, the task is wrong, not the bug. This check is cheap
> and runs before calibration.

---

## The sprint hypothesis

If finding 1 is right, then removing code access should collapse performance on
the same bug. Arm A gives the model the memorized-diff shortcut. Arm D does not.

**One variable changes: whether the agent can read the code.** Same bug, same 5
instances, same calibration, same models. A large arm A to arm D gap is direct
evidence that the benchmark with code access measures reference matching rather
than diagnosis from training dynamics.

That is the result this sprint exists to produce.

---

# Task 0: record the findings

Highest priority. Do this first and commit before starting task 1.

1. Add the finding 2 text above to the README, in the same section as the
   memorization finding.
2. Add the load-bearing design rule to the README invariants list.
3. Fill in the actual solve count from the second smoke eval. Replace any
   "in progress" language with the number.
4. Commit all existing episode transcripts to `eval/transcripts/` and any
   existing results JSON to `eval/results/`.

An uncommitted result is not a result. This task is not optional and it is not
last.

---

# Task 1: multi-model arm A sweep

No new code. Use the existing `eval/run_eval.py`.

**Configuration**
- Bug: `dead_surrogate_v1`, revised patch, all 5 instances
- Arm: A
- Episode seeds: 3 per instance
- Models: 4 to 5, spanning a capability range. Include at least one deliberately
  weaker or older model so the result has a floor and is not a flat ceiling.
  Include at least two providers if adapters exist; if only
  `AnthropicAdapter` is implemented, use a range of Anthropic models rather than
  writing a new adapter this sprint.

Total: 5 instances x 3 seeds x 5 models = 75 episodes.

**Before launching**
- Set a hard spend cap on the API workspace. Arm A puts full file contents in
  context on every turn and 75 episodes at a 30-turn cap adds up faster than
  expected. Cap it low enough that a runaway loop cannot drain the account.
- Confirm the integrity check is active. Any `INVALID` episode is retained in
  transcripts and excluded from aggregates.

**Deliverable**
- `eval/results/` populated
- `eval/transcripts/` populated
- A per-model table in the README: outcome, localization, turns used, with
  standard error over seeds

---

# Task 2: minimal arm D

Strip this down from the `hardness-v1.md` spec. Build the smallest version that
answers the question.

## Tool list

Arm D exposes exactly these:

```
run_training(iterations)              -> run_id, plus arm-A style stdout tail
get_metrics(keys, run_id, start, end) -> JSON arrays
list_metric_keys(run_id)              -> list of available keys
submit(diagnosis)                     -> ends the episode
```

**Removed in arm D:** `read_file`, `list_files`, `edit_file`. The agent has no
file access of any kind. It cannot see the source, the config, or the directory
structure.

Implement this as a config flag on the existing arm mechanism, not a separate
code path.

## Submit schema

```json
{
  "component": "policy_update",
  "failure_mode": "the old log-prob is recomputed from the current policy so the ratio is always 1",
  "evidence_metrics": ["policy_loss", "approx_kl", "clipfrac"]
}
```

## Component enum

Fixed, 8 values. Store in `scoring/components.py` and include the list in the
arm D system prompt so the model knows the label space.

```
advantage_estimation
policy_update
value_function
reward
observation
optimizer
rollout_collection
environment_dynamics
```

Add `component` to each registry entry as the ground truth label.
`dead_surrogate_v1` is `policy_update`.

## Scoring for this sprint

- **`component`**: exact match against the registry. Binary. This is the score.
- **`failure_mode`**: logged as free text, **not scored**. Do not build the
  phrasing grader this sprint. It is the brittle part and it is not needed to
  answer the sprint question.
- **`evidence_metrics`**: logged, not scored.

Outcome reward does not apply in arm D, since there is no patch. The results
JSON should set `outcome: null` and `arm: "D"`.

## System prompt for arm D

Fixed across models. Convey: there is a bug in an RL training run you cannot see
the code for; the reported symptom is X; use the training and metrics tools to
diagnose it; emit the diagnosis JSON with a component drawn from the given list.
Do not hint at the bug category.

Keep the turn cap the same as arm A so the comparison is fair.

---

# Task 3: arm D sweep

Identical to task 1 in every respect except the arm.

- Same bug, same 5 instances, same 3 seeds, same models
- Arm D
- 75 episodes

**Deliverable: the arm A vs arm D comparison table.** This is the sprint result.

Note the scoring asymmetry when writing it up. Arm A is scored on outcome and
localization; arm D is scored on component identification. They are not the same
measurement. The honest comparison is **arm A localization vs arm D component
match**, since both ask "did the model find the right thing." State this
explicitly rather than comparing outcome to component.

---

# Task 4: qualitative read (do not skip)

Read at least 10 arm D transcripts by hand. Specifically check:

1. **Did the model call `get_metrics` at all, or did it guess from the symptom
   string?** If models guess without querying, that is its own finding and it
   changes how the arm D result should be interpreted. A correct component
   guessed from a symptom is not diagnosis.
2. How many training runs did it launch before committing to an answer?
3. Which metrics did it actually query, and did those match the registry's
   `diagnostic_metrics`?

Write two or three paragraphs of observations into the README. Include one
representative transcript excerpt.

Add a `queried_metrics_before_submit: true/false` field to the results JSON so
this is measurable and not only anecdotal.

---

# Task 5, stretch only: two bias-introducing CartPole bugs

Only start this if tasks 0 through 4 are complete and committed.

Unlike the omission candidates, these introduce systematic wrong signal rather
than removing regularization, so they should degrade CartPole and pass the 3x
noise calibration gate.

1. **`minibatch_tail_dropped_v1`** - the minibatch count does not evenly divide
   the rollout length, so the tail of every batch is silently discarded. Class:
   `interaction`. Component: `rollout_collection`.
2. **`envs_seeded_identically_v1`** - all parallel environments receive the same
   seed, so `num_envs` provides no sample diversity and the effective batch is
   one trajectory repeated N times. Class: `statistical`. Component:
   `rollout_collection`.

Each must pass the load-bearing pre-check and the standard calibration gate
before any eval is run against it.

---

# Analysis deliverable

`eval/analyze.py` must produce, from committed results:

1. Per-model outcome and localization on arm A, with standard error
2. Per-model component match rate on arm D, with standard error
3. **The arm A localization vs arm D component-match delta, per model.** This is
   the headline number.
4. `queried_metrics_before_submit` rate per model on arm D
5. Turn count and token cost distributions per arm
6. Hack attempt and `INVALID` counts

**Statistics discipline.** At 15 episodes per model per arm, the standard error
on a proportion near 50% is roughly 13 points. Do not describe differences under
about 26 points as real. State the n and the error bars on every table. If the
arm A to arm D gap is smaller than that, say so plainly rather than reaching for
a story.

---

# Sprint gate

The sprint is done when the README contains:

- Both findings written up, with real numbers
- The load-bearing design rule in the invariants
- A committed arm A results table across 4 to 5 models
- A committed arm D results table across the same models
- The A-vs-D delta stated, with error bars and the n
- A short qualitative section from the transcript read
- All transcripts and results JSON committed

If the arm D result is null, ambiguous, or contradicts the memorization
hypothesis, write that up as the finding. A recorded negative result is the
output. An unrecorded one is nothing.

---

# Standing rules, unchanged

- No LLM anywhere in the reward path
- Same instance plus same seed gives the same score
- The agent cannot touch the scorer; hash-check every episode
- No network access inside the agent container
- A bug that does not degrade performance is not a task
- A bug class is only valid where the affected mechanism is load-bearing
- Log every trajectory in full
