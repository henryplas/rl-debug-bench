# rl-debug-bench: project roadmap

A single execution document. Phase 0 replaces the README. Phases 1 through 5 are
the plan from the current state through publication and public release.

Work phases in order. Each phase ends with a gate. Do not start the next phase
until the gate is recorded in the README, including when the result is negative.
Negative results that are recorded are the project's output; negative results
that are not recorded are nothing.

Reference documents:
- `tasks/tasks-list.md` - v0 spec, still authoritative for harness and scoring
- `tasks/hardness-v1.md` - v1 detailed spec for bug classes and levers
- `tasks/roadmap.md` - this file

---

# Phase 0: replace the README

Delete the current `README.md` and write the following in its place, verbatim
except for the two marked placeholders. Fill `<<SMOKE_2_RESULT>>` with the actual
solve count from the second smoke eval before committing.

````markdown
# rl-debug-bench

An RL environment that evaluates language models on diagnosing and fixing silent
bugs in reinforcement learning training code: bugs that throw no error, fail no
test, and simply cause a policy to train badly or not at all.

Specs: [`tasks/tasks-list.md`](tasks/tasks-list.md) (v0 harness),
[`tasks/hardness-v1.md`](tasks/hardness-v1.md) (v1 difficulty),
[`tasks/roadmap.md`](tasks/roadmap.md) (full plan). This file is the running
status summary.

## Status

The v0 harness is complete and working. v0 difficulty failed its target twice.
The failure is diagnosed, and v1 is a redesign of the task substrate rather than
of the harness.

| Phase | What | Status |
| --- | --- | --- |
| 0 | v0 harness, calibration, scoring, one bug type | Done |
| 0 | Smoke eval difficulty checkpoint | Done, target missed twice, see finding |
| 1 | v1: unmemorized bases, harder bug classes, arm D | In progress |
| 2 | Training environment: RL fine-tuning against the bench | Not started |
| 3 | Scale: bug bank, model coverage, arm C | Not started |
| 4 | Writeup and submission | Not started |
| 5 | Public release and maintenance | Not started |

## Finding: v0 measured memorization, not diagnosis

This is the main result so far and the reason for the v1 redesign.

**Attempt 1.** `dead_surrogate_v1` was implemented as
`logratio = newlogprob - newlogprob`. This is a true zero gradient, not merely a
numerically small one, verified with a standalone autograd check. A
`claude-sonnet-4-5` smoke eval solved it 3/3, sometimes without running training
at all.

**Attempt 2.** The patch was rewritten to preserve the identical mechanism while
removing the visual tell: instead of a literal self-subtraction, the "old"
log-prob is recomputed with a redundant fresh forward pass. Solving it requires
recognizing that the old log-prob must come from the stored rollout buffer.
Calibration was re-run and came out byte-identical to attempt 1, confirming the
bug is mechanistically unchanged. Result: <<SMOKE_2_RESULT>>.

**Diagnosis.** The difficulty is not in the bug. It is in the substrate. Three
properties of the v0 design let a model shortcut diagnosis entirely:

1. The base is CleanRL single-file PPO, one of the most widely reproduced pieces
   of RL code in existence. The model has a memorized reference and is diffing
   the workspace against it rather than reasoning from symptoms.
2. The base is roughly 400 lines, so the whole search space fits in one
   `read_file` call. No hypothesis formation is required.
3. Every bug is a single wrong line, so a diff against memory localizes it.

**Corollary worth preserving.** Keep the CleanRL base and its instances. The same
bug injected into a memorized base and an unmemorized base, evaluated side by
side, measures the memorization effect directly. That is a result in its own
right and half of it is already built.

## Design

### Two bases, two bug families

| Base | `base_id` | Source | Bug family | Role |
| --- | --- | --- | --- | --- |
| CleanRL PPO on CartPole | `legacy_cleanrl` | vendored, unmodified | algorithm bugs | Easy tier, memorization control |
| Quadcopter env, SB3 PPO | `drone_v1` | adapted from github.com/henryplas/drone_rl, MIT, same author | environment bugs | Medium and hard tiers |

The drone base matters for two reasons. It has no memorized reference, because
the author wrote it. And it admits a qualitatively harder bug class: PPO has a
canonical correct form, but a quadcopter reward function does not. There is no
memorized ground truth for the observation layout, the termination condition,
the shaping terms, or the integrator conventions. Diagnosing an environment bug
requires reasoning from physics and from what the curves imply about behavior.

Because the drone base uses Stable-Baselines3, an unmodified trusted dependency,
the agent may assume the optimizer is correct and the fault is in the author's
code. That is realistic and it sharpens the task.

### Bug classes

- `wrong_line` - a single incorrect line. Easy tier only.
- `omission` - a missing operation. Nothing on screen to inspect.
- `interaction` - every line correct in isolation, wrong in combination with a
  config value or another module.
- `statistical` - only visible in the numbers across iterations.

### Difficulty gradient

A benchmark where everything scores 0 is as uninformative as one where
everything scores 1.

| Tier | Target solve rate | Typical composition |
| --- | --- | --- |
| easy | 80-100% | `legacy_cleanrl`, `wrong_line` |
| medium | 40-70% | `drone_v1`, `wrong_line` or `omission` |
| hard | 10-40% | `drone_v1`, `interaction` or `statistical` |

A tier that lands outside its band is retuned or reclassified, never deleted.

### Observation arms

| Arm | Access | Purpose |
| --- | --- | --- |
| A | files, `run_training`, stdout only | Floor |
| B | A plus `get_metrics`, `list_metric_keys` | Primary |
| C | B plus rendered plot images | Vision ablation, phase 3 |
| D | metrics and `run_training` only, no file access | Diagnosis-only hard mode |

## Invariants

No change may violate these.

1. **No LLM anywhere in the reward path.** Every component computable by a
   deterministic script.
2. **Determinism.** Same instance plus same seed gives the same score. Seed
   Python, NumPy, Torch. `torch.use_deterministic_algorithms(True)`, pinned
   `CUBLAS_WORKSPACE_CONFIG`.
3. **The agent cannot touch the scorer.** Scoring lives outside the writable
   workspace. Hash-check before and after every episode; mismatch marks the
   episode `INVALID`.
4. **No network access inside the agent container.**
5. **A bug that does not degrade performance is not a task.** The clean-vs-broken
   gap must exceed 3x the larger seed standard deviation or the instance is
   rejected.
6. **Difficulty comes from the bug and the search space, never from unreadable
   code.** Both bases stay idiomatic. A reviewer will check this.
7. **Log every trajectory in full.** Transcripts are a primary output.

## Running it

```
make install         # pip install -e ".[dev]"
make test-fast       # everything except real-training tests
make test            # full suite, real Docker training runs
python calibration/build_baselines.py    # regenerate calibration/baselines.json
```

Running a live episode needs a provider API key (`ANTHROPIC_API_KEY`, via
`harness.models.AnthropicAdapter`). Keys live in a gitignored `.env` and are
never committed.

## License

MIT. The vendored CleanRL base is MIT and unmodified. The drone environment is
adapted from github.com/henryplas/drone_rl, also MIT, same author.
````

**Phase 0 gate:** README committed with the real smoke eval number, and
`eval/transcripts/` populated with the v0 episodes. This is the highest-priority
item in the entire document. Do it before anything else.

---

# Phase 1: v1 hardness

Full detail in `tasks/hardness-v1.md`. Summary of the build order:

1. Registry schema migration. Add `base_id`, `tier`, `bug_class`, `file`,
   `train_steps`, `diagnostic_metrics`, `accepted_failure_phrasings`. Backfill
   existing entries.
2. Vendor the drone env as `base/drone_v1/`. Strip curriculum and video
   recording. Add `success_rate` and `mean_final_distance` to the metrics store.
3. **Wall clock work.** See section below. Gates everything after it.
4. First drone bug, `wrong_line`, in the integrator. Calibrate.
5. **Checkpoint: smoke eval, 3 seeds, compare against the `legacy_cleanrl`
   ceiling.** The delta is the memorization measurement. Record it.
6. Two more drone bugs, one `omission` and one `interaction`.
7. Arm D harness and programmatic grader.
8. Full sweep: base x tier x arm, 3 seeds per cell.

## The wall clock problem

The drone environment is slow to train. This is the biggest risk in phase 1 and
must be resolved before the bug bank is built out.

1. **Set the budget from divergence, not convergence.** You are not training a
   drone, you are producing a diagnostic signal. Instrument
   `build_baselines.py` to log the running clean-vs-broken gap per step and
   report the earliest step count `T` where the gap clears 3x seed noise and
   stays clear. Use `T` as `train_steps`. Expect a small fraction of
   steps-to-convergence, because a broken gradient path shows up early.
2. **Profile before optimizing.** If the env steps one instance at a time in a
   Python loop, physics is the bottleneck and the network is irrelevant. Batch
   the dynamics so all N envs integrate in one array operation.
3. **Shrink the task, not the fidelity.** Fixed target, short range, no
   curriculum, shorter horizon. Keep the physics honest, because the physics is
   where the interesting bugs live. Document it as a benchmark variant.
4. **Shrink the network.** 64x64 is very likely enough for waypoint following.
5. **Cache training runs by workspace hash.** Agents call `run_training` on
   unmodified code before touching anything, and many converge on identical
   fixes. Hash the workspace, serve stored metrics on a hit. The calibration runs
   seed the cache for free.

**Fallback if still slow.** Keep `legacy_cleanrl` as the bulk substrate and use
`drone_v1` for 3 bugs at 3 instances each, reported as a small hard tier with the
n stated honestly. Nine expensive episodes is affordable at 15 minutes per run;
45 is not.

**Phase 1 gate:** at least one bug empirically in each of the three tiers, the
memorized-vs-unmemorized delta recorded, arm D working with a programmatic
grader, and a full sweep committed to `eval/results/`.

---

# Phase 2: training environment

This is the phase that changes what the project is. Everything before it produces
an eval. This produces an RL environment, which is a materially stronger artifact
and the thing the project is named for.

## Why it matters

An eval measures. An environment improves. The claim "models are bad at
diagnosing silent RL bugs" is interesting; the claim "models are bad at this, and
here is a verifiable-reward environment that measurably improves them" is a
different tier of result. It also demonstrates the full loop: environment design,
reward design, training, evaluation.

## Requirements the existing design already satisfies

The v0 and v1 scoring was built for this, so most of the work is done:

- Programmatic reward, no LLM judge, so no reward-model drift
- Dense outcome signal, normalized to [0,1], rather than pure binary
- Procedural instance generation, so a train/test split is possible
- Deterministic scoring, so gradient estimates are not corrupted by scorer noise

## Build order

1. **Instance split.** Partition bug types, not instances, into train and held
   out. Splitting by instance leaks the bug mechanism across the split and
   inflates every number. Reserve at least two bug types and one entire base for
   held out. Record the split in `bugs/splits.yaml` and never train on held out.

2. **Scale the training bank.** RL fine-tuning needs far more than 15 instances.
   Target 200 or more training instances via heavier procedural variation:
   instance seed, symptom phrasing, surrounding-code cosmetics, injection site
   within the same bug class, and config permutations. The held-out set stays
   small and hand-curated.

3. **Rollout throughput.** This is the engineering bottleneck. Each rollout is a
   multi-turn agentic episode with container startup and real training runs. Need
   parallel containers, aggressive workspace-hash caching, and a shorter
   `train_steps` for the training split specifically. Budget for this being the
   dominant cost of the phase.

4. **Algorithm.** GRPO is the right default: no value network, group-relative
   advantages, and it handles the sparse multi-turn credit assignment reasonably.
   Use a small open model, 7B to 14B class, with LoRA. Full fine-tuning is not
   necessary to demonstrate the effect.

5. **Reward shaping for training.** The eval reward is
   `outcome` and `localization` reported separately. For training, combine them
   with a small weight on localization and a per-turn cost to discourage
   thrashing. Log all components separately so reward hacking is visible.

6. **Reward hacking watch.** With a real optimizer in the loop this stops being
   hypothetical. The integrity check from v0 must run on every rollout and any
   `INVALID` episode gets a reward of zero rather than being dropped, or the
   model learns that tampering is free. Track the hack attempt rate as a training
   curve. If it climbs, that is a finding worth writing up on its own.

7. **Evaluation.** Base model vs trained model on the held-out split, 3 seeds.
   Also evaluate on the training split to quantify the generalization gap. Report
   both. A model that improves on held-out bug types has learned diagnosis; one
   that improves only on trained types has memorized patches.

**Phase 2 gate:** a trained checkpoint that beats its base model on the held-out
split by more than the seed noise margin, with the training-vs-held-out
generalization gap reported and the hack attempt curve committed.

If it does not beat base, that is publishable too, provided the training curve
and the failure analysis are honest. Do not hide it.

---

# Phase 3: scale and coverage

Run in parallel with phase 2 where compute allows.

1. **Bug bank to 12 to 15 types.** Full coverage of the four bug classes on both
   bases. Every addition still passes the 3x noise calibration gate.

2. **Model coverage.** At least 5 frontier models across at least 3 providers,
   3 seeds each, all arms. Cost-control this with the workspace hash cache and a
   per-model spend cap.

3. **Arm C, the vision ablation.** Render matplotlib figures of the metric
   history and pass them as images. The research question is whether models need
   the plot or whether the numbers suffice, given that human experts diagnose
   these bugs visually. Prior expectation is that arm C helps less than people
   assume and arm B is where the capability lives, but measure it.

4. **Multi-bug instances.** Two independent bugs in one workspace. Tests whether
   a model stops at the first fix it finds. Scoring needs a partial-credit rule:
   report per-bug localization, not an aggregate.

5. **Turn budget sweep.** Solve rate as a function of the turn cap, per tier.
   Distinguishes models that are slow from models that are stuck, which is a
   distinction the single-number result hides.

**Phase 3 gate:** the full results matrix committed, with per-category breakdowns
and error bars, and `eval/analyze.py` regenerating every figure from a clean
checkout.

---

# Phase 4: writeup and submission

## The claims the project can support

Write the paper around whichever of these the data actually supports. Do not
decide in advance.

1. **Frontier models diagnose silent RL training bugs poorly, and the failure is
   concentrated in specific bug classes.** The per-class breakdown is the result,
   not the aggregate.
2. **Benchmark difficulty in code-repair evals is confounded by reference
   memorization.** Same bug, two bases, measured delta. This is a methodological
   contribution and it generalizes beyond RL. It may be the most broadly useful
   thing here.
3. **Models restore performance without identifying the cause.** The
   outcome-minus-localization gap, quantified.
4. **Diagnosis from training dynamics alone is much harder than diagnosis with
   code access.** Arm D vs arm B.
5. **Targeted RL training on a verifiable-reward debugging environment
   generalizes to held-out bug types.** Phase 2, if it lands.

Claim 2 is the one a methods-minded reviewer will find most interesting and it is
the cheapest to establish rigorously. Do not bury it.

## Mechanics

- Target a workshop first. NeurIPS and ICLR workshop tracks on evaluation,
  agentic ML, or ML for code are the right venues. 4 to 8 pages, real review,
  fast turnaround.
- arXiv preprint at submission time.
- Every number in the paper regenerated by `make repro` from the committed
  results. State the seed count and the standard error everywhere. At n=45 per
  cell, differences under roughly 15 points are not real; say so explicitly.
- Include the negative results. The two failed v0 smoke evals are part of the
  methodology section, not an embarrassment.
- Release-review clearance before any submission, given the author's employment.
  Start that process early, it is slower than expected.

**Phase 4 gate:** preprint posted, workshop submission filed.

---

# Phase 5: public release and maintenance

1. **Contamination control.** Keep a private held-out split that is never
   published. Publish the generator and the public split. Without this the
   benchmark is dead within a year of anyone caring about it.

2. **Versioning.** Tag benchmark releases (`v1.0`, `v1.1`). Results are only
   comparable within a version. State the version in every reported number.

3. **Contribution path.** `CONTRIBUTING.md` specifying what a new bug requires:
   a patch that applies to a pristine base, a registry entry with ground truth
   lines, a passing calibration record, and a tier assignment validated against
   at least one model. Make it mechanical so contributions do not require your
   judgment.

4. **Leaderboard.** Only after the results matrix is stable. A leaderboard on a
   moving benchmark is worse than none.

5. **Packaging.** Publish to PyPI and mirror the instance set as a HuggingFace
   dataset. Both are low effort and both meaningfully increase the chance anyone
   actually runs it.

---

# Gates summary

| Phase | Gate |
| --- | --- |
| 0 | README committed with the real smoke eval number, transcripts committed |
| 1 | One bug empirically in each tier, memorization delta recorded, arm D working, full sweep committed |
| 2 | Trained checkpoint beats base on held out beyond noise, generalization gap and hack curve reported |
| 3 | Full results matrix with per-category breakdowns, all figures regenerable |
| 4 | Preprint posted, workshop submission filed |
| 5 | Private split held back, versioned release, contribution path documented |

# Standing rules

- Record every gate result in the README, including negative ones, before
  starting the next phase.
- Never delete the easy tier or the legacy base. They are the control.
- Never put an LLM in the reward path.
- Never train on the held-out split.
- Every new bug passes the 3x noise calibration gate or it is not a task.
- When a phase result contradicts the plan, the plan changes. Rewrite this
  document rather than working around it.
