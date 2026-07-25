# rl-debug-bench

An RL environment that evaluates language models on diagnosing and fixing silent
bugs in reinforcement learning training code: bugs that throw no error, fail no
test, and just cause a policy to train badly or not at all. Existing code-repair
benchmarks test bugs that announce themselves through failing tests or
tracebacks; silent numerical failures in training loops do not.

## Status

| Phase | What | Status |
| --- | --- | --- |
| v0 | Harness: container, tools, episode loop, scoring | Done |
| v0 | Calibration (`dead_surrogate_v1`, `logprob_broadcast_v1`, `dead_surrogate_v1_modular`) | Done, 11/11 instances accepted |
| v0 | Smoke eval difficulty checkpoint | Done, target missed twice, see Finding 1 |
| v1 lever 1 | Modular (de-memorized) base, legacy-vs-modular checkpoint | Done, see Finding 1 |
| v1 lever 3a | Omission-bug ablations on the modular base | Done, 0/4 accepted, see Finding 2 |
| sprint task 0 | Record findings, commit transcripts | Done |
| sprint task 1 | Multi-model arm A sweep | Done |
| sprint task 2 | Minimal arm D (diagnosis-only, no file access) | Done |
| sprint task 3 | Arm D sweep | Done |
| sprint task 4 | Qualitative read of arm D transcripts | Not started |
| sprint analysis | `eval/analyze.py` | Not started |

Nothing is currently running. "Not started" rows above are exactly that, not
work stalled mid-flight.

## Findings

### Finding 1: v0 measured memorization, not diagnosis

**Attempt 1.** `dead_surrogate_v1` as `logratio = newlogprob - newlogprob`: a
true zero gradient, not a numerically small one, verified with a standalone
autograd check. A `claude-sonnet-4-5` smoke eval, 3 episode seeds, solved it
3/3, sometimes without running training at all.

**Attempt 2.** Rewrote the patch to preserve the identical mechanism while
removing the visual tell: the "old" log-prob is recomputed with a redundant
fresh forward pass instead of a literal self-subtraction, requiring
recognition that it must come from the stored rollout buffer. Calibration
came out byte-identical to attempt 1's. Result: 3/3 again.

**Attempt 3, the legacy-vs-modular checkpoint.** Reimplemented the base as a
7-module package (`base/modular_v1/`) with de-CleanRL'd naming and
structure, verified deterministic and equivalent to the legacy base within
seed noise (clean baseline 176.8 vs. legacy's 175.5 at matched seeds). The
same bug, ported unchanged, was smoke-evaled on both bases, 3 seeds each:
legacy 3/3, modular 3/3, delta 0.000. The model needed far more exploration
on the modular base (8 `read_file` calls across modules vs. 1-2 on the single
legacy file) but still found and fixed the bug every time.

**Diagnosis.** De-memorizing the code did not move the needle. The model is
not shortcutting via a literal diff against a memorized reference file; it
recognizes a well-understood PPO mistake (the old log-prob must come from
the stored rollout policy, not a fresh recompute) from domain knowledge of
the algorithm, independent of code layout or naming. That class of mistake
is recognizable on sight once a model understands PPO, the base fits inside
a handful of `read_file` calls even split across modules, and there is
always a canonical correct form for "the PPO ratio" to reason against,
memorized or not.

### Finding 2: omission bugs do not degrade a task where the omitted mechanism is not load-bearing

Four omission-bug candidates were built on the modular base and calibrated
against the same 40k-timestep, CartPole-v1 budget as the other instances,
3 seeds each. Every one improved the clean baseline instead of degrading it,
and every one was rejected:

| Bug | Mechanism removed | Clean baseline | Broken baseline | Margin | Threshold | Result |
| --- | --- | --- | --- | --- | --- | --- |
| `grad_clip_omitted_v1` | Gradient clipping | 176.8 | 318.0 | -141.2 | 140.0 | Rejected |
| `entropy_omitted_v1` | Entropy bonus in the total loss | 176.8 | 208.8 | -32.0 | 37.8 | Rejected |
| `adv_norm_omitted_v1` | Advantage normalization | 176.8 | 214.0 | -37.1 | 51.6 | Rejected |
| `stale_bootstrap_v1` | Fresh bootstrap value (reused a one-step-stale value instead) | 176.8 | 211.4 | -34.6 | 85.8 | Rejected |

Margin is clean minus broken; a bug must clear a positive margin exceeding
the threshold (3x the larger seed standard deviation) to be accepted. All
four margins are negative: broken outperformed clean at every seed tested.

This is the expected result, not a fluke. Gradient clipping, the entropy
bonus, and advantage normalization are variance-reduction and stability
machinery that pays off on hard or unstable tasks and costs throughput on
easy, well-scaled ones. CartPole-v1 has two actions, dense uniform reward,
no real exploration problem, and advantages already at reasonable scale, so
removing machinery that is not doing any work cannot degrade the task; it
just lets the optimizer move faster. Stale bootstrapping is a plausible bias
source in principle, but CartPole's short, mostly-truncated episodes and
small value-function error do not make it large enough to matter here.

Longer budgets were considered and rejected as a way to retry these:
CartPole-v1 caps episodic return at 500, so both arms saturate near the cap
given enough steps, hiding degradation rather than revealing it.

## What the findings imply

Both findings point the same direction: the substrate is too easy, not that
the bugs are wrong. `dead_surrogate_v1` is solved at ceiling because a
well-known algorithm mistake is recognizable regardless of memorization or
code layout, and omission bugs cannot degrade a task where the omitted
mechanism was never load-bearing to begin with. The direction implied is
bases with no memorized reference and tasks where the affected mechanisms
are genuinely load-bearing, which is what `tasks/roadmap.md` lays out in
full. The current, narrower, time-boxed slice of that plan is
`tasks/weekend-sprint.md`: test the memorization hypothesis directly by
comparing an arm with file access against one without it, same bug, same
models.

## Design

Compact summary; see `tasks/roadmap.md` and `tasks/hardness-v1.md` for the
full detail behind each of these.

**Bases.** `legacy_cleanrl` (vendored, unmodified CleanRL PPO) is built and
serves as the easy tier and memorization control. `drone_v1` (a quadcopter
environment, adapted from a separate MIT-licensed repository) is planned for
environment-level bugs with no memorized reference, and is not built yet.

**Bug classes.** `wrong_line` (a single incorrect line), `omission` (a
missing operation, nothing on screen to inspect), `interaction` (every line
correct in isolation, wrong only in combination with a config value or
another module), `statistical` (only visible in the numbers across
iterations).

**Observation arms.**

| Arm | Access | Implemented |
| --- | --- | --- |
| A | Files, `run_training`, stdout only | Yes |
| B | A plus `get_metrics`, `list_metric_keys` | Yes |
| C | B plus rendered plot images | No |
| D | `run_training` and metrics tools only, no file access | Yes |

**Difficulty gradient.** A benchmark where every task scores 0 is as
uninformative as one where every task scores 1.

| Tier | Target solve rate | Typical composition |
| --- | --- | --- |
| Easy | 80-100% | `legacy_cleanrl`, `wrong_line` |
| Medium | 40-70% | Unmemorized base, `wrong_line` or `omission` |
| Hard | 10-40% | Unmemorized base, `interaction` or `statistical` |

## Results

### Arm A: multi-model sweep, `dead_surrogate_v1`

All 5 instances, 3 episode seeds each, arm A. `claude-opus-4-8` was skipped
after `claude-sonnet-4-5`'s results came back statistically indistinguishable
from `claude-haiku-4-5`'s.

| Model | n | Outcome | Localization |
| --- | --- | --- | --- |
| `claude-haiku-4-5` | 15 | 0.950 +/- 0.008 | 1.000 +/- 0.000 |
| `claude-sonnet-4-5` | 15 | 0.950 +/- 0.008 | 1.000 +/- 0.000 |

All 30 episodes: `status: OK`, `hack_attempt: false`. Localization has zero
variance in either group: every episode identified and fixed exactly the
right line.

### Arm D: same bug, no file access

Same instances and seeds, arm D, same two models. `component_match` is exact
match of the submitted diagnosis's component against the registry ground
truth (`policy_update`).

| Model | n | Component match | Queried metrics before submit |
| --- | --- | --- | --- |
| `claude-haiku-4-5` | 15 | 0.667 +/- 0.122 | 1.000 |
| `claude-sonnet-4-5` | 15 | 0.333 +/- 0.122 | 1.000 |

All 30 episodes: `status: OK`, `hack_attempt: false`. Every episode queried
`get_metrics` or `list_metric_keys` at least once before submitting; this was
not a cold guess from the symptom string in any episode.

**The headline comparison: arm A localization vs. arm D component match**,
since both ask "did the model find the right thing," not "did it fix it"
(arm D has no patch to re-run, so outcome does not apply there). Localization
is 1.000 for both models; component match drops to 0.667 (haiku) and 0.333
(sonnet). Removing file access produces a real, measurable gap on this bug,
unlike the omission-bug ablations above, where nothing moved.

**Statistics discipline.** At n=15 per model per arm, the standard error on a
proportion near 50% is `sqrt(0.5 x 0.5 / 15)` = ~0.129, about 13 points.
Differences under roughly 26 points (2x that) should not be read as
reliable. The arm A vs. arm D gap for haiku (33 points) and sonnet (67
points) both clear that bar; the haiku vs. sonnet difference within arm D
(33 points) is right at the edge of it.

**A genuine nuance, not noise.** Both models' wrong guesses cluster on
`advantage_estimation` rather than scattering randomly, and that is not a
careless guess: a frozen policy alongside a normally-learning value function
(zero KL, zero clip fraction, near-zero policy loss, rising explained
variance) is consistent with either a dead policy-gradient term (the actual
bug, `policy_update`) or collapsed advantages feeding that same term
(`advantage_estimation`). Sonnet's stated reasoning for the alternative was
well-argued in the transcripts read so far. The single fixed ground-truth
label may be underdetermined by the current `diagnostic_metrics` for this
bug, which is worth recording rather than reading as "sonnet performed worse
than haiku."

## Invariants

No change may violate these.

1. **No LLM anywhere in the reward path.** Every scoring component is
   computable by a deterministic script.
2. **Determinism.** Same instance plus same seed gives the same score. Seed
   Python, NumPy, Torch; pin `CUBLAS_WORKSPACE_CONFIG`.
3. **The agent cannot touch the scorer.** Scoring lives outside the writable
   workspace. Hash-check before and after every episode; a mismatch marks
   the episode `INVALID`.
4. **No network access inside the agent container.**
5. **A bug that does not degrade performance is not a task.** The
   clean-vs-broken gap must exceed 3x the larger seed standard deviation, or
   the instance is rejected.
6. **A bug class is only valid on a base where the affected mechanism is
   load-bearing.** Before writing an omission or interaction patch, ablate
   the mechanism on the clean base and confirm it measurably degrades the
   clean baseline. If it does not, the task is wrong, not the bug. This
   check runs before calibration and is cheap.
7. **Difficulty comes from the bug and the search space, never from
   unreadable code.** Bases stay idiomatic; obfuscation is not a valid
   difficulty source.
8. **Log every trajectory in full.** Transcripts are a primary output, not a
   byproduct.

## Repository layout

Every directory below exists in the pushed tree.

```
base/legacy_cleanrl/   vendored, unmodified CleanRL PPO -- the memorization control
base/modular_v1/       7-module reimplementation, same algorithm, de-CleanRL'd naming
bugs/                  registry.yaml + patches/
calibration/           build_baselines.py + committed baselines.json
harness/               container lifecycle, tools, episode loop, metrics store, model adapters, base registry
scoring/               outcome, localization, integrity/hack-detection, score_episode entrypoint
tests/                 pytest suite (fast tests + a `slow` marker for real-training tests)
eval/                  transcripts/ and results/, committed episode logs (63 episodes as of this writing)
tasks/                 spec and planning documents
```

## Running it

```
make install         # pip install -e ".[dev]"
make test-fast       # everything except real-training tests
make test            # full suite, real Docker training runs
python calibration/build_baselines.py    # regenerate calibration/baselines.json
```

`make repro` is not implemented yet; it currently exits with an error
pointing back to this file rather than pretending to run a sweep that does
not exist end to end.

Running a live episode needs a provider API key (`ANTHROPIC_API_KEY`, via
`harness.models.AnthropicAdapter`). Keys live in a gitignored `.env` and are
never committed. There is no pre-commit hook scanning for a leaked key yet;
every commit in this repository has been checked by hand before pushing, but
an automated `grep -r "sk-ant-"` pre-commit hook would be a reasonable
addition and does not exist today.

## Specs and plan

- [`tasks/tasks-list.md`](tasks/tasks-list.md): the v0 spec, harness and
  scoring design, still authoritative for those pieces.
- [`tasks/hardness-v1.md`](tasks/hardness-v1.md): the v1 difficulty redesign,
  bug classes and levers.
- [`tasks/roadmap.md`](tasks/roadmap.md): the full project plan, phases 0
  through 5.
- [`tasks/weekend-sprint.md`](tasks/weekend-sprint.md): the current, scoped
  sprint that produced the arm A vs. arm D results above.

## License

MIT. The vendored CleanRL base (`base/legacy_cleanrl/`) is MIT and
unmodified.
