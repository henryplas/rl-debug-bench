# rl-debug-bench

An RL environment that evaluates language models on diagnosing and fixing silent
bugs in reinforcement learning training code: bugs that throw no error, fail no
test, and simply cause a policy to train badly or not at all.

Specs: [`tasks/tasks-list.md`](tasks/tasks-list.md) (v0 harness spec),
[`tasks/hardness-v1.md`](tasks/hardness-v1.md) (v1 difficulty redesign),
[`tasks/roadmap.md`](tasks/roadmap.md) (full project plan),
[`tasks/weekend-sprint.md`](tasks/weekend-sprint.md) (current, scoped sprint —
supersedes the other two for its duration). This file is the running status
summary.

## Status

The v0 harness (container, tools, episode loop, scoring) is complete. v0
difficulty missed its target twice. The failure is diagnosed (see Findings
below), and the current work — the weekend sprint — tests that diagnosis
directly by comparing an arm with file access against one without it, on the
same bug, same models.

| Phase | What | Status |
| --- | --- | --- |
| v0 | Harness, calibration, scoring, `dead_surrogate_v1` (one bug type) | Done |
| v0 | Smoke eval difficulty checkpoint | Done, target missed twice, see Finding 1 |
| v1 lever 1 | Modular (de-memorized) base, legacy-vs-modular checkpoint | Done, see Finding 1 |
| v1 lever 3a | Omission-bug candidates on the modular base | Done, negative result, see Finding 2 |
| sprint task 1 | Multi-model arm A sweep, `dead_surrogate_v1`, 5 instances x 3 seeds | Done, see Finding 3 |
| sprint task 2 | Minimal arm D (diagnosis-only, no file access) | Done |
| sprint task 3-4 | Arm D sweep, qualitative transcript read | Not started |

## Findings

### Finding 1: v0 measured memorization, not diagnosis

**Attempt 1.** `dead_surrogate_v1` was implemented as
`logratio = newlogprob - newlogprob`. This is a true zero gradient, not merely a
numerically small one, verified with a standalone autograd check. A
`claude-sonnet-4-5` smoke eval (3 episode seeds) solved it **3/3**, sometimes
without running training at all.

**Attempt 2.** The patch was rewritten to preserve the identical mechanism while
removing the visual tell: instead of a literal self-subtraction, the "old"
log-prob is recomputed with a redundant fresh forward pass. Solving it requires
recognizing that the old log-prob must come from the stored rollout buffer, not
a fresh policy evaluation. Calibration was re-run and came out byte-identical to
attempt 1's, confirming the bug is mechanistically unchanged. Result: **3/3**
again.

**Attempt 3, the legacy-vs-modular checkpoint (`tasks/hardness-v1.md` lever 1).**
The base was reimplemented as a 7-module package (`base/modular_v1/`) with
de-CleanRL'd internal naming and structure, verified deterministic and
equivalent to the legacy base within seed noise (clean baseline 176.8 vs.
legacy's 175.5 at matched seeds). The same bug, ported unchanged, was smoke-evaled
on both bases, 3 seeds each: legacy **3/3**, modular **3/3**, delta **0.000**.
The model needed substantially more exploration on the modular base (8
`read_file` calls across the module structure vs. 1-2 on the single legacy
file, visible directly in the transcripts) but still found and fixed the bug
every time.

**Diagnosis.** De-memorizing the code did not move the needle, which sharpens
rather than overturns the original hypothesis: the model isn't shortcutting via
a literal diff against a memorized reference file. It's recognizing a
well-understood PPO implementation mistake ("the old log-prob must come from
the stored rollout policy, not a fresh recompute") from domain knowledge of the
algorithm, independent of code layout or naming. Three properties of the
substrate let this happen regardless of memorization:

1. The bug is a single wrong line (`bug_class: wrong_line`), and once a model
   understands PPO, that class of mistake is recognizable on sight regardless
   of variable names.
2. The base fits inside a handful of `read_file` calls even split across
   modules, so exhaustive reading is still cheap.
3. There is a well-known canonical correct form for "the PPO ratio," so there
   is always a reference to reason against, memorized or not.

### Finding 2: non-load-bearing omissions don't degrade CartPole

Four omission-bug candidates were built on the modular base and tested across 3
seeds each at the 40k-timestep calibration budget: **gradient clipping
removed**, **entropy bonus dropped from the total loss**, **advantage
normalization removed**, and **a stale (one-step-old) bootstrap value reused
instead of a fresh critic evaluation**. All four *improved* mean final return
relative to clean training, at every seed tested. None were viable bug
instances.

This is the expected result, not a fluke. Gradient clipping, the entropy bonus,
and advantage normalization are variance-reduction and stability machinery that
pays off on hard or unstable tasks and costs throughput on easy, well-scaled
ones. CartPole-v1 has two actions, dense uniform reward, essentially no
exploration problem, and advantages that are already well-scaled — removing
safety machinery that isn't doing any work cannot degrade the task, it just
lets the optimizer move faster. (The stale-bootstrap case is a plausible bias
source in principle, but CartPole's short, mostly-truncated episodes and small
value-function error apparently don't make that bias large enough to matter at
this budget either.)

**Do not retry these candidates at a longer training budget.** CartPole-v1 caps
episodic return at 500; both the clean and broken arms saturate near the cap
given enough steps, producing a ceiling effect that hides any real degradation
rather than revealing one — more budget makes the comparison *less*
informative, not more.

**Design rule (added to Invariants below):** a bug class is only a valid task
on a base where the affected mechanism is load-bearing. Ablate the mechanism on
the clean base and confirm it measurably degrades the clean baseline *before*
writing an omission or interaction patch — this check is cheap and runs before
calibration, and it would have saved four failed attempts here if it had
existed first.

### Finding 3: two model tiers are statistically indistinguishable on this bug (arm A)

Sprint task 1: `dead_surrogate_v1`, all 5 instances, 3 episode seeds each, arm A,
against `claude-haiku-4-5` and `claude-sonnet-4-5` (15 episodes per model, 30
total; `claude-opus-4-8` was skipped after sonnet's results came back —
see below).

| Model | n | Outcome | Localization | Turns used |
| --- | --- | --- | --- | --- |
| `claude-haiku-4-5` | 15 | 0.950 ± 0.008 | 1.000 ± 0.000 | 7.7 |
| `claude-sonnet-4-5` | 15 | 0.950 ± 0.008 | 1.000 ± 0.000 | 6.1 |

All 30 episodes: `status: OK`, `hack_attempt: false`. Standard error is over
seeds within each model; at n=15 it is small enough here that the two models'
outcome means agree to three decimal places, and localization has zero
variance in either group — every single episode identified and fixed exactly
the right line.

**Opus was not run.** The sprint doc calls for 4-5 models spanning a
capability range, but haiku (the fast/cheap floor) and sonnet came back
statistically indistinguishable from each other rather than showing the
expected capability gradient. Running the frontier model too would almost
certainly extend the same ceiling rather than add information, at real
additional cost, so the sweep was stopped after sonnet. This is itself
informative: it means the "floor" this bug provides is not actually a floor
in the useful sense — even the cheapest model tested saturates it, so
`dead_surrogate_v1` under arm A cannot discriminate between these models at
all, consistent with Findings 1 and 2's shared diagnosis that the substrate,
not the models, sets the ceiling here.

## Invariants

No change may violate these.

1. **No LLM anywhere in the reward path.** Every scoring component is
   computable by a deterministic script.
2. **Determinism.** Same instance plus same seed gives the same score. Seed
   Python, NumPy, Torch; pin `CUBLAS_WORKSPACE_CONFIG`.
3. **The agent cannot touch the scorer.** Scoring lives outside the writable
   workspace. Hash-check before and after every episode; a mismatch marks the
   episode `INVALID`.
4. **No network access inside the agent container.**
5. **A bug that does not degrade performance is not a task.** The clean-vs-broken
   gap must exceed 3x the larger seed standard deviation, or the instance is
   rejected.
6. **A bug class is only valid on a base where the affected mechanism is
   load-bearing.** Ablate the mechanism on the clean base and confirm it
   measurably degrades the clean baseline before writing an omission or
   interaction patch. If it does not degrade the baseline, the task is wrong,
   not the bug — this check is cheap and runs before calibration. (Added after
   Finding 2.)
7. **Difficulty comes from the bug and the search space, never from unreadable
   code.** Bases stay idiomatic; obfuscation is not a valid difficulty source.
8. **Log every trajectory in full.** Transcripts are a primary output, not a
   byproduct.

## Repository layout

```
base/legacy_cleanrl/   vendored, unmodified CleanRL PPO -- the memorization control
base/modular_v1/       7-module reimplementation, same algorithm, de-CleanRL'd naming
bugs/                  registry.yaml + patches/
calibration/           build_baselines.py + committed baselines.json
harness/               container lifecycle, tools, episode loop, metrics store, model adapters, base registry
scoring/                outcome, localization, integrity/hack-detection, score_episode entrypoint
tests/                 pytest suite (fast tests + a `slow` marker for real-training tests)
eval/                  transcripts/ and results/ (committed episode logs)
```

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

MIT. The vendored CleanRL base (`base/legacy_cleanrl/`) is MIT and unmodified.
