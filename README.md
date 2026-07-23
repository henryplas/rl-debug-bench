# rl-debug-bench

An RL environment that evaluates language models on diagnosing and fixing silent
bugs in reinforcement learning training code: bugs that don't throw errors or
fail a test, they just cause a policy to train badly or not at all.

The full v0 design spec and build order this project is being built against
lives in [`tasks/tasks-list.md`](tasks/tasks-list.md). This file is a running
status summary — read the tasks list for the actual specification.

## Status: build order steps 1–5 done, step 6 (smoke eval) in progress

| Step | What | Status |
|---|---|---|
| 1 | Repo skeleton, vendored CleanRL PPO base, determinism tests | Done |
| 2 | `dead_surrogate_v1` bug patch + registry entry | Done (rewritten once, see below) |
| 3 | Calibration (clean vs. broken baseline, noise threshold) | Done — all 5 instances accepted with wide margins |
| 4 | Harness: Docker container, tools, episode loop, transcripts (arm A) | Done |
| 5 | Scoring: outcome, localization, integrity/hack-detection | Done |
| 6 | Smoke eval: one model, one bug, 3 seeds — the difficulty checkpoint | In progress |

### Where step 6 stands

The README's own build order treats this as a hard stop: test difficulty
after the *first* bug type, before building anything else, because clearing
~80% overall means the bug bank is too easy and needs to be harder before
more is built on top of it.

The first version of `dead_surrogate_v1` (`logratio = newlogprob -
newlogprob`) was mathematically correct — it produces a true zero gradient,
not just a numerically-zero one — but a `claude-sonnet-4-5` smoke eval solved
it 3/3 times, sometimes without even running training, because the bug is a
visible code smell (a variable subtracted from itself) rather than something
that requires behavioral diagnosis.

The patch has been rewritten to use the same underlying mechanism (verified
with a standalone autograd check) expressed as a redundant, non-obvious
recompute of the "old" log-prob instead of a literal self-subtraction — it
now requires recognizing that the old log-prob should come from the stored
rollout buffer, not a fresh forward pass. Calibration was re-run and is
byte-identical to the original patch's (confirming it's the same bug
mechanistically). A second smoke eval against the revised patch is the
current in-flight work.

## Repository layout

See `tasks/tasks-list.md` section 4 for the intended full layout. As built so
far:

```
base/        vendored, unmodified CleanRL PPO (base/ppo_cartpole.py)
bugs/        registry.yaml + patches/ (one bug type so far: dead_surrogate_v1)
calibration/ build_baselines.py + committed baselines.json
harness/     container lifecycle, tools, episode loop, metrics store, model adapters
scoring/     outcome, localization, integrity/hack-detection, score_episode entrypoint
tests/       pytest suite (fast tests + a `slow` marker for real-training tests)
eval/        transcripts/ and results/ (populated once episodes are run)
```

## Running it

```
make install        # pip install -e ".[dev]"
make test-fast       # everything except real-training tests
make test            # full suite, ~7 min (real Docker training runs)
python calibration/build_baselines.py   # regenerate calibration/baselines.json
```

Running an actual episode against a live model needs a provider API key
(currently `ANTHROPIC_API_KEY`, used via `harness.models.AnthropicAdapter`).
Keys are kept in a local, gitignored `.env` file — never committed.
