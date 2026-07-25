# Task: rewrite README.md

A specification for the repo's README. Follow it section by section.

**Why this task exists.** The repo has real results sitting on the local machine
and none of them are visible publicly. The README still says "step 6 in
progress" and `eval/` does not exist in the pushed tree. The work is invisible,
which for a public research artifact is the same as it not existing.

**This task is not "polish the README."** It is "make the repository state
honest and legible to a reader who arrives cold." Most of the work is finding
numbers that already exist and writing them down.

---

## Rule 0: numbers come from artifacts, never from memory

Every quantitative claim in the README must trace to a committed file in
`eval/results/`, `eval/transcripts/`, or `calibration/baselines.json`.

- **Do not invent, estimate, or approximate any number.**
- **Do not write a number you cannot point at a file for.**
- If an experiment was run but the artifact was not saved, say so explicitly in
  the README rather than reporting a remembered number: "run not preserved,
  needs rerun."
- If an experiment was not run, say it was not run. Do not write "in progress"
  for something that has stalled. "In progress" is only valid for work actively
  underway right now.

A README that says "we have not measured this yet" is credible. A README with an
unverifiable number in it is not.

---

## Step 1: inventory before writing

Before touching the README, produce a written inventory. Put it in the PR
description or a scratch file, not in the repo.

1. `git status` and `git log origin/master..HEAD`. List what is uncommitted and
   what is committed but unpushed.
2. Find every episode artifact on disk, wherever it landed: results JSON,
   transcripts, logs from smoke evals, calibration output from the omission
   experiments.
3. For each, record: which bug, which patch version, which model, how many
   seeds, what the outcome and localization scores were, and whether the
   artifact is currently in the repo.

**This inventory determines what the README can say.** Write it first.

---

## Step 2: commit the artifacts

`eval/` does not exist in the pushed repository. Create it and populate it.

- `eval/results/` - one JSON per episode, per the v0 schema
- `eval/transcripts/` - full trajectory logs, one per episode

Commit every episode that was actually run, including the ones that failed,
were `INVALID`, or were solved trivially. Especially those. The trivially-solved
episodes are the evidence for the memorization finding.

If transcripts were not saved for some earlier episodes, note that gap in the
README rather than quietly omitting it.

If any transcript contains an API key, a local absolute path with a username, or
anything else that should not be public, scrub it before committing and note
that transcripts are scrubbed.

---

## Step 3: README structure

Replace `README.md` entirely. Required sections in this order.

### 3.1 Title and one-paragraph description

Keep the existing framing. An RL environment that evaluates language models on
diagnosing and fixing silent bugs in RL training code: bugs that throw no error,
fail no test, and just cause a policy to train badly or not at all.

Add one sentence stating what makes this different from existing code-repair
benchmarks: those bugs announce themselves through failing tests or tracebacks,
and silent numerical failures in training loops do not.

### 3.2 Status table

One row per phase. Every row needs a status that is either **Done**, **Not
started**, or **In progress** with a note on what is actively running. No row may
say "in progress" if nothing is running.

Required columns: phase, what, status.

Cover at minimum: v0 harness, calibration, scoring, the smoke eval difficulty
checkpoint, the omission bug attempt, and whatever comes next.

### 3.3 Findings

The most important section. Two findings exist. Both need a real number.

**Finding 1: v0 measured memorization, not diagnosis.**

Must state:
- The first patch formulation (`logratio = newlogprob - newlogprob`), that it
  produces a true zero gradient, and that this was verified with a standalone
  autograd check
- The solve rate against `claude-sonnet-4-5`, with the number of seeds, and the
  observation that some episodes were solved without running training at all
- That the patch was rewritten to preserve the mechanism while removing the
  visual tell, and that calibration came out byte-identical, confirming it is
  mechanistically the same bug
- **The second smoke eval result, as a number.** Pull it from the artifacts.
  This is the currently missing piece and the whole reason this section reads as
  unfinished.
- The diagnosis: CleanRL single-file PPO is heavily reproduced, fits in one
  `read_file` call, and every bug is a single wrong line, so the model diffs
  against a memorized reference rather than diagnosing from symptoms.

**Finding 2: omission bugs do not degrade a task where the omitted mechanism is
not load-bearing.**

Must state:
- Which mechanisms were tested (gradient clipping, entropy bonus, advantage
  normalization, and the fourth candidate by name)
- **That 4/4 improved rather than degraded training, with the actual measured
  clean-vs-broken deltas from calibration.** Not just the direction, the numbers.
- The budget and task: 40k timesteps, CartPole-v1
- The interpretation: these are variance-reduction and stability mechanisms that
  pay off on hard tasks and cost on easy ones. CartPole has two actions, uniform
  +1 reward, no exploration problem worth an entropy bonus, and advantages
  already at reasonable scale. Removing them cannot degrade a task where they
  are not doing work.
- That longer budgets were considered and rejected, because CartPole caps at 500
  return so both arms saturate and produce a ceiling effect rather than
  separation.

Both findings should be written as results, not as apologies. They are the
project's current output.

### 3.4 What the findings imply

Three or four sentences. Both findings say the substrate is too easy, not that
the bugs are wrong. The direction is toward bases with no memorized reference
and tasks where the affected mechanisms are actually load-bearing. Point at
`tasks/roadmap.md` for the plan rather than restating it.

### 3.5 Design

Compact. The reader needs enough to understand the results table.

- The two-base plan (`legacy_cleanrl` as the easy tier and memorization control,
  `drone_v1` for environment bugs) with a one-line note that the drone base is
  not built yet
- The four bug classes: `wrong_line`, `omission`, `interaction`, `statistical`
- The observation arms table: A, B, C, D, with what each exposes and which are
  implemented
- The difficulty gradient table with target solve rates per tier, and a sentence
  saying a benchmark where everything scores 0 is as uninformative as one where
  everything scores 1

Mark clearly which parts of this are built and which are planned. A reader must
not come away thinking the drone base or arm D exists.

### 3.6 Results

If any multi-model or multi-arm sweep has been run, put the table here with
standard error and the n. If none has been run, this section says so in one
line and nothing more. Do not create a table of placeholder rows.

State the statistics discipline explicitly: at the current n, the standard error
on a proportion is roughly X points, so differences under about 2X are not
meaningful. Compute X from the actual n rather than copying a number from a spec
document.

### 3.7 Invariants

Carry over the existing list and add the new rule:

> **A bug class is only valid on a base where the affected mechanism is
> load-bearing.** Before writing an omission or interaction patch, ablate the
> mechanism on the clean base and confirm it measurably degrades the clean
> baseline. If it does not, the task is wrong, not the bug. This check runs
> before calibration and is cheap.

Full list: no LLM in the reward path; determinism; the agent cannot touch the
scorer; no network in the agent container; a bug that does not degrade
performance is not a task; the load-bearing rule; log every trajectory in full;
difficulty comes from the bug and the search space, never from unreadable code.

### 3.8 Repository layout

Update it to match what actually exists. The current layout section lists
`eval/` as if it were present. Either commit it (step 2) or remove the line.
Every directory listed must exist in the pushed tree.

### 3.9 Running it

Keep the existing make targets. Add the `.env` and API key note. Add one line on
the pre-commit hook that greps for `sk-ant-` if it exists, or a note to add one
if it does not.

### 3.10 Specs and plan

Link the task documents: `tasks/tasks-list.md` (v0 spec), `tasks/hardness-v1.md`
(v1 difficulty), `tasks/roadmap.md` (full plan), `tasks/weekend-sprint.md`
(current scoped work). One line each on what they cover.

### 3.11 License

MIT. Note that the vendored CleanRL base is MIT and unmodified.

---

## Style rules

- Plain hyphens only. No em dashes or en dashes anywhere in the file.
- Tables over prose for anything with more than three parallel items.
- No hedging language around the findings. "Solved 3/3" not "appeared to be
  solved fairly reliably."
- No forward-looking claims stated as fact. "The drone base will use..." is
  fine; "The drone base uses..." is not, until it does.
- Keep the whole file under roughly 250 lines. Detail belongs in `tasks/`.

---

## Acceptance checklist

The task is complete when all of these are true:

- [ ] `eval/results/` and `eval/transcripts/` exist in the pushed tree and
      contain every episode that was actually run
- [ ] No section of the README says "in progress" for work that is not actively
      running
- [ ] The second smoke eval result appears as a number, traceable to a committed
      artifact
- [ ] The omission finding appears with the actual measured deltas, traceable to
      committed calibration output
- [ ] Every directory named in the layout section exists
- [ ] Every quantitative claim traces to a committed file
- [ ] The load-bearing rule is in the invariants
- [ ] No placeholder text, no `<<TODO>>`, no bracketed fill-ins remain
- [ ] Nothing in the file describes unbuilt components in the present tense
- [ ] Everything is committed and pushed to `origin/master`

Report the inventory from step 1 alongside the finished README so the numbers
can be checked against their sources.
