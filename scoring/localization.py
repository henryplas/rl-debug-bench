"""Diff overlap against ground truth lines (README.md section 8.2).

localization = |changed ∩ truth| / |changed ∪ truth|, where `truth` is the
registry's ground_truth_lines and `changed` is the set of base-file line
numbers the agent's edits touched.

Note: this diffs the agent's final file against the *broken file it started
from* (pristine + bug patch applied), not against the pristine base directly.
Diffing straight against pristine breaks for exactly the bugs this benchmark
cares about: dead_surrogate_v1's correct fix reconstructs the pristine text
byte-for-byte, so a final-vs-pristine diff would show *no* change at the bug
site for a perfect fix (localization 0) while a no-op agent -- whose file
still differs from pristine at the bug's own footprint -- would score 1.0.
Diffing against the broken starting point fixes both: a no-op agent touches
nothing (0), and reverting the bug line touches exactly that line (credit).
This assumes a bug patch doesn't change the base file's total line count,
true for every v0 patch (single-line replacements); a future line-adding
patch would need real coordinate remapping, not needed yet.
"""

import difflib
import os
import shutil

from harness.container import BASE_SCRIPT, build_workspace


def broken_source_for(patch_relpath):
    """The pristine base script with a bug patch applied, as a string."""
    workdir = build_workspace(patch_relpath=patch_relpath)
    try:
        with open(os.path.join(workdir, "ppo_cartpole.py")) as f:
            return f.read()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def pristine_source():
    with open(BASE_SCRIPT) as f:
        return f.read()


def changed_lines(starting_source, final_source):
    """1-indexed line numbers touched going from starting_source to
    final_source. A pure insertion is attributed to the preceding line."""
    starting_lines = starting_source.splitlines(keepends=True)
    final_lines = final_source.splitlines(keepends=True)

    matcher = difflib.SequenceMatcher(a=starting_lines, b=final_lines, autojunk=False)
    changed = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i1 == i2:
            changed.add(max(1, i1))
        else:
            changed.update(range(i1 + 1, i2 + 1))
    return changed


def score_localization(final_source, starting_source, ground_truth_lines):
    changed = changed_lines(starting_source, final_source)
    truth = set(ground_truth_lines)

    union = changed | truth
    localization = len(changed & truth) / len(union) if union else 0.0

    return {
        "localization": localization,
        "localization_binary": localization > 0,
        "changed_lines": sorted(changed),
    }
