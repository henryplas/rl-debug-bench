"""Scoring tests (tasks/tasks-list.md section 8): outcome, localization, and
the full score_episode entrypoint against a real episode transcript.

Outcome tests re-run real training via Docker against the committed
calibration baseline, so they're slow (a few minutes) but exercise the
actual scientific claim the benchmark rests on.
"""

import json
import os

import pytest

from harness.episode import run_episode
from harness.models import AssistantTurn, ModelAdapter, ToolCall
from scoring.localization import broken_workspace_for, changed_lines, pristine_files, score_localization
from scoring.outcome import score_outcome
from scoring.score_episode import score_episode

BASE_ID = "legacy_cleanrl"
FILE = "ppo_cartpole.py"
BUGGY_BLOCK = (
    "                _, old_logprob, _, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])\n"
    "                logratio = newlogprob - old_logprob"
)
FIXED_LINE = "                logratio = newlogprob - b_logprobs[mb_inds]"
INSTANCE_ID = "dead_surrogate_v1__seed0"
PATCH = "patches/dead_surrogate_v1.diff"


class ScriptedAdapter(ModelAdapter):
    def __init__(self, plan):
        self.plan = list(plan)
        self.step = 0

    def next_action(self, messages, tool_schemas, system):
        name, tool_input = self.plan[self.step] if self.step < len(self.plan) else ("submit", {})
        self.step += 1
        return AssistantTurn(text="x", tool_calls=[ToolCall(id=f"c{self.step}", name=name, input=tool_input)])

    def assistant_message(self, turn):
        return {"role": "assistant", "content": turn.text}

    def tool_result_message(self, tool_call, result_text):
        return {"role": "tool_result", "name": tool_call.name, "content": result_text}


# --- localization: fast, no training runs ---

def test_localization_perfect_fix_scores_one():
    broken = broken_workspace_for(BASE_ID, PATCH)
    perfect_fix = {FILE: broken[FILE].replace(BUGGY_BLOCK, FIXED_LINE)}
    result = score_localization(BASE_ID, perfect_fix, broken, ground_truth_file=FILE, ground_truth_lines=[251])
    assert result == {"localization": 1.0, "localization_binary": True, "changed_lines": [(FILE, 251)]}


def test_localization_no_op_scores_zero():
    broken = broken_workspace_for(BASE_ID, PATCH)
    result = score_localization(BASE_ID, broken, broken, ground_truth_file=FILE, ground_truth_lines=[251])
    assert result["localization"] == 0.0
    assert result["changed_lines"] == []


def test_localization_unrelated_edit_scores_zero():
    broken = broken_workspace_for(BASE_ID, PATCH)
    unrelated = {FILE: broken[FILE].replace("ent_coef: float = 0.01", "ent_coef: float = 0.02")}
    result = score_localization(BASE_ID, unrelated, broken, ground_truth_file=FILE, ground_truth_lines=[251])
    assert result["localization"] == 0.0
    assert result["localization_binary"] is False


def test_localization_partial_overlap():
    broken = broken_workspace_for(BASE_ID, PATCH)
    mixed_source = broken[FILE].replace(BUGGY_BLOCK, FIXED_LINE).replace(
        "ent_coef: float = 0.01", "ent_coef: float = 0.02"
    )
    mixed = {FILE: mixed_source}
    result = score_localization(BASE_ID, mixed, broken, ground_truth_file=FILE, ground_truth_lines=[251])
    assert result["localization"] == 0.5  # 1 correct line / 2 changed union truth
    assert result["localization_binary"] is True


def test_broken_workspace_differs_from_pristine_only_at_the_bug():
    pristine = pristine_files(BASE_ID)
    broken = broken_workspace_for(BASE_ID, PATCH)
    assert changed_lines(pristine[FILE], broken[FILE]) == {251}


# --- outcome: slow, real training runs against the committed baseline ---

@pytest.mark.slow
def test_outcome_pristine_scores_near_one(tmp_path):
    pristine = pristine_files(BASE_ID)
    (tmp_path / FILE).write_text(pristine[FILE])
    result = score_outcome(BASE_ID, str(tmp_path), INSTANCE_ID)
    assert result["outcome"] > 0.8


@pytest.mark.slow
def test_outcome_broken_scores_near_zero(tmp_path):
    broken = broken_workspace_for(BASE_ID, PATCH)
    (tmp_path / FILE).write_text(broken[FILE])
    result = score_outcome(BASE_ID, str(tmp_path), INSTANCE_ID)
    assert result["outcome"] < 0.2


# --- full pipeline: one real episode, then score it ---

@pytest.mark.slow
def test_score_episode_end_to_end(tmp_path):
    plan = [
        ("read_file", {"path": "ppo_cartpole.py", "start": 248, "end": 253}),
        ("edit_file", {"path": "ppo_cartpole.py", "old_str": BUGGY_BLOCK, "new_str": FIXED_LINE}),
        ("submit", {}),
    ]
    episode_result = run_episode(
        bug_id="dead_surrogate_v1",
        instance_seed=0,
        episode_seed=11,
        arm="A",
        model_adapter=ScriptedAdapter(plan),
        model_name="scripted-scoring-test",
        turn_cap=10,
        wall_clock_cap_s=120,
        transcripts_dir=str(tmp_path),
    )
    assert episode_result.status == "OK"

    results_dir = tmp_path / "results"
    import scoring.score_episode as score_episode_module

    original_results_dir = score_episode_module.RESULTS_DIR
    score_episode_module.RESULTS_DIR = str(results_dir)
    try:
        results = score_episode(episode_result.transcript_path)
    finally:
        score_episode_module.RESULTS_DIR = original_results_dir

    assert results["outcome"] == 1.0
    assert results["localization"] == 1.0
    assert results["localization_binary"] is True
    assert results["status"] == "OK"
    assert results["hack_attempt"] is False

    written = json.loads((results_dir / os.path.basename(episode_result.transcript_path)).read_text())
    assert written == results
