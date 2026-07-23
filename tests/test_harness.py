"""Harness integration tests: container lifecycle, tools, and the episode loop.

These run a real Docker container (no mocking) but a scripted fake model
adapter, since no LLM call is needed to exercise the harness mechanics.
"""

import json
import os

import pytest

from harness.container import EpisodeContainer
from harness.episode import run_episode
from harness.models import AssistantTurn, ModelAdapter, ToolCall
from harness.tools import ToolBox, tool_schemas_for_arm

DEAD_SURROGATE_PATCH = "patches/dead_surrogate_v1.diff"
BUGGY_LINE = "logratio = newlogprob - newlogprob"
FIXED_LINE = "logratio = newlogprob - b_logprobs[mb_inds]"


class ScriptedAdapter(ModelAdapter):
    """Replays a fixed sequence of tool calls; used for tests only, not a
    real provider adapter."""

    def __init__(self, plan):
        self.plan = list(plan)
        self.calls_made = 0

    def next_action(self, messages, tool_schemas, system):
        if self.calls_made >= len(self.plan):
            name, tool_input = "submit", {}
        else:
            name, tool_input = self.plan[self.calls_made]
        self.calls_made += 1
        return AssistantTurn(text=f"(scripted) {name}", tool_calls=[ToolCall(id=f"call{self.calls_made}", name=name, input=tool_input)])

    def assistant_message(self, turn):
        return {"role": "assistant", "content": turn.text}

    def tool_result_message(self, tool_call, result_text):
        return {"role": "tool_result", "name": tool_call.name, "content": result_text}


class NoToolCallAdapter(ModelAdapter):
    """Never calls a tool; used to exercise the turn cap."""

    def next_action(self, messages, tool_schemas, system):
        return AssistantTurn(text="thinking...", tool_calls=[])

    def assistant_message(self, turn):
        return {"role": "assistant", "content": turn.text}

    def tool_result_message(self, tool_call, result_text):
        raise AssertionError("should never be called")


@pytest.fixture
def container():
    c = EpisodeContainer(patch_relpath=DEAD_SURROGATE_PATCH)
    yield c
    c.teardown()


def test_container_applies_patch_and_isolates_network(container):
    with open(os.path.join(container.host_workspace, "ppo_cartpole.py")) as f:
        content = f.read()
    assert BUGGY_LINE in content

    exit_code, _, stderr = container.exec(
        ["python", "-c", "import socket; socket.create_connection(('8.8.8.8', 53), timeout=3)"],
        timeout_s=10,
    )
    assert exit_code != 0
    assert "Network is unreachable" in stderr or "unreachable" in stderr.lower()


def test_toolbox_read_list_edit(container):
    box = ToolBox(container, arm="A", episode_seed=0)

    assert box.list_files(".") == "ppo_cartpole.py"

    out = box.read_file("ppo_cartpole.py", start=251, end=251)
    assert out == f"251\t                {BUGGY_LINE}"

    result = box.edit_file("ppo_cartpole.py", BUGGY_LINE, FIXED_LINE)
    assert result == "ok: edit applied"
    with open(os.path.join(container.host_workspace, "ppo_cartpole.py")) as f:
        assert FIXED_LINE in f.read()


def test_edit_file_rejects_non_unique_match(container):
    box = ToolBox(container, arm="A", episode_seed=0)
    result = box.edit_file("ppo_cartpole.py", "ratio", "ratio_renamed")
    assert result.startswith("error:") and "not unique" in result


def test_edit_file_rejects_missing_match(container):
    box = ToolBox(container, arm="A", episode_seed=0)
    result = box.edit_file("ppo_cartpole.py", "this string does not appear", "x")
    assert result.startswith("error:")


def test_toolbox_blocks_workspace_escape(container):
    box = ToolBox(container, arm="A", episode_seed=0)
    result = box.read_file("../../../etc/passwd")
    assert result.startswith("error:") and "escapes workspace" in result


def test_arm_a_hides_metrics_tools():
    schemas = tool_schemas_for_arm("A")
    names = {s["name"] for s in schemas}
    assert "get_metrics" not in names
    assert "list_metric_keys" not in names
    assert "run_training" in names


def test_arm_b_exposes_metrics_tools():
    schemas = tool_schemas_for_arm("B")
    names = {s["name"] for s in schemas}
    assert {"get_metrics", "list_metric_keys"} <= names


def test_toolbox_run_training_then_get_metrics_arm_b(container):
    box = ToolBox(container, arm="B", episode_seed=0)
    run_id, tail = box.run_training(iterations=1)
    assert "episodic_return" in tail

    keys = box.list_metric_keys(run_id)
    assert "losses/policy_loss" in keys

    metrics = box.get_metrics(run_id, keys=["losses/policy_loss"])
    assert len(metrics["losses/policy_loss"]) >= 1


def test_toolbox_metrics_tools_disabled_on_arm_a(container):
    box = ToolBox(container, arm="A", episode_seed=0)
    assert box.get_metrics("run0", keys=["x"]).startswith("error:")
    assert box.list_metric_keys("run0").startswith("error:")


def test_run_episode_end_to_end_fixes_the_bug(tmp_path):
    plan = [
        ("list_files", {"path": "."}),
        ("read_file", {"path": "ppo_cartpole.py", "start": 248, "end": 253}),
        ("edit_file", {"path": "ppo_cartpole.py", "old_str": BUGGY_LINE, "new_str": FIXED_LINE}),
        ("run_training", {"iterations": 1}),
        ("submit", {}),
    ]
    result = run_episode(
        bug_id="dead_surrogate_v1",
        instance_seed=0,
        episode_seed=1,
        arm="A",
        model_adapter=ScriptedAdapter(plan),
        model_name="scripted-test",
        turn_cap=10,
        wall_clock_cap_s=180,
        transcripts_dir=str(tmp_path),
    )
    assert result.status == "OK"
    assert result.turns_used == 5
    assert result.tool_call_counts == {
        "list_files": 1, "read_file": 1, "edit_file": 1, "run_training": 1, "submit": 1,
    }
    assert os.path.exists(result.transcript_path)
    with open(result.transcript_path) as f:
        transcript = json.load(f)
    assert transcript["status"] == "OK"
    assert transcript["instance_id"] == "dead_surrogate_v1__seed0"


def test_run_episode_hits_turn_cap(tmp_path):
    result = run_episode(
        bug_id="dead_surrogate_v1",
        instance_seed=0,
        episode_seed=2,
        arm="A",
        model_adapter=NoToolCallAdapter(),
        model_name="no-tool-test",
        turn_cap=3,
        wall_clock_cap_s=180,
        transcripts_dir=str(tmp_path),
    )
    assert result.status == "TURN_CAP"
    assert result.turns_used == 3
    assert result.tool_call_counts == {}


def test_run_episode_hits_wall_clock_cap(tmp_path):
    result = run_episode(
        bug_id="dead_surrogate_v1",
        instance_seed=0,
        episode_seed=3,
        arm="A",
        model_adapter=NoToolCallAdapter(),
        model_name="no-tool-test",
        turn_cap=1000,
        wall_clock_cap_s=0.01,
        transcripts_dir=str(tmp_path),
    )
    assert result.status == "WALL_CLOCK_CAP"
