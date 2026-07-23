"""Main agent loop: turn cap, wall clock cap, transcript logging.

See README.md section 7.1 (episode structure) and 7.3 (system prompt).
"""

import json
import os
import re
import time
from collections import defaultdict

import yaml

from harness.container import EpisodeContainer
from harness.tools import ToolBox, tool_schemas_for_arm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(REPO_ROOT, "bugs", "registry.yaml")
TRANSCRIPTS_DIR = os.path.join(REPO_ROOT, "eval", "transcripts")

TURN_CAP = 30
WALL_CLOCK_CAP_S = 20 * 60

SYSTEM_PROMPT_TEMPLATE = (
    "The workspace contains an RL training script, ppo_cartpole.py, with a bug.\n"
    "Reported symptom: {symptom}\n\n"
    "Use the available tools to diagnose and fix the bug in the script. "
    "Call submit once you believe it is fixed."
)

NUDGE_MESSAGE = "Continue by calling a tool, or call submit() once you believe the bug is fixed."


def _load_registry_entry(bug_id):
    with open(REGISTRY_PATH) as f:
        registry = yaml.safe_load(f)
    for entry in registry:
        if entry["bug_id"] == bug_id:
            return entry
    raise KeyError(f"no registry entry for bug_id={bug_id!r}")


def _sanitize(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


class EpisodeResult:
    def __init__(self, instance_id, status, turns_used, wall_clock_s, tool_call_counts, transcript_path):
        self.instance_id = instance_id
        self.status = status
        self.turns_used = turns_used
        self.wall_clock_s = wall_clock_s
        self.tool_call_counts = dict(tool_call_counts)
        self.transcript_path = transcript_path


def run_episode(
    bug_id,
    instance_seed,
    episode_seed,
    arm,
    model_adapter,
    model_name,
    turn_cap=TURN_CAP,
    wall_clock_cap_s=WALL_CLOCK_CAP_S,
    transcripts_dir=None,
):
    entry = _load_registry_entry(bug_id)
    instance_id = f"{bug_id}__seed{instance_seed}"
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(symptom=entry["symptom"])
    tool_schemas = tool_schemas_for_arm(arm)

    transcript = {
        "instance_id": instance_id,
        "bug_id": bug_id,
        "instance_seed": instance_seed,
        "episode_seed": episode_seed,
        "arm": arm,
        "model": model_name,
        "system_prompt": system_prompt,
        "turns": [],
    }

    tool_call_counts = defaultdict(int)
    status = None
    start = time.time()

    with EpisodeContainer(patch_relpath=entry["patch"]) as container:
        toolbox = ToolBox(container, arm=arm, episode_seed=episode_seed)
        messages = []
        turn = 0

        while True:
            if turn >= turn_cap:
                status = "TURN_CAP"
                break
            if time.time() - start >= wall_clock_cap_s:
                status = "WALL_CLOCK_CAP"
                break

            assistant_turn = model_adapter.next_action(messages, tool_schemas, system_prompt)
            messages.append(model_adapter.assistant_message(assistant_turn))
            transcript["turns"].append({
                "turn": turn,
                "role": "assistant",
                "text": assistant_turn.text,
                "tool_calls": [{"name": tc.name, "input": tc.input} for tc in assistant_turn.tool_calls],
                "elapsed_s": time.time() - start,
            })

            submitted = False
            cap_hit = False
            for tc in assistant_turn.tool_calls:
                tool_call_counts[tc.name] += 1
                remaining = wall_clock_cap_s - (time.time() - start)
                if remaining <= 0:
                    cap_hit = True
                    break

                result_text = toolbox.dispatch(tc.name, tc.input, max_wall_s=remaining)
                messages.append(model_adapter.tool_result_message(tc, result_text))
                transcript["turns"].append({
                    "turn": turn,
                    "role": "tool_result",
                    "name": tc.name,
                    "input": tc.input,
                    "output": result_text,
                    "elapsed_s": time.time() - start,
                })
                if tc.name == "submit":
                    submitted = True
                    break

            if not assistant_turn.tool_calls:
                messages.append({"role": "user", "content": NUDGE_MESSAGE})
                transcript["turns"].append({"turn": turn, "role": "nudge", "text": NUDGE_MESSAGE})

            turn += 1
            if submitted:
                status = "OK"
                break
            if cap_hit:
                status = "WALL_CLOCK_CAP"
                break

    wall_clock_s = time.time() - start
    status = status or "ERROR"

    out_dir = transcripts_dir or TRANSCRIPTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{_sanitize(instance_id)}__{_sanitize(model_name)}__arm{arm}__epseed{episode_seed}.json"
    transcript_path = os.path.join(out_dir, fname)

    transcript["status"] = status
    transcript["turns_used"] = turn
    transcript["wall_clock_s"] = wall_clock_s
    transcript["tool_call_counts"] = dict(tool_call_counts)
    with open(transcript_path, "w") as f:
        json.dump(transcript, f, indent=2, default=str)

    return EpisodeResult(
        instance_id=instance_id,
        status=status,
        turns_used=turn,
        wall_clock_s=wall_clock_s,
        tool_call_counts=tool_call_counts,
        transcript_path=transcript_path,
    )
