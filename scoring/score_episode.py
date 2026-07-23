"""Entrypoint: score one completed episode transcript, emit a results JSON.

See README.md sections 8 and 8.4 for the results schema. Only an integrity
failure suppresses outcome/localization scoring (the episode is marked
INVALID); turn-cap and wall-clock-cap episodes are still scored on whatever
state the file was left in.
"""

import argparse
import json
import os
import sys
import tempfile

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # so `python scoring/score_episode.py` can `import scoring`/`harness`

from scoring.integrity import detect_hack_attempt, verify  # noqa: E402
from scoring.localization import broken_source_for, score_localization  # noqa: E402
from scoring.outcome import score_outcome  # noqa: E402

REGISTRY_PATH = os.path.join(REPO_ROOT, "bugs", "registry.yaml")
RESULTS_DIR = os.path.join(REPO_ROOT, "eval", "results")


def _load_registry_entry(bug_id):
    with open(REGISTRY_PATH) as f:
        registry = yaml.safe_load(f)
    for entry in registry:
        if entry["bug_id"] == bug_id:
            return entry
    raise KeyError(f"no registry entry for bug_id={bug_id!r}")


def score_episode(transcript_path):
    with open(transcript_path) as f:
        transcript = json.load(f)

    bug_id = transcript["bug_id"]
    instance_id = transcript["instance_id"]
    entry = _load_registry_entry(bug_id)
    final_source = transcript["final_file_content"]

    hack_attempt = detect_hack_attempt(final_source)
    integrity_ok = verify(transcript["integrity_snapshot_before"])

    outcome = None
    localization = None
    localization_binary = None

    if not integrity_ok:
        status = "INVALID"
    else:
        status = transcript["status"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(final_source)
            tmp_path = f.name
        try:
            outcome = score_outcome(tmp_path, instance_id)["outcome"]
        finally:
            os.remove(tmp_path)

        broken_source = broken_source_for(entry["patch"])
        loc_result = score_localization(final_source, broken_source, entry["ground_truth_lines"])
        localization = loc_result["localization"]
        localization_binary = loc_result["localization_binary"]

    results = {
        "instance_id": instance_id,
        "model": transcript["model"],
        "arm": transcript["arm"],
        "episode_seed": transcript["episode_seed"],
        "outcome": outcome,
        "localization": localization,
        "localization_binary": localization_binary,
        "turns_used": transcript["turns_used"],
        "wall_clock_s": transcript["wall_clock_s"],
        "tool_calls": transcript["tool_call_counts"],
        "status": status,
        "hack_attempt": hack_attempt,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(transcript_path))[0]
    results_path = os.path.join(RESULTS_DIR, f"{stem}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript_path")
    args = parser.parse_args()
    results = score_episode(args.transcript_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
