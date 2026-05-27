#!/usr/bin/env python3
"""Build post-answer belief-probe prompts from selected case trajectories."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "belief_probe_dataset_task_a.json"
DEFAULT_SUMMARY = ROOT / "outputs" / "belief_probe_dataset_task_a.summary.json"

ROLE_SET = {"system", "user", "assistant"}
QUOTED_RULE_RE = re.compile(r'^\s*-\s*"([^"]+)":', re.MULTILINE)
FAULT_ID_RE = re.compile(r"^\s*([A-Z]):\s+", re.MULTILINE)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_case_files(inputs: Sequence[Path]) -> Iterable[Path]:
    for item in inputs:
        if item.is_file():
            yield item
        elif item.is_dir():
            yield from sorted(
                path
                for path in item.rglob("*.json")
                if path.name not in {"summary.json", "augment_summary.json", "stats_report.json"}
            )


def infer_scenario(path: Path) -> str:
    text = str(path).replace("\\", "/")
    if "/task_a/" in text or "test_a" in text or "_task_a" in text:
        return "task_a"
    if "/task_b/" in text or "test_b" in text or "_task_b" in text:
        return "task_b"
    return "unknown"


def keep_scenario(path: Path, scenario: str) -> bool:
    return scenario == "all" or infer_scenario(path) == f"scenario_{scenario}"


def normalize_challenge(value: Any) -> str:
    text = str(value or "").strip().lower()
    return "failed_isolation" if text in {"failed_isolation", "failed_isolation", "failed_isolation"} else text


def base_case_id(payload: Dict[str, Any], path: Path) -> str:
    for key in ("case_id", "sample_id", "experiment_id"):
        if payload.get(key):
            return str(payload[key])
    return path.stem


def normalize_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        if role in ROLE_SET:
            item = dict(message)
            item["role"] = role
            item["content"] = str(message.get("content", ""))
            output.append(item)
    return output


def first_system(messages: Sequence[Dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "system":
            return str(message.get("content", ""))
    return ""


def trajectory_messages(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns = trajectory.get("turns")
    if not isinstance(turns, list):
        messages = trajectory.get("conversation") or trajectory.get("messages") or []
        return normalize_messages(messages if isinstance(messages, list) else [])

    system_prompt = str(trajectory.get("system_prompt") or first_system(trajectory.get("messages") or []))
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        turn_index = int(turn.get("turn", index) or index)
        gold = list(turn.get("golden_hypotheses") or turn.get("golden") or [])
        messages.append(
            {
                "role": "user",
                "content": str(turn.get("prompt", "")),
                "turn": turn_index,
                "golden_hypotheses": gold,
            }
        )
        if turn.get("response"):
            messages.append(
                {
                    "role": "assistant",
                    "content": str(turn.get("response", "")),
                    "turn": turn_index,
                    "model_hypotheses": list(turn.get("model_hypotheses") or []),
                    "model_matches_golden": turn.get("model_matches_golden"),
                    "parse_ok": turn.get("parse_ok"),
                }
            )
    return messages


def candidate_rules(trajectory: Dict[str, Any], system_prompt: str) -> List[str]:
    raw = trajectory.get("candidate_rules") or trajectory.get("candidate_ids") or trajectory.get("fault_ids")
    if isinstance(raw, dict):
        return [str(key) for key in raw.keys()]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    quoted = QUOTED_RULE_RE.findall(system_prompt)
    if quoted:
        return quoted
    return FAULT_ID_RE.findall(system_prompt)


def probe_question(candidates: Sequence[str]) -> str:
    candidate_block = "\n".join(f"- {candidate}" for candidate in candidates)
    return (
        "Rank the candidate hypotheses from most likely to least likely under the current active evidence.\n"
        "Return strict JSON only in the form:\n"
        '{"ranking":["candidate_id_1","candidate_id_2",...]}\n\n'
        f"Candidates:\n{candidate_block}"
    )


def iter_turns(messages: Sequence[Dict[str, Any]]) -> Iterable[Tuple[int, int, Dict[str, Any], Dict[str, Any] | None]]:
    user_indices = [idx for idx, message in enumerate(messages) if message.get("role") == "user"]
    for fallback_turn, user_idx in enumerate(user_indices):
        user_msg = messages[user_idx]
        assistant_msg = (
            messages[user_idx + 1]
            if user_idx + 1 < len(messages) and messages[user_idx + 1].get("role") == "assistant"
            else None
        )
        try:
            turn_index = int(user_msg.get("turn", fallback_turn))
        except Exception:
            turn_index = fallback_turn
        yield turn_index, user_idx, user_msg, assistant_msg


def repeat_entries(payload: Dict[str, Any], all_repeats: bool) -> List[Dict[str, Any]]:
    repeats = payload.get("repeat_trajectories")
    if isinstance(repeats, list) and repeats:
        entries = [entry for entry in repeats if isinstance(entry, dict)]
        if all_repeats:
            return entries
        target_category = payload.get("category")
        return [next((entry for entry in entries if entry.get("category") == target_category), entries[0])]
    return [{"trajectory": payload, "repeat_index": int(payload.get("repeat_index", 0) or 0), "category": payload.get("category")}]


def build_records_for_case(path: Path, payload: Dict[str, Any], entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    trajectory = entry.get("trajectory") if isinstance(entry.get("trajectory"), dict) else entry
    repeat_index = int(entry.get("repeat_index", trajectory.get("repeat_index", 0)) or 0)
    messages = trajectory_messages(trajectory)
    if not messages:
        raise ValueError(f"missing messages: {path}")

    system_prompt = first_system(messages)
    candidates = candidate_rules(trajectory, system_prompt)
    if not candidates:
        raise ValueError(f"missing candidate rules: {path}")

    case_id = base_case_id(payload, path)
    oracle = trajectory.get("oracle") or payload.get("oracle")
    challenge = normalize_challenge(trajectory.get("challenge_type") or payload.get("challenge_type") or trajectory.get("mode") or payload.get("mode"))
    scenario = infer_scenario(path)
    question = probe_question(candidates)
    records: List[Dict[str, Any]] = []

    for turn_index, user_idx, user_msg, assistant_msg in iter_turns(messages):
        history = deepcopy(messages[: user_idx + (2 if assistant_msg is not None else 1)])
        probe_messages = history + [{"role": "user", "content": question}]
        records.append(
            {
                "probe_id": f"{case_id}_rep{repeat_index}_t{turn_index}",
                "source_file": str(path),
                "scenario": scenario,
                "case_id": case_id,
                "repeat_index": repeat_index,
                "challenge_type": challenge,
                "turn_index": turn_index,
                "candidate_rules": candidates,
                "oracle": oracle,
                "gold_hypotheses": list(user_msg.get("golden_hypotheses") or []),
                "model_hypotheses": list((assistant_msg or {}).get("model_hypotheses") or []),
                "model_matches_golden": (assistant_msg or {}).get("model_matches_golden"),
                "probe_position": "post_answer",
                "probe_question": question,
                "source_user_prompt": str(user_msg.get("content", "")),
                "source_assistant_response": str((assistant_msg or {}).get("content", "")),
                "messages": probe_messages,
            }
        )
    turn_count = len(records)
    for record in records:
        record["turn_count"] = turn_count
    return records


def build_dataset(inputs: Sequence[Path], scenario: str, all_repeats: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    skipped: List[str] = []
    source_files = 0
    repeat_count = 0

    for path in iter_case_files(inputs):
        if not keep_scenario(path, scenario):
            skipped.append(str(path))
            continue
        payload = read_json(path)
        source_files += 1
        for entry in repeat_entries(payload, all_repeats):
            records.extend(build_records_for_case(path, payload, entry))
            repeat_count += 1

    summary = {
        "input_files": [str(path) for path in inputs],
        "scenario": scenario,
        "probe_position": "post_answer",
        "source_file_count": source_files,
        "repeat_trajectory_count": repeat_count,
        "probe_record_count": len(records),
        "skipped_files": skipped,
    }
    return records, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build post-answer belief-probe dataset")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--scenario", choices=["a", "b", "all"], default="a")
    parser.add_argument("--all-repeats", action="store_true")
    args = parser.parse_args()

    records, summary = build_dataset(args.inputs, scenario=args.scenario, all_repeats=args.all_repeats)
    write_json(args.output, records)
    write_json(args.summary, summary)
    print(f"[build-probe] records={len(records)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
