#!/usr/bin/env python3
"""Export compact case-level JSON for manual probing analysis."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "task_a" / "9B" / "probing" / "base" / "probe_ranking_results.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "task_a" / "9B" / "probing" / "base" / "probe_ranking_simple_cases.json"


def read_json_records(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [item for item in payload["records"] if isinstance(item, dict)]
    raise ValueError(f"unsupported JSON records: {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role == "assistant":
            content = content.replace("<think>", "").replace("</think>", "").strip()
        if role and content:
            output.append({"role": role, "content": content})
    return output


def extract_triple_and_result(text: Any) -> Dict[str, str]:
    lines = str(text or "").splitlines()
    triple_line = ""
    result = ""
    for line in lines:
        if "Triple" in line and not triple_line:
            triple_line = line.strip()
        if "**YES**" in line:
            result = "YES"
            if not triple_line:
                triple_line = line.strip()
            break
        if "**NO**" in line:
            result = "NO"
            if not triple_line:
                triple_line = line.strip()
            break
    return {"triple_line": triple_line, "result": result}


def history_without_probe(row: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = list(row.get("messages") or [])
    if messages and messages[-1].get("role") in {"user", "system"} and "Rank the candidate hypotheses" in str(
        messages[-1].get("content", "")
    ):
        messages = messages[:-1]
    return compact_messages(messages)


def load_original_case(source_file: str, repeat_index: int) -> Dict[str, Any]:
    payload = json.loads(Path(source_file).read_text(encoding="utf-8"))
    repeats = payload.get("repeat_trajectories")
    if isinstance(repeats, list):
        for entry in repeats:
            if not isinstance(entry, dict):
                continue
            trajectory = entry.get("trajectory")
            if not isinstance(trajectory, dict):
                continue
            idx = int(entry.get("repeat_index", trajectory.get("repeat_index", 0)) or 0)
            if idx == repeat_index:
                return trajectory
    return payload


def compact_original_trajectory(source_file: str, repeat_index: int) -> Dict[str, Any]:
    trajectory = load_original_case(source_file, repeat_index)
    system_prompt = str(trajectory.get("system_prompt") or "")
    if not system_prompt:
        messages = trajectory.get("conversation") or trajectory.get("messages") or []
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "system":
                    system_prompt = str(message.get("content", ""))
                    break
    turns = trajectory.get("turns")
    if isinstance(turns, list):
        compact_turns: List[Dict[str, Any]] = []
        for idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            prompt = str(turn.get("prompt", ""))
            probe_bits = extract_triple_and_result(prompt)
            compact_turns.append(
                {
                    "turn_index": int(turn.get("turn", idx) or idx),
                    "triple_line": probe_bits["triple_line"],
                    "result": probe_bits["result"],
                    "gold_hypotheses": list(turn.get("golden_hypotheses") or turn.get("golden") or []),
                    "model_hypotheses": list(turn.get("model_hypotheses") or []),
                    "original_correct": turn.get("model_matches_golden"),
                }
            )
        return {"system_prompt": system_prompt, "turns": compact_turns}

    messages = trajectory.get("conversation") or trajectory.get("messages") or []
    compact_messages_out: List[Dict[str, Any]] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = str(message.get("content", ""))
            if role == "user":
                probe_bits = extract_triple_and_result(content)
                compact_messages_out.append(
                    {
                        "role": "user",
                        "triple_line": probe_bits["triple_line"],
                        "result": probe_bits["result"],
                    }
                )
            elif role == "assistant":
                compact_messages_out.append(
                    {
                        "role": "assistant",
                        "hypotheses": list(message.get("model_hypotheses") or []),
                    }
                )
    return {"system_prompt": system_prompt, "messages": compact_messages_out}


def group_key(row: Dict[str, Any]) -> Tuple[str, str, int]:
    return (
        str(row.get("source_file", "")),
        str(row.get("case_id", "")),
        int(row.get("repeat_index", 0) or 0),
    )


def original_trajectory_from_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def shorten(text: Any, limit: int = 180) -> str:
        value = str(text or "").replace("<think>", "").replace("</think>", "").replace("\n", " ").strip()
        return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."

    turns: List[Dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item.get("turn_index", 0) or 0)):
        turns.append(
            {
                "turn_index": int(row.get("turn_index", 0) or 0),
                "user": shorten(row.get("source_user_prompt"), 220),
                "assistant": shorten(row.get("source_assistant_response"), 220),
                "gold_hypotheses": row.get("gold_hypotheses"),
                "model_hypotheses": row.get("model_hypotheses"),
                "original_correct": row.get("model_matches_golden"),
            }
        )
    return turns


def export_cases(rows: Sequence[Dict[str, Any]], final_wrong_only: bool) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    cases: List[Dict[str, Any]] = []
    for (source_file, case_id, repeat_index), items in sorted(grouped.items()):
        items.sort(key=lambda item: int(item.get("turn_index", 0) or 0))
        final_wrong = bool(items and not bool(items[-1].get("model_matches_golden")))
        if final_wrong_only and not final_wrong:
            continue
        first = items[0]
        cases.append(
            {
                "case_id": case_id,
                "repeat_index": repeat_index,
                "challenge_type": first.get("challenge_type"),
                "oracle": first.get("oracle"),
                "final_wrong": final_wrong,
                "source_file": source_file,
                "original_trajectory": compact_original_trajectory(source_file, repeat_index),
                "turn_probes": [
                    {
                        "turn_index": int(row.get("turn_index", 0) or 0),
                        "history": history_without_probe(row),
                        "probe_question": row.get("probe_question") or row.get("probe_instruction"),
                        "probe_answer": row.get("response_text"),
                        "parsed_ranking": row.get("parsed_ranking"),
                        "completed_ranking": row.get("completed_ranking"),
                        "gold_hypotheses": row.get("gold_hypotheses"),
                        "oracle_rank": row.get("oracle_rank"),
                        "oracle_top1": row.get("oracle_top1"),
                        "oracle_top3": row.get("oracle_top3"),
                        "original_correct": row.get("model_matches_golden"),
                    }
                    for row in items
                ],
            }
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Export compact case-level belief-probe JSON")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--final-wrong-only", action="store_true")
    args = parser.parse_args()

    cases = export_cases(read_json_records(args.input), final_wrong_only=args.final_wrong_only)
    write_json(args.output, cases)
    print(f"[export-probe-cases] cases={len(cases)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
