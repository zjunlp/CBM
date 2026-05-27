"""Normalize raw failed_isolation source outputs into analysis cases."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List


def _role_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
        if message.get("role") in {"system", "user", "assistant"}
    ]


def _assistant_records(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [message for message in messages if message.get("role") == "assistant"]


def _oracle_from_b_filename(name: str) -> str:
    match = re.search(r"_([A-Z])_tpl", name)
    return match.group(1) if match else ""


def _fault_ids_from_system(system_prompt: str) -> List[str]:
    return re.findall(r"^\s*([A-Z]):\s+", system_prompt, flags=re.MULTILINE)


def normalize_failed_isolation_source_case(path: Path, payload: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    if str(payload.get("challenge_type", "")).lower() == "failed_isolation" and payload.get("turns"):
        return payload

    if scenario == "a" or "repeat_trajectories" in payload:
        repeats = payload.get("repeat_trajectories") or []
        if not repeats:
            raise ValueError(f"{path} has no repeat_trajectories")
        source = repeats[0].get("trajectory") or {}
        messages = _role_messages(source.get("messages") or [])
        assistants = _assistant_records(source.get("messages") or [])
        system = next((message["content"] for message in messages if message["role"] == "system"), None)
        if system is None:
            raise ValueError(f"{path} has no system message")
        candidate_rules = source.get("candidate_rules") or []
        if isinstance(candidate_rules, dict):
            candidate_ids = list(candidate_rules)
        else:
            candidate_ids = list(candidate_rules) if isinstance(candidate_rules, list) else []
        return {
            "case_id": path.stem,
            "experiment_id": str(payload.get("experiment_id") or path.stem),
            "challenge_type": "failed_isolation",
            "failed_isolation_scenario": "a",
            "source_file": str(path),
            "oracle": payload.get("oracle") or source.get("oracle"),
            "candidate_ids": candidate_ids,
            "system_prompt": system,
            "turns": [
                {"prompt": message["content"], "golden": list(golden.get("golden_hypotheses") or [])}
                for message, golden in zip([m for m in messages if m["role"] == "user"], assistants, strict=False)
            ],
        }

    conversations = payload.get("conversation") or []
    if not conversations:
        raise ValueError(f"{path} has no conversation repeats")
    source = conversations[0]
    messages = _role_messages(source.get("conversation") or [])
    survivors = source.get("turn_survivors") or []
    system = next((message["content"] for message in messages if message["role"] == "system"), None)
    if system is None:
        raise ValueError(f"{path} has no system message")
    return {
        "case_id": path.stem,
        "challenge_type": "failed_isolation",
        "failed_isolation_scenario": "b",
        "source_file": str(path),
        "oracle": _oracle_from_b_filename(path.name),
        "candidate_ids": _fault_ids_from_system(system),
        "system_prompt": system,
        "turns": [
            {"prompt": message["content"], "golden": list(golden.get("golden_survivors") or [])}
            for message, golden in zip([m for m in messages if m["role"] == "user"], survivors, strict=False)
        ],
    }
