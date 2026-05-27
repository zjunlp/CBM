"""Load fixed multi-turn cases into HuggingFace Dataset objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset


def _row_from_case(case: dict[str, Any]) -> dict[str, Any]:
    turns = case.get("turns") or []
    turn_prompts = [turn["prompt"] for turn in turns]
    turn_gts = [turn["golden"] for turn in turns]
    if not turn_prompts and "selected_prompt" in case:
        turn_prompts = [case["selected_prompt"]]
        turn_gts = [case["gt_survivors"]]
    row = {
        "case_id": case["case_id"],
        "challenge_type": case["challenge_type"],
        "oracle": case["oracle"],
        "system_prompt": case["system_prompt"],
        "turn_prompts": turn_prompts,
        "turn_gts": turn_gts,
        "prompt": turn_prompts[0],
        "gt_survivors": case.get("gt_survivors", turn_gts[-1]),
    }
    if "messages" in case:
        row["messages"] = case["messages"]
    if "selected_turn" in case:
        row["selected_turn"] = case["selected_turn"]
    if "source_category" in case:
        row["source_category"] = case["source_category"]
    return row


def load_cases_dataset(cases_json_path: str) -> Dataset:
    path = Path(cases_json_path)
    cases = json.loads(path.read_text())
    rows = [_row_from_case(case) for case in cases]
    if not rows:
        raise ValueError(f"No cases found in {cases_json_path}")
    return Dataset.from_list(rows)
