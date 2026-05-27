"""Load task_b training cases into HuggingFace Dataset objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset


def _row_from_case(case: dict) -> dict:
    turns = case["turns"]
    turn_prompts = [turn["prompt"] for turn in turns]
    turn_gts = [turn["golden"] for turn in turns]
    return {
        "case_id": case["case_id"],
        "challenge_type": case["challenge_type"],
        "oracle": case["oracle"],
        "circuit_type": case.get("circuit_type", ""),
        "system_prompt": case["system_prompt"],
        "turn_prompts": turn_prompts,
        "turn_gts": turn_gts,
        "prompt": turn_prompts[0],
        "gt_survivors": json.dumps(turn_gts[2]),
    }


def load_cases_dataset(cases_json_path: str) -> Dataset:
    path = Path(cases_json_path)
    cases = json.loads(path.read_text())
    rows = [_row_from_case(case) for case in cases]
    if not rows:
        raise ValueError(f"No cases found in {cases_json_path}")
    return Dataset.from_list(rows)
