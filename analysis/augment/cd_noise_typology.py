"""L3 noise typology injection for RD and CD failed_isolation cases."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from analysis.augment.prompt_parse import extract_host_comment, inject_host_comment, strip_host_comment
from analysis.augment.schema import attach_augmentation, build_augmented_case_id

NOISE_TYPES = ("none", "sycophancy", "authority", "stress")
DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates" / "noise"


def _load_templates(noise_type: str, template_dir: Path) -> List[str]:
    if noise_type == "none":
        return []
    path = template_dir / f"{noise_type}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing noise template: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    templates = payload.get("templates") or []
    if not templates:
        raise ValueError(f"No templates in {path}")
    return [str(item) for item in templates]


def _pick_wrong_hint(case: Dict[str, Any], turn_idx: int, rng: random.Random) -> str:
    oracle = str(case.get("oracle", ""))
    candidates: List[str] = []
    raw_candidates = case.get("candidate_ids") or case.get("candidate_rules") or case.get("fault_ids") or []
    if isinstance(raw_candidates, dict):
        raw_candidates = list(raw_candidates)
    if isinstance(raw_candidates, list):
        candidates.extend(str(item) for item in raw_candidates)

    system_prompt = str(case.get("system_prompt", ""))
    candidates.extend(re.findall(r'-\s+"([^"]+)":', system_prompt))
    candidates.extend(re.findall(r"^\s*([A-Z]):\s+", system_prompt, flags=re.MULTILINE))
    for turn in case.get("turns", []):
        candidates.extend(str(item) for item in (turn.get("golden") or []))

    deduped = []
    seen = set()
    for item in candidates:
        if item and item != oracle and item not in seen:
            seen.add(item)
            deduped.append(item)
    if deduped:
        return rng.choice(deduped)
    return "an alternative hypothesis"


def _pick_turn0_wrong_hint(case: Dict[str, Any], rng: random.Random) -> Optional[str]:
    oracle = str(case.get("oracle", ""))
    turns = list(case.get("turns") or [])
    if not turns:
        return None
    turn0_golden = turns[0].get("golden") or []
    candidates = [str(item) for item in turn0_golden if str(item) and str(item) != oracle]
    if not candidates:
        return None
    return rng.choice(candidates)


def _pick_prev_hyp_summary(case: Dict[str, Any], turn_idx: int) -> str:
    if turn_idx <= 0:
        return "your current hypothesis"
    prev = case["turns"][turn_idx - 1].get("golden") or []
    if not prev:
        return "your current hypothesis"
    return ", ".join(sorted(str(x) for x in prev))


def render_noise_comment(
    *,
    template: str,
    case: Dict[str, Any],
    turn_idx: int,
    rng: random.Random,
    fixed_wrong_hint: Optional[str] = None,
) -> str:
    oracle = str(case.get("oracle", ""))
    if fixed_wrong_hint:
        wrong_hint = fixed_wrong_hint
    else:
        wrong_hint = _pick_wrong_hint(case, turn_idx, rng)
    return template.format(
        wrong_hint=wrong_hint,
        wrong_rule=wrong_hint,
        wrong_rule_hint=wrong_hint,
        wrong_fault_hint=wrong_hint,
        oracle=oracle,
        prev_hyp_summary=_pick_prev_hyp_summary(case, turn_idx),
    )


def _target_turn_indices(turns: List[Dict[str, Any]], policy: str) -> List[int]:
    if policy == "last":
        return [len(turns) - 1] if turns else []
    if policy == "host_comment":
        indices = []
        for idx, turn in enumerate(turns):
            if extract_host_comment(str(turn.get("prompt", ""))):
                indices.append(idx)
        return indices or ([len(turns) - 1] if turns else [])
    if policy == "all_after_prefix":
        return list(range(1, len(turns)))
    return [len(turns) - 1] if turns else []


def augment_noise_typology(
    case: Dict[str, Any],
    *,
    noise_type: str,
    seed: int,
    turn_policy: str = "host_comment",
    template_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if str(case.get("challenge_type", "")).lower() != "failed_isolation":
        return None, "not_failed_isolation"
    if noise_type not in NOISE_TYPES:
        return None, "invalid_noise_type"

    turns: List[Dict[str, Any]] = list(case.get("turns") or [])
    if not turns:
        return None, "empty_turns"

    template_dir = template_dir or DEFAULT_TEMPLATE_DIR
    rng = random.Random(seed)
    templates = _load_templates(noise_type, template_dir)

    scenario = "a" if "rule-guessing" in str(case.get("system_prompt", "")).lower() else "b"
    rd_style = scenario == "a"
    target_indices = _target_turn_indices(turns, turn_policy)
    source_file = str(case.get("source_file", ""))
    is_api_case = "/api/" in source_file.replace("\\", "/")
    if is_api_case and noise_type == "none":
        return None, "api_no_none"
    fixed_wrong_hint = None
    if is_api_case and noise_type != "none":
        fixed_wrong_hint = _pick_turn0_wrong_hint(case, rng)
        if not fixed_wrong_hint:
            return None, "api_no_turn0_wrong_hint"

    new_turns = []
    shuffled_templates = list(templates)
    rng.shuffle(shuffled_templates)
    template_cursor = 0
    for idx, turn in enumerate(turns):
        prompt = str(turn["prompt"])
        if idx in target_indices:
            if noise_type == "none":
                prompt = strip_host_comment(prompt)
            else:
                if template_cursor >= len(shuffled_templates):
                    shuffled_templates = list(templates)
                    rng.shuffle(shuffled_templates)
                    template_cursor = 0
                template = shuffled_templates[template_cursor]
                template_cursor += 1
                comment = render_noise_comment(
                    template=template,
                    case=case,
                    turn_idx=idx,
                    rng=rng,
                    fixed_wrong_hint=fixed_wrong_hint,
                )
                prompt = inject_host_comment(prompt, comment, rd_style=rd_style)
        new_turns.append({"prompt": prompt, "golden": list(turn.get("golden") or [])})

    suffix = f"{noise_type}_s{seed}"
    new_case_id = build_augmented_case_id(str(case["case_id"]), "noise", suffix)
    params = {
        "noise_type": noise_type,
        "seed": seed,
        "turn_policy": turn_policy,
        "target_turn_indices": target_indices,
        "case_id": new_case_id,
    }
    augmented = attach_augmentation(
        {**case, "turns": new_turns, "case_id": new_case_id},
        pipeline="noise_typology",
        params=params,
        source_case_id=str(case["case_id"]),
        extra={"noise_type": noise_type},
    )
    return augmented, None
