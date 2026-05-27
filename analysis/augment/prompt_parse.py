"""Parse rendered RD prompts back into structured evidence."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from task_a.experiments.belief_stats import Triple

TRIPLE_EVIDENCE_RE = re.compile(
    r"Triple\s*\((-?\d+),\s*(-?\d+),\s*(-?\d+)\):\s*\*\*(YES|NO)\*\*",
    re.IGNORECASE,
)
CORRECTION_RE = re.compile(
    r"CORRECTION:.*?triple\s*\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)\s*→\s*(YES|NO)\).*?"
    r"correct result for that same triple is\s*\*\*(YES|NO)\*\*",
    re.IGNORECASE | re.DOTALL,
)
TURN_NUMBER_RE = re.compile(r"\*\*Turn\s+(\d+)\s+evidence:\*\*")
RD_TURN_IN_CORRECTION_RE = re.compile(r"from Turn\s+(\d+)", re.IGNORECASE)
CD_TURN_NUMBER_RE = re.compile(r"^Turn\s+(\d+):", re.MULTILINE)
HOST_COMMENT_RE = re.compile(r"\nHost comment:\s*(.*)\Z", re.DOTALL)
HOST_NOTE_RE = re.compile(r"\n\*\*Host note:\*\*\s*(.*)\Z", re.DOTALL)
TURN_MEASUREMENTS_RE = re.compile(
    r"(?P<prefix>.*?^Turn\s+\d+:.*?^-\s*Measurements:\s*\n(?:\s*-\s+.*\n)+)"
    r"(?P<comment>.*?)"
    r"(?P<suffix>\nPlease update your hypotheses\.?)\s*\Z",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
INLINE_UPDATE_RE = re.compile(
    r"(?P<prefix>(?:\*\*Turn\s+\d+\s+evidence:\*\*\n)?(?:Triple\s*\([^)]*\):\s*\*\*(?:YES|NO)\*\*\.?))"
    r"(?P<comment>\s+.*?)"
    r"(?P<suffix>\s+Please update your hypotheses\.?)\s*\Z",
    re.IGNORECASE | re.DOTALL,
)


def _to_triple(groups: Tuple[str, str, str]) -> Triple:
    return (int(groups[0]), int(groups[1]), int(groups[2]))


def parse_triple_evidence(prompt: str) -> Optional[Tuple[Triple, str]]:
    if "CORRECTION" in prompt:
        return None
    match = TRIPLE_EVIDENCE_RE.search(prompt)
    if not match:
        return None
    return _to_triple(match.groups()[:3]), match.group(4).upper()


def parse_correction(prompt: str) -> Optional[Tuple[Triple, str, str]]:
    match = CORRECTION_RE.search(prompt)
    if not match:
        return None
    triple = _to_triple(match.groups()[:3])
    old_label = match.group(4).upper()
    new_label = match.group(5).upper()
    return triple, old_label, new_label


def has_correction(prompt: str) -> bool:
    return "CORRECTION" in prompt


def rebuild_active_evidence(turns: List[dict], up_to_idx: int) -> List[Tuple[Triple, str]]:
    evidence: List[Tuple[Triple, str]] = []
    for idx in range(up_to_idx + 1):
        prompt = str(turns[idx].get("prompt", ""))
        correction = parse_correction(prompt)
        if correction is not None:
            triple, _old_label, new_label = correction
            evidence = [(t, label) for t, label in evidence if t != triple]
            evidence.append((triple, new_label))
            continue
        parsed = parse_triple_evidence(prompt)
        if parsed is not None:
            evidence.append(parsed)
    return evidence


def renumber_rd_turn_prompt(prompt: str, turn_map: dict[int, int]) -> str:
    def _replace_turn(match: re.Match[str]) -> str:
        old = int(match.group(1))
        new = turn_map.get(old, old)
        return f"**Turn {new} evidence:**"

    updated = TURN_NUMBER_RE.sub(_replace_turn, prompt)

    def _replace_corr_turn(match: re.Match[str]) -> str:
        old = int(match.group(1))
        new = turn_map.get(old, old)
        return f"from Turn {new}"

    return RD_TURN_IN_CORRECTION_RE.sub(_replace_corr_turn, updated)


def renumber_cd_turn_prompt(prompt: str, new_turn: int) -> str:
    return CD_TURN_NUMBER_RE.sub(f"Turn {new_turn}:", prompt, count=1)


def extract_host_comment(prompt: str) -> Optional[str]:
    match = HOST_COMMENT_RE.search(prompt)
    if match:
        return match.group(1).strip()
    match = HOST_NOTE_RE.search(prompt)
    if match:
        return match.group(1).strip()
    match = TURN_MEASUREMENTS_RE.search(prompt)
    if match:
        return match.group("comment").strip()
    match = INLINE_UPDATE_RE.search(prompt)
    if match:
        return match.group("comment").strip()
    return None


def strip_host_comment(prompt: str) -> str:
    prompt = HOST_COMMENT_RE.sub("", prompt).rstrip()
    prompt = HOST_NOTE_RE.sub("", prompt).rstrip()
    match = TURN_MEASUREMENTS_RE.search(prompt)
    if match:
        return f"{match.group('prefix').rstrip()}\n\n{match.group('suffix').strip()}".rstrip()
    match = INLINE_UPDATE_RE.search(prompt)
    if match:
        return f"{match.group('prefix')} {match.group('suffix').strip()}".rstrip()
    return prompt


def inject_host_comment(prompt: str, comment: str, *, rd_style: bool = False) -> str:
    match = TURN_MEASUREMENTS_RE.search(prompt)
    if match:
        return (
            f"{match.group('prefix').rstrip()}\n\n{comment.strip()}\n"
            f"{match.group('suffix').strip()}"
        ).rstrip()
    match = INLINE_UPDATE_RE.search(prompt)
    if match:
        return f"{match.group('prefix')} {comment.strip()} {match.group('suffix').strip()}".rstrip()
    base = strip_host_comment(prompt)
    if rd_style:
        return f"{base}\n**Host note:** {comment}"
    return f"{base}\nHost comment: {comment}"
