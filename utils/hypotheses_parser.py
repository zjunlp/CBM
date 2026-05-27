"""Hypotheses parsing utilities shared by training and evaluation code."""

from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional


HYPOTHESIS_TAG_RE = re.compile(r"<hypothesis>(.*?)</hypothesis>", re.DOTALL | re.IGNORECASE)
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
ITEM_SEP_RE = re.compile(r"[,，\s]+")

PARSE_MODES = {"tag", "json", "auto"}
_default_parse_mode = "tag"


def parse_hypotheses(text: str, parse_mode: str | None = None) -> Optional[List[str]]:
    mode = parse_mode or _default_parse_mode
    if mode == "tag":
        return parse_hypotheses_tag(text)
    if mode == "json":
        return parse_hypotheses_json(text)
    if mode == "auto":
        parsed = parse_hypotheses_json(text)
        return parsed if parsed is not None else parse_hypotheses_tag(text)
    raise ValueError(f"Unsupported hypotheses parse mode: {mode}")


def parse_hypotheses_tag(text: str) -> Optional[List[str]]:
    if not isinstance(text, str) or not text:
        return None

    lower_text = text.lower()
    last_think_end = lower_text.rfind("</think>")
    search_text = text[last_think_end + len("</think>") :] if last_think_end >= 0 else text

    matches = HYPOTHESIS_TAG_RE.findall(search_text)
    if not matches and last_think_end >= 0:
        stripped = THINK_TAG_RE.sub("", text)
        matches = HYPOTHESIS_TAG_RE.findall(stripped)
    if not matches:
        return None

    content = matches[-1].strip()
    if content == "" or content.lower() == "none":
        return []
    return dedupe_nonempty(ITEM_SEP_RE.split(content))


def parse_hypotheses_json(text: str) -> Optional[List[str]]:
    if not isinstance(text, str) or not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    hypotheses = payload.get("hypotheses")
    if hypotheses is None:
        return []
    if not isinstance(hypotheses, list):
        return None

    return dedupe_nonempty(item for item in hypotheses if isinstance(item, str))


def dedupe_nonempty(items: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for item in items:
        value = item.strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def set_default_hypotheses_parse_mode(parse_mode: str) -> None:
    global _default_parse_mode
    if parse_mode not in PARSE_MODES:
        raise ValueError(f"Unsupported hypotheses parse mode: {parse_mode}")
    _default_parse_mode = parse_mode


def get_default_hypotheses_parse_mode() -> str:
    return _default_parse_mode
