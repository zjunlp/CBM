"""Utilities for Scenario A belief-steering experiments."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from utils.hypotheses_parser import parse_hypotheses_tag


CATEGORY_INSUFFICIENT = "insufficient_capability"
CATEGORY_VALID = "belief_failure"
CATEGORY_BELIEF = "oracle_match"

DEFAULT_EVAL_ROOT = Path(
    "task_a/outputs/"
    "swift_train_a_with_thinking_rollout_8_test_a_ckpt_520"
)
DEFAULT_BASE_EVAL_ROOT = Path(
    "task_a/outputs/"
    "swift_train_a_with_thinking_rollout_8_test_a_ckpt_520"
)
DEFAULT_RL_EVAL_ROOT = Path(
    "task_a/outputs/"
    "swift_train_a_with_thinking_rollout_8_test_a_ckpt_520"
)
DEFAULT_BASE_MODEL = "models/Qwen3.5-9B"
DEFAULT_RL_MODEL = (
    "task_a/training/checkpoints_multi_turn_online_swift_grpo_9B_thinking/"
    "v4-20260511-171302/checkpoint-315_merged"
)
DEFAULT_RL_MODEL_520 = (
    "task_a/training/checkpoints_multi_turn_online_swift_grpo_9B_thinking_rollout_8/"
    "v1-20260515-041104/checkpoint-520_merged"
)

CHALLENGE_TYPES = ("failed_stay", "failed_update")
CATEGORY_DIRS = {
    CATEGORY_INSUFFICIENT: "insufficient_capability",
    CATEGORY_VALID: "belief_failure",
    CATEGORY_BELIEF: "oracle_match",
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_HYPOTHESIS_BLOCK_RE = re.compile(r"<hypothesis\b[^>]*>[^<>]*</hypothesis>", re.DOTALL | re.IGNORECASE)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def response_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def extract_hypothesis_block(text: str) -> str:
    matches = _HYPOTHESIS_BLOCK_RE.findall(text or "")
    if not matches:
        return ""
    return matches[-1].strip()


def assistant_hypothesis_only_messages(messages: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    converted: List[Dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "assistant":
            content = extract_hypothesis_block(content)
        converted.append({"role": role, "content": content})
    return converted


def parse_hypotheses(text: str) -> Optional[List[str]]:
    parsed = parse_hypotheses_tag(text)
    if parsed is None:
        return None
    return list(dict.fromkeys(str(item).strip() for item in parsed if str(item).strip()))


def hypotheses_match(predicted: Optional[Sequence[str]], golden: Sequence[str]) -> bool:
    return predicted is not None and set(predicted) == set(golden)


def canonical_hypothesis_response(golden: Sequence[str]) -> str:
    return "<hypothesis>" + ", ".join(str(item) for item in golden) + "</hypothesis>"


def load_model_samples(eval_root: Path, model_label: str) -> Dict[str, Dict[str, Any]]:
    stats_path = eval_root / model_label / "stats_report.json"
    stats = read_json(stats_path)
    return {str(item["case_id"]): item for item in stats.get("sample_results", [])}


def index_case_payloads(eval_root: Path, model_label: str) -> Dict[str, Dict[str, Any]]:
    root = eval_root / model_label
    index: Dict[str, Dict[str, Any]] = {}
    for challenge_type in CHALLENGE_TYPES:
        for category_dir in CATEGORY_DIRS.values():
            case_dir = root / challenge_type / category_dir
            if not case_dir.exists():
                continue
            for path in sorted(case_dir.glob("*.json")):
                if path.name in {"stats_report.json", "summary.json", "comparison.json"}:
                    continue
                payload = read_json(path)
                case_id = str(payload.get("case_id") or path.stem)
                index[case_id] = payload
    return index


def get_repeat(payload: Dict[str, Any], repeat_index: int) -> Dict[str, Any]:
    repeats = payload.get("repeat_trajectories") or []
    for repeat in repeats:
        if int(repeat.get("repeat_index", -1)) == int(repeat_index):
            return repeat
    raise KeyError(f"repeat_index={repeat_index} not found for {payload.get('case_id')}")


def turn_records(repeat: Dict[str, Any]) -> List[Dict[str, Any]]:
    trajectory = repeat.get("trajectory") or {}
    return list(trajectory.get("turns") or [])


def first_failure_turn(repeat: Dict[str, Any]) -> int:
    turns = turn_records(repeat)
    for idx, turn in enumerate(turns):
        if not bool(turn.get("model_matches_golden")):
            return idx
    return max(0, len(turns) - 1)


def final_turn(repeat: Dict[str, Any]) -> int:
    return max(0, len(turn_records(repeat)) - 1)


def build_canonical_messages(repeat: Dict[str, Any], target_turn: int) -> List[Dict[str, str]]:
    trajectory = repeat.get("trajectory") or {}
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": str(trajectory.get("system_prompt") or "")}
    ]
    turns = turn_records(repeat)
    for idx, turn in enumerate(turns[: target_turn + 1]):
        messages.append({"role": "user", "content": str(turn.get("prompt") or "")})
        if idx < target_turn:
            golden = list(turn.get("golden_hypotheses") or [])
            messages.append({"role": "assistant", "content": canonical_hypothesis_response(golden)})
    return messages


def build_base_context_messages(repeat: Dict[str, Any], target_turn: int) -> List[Dict[str, str]]:
    trajectory = repeat.get("trajectory") or {}
    conversation = trajectory.get("conversation") or []
    if not conversation:
        return build_canonical_messages(repeat, target_turn)

    messages: List[Dict[str, str]] = []
    seen_user_turn = -1
    for message in conversation:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        if role == "user":
            seen_user_turn += 1
            messages.append({"role": "user", "content": str(message.get("content") or "")})
            if seen_user_turn >= target_turn:
                break
            continue
        messages.append({"role": str(role), "content": str(message.get("content") or "")})
    return messages


def render_chat_prompt(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    *,
    enable_thinking: Optional[bool] = False,
) -> str:
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        rendered = tokenizer.apply_chat_template(list(messages), **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        rendered = tokenizer.apply_chat_template(list(messages), **kwargs)
    if isinstance(rendered, list):
        return str(rendered[0])
    return str(rendered)


def normalize_layers(raw_layers: Iterable[int], num_layers: int) -> List[int]:
    layers: List[int] = []
    for layer in raw_layers:
        value = int(layer)
        if value < 0:
            value = num_layers + value
        if value < 0 or value >= num_layers:
            raise ValueError(f"layer {layer} out of range for num_layers={num_layers}")
        if value not in layers:
            layers.append(value)
    return layers


def get_transformer_layers(model: Any) -> Sequence[Any]:
    candidates = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj
    raise AttributeError("cannot locate transformer layers on model")


def short_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "record_id": record.get("record_id"),
        "case_id": record.get("case_id"),
        "repeat_index": record.get("repeat_index"),
        "challenge_type": record.get("challenge_type"),
        "target_turn": record.get("target_turn"),
        "golden_hypotheses": record.get("golden_hypotheses"),
    }
