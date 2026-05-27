"""Host-driven belief statistics for larger candidate spaces.

This experiment differs from self-proposal mode:
- The environment provides triples and acceptance/rejection outcomes each round.
- The model only updates the hypothesis space.
- Triple generation logic is reused from task_a.experiments.generate_sequences.

Supported modes:
- failed_stay: uses generate_failed_stay_sequence
- failed_update: uses generate_failed_update_sequence_v2

Supported inference backend:
- api: OpenAI-compatible API calls with thread-pool parallelism.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import httpx
from openai import OpenAI

from task_a.core.rules import (
    EXTENDED_BENCHMARK_RULES as BENCHMARK_RULES,
    EXTENDED_RULES as RULES,
    Triple,
    get_rule,
    resolve_heldout_rules,
)
from task_a.experiments.generate_sequences import (
    generate_failed_stay_sequence,
    generate_failed_update_sequence_v2 as generate_failed_update_sequence_v2_legacy,
)
from task_a.experiments.host_driven_sequences import (
    generate_failed_stay_sequence_v2,
    generate_failed_update_sequence_v2_host,
)


API_KEY_ENV = "DMX_API_KEY"
BASE_URL_DEFAULT = ""
MODEL_DEFAULT = "deepseek-v3.2"
DEFAULT_ACCEPTED_TRIPLE: Triple = (2, 4, 6)
TRIPLE_RANGE: Tuple[int, int] = (-20, 20)
MAX_SAMPLE_ATTEMPTS = 12000
CATEGORIES = ["insufficient_capability", "oracle_match", "belief_failure", "unstable"]
ROUND_MATCH_CATEGORIES = CATEGORIES
BASE_ROUND_MATCH_CATEGORIES = CATEGORIES[:3]
CATEGORY_DIR_NAMES = {
    "insufficient_capability": "insufficient_capability",
    "oracle_match": "oracle_match",
    "belief_failure": "belief_failure",
    "unstable": "unstable",
}


def _ordered_thread_map(func: Callable[[Any], Any], items: List[Any], max_workers: int) -> List[Any]:
    if not items:
        return []
    if max_workers <= 1 or len(items) == 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
        return list(executor.map(func, items))


def build_category_dirs(output_dir: str) -> Dict[str, str]:
    return {
        category: os.path.join(output_dir, dirname)
        for category, dirname in CATEGORY_DIR_NAMES.items()
    }


def _write_result_turn_checkpoints(
    *,
    checkpoint_dir: str,
    results: List[Dict[str, Any]],
) -> None:
    rows_by_turn: Dict[int, List[Dict[str, Any]]] = {}
    for result in results:
        for message in result.get("messages", []):
            if message.get("role") != "assistant":
                continue
            turn = message.get("turn")
            if not isinstance(turn, int):
                continue
            rows_by_turn.setdefault(turn, []).append(
                {
                    "turn": turn,
                    "turn_number": turn + 1,
                    "sample_id": result.get("sample_id"),
                    "experiment_id": result.get("experiment_id"),
                    "mode": result.get("mode"),
                    "oracle": result.get("oracle"),
                    "run_idx": result.get("run_idx"),
                    "repeat_index": result.get("repeat_index"),
                    "phase": message.get("phase"),
                    "user_message": message.get("user_prompt", ""),
                    "host_triple": message.get("host_triple"),
                    "host_result": message.get("host_result"),
                    "host_evidence_batch": message.get("host_evidence_batch"),
                    "golden_hypotheses": message.get("golden_hypotheses"),
                    "model_hypotheses": message.get("model_hypotheses"),
                    "model_matches_golden": message.get("model_matches_golden"),
                    "parse_ok": message.get("parse_ok"),
                    "assistant_response": message.get("content", ""),
                    "turn_result": message,
                }
            )

    if not rows_by_turn:
        return

    os.makedirs(checkpoint_dir, exist_ok=True)
    max_turn = max(rows_by_turn)
    for turn in sorted(rows_by_turn):
        path = os.path.join(checkpoint_dir, f"turn_{turn + 1:02d}.jsonl")
        tmp_path = f"{path}.tmp"
        rows = rows_by_turn[turn]
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in rows:
                row["max_turns_observed"] = max_turn + 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
        print(
            f"[checkpoint] saved turn {turn + 1}/{max_turn + 1}: "
            f"{path} rows={len(rows)}",
            flush=True,
        )


def normalize_category(category: str) -> str:
    if category == "oracle_match":
        return "oracle_match"
    if category == "insufficient_belief":
        return "belief_failure"
    if category in CATEGORIES:
        return category
    return "insufficient_capability"


def to_jsonable(obj: Any) -> Any:
    """Recursively convert Python objects to JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        # Keep output stable for easier diffing and reproducibility.
        return sorted(to_jsonable(v) for v in obj)
    return obj


def extract_round_matches_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract round-level golden/model hypothesis alignment from a result item."""
    rounds = []
    for message in result.get("messages", []):
        golden = message.get("golden_hypotheses")
        model = message.get("model_hypotheses")
        if golden is None or model is None:
            continue
        rounds.append(
            {
                "turn": message.get("turn"),
                "phase": message.get("phase"),
                "golden_hypotheses": golden,
                "model_hypotheses": model,
                "match": set(golden) == set(model),
            }
        )
    return rounds


def _post_interference_start_turn(result: Dict[str, Any]) -> Optional[int]:
    sequence = result.get("challenge_sequence") or {}
    value = sequence.get("interference_round_start_turn")
    if not isinstance(value, int):
        value = result.get("failed_stay_interference_round_start")
    return int(value) if isinstance(value, int) else None


def collect_failed_stay_post_events_from_rounds(
    rounds: List[Dict[str, Any]],
    *,
    oracle: str,
    post_start_turn: Optional[int],
) -> List[Dict[str, Any]]:
    if post_start_turn is None:
        return []
    events = []
    for round_info in rounds:
        turn = round_info.get("turn")
        if not isinstance(turn, int) or turn < post_start_turn:
            continue
        golden_set = set(round_info.get("golden_hypotheses") or [])
        model_set = set(round_info.get("model_hypotheses") or [])
        if model_set == golden_set:
            continue
        extra_rules = sorted(model_set - {oracle})
        missing_rules = sorted(golden_set - model_set)
        events.append(
            {
                "turn": turn,
                "extra_rules": extra_rules,
                "missing_rules": missing_rules,
                "model_hypotheses": sorted(model_set),
                "golden_hypotheses": sorted(golden_set),
            }
        )
    return events


def collect_failed_stay_post_events(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find post-convergence failed_stay events: post turns with non-oracle rules reintroduced."""
    oracle = result.get("oracle")
    if not oracle:
        return []
    return collect_failed_stay_post_events_from_rounds(
        extract_round_matches_from_result(result),
        oracle=str(oracle),
        post_start_turn=_post_interference_start_turn(result),
    )


def classify_round_match_result(result: Dict[str, Any]) -> str:
    """Classify a result using mode-specific round-match patterns."""
    rounds = extract_round_matches_from_result(result)
    pattern = [r["match"] for r in rounds]
    mode = result.get("mode")

    if mode == "failed_update":
        if len(pattern) < 3 or not all(pattern[:2]):
            return "insufficient_capability"
        if pattern[2]:
            return "oracle_match"
        return "belief_failure"

    if mode == "failed_stay":
        post_start = _post_interference_start_turn(result)
        if post_start is None:
            if not rounds:
                return "insufficient_capability"
            return "oracle_match" if all(r["match"] for r in rounds) else "belief_failure"

        pre_post_rounds = [
            r for r in rounds
            if isinstance(r.get("turn"), int) and r["turn"] < post_start
        ]
        if not pre_post_rounds or not all(r["match"] for r in pre_post_rounds):
            return "insufficient_capability"

        post_rounds = [
            r for r in rounds
            if isinstance(r.get("turn"), int) and r["turn"] >= post_start
        ]
        if not post_rounds:
            return "oracle_match"
        return "oracle_match" if all(r["match"] for r in post_rounds) else "belief_failure"

    return "insufficient_capability"


def aggregate_repeat_categories(
    per_run_categories: List[str],
    threshold: float = 0.5,
) -> str:
    if not per_run_categories:
        return "insufficient_capability"
    normalized = [normalize_category(category) for category in per_run_categories]
    counts = {category: normalized.count(category) for category in BASE_ROUND_MATCH_CATEGORIES}
    category, count = max(counts.items(), key=lambda item: item[1])
    return category if count / len(normalized) > threshold else "unstable"


def analyze_round_match_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate classification counts for offline round-match analysis."""
    counts = {cat: 0 for cat in ROUND_MATCH_CATEGORIES}
    classified_results = []

    for result in results:
        category = classify_round_match_result(result)
        rounds = extract_round_matches_from_result(result)
        counts[category] += 1
        classified_results.append(
            {
                "experiment_id": result.get("experiment_id"),
                "oracle": result.get("oracle"),
                "category": category,
                "label": category,
                "round_matches": rounds,
                "round_match_pattern": [r["match"] for r in rounds],
            }
        )

    total = len(results)
    return {
        "total": total,
        "counts": counts,
        "percentages": {
            cat: round(counts[cat] / max(total, 1) * 100, 2)
            for cat in ROUND_MATCH_CATEGORIES
        },
        "results": classified_results,
    }


# ---------- API ----------

def make_client(api_key: str, base_url: str) -> OpenAI:
    if not api_key:
        raise ValueError(f"API key is empty. Set --api-key or env {API_KEY_ENV}.")
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.Client(trust_env=False, timeout=120.0),
    )


def chat(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 1024,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    max_retries: int = 3,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if top_p is not None:
                kwargs["top_p"] = top_p
            if presence_penalty is not None:
                kwargs["presence_penalty"] = presence_penalty
            extra_body: Dict[str, Any] = {}
            if top_k is not None:
                extra_body["top_k"] = top_k
            if repetition_penalty is not None:
                extra_body["repetition_penalty"] = repetition_penalty
            if extra_body:
                kwargs["extra_body"] = extra_body
            r = client.chat.completions.create(
                **kwargs,
            )
            content = r.choices[0].message.content or ""
            if content.strip():
                return content
            last_err = RuntimeError(
                f"empty content (finish_reason={r.choices[0].finish_reason})"
            )
            print(f"  [chat] empty response on attempt {attempt + 1}; retrying", flush=True)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  [chat] error on attempt {attempt + 1}: {e}", flush=True)
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"chat failed after {max_retries} retries: {last_err}")


# ---------- Parse / normalize ----------

_HYPOTHESIS_TAG_RE = re.compile(r"<hypothesis>(.*?)</hypothesis>", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ITEM_SEP_RE = re.compile(r"[,，\s]+")


def parse_agent_output(text: str) -> Optional[List[str]]:
    """Return list of hypothesis rule IDs from `<hypothesis>...</hypothesis>`,
    `[]` for `<hypothesis>none</hypothesis>`, or `None` if the tag is missing."""
    if not isinstance(text, str) or not text:
        return None
    stripped = _THINK_TAG_RE.sub("", text)
    matches = _HYPOTHESIS_TAG_RE.findall(stripped)
    if not matches:
        return None
    content = matches[-1].strip()
    if content.lower() == "none" or content == "":
        return []
    result: List[str] = []
    seen: Set[str] = set()
    for token in _ITEM_SEP_RE.split(content):
        rule_id = token.strip()
        if rule_id and rule_id not in seen:
            seen.add(rule_id)
            result.append(rule_id)
    return result


def normalize_hypotheses(raw: Optional[List[str]], candidates: List[str]) -> Set[str]:
    candidate_set = set(candidates)
    if not raw:
        return set()
    return {name for name in raw if name in candidate_set}


# ---------- Rules / evidence ----------

def all_rules(heldout_set: str = "easy") -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    merged.update(RULES)
    merged.update(resolve_heldout_rules(heldout_set))
    return merged


def candidate_rules_map(candidate_names: List[str], heldout_set: str = "easy") -> Dict[str, Any]:
    rules = all_rules(heldout_set)
    return {n: rules[n] for n in candidate_names}


def oracle_answer(oracle: str, triple: Triple) -> str:
    return "YES" if get_rule(oracle).validate(triple) else "NO"


def compute_golden_hypotheses(rules: Dict[str, Any], evidence: List[Tuple[Triple, str]]) -> Set[str]:
    out: Set[str] = set()
    for name, rule in rules.items():
        ok = True
        for triple, label in evidence:
            pred = "YES" if rule.validate(triple) else "NO"
            if pred != label:
                ok = False
                break
        if ok:
            out.add(name)
    return out


def _random_triple(rng: random.Random) -> Triple:
    lo, hi = TRIPLE_RANGE
    return (rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi))


def _sample_constrained_evidence_triple(
    *,
    rng: random.Random,
    rules: Dict[str, Any],
    oracle: str,
    active_evidence: List[Tuple[Triple, str]],
    target_survivors: Set[str],
    required_true_rules: Set[str],
) -> Optional[Triple]:
    """Sample a triple preserving target survivors and satisfying required rules."""
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        triple = _random_triple(rng)
        if any(not rules[name].validate(triple) for name in required_true_rules):
            continue
        result = "YES" if rules[oracle].validate(triple) else "NO"
        survivors = compute_golden_hypotheses(rules, active_evidence + [(triple, result)])
        if survivors == target_survivors:
            return triple
    return None


# ---------- Prompt ----------

def triple_text(triple: Triple) -> str:
    return f"({triple[0]}, {triple[1]}, {triple[2]})"


def format_candidate_rule_text(candidate_names: List[str]) -> str:
    lines = []
    for name in candidate_names:
        lines.append(f'- "{name}": {get_rule(name).description}')
    return "\n".join(lines)


def build_system_prompt(
    candidate_names: List[str],
) -> str:
    candidates_block = format_candidate_rule_text(candidate_names)
    turn_input_block = (
        "Each turn, the host provides:\n"
        "1. A triple and its currently recorded result: YES or NO.\n"
        "2. For some non-corrected turns, what each candidate rule predicts for that triple.\n"
        "If rule predictions are omitted, use the rule definitions and all active evidence directly.\n\n"
    )
    return (
        "You are a cautious scientist tracking candidate rules in a rule-guessing game.\n\n"
        f"{turn_input_block}"
        "Sometimes the host may correct an earlier recorded result.\n"
        "When that happens, the old record becomes inactive and the corrected one replaces it.\n"
        "Do not revise old evidence unless the host explicitly issues a correction.\n\n"
        "Your job is to maintain the set of rule IDs that are still consistent with all currently active recorded evidence.\n"
        "A rule is contradicted if its prediction differs from the currently recorded result for any active evidence item.\n\n"
        "Available rule IDs:\n"
        f"{candidates_block}\n\n"
        "Output format (strict):\n"
        "<hypothesis>rule_id_1, rule_id_2</hypothesis>\n\n"
        "- Inside `<hypothesis>`: comma-separated rule IDs that are still consistent with ALL active evidence.\n"
        "- Do not output anything outside the `<hypothesis>` tag."
    )


def _turn_prefix(turn: int) -> str:
    return "Let's begin.\n\n" if turn == 0 else ""


def _format_rule_predictions(
    candidate_names: List[str],
    triple: Triple,
    result: str,
    *,
    prediction_overrides: Optional[Dict[str, bool]] = None,
) -> str:
    lines = []
    _ = result
    for name in candidate_names:
        pred_yes = (
            bool(prediction_overrides[name])
            if prediction_overrides is not None and name in prediction_overrides
            else get_rule(name).validate(triple)
        )
        pred = "YES" if pred_yes else "NO"
        lines.append(f"  - {name} → {pred}")
    return "\n".join(lines)


def build_evidence_message(
    triple: Triple,
    result: str,
    turn: int,
    *,
    candidate_names: Optional[List[str]] = None,
    include_rule_predictions: bool = True,
    prediction_overrides: Optional[Dict[str, bool]] = None,
) -> str:
    body = (
        f"{_turn_prefix(turn)}"
        f"**Turn {turn} evidence:**\n"
        f"Triple {triple_text(triple)}: **{result}**\n\n"
    )
    if include_rule_predictions:
        if candidate_names is None:
            raise ValueError("candidate_names is required when include_rule_predictions=True")
        body += (
            "Rule predictions for this triple:\n"
            f"{_format_rule_predictions(candidate_names, triple, result, prediction_overrides=prediction_overrides)}\n\n"
        )
    if turn == 0:
        body += "Please update your hypotheses based on this evidence."
    else:
        body += "Please update your hypotheses using all currently active evidence."
    return body


def _include_rule_predictions_for_turn(
    *,
    mode: str,
    final_turn: bool,
    event_type: str,
) -> bool:
    if mode == "failed_update" and final_turn and event_type == "retraction":
        return False
    return event_type != "retraction"


def build_evidence_batch_message(
    batch: List[Tuple[Triple, str]],
    turn: int,
    *,
    candidate_names: Optional[List[str]] = None,
    include_rule_predictions: bool = True,
    prediction_overrides: Optional[Dict[str, bool]] = None,
    prediction_overrides_by_item: Optional[List[Optional[Dict[str, bool]]]] = None,
) -> str:
    lines = [_turn_prefix(turn) + f"**Turn {turn} evidence:**"]
    for idx, (triple, result) in enumerate(batch, 1):
        lines.append(f"{idx}. Triple {triple_text(triple)}: **{result}**")
        if include_rule_predictions:
            if candidate_names is None:
                raise ValueError("candidate_names is required when include_rule_predictions=True")
            lines.append("   Rule predictions for this triple:")
            item_prediction_overrides = (
                prediction_overrides_by_item[idx - 1]
                if prediction_overrides_by_item is not None
                else prediction_overrides
            )
            for prediction_line in _format_rule_predictions(
                candidate_names,
                triple,
                result,
                prediction_overrides=item_prediction_overrides,
            ).splitlines():
                lines.append(f"   {prediction_line.strip()}")
    lines.append("")
    if turn == 0:
        lines.append("Please update your hypotheses based on this evidence.")
    else:
        lines.append("Please update your hypotheses using all currently active evidence.")
    return "\n".join(lines)


def build_correction_message(triple: Triple, result: str, retract_turn: int, turn: int) -> str:
    return (
        f"**Turn {turn} evidence:**\n"
        f"CORRECTION for Turn {retract_turn}:\n"
        f"Triple {triple_text(triple)}: **{result}**\n\n"
        "Please update your hypotheses using all currently active evidence."
    )


# ---------- Sequence application ----------

def apply_event_update(
    *,
    oracle: str,
    event_idx: int,
    event: Dict[str, Any],
    active_evidence: List[Tuple[Triple, str]],
    event_to_active: Dict[int, int],
    candidate_names: Optional[List[str]] = None,
    include_rule_predictions: bool = True,
    prediction_overrides: Optional[Dict[str, bool]] = None,
    flip_prediction_for_rule: Optional[str] = None,
    display_turn: Optional[int] = None,
    display_retract_turn: Optional[int] = None,
) -> Tuple[str, Triple, str, str]:
    """Apply one event and return (prompt, triple, result, event_type)."""
    event_type = event["type"]
    prompt_turn = event_idx if display_turn is None else display_turn

    if event_type == "evidence":
        triple = tuple(event["triple"])  # type: ignore[assignment]
        result = event.get("recorded_result") or oracle_answer(oracle, triple)
        event_to_active[event_idx] = len(active_evidence)
        active_evidence.append((triple, result))
        effective_prediction_overrides = prediction_overrides
        if flip_prediction_for_rule is not None:
            effective_prediction_overrides = dict(prediction_overrides or {})
            effective_prediction_overrides[flip_prediction_for_rule] = not get_rule(
                flip_prediction_for_rule
            ).validate(triple)
        return (
            build_evidence_message(
                triple,
                result,
                prompt_turn,
                candidate_names=candidate_names,
                include_rule_predictions=include_rule_predictions,
                prediction_overrides=effective_prediction_overrides,
            ),
            triple,
            result,
            "evidence",
        )

    if event_type == "retraction":
        retract_turn = int(event["retract_turn"])
        prompt_retract_turn = (
            retract_turn if display_retract_turn is None else display_retract_turn
        )
        if retract_turn not in event_to_active:
            raise RuntimeError(f"Invalid retract_turn={retract_turn}; missing active mapping")
        remove_idx = event_to_active[retract_turn]
        removed_triple, _removed_result = active_evidence.pop(remove_idx)

        # shift index map after pop
        for k, v in list(event_to_active.items()):
            if v > remove_idx:
                event_to_active[k] = v - 1

        use_retracted_triple = bool(event.get("use_retracted_triple", False))
        triple = removed_triple if use_retracted_triple else tuple(event["new_triple"])  # type: ignore[assignment]
        result = event.get("new_result") or oracle_answer(oracle, triple)
        event_to_active[event_idx] = len(active_evidence)
        active_evidence.append((triple, result))
        return (
            build_correction_message(triple, result, prompt_retract_turn, prompt_turn),
            triple,
            result,
            "retraction",
        )

    raise ValueError(f"Unknown event type: {event_type}")


# ---------- Run one sequence ----------

def run_host_driven_sequence(
    *,
    client: Optional[OpenAI],
    model: str,
    mode: str,
    oracle: str,
    candidate_names: List[str],
    sequence: Dict[str, Any],
    temperature: float,
    max_tokens: int,
    top_p: Optional[float],
    top_k: Optional[int],
    presence_penalty: Optional[float],
    repetition_penalty: Optional[float],
    heldout_set: str,
    include_rule_predictions: bool = True,
    perturb_oracle_rule_prediction_in_post: bool = False,
    chat_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> Dict[str, Any]:
    rules = candidate_rules_map(candidate_names, heldout_set)
    rng = random.Random(sum(ord(ch) for ch in f"{oracle}:{mode}:{len(candidate_names)}"))

    system_prompt = build_system_prompt(candidate_names)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    message_records: List[Dict[str, Any]] = []
    message_records.append({"role": "system", "content": system_prompt, "phase": "init"})

    active_evidence: List[Tuple[Triple, str]] = []
    event_to_active: Dict[int, int] = {}
    event_to_prompt_turn: Dict[int, int] = {}
    golden = compute_golden_hypotheses(rules, active_evidence)
    model_hyp = set(candidate_names)

    turn_match_flags: List[bool] = []
    self_eliminated: Set[str] = set()
    failed_stay_events: List[Dict[str, Any]] = []

    # FailedStay constraint state (enabled from the first event where target survivor size reaches 5).
    failed_stay_rule_constraints_active = False
    failed_stay_required_rules: Set[str] = set()
    failed_stay_ever_excluded_by_golden: Set[str] = set()
    failed_stay_start_turn: Optional[int] = None
    failed_stay_convergence_turn = int(sequence.get("convergence_turn", -1))
    failed_stay_interference_round_start = sequence.get("interference_round_start_turn")

    gt_survivors_by_turn: Dict[int, Set[str]] = {}
    for step in sequence.get("ground_truth", []):
        turn = step.get("turn")
        survivors = step.get("survivors")
        if isinstance(turn, int) and isinstance(survivors, list):
            gt_survivors_by_turn[turn] = set(survivors)

    if failed_stay_rule_constraints_active and gt_survivors_by_turn:
        for t in sorted(gt_survivors_by_turn):
            if len(gt_survivors_by_turn[t]) == 5:
                failed_stay_start_turn = t
                break

    events = sequence["events"]
    batch_start_to_end: Dict[int, int] = {}
    for pair in sequence.get("evidence_batch_ranges", []):
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        start, end = int(pair[0]), int(pair[1])
        if start <= end:
            batch_start_to_end[start] = end
    i = 0
    prompt_turn = 0
    while i < len(events):
        event = events[i]
        prev_model = set(model_hyp)

        def should_flip_oracle_prediction(turn_idx: int) -> bool:
            return (
                mode == "failed_stay"
                and perturb_oracle_rule_prediction_in_post
                and isinstance(failed_stay_interference_round_start, int)
                and turn_idx >= failed_stay_interference_round_start
            )

        if (
            i in batch_start_to_end
            and all(events[k].get("type") == "evidence" for k in range(i, batch_start_to_end[i] + 1))
        ):
            batch_info: List[Tuple[Triple, str]] = []
            for k in range(i, batch_start_to_end[i] + 1):
                event_to_prompt_turn[k] = prompt_turn
                if (
                    failed_stay_rule_constraints_active
                    and failed_stay_start_turn is not None
                    and k >= failed_stay_start_turn
                    and failed_stay_required_rules
                    and k in gt_survivors_by_turn
                ):
                    sampled = _sample_constrained_evidence_triple(
                        rng=rng,
                        rules=rules,
                        oracle=oracle,
                        active_evidence=active_evidence,
                        target_survivors=gt_survivors_by_turn[k],
                        required_true_rules=failed_stay_required_rules,
                    )
                    if sampled is not None:
                        events[k]["triple"] = list(sampled)

                _prompt, triple_k, result_k, _etype = apply_event_update(
                    oracle=oracle,
                    event_idx=k,
                    event=events[k],
                    active_evidence=active_evidence,
                    event_to_active=event_to_active,
                    include_rule_predictions=False,
                )
                batch_info.append((triple_k, result_k))

            golden = compute_golden_hypotheses(rules, active_evidence)
            prompt = build_evidence_batch_message(
                batch_info,
                prompt_turn,
                candidate_names=candidate_names,
                include_rule_predictions=include_rule_predictions,
                prediction_overrides_by_item=[
                    {oracle: not get_rule(oracle).validate(triple)}
                    if should_flip_oracle_prediction(k)
                    else None
                    for k, (triple, _result) in zip(
                        range(i, batch_start_to_end[i] + 1),
                        batch_info,
                    )
                ],
            )
            messages.append({"role": "user", "content": prompt})
            message_records.append(
                {
                    "role": "user",
                    "content": prompt,
                    "turn": i,
                    "prompt_turn": prompt_turn,
                    "phase": "evidence_batch",
                    "host_evidence_batch": [
                        {"triple": list(t), "result": r} for t, r in batch_info
                    ],
                    "golden_hypotheses": sorted(golden),
                }
            )

            response = (
                chat_fn(messages)
                if chat_fn is not None
                else chat(
                    client=client,  # type: ignore[arg-type]
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    presence_penalty=presence_penalty,
                    repetition_penalty=repetition_penalty,
                )
            )
            parsed = parse_agent_output(response)
            model_hyp = normalize_hypotheses(parsed, candidate_names)

            removed = sorted(prev_model - model_hyp)
            self_eliminated.update(removed)
            readded = sorted(model_hyp & self_eliminated)
            match = model_hyp == golden
            turn_match_flags.append(match)

            if failed_stay_rule_constraints_active:
                failed_stay_ever_excluded_by_golden.update(set(candidate_names) - golden)
                if failed_stay_start_turn is not None and i >= failed_stay_start_turn:
                    added_excluded = (model_hyp - golden) & failed_stay_ever_excluded_by_golden
                    if added_excluded:
                        failed_stay_required_rules.update(added_excluded)

            message_records.append(
                {
                    "role": "assistant",
                    "content": response,
                    "turn": i,
                    "prompt_turn": prompt_turn,
                    "phase": "evidence_batch",
                    "host_evidence_batch": [
                        {"triple": list(t), "result": r} for t, r in batch_info
                    ],
                    "golden_hypotheses": sorted(golden),
                    "model_hypotheses": sorted(model_hyp),
                    "model_removed_this_turn": removed,
                    "readded_self_eliminated_rules": readded,
                    "model_matches_golden": match,
                    "parse_ok": parsed is not None,
                }
            )
            messages.append({"role": "assistant", "content": response})
            i = batch_start_to_end[i] + 1
            prompt_turn += 1
            continue

        display_retract_turn = None
        if event.get("type") == "retraction":
            display_retract_turn = event_to_prompt_turn.get(
                int(event["retract_turn"]),
                int(event["retract_turn"]),
            )
        prompt, triple, result, event_type = apply_event_update(
            oracle=oracle,
            event_idx=i,
            event=event,
            active_evidence=active_evidence,
            event_to_active=event_to_active,
            candidate_names=candidate_names,
            include_rule_predictions=_include_rule_predictions_for_turn(
                mode=mode,
                final_turn=(i == len(events) - 1),
                event_type=str(event.get("type", "")),
            ),
            flip_prediction_for_rule=oracle if should_flip_oracle_prediction(i) else None,
            display_turn=prompt_turn,
            display_retract_turn=display_retract_turn,
        )
        event_to_prompt_turn[i] = prompt_turn

        if (
            failed_stay_rule_constraints_active
            and event_type == "evidence"
            and failed_stay_start_turn is not None
            and i >= failed_stay_start_turn
            and failed_stay_required_rules
            and i in gt_survivors_by_turn
        ):
            sampled = _sample_constrained_evidence_triple(
                rng=rng,
                rules=rules,
                oracle=oracle,
                active_evidence=active_evidence[:-1],
                target_survivors=gt_survivors_by_turn[i],
                required_true_rules=failed_stay_required_rules,
            )
            if sampled is not None:
                # Re-apply this event with sampled triple so it honors failed_stay constraints.
                active_evidence.pop()
                event_to_active.pop(i, None)
                events[i]["triple"] = list(sampled)
                prompt, triple, result, event_type = apply_event_update(
                    oracle=oracle,
                    event_idx=i,
                    event=events[i],
                    active_evidence=active_evidence,
                    event_to_active=event_to_active,
                    candidate_names=candidate_names,
                    include_rule_predictions=_include_rule_predictions_for_turn(
                        mode=mode,
                        final_turn=(i == len(events) - 1),
                        event_type=str(events[i].get("type", "")),
                    ),
                    flip_prediction_for_rule=oracle if should_flip_oracle_prediction(i) else None,
                    display_turn=prompt_turn,
                    display_retract_turn=display_retract_turn,
                )
                event_to_prompt_turn[i] = prompt_turn

        golden = compute_golden_hypotheses(rules, active_evidence)

        messages.append({"role": "user", "content": prompt})
        message_records.append(
            {
                "role": "user",
                "content": prompt,
                "turn": i,
                "prompt_turn": prompt_turn,
                "phase": event_type,
                "user_prompt": prompt,
                "host_triple": list(triple),
                "host_result": result,
                "golden_hypotheses": sorted(golden),
            }
        )

        response = (
            chat_fn(messages)
            if chat_fn is not None
            else chat(
                client=client,  # type: ignore[arg-type]
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
            )
        )
        parsed = parse_agent_output(response)
        model_hyp = normalize_hypotheses(parsed, candidate_names)

        removed = sorted(prev_model - model_hyp)
        self_eliminated.update(removed)
        readded = sorted(model_hyp & self_eliminated)

        if failed_stay_rule_constraints_active:
            failed_stay_ever_excluded_by_golden.update(set(candidate_names) - golden)
            if failed_stay_start_turn is not None and i >= failed_stay_start_turn:
                added_excluded = (model_hyp - golden) & failed_stay_ever_excluded_by_golden
                if added_excluded:
                    failed_stay_required_rules.update(added_excluded)

        match = model_hyp == golden
        turn_match_flags.append(match)

        message_records.append(
            {
                "role": "assistant",
                "content": response,
                "turn": i,
                "prompt_turn": prompt_turn,
                "phase": event_type,
                "user_prompt": prompt,
                "host_triple": list(triple),
                "host_result": result,
                "golden_hypotheses": sorted(golden),
                "model_hypotheses": sorted(model_hyp),
                "model_removed_this_turn": removed,
                "readded_self_eliminated_rules": readded,
                "model_matches_golden": match,
                "parse_ok": bool(parsed),
            }
        )
        messages.append({"role": "assistant", "content": response})
        i += 1
        prompt_turn += 1

    result: Dict[str, Any] = {
        "mode": mode,
        "oracle": oracle,
        "oracle_description": get_rule(oracle).description,
        "candidate_rules": candidate_names,
        "include_rule_predictions": include_rule_predictions,
        "challenge_sequence": sequence,
        "messages": message_records,
        "final_golden_hypotheses": sorted(golden),
        "final_model_hypotheses": sorted(model_hyp),
        "final_match": model_hyp == golden,
        "turn_match_rate": sum(1 for x in turn_match_flags if x) / len(turn_match_flags),
    }

    if mode == "failed_stay":
        convergence_match = None
        convergence_golden_is_singleton_oracle = None
        for m in message_records:
            if m.get("role") != "assistant":
                continue
            if m.get("turn") != failed_stay_convergence_turn:
                continue
            convergence_match = bool(m.get("model_matches_golden"))
            golden_h = set(m.get("golden_hypotheses") or [])
            convergence_golden_is_singleton_oracle = golden_h == {oracle}
            break

        rounds = extract_round_matches_from_result(result)
        post_failed_stay_events = collect_failed_stay_post_events_from_rounds(
            rounds,
            oracle=oracle,
            post_start_turn=failed_stay_interference_round_start,
        )
        final_model_set = set(result["final_model_hypotheses"])
        final_wrong_rules = sorted(final_model_set - {oracle})
        retained_readded_rules = sorted(
            {rule for event in post_failed_stay_events for rule in event["extra_rules"]}
        )
        final_is_singleton_oracle = final_model_set == {oracle}

        result["failed_stay_detected"] = bool(post_failed_stay_events)
        result["failed_stay_events"] = post_failed_stay_events
        result["failed_stay_judgement_turn"] = failed_stay_interference_round_start
        result["failed_stay_required_rules"] = sorted(failed_stay_required_rules)
        result["failed_stay_start_turn"] = failed_stay_start_turn
        result["failed_stay_interference_round_start"] = failed_stay_interference_round_start
        result["failed_stay_post_interference_events"] = post_failed_stay_events
        result["failed_stay_post_extra_rules"] = retained_readded_rules
        result["failed_stay_convergence_match"] = convergence_match
        result["failed_stay_convergence_golden_is_singleton_oracle"] = (
            convergence_golden_is_singleton_oracle
        )
        result["failed_stay_final_wrong_rules"] = final_wrong_rules
        result["failed_stay_final_retained_readded_rules"] = retained_readded_rules
        result["failed_stay_final_is_singleton_oracle"] = final_is_singleton_oracle

    if mode == "failed_update":
        correction_checks: List[Dict[str, Any]] = []
        for m in message_records:
            if m.get("role") != "assistant" or m.get("phase") != "retraction":
                continue
            turn = m.get("turn")
            if turn is None:
                continue

            pre_rec = None
            post_rec = None
            for x in message_records:
                if x.get("role") != "assistant":
                    continue
                if x.get("turn") is not None and x.get("turn") < turn:
                    pre_rec = x
                if x.get("turn") == turn and x.get("phase") == "retraction":
                    post_rec = x

            if pre_rec is None or post_rec is None:
                continue

            pre = set(pre_rec.get("model_hypotheses") or [])
            post = set(post_rec.get("model_hypotheses") or [])
            expected = set(post_rec.get("golden_hypotheses") or [])
            missing_expected = sorted(expected - post)
            residual_wrong = sorted((pre - expected) & post)
            correction_checks.append(
                {
                    "turn": turn,
                    "expected_after_correction": sorted(expected),
                    "wrong_before_correction": sorted(pre),
                    "model_after_correction": sorted(post),
                    "missing_expected_rules": missing_expected,
                    "residual_wrong_rules": residual_wrong,
                    "is_failed_update_violation": bool(missing_expected or residual_wrong),
                }
            )

        result["failed_update_correction_checks"] = correction_checks
        result["failed_update_detected"] = bool(correction_checks and correction_checks[0]["is_failed_update_violation"])

    return result


def _init_host_session(
    *,
    row: Dict[str, Any],
    model: str,
    mode: str,
    candidate_names: List[str],
    heldout_set: str,
    include_rule_predictions: bool = True,
    perturb_oracle_rule_prediction_in_post: bool = False,
    hide_rule_predictions_in_failed_stay_post: bool = False,
) -> Dict[str, Any]:
    oracle = row["oracle"]
    sequence = json.loads(json.dumps(to_jsonable(row["sequence"])))
    rules = candidate_rules_map(candidate_names, heldout_set)
    rng = random.Random(sum(ord(ch) for ch in f"{oracle}:{mode}:{len(candidate_names)}"))
    system_prompt = build_system_prompt(candidate_names)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    message_records: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt, "phase": "init"}
    ]

    gt_survivors_by_turn: Dict[int, Set[str]] = {}
    for step in sequence.get("ground_truth", []):
        turn = step.get("turn")
        survivors = step.get("survivors")
        if isinstance(turn, int) and isinstance(survivors, list):
            gt_survivors_by_turn[turn] = set(survivors)

    failed_stay_start_turn = None
    if mode == "failed_stay" and gt_survivors_by_turn:
        for t in sorted(gt_survivors_by_turn):
            if len(gt_survivors_by_turn[t]) == 5:
                failed_stay_start_turn = t
                break

    batch_start_to_end: Dict[int, int] = {}
    for pair in sequence.get("evidence_batch_ranges", []):
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        start, end = int(pair[0]), int(pair[1])
        if start <= end:
            batch_start_to_end[start] = end

    return {
        "row": row,
        "mode": mode,
        "model": model,
        "oracle": oracle,
        "candidate_names": candidate_names,
        "heldout_set": heldout_set,
        "include_rule_predictions": include_rule_predictions,
        "perturb_oracle_rule_prediction_in_post": perturb_oracle_rule_prediction_in_post,
        "hide_rule_predictions_in_failed_stay_post": hide_rule_predictions_in_failed_stay_post,
        "rules": rules,
        "rng": rng,
        "sequence": sequence,
        "messages": messages,
        "message_records": message_records,
        "active_evidence": [],
        "event_to_active": {},
        "golden": compute_golden_hypotheses(rules, []),
        "model_hyp": set(candidate_names),
        "turn_match_flags": [],
        "self_eliminated": set(),
        "failed_stay_events": [],
        "failed_stay_rule_constraints_active": False,
        "failed_stay_required_rules": set(),
        "failed_stay_ever_excluded_by_golden": set(),
        "failed_stay_start_turn": failed_stay_start_turn,
        "failed_stay_convergence_turn": int(sequence.get("convergence_turn", -1)),
        "failed_stay_interference_round_start": sequence.get("interference_round_start_turn"),
        "gt_survivors_by_turn": gt_survivors_by_turn,
        "batch_start_to_end": batch_start_to_end,
        "event_to_prompt_turn": {},
        "prompt_turn": 0,
        "i": 0,
        "pending": None,
        "done": False,
    }


def _prepare_host_session_prompt(session: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    if session.get("done"):
        return None

    events = session["sequence"]["events"]
    i = session["i"]
    if i >= len(events):
        session["done"] = True
        return None

    rules = session["rules"]
    oracle = session["oracle"]
    rng = session["rng"]
    active_evidence = session["active_evidence"]
    event_to_active = session["event_to_active"]
    prev_model = set(session["model_hyp"])
    mode = session["mode"]
    gt_survivors_by_turn = session["gt_survivors_by_turn"]
    failed_stay_rule_constraints_active = session["failed_stay_rule_constraints_active"]
    failed_stay_start_turn = session["failed_stay_start_turn"]
    failed_stay_required_rules = session["failed_stay_required_rules"]
    batch_start_to_end = session["batch_start_to_end"]
    failed_stay_interference_round_start = session["failed_stay_interference_round_start"]
    event_to_prompt_turn = session["event_to_prompt_turn"]
    prompt_turn = int(session["prompt_turn"])

    def should_flip_oracle_prediction(turn_idx: int) -> bool:
        return (
            mode == "failed_stay"
            and bool(session.get("perturb_oracle_rule_prediction_in_post"))
            and isinstance(failed_stay_interference_round_start, int)
            and turn_idx >= failed_stay_interference_round_start
        )

    def include_predictions_for_turn(turn_idx: int, event_type: str, final_turn: bool) -> bool:
        if (
            mode == "failed_stay"
            and bool(session.get("hide_rule_predictions_in_failed_stay_post"))
            and isinstance(failed_stay_interference_round_start, int)
            and turn_idx >= failed_stay_interference_round_start
        ):
            return False
        return bool(session.get("include_rule_predictions")) and _include_rule_predictions_for_turn(
            mode=mode,
            final_turn=final_turn,
            event_type=event_type,
        )

    if (
        i in batch_start_to_end
        and all(events[k].get("type") == "evidence" for k in range(i, batch_start_to_end[i] + 1))
    ):
        batch_info: List[Tuple[Triple, str]] = []
        for k in range(i, batch_start_to_end[i] + 1):
            event_to_prompt_turn[k] = prompt_turn
            if (
                failed_stay_rule_constraints_active
                and failed_stay_start_turn is not None
                and k >= failed_stay_start_turn
                and failed_stay_required_rules
                and k in gt_survivors_by_turn
            ):
                sampled = _sample_constrained_evidence_triple(
                    rng=rng,
                    rules=rules,
                    oracle=oracle,
                    active_evidence=active_evidence,
                    target_survivors=gt_survivors_by_turn[k],
                    required_true_rules=failed_stay_required_rules,
                )
                if sampled is not None:
                    events[k]["triple"] = list(sampled)

            _prompt, triple_k, result_k, _etype = apply_event_update(
                oracle=oracle,
                event_idx=k,
                event=events[k],
                active_evidence=active_evidence,
                event_to_active=event_to_active,
                include_rule_predictions=False,
            )
            batch_info.append((triple_k, result_k))

        golden = compute_golden_hypotheses(rules, active_evidence)
        prompt = build_evidence_batch_message(
            batch_info,
            prompt_turn,
            candidate_names=session["candidate_names"],
            include_rule_predictions=include_predictions_for_turn(
                i,
                "evidence",
                batch_start_to_end[i] == len(events) - 1,
            ),
            prediction_overrides_by_item=[
                {oracle: not get_rule(oracle).validate(triple)}
                if should_flip_oracle_prediction(k)
                else None
                for k, (triple, _result) in zip(
                    range(i, batch_start_to_end[i] + 1),
                    batch_info,
                )
            ],
        )
        session["messages"].append({"role": "user", "content": prompt})
        session["message_records"].append(
            {
                "role": "user",
                "content": prompt,
                "turn": i,
                "prompt_turn": prompt_turn,
                "phase": "evidence_batch",
                "host_evidence_batch": [
                    {"triple": list(t), "result": r} for t, r in batch_info
                ],
                "golden_hypotheses": sorted(golden),
            }
        )
        session["golden"] = golden
        session["pending"] = {
            "turn": i,
            "prompt_turn": prompt_turn,
            "phase": "evidence_batch",
            "prompt": prompt,
            "golden": golden,
            "prev_model": prev_model,
            "host_evidence_batch": [
                {"triple": list(t), "result": r} for t, r in batch_info
            ],
            "next_i": batch_start_to_end[i] + 1,
        }
        return session["messages"]

    event = events[i]
    display_retract_turn = None
    if event.get("type") == "retraction":
        display_retract_turn = event_to_prompt_turn.get(
            int(event["retract_turn"]),
            int(event["retract_turn"]),
        )
    prompt, triple, result, event_type = apply_event_update(
        oracle=oracle,
        event_idx=i,
        event=event,
        active_evidence=active_evidence,
        event_to_active=event_to_active,
        candidate_names=session["candidate_names"],
        include_rule_predictions=include_predictions_for_turn(
            i,
            str(event.get("type", "")),
            i == len(events) - 1,
        ),
        flip_prediction_for_rule=oracle if should_flip_oracle_prediction(i) else None,
        display_turn=prompt_turn,
        display_retract_turn=display_retract_turn,
    )
    event_to_prompt_turn[i] = prompt_turn

    if (
        failed_stay_rule_constraints_active
        and event_type == "evidence"
        and failed_stay_start_turn is not None
        and i >= failed_stay_start_turn
        and failed_stay_required_rules
        and i in gt_survivors_by_turn
    ):
        sampled = _sample_constrained_evidence_triple(
            rng=rng,
            rules=rules,
            oracle=oracle,
            active_evidence=active_evidence[:-1],
            target_survivors=gt_survivors_by_turn[i],
            required_true_rules=failed_stay_required_rules,
        )
        if sampled is not None:
            active_evidence.pop()
            event_to_active.pop(i, None)
            events[i]["triple"] = list(sampled)
            prompt, triple, result, event_type = apply_event_update(
                oracle=oracle,
                event_idx=i,
                event=events[i],
                active_evidence=active_evidence,
                event_to_active=event_to_active,
                candidate_names=session["candidate_names"],
                include_rule_predictions=include_predictions_for_turn(
                    i,
                    str(events[i].get("type", "")),
                    i == len(events) - 1,
                ),
                flip_prediction_for_rule=oracle if should_flip_oracle_prediction(i) else None,
                display_turn=prompt_turn,
                display_retract_turn=display_retract_turn,
            )
            event_to_prompt_turn[i] = prompt_turn

    golden = compute_golden_hypotheses(rules, active_evidence)
    session["golden"] = golden
    session["messages"].append({"role": "user", "content": prompt})
    session["message_records"].append(
        {
            "role": "user",
            "content": prompt,
            "turn": i,
            "prompt_turn": prompt_turn,
            "phase": event_type,
            "user_prompt": prompt,
            "host_triple": list(triple),
            "host_result": result,
            "golden_hypotheses": sorted(golden),
        }
    )
    session["pending"] = {
        "turn": i,
        "prompt_turn": prompt_turn,
        "phase": event_type,
        "prompt": prompt,
        "triple": triple,
        "result": result,
        "golden": golden,
        "prev_model": prev_model,
        "next_i": i + 1,
    }
    return session["messages"]


def _apply_host_session_response(session: Dict[str, Any], response: str) -> None:
    pending = session["pending"]
    if pending is None:
        raise RuntimeError("Missing pending prompt for host session")

    candidate_names = session["candidate_names"]
    parsed = parse_agent_output(response)
    model_hyp = normalize_hypotheses(parsed, candidate_names)
    prev_model = set(pending["prev_model"])
    golden = set(pending["golden"])
    removed = sorted(prev_model - model_hyp)
    session["self_eliminated"].update(removed)
    readded = sorted(model_hyp & session["self_eliminated"])

    turn = pending["turn"]
    phase = pending["phase"]
    if session["failed_stay_rule_constraints_active"]:
        session["failed_stay_ever_excluded_by_golden"].update(set(candidate_names) - golden)
        failed_stay_start_turn = session["failed_stay_start_turn"]
        if failed_stay_start_turn is not None and turn >= failed_stay_start_turn:
            added_excluded = (model_hyp - golden) & session["failed_stay_ever_excluded_by_golden"]
            if added_excluded:
                session["failed_stay_required_rules"].update(added_excluded)

    match = model_hyp == golden
    session["turn_match_flags"].append(match)

    if phase == "evidence_batch":
        session["message_records"].append(
            {
                "role": "assistant",
                "content": response,
                "turn": turn,
                "prompt_turn": pending.get("prompt_turn"),
                "phase": phase,
                "host_evidence_batch": pending["host_evidence_batch"],
                "golden_hypotheses": sorted(golden),
                "model_hypotheses": sorted(model_hyp),
                "model_removed_this_turn": removed,
                "readded_self_eliminated_rules": readded,
                "model_matches_golden": match,
                "parse_ok": parsed is not None,
            }
        )
    else:
        session["message_records"].append(
            {
                "role": "assistant",
                "content": response,
                "turn": turn,
                "prompt_turn": pending.get("prompt_turn"),
                "phase": phase,
                "user_prompt": pending["prompt"],
                "host_triple": list(pending["triple"]),
                "host_result": pending["result"],
                "golden_hypotheses": sorted(golden),
                "model_hypotheses": sorted(model_hyp),
                "model_removed_this_turn": removed,
                "readded_self_eliminated_rules": readded,
                "model_matches_golden": match,
                "parse_ok": bool(parsed),
            }
        )

    session["messages"].append({"role": "assistant", "content": response})
    session["model_hyp"] = model_hyp
    session["golden"] = golden
    session["i"] = pending["next_i"]
    session["prompt_turn"] = int(session.get("prompt_turn", 0)) + 1
    session["pending"] = None
    if session["i"] >= len(session["sequence"]["events"]):
        session["done"] = True


def _finalize_host_session_result(session: Dict[str, Any]) -> Dict[str, Any]:
    row = session["row"]
    model_hyp = set(session["model_hyp"])
    golden = set(session["golden"])
    turn_match_flags = session["turn_match_flags"]
    result: Dict[str, Any] = {
        "mode": session["mode"],
        "oracle": session["oracle"],
        "oracle_description": get_rule(session["oracle"]).description,
        "candidate_rules": session["candidate_names"],
        "include_rule_predictions": bool(session.get("include_rule_predictions")),
        "hide_rule_predictions_in_failed_stay_post": bool(
            session.get("hide_rule_predictions_in_failed_stay_post")
        ),
        "challenge_sequence": session["sequence"],
        "messages": session["message_records"],
        "final_golden_hypotheses": sorted(golden),
        "final_model_hypotheses": sorted(model_hyp),
        "final_match": model_hyp == golden,
        "turn_match_rate": (
            sum(1 for x in turn_match_flags if x) / len(turn_match_flags)
            if turn_match_flags
            else 0.0
        ),
    }

    if session["mode"] == "failed_stay":
        convergence_match = None
        convergence_golden_is_singleton_oracle = None
        for m in session["message_records"]:
            if m.get("role") != "assistant":
                continue
            if m.get("turn") != session["failed_stay_convergence_turn"]:
                continue
            convergence_match = bool(m.get("model_matches_golden"))
            golden_h = set(m.get("golden_hypotheses") or [])
            convergence_golden_is_singleton_oracle = golden_h == {session["oracle"]}
            break

        rounds = extract_round_matches_from_result(result)
        post_failed_stay_events = collect_failed_stay_post_events_from_rounds(
            rounds,
            oracle=session["oracle"],
            post_start_turn=session["failed_stay_interference_round_start"],
        )
        final_model_set = set(result["final_model_hypotheses"])
        final_wrong_rules = sorted(final_model_set - {session["oracle"]})
        retained_readded_rules = sorted(
            {rule for event in post_failed_stay_events for rule in event["extra_rules"]}
        )
        final_is_singleton_oracle = final_model_set == {session["oracle"]}

        result["failed_stay_detected"] = bool(post_failed_stay_events)
        result["failed_stay_events"] = post_failed_stay_events
        result["failed_stay_judgement_turn"] = session["failed_stay_interference_round_start"]
        result["failed_stay_required_rules"] = sorted(session["failed_stay_required_rules"])
        result["failed_stay_start_turn"] = session["failed_stay_start_turn"]
        result["failed_stay_interference_round_start"] = session["failed_stay_interference_round_start"]
        result["failed_stay_post_interference_events"] = post_failed_stay_events
        result["failed_stay_post_extra_rules"] = retained_readded_rules
        result["failed_stay_convergence_match"] = convergence_match
        result["failed_stay_convergence_golden_is_singleton_oracle"] = (
            convergence_golden_is_singleton_oracle
        )
        result["failed_stay_final_wrong_rules"] = final_wrong_rules
        result["failed_stay_final_retained_readded_rules"] = retained_readded_rules
        result["failed_stay_final_is_singleton_oracle"] = final_is_singleton_oracle

    if session["mode"] == "failed_update":
        correction_checks: List[Dict[str, Any]] = []
        for m in session["message_records"]:
            if m.get("role") != "assistant" or m.get("phase") != "retraction":
                continue
            turn = m.get("turn")
            if turn is None:
                continue
            pre_rec = None
            post_rec = None
            for x in session["message_records"]:
                if x.get("role") != "assistant":
                    continue
                if x.get("turn") is not None and x.get("turn") < turn:
                    pre_rec = x
                if x.get("turn") == turn and x.get("phase") == "retraction":
                    post_rec = x
            if pre_rec is None or post_rec is None:
                continue
            pre = set(pre_rec.get("model_hypotheses") or [])
            post = set(post_rec.get("model_hypotheses") or [])
            expected = set(post_rec.get("golden_hypotheses") or [])
            missing_expected = sorted(expected - post)
            residual_wrong = sorted((pre - expected) & post)
            correction_checks.append(
                {
                    "turn": turn,
                    "expected_after_correction": sorted(expected),
                    "wrong_before_correction": sorted(pre),
                    "model_after_correction": sorted(post),
                    "missing_expected_rules": missing_expected,
                    "residual_wrong_rules": residual_wrong,
                    "is_failed_update_violation": bool(missing_expected or residual_wrong),
                }
            )
        result["failed_update_correction_checks"] = correction_checks
        result["failed_update_detected"] = bool(
            correction_checks and correction_checks[0]["is_failed_update_violation"]
        )

    result["experiment_id"] = row["repeat_experiment_id"]
    result["sample_id"] = row["sample_id"]
    result["model"] = session["model"]
    result["seed"] = row["seed"]
    result["run_idx"] = row["run_idx"]
    result["repeat_index"] = row["repeat_index"]
    result.pop("challenge_sequence", None)
    return to_jsonable(result)


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host-driven API reasoning (failed_stay/failed_update)")
    parser.add_argument(
        "--analyze-results-json",
        type=str,
        default="",
        help="Offline analysis mode: path to an existing all_results.json file",
    )
    parser.add_argument(
        "--save-labeled-results",
        type=str,
        default="",
        help="Optional path to save per-item labels for --analyze-results-json",
    )
    parser.add_argument("--mode", choices=["failed_stay", "failed_update"], default="failed_stay")
    parser.add_argument("--backend", choices=["api"], default="api")
    parser.add_argument(
        "--evidence-per-round",
        choices=[1, 4],
        type=int,
        default=4,
        help=(
            "Host evidence count per round for failed_stay/failed_update host-v2 heldout generation. "
            "Use 4 for batched rounds or 1 for single-evidence rounds."
        ),
    )
    parser.set_defaults(perturb_oracle_rule_prediction_in_post=False)
    parser.add_argument(
        "--perturb-oracle-rule-prediction-in-post",
        dest="perturb_oracle_rule_prediction_in_post",
        action="store_true",
        help=(
            "During failed_stay post-interference turns, flip only the oracle row in "
            "the displayed rule_predictions. The recorded YES/NO evidence is unchanged."
        ),
    )
    parser.add_argument(
        "--no-perturb-oracle-rule-prediction-in-post",
        dest="perturb_oracle_rule_prediction_in_post",
        action="store_false",
        help="Keep oracle rule_predictions truthful during failed_stay post-interference turns.",
    )
    parser.add_argument("--rules", nargs="+", default=list(BENCHMARK_RULES))
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help=(
            "Optional target/oracle rule names. If omitted with --include-heldout, selected heldout rules "
            "are used as targets; otherwise all candidate rules are used. "
            "Targets must be a subset of final candidate rules (rules + optional heldout)."
        ),
    )
    parser.add_argument("--include-heldout", action="store_true")
    parser.add_argument("--heldout-set", choices=["easy", "hard"], default="easy")
    parser.add_argument(
        "--heldout-rules",
        nargs="+",
        default=None,
        help=(
            "Heldout rules to add to the candidate space and sample as default targets. "
            "Only used with --include-heldout. If omitted, all rules in --heldout-set are used."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducible sampling.")
    parser.add_argument("--num-runs", type=int, default=1, help="Sequences per target/oracle rule.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeated samples per generated sequence.")
    parser.add_argument(
        "--post-convergence-interference-rounds",
        type=int,
        default=1,
        help=(
            "For failed_stay_v2 heldout generation: number of post-test interference rounds after n+5->5->1 "
            "Interference evidences will keep singleton oracle while sharing a fixed matched-rule profile."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-output-tokens", "--max-tokens", dest="max_output_tokens", type=int, default=1024)
    parser.add_argument("--top-p", dest="top_p", type=float, default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, default=None)
    parser.add_argument("--presence-penalty", dest="presence_penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=None)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--base-url", type=str, default=BASE_URL_DEFAULT)
    parser.add_argument("--model", type=str, default=MODEL_DEFAULT)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument(
        "--enable-concurrency",
        action="store_true",
        help="Deprecated: API backend always uses thread-pool execution; kept for compatibility.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Thread-pool worker count for --backend api.",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=1,
        help="Thread workers for sequence generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.analyze_results_json:
        with open(args.analyze_results_json, "r", encoding="utf-8") as f:
            results = json.load(f)

        report = analyze_round_match_results(results)
        print(f"Offline analysis: {args.analyze_results_json}", flush=True)
        print(f"Total: {report['total']}", flush=True)
        for cat in ROUND_MATCH_CATEGORIES:
            n = report["counts"][cat]
            pct = report["percentages"][cat]
            print(f"{cat}: {n}/{report['total']} ({pct:.1f}%)", flush=True)
        if args.save_labeled_results:
            with open(args.save_labeled_results, "w", encoding="utf-8") as f:
                json.dump(report["results"], f, ensure_ascii=False, indent=2)
            print(f"Labeled results saved to: {args.save_labeled_results}", flush=True)
        return

    effective_post_convergence_interference_rounds = args.post_convergence_interference_rounds
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.preprocess_workers < 1:
        raise ValueError("--preprocess-workers must be >= 1")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    candidate_names = list(args.rules)
    selected_heldout_names: List[str] = []
    if args.include_heldout:
        heldout_pool = resolve_heldout_rules(args.heldout_set)
        if args.heldout_rules is None:
            selected_heldout_names = list(heldout_pool.keys())
        else:
            selected_heldout_names = list(dict.fromkeys(args.heldout_rules))
            unknown_heldout = [n for n in selected_heldout_names if n not in heldout_pool]
            if unknown_heldout:
                raise ValueError(
                    f"Unknown heldout rules for set={args.heldout_set}: {unknown_heldout}. "
                    f"Available: {sorted(heldout_pool)}"
                )
        for n in selected_heldout_names:
            if n not in candidate_names:
                candidate_names.append(n)
    for n in candidate_names:
        get_rule(n)

    if args.targets is None:
        target_names = (
            list(selected_heldout_names)
            if args.include_heldout and selected_heldout_names
            else list(candidate_names)
        )
    else:
        target_names = list(dict.fromkeys(args.targets))
        candidate_set = set(candidate_names)
        unknown_targets = [t for t in target_names if t not in candidate_set]
        if unknown_targets:
            raise ValueError(
                "Unknown targets (or not in candidate set): "
                f"{unknown_targets}. Candidate set: {sorted(candidate_names)}"
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_label = args.model
    output_dir = args.output_dir or os.path.join(
        "outputs", f"{args.backend}_host_driven_{args.mode}_{timestamp}"
    )
    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    category_dirs = build_category_dirs(output_dir)
    for category_dir in category_dirs.values():
        os.makedirs(category_dir, exist_ok=True)

    print(f"Output dir: {output_dir}", flush=True)
    print(f"Backend: {args.backend}", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Evidence per round: {args.evidence_per_round}", flush=True)
    print("Include rule predictions: True (correction turns remain evidence-only)", flush=True)
    print(
        "Perturb oracle rule prediction in failed_stay post turns: "
        f"{args.perturb_oracle_rule_prediction_in_post}",
        flush=True,
    )
    print(
        f"Post-convergence interference rounds: {effective_post_convergence_interference_rounds}",
        flush=True,
    )
    print(f"Rules: {candidate_names}", flush=True)
    if args.include_heldout:
        print(f"Selected heldout rules: {selected_heldout_names}", flush=True)
    print(f"Targets: {target_names}", flush=True)
    print(f"Seed: {args.seed} | num_runs={args.num_runs} | repeats={args.repeats}", flush=True)
    print(
        f"API thread workers: {args.num_workers}",
        flush=True,
    )
    print(
        f"Preprocess workers: {args.preprocess_workers}",
        flush=True,
    )
    print(
        "Sampling: "
        f"temperature={args.temperature}, max_tokens={args.max_output_tokens}, "
        f"top_p={args.top_p}, top_k={args.top_k}, "
        f"presence_penalty={args.presence_penalty}, "
        f"repetition_penalty={args.repetition_penalty}",
        flush=True,
    )

    use_failed_stay_v2_heldout = args.mode == "failed_stay" and args.include_heldout
    use_failed_update_v2_heldout = args.mode == "failed_update" and args.include_heldout
    generator = generate_failed_stay_sequence if args.mode == "failed_stay" else generate_failed_update_sequence_v2_legacy

    combos: List[Tuple[str, int]] = []
    for oracle in target_names:
        for run_idx in range(args.num_runs):
            combos.append((oracle, run_idx))

    def _case_seed(oracle: str, run_idx: int) -> int:
        oracle_offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(oracle))
        return args.seed * 1000003 + oracle_offset * 9176 + run_idx

    def build_base_case(oracle: str, run_idx: int) -> Dict[str, Any]:
        exp_id = f"{args.mode}_{oracle}_r{run_idx}"
        rng = random.Random(_case_seed(oracle, run_idx))

        seq = None
        for _ in range(30):
            if use_failed_stay_v2_heldout:
                seq = generate_failed_stay_sequence_v2(
                    oracle=oracle,
                    rng=rng,
                    candidate_names=candidate_names,
                    target_sizes=[5, 1],
                    heldout_set=args.heldout_set,
                    evidence_per_round=args.evidence_per_round,
                    post_convergence_interference_rounds=effective_post_convergence_interference_rounds,
                )
            elif use_failed_update_v2_heldout:
                seq = generate_failed_update_sequence_v2_host(
                    oracle=oracle,
                    rng=rng,
                    candidate_names=candidate_names,
                    heldout_set=args.heldout_set,
                    evidence_per_round=args.evidence_per_round,
                )
            else:
                seq = generator(oracle, rng)
            if seq is not None:
                break

        if seq is None:
            return {
                "sample_id": exp_id,
                "mode": args.mode,
                "oracle": oracle,
                "seed": args.seed,
                "run_idx": run_idx,
                "error": "failed to generate sequence",
            }

        return {
            "sample_id": exp_id,
            "mode": args.mode,
            "oracle": oracle,
            "seed": args.seed,
            "run_idx": run_idx,
            "sequence": seq,
        }

    def add_grouped_case(result: Dict[str, Any]) -> None:
        oracle = str(result.get("oracle"))
        seed = int(result.get("seed"))
        key = (oracle, seed)
        grouped_cases.setdefault(key, []).append(result)

    def build_combo_case(combo: Tuple[str, int]) -> Dict[str, Any]:
        oracle, run_idx = combo
        return build_base_case(oracle, run_idx)

    print(
        f"[generate] base cases={len(combos)} preprocess_workers={args.preprocess_workers}",
        flush=True,
    )
    base_cases: List[Dict[str, Any]] = _ordered_thread_map(
        build_combo_case,
        combos,
        args.preprocess_workers,
    )
    for done, case in enumerate(base_cases, 1):
        status = "error" if "error" in case else "ok"
        print(f"[generate {done}/{len(base_cases)}] {case['sample_id']} {status}", flush=True)
    base_cases_by_sample = {case["sample_id"]: case for case in base_cases}

    expanded_rows: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []
    grouped_cases: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

    for case in base_cases:
        for repeat_idx in range(args.repeats):
            repeat_exp_id = f"{case['sample_id']}_rep{repeat_idx}"
            if "error" in case:
                result = {
                    "mode": args.mode,
                    "oracle": case["oracle"],
                    "seed": case["seed"],
                    "run_idx": case["run_idx"],
                    "repeat_index": repeat_idx,
                    "experiment_id": repeat_exp_id,
                    "sample_id": case["sample_id"],
                    "model": model_label,
                    "backend": args.backend,
                    "error": case["error"],
                    "category": "insufficient_capability",
                }
                result = to_jsonable(result)
                all_results.append(result)
                add_grouped_case(result)
                continue
            expanded_rows.append(
                {
                    "sample_id": case["sample_id"],
                    "repeat_experiment_id": repeat_exp_id,
                    "oracle": case["oracle"],
                    "seed": case["seed"],
                    "run_idx": case["run_idx"],
                    "repeat_index": repeat_idx,
                    "sequence": json.loads(json.dumps(to_jsonable(case["sequence"]))),
                }
            )

    if args.backend == "api":
        api_key = args.api_key or os.getenv(API_KEY_ENV, "")
        client = make_client(api_key=api_key, base_url=args.base_url)

        def run_api_row(row: Dict[str, Any]) -> Dict[str, Any]:
            try:
                result = run_host_driven_sequence(
                    client=client,
                    model=args.model,
                    mode=args.mode,
                    oracle=row["oracle"],
                    candidate_names=candidate_names,
                    sequence=json.loads(json.dumps(to_jsonable(row["sequence"]))),
                    temperature=args.temperature,
                    max_tokens=args.max_output_tokens,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    presence_penalty=args.presence_penalty,
                    repetition_penalty=args.repetition_penalty,
                    heldout_set=args.heldout_set,
                    include_rule_predictions=True,
                    perturb_oracle_rule_prediction_in_post=args.perturb_oracle_rule_prediction_in_post,
                )
            except Exception as e:  # noqa: BLE001
                result = {
                    "mode": args.mode,
                    "oracle": row["oracle"],
                    "seed": row["seed"],
                    "run_idx": row["run_idx"],
                    "repeat_index": row["repeat_index"],
                    "experiment_id": row["repeat_experiment_id"],
                    "sample_id": row["sample_id"],
                    "model": args.model,
                    "backend": args.backend,
                    "error": str(e),
                }
                return to_jsonable(result)

            result["experiment_id"] = row["repeat_experiment_id"]
            result["sample_id"] = row["sample_id"]
            result["model"] = args.model
            result["backend"] = args.backend
            result["seed"] = row["seed"]
            result["run_idx"] = row["run_idx"]
            result["repeat_index"] = row["repeat_index"]
            result.pop("challenge_sequence", None)
            return to_jsonable(result)

        future_to_row = {}
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for row in expanded_rows:
                future = executor.submit(run_api_row, row)
                future_to_row[future] = row
            done_count = 0
            for future in as_completed(future_to_row):
                done_count += 1
                row = future_to_row[future]
                print(f"\n[api {done_count}/{len(expanded_rows)}] {row['repeat_experiment_id']}", flush=True)
                try:
                    result = future.result()
                except Exception as e:  # noqa: BLE001
                    result = to_jsonable(
                        {
                            "mode": args.mode,
                            "oracle": row["oracle"],
                            "seed": row["seed"],
                            "run_idx": row["run_idx"],
                            "repeat_index": row["repeat_index"],
                            "experiment_id": row["repeat_experiment_id"],
                            "sample_id": row["sample_id"],
                            "model": args.model,
                            "backend": args.backend,
                            "error": str(e),
                        }
                    )
                all_results.append(result)
                add_grouped_case(result)
                if "error" not in result:
                    print(
                        f"  -> final_match={result.get('final_match')} "
                        f"turn_match_rate={result.get('turn_match_rate', 0.0):.1%}",
                        flush=True,
                    )
    for result in all_results:
        if "category" not in result:
            result["category"] = (
                "insufficient_capability" if "error" in result else classify_round_match_result(result)
            )
        result["category"] = normalize_category(str(result.get("category", "insufficient_capability")))

    _write_result_turn_checkpoints(
        checkpoint_dir=os.path.join(output_dir, "turn_checkpoints"),
        results=all_results,
    )

    sample_grouped: Dict[str, List[Dict[str, Any]]] = {}
    for result in all_results:
        sample_grouped.setdefault(str(result.get("sample_id")), []).append(result)

    sample_results: List[Dict[str, Any]] = []
    category_counts = {category: 0 for category in ROUND_MATCH_CATEGORIES}
    for sample_id, repeats in sorted(sample_grouped.items()):
        repeats_sorted = sorted(repeats, key=lambda item: int(item.get("repeat_index", 0)))
        per_run_categories = [
            normalize_category(str(item.get("category", "insufficient_capability")))
            for item in repeats_sorted
        ]
        for item, normalized_category in zip(repeats_sorted, per_run_categories):
            item["category"] = normalized_category
        category = aggregate_repeat_categories(per_run_categories)
        category_counts[category] += 1
        first = repeats_sorted[0]
        base_case = base_cases_by_sample.get(sample_id, {})
        repeat_trajectories = [
            {
                "repeat_index": int(item.get("repeat_index", idx)),
                "category": per_run_categories[idx],
                "trajectory": item,
            }
            for idx, item in enumerate(repeats_sorted)
        ]
        sample_payload = {
            "experiment_id": sample_id,
            "mode": args.mode,
            "oracle": first.get("oracle"),
            "category": category,
            "seed": first.get("seed"),
            "run_idx": first.get("run_idx"),
            "repeats": args.repeats,
            "per_run_categories": per_run_categories,
            "challenge_sequence": base_case.get("sequence"),
            "repeat_trajectories": repeat_trajectories,
        }
        with open(
            os.path.join(category_dirs[category], f"{sample_id}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(to_jsonable(sample_payload), f, indent=2, ensure_ascii=False)
        sample_results.append(
            {
                "sample_id": sample_id,
                "mode": args.mode,
                "oracle": first.get("oracle"),
                "seed": first.get("seed"),
                "run_idx": first.get("run_idx"),
                "category": category,
                "per_run_categories": per_run_categories,
                "repeat_experiment_ids": [item.get("experiment_id") for item in repeats_sorted],
            }
        )

    # Save grouped trajectories: one file per (oracle, seed) containing all num-runs cases.
    for (oracle, seed), cases in grouped_cases.items():
        cases_sorted = sorted(
            cases,
            key=lambda x: (int(x.get("run_idx", 0)), int(x.get("repeat_index", 0))),
        )
        payload = {
            "mode": args.mode,
            "backend": args.backend,
            "oracle": oracle,
            "seed": seed,
            "model": model_label,
            "repeats": args.repeats,
            "n_cases": len(cases_sorted),
            "cases": cases_sorted,
        }
        group_file = f"{args.mode}_{oracle}_s{seed}.json"
        with open(os.path.join(traj_dir, group_file), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    valid = [r for r in all_results if "error" not in r]
    summary = {
        "mode": args.mode,
        "backend": args.backend,
        "evidence_per_round": args.evidence_per_round,
        "include_rule_predictions": True,
        "perturb_oracle_rule_prediction_in_post": bool(
            args.perturb_oracle_rule_prediction_in_post
        ),
        "post_convergence_interference_rounds": effective_post_convergence_interference_rounds,
        "model": model_label,
        "rules": candidate_names,
        "selected_heldout_rules": selected_heldout_names,
        "targets": target_names,
        "seed": args.seed,
        "num_runs_per_target": args.num_runs,
        "repeats": args.repeats,
        "preprocess_workers": args.preprocess_workers,
        "api_num_workers": args.num_workers,
        "n_repeat_runs": len(all_results),
        "n_samples": len(sample_results),
        "n_valid": len(valid),
        "final_match_rate": (
            sum(1 for r in valid if r.get("final_match")) / len(valid) if valid else None
        ),
        "avg_turn_match_rate": (
            sum(r.get("turn_match_rate", 0.0) for r in valid) / len(valid) if valid else None
        ),
        "category_counts": category_counts,
        "category_percentages": {
            category: round(category_counts[category] / max(len(sample_results), 1) * 100, 2)
            for category in ROUND_MATCH_CATEGORIES
        },
        "total": {
            "total": len(sample_results),
            **{category: category_counts[category] for category in ROUND_MATCH_CATEGORIES},
            **{
                f"{category}_pct": round(
                    category_counts[category] / max(len(sample_results), 1) * 100,
                    2,
                )
                for category in ROUND_MATCH_CATEGORIES
            },
        },
        "per_oracle": {},
    }

    for oracle in target_names:
        subset = [r for r in valid if r.get("oracle") == oracle]
        sample_subset = [r for r in sample_results if r.get("oracle") == oracle]
        if not subset and not sample_subset:
            continue
        oracle_category_counts = {
            category: sum(1 for item in sample_subset if item.get("category") == category)
            for category in ROUND_MATCH_CATEGORIES
        }
        summary["per_oracle"][oracle] = {
            "n": len(subset),
            "n_samples": len(sample_subset),
            "final_match_rate": (
                sum(1 for r in subset if r.get("final_match")) / len(subset)
                if subset
                else None
            ),
            "avg_turn_match_rate": (
                sum(r.get("turn_match_rate", 0.0) for r in subset) / len(subset)
                if subset
                else None
            ),
            "category_counts": oracle_category_counts,
        }

    with open(os.path.join(output_dir, "stats_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(all_results), f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "sample_results.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(sample_results), f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60, flush=True)
    if summary["final_match_rate"] is None:
        print("No valid results.", flush=True)
    else:
        print(
            f"Done. final_match_rate={summary['final_match_rate']:.1%} "
            f"avg_turn_match_rate={summary['avg_turn_match_rate']:.1%}",
            flush=True,
        )
    for category in ROUND_MATCH_CATEGORIES:
        n = category_counts[category]
        pct = n / max(len(sample_results), 1) * 100
        print(f"{category}: {n}/{len(sample_results)} ({pct:.1f}%)", flush=True)
    print(f"Stats report: {os.path.join(output_dir, 'stats_report.json')}", flush=True)
    print(f"Sample results: {os.path.join(output_dir, 'sample_results.json')}", flush=True)


if __name__ == "__main__":
    main()
