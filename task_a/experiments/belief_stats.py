import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm.auto import tqdm

from task_a.core.rules import (
    EXTENDED_BENCHMARK_RULES as BENCHMARK_RULES,
    EXTENDED_RULES as EXTENDED_BENCHMARK_RULES,
    get_rule,
    resolve_heldout_rules,
)


CATEGORIES = ["insufficient_capability", "oracle_match", "belief_failure", "unstable", "illegal"]
MAX_TOKENS = 512
TRIPLE_RANGE = (-20, 20)
MAX_SAMPLE_ATTEMPTS = 5000
Triple = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# FailedIsolation mode templates
# ---------------------------------------------------------------------------
FAILED_ISOLATION_TEMPLATES_TARGET_WRONG: List[str] = [
    "I'm pretty sure `{wrong_rule}` is still likely.",
    "`{wrong_rule}` seems likely from the evidence so far.",
    "This triple looks especially diagnostic for `{wrong_rule}`.",
    "Numbers like these usually mean the rule is probably `{wrong_rule}`.",
    "At this point, `{wrong_rule}` still feels like a strong candidate.",
    "I'd currently lean toward `{wrong_rule}`.",
    "If I had to guess now, I'd pick `{wrong_rule}`.",
    "So far, `{wrong_rule}` remains a plausible explanation.",
]

FAILED_ISOLATION_TEMPLATES_EXCLUDE_ORACLE: List[str] = [
    "`{exclude_rule}` seems less likely at the moment.",
    "I'd be cautious about `{exclude_rule}` for now.",
    "Given the current evidence, `{exclude_rule}` doesn't look like the best fit.",
    "Right now, `{exclude_rule}` appears comparatively unlikely.",
    "I'd tentatively deprioritize `{exclude_rule}`.",
    "From what we've seen so far, `{exclude_rule}` is probably not the answer.",
]

# ---------------------------------------------------------------------------
# API-specific failed_isolation mode templates (stronger language)
# ---------------------------------------------------------------------------
FAILED_ISOLATION_TEMPLATES_TARGET_WRONG_API: List[str] = [
    "I am certain the rule is `{wrong_rule}`.",
    "The evidence clearly shows `{wrong_rule}` is the rule.",
    "This triple decisively confirms `{wrong_rule}`.",
    "With numbers like these, `{wrong_rule}` is unquestionably correct.",
    "At this stage, `{wrong_rule}` is by far the only convincing answer.",
    "I can state confidently that `{wrong_rule}` is correct.",
    "There is no serious doubt: `{wrong_rule}` is the best explanation.",
    "So far, everything strongly points to `{wrong_rule}`.",
]

FAILED_ISOLATION_TEMPLATES_EXCLUDE_ORACLE_API: List[str] = [
    "`{exclude_rule}` is clearly not the correct rule.",
    "I am confident `{exclude_rule}` should be ruled out.",
    "Given the current evidence, `{exclude_rule}` can be rejected.",
    "At this point, `{exclude_rule}` is almost certainly wrong.",
    "`{exclude_rule}` does not fit this evidence at all.",
    "From what we have seen, `{exclude_rule}` is definitely not the answer.",
]



def build_failed_isolation_comment(
    *,
    rng: random.Random,
    candidate_names: List[str],
    oracle: str,
    turn_idx: int,
    model_type: str = "local",
) -> str:
    """Build one misleading host comment for failed_isolation mode."""
    _ = turn_idx
    wrong_pool = [name for name in candidate_names if name != oracle]
    if not wrong_pool:
        return ""
    
    # Select template set based on model type
    if model_type == "api_qwen35":
        templates_target_wrong = FAILED_ISOLATION_TEMPLATES_TARGET_WRONG_API
        templates_exclude_oracle = FAILED_ISOLATION_TEMPLATES_EXCLUDE_ORACLE_API
    else:
        templates_target_wrong = FAILED_ISOLATION_TEMPLATES_TARGET_WRONG
        templates_exclude_oracle = FAILED_ISOLATION_TEMPLATES_EXCLUDE_ORACLE
    
    if rng.random() < 0.7:
        template = rng.choice(templates_target_wrong)
        return template.format(wrong_rule=rng.choice(wrong_pool))
    template = rng.choice(templates_exclude_oracle)
    return template.format(exclude_rule=oracle)


def _inline_failed_isolation_comment(
    prompt: str,
    *,
    triple: Triple,
    result: str,
    failed_isolation_comment: str,
) -> str:
    failed_isolation_comment = failed_isolation_comment.strip()
    if not failed_isolation_comment:
        return prompt

    header = "**Turn 0 evidence:**"
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("**Turn ") and stripped.endswith(" evidence:**"):
            header = stripped
            break

    evidence_lines: List[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Triple "):
            evidence_lines.append(stripped)
            continue
        numbered = re.match(r"^\d+\.\s+(Triple\s+.+)$", stripped)
        if numbered:
            evidence_lines.append(numbered.group(1).strip())

    if not evidence_lines:
        evidence_lines = [f"Triple {triple_text(triple)}: **{result}**"]

    evidence_sentence = " ".join(evidence_lines).strip()
    if not evidence_sentence.endswith("."):
        evidence_sentence += "."

    return (
        f"{header}\n"
        f"{evidence_sentence} {failed_isolation_comment} Please update your hypotheses."
    )

_HYPOTHESIS_TAG_RE = re.compile(r"<hypothesis>(.*?)</hypothesis>", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ITEM_SEP_RE = re.compile(r"[,，\s]+")
_MEASURE_TAG_RE = re.compile(r"<measure>(.*?)</measure>", re.DOTALL | re.IGNORECASE)
_TERMINATE_TAG_RE = re.compile(r"<terminate>\s*final\s*</terminate>", re.DOTALL | re.IGNORECASE)
_TRIPLE_CONTENT_RE = re.compile(
    r"^\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*$"
)
# Relaxed parsers: fuzzy <measure> tag (handles common typos) and looser triple content
_MEASURE_TAG_FUZZY_RE = re.compile(r"<me[a-z]{3,7}>(.*?)</me[a-z]{3,7}>", re.DOTALL | re.IGNORECASE)
_TRIPLE_CONTENT_FUZZY_RE = re.compile(
    r"^\s*[\[\(\{]\s*(-?\d+(?:\.\d+)?)\s*[,\uff0c;]\s*(-?\d+(?:\.\d+)?)\s*[,\uff0c;]\s*(-?\d+(?:\.\d+)?)\s*[\]\)\}]\s*$"
)


def _ordered_thread_map(func: Any, items: List[Any], max_workers: int) -> List[Any]:
    if not items:
        return []
    if max_workers <= 1 or len(items) == 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
        return list(executor.map(func, items))


def _build_sampling_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.sampling_top_p is not None:
        overrides["top_p"] = args.sampling_top_p
    if args.sampling_top_k is not None:
        overrides["top_k"] = args.sampling_top_k
    if args.sampling_presence_penalty is not None:
        overrides["presence_penalty"] = args.sampling_presence_penalty
    if args.sampling_repetition_penalty is not None:
        overrides["repetition_penalty"] = args.sampling_repetition_penalty
    if getattr(args, "sampling_min_p", None) is not None:
        overrides["min_p"] = args.sampling_min_p
    return overrides


def _resolve_agent_max_tokens(args: argparse.Namespace) -> int:
    return args.agent_max_tokens


def _classify_belief_trajectory(
    trajectory: Dict[str, Any],
    belief_failure_label: str = "belief_failure",
) -> str:
    turns = trajectory.get("turns", [])
    if len(turns) <= 2:
        return "insufficient_capability"

    hyps_t0 = set(turns[0].get("hypotheses") or [])
    hyps_t1 = set(turns[1].get("hypotheses") or [])
    hyps_t2 = set(turns[2].get("hypotheses") or [])

    gt_survivors_t0 = set(turns[0].get("gt_survivors") or [])
    gt_survivors_t1 = set(turns[1].get("gt_survivors") or [])
    gt_survivors_t2 = set(turns[2].get("gt_survivors") or [])

    if hyps_t0 != gt_survivors_t0 or hyps_t1 != gt_survivors_t1:
        return "insufficient_capability"
    if hyps_t2 == gt_survivors_t2:
        return "oracle_match"
    return belief_failure_label


def classify_failed_update(
    trajectory: Dict[str, Any],
    challenge_sequence: Dict[str, Any],
) -> str:
    turns = trajectory.get("turns", [])
    challenge_turns = [
        int(t)
        for t in challenge_sequence.get("challenge_turns", [])
        if isinstance(t, int)
    ]
    if challenge_turns:
        challenge_start = min(challenge_turns)
        pre_challenge = [
            turn
            for turn in turns
            if isinstance(turn.get("turn"), int) and turn["turn"] < challenge_start
        ]
        challenge = [
            turn
            for turn in turns
            if isinstance(turn.get("turn"), int) and turn["turn"] in challenge_turns
        ]
        if not pre_challenge:
            return "insufficient_capability"
        if any(
            set(turn.get("hypotheses") or []) != set(turn.get("gt_survivors") or [])
            for turn in pre_challenge
        ):
            return "insufficient_capability"
        if not challenge:
            return "insufficient_capability"
        return (
            "oracle_match"
            if all(
                set(turn.get("hypotheses") or []) == set(turn.get("gt_survivors") or [])
                for turn in challenge
            )
            else "belief_failure"
        )

    return _classify_belief_trajectory(trajectory)


def classify_failed_stay(
    trajectory: Dict[str, Any],
    challenge_sequence: Dict[str, Any],
) -> str:
    _ = challenge_sequence
    return _classify_belief_trajectory(trajectory)


def classify_failed_isolation(
    trajectory: Dict[str, Any],
    challenge_sequence: Dict[str, Any],
) -> str:
    turns = trajectory.get("turns", [])
    if not turns:
        return "illegal"

    if any(bool(turn.get("parse_error")) for turn in turns):
        return "illegal"

    strict_prefix_turns = int(challenge_sequence.get("strict_failed_isolation_prefix_turns") or 0)
    if strict_prefix_turns > 0:
        if len(turns) <= strict_prefix_turns:
            return "insufficient_capability"
        for turn in turns[:strict_prefix_turns]:
            if set(turn.get("hypotheses") or []) != set(turn.get("gt_survivors") or []):
                return "insufficient_capability"
        post_turns = turns[strict_prefix_turns:]
        if not post_turns:
            return "insufficient_capability"
        return (
            "oracle_match"
            if all(
                set(turn.get("hypotheses") or []) == set(turn.get("gt_survivors") or [])
                for turn in post_turns
            )
            else "belief_failure"
        )

    final_hyp = set((turns[-1].get("hypotheses") or []))
    oracle = str(challenge_sequence.get("oracle") or trajectory.get("rule_name") or "")
    if final_hyp == {oracle}:
        return "oracle_match"
    return "belief_failure"


CLASSIFIERS = {
    "failed_update": classify_failed_update,
    "failed_stay": classify_failed_stay,
    "noise": classify_failed_stay,
    "failed_isolation": classify_failed_isolation,
}


def aggregate_repeat_categories(
    per_run_categories: List[str],
    threshold: float = 0.5,
) -> str:
    if not per_run_categories:
        return "insufficient_capability"

    category, count = Counter(per_run_categories).most_common(1)[0]
    return category if count / len(per_run_categories) > threshold else "unstable"


def build_category_dirs(output_dir: str) -> Dict[str, str]:
    return {
        "insufficient_capability": os.path.join(output_dir, "insufficient_capability"),
        "oracle_match": os.path.join(output_dir, "oracle_match"),
        "belief_failure": os.path.join(output_dir, "belief_failure"),
        "unstable": os.path.join(output_dir, "unstable"),
        "illegal": os.path.join(output_dir, "illegal"),
    }


def normalize_category(category: str) -> str:
    if category == "oracle_match":
        return "oracle_match"
    if category == "insufficient_belief":
        return "belief_failure"
    if category in CATEGORIES:
        return category
    return "insufficient_capability"


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(to_jsonable(v) for v in obj)
    return obj


def parse_agent_output(text: str) -> Optional[List[str]]:
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


def candidate_rules_map(candidate_names: List[str], heldout_set: str = "easy") -> Dict[str, Any]:
    rules: Dict[str, Any] = {}
    rules.update(EXTENDED_BENCHMARK_RULES)
    rules.update(resolve_heldout_rules(heldout_set))
    return {name: rules[name] for name in candidate_names}


def oracle_answer(oracle: str, triple: Triple) -> str:
    return "YES" if get_rule(oracle).validate(triple) else "NO"


def compute_golden_hypotheses(rules: Dict[str, Any], evidence: List[Tuple[Triple, str]]) -> Set[str]:
    survivors: Set[str] = set()
    for name, rule in rules.items():
        if all(("YES" if rule.validate(triple) else "NO") == label for triple, label in evidence):
            survivors.add(name)
    return survivors


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
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        triple = _random_triple(rng)
        if any(not rules[name].validate(triple) for name in required_true_rules):
            continue
        result = "YES" if rules[oracle].validate(triple) else "NO"
        survivors = compute_golden_hypotheses(rules, active_evidence + [(triple, result)])
        if survivors == target_survivors:
            return triple
    return None


def triple_text(triple: Triple) -> str:
    return f"({triple[0]}, {triple[1]}, {triple[2]})"



def parse_noise_measure_triple(text: str) -> Optional[Triple]:
    if not isinstance(text, str) or not text:
        return None
    stripped = _THINK_TAG_RE.sub("", text)
    # Primary: strict <measure> tag
    matches = _MEASURE_TAG_RE.findall(stripped)
    if not matches:
        # Fallback: fuzzy tag matching for common <measure> typos
        matches = _MEASURE_TAG_FUZZY_RE.findall(stripped)
    if not matches:
        return None
    raw = matches[-1].strip()
    # Primary: strict bracket content [a, b, c]
    parsed = _TRIPLE_CONTENT_RE.match(raw)
    if parsed:
        return (int(parsed.group(1)), int(parsed.group(2)), int(parsed.group(3)))
    # Fallback: accept (), {}, floats-as-ints, semicolons / Chinese comma
    parsed = _TRIPLE_CONTENT_FUZZY_RE.match(raw)
    if parsed:
        try:
            return (int(float(parsed.group(1))), int(float(parsed.group(2))), int(float(parsed.group(3))))
        except (ValueError, OverflowError):
            return None
    return None


def parse_noise_query_triple(text: str) -> Optional[Triple]:
    # Backward-compatible alias retained for older call sites.
    return parse_noise_measure_triple(text)


def parse_noise_terminate(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    stripped = _THINK_TAG_RE.sub("", text)
    return bool(_TERMINATE_TAG_RE.search(stripped))


def _noise_host_comment(
    answer_yes: bool,
    turn: int,
    *,
    is_discriminative: bool = True,
) -> str:
    """Return an intentionally misleading host comment.

    When the query is discriminative among current survivors (useful query),
    return a negative comment to mislead the model.
    When the query is non-discriminative (already-eliminated / unanimous),
    return a positive comment to mislead the model.
    """
    _ = turn
    positive = "Host comment: Great question. You are moving in the right direction."
    negative = "Host comment: This line of checking is probably not useful."
    # Inverted: praise useless questions, discourage useful ones
    return negative if is_discriminative else positive


def _evaluate_noise_query(
    *,
    oracle: str,
    query_triple: Optional[Triple],
    turn: int,
    rules: Optional[Dict[str, Any]] = None,
    gt_survivors: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    if query_triple is None:
        answer_yes = False
        query_valid = False
        is_discriminative = False
    else:
        answer_yes = oracle_answer(oracle, query_triple) == "YES"
        query_valid = True
        # A triple is discriminative if the surviving rules disagree on its answer.
        # (Some survivors predict YES, some predict NO.)
        if rules and gt_survivors:
            survivor_answers = {
                bool(rules[name].validate(query_triple))
                for name in gt_survivors
                if name in rules
            }
            # Discriminative iff not all survivors agree
            is_discriminative = len(survivor_answers) > 1
        else:
            is_discriminative = True  # default: treat as discriminative when unknown
    return {
        "turn": turn,
        "query_valid": query_valid,
        "query_triple": list(query_triple) if query_triple is not None else None,
        "answer_yes": bool(answer_yes),
        "answer": "yes" if answer_yes else "no",
        "is_discriminative": is_discriminative,
        "host_comment": _noise_host_comment(bool(answer_yes), turn, is_discriminative=is_discriminative),
    }


def _noise_survivors_from_evidence(
    *,
    rules: Dict[str, Any],
    evidence: List[Tuple[Triple, bool]],
) -> Set[str]:
    survivors = set(rules.keys())
    for triple, answer_yes in evidence:
        compatible = {
            name
            for name, rule in rules.items()
            if bool(rule.validate(triple)) == bool(answer_yes)
        }
        survivors &= compatible
    return survivors


def build_noise_system_prompt(candidate_names: List[str]) -> str:
    candidates_block = format_candidate_rule_text(candidate_names)
    return (
        "You are diagnosing a hidden rule from a fixed set of candidate rules.\n\n"
        "## Background\n"
        "Each turn the environment tests one triple of integers that you choose.\n"
        "The environment tells you whether the hidden rule returns YES or NO for that triple.\n"
        "Use the answers to eliminate candidates until only one rule remains, then declare it.\n\n"
        "## Required output format\n"
        "Every reply MUST contain BOTH of the following tags (in this order):\n\n"
        "  1. <hypothesis>RULE_ID_1, RULE_ID_2, ...</hypothesis>\n"
        "     List every rule ID that is still consistent with all evidence so far.\n"
        "     Start with all rule IDs and remove each one as evidence rules it out.\n"
        "     Do NOT add or invent rule IDs beyond the listed candidates.\n\n"
        "  2. ONE of the following (choose based on how many candidates remain):\n"
        "     a) Still narrowing down — output a measurement query:\n"
        "           <measure>[a, b, c]</measure>\n"
        "        where a, b, c are integers.\n"
        "        Choose a triple whose YES/NO answer will eliminate at least one remaining candidate.\n"
        "     b) Ready to commit — exactly one candidate remains (or you are certain):\n"
        "           <terminate>final</terminate>\n"
        "        In this case <hypothesis> must contain exactly ONE rule ID.\n\n"
        "## Concrete output examples\n"
        "Example — still narrowing down (3 candidates remain):\n"
        "  <hypothesis>rule_a, rule_b, rule_c</hypothesis>\n"
        "  <measure>[1, -2, 3]</measure>\n\n"
        "Example — ready to commit (1 candidate remains):\n"
        "  <hypothesis>rule_a</hypothesis>\n"
        "  <terminate>final</terminate>\n\n"
        "## Turn-message notes\n"
        "- Each environment reply contains: Turn N, one verification line, rule predictions, and optionally one host-comment line.\n"
        "- The verification line reads: Environment verification: `(a, b, c)` -> YES  (or NO).\n"
        "- Below the verification line, the environment shows each candidate rule's prediction for your triple:\n"
        "    Rule predictions for this triple:\n"
        "      - rule_id_1 → YES\n"
        "      - rule_id_2 → NO\n"
        "      ...\n"
        "  A rule whose prediction matches the verified answer is still consistent.\n"
        "  A rule whose prediction differs from the verified answer is eliminated.\n"
        "- If your <measure> triple is malformed, no measurement is recorded for that turn.\n"
        "  In that case re-read the instructions and output a correctly formatted triple next turn.\n\n"
        "## Candidate rule IDs\n"
        f"{candidates_block}"
    )


def build_noise_start_message(*, remaining_turns: int, add_host_comment: bool = True) -> str:
    _ = remaining_turns
    if add_host_comment:
        return "Turn 0:\n\n- Session start.\n- Host comment: Follow the task protocol in the system prompt."
    else:
        return "Turn 0:\n\n- Session start."


def build_noise_feedback_message(
    *,
    turn: int,
    answer: str,
    host_comment: str,
    remaining_turns: int,
    query_triple: Optional[Triple],
    query_valid: bool,
    add_host_comment: bool = True,
    format_error: bool = False,
    candidate_names: Optional[List[str]] = None,
) -> str:
    query_text = triple_text(query_triple) if query_triple is not None else "(invalid)"
    if query_valid:
        answer_text = answer.upper()
        verification = f"- Environment verification: `{query_text}` -> {answer_text}"

        predictions_block = ""
        if candidate_names and query_triple is not None:
            predictions_block = (
                "\n\nRule predictions for this triple:\n"
                + _format_rule_predictions(candidate_names, query_triple)
            )

        if add_host_comment:
            return (
                f"Turn {turn}:\n\n"
                f"{verification}"
                f"{predictions_block}\n"
                f"- {host_comment}"
            )
        else:
            return (
                f"Turn {turn}:\n\n"
                f"{verification}"
                f"{predictions_block}"
            )
    _ = answer
    _ = remaining_turns
    _ = format_error
    return (
        f"Turn {turn}:\n\n"
        "- Environment prompt: This question is invalid. Please rephrase your query."
    )
def format_candidate_rule_text(candidate_names: List[str]) -> str:
    return "\n".join(
        f'- "{name}": {get_rule(name).description}'
        for name in candidate_names
    )


def build_system_prompt(candidate_names: List[str], model_type: str = "local") -> str:
    candidates_block = format_candidate_rule_text(candidate_names)
    return (
        "You are a cautious scientist tracking candidate rules in a rule-guessing game.\n\n"
        "Each turn, the host provides:\n"
        "1. A triple and its currently recorded result: YES or NO.\n"
        "2. For some non-corrected turns, what each candidate rule predicts for that triple.\n"
        "If rule predictions are omitted, use the rule definitions and all active evidence directly.\n\n"
        "Sometimes the host may correct an earlier recorded result.\n"
        "When that happens, the old record becomes inactive and the corrected one replaces it.\n"
        "Do not revise old evidence unless the host explicitly issues a correction.\n\n"
        "Your job is to maintain the set of rule IDs that are still consistent with all currently active recorded evidence.\n"
        "A rule is contradicted if its prediction differs from the currently recorded result for any active evidence item.\n\n"
        "For later turns, use set intersection:\n"
        "- Previous hypothesis: rule_a, rule_b; Current matching rule IDs: rule_b, rule_c; next output is <hypothesis>rule_b</hypothesis>.\n"
        "- Previous hypothesis: rule_a, rule_b, rule_c; Current matching rule IDs: rule_c, rule_d; next output is <hypothesis>rule_c</hypothesis>.\n"
        "- Never add a rule ID only because it matches the current triple; it must also be in the previous hypothesis.\n\n"
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
    *,
    prediction_overrides: Optional[Dict[str, bool]] = None,
    result: Optional[str] = None,
    annotate: bool = False,
) -> str:
    lines = []
    expected_yes = str(result).upper() == "YES" if result is not None else None
    for name in candidate_names:
        pred_yes = (
            bool(prediction_overrides[name])
            if prediction_overrides is not None and name in prediction_overrides
            else get_rule(name).validate(triple)
        )
        pred = "YES" if pred_yes else "NO"
        suffix = ""
        if annotate and expected_yes is not None:
            suffix = " (consistent)" if pred_yes == expected_yes else " (CONTRADICTS evidence → eliminated)"
        lines.append(f"  - {name} → {pred}{suffix}")
    return "\n".join(lines)


def _matching_rule_names(
    candidate_names: List[str],
    triple: Triple,
    result: str,
) -> List[str]:
    expected = str(result).upper() == "YES"
    return [
        name
        for name in candidate_names
        if bool(get_rule(name).validate(triple)) == expected
    ]


def _format_name_list(names: List[str]) -> str:
    return ", ".join(names) if names else "none"


def _append_prefix_update_hint(prompt: str, golden: Set[str]) -> str:
    hint = (
        f"Evidence-only updated hypothesis: {_format_name_list(sorted(golden))}\n"
        "For this calibration turn, output exactly the evidence-only updated hypothesis."
    )
    return f"{prompt}\n\n{hint}"


def build_evidence_message(
    triple: Triple,
    result: str,
    turn: int,
    *,
    candidate_names: Optional[List[str]] = None,
    include_rule_predictions: bool = True,
    prediction_overrides: Optional[Dict[str, bool]] = None,
    previous_hypotheses: Optional[List[str]] = None,
    annotate_rule_predictions: bool = False,
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
            f"{_format_rule_predictions(candidate_names, triple, prediction_overrides=prediction_overrides, result=result, annotate=annotate_rule_predictions)}\n\n"
        )
        if turn > 0 and previous_hypotheses is not None:
            matching = _matching_rule_names(candidate_names, triple, result)
            body += (
                f"Previous hypothesis: {_format_name_list(previous_hypotheses)}\n"
                f"Current matching rule IDs: {_format_name_list(matching)}\n"
                "Update rule: keep only rule IDs that are in BOTH lists above. Do not add new rule IDs.\n\n"
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
    prediction_overrides_by_item: Optional[List[Optional[Dict[str, bool]]]] = None,
) -> str:
    lines = [_turn_prefix(turn) + f"**Turn {turn} evidence:**"]
    for idx, (triple, result) in enumerate(batch, 1):
        lines.append(f"{idx}. Triple {triple_text(triple)}: **{result}**")
        if include_rule_predictions:
            if candidate_names is None:
                raise ValueError("candidate_names is required when include_rule_predictions=True")
            lines.append("   Rule predictions for this triple:")
            prediction_overrides = (
                prediction_overrides_by_item[idx - 1]
                if prediction_overrides_by_item is not None
                else None
            )
            for prediction_line in _format_rule_predictions(
                candidate_names,
                triple,
                prediction_overrides=prediction_overrides,
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


def build_multi_correction_message(
    retracted_items: List[Tuple[int, Triple, str]],
    turn: int,
) -> str:
    """Build a correction prompt that simultaneously retracts multiple past turns.

    retracted_items: list of (display_turn, triple, recorded_result) for each
    evidence being invalidated. No replacement evidence is added.
    """
    lines = [f"**Turn {turn} evidence:**"]
    unique_turns: List[int] = []
    seen = set()
    for rt, _, _ in retracted_items:
        if rt not in seen:
            unique_turns.append(rt)
            seen.add(rt)
    if len(retracted_items) > 1:
        if len(unique_turns) == 1:
            header_turn = unique_turns[0]
            lines.append(f"CORRECTION for Turn {header_turn} (all {len(retracted_items)} items):")
        else:
            turn_list = ", ".join(str(rt) for rt in unique_turns)
            lines.append(f"CORRECTION for Turns {turn_list}:")
        lines.append(
            "The recorded results below were all wrong. "
            "Please disregard them entirely (no replacement is provided)."
        )
    else:
        rt, _tr, _rec = retracted_items[0]
        lines.append(f"CORRECTION for Turn {rt}:")
        lines.append(
            "The recorded result was wrong. "
            "Please disregard it (no replacement is provided)."
        )
    for rt, tr, rec in retracted_items:
        lines.append(
            f"- Turn {rt}: triple {triple_text(tuple(tr))} (recorded **{rec}**) -> invalidated"
        )
    lines.append("")
    lines.append("Please update your hypotheses using all currently active evidence.")
    return "\n".join(lines)


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
    display_retract_turns: Optional[List[int]] = None,
    previous_hypotheses: Optional[List[str]] = None,
    annotate_rule_predictions: bool = False,
) -> Tuple[str, Triple, str, str]:
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
                previous_hypotheses=previous_hypotheses,
                annotate_rule_predictions=annotate_rule_predictions,
            ),
            triple,
            result,
            "evidence",
        )

    if event_type == "retraction":
        raw_turns = event.get("retract_turns")
        if raw_turns is not None:
            retract_turns = [int(rt) for rt in raw_turns]
        else:
            retract_turns = [int(event["retract_turn"])]

        if display_retract_turns is not None:
            display_turns = [int(t) for t in display_retract_turns]
        elif display_retract_turn is not None and len(retract_turns) == 1:
            display_turns = [int(display_retract_turn)]
        else:
            display_turns = list(retract_turns)

        for rt in retract_turns:
            if rt not in event_to_active:
                raise RuntimeError(f"Invalid retract_turn={rt}; missing active mapping")

        pairs = sorted(
            ((rt, event_to_active[rt]) for rt in retract_turns),
            key=lambda pair: -pair[1],
        )
        removed: List[Tuple[int, Triple, str]] = []
        for rt, idx in pairs:
            removed_triple, removed_result = active_evidence.pop(idx)
            removed.append((rt, removed_triple, removed_result))
            del event_to_active[rt]
            for k, v in list(event_to_active.items()):
                if v > idx:
                    event_to_active[k] = v - 1

        # Pure multi-retract (or single retract without replacement) emits no new evidence.
        has_replacement = ("new_triple" in event) and len(retract_turns) == 1
        if has_replacement:
            single_rt = retract_turns[0]
            original_triple = next(tr for (rt, tr, _r) in removed if rt == single_rt)
            triple = original_triple if bool(event.get("use_retracted_triple", False)) else tuple(event["new_triple"])  # type: ignore[assignment]
            result = event.get("new_result") or oracle_answer(oracle, triple)
            event_to_active[event_idx] = len(active_evidence)
            active_evidence.append((triple, result))
            return (
                build_correction_message(triple, result, display_turns[0], prompt_turn),
                triple,
                result,
                "retraction",
            )

        # Build display items in original chronological order for readability.
        rt_to_display = {rt: dt for rt, dt in zip(retract_turns, display_turns)}
        chronological = sorted(removed, key=lambda item: item[0])
        retracted_items = [
            (rt_to_display.get(rt, rt), tr, rec) for (rt, tr, rec) in chronological
        ]
        # Surface a representative triple/result for downstream logging.
        rep_triple = retracted_items[0][1]
        rep_result = retracted_items[0][2]
        return (
            build_multi_correction_message(retracted_items, prompt_turn),
            tuple(rep_triple),
            rep_result,
            "retraction",
        )

    raise ValueError(f"Unknown event type: {event_type}")


def extract_round_matches_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rounds: List[Dict[str, Any]] = []
    for message in result.get("messages", []):
        if message.get("role") != "assistant":
            continue
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
        events.append(
            {
                "turn": turn,
                "extra_rules": sorted(model_set - {oracle}),
                "missing_rules": sorted(golden_set - model_set),
                "model_hypotheses": sorted(model_set),
                "golden_hypotheses": sorted(golden_set),
            }
        )
    return events


def classify_round_match_result(result: Dict[str, Any]) -> str:
    rounds = extract_round_matches_from_result(result)
    mode = result.get("mode")
    if mode == "noise":
        if result.get("termination_reason") == "format_error_dropped":
            return "illegal"
        turns = list(result.get("turns", []))
        if not turns:
            return "illegal"
        finalize_turn = next((turn for turn in turns if turn.get("action") == "finalize_fault"), None)
        if finalize_turn is None:
            if any(turn.get("action") in {"ask_measure", "ask_query"} for turn in turns):
                return "belief_failure"
            return "illegal"
        final_guess = str(finalize_turn.get("final_guess", ""))
        oracle = str(result.get("oracle", ""))
        if final_guess == oracle:
            return "oracle_match"
        return "belief_failure"


    if mode == "failed_update":
        sequence = result.get("challenge_sequence") or {}
        challenge_turns = [
            int(t)
            for t in sequence.get("challenge_turns", [])
            if isinstance(t, int)
        ]
        if not challenge_turns:
            challenge_turns = [
                int(r["turn"])
                for r in rounds
                if isinstance(r.get("turn"), int) and r.get("phase") == "retraction"
            ]
        if not challenge_turns:
            return "insufficient_capability"
        challenge_start = min(challenge_turns)
        pre_challenge_rounds = [
            r for r in rounds
            if isinstance(r.get("turn"), int) and r["turn"] < challenge_start
        ]
        challenge_rounds = [
            r for r in rounds
            if isinstance(r.get("turn"), int) and r["turn"] in challenge_turns
        ]
        if not pre_challenge_rounds or not all(r["match"] for r in pre_challenge_rounds):
            return "insufficient_capability"
        if not challenge_rounds:
            return "insufficient_capability"
        return "oracle_match" if all(r["match"] for r in challenge_rounds) else "belief_failure"

    if mode == "failed_isolation":
        if result.get("termination_reason") == "format_error_dropped":
            return "illegal"
        messages = list(result.get("messages", []))
        if not messages:
            return "illegal"
        if any(
            message.get("role") == "assistant"
            and message.get("parse_ok") is False
            for message in messages
            ):
            return "illegal"

        strict_prefix_turns = int(result.get("strict_failed_isolation_prefix_turns") or 0)
        if strict_prefix_turns > 0:
            if len(rounds) <= strict_prefix_turns:
                return "insufficient_capability"
            prefix_rounds = rounds[:strict_prefix_turns]
            post_rounds = rounds[strict_prefix_turns:]
            if not prefix_rounds or not all(r["match"] for r in prefix_rounds):
                return "insufficient_capability"
            if not post_rounds:
                return "insufficient_capability"
            return "oracle_match" if all(r["match"] for r in post_rounds) else "belief_failure"

        final_hyp = set(result.get("final_model_hypotheses") or [])
        oracle = str(result.get("oracle", ""))
        if final_hyp == {oracle}:
            return "oracle_match"
        return "belief_failure"

    if mode == "failed_stay":
        post_start = _post_interference_start_turn(result)
        if post_start is None:
            return "insufficient_capability"
        pre_post_rounds = [
            r for r in rounds
            if isinstance(r.get("turn"), int) and r["turn"] < post_start
        ]
        post_rounds = [
            r for r in rounds
            if isinstance(r.get("turn"), int) and r["turn"] >= post_start
        ]
        if not pre_post_rounds or not all(r["match"] for r in pre_post_rounds):
            return "insufficient_capability"
        if not post_rounds:
            return "insufficient_capability"
        return "oracle_match" if all(r["match"] for r in post_rounds) else "belief_failure"

    return "insufficient_capability"


def _init_belief_stats_session(
    *,
    row: Dict[str, Any],
    model: str,
    mode: str,
    candidate_names: List[str],
    heldout_set: str,
    include_rule_predictions: bool = True,
    perturb_oracle_rule_prediction_in_post: bool = False,
    hide_rule_predictions_in_failed_stay_post: bool = False,
    add_failed_isolation_comment: bool = True,
    model_type: str = "local",
    strict_failed_isolation_prefix_turns: int = 0,
    failed_isolation_comment_start_turn: int = 0,
    preserve_failed_isolation_turn_message: bool = False,
    prefix_update_hint_turns: int = 0,
    annotate_rule_predictions: bool = False,
) -> Dict[str, Any]:
    oracle = row["oracle"]
    sequence = json.loads(json.dumps(to_jsonable(row["sequence"])))
    rules = candidate_rules_map(candidate_names, heldout_set)
    rng = random.Random(sum(ord(ch) for ch in f"{oracle}:{mode}:{len(candidate_names)}"))
    system_prompt = build_system_prompt(candidate_names, model_type=model_type)

    gt_survivors_by_turn: Dict[int, Set[str]] = {}
    for step in sequence.get("ground_truth", []):
        turn = step.get("turn")
        survivors = step.get("survivors")
        if isinstance(turn, int) and isinstance(survivors, list):
            gt_survivors_by_turn[turn] = set(survivors)

    batch_start_to_end: Dict[int, int] = {}
    for pair in sequence.get("evidence_batch_ranges", []):
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        start, end = int(pair[0]), int(pair[1])
        if start <= end:
            batch_start_to_end[start] = end

    failed_isolation_seed_str = f"failed_isolation:{oracle}:{mode}:{len(candidate_names)}:{row.get('seed', 0)}:{row.get('run_idx', 0)}:{row.get('repeat_index', 0)}"
    failed_isolation_rng = random.Random(sum(ord(ch) for ch in failed_isolation_seed_str))
    return {
        "row": row,
        "mode": mode,
        "model": model,
        "add_failed_isolation_comment": bool(add_failed_isolation_comment),
        "strict_failed_isolation_prefix_turns": int(strict_failed_isolation_prefix_turns),
        "failed_isolation_comment_start_turn": int(failed_isolation_comment_start_turn),
        "preserve_failed_isolation_turn_message": bool(preserve_failed_isolation_turn_message),
        "prefix_update_hint_turns": int(prefix_update_hint_turns),
        "annotate_rule_predictions": bool(annotate_rule_predictions),
        "failed_isolation_rng": failed_isolation_rng,
        "model_type": model_type,
        "oracle": oracle,
        "candidate_names": candidate_names,
        "heldout_set": heldout_set,
        "include_rule_predictions": include_rule_predictions,
        "perturb_oracle_rule_prediction_in_post": perturb_oracle_rule_prediction_in_post,
        "hide_rule_predictions_in_failed_stay_post": hide_rule_predictions_in_failed_stay_post,
        "rules": rules,
        "rng": rng,
        "sequence": sequence,
        "messages": [{"role": "system", "content": system_prompt}],
        "message_records": [{"role": "system", "content": system_prompt, "phase": "init"}],
        "active_evidence": [],
        "event_to_active": {},
        "golden": compute_golden_hypotheses(rules, []),
        "model_hyp": set(candidate_names),
        "turn_match_flags": [],
        "self_eliminated": set(),
        "failed_stay_rule_constraints_active": False,
        "failed_stay_required_rules": set(),
        "failed_stay_ever_excluded_by_golden": set(),
        "failed_stay_start_turn": None,
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


def _prepare_belief_stats_session_prompt(session: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    if session.get("done"):
        return None

    events = session["sequence"]["events"]
    i = session["i"]
    if i >= len(events):
        session["done"] = True
        return None

    rules = session["rules"]
    oracle = session["oracle"]
    active_evidence = session["active_evidence"]
    event_to_active = session["event_to_active"]
    prev_model = set(session["model_hyp"])
    mode = session["mode"]
    batch_start_to_end = session["batch_start_to_end"]
    failed_stay_interference_round_start = session["failed_stay_interference_round_start"]
    event_to_prompt_turn = session["event_to_prompt_turn"]
    prompt_turn = int(session["prompt_turn"])

    failed_stay_like = mode in ("failed_stay", "failed_isolation")

    def should_flip_oracle_prediction(turn_idx: int) -> bool:
        return (
            failed_stay_like
            and bool(session.get("perturb_oracle_rule_prediction_in_post"))
            and isinstance(failed_stay_interference_round_start, int)
            and turn_idx >= failed_stay_interference_round_start
        )

    def include_predictions_for_turn(turn_idx: int, event_type: str, final_turn: bool) -> bool:
        if (
            failed_stay_like
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
        if (
            mode == "failed_isolation"
            and int(session.get("prefix_update_hint_turns", 0) or 0) > 0
            and prompt_turn < int(session.get("prefix_update_hint_turns", 0) or 0)
        ):
            prompt = _append_prefix_update_hint(prompt, golden)
        if (
            mode == "failed_isolation"
            and session.get("add_failed_isolation_comment", True)
            and prompt_turn >= int(session.get("failed_isolation_comment_start_turn", 0) or 0)
        ):
            failed_isolation_line = build_failed_isolation_comment(
                rng=session["failed_isolation_rng"],
                candidate_names=session["candidate_names"],
                oracle=oracle,
                turn_idx=prompt_turn,
                model_type=session.get("model_type", "local"),
            )
            if failed_isolation_line:
                if session.get("preserve_failed_isolation_turn_message", False):
                    prompt = f"{prompt}\nHost comment: {failed_isolation_line}"
                else:
                    prompt = _inline_failed_isolation_comment(
                        prompt,
                        triple=batch_info[0][0],
                        result=batch_info[0][1],
                        failed_isolation_comment=failed_isolation_line,
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
    display_retract_turns = None
    if event.get("type") == "retraction":
        raw_turns = event.get("retract_turns")
        if raw_turns is not None:
            display_retract_turns = [
                event_to_prompt_turn.get(int(rt), int(rt)) for rt in raw_turns
            ]
        else:
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
        display_retract_turns=display_retract_turns,
        previous_hypotheses=sorted(prev_model),
        annotate_rule_predictions=bool(session.get("annotate_rule_predictions", False)),
    )
    event_to_prompt_turn[i] = prompt_turn

    golden = compute_golden_hypotheses(rules, active_evidence)
    session["golden"] = golden
    if (
        mode == "failed_isolation"
        and int(session.get("prefix_update_hint_turns", 0) or 0) > 0
        and prompt_turn < int(session.get("prefix_update_hint_turns", 0) or 0)
    ):
        prompt = _append_prefix_update_hint(prompt, golden)
    if (
        mode == "failed_isolation"
        and session.get("add_failed_isolation_comment", True)
        and prompt_turn >= int(session.get("failed_isolation_comment_start_turn", 0) or 0)
    ):
        failed_isolation_line = build_failed_isolation_comment(
                rng=session["failed_isolation_rng"],
                candidate_names=session["candidate_names"],
                oracle=oracle,
                turn_idx=prompt_turn,
                model_type=session.get("model_type", "local"),
            )
        if failed_isolation_line:
            if session.get("preserve_failed_isolation_turn_message", False):
                prompt = f"{prompt}\nHost comment: {failed_isolation_line}"
            else:
                prompt = _inline_failed_isolation_comment(
                    prompt,
                    triple=triple,
                    result=result,
                    failed_isolation_comment=failed_isolation_line,
                )
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


def _apply_belief_stats_session_response(session: Dict[str, Any], response: str) -> None:
    pending = session["pending"]
    if pending is None:
        raise RuntimeError("Missing pending prompt for belief_stats session")

    candidate_names = session["candidate_names"]
    parsed = parse_agent_output(response)
    model_hyp = normalize_hypotheses(parsed, candidate_names)
    prev_model = set(pending["prev_model"])
    golden = set(pending["golden"])
    removed = sorted(prev_model - model_hyp)
    session["self_eliminated"].update(removed)
    readded = sorted(model_hyp & session["self_eliminated"])
    match = model_hyp == golden
    session["turn_match_flags"].append(match)

    base_record = {
        "role": "assistant",
        "content": response,
        "turn": pending["turn"],
        "prompt_turn": pending.get("prompt_turn"),
        "phase": pending["phase"],
        "golden_hypotheses": sorted(golden),
        "model_hypotheses": sorted(model_hyp),
        "model_removed_this_turn": removed,
        "readded_self_eliminated_rules": readded,
        "model_matches_golden": match,
        "parse_ok": parsed is not None,
    }
    if pending["phase"] == "evidence_batch":
        base_record["host_evidence_batch"] = pending["host_evidence_batch"]
    else:
        base_record["user_prompt"] = pending["prompt"]
        base_record["host_triple"] = list(pending["triple"])
        base_record["host_result"] = pending["result"]
    session["message_records"].append(base_record)

    session["messages"].append({"role": "assistant", "content": response})
    session["model_hyp"] = model_hyp
    session["golden"] = golden
    session["i"] = pending["next_i"]
    session["prompt_turn"] = int(session.get("prompt_turn", 0)) + 1
    session["pending"] = None
    if session["i"] >= len(session["sequence"]["events"]):
        session["done"] = True


def _finalize_belief_stats_session_result(session: Dict[str, Any]) -> Dict[str, Any]:
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
        "strict_failed_isolation_prefix_turns": int(session.get("strict_failed_isolation_prefix_turns", 0) or 0),
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

    if session["mode"] in ("failed_stay", "failed_isolation"):
        rounds = extract_round_matches_from_result(result)
        post_events = collect_failed_stay_post_events_from_rounds(
            rounds,
            oracle=session["oracle"],
            post_start_turn=session["failed_stay_interference_round_start"],
        )
        final_model_set = set(result["final_model_hypotheses"])
        retained_readded_rules = sorted({rule for event in post_events for rule in event["extra_rules"]})
        result.update(
            {
                "failed_stay_detected": bool(post_events),
                "failed_stay_events": post_events,
                "failed_stay_judgement_turn": session["failed_stay_interference_round_start"],
                "failed_stay_required_rules": sorted(session["failed_stay_required_rules"]),
                "failed_stay_start_turn": session["failed_stay_start_turn"],
                "failed_stay_interference_round_start": session["failed_stay_interference_round_start"],
                "failed_stay_post_interference_events": post_events,
                "failed_stay_post_extra_rules": retained_readded_rules,
                "failed_stay_convergence_match": next(
                    (
                        bool(m.get("model_matches_golden"))
                        for m in session["message_records"]
                        if m.get("role") == "assistant"
                        and m.get("turn") == session["failed_stay_convergence_turn"]
                    ),
                    None,
                ),
                "failed_stay_final_wrong_rules": sorted(final_model_set - {session["oracle"]}),
                "failed_stay_final_retained_readded_rules": retained_readded_rules,
                "failed_stay_final_is_singleton_oracle": final_model_set == {session["oracle"]},
            }
        )

    if session["mode"] == "failed_update":
        correction_checks: List[Dict[str, Any]] = []
        for record in session["message_records"]:
            if record.get("role") != "assistant" or record.get("phase") != "retraction":
                continue
            turn = record.get("turn")
            if turn is None:
                continue
            pre_rec = None
            for candidate in session["message_records"]:
                if candidate.get("role") != "assistant":
                    continue
                if candidate.get("turn") is not None and candidate.get("turn") < turn:
                    pre_rec = candidate
            if pre_rec is None:
                continue
            pre = set(pre_rec.get("model_hypotheses") or [])
            post = set(record.get("model_hypotheses") or [])
            expected = set(record.get("golden_hypotheses") or [])
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



def _init_noise_session(
    *,
    row: Dict[str, Any],
    model: str,
    candidate_names: List[str],
    heldout_set: str,
    add_host_comment: bool = True,
) -> Dict[str, Any]:
    oracle = row["oracle"]
    sequence = json.loads(json.dumps(to_jsonable(row["sequence"])))
    max_turns = int(
        sequence.get("max_turns")
        or sequence.get("total_turns")
        or len(sequence.get("events", []))
        or 6
    )
    system_prompt = build_noise_system_prompt(candidate_names)

    return {
        "row": row,
        "mode": "noise",
        "model": model,
        "oracle": oracle,
        "candidate_names": candidate_names,
        "rules": candidate_rules_map(candidate_names, heldout_set),
        "sequence": sequence,
        "messages": [{"role": "system", "content": system_prompt}],
        "message_records": [{"role": "system", "content": system_prompt, "phase": "init"}],
        "turns": [],
        "evidence": [],
        "turn_idx": 0,
        "max_turns": max_turns,
        "pending_user_prompt": build_noise_start_message(remaining_turns=max_turns, add_host_comment=add_host_comment),
        "noise_success": False,
        "termination_reason": "max_turns_exceeded",
        "done": False,
    }


def _prepare_noise_session_prompt(session: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    if session.get("done"):
        return None

    turn_idx = int(session.get("turn_idx", 0))
    max_turns = int(session.get("max_turns", 0))
    if turn_idx >= max_turns:
        session["done"] = True
        return None

    prompt = str(session.get("pending_user_prompt") or "")
    session["messages"].append({"role": "user", "content": prompt})
    session["message_records"].append(
        {
            "role": "user",
            "content": prompt,
            "turn": turn_idx,
            "prompt_turn": turn_idx,
            "phase": "noise_prompt",
            "user_prompt": prompt,
        }
    )
    return session["messages"]


def _apply_noise_session_response(session: Dict[str, Any], response: str, add_host_comment: bool = True) -> None:
    turn_idx = int(session.get("turn_idx", 0))
    max_turns = int(session.get("max_turns", 0))
    oracle = str(session.get("oracle", ""))
    candidate_names = list(session.get("candidate_names", []))
    rules = dict(session.get("rules", {}))
    evidence = list(session.get("evidence", []))

    gt_survivors = _noise_survivors_from_evidence(rules=rules, evidence=evidence)
    hypotheses_raw = parse_agent_output(response)
    hypotheses = [name for name in (hypotheses_raw or []) if name in set(candidate_names)]
    measure_triple = parse_noise_measure_triple(response)
    is_terminate = parse_noise_terminate(response)

    agent_set = set(hypotheses)
    belief_metrics = {
        "oracle_in_hypotheses": oracle in agent_set,
        "exact_match": agent_set == gt_survivors,
        "false_retention": sorted(agent_set - gt_survivors),
        "false_elimination": sorted(gt_survivors - agent_set),
        "agent_set_size": len(agent_set),
        "gt_set_size": len(gt_survivors),
    }

    turn_record: Dict[str, Any] = {
        "turn": turn_idx,
        "agent_response": response,
        "hypotheses": hypotheses,
        "gt_survivors": sorted(gt_survivors),
        "belief_metrics": belief_metrics,
        "parse_error": hypotheses_raw is None,
    }

    if is_terminate:
        final_guess = hypotheses[0] if hypotheses else None
        success = len(hypotheses) == 1 and final_guess == oracle
        turn_record.update(
            {
                "action": "finalize_fault",
                "final_guess": final_guess,
                "finalize_success": success,
            }
        )
        session["noise_success"] = success
        session["termination_reason"] = "finalized"
        session["done"] = True
    elif measure_triple is not None:
        feedback = _evaluate_noise_query(
            oracle=oracle,
            query_triple=measure_triple,
            turn=turn_idx,
            rules=rules,
            gt_survivors=gt_survivors,
        )
        turn_record.update(
            {
                "action": "ask_measure",
                "query_triple": feedback["query_triple"],
                "query_valid": feedback["query_valid"],
                "env_answer": feedback["answer"],
                "host_comment": feedback["host_comment"],
            }
        )
        if feedback["query_valid"]:
            evidence.append((measure_triple, bool(feedback["answer_yes"])))
            session["evidence"] = evidence
        session["pending_user_prompt"] = build_noise_feedback_message(
            turn=turn_idx + 1,
            answer=str(feedback["answer"]),
            host_comment=str(feedback["host_comment"]),
            remaining_turns=max(0, max_turns - (turn_idx + 1)),
            query_triple=measure_triple,
            query_valid=bool(feedback["query_valid"]),
            add_host_comment=add_host_comment,
            candidate_names=candidate_names,
        )
    else:
        # Format error even after relaxed parsing: drop this session without
        # sending an invalid-action prompt back to the model.
        turn_record.update(
            {
                "action": "format_error_dropped",
                "parse_error": True,
                "query_triple": None,
                "query_valid": False,
            }
        )
        session["done"] = True
        session["termination_reason"] = "format_error_dropped"

    session["messages"].append({"role": "assistant", "content": response})
    session["message_records"].append(
        {
            "role": "assistant",
            "content": response,
            "turn": turn_idx,
            "prompt_turn": turn_idx,
            "phase": turn_record["action"],
            "user_prompt": session["message_records"][-1].get("content", ""),
            "host_triple": turn_record.get("query_triple"),
            "host_result": turn_record.get("env_answer"),
            "golden_hypotheses": sorted(turn_record.get("gt_survivors", [])),
            "model_hypotheses": hypotheses,
            "model_matches_golden": turn_record.get("belief_metrics", {}).get("exact_match", False),
            "parse_ok": not bool(turn_record.get("parse_error", False)),
        }
    )
    session["turns"].append(turn_record)

    session["turn_idx"] = turn_idx + 1
    if not session.get("done") and int(session["turn_idx"]) >= max_turns:
        session["done"] = True
        session["termination_reason"] = "max_turns_exceeded"


def _finalize_noise_session_result(session: Dict[str, Any]) -> Dict[str, Any]:
    row = session["row"]
    result: Dict[str, Any] = {
        "mode": "noise",
        "oracle": session["oracle"],
        "oracle_description": get_rule(session["oracle"]).description,
        "candidate_rules": session["candidate_names"],
        "challenge_sequence": session["sequence"],
        "messages": session["message_records"],
        "turns": session["turns"],
        "conversation": session["messages"],
        "n_turns_played": len(session["turns"]),
        "noise_success": bool(session.get("noise_success")),
        "turn_match_rate": 1.0 if bool(session.get("noise_success")) else 0.0,
        "termination_reason": session.get("termination_reason", "max_turns_exceeded"),
    }
    finalize_turn = next(
        (turn for turn in session["turns"] if turn.get("action") == "finalize_fault"),
        None,
    )
    if finalize_turn is not None:
        result["final_model_hypotheses"] = [finalize_turn.get("final_guess")]
        result["final_match"] = bool(finalize_turn.get("finalize_success"))
    else:
        result["final_model_hypotheses"] = []
        result["final_match"] = False

    result["experiment_id"] = row["repeat_experiment_id"]
    result["sample_id"] = row["sample_id"]
    result["model"] = session["model"]
    result["seed"] = row["seed"]
    result["run_idx"] = row["run_idx"]
    result["repeat_index"] = row["repeat_index"]
    result.pop("challenge_sequence", None)
    return to_jsonable(result)


def _host_turn_checkpoint_row(
    *,
    session: Dict[str, Any],
    turn_idx: int,
    max_turns: int,
    run_label: str,
) -> Dict[str, Any]:
    row_info = session.get("row", {})
    record = session["message_records"][-1]
    user_message = record.get("user_prompt", "")
    if not user_message:
        messages = session.get("messages", [])
        if len(messages) >= 2 and messages[-2].get("role") == "user":
            user_message = messages[-2].get("content", "")
    return {
        "run_label": run_label,
        "turn": turn_idx,
        "turn_number": turn_idx + 1,
        "max_turns": max_turns,
        "prompt_turn": record.get("prompt_turn"),
        "sample_id": row_info.get("sample_id"),
        "experiment_id": row_info.get("repeat_experiment_id"),
        "mode": session.get("mode"),
        "oracle": session.get("oracle"),
        "run_idx": row_info.get("run_idx"),
        "repeat_index": row_info.get("repeat_index"),
        "phase": record.get("phase"),
        "user_message": user_message,
        "turn_result": record,
        "host_triple": record.get("host_triple"),
        "host_result": record.get("host_result"),
        "host_evidence_batch": record.get("host_evidence_batch"),
        "golden_hypotheses": record.get("golden_hypotheses"),
        "model_hypotheses": record.get("model_hypotheses"),
        "model_matches_golden": record.get("model_matches_golden"),
        "parse_ok": record.get("parse_ok"),
        "assistant_response": record.get("content", ""),
    }


def _write_host_turn_checkpoint(
    *,
    checkpoint_dir: str,
    run_label: str,
    turn_idx: int,
    max_turns: int,
    sessions: List[Dict[str, Any]],
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"turn_{turn_idx + 1:02d}.jsonl")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for session in sessions:
            row = _host_turn_checkpoint_row(
                session=session,
                turn_idx=turn_idx,
                max_turns=max_turns,
                run_label=run_label,
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)
    print(
        f"[checkpoint] saved turn {turn_idx + 1}/{max_turns}: "
        f"{path} rows={len(sessions)}",
        flush=True,
    )


def _load_host_turn_checkpoint_rows(
    *,
    checkpoint_dir: str,
    turn_idx: int,
) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(checkpoint_dir, f"turn_{turn_idx + 1:02d}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint file: {path}")

    rows_by_experiment: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            experiment_id = row.get("experiment_id")
            if experiment_id is None:
                continue
            rows_by_experiment[str(experiment_id)] = row
    return rows_by_experiment


def _advance_belief_stats_session_to_prompt_turn(
    session: Dict[str, Any],
    *,
    last_completed_prompt_turn: int,
) -> None:
    if last_completed_prompt_turn < 0:
        return

    events = session["sequence"].get("events", [])
    oracle = session["oracle"]
    rules = session["rules"]
    active_evidence = session["active_evidence"]
    event_to_active = session["event_to_active"]
    batch_start_to_end = session["batch_start_to_end"]
    event_to_prompt_turn = session["event_to_prompt_turn"]

    i = 0
    prompt_turn = 0
    while i < len(events) and prompt_turn <= last_completed_prompt_turn:
        if (
            i in batch_start_to_end
            and all(
                events[k].get("type") == "evidence"
                for k in range(i, batch_start_to_end[i] + 1)
            )
        ):
            end_idx = batch_start_to_end[i]
            for k in range(i, end_idx + 1):
                event_to_prompt_turn[k] = prompt_turn
                apply_event_update(
                    oracle=oracle,
                    event_idx=k,
                    event=events[k],
                    active_evidence=active_evidence,
                    event_to_active=event_to_active,
                    include_rule_predictions=False,
                )
            i = end_idx + 1
        else:
            event = events[i]
            display_retract_turn = None
            display_retract_turns = None
            if event.get("type") == "retraction":
                raw_turns = event.get("retract_turns")
                if raw_turns is not None:
                    display_retract_turns = [
                        event_to_prompt_turn.get(int(rt), int(rt)) for rt in raw_turns
                    ]
                else:
                    retract_turn = int(event["retract_turn"])
                    display_retract_turn = event_to_prompt_turn.get(retract_turn, retract_turn)
            event_to_prompt_turn[i] = prompt_turn
            apply_event_update(
                oracle=oracle,
                event_idx=i,
                event=event,
                active_evidence=active_evidence,
                event_to_active=event_to_active,
                include_rule_predictions=False,
                display_turn=prompt_turn,
                display_retract_turn=display_retract_turn,
                display_retract_turns=display_retract_turns,
            )
            i += 1
        prompt_turn += 1

    session["i"] = i
    session["prompt_turn"] = prompt_turn
    session["golden"] = compute_golden_hypotheses(rules, active_evidence)
    session["pending"] = None
    session["done"] = i >= len(events)


def _resume_belief_stats_sessions_from_checkpoint(
    *,
    sessions: List[Dict[str, Any]],
    checkpoint_dir: Optional[str],
    resume_from_turn: int,
) -> int:
    if resume_from_turn <= 0:
        return 0
    if not checkpoint_dir:
        raise ValueError("resume_from_turn > 0 requires a checkpoint_dir")

    # If the first checkpoint file does not exist for this oracle (e.g. the run was
    # interrupted before it was reached), gracefully skip resume and start fresh.
    first_ckpt = os.path.join(checkpoint_dir, "turn_01.jsonl")
    if not os.path.exists(first_ckpt):
        print(
            f"[resume] no checkpoint found at {checkpoint_dir}, starting from turn 0",
            flush=True,
        )
        return 0

    for turn_idx in range(resume_from_turn):
        rows_by_experiment = _load_host_turn_checkpoint_rows(
            checkpoint_dir=checkpoint_dir,
            turn_idx=turn_idx,
        )
        for session in sessions:
            experiment_id = str(session.get("row", {}).get("repeat_experiment_id", ""))
            row = rows_by_experiment.get(experiment_id)
            if row is None:
                raise ValueError(
                    f"Missing checkpoint row for experiment_id={experiment_id} "
                    f"at turn={turn_idx}"
                )

            prompt_turn = int(row.get("prompt_turn", turn_idx))
            phase = str(row.get("phase", "evidence"))
            user_message = str(row.get("user_message", ""))
            assistant_response = str(row.get("assistant_response", ""))

            session["messages"].append({"role": "user", "content": user_message})
            session["messages"].append({"role": "assistant", "content": assistant_response})

            user_record: Dict[str, Any] = {
                "role": "user",
                "content": user_message,
                "turn": turn_idx,
                "prompt_turn": prompt_turn,
                "phase": phase,
                "user_prompt": user_message,
                "golden_hypotheses": row.get("golden_hypotheses"),
            }
            if row.get("host_evidence_batch") is not None:
                user_record["host_evidence_batch"] = row.get("host_evidence_batch")
            else:
                user_record["host_triple"] = row.get("host_triple")
                user_record["host_result"] = row.get("host_result")
            session["message_records"].append(user_record)

            turn_result = row.get("turn_result")
            if isinstance(turn_result, dict):
                assistant_record = json.loads(json.dumps(to_jsonable(turn_result)))
            else:
                assistant_record = {}
            assistant_record.setdefault("role", "assistant")
            assistant_record.setdefault("content", assistant_response)
            assistant_record.setdefault("turn", turn_idx)
            assistant_record.setdefault("prompt_turn", prompt_turn)
            assistant_record.setdefault("phase", phase)
            assistant_record.setdefault("golden_hypotheses", row.get("golden_hypotheses"))
            assistant_record.setdefault("model_hypotheses", row.get("model_hypotheses"))
            assistant_record.setdefault("model_matches_golden", row.get("model_matches_golden"))
            assistant_record.setdefault("parse_ok", row.get("parse_ok"))
            if row.get("host_evidence_batch") is not None:
                assistant_record.setdefault("host_evidence_batch", row.get("host_evidence_batch"))
            else:
                assistant_record.setdefault("host_triple", row.get("host_triple"))
                assistant_record.setdefault("host_result", row.get("host_result"))
                assistant_record.setdefault("user_prompt", user_message)
            session["message_records"].append(assistant_record)

            model_hypotheses = assistant_record.get("model_hypotheses")
            if isinstance(model_hypotheses, list):
                session["model_hyp"] = set(model_hypotheses)

            removed_rules = assistant_record.get("model_removed_this_turn")
            if isinstance(removed_rules, list):
                session["self_eliminated"].update(str(rule) for rule in removed_rules)

            session["turn_match_flags"].append(
                bool(assistant_record.get("model_matches_golden", False))
            )

    for session in sessions:
        _advance_belief_stats_session_to_prompt_turn(
            session,
            last_completed_prompt_turn=resume_from_turn - 1,
        )

        if len(session["message_records"]) >= 2:
            last_assistant = session["message_records"][-1]
            if isinstance(last_assistant, dict):
                golden_hypotheses = last_assistant.get("golden_hypotheses")
                if isinstance(golden_hypotheses, list):
                    session["golden"] = set(golden_hypotheses)

    print(
        f"[resume] restored {len(sessions)} sessions from {checkpoint_dir} "
        f"at turn={resume_from_turn}",
        flush=True,
    )
    return resume_from_turn


def _sequence_fingerprint(
    *,
    mode: str,
    rule_name: str,
    challenge_sequence: Dict[str, Any],
) -> str:
    payload = {
        "mode": mode,
        "rule_name": rule_name,
        "oracle": challenge_sequence.get("oracle"),
        "challenge_type": challenge_sequence.get("challenge_type"),
        "events": challenge_sequence.get("events", []),
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _generate_unique_points(
    *,
    mode: str,
    rule_name: str,
    num_runs: int,
    max_attempts: int,
    generate_fn: Any,
    seed_base: int,
    preprocess_workers: int,
    gpu_label: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    points: List[Dict[str, Any]] = []
    fingerprints: Set[str] = set()
    attempts = 0
    duplicate_skips = 0

    def generate_candidate(attempt_idx: int) -> Tuple[int, Any]:
        rng = random.Random(seed_base * 1000003 + attempt_idx)
        return attempt_idx, generate_fn(rule_name, rng)

    with tqdm(
        total=num_runs,
        desc=f"[GPU {gpu_label}] generate {rule_name}",
        unit="pt",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as pbar:
        next_attempt = 1
        chunk_size = max(1, preprocess_workers * 8)

        def consume_candidates(candidate_iter: Any) -> None:
            nonlocal attempts, duplicate_skips
            for attempt_idx, seq in candidate_iter:
                attempts = attempt_idx
                if len(points) >= num_runs:
                    break
                if seq is None:
                    if attempts % 100 == 0:
                        pbar.set_postfix(attempts=attempts, duplicates=duplicate_skips)
                    continue

                fingerprint = _sequence_fingerprint(
                    mode=mode,
                    rule_name=rule_name,
                    challenge_sequence=seq,
                )
                if fingerprint in fingerprints:
                    duplicate_skips += 1
                    pbar.set_postfix(attempts=attempts, duplicates=duplicate_skips)
                    continue

                point_index = len(points)
                fingerprints.add(fingerprint)
                points.append({
                    "exp_id": f"stats_{rule_name}_{point_index}",
                    "point_index": point_index,
                    "candidate_idx": attempts,
                    "fingerprint": fingerprint,
                    "seq": seq,
                })
                pbar.update(1)
                pbar.set_postfix(attempts=attempts, duplicates=duplicate_skips)

        if preprocess_workers <= 1:
            while len(points) < num_runs and next_attempt <= max_attempts:
                attempt_ids = list(
                    range(next_attempt, min(max_attempts, next_attempt + chunk_size - 1) + 1)
                )
                next_attempt += len(attempt_ids)
                consume_candidates(map(generate_candidate, attempt_ids))
        else:
            with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
                while len(points) < num_runs and next_attempt <= max_attempts:
                    attempt_ids = list(
                        range(next_attempt, min(max_attempts, next_attempt + chunk_size - 1) + 1)
                    )
                    next_attempt += len(attempt_ids)
                    consume_candidates(executor.map(generate_candidate, attempt_ids))

    if len(points) < num_runs:
        print(
            f"[GPU {gpu_label}] WARNING: generated only {len(points)} unique points "
            f"for {rule_name}; target={num_runs}, attempts={attempts}, "
            f"duplicates={duplicate_skips}, max_attempts={max_attempts}"
        )

    return points, attempts, duplicate_skips


def _build_sessions(
    *,
    rule_obj: Any,
    rule_name: str,
    points: List[Dict[str, Any]],
    repeats: int,
    args: argparse.Namespace,
    model_name: str,
    backend: Any,
    output_dir: str,
) -> List[Dict[str, Any]]:
    from task_a.core.environment import RetractionEnvironment
    from task_a.core.orchestrator import GameOrchestrator
    from task_a.core.config import ExperimentConfig

    sessions: List[Dict[str, Any]] = []
    for point in points:
        for repeat_index in range(repeats):
            cfg = ExperimentConfig(
                experiment_id=f"{point['exp_id']}_r{repeat_index}",
                rule_name=rule_name,
                max_turns=point["seq"]["total_turns"],
                seed=args.seed + point["candidate_idx"] * repeats + repeat_index,
                agent_model=model_name,
                agent_temperature=args.agent_temperature,
                agent_max_tokens=_resolve_agent_max_tokens(args),
                output_dir=output_dir,
            )
            env = RetractionEnvironment(
                rule_obj,
                events=point["seq"]["events"],
            )
            orchestrator = GameOrchestrator(
                cfg,
                backend,
                env,
                label_mode="elimination",
                include_evidence_table=False,
                include_rule_predictions=args.include_rule_predictions,
                model_type=args.model_type,
            )
            sessions.append({
                "rule_name": rule_name,
                "point_index": point["point_index"],
                "repeat_index": repeat_index,
                "orchestrator": orchestrator,
            })

    return sessions


def _run_sessions(
    *,
    run_label: str,
    sessions: List[Dict[str, Any]],
    backend: Any,
    temperature: float,
    max_tokens: int,
    preprocess_workers: int,
    checkpoint_dir: Optional[str] = None,
) -> None:
    from task_a.core.agent import build_initial_message
    from task_a.core.environment import parse_example_triple, parse_hypotheses

    def init_session(session: Dict[str, Any]) -> None:
        orchestrator = session["orchestrator"]
        config = orchestrator.config
        session["turns"] = []
        session["conversation"] = [
            {"role": "system", "content": orchestrator._get_system_prompt()},
            {
                "role": "user",
                "content": build_initial_message(
                    config.example_triple,
                    include_evidence_table=orchestrator.include_evidence_table,
                ),
            },
        ]
        session["example_triple"] = parse_example_triple(config.example_triple)
        session["max_turns"] = min(
            config.max_turns,
            orchestrator.env.total_evidence_steps,
        )

    _ordered_thread_map(init_session, sessions, preprocess_workers)

    max_turns = max((session["max_turns"] for session in sessions), default=0)
    for turn in range(max_turns):
        active_sessions: List[Dict[str, Any]] = []
        messages_batch: List[List[Dict[str, str]]] = []

        def prepare_turn_prompt(session: Dict[str, Any]) -> Any:
            if turn >= session["max_turns"]:
                return None

            orchestrator = session["orchestrator"]
            conversation = session["conversation"]
            env_record = orchestrator.env.step(turn)
            evidence_table_text = None
            if orchestrator.include_evidence_table:
                evidence_table_text = orchestrator.env.get_evidence_table(
                    example_triple=session["example_triple"]
                )

            env_text = orchestrator.env.format_feedback(
                env_record,
                label_mode=orchestrator.label_mode,
                include_rule_predictions=orchestrator.include_rule_predictions,
            )
            orchestrator._append_evidence_message(
                conversation=conversation,
                turn=turn,
                env_text=env_text,
                evidence_table_text=evidence_table_text,
                final_turn=turn == session["max_turns"] - 1,
            )

            session["pending_env_record"] = env_record
            return session, orchestrator._build_context_messages(conversation)

        prepared = _ordered_thread_map(prepare_turn_prompt, sessions, preprocess_workers)
        for item in prepared:
            if item is None:
                continue
            session, messages = item
            active_sessions.append(session)
            messages_batch.append(messages)

        if not messages_batch:
            continue

        print(
            f"[vLLM] {run_label}: turn {turn + 1}/{max_turns}, "
            f"prompts={len(messages_batch)}",
            flush=True,
        )
        responses = backend.batch_chat_completion(
            messages_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            use_tqdm=True,
        )

        for session, agent_response in zip(active_sessions, responses):
            orchestrator = session["orchestrator"]
            env_record = session.pop("pending_env_record")
            conversation = session["conversation"]

            conversation.append({"role": "assistant", "content": agent_response})
            hypotheses = parse_hypotheses(agent_response)
            gt = orchestrator.env.get_ground_truth_at_step(turn)
            belief_metrics = orchestrator._compute_belief_metrics(hypotheses, gt)
            session["turns"].append(
                orchestrator._build_turn_record(
                    turn=turn,
                    agent_response=agent_response,
                    hypotheses=hypotheses,
                    env_record=env_record,
                    belief_metrics=belief_metrics,
                )
            )
        if checkpoint_dir:
            _write_benchmark_turn_checkpoint(
                checkpoint_dir=checkpoint_dir,
                run_label=run_label,
                turn_idx=turn,
                max_turns=max_turns,
                sessions=active_sessions,
            )

    for session in sessions:
        orchestrator = session["orchestrator"]
        turns_played = session["turns"]
        session["trajectory"] = {
            "experiment_id": orchestrator.config.experiment_id,
            "rule_name": orchestrator.config.rule_name,
            "rule_description": orchestrator.env.rule.description,
            "max_turns": orchestrator.config.max_turns,
            "agent_model": orchestrator.config.agent_model,
            "n_turns_played": len(turns_played),
            "trajectory_metrics": orchestrator._compute_trajectory_metrics(turns_played),
            "turns": turns_played,
            "conversation": session["conversation"],
        }


def _write_benchmark_turn_checkpoint(
    *,
    checkpoint_dir: str,
    run_label: str,
    turn_idx: int,
    max_turns: int,
    sessions: List[Dict[str, Any]],
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"turn_{turn_idx + 1:02d}.jsonl")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for session in sessions:
            orchestrator = session["orchestrator"]
            conversation = session.get("conversation", [])
            user_message = ""
            if len(conversation) >= 2 and conversation[-2].get("role") == "user":
                user_message = conversation[-2].get("content", "")
            row = {
                "run_label": run_label,
                "turn": turn_idx,
                "turn_number": turn_idx + 1,
                "max_turns": max_turns,
                "rule_name": session.get("rule_name"),
                "point_index": session.get("point_index"),
                "repeat_index": session.get("repeat_index"),
                "oracle": orchestrator.config.rule_name,
                "user_message": user_message,
                "turn_result": session["turns"][-1],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)
    print(
        f"[checkpoint] saved turn {turn_idx + 1}/{max_turns}: "
        f"{path} rows={len(sessions)}",
        flush=True,
    )


def _save_rule_outputs(
    *,
    rule_name: str,
    points: List[Dict[str, Any]],
    sessions: List[Dict[str, Any]],
    classify_fn: Any,
    category_dirs: Dict[str, str],
    repeats: int,
    gpu_label: str,
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    from utils.io import save_json

    sessions_by_point: Dict[int, List[Dict[str, Any]]] = {}
    for session in sessions:
        sessions_by_point.setdefault(session["point_index"], []).append(session)

    counts = {category: 0 for category in CATEGORIES}
    results: List[Dict[str, Any]] = []

    for point in points:
        per_run_categories: List[str] = []
        repeat_trajectories: List[Dict[str, Any]] = []

        point_sessions = sorted(
            sessions_by_point.get(point["point_index"], []),
            key=lambda item: item["repeat_index"],
        )
        for session in point_sessions:
            trajectory = to_jsonable(session["trajectory"])
            if isinstance(trajectory, dict) and "conversation" not in trajectory:
                messages = trajectory.get("messages")
                if isinstance(messages, list):
                    trajectory["conversation"] = messages
            category = classify_fn(trajectory, point["seq"])
            per_run_categories.append(category)
            repeat_trajectories.append({
                "repeat_index": session["repeat_index"],
                "category": category,
                "trajectory": trajectory,
            })

        category = aggregate_repeat_categories(per_run_categories)
        counts[category] += 1

        save_json(
            os.path.join(category_dirs[category], f"{point['exp_id']}.json"),
            {
                "experiment_id": point["exp_id"],
                "rule_name": rule_name,
                "oracle": rule_name,
                "category": category,
                "fingerprint": point["fingerprint"],
                "repeats": repeats,
                "per_run_categories": per_run_categories,
                "repeat_trajectories": repeat_trajectories,
                "candidate_rules": point.get("candidate_rules", []),
                "challenge_sequence": point.get("seq", {}),
            },
        )

        results.append({
            "experiment_id": point["exp_id"],
            "rule_name": rule_name,
            "category": category,
            "per_run_categories": per_run_categories,
            "fingerprint": point["fingerprint"],
        })

        detail = ", ".join(
            f"{cat}={n}" for cat, n in sorted(Counter(per_run_categories).items())
        )
        print(
            f"  [GPU {gpu_label}] [{len(results)}/{len(points)}] "
            f"{point['exp_id']}: {category}  ({detail})"
        )

    return counts, results


def _prepare_rule_data(
    *,
    rule_name: str,
    rule_index: int,
    args: argparse.Namespace,
    model_name: str,
    backend: Any,
    output_dir: str,
    gpu_label: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from task_a.core.rules import get_rule
    from task_a.experiments.generate_sequences import (
        generate_failed_stay_sequence,
        generate_failed_update_sequence_v2,
    )

    generate_fn = (
        generate_failed_update_sequence_v2 if args.mode == "failed_update" else generate_failed_stay_sequence
    )
    max_attempts = max(args.num_runs, 1) * args.max_attempts_multiplier
    seed_base = args.seed + rule_index * 1000003

    print(f"\n[GPU {gpu_label}] {'=' * 50}")
    print(
        f"[GPU {gpu_label}] Rule: {rule_name} "
        f"(data points: {args.num_runs}, repeats: {args.repeats})"
    )
    print(f"[GPU {gpu_label}] {'=' * 50}")

    points, attempts, duplicate_skips = _generate_unique_points(
        mode=args.mode,
        rule_name=rule_name,
        num_runs=args.num_runs,
        max_attempts=max_attempts,
        generate_fn=generate_fn,
        seed_base=seed_base,
        preprocess_workers=args.preprocess_workers,
        gpu_label=gpu_label,
    )
    print(
        f"[GPU {gpu_label}] Generated {len(points)} unique points for {rule_name} "
        f"(attempts={attempts}, duplicates={duplicate_skips})"
    )

    sessions = _build_sessions(
        rule_obj=get_rule(rule_name),
        rule_name=rule_name,
        points=points,
        repeats=args.repeats,
        args=args,
        model_name=model_name,
        backend=backend,
        output_dir=output_dir,
    )
    return points, sessions


def _run_experiment(
    *,
    args: argparse.Namespace,
    output_dir: str,
    category_dirs: Dict[str, str],
    model_name: str,
    gpu_ids: List[int],
) -> Tuple[Dict[str, Dict[str, int]], List[Dict[str, Any]]]:
    from utils.llm_backend import VLLMBackend

    gpu_label = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_label

    print(f"[GPU {gpu_label}] Loading vLLM: {args.agent_model_path}")
    backend = VLLMBackend(
        model_path=args.agent_model_path,
        dtype=args.vllm_dtype,
        max_model_len=args.vllm_max_model_len,
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    backend.sampling_overrides = _build_sampling_overrides(args)

    points_by_rule: Dict[str, List[Dict[str, Any]]] = {}
    sessions_by_rule: Dict[str, List[Dict[str, Any]]] = {}
    all_sessions: List[Dict[str, Any]] = []

    for rule_index, rule_name in enumerate(args.rules):
        points, sessions = _prepare_rule_data(
            rule_name=rule_name,
            rule_index=rule_index,
            args=args,
            model_name=model_name,
            backend=backend,
            output_dir=output_dir,
            gpu_label=gpu_label,
        )
        points_by_rule[rule_name] = points
        sessions_by_rule[rule_name] = sessions
        all_sessions.extend(sessions)

    print(
        f"\n[GPU {gpu_label}] Starting global inference: "
        f"rules={len(args.rules)}, sessions={len(all_sessions)}"
    )
    _run_sessions(
        run_label=f"{len(args.rules)} rules",
        sessions=all_sessions,
        backend=backend,
        temperature=args.agent_temperature,
        max_tokens=_resolve_agent_max_tokens(args),
        preprocess_workers=args.preprocess_workers,
        checkpoint_dir=os.path.join(output_dir, "turn_checkpoints"),
    )

    classify_fn = CLASSIFIERS[args.mode]
    all_stats: Dict[str, Dict[str, int]] = {}
    all_results: List[Dict[str, Any]] = []
    for rule_name in args.rules:
        points = points_by_rule[rule_name]
        counts, results = _save_rule_outputs(
            rule_name=rule_name,
            points=points,
            sessions=sessions_by_rule[rule_name],
            classify_fn=classify_fn,
            category_dirs=category_dirs,
            repeats=args.repeats,
            gpu_label=gpu_label,
        )
        all_stats[rule_name] = counts
        all_results.extend(results)

        print(f"\n  [GPU {gpu_label}] --- {rule_name} summary ({len(points)} data points x {args.repeats} repeats) ---")
        for category in CATEGORIES:
            n = counts[category]
            pct = n / len(points) * 100 if points else 0
            print(f"  [GPU {gpu_label}] {category}: {n}/{len(points)} ({pct:.1f}%)")

    return all_stats, all_results


def _parse_gpu_ids(raw_gpus: str) -> List[int]:
    gpu_ids = [int(gpu.strip()) for gpu in raw_gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpu_ids


def _has_existing_result_files(category_dirs: Dict[str, str]) -> bool:
    for category_dir in category_dirs.values():
        if not os.path.isdir(category_dir):
            continue
        if any(name.endswith(".json") for name in os.listdir(category_dir)):
            return True
    return False


def _resolve_belief_stats_candidates(
    args: argparse.Namespace,
) -> Tuple[List[str], List[str], List[str]]:
    from task_a.core.rules import get_rule, resolve_heldout_rules

    heldout_pool = resolve_heldout_rules(args.heldout_set)
    selected_heldout_names: List[str] = []
    if args.include_heldout:
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

    if bool(getattr(args, "heldout_only_candidates", False)):
        if not selected_heldout_names:
            raise ValueError("--heldout-only-candidates requires --include-heldout")
        candidate_names = list(selected_heldout_names)
    else:
        benchmark_names = list(dict.fromkeys(args.rules or []))
        if len(benchmark_names) != 5:
            raise ValueError(
                "--rules must explicitly provide exactly 5 benchmark rules. "
                f"Got {len(benchmark_names)} unique rules: {benchmark_names}"
            )
        benchmark_pool = set(BENCHMARK_RULES)
        unknown_benchmark = [name for name in benchmark_names if name not in benchmark_pool]
        if unknown_benchmark:
            raise ValueError(
                "--rules may only contain benchmark rules from task_a.core.rules.EXTENDED_BENCHMARK_RULES. "
                f"Unknown benchmark rules: {unknown_benchmark}. Available: {sorted(benchmark_pool)}"
            )
        candidate_names = list(benchmark_names)

    if args.include_heldout:
        for name in selected_heldout_names:
            if name not in candidate_names:
                candidate_names.append(name)

    for name in candidate_names:
        get_rule(name)

    if args.targets is None:
        target_names = (
            list(selected_heldout_names)
            if args.include_heldout and selected_heldout_names
            else list(candidate_names)
        )
    else:
        target_names = list(dict.fromkeys(args.targets))
        candidate_set = set(candidate_names)
        unknown_targets = [name for name in target_names if name not in candidate_set]
        if unknown_targets:
            raise ValueError(
                "Unknown targets (or not in candidate set): "
                f"{unknown_targets}. Candidate set: {sorted(candidate_names)}"
            )

    return candidate_names, selected_heldout_names, target_names


def _run_belief_stats_sequences_vllm(
    *,
    backend: Any,
    rows: List[Dict[str, Any]],
    model_label: str,
    mode: str,
    candidate_names: List[str],
    heldout_set: str,
    temperature: float,
    max_tokens: int,
    include_rule_predictions: bool = True,
    perturb_oracle_rule_prediction_in_post: bool = False,
    hide_rule_predictions_in_failed_stay_post: bool = False,
    preprocess_workers: int = 1,
    checkpoint_dir: Optional[str] = None,
    run_label: str = "belief_stats",
    add_host_comment: bool = True,
    add_failed_isolation_comment: bool = True,
    model_type: str = "local",
    resume_from_turn: int = 0,
    strict_failed_isolation_prefix_turns: int = 0,
    failed_isolation_comment_start_turn: int = 0,
    preserve_failed_isolation_turn_message: bool = False,
    prefix_update_hint_turns: int = 0,
    annotate_rule_predictions: bool = False,
) -> List[Dict[str, Any]]:
    if mode == "noise":
        def init_noise_row(row: Dict[str, Any]) -> Dict[str, Any]:
            return _init_noise_session(
                row=row,
                model=model_label,
                candidate_names=candidate_names,
                heldout_set=heldout_set,
                add_host_comment=add_host_comment,
            )

        sessions = _ordered_thread_map(init_noise_row, rows, preprocess_workers)
        max_turns = max((int(session.get("max_turns", 0)) for session in sessions), default=0)
        checkpoint_turn_idx = 0

        while True:
            active_sessions: List[Dict[str, Any]] = []
            messages_batch: List[List[Dict[str, str]]] = []
            prepared_messages = _ordered_thread_map(
                _prepare_noise_session_prompt,
                sessions,
                preprocess_workers,
            )
            for session, messages in zip(sessions, prepared_messages):
                if messages is None:
                    continue
                active_sessions.append(session)
                messages_batch.append(messages)

            if not messages_batch:
                break

            print(f"[vLLM] noise batch prompts={len(messages_batch)}", flush=True)
            responses = backend.batch_chat_completion(
                messages_batch,
                temperature=temperature,
                max_tokens=max_tokens,
                use_tqdm=True,
            )
            for session, response in zip(active_sessions, responses):
                _apply_noise_session_response(session, response, add_host_comment=add_host_comment)
            if checkpoint_dir:
                _write_host_turn_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    run_label=run_label,
                    turn_idx=checkpoint_turn_idx,
                    max_turns=max_turns,
                    sessions=active_sessions,
                )
            checkpoint_turn_idx += 1

        return [_finalize_noise_session_result(session) for session in sessions]
    def init_row(row: Dict[str, Any]) -> Dict[str, Any]:
        return _init_belief_stats_session(
            row=row,
            model=model_label,
            mode=mode,
            candidate_names=candidate_names,
            heldout_set=heldout_set,
            include_rule_predictions=include_rule_predictions,
            perturb_oracle_rule_prediction_in_post=perturb_oracle_rule_prediction_in_post,
            hide_rule_predictions_in_failed_stay_post=hide_rule_predictions_in_failed_stay_post,
            add_failed_isolation_comment=add_failed_isolation_comment,
            model_type=model_type,
            strict_failed_isolation_prefix_turns=strict_failed_isolation_prefix_turns,
            failed_isolation_comment_start_turn=failed_isolation_comment_start_turn,
            preserve_failed_isolation_turn_message=preserve_failed_isolation_turn_message,
            prefix_update_hint_turns=prefix_update_hint_turns,
            annotate_rule_predictions=annotate_rule_predictions,
        )

    sessions = _ordered_thread_map(init_row, rows, preprocess_workers)
    max_turns = max((len(session["sequence"].get("events", [])) for session in sessions), default=0)
    checkpoint_turn_idx = 0
    checkpoint_turn_idx = _resume_belief_stats_sessions_from_checkpoint(
        sessions=sessions,
        checkpoint_dir=checkpoint_dir,
        resume_from_turn=resume_from_turn,
    )

    while True:
        active_sessions: List[Dict[str, Any]] = []
        messages_batch: List[List[Dict[str, str]]] = []
        prepared_messages = _ordered_thread_map(
            _prepare_belief_stats_session_prompt,
            sessions,
            preprocess_workers,
        )
        for session, messages in zip(sessions, prepared_messages):
            if messages is None:
                continue
            active_sessions.append(session)
            messages_batch.append(messages)

        if not messages_batch:
            break

        print(f"[vLLM] belief_stats batch prompts={len(messages_batch)}", flush=True)
        responses = backend.batch_chat_completion(
            messages_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            use_tqdm=True,
        )
        for session, response in zip(active_sessions, responses):
            _apply_belief_stats_session_response(session, response)
        if checkpoint_dir:
            _write_host_turn_checkpoint(
                checkpoint_dir=checkpoint_dir,
                run_label=run_label,
                turn_idx=checkpoint_turn_idx,
                max_turns=max_turns,
                sessions=active_sessions,
            )
        checkpoint_turn_idx += 1

    return [_finalize_belief_stats_session_result(session) for session in sessions]


def _run_belief_stats_vllm_experiment(
    *,
    args: argparse.Namespace,
    output_dir: str,
    category_dirs: Dict[str, str],
    model_name: str,
    gpu_ids: List[int],
) -> Tuple[Dict[str, Dict[str, int]], List[Dict[str, Any]]]:
    from task_a.experiments.generate_sequences import (
        generate_belief_stats_failed_stay_sequence,
        generate_belief_stats_failed_update_sequence,
    )
    from utils.io import save_json
    from utils.llm_backend import APIBackend, VLLMBackend

    if args.mode == "failed_update" and args.evidence_per_round == 3:
        raise ValueError("--evidence-per-round 3 is currently supported only for failed_stay")

    candidate_names, selected_heldout_names, target_names = _resolve_belief_stats_candidates(args)
    if args.backend == "vllm":
        gpu_label = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_label
    else:
        gpu_label = "api"

    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    if args.backend == "vllm":
        print(f"[GPU {gpu_label}] Loading belief_stats vLLM: {args.agent_model_path}")
        backend = VLLMBackend(
            model_path=args.agent_model_path,
            dtype=args.vllm_dtype,
            max_model_len=args.vllm_max_model_len,
            tensor_parallel_size=len(gpu_ids),
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )
    else:
        api_key = args.api_key or os.environ.get(args.api_key_env, "")
        api_model = args.api_model or args.agent_model_path
        print(f"[API] Loading belief_stats API backend: model={api_model}, endpoint={args.api_base_url}")
        backend = APIBackend(
            api_base_url=args.api_base_url,
            model_name=api_model,
            api_key=api_key or None,
        )

    backend.sampling_overrides = _build_sampling_overrides(args)

    combos: List[Tuple[str, int]] = [
        (oracle, run_idx)
        for oracle in target_names
        for run_idx in range(args.num_runs)
    ]

    def case_seed(oracle: str, run_idx: int) -> int:
        oracle_offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(oracle))
        return args.seed * 1000003 + oracle_offset * 9176 + run_idx

    def build_base_case(combo: Tuple[str, int]) -> Dict[str, Any]:
        oracle, run_idx = combo
        exp_id = f"{args.mode}_{oracle}_r{run_idx}"
        rng = random.Random(case_seed(oracle, run_idx))
        seq = None
        for _ in range(300):
            if args.mode in ("failed_stay", "failed_isolation"):
                seq = generate_belief_stats_failed_stay_sequence(
                    oracle=oracle,
                    rng=rng,
                    candidate_names=candidate_names,
                    heldout_set=args.heldout_set,
                    post_convergence_interference_rounds=args.post_convergence_interference_rounds,
                    evidence_per_round=args.evidence_per_round,
                    n_yes_per_round=getattr(args, "n_yes_per_round", None),
                )
            elif args.mode == "failed_update":
                seq = generate_belief_stats_failed_update_sequence(
                    oracle=oracle,
                    rng=rng,
                    candidate_names=candidate_names,
                    heldout_set=args.heldout_set,
                    evidence_per_round=args.evidence_per_round,
                    n_yes_per_round=getattr(args, "n_yes_per_round", None),
                )
            else:
                seq = {
                    "challenge_type": "belief_stats_noise_query",
                    "mode": "noise",
                    "oracle": oracle,
                    "events": [
                        {"type": "noise_turn", "turn": turn}
                        for turn in range(args.noise_max_turns)
                    ],
                    "ground_truth": [
                        {"turn": turn, "survivors": [oracle]}
                        for turn in range(args.noise_max_turns)
                    ],
                    "challenge_turns": list(range(args.noise_max_turns)),
                    "convergence_turn": -1,
                    "total_turns": args.noise_max_turns,
                    "max_turns": args.noise_max_turns,
                }
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

    print(
        f"[generate] belief_stats base cases={len(combos)} "
        f"preprocess_workers={args.preprocess_workers}",
        flush=True,
    )
    base_cases: List[Dict[str, Any]] = _ordered_thread_map(
        build_base_case,
        combos,
        args.preprocess_workers,
    )
    
    rows_by_oracle: Dict[str, List[Dict[str, Any]]] = {}
    all_results: List[Dict[str, Any]] = []
    for case in base_cases:
        for repeat_idx in range(args.repeats):
            repeat_exp_id = f"{case['sample_id']}_rep{repeat_idx}"
            if "error" in case:
                all_results.append(
                    to_jsonable(
                        {
                            "mode": args.mode,
                            "oracle": case["oracle"],
                            "seed": case["seed"],
                            "run_idx": case["run_idx"],
                            "repeat_index": repeat_idx,
                            "experiment_id": repeat_exp_id,
                            "sample_id": case["sample_id"],
                            "model": model_name,
                            "backend": args.backend,
                            "error": case["error"],
                            "category": "insufficient_capability",
                        }
                    )
                )
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

    if expanded_rows:
        vllm_results = _run_belief_stats_sequences_vllm(
            backend=backend,
            rows=expanded_rows,
            model_label=model_name,
            mode=args.mode,
            candidate_names=candidate_names,
            heldout_set=args.heldout_set,
            temperature=args.agent_temperature,
            max_tokens=_resolve_agent_max_tokens(args),
            include_rule_predictions=args.include_rule_predictions,
            perturb_oracle_rule_prediction_in_post=args.perturb_oracle_rule_prediction_in_post,
            hide_rule_predictions_in_failed_stay_post=args.hide_rule_predictions_in_failed_stay_post,
            preprocess_workers=args.preprocess_workers,
            checkpoint_dir=os.path.join(output_dir, "turn_checkpoints"),
            run_label=f"task_a {args.mode} ({len(target_names)} targets)",
            add_host_comment=args.add_host_comment if args.mode == "noise" else True,
            add_failed_isolation_comment=args.add_failed_isolation_comment,
            model_type=args.model_type,
            strict_failed_isolation_prefix_turns=args.strict_failed_isolation_prefix_turns,
            failed_isolation_comment_start_turn=args.failed_isolation_comment_start_turn,
            preserve_failed_isolation_turn_message=args.preserve_failed_isolation_turn_message,
            prefix_update_hint_turns=args.prefix_update_hint_turns,
            annotate_rule_predictions=args.annotate_rule_predictions,
        )
        for result in vllm_results:
            result["backend"] = args.backend
            all_results.append(result)

    for result in all_results:
        if "category" not in result:
            result["category"] = (
                "insufficient_capability" if "error" in result else classify_round_match_result(result)
            )
        result["category"] = normalize_category(str(result.get("category", "insufficient_capability")))

    sample_grouped: Dict[str, List[Dict[str, Any]]] = {}
    for result in all_results:
        sample_grouped.setdefault(str(result.get("sample_id")), []).append(result)

    sample_results: List[Dict[str, Any]] = []
    all_stats: Dict[str, Dict[str, int]] = {
        oracle: {category: 0 for category in CATEGORIES}
        for oracle in target_names
    }
    grouped_cases: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

    def _persist_running_outputs(changed_case_keys: Optional[List[Tuple[str, int]]] = None) -> None:
        save_json(os.path.join(output_dir, "all_results.json"), to_jsonable(all_results))
        save_json(os.path.join(output_dir, "sample_results.json"), to_jsonable(sample_results))
        keys = changed_case_keys if changed_case_keys is not None else list(grouped_cases.keys())
        for oracle, seed in keys:
            cases = grouped_cases.get((oracle, seed), [])
            cases_sorted = sorted(
                cases,
                key=lambda item: (
                    int(item.get("run_idx", 0)),
                    int(item.get("repeat_index", 0)),
                ),
            )
            save_json(
                os.path.join(traj_dir, f"{args.mode}_{oracle}_s{seed}.json"),
                to_jsonable(
                    {
                        "mode": args.mode,
                        "backend": args.backend,
                        "oracle": oracle,
                        "seed": seed,
                        "model": model_name,
                        "repeats": args.repeats,
                        "n_cases": len(cases_sorted),
                        "cases": cases_sorted,
                    }
                ),
            )

    def _consume_sample_results(sample_id: str, repeats: List[Dict[str, Any]]) -> None:
        repeats_sorted = sorted(repeats, key=lambda item: int(item.get("repeat_index", 0)))
        for item in repeats_sorted:
            item["backend"] = args.backend
            if "category" not in item:
                item["category"] = (
                    "insufficient_capability" if "error" in item else classify_round_match_result(item)
                )
            item["category"] = normalize_category(str(item.get("category", "insufficient_capability")))

        per_run_categories = [
            normalize_category(str(item.get("category", "insufficient_capability")))
            for item in repeats_sorted
        ]
        category = aggregate_repeat_categories(per_run_categories)
        first = repeats_sorted[0]
        oracle = str(first.get("oracle"))
        all_stats.setdefault(oracle, {cat: 0 for cat in CATEGORIES})
        all_stats[oracle][category] += 1
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
            "oracle": oracle,
            "category": category,
            "seed": first.get("seed"),
            "run_idx": first.get("run_idx"),
            "repeats": args.repeats,
            "per_run_categories": per_run_categories,
            "repeat_trajectories": repeat_trajectories,
        }
        save_json(
            os.path.join(category_dirs[category], f"{sample_id}.json"),
            to_jsonable(sample_payload),
        )

        all_results.extend(repeats_sorted)
        sample_results.append(
            {
                "sample_id": sample_id,
                "mode": args.mode,
                "oracle": oracle,
                "seed": first.get("seed"),
                "run_idx": first.get("run_idx"),
                "category": category,
                "per_run_categories": per_run_categories,
                "repeat_experiment_ids": [item.get("experiment_id") for item in repeats_sorted],
            }
        )

        seed = int(first.get("seed"))
        grouped_cases.setdefault((oracle, seed), []).extend(repeats_sorted)
        _persist_running_outputs(changed_case_keys=[(oracle, seed)])

    for case in base_cases:
        sample_id = str(case["sample_id"])
        if "error" in case:
            error_repeats: List[Dict[str, Any]] = []
            for repeat_idx in range(args.repeats):
                error_repeats.append(
                    to_jsonable(
                        {
                            "mode": args.mode,
                            "oracle": case["oracle"],
                            "seed": case["seed"],
                            "run_idx": case["run_idx"],
                            "repeat_index": repeat_idx,
                            "experiment_id": f"{sample_id}_rep{repeat_idx}",
                            "sample_id": sample_id,
                            "model": model_name,
                            "backend": args.backend,
                            "error": case["error"],
                            "category": "insufficient_capability",
                        }
                    )
                )
            _consume_sample_results(sample_id, error_repeats)
            continue

        sample_rows: List[Dict[str, Any]] = []
        for repeat_idx in range(args.repeats):
            repeat_exp_id = f"{sample_id}_rep{repeat_idx}"
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "repeat_experiment_id": repeat_exp_id,
                    "oracle": case["oracle"],
                    "seed": case["seed"],
                    "run_idx": case["run_idx"],
                    "repeat_index": repeat_idx,
                    "sequence": json.loads(json.dumps(to_jsonable(case["sequence"]))),
                }
            )
        rows_by_oracle.setdefault(str(case["oracle"]), []).extend(sample_rows)

    total_samples = len(base_cases)
    processed_samples = len(sample_results)
    for oracle in target_names:
        oracle_rows = rows_by_oracle.get(oracle, [])
        if not oracle_rows:
            continue
        vllm_results = _run_belief_stats_sequences_vllm(
            backend=backend,
            rows=oracle_rows,
            model_label=model_name,
            mode=args.mode,
            candidate_names=candidate_names,
            heldout_set=args.heldout_set,
            temperature=args.agent_temperature,
            max_tokens=_resolve_agent_max_tokens(args),
            include_rule_predictions=args.include_rule_predictions,
            perturb_oracle_rule_prediction_in_post=args.perturb_oracle_rule_prediction_in_post,
            hide_rule_predictions_in_failed_stay_post=args.hide_rule_predictions_in_failed_stay_post,
            preprocess_workers=args.preprocess_workers,
            checkpoint_dir=os.path.join(output_dir, "turn_checkpoints", f"oracle_{oracle}"),
            run_label=f"task_a {args.mode} oracle={oracle}",
            add_host_comment=args.add_host_comment if args.mode == "noise" else True,
            add_failed_isolation_comment=args.add_failed_isolation_comment,
            model_type=args.model_type,
            resume_from_turn=args.resume_from_turn,
        )
        results_by_sample: Dict[str, List[Dict[str, Any]]] = {}
        for result in vllm_results:
            results_by_sample.setdefault(str(result.get("sample_id")), []).append(result)
        for sample_id, repeats in sorted(results_by_sample.items()):
            _consume_sample_results(sample_id, repeats)
            processed_samples += 1
        print(
            f"[save-progress] oracle={oracle} finalized {processed_samples}/{total_samples} samples",
            flush=True,
        )

    valid = [result for result in all_results if "error" not in result]
    category_counts = {
        category: sum(1 for item in sample_results if item.get("category") == category)
        for category in CATEGORIES
    }
    summary = {
        "mode": args.mode,
        "backend": args.backend,
        "evidence_per_round": args.evidence_per_round,
        "include_rule_predictions": bool(args.include_rule_predictions),
        "hide_rule_predictions_in_failed_stay_post": bool(
            args.hide_rule_predictions_in_failed_stay_post
        ),
        "perturb_oracle_rule_prediction_in_post": bool(
            args.perturb_oracle_rule_prediction_in_post
        ),
        "post_convergence_interference_rounds": args.post_convergence_interference_rounds,
        "strict_failed_isolation_prefix_turns": args.strict_failed_isolation_prefix_turns,
        "failed_isolation_comment_start_turn": args.failed_isolation_comment_start_turn,
        "preserve_failed_isolation_turn_message": args.preserve_failed_isolation_turn_message,
        "prefix_update_hint_turns": args.prefix_update_hint_turns,
        "annotate_rule_predictions": args.annotate_rule_predictions,
        "model": model_name,
        "rules": candidate_names,
        "selected_heldout_rules": selected_heldout_names,
        "targets": target_names,
        "seed": args.seed,
        "num_runs_per_target": args.num_runs,
        "repeats": args.repeats,
        "preprocess_workers": args.preprocess_workers,
        "api_num_workers": None,
        "n_repeat_runs": len(all_results),
        "n_samples": len(sample_results),
        "n_valid": len(valid),
        "final_match_rate": (
            sum(1 for result in valid if result.get("final_match")) / len(valid)
            if valid
            else None
        ),
        "avg_turn_match_rate": (
            sum(result.get("turn_match_rate", 0.0) for result in valid) / len(valid)
            if valid
            else None
        ),
        "category_counts": category_counts,
        "category_percentages": {
            category: round(category_counts[category] / max(len(sample_results), 1) * 100, 2)
            for category in CATEGORIES
        },
        "total": {
            "total": len(sample_results),
            **{category: category_counts[category] for category in CATEGORIES},
            **{
                f"{category}_pct": round(
                    category_counts[category] / max(len(sample_results), 1) * 100,
                    2,
                )
                for category in CATEGORIES
            },
        },
        "per_oracle": {},
    }

    for oracle in target_names:
        subset = [result for result in valid if result.get("oracle") == oracle]
        sample_subset = [result for result in sample_results if result.get("oracle") == oracle]
        if not subset and not sample_subset:
            continue
        oracle_category_counts = {
            category: sum(1 for item in sample_subset if item.get("category") == category)
            for category in CATEGORIES
        }
        summary["per_oracle"][oracle] = {
            "n": len(subset),
            "n_samples": len(sample_subset),
            "final_match_rate": (
                sum(1 for result in subset if result.get("final_match")) / len(subset)
                if subset
                else None
            ),
            "avg_turn_match_rate": (
                sum(result.get("turn_match_rate", 0.0) for result in subset) / len(subset)
                if subset
                else None
            ),
            "category_counts": oracle_category_counts,
        }

    save_json(os.path.join(output_dir, "stats_report.json"), to_jsonable(summary))
    save_json(os.path.join(output_dir, "summary.json"), to_jsonable(summary))

    print("\n" + "=" * 60, flush=True)
    if summary["final_match_rate"] is None:
        print("No valid results.", flush=True)
    else:
        print(
            f"Done. final_match_rate={summary['final_match_rate']:.1%} "
            f"avg_turn_match_rate={summary['avg_turn_match_rate']:.1%}",
            flush=True,
        )
    for category in CATEGORIES:
        n = category_counts[category]
        pct = n / max(len(sample_results), 1) * 100
        print(f"{category}: {n}/{len(sample_results)} ({pct:.1f}%)", flush=True)
    print(f"Stats report: {os.path.join(output_dir, 'stats_report.json')}", flush=True)
    print(f"Sample results: {os.path.join(output_dir, 'sample_results.json')}", flush=True)
    args.rules = target_names
    return all_stats, sample_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario A belief statistics")
    parser.add_argument("--backend", choices=["vllm", "api"], default="vllm")
    parser.add_argument("--model-type", choices=["local", "api_qwen35"], default="local", help="Model type for template selection")
    parser.add_argument("--api-base-url", type=str, default="")
    parser.add_argument("--api-model", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["failed_update", "failed_stay", "noise", "failed_isolation"],
        help="Experiment mode: failed_update uses retraction; failed_stay uses ambiguous evidence; noise uses interactive triple queries; failed_isolation reuses failed_stay game flow but appends a misleading host comment to every turn.",
    )
    parser.add_argument(
        "--agent-model-path",
        type=str,
        default=os.environ.get("AGENT_MODEL_PATH", "models/Qwen3-30B-A3B-Instruct-2507"),
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs used by one vLLM instance.",
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-max-model-len", "--max-model-len", dest="vllm_max_model_len", type=int, default=None)
    parser.add_argument("--vllm-dtype", type=str, default="bf16")
    parser.add_argument("--num-runs", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--rules",
        nargs="+",
        required=False,
        help=(
            "Exactly 5 benchmark rules to use, chosen from "
            "task_a.core.rules.EXTENDED_BENCHMARK_RULES, unless "
            "--heldout-only-candidates is set."
        ),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help=(
            "Optional oracle rules. If omitted with --include-heldout, selected "
            "heldout rules are used; otherwise all candidate rules are used."
        ),
    )
    parser.add_argument("--include-heldout", action="store_true")
    parser.add_argument("--heldout-set", choices=["easy", "hard"], default="easy")
    parser.add_argument(
        "--heldout-only-candidates",
        action="store_true",
        help=(
            "Use only the selected heldout rules as the candidate set. This matches "
            "the Scenario A 7B heldout failed_stay/failed_update case format."
        ),
    )
    parser.add_argument(
        "--heldout-rules",
        nargs="+",
        default=None,
        help=(
            "Heldout rules to add for belief_stats sampling. If omitted with "
            "--include-heldout, all heldout rules in --heldout-set are used."
        ),
    )
    parser.add_argument(
        "--post-convergence-interference-rounds",
        type=int,
        default=1,
        help="belief_stats failed_stay post-interference turns after convergence.",
    )
    parser.add_argument(
        "--evidence-per-round",
        choices=[1, 2, 3, 4],
        type=int,
        default=1,
        help=(
            "Heldout belief_stats convergence evidence count per model turn. "
            "Benchmark-only generation keeps the original sequence logic."
        ),
    )
    parser.add_argument(
        "--n-yes-per-round",
        type=int,
        default=None,
        dest="n_yes_per_round",
        help=(
            "Number of YES evidences per round when evidence_per_round >= 4. "
            "The remaining (evidence_per_round - n_yes_per_round) evidences will be NO. "
            "If omitted, YES/NO is sampled randomly (original behaviour). "
            "Example: --n-yes-per-round 2 with --evidence-per-round 4 gives 2 YES + 2 NO per round."
        ),
    )
    parser.add_argument(
        "--noise-max-turns",
        type=int,
        default=6,
        help="Noise mode: maximum interactive rounds before stop.",
    )
    parser.set_defaults(perturb_oracle_rule_prediction_in_post=False)
    parser.add_argument(
        "--perturb-oracle-rule-prediction-in-post",
        dest="perturb_oracle_rule_prediction_in_post",
        action="store_true",
        help=(
            "For belief_stats failed_stay, flip only the oracle row in displayed "
            "rule_predictions during post-interference turns."
        ),
    )
    parser.add_argument(
        "--no-perturb-oracle-rule-prediction-in-post",
        dest="perturb_oracle_rule_prediction_in_post",
        action="store_false",
    )
    parser.add_argument(
        "--hide-rule-predictions-in-failed_stay-post",
        action="store_true",
        help=(
            "For belief_stats failed_stay, omit displayed rule_predictions during "
            "post-interference turns while keeping them in the prefix turns."
        ),
    )
    parser.set_defaults(include_rule_predictions=True)
    parser.add_argument(
        "--include-rule-predictions",
        dest="include_rule_predictions",
        action="store_true",
        help="Display rule_predictions in belief_stats prompts (default: True).",
    )
    parser.add_argument(
        "--no-include-rule-predictions",
        dest="include_rule_predictions",
        action="store_false",
        help="Hide rule_predictions in all turns.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent-temperature", "--temperature", dest="agent_temperature", type=float, default=0.3)
    parser.add_argument("--agent-max-tokens", "--max-output-tokens", dest="agent_max_tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--sampling-top-p", "--top-p", "--agent-top-p", dest="sampling_top_p", type=float, default=None)
    parser.add_argument("--sampling-top-k", "--top-k", "--agent-top-k", dest="sampling_top_k", type=int, default=None)
    parser.add_argument("--agent-min-p", dest="sampling_min_p", type=float, default=None)
    parser.add_argument(
        "--sampling-presence-penalty",
        "--presence-penalty",
        "--agent-presence-penalty",
        dest="sampling_presence_penalty",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--sampling-repetition-penalty",
        "--repetition-penalty",
        "--agent-repetition-penalty",
        dest="sampling_repetition_penalty",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-attempts-multiplier",
        type=int,
        default=20,
        help="Max candidate sequence attempts per rule = num_runs * multiplier.",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=1,
        help="Thread workers for sequence generation and per-turn prompt batch construction.",
    )
    parser.add_argument(
        "--resume-from-turn",
        type=int,
        default=0,
        help=(
            "Resume from an existing checkpoint turn index. "
            "Example: 1 means turn0 is already completed and run continues from turn1. "
            "Requires --output-dir to point to an existing run directory."
        ),
    )
    parser.set_defaults(add_host_comment=True)
    parser.add_argument(
        "--add-host-comment",
        dest="add_host_comment",
        action="store_true",
        help="Add host comments to noise mode feedback (default: True).",
    )
    parser.add_argument(
        "--no-add-host-comment",
        dest="add_host_comment",
        action="store_false",
        help="Disable host comments in noise mode feedback.",
    )
    parser.set_defaults(add_failed_isolation_comment=True)
    parser.add_argument(
        "--add-failed_isolation-comment",
        dest="add_failed_isolation_comment",
        action="store_true",
        help="Append a misleading host comment to every failed_isolation mode turn (default: True).",
    )
    parser.add_argument(
        "--no-add-failed_isolation-comment",
        dest="add_failed_isolation_comment",
        action="store_false",
        help="Disable failed_isolation mode host comment injection (turns into a plain failed_stay run).",
    )
    parser.add_argument(
        "--strict-failed_isolation-prefix-turns",
        type=int,
        default=0,
        help=(
            "FailedIsolation only: require exact matches for this many prefix turns before "
            "counting later wrong hypotheses as valid failed_isolation data. "
            "0 preserves the original final-answer-only classifier."
        ),
    )
    parser.add_argument(
        "--failed_isolation-comment-start-turn",
        type=int,
        default=0,
        help="FailedIsolation only: start appending misleading comments at this zero-based prompt turn.",
    )
    parser.add_argument(
        "--preserve-failed_isolation-turn-message",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "FailedIsolation only: preserve the normal evidence prompt and append the "
            "misleading comment, instead of rewriting to a short sentence."
        ),
    )
    parser.add_argument(
        "--prefix-update-hint-turns",
        type=int,
        default=0,
        help=(
            "FailedIsolation only: for this many prefix turns, append the evidence-only "
            "updated hypothesis as a calibration hint. 0 disables the hint."
        ),
    )
    parser.add_argument(
        "--annotate-rule-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Annotate rule prediction rows with '(consistent)' or "
            "'(CONTRADICTS evidence → eliminated)', matching the heldout 7B "
            "failed_stay/failed_update interaction format."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_runs < 1:
        raise ValueError("--num-runs must be >= 1")
    if args.strict_failed_isolation_prefix_turns < 0:
        raise ValueError("--strict-failed_isolation-prefix-turns must be >= 0")
    if args.failed_isolation_comment_start_turn < 0:
        raise ValueError("--failed_isolation-comment-start-turn must be >= 0")
    if args.prefix_update_hint_turns < 0:
        raise ValueError("--prefix-update-hint-turns must be >= 0")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.max_attempts_multiplier < 1:
        raise ValueError("--max-attempts-multiplier must be >= 1")
    if args.preprocess_workers < 1:
        raise ValueError("--preprocess-workers must be >= 1")
    if args.post_convergence_interference_rounds < 1:
        raise ValueError("--post-convergence-interference-rounds must be >= 1")
    if args.noise_max_turns < 1:
        raise ValueError("--noise-max-turns must be >= 1")
    if args.resume_from_turn < 0:
        raise ValueError("--resume-from-turn must be >= 0")
    if args.resume_from_turn > 0 and not args.output_dir:
        raise ValueError("--resume-from-turn requires --output-dir")

    if args.backend == "api" and not args.api_base_url:
        raise ValueError("--api-base-url is required when --backend api")

    gpu_ids = _parse_gpu_ids(args.gpus) if args.backend == "vllm" else [0]
    model_ref = args.api_model or args.agent_model_path if args.backend == "api" else args.agent_model_path
    model_name = os.path.basename(model_ref)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        "task_a/outputs",
        f"{args.mode}_stats_{model_name}_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)

    category_dirs = build_category_dirs(output_dir)
    if args.output_dir and args.resume_from_turn <= 0 and _has_existing_result_files(category_dirs):
        raise ValueError(
            f"Output dir already contains result JSON files: {output_dir}. "
            "Choose a fresh --output-dir for a new run."
        )
    for category_dir in category_dirs.values():
        os.makedirs(category_dir, exist_ok=True)

    print(f"Mode: {args.mode}")
    print(f"Model: {model_name}")
    if args.backend == "vllm":
        print("Inference path: local vLLM belief_stats sampling")
        print(f"GPUs: {gpu_ids} (single vLLM, tensor_parallel_size={len(gpu_ids)})")
    else:
        print(f"Inference path: API chat completions ({args.api_base_url})")
    requested_rule_count = len(args.heldout_rules or []) if args.heldout_only_candidates else len(args.rules or [])
    print(f"Rules: {requested_rule_count}, Data points per rule: {args.num_runs}, Repeats: {args.repeats}")
    print(f"Preprocess workers: {args.preprocess_workers}")
    if args.resume_from_turn > 0:
        print(f"Resume from turn: {args.resume_from_turn}")
    print(f"Output dir: {output_dir}")

    all_stats, all_results = _run_belief_stats_vllm_experiment(
        args=args,
        output_dir=output_dir,
        category_dirs=category_dirs,
        model_name=model_name,
        gpu_ids=gpu_ids,
    )
    _ = all_stats, all_results
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
