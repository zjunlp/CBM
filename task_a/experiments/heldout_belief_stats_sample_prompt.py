import argparse
import hashlib
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm.auto import tqdm

from task_a.core.rules import get_rule, resolve_heldout_rules


CATEGORIES = ["insufficient_capability", "oracle_match", "belief_failure", "unstable"]
_HYPOTHESIS_TAG_RE = re.compile(r"<hypothesis>(.*?)</hypothesis>", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ITEM_SEP_RE = re.compile(r"[,，\s]+")
TRIPLE_RANGE: Tuple[int, int] = (-20, 20)
MAX_SAMPLE_ATTEMPTS = 5000


def _ordered_thread_map(func: Any, items: List[Any], max_workers: int) -> List[Any]:
    if not items:
        return []
    if max_workers <= 1 or len(items) == 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
        return list(executor.map(func, items))


def _stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def _build_sampling_kwargs(
    *,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    presence_penalty: Optional[float],
    repetition_penalty: Optional[float],
    max_tokens: int,
    eos_token_id: Optional[int],
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"max_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty
    if eos_token_id is not None:
        kwargs["stop_token_ids"] = [eos_token_id]
    return kwargs


def parse_agent_output(text: str) -> Optional[List[str]]:
    return parse_agent_output_by_mode(text, parse_mode="json")


def parse_agent_output_by_mode(text: str, parse_mode: str) -> Optional[List[str]]:
    if not isinstance(text, str) or not text:
        return None
    if parse_mode == "json":
        try:
            parsed_json = json.loads(text)
        except Exception:
            return None
        hypotheses = parsed_json.get("hypotheses")
        if isinstance(hypotheses, list):
            out: List[str] = []
            seen: Set[str] = set()
            for item in hypotheses:
                if isinstance(item, str):
                    rule_id = item.strip()
                    if rule_id and rule_id not in seen:
                        seen.add(rule_id)
                        out.append(rule_id)
            return out
        if hypotheses is None:
            return []
        return None

    if parse_mode == "tag":
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

    raise ValueError(f"Unsupported parse_mode: {parse_mode}")


def normalize_hypotheses(raw: Optional[List[str]], candidates: List[str]) -> Set[str]:
    candidate_set = set(candidates)
    if not raw:
        return set()
    return {name for name in raw if name in candidate_set}


def candidate_rules_map(candidate_names: List[str], heldout_set: str) -> Dict[str, Any]:
    heldout_rules = resolve_heldout_rules(heldout_set)
    return {name: heldout_rules[name] for name in candidate_names}


def oracle_answer(oracle: str, triple: Tuple[int, int, int]) -> str:
    return "YES" if get_rule(oracle).validate(triple) else "NO"


def compute_golden_hypotheses(rules: Dict[str, Any], evidence: List[Tuple[Tuple[int, int, int], str]]) -> Set[str]:
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


def _random_triple(rng: random.Random) -> Tuple[int, int, int]:
    lo, hi = TRIPLE_RANGE
    return (rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi))


def _generate_partial_convergence_prefix(
    oracle: str,
    rng: random.Random,
    candidate_names: List[str],
    heldout_set: str,
) -> Optional[Tuple[Tuple[int, int, int], List[Tuple[Tuple[int, int, int], str]], Set[str]]]:
    rules = candidate_rules_map(candidate_names, heldout_set)
    oracle_rule = rules[oracle]
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        triple = _random_triple(rng)
        result = "YES" if oracle_rule.validate(triple) else "NO"
        survivors = compute_golden_hypotheses(rules, [(triple, result)])
        if oracle in survivors and 2 <= len(survivors) <= 3:
            return triple, [(triple, result)], survivors
    return None


def generate_failed_update_sequence_heldout(
    oracle: str,
    rng: random.Random,
    candidate_names: List[str],
    heldout_set: str,
) -> Optional[Dict[str, Any]]:
    rules = candidate_rules_map(candidate_names, heldout_set)
    oracle_rule = rules[oracle]
    prefix = _generate_partial_convergence_prefix(oracle, rng, candidate_names, heldout_set)
    if prefix is None:
        return None
    prefix_triple, prefix_evidence, prefix_survivors = prefix
    misrecord_triple = None
    wrong_recorded = None
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        triple = _random_triple(rng)
        true_result = "YES" if oracle_rule.validate(triple) else "NO"
        flipped = "NO" if true_result == "YES" else "YES"
        survivors = compute_golden_hypotheses(rules, prefix_evidence + [(triple, flipped)])
        if oracle not in survivors and len(survivors) >= 1:
            misrecord_triple = triple
            wrong_recorded = flipped
            break
    if misrecord_triple is None or wrong_recorded is None:
        return None
    active_evidence = list(prefix_evidence)
    ground_truth: List[Dict[str, Any]] = []
    ground_truth.append(
        {"event_type": "evidence", "triple": list(prefix_triple), "result": prefix_evidence[0][1], "survivors": sorted(prefix_survivors), "turn": 0}
    )
    active_evidence.append((misrecord_triple, wrong_recorded))
    wrong_survivors = compute_golden_hypotheses(rules, active_evidence)
    ground_truth.append(
        {"event_type": "evidence", "triple": list(misrecord_triple), "result": wrong_recorded, "survivors": sorted(wrong_survivors), "turn": 1}
    )
    active_evidence.pop()
    corrected_result = "YES" if oracle_rule.validate(misrecord_triple) else "NO"
    active_evidence.append((misrecord_triple, corrected_result))
    corrected_survivors = compute_golden_hypotheses(rules, active_evidence)
    ground_truth.append(
        {
            "event_type": "retraction",
            "triple": list(misrecord_triple),
            "result": corrected_result,
            "survivors": sorted(corrected_survivors),
            "retract_turn": 1,
            "reinstated_rules": sorted(corrected_survivors - wrong_survivors),
            "pre_retraction_survivors": sorted(wrong_survivors),
            "turn": 2,
        }
    )
    return {
        "challenge_type": "heldout_failed_update",
        "oracle": oracle,
        "events": [
            {"type": "evidence", "triple": list(prefix_triple)},
            {"type": "evidence", "triple": list(misrecord_triple), "recorded_result": wrong_recorded},
            {"type": "retraction", "retract_turn": 1, "new_triple": list(misrecord_triple)},
        ],
        "triples": None,
        "ground_truth": ground_truth,
        "challenge_turns": [2],
        "convergence_turn": 0,
        "total_turns": 3,
        "prefix_survivors": sorted(prefix_survivors),
    }


def generate_failed_stay_sequence_heldout(
    oracle: str,
    rng: random.Random,
    candidate_names: List[str],
    heldout_set: str,
) -> Optional[Dict[str, Any]]:
    rules = candidate_rules_map(candidate_names, heldout_set)
    oracle_rule = rules[oracle]
    prefix = _generate_partial_convergence_prefix(oracle, rng, candidate_names, heldout_set)
    if prefix is None:
        return None
    prefix_triple, prefix_evidence, prefix_survivors = prefix
    disc_triple = None
    disc_evidence = None
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        triple = _random_triple(rng)
        result = "YES" if oracle_rule.validate(triple) else "NO"
        evidence = prefix_evidence + [(triple, result)]
        survivors = compute_golden_hypotheses(rules, evidence)
        if survivors == {oracle}:
            disc_triple = triple
            disc_evidence = evidence
            break
    if disc_triple is None or disc_evidence is None:
        return None
    ambig_triple = None
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        triple = _random_triple(rng)
        oracle_result = oracle_rule.validate(triple)
        agreeing = [name for name, rule in rules.items() if name != oracle and rule.validate(triple) == oracle_result]
        if len(agreeing) < 1:
            continue
        evidence = disc_evidence + [(triple, "YES" if oracle_result else "NO")]
        survivors = compute_golden_hypotheses(rules, evidence)
        if survivors == {oracle}:
            ambig_triple = triple
            break
    if ambig_triple is None:
        return None
    return {
        "challenge_type": "heldout_failed_stay",
        "oracle": oracle,
        "events": [
            {"type": "evidence", "triple": list(prefix_triple)},
            {"type": "evidence", "triple": list(disc_triple)},
            {"type": "evidence", "triple": list(ambig_triple)},
        ],
        "triples": None,
        "ground_truth": [
            {"event_type": "evidence", "triple": list(prefix_triple), "result": prefix_evidence[0][1], "survivors": sorted(prefix_survivors), "turn": 0},
            {"event_type": "evidence", "triple": list(disc_triple), "result": disc_evidence[1][1], "survivors": [oracle], "turn": 1},
            {"event_type": "evidence", "triple": list(ambig_triple), "result": "YES" if oracle_rule.validate(ambig_triple) else "NO", "survivors": [oracle], "turn": 2},
        ],
        "challenge_turns": [2],
        "convergence_turn": 1,
        "total_turns": 3,
        "prefix_survivors": sorted(prefix_survivors),
    }


def apply_event_update(
    *,
    oracle: str,
    event_idx: int,
    event: Dict[str, Any],
    active_evidence: List[Tuple[Tuple[int, int, int], str]],
    event_to_active: Dict[int, int],
) -> Tuple[str, Tuple[int, int, int], str, str]:
    event_type = event["type"]
    if event_type == "evidence":
        triple = tuple(event["triple"])
        result = event.get("recorded_result") or oracle_answer(oracle, triple)
        event_to_active[event_idx] = len(active_evidence)
        active_evidence.append((triple, result))
        return "", triple, result, "evidence"
    if event_type == "retraction":
        retract_turn = int(event["retract_turn"])
        if retract_turn not in event_to_active:
            raise RuntimeError(f"Invalid retract_turn={retract_turn}")
        remove_idx = event_to_active[retract_turn]
        removed_triple, _ = active_evidence.pop(remove_idx)
        for k, v in list(event_to_active.items()):
            if v > remove_idx:
                event_to_active[k] = v - 1
        use_retracted_triple = bool(event.get("use_retracted_triple", False))
        triple = removed_triple if use_retracted_triple else tuple(event["new_triple"])
        result = event.get("new_result") or oracle_answer(oracle, triple)
        event_to_active[event_idx] = len(active_evidence)
        active_evidence.append((triple, result))
        return "", triple, result, "retraction"
    raise ValueError(f"Unknown event type: {event_type}")


def classify_belief_trajectory(trajectory: Dict[str, Any]) -> str:
    turns = trajectory.get("turns", [])
    if len(turns) < 2:
        return "insufficient_capability"
    for turn in turns[:-1]:
        if set(turn.get("hypotheses") or []) != set(turn.get("gt_survivors") or []):
            return "insufficient_capability"
    if set(turns[-1].get("hypotheses") or []) == set(turns[-1].get("gt_survivors") or []):
        return "oracle_match"
    return "belief_failure"


def aggregate_repeat_categories(per_run_categories: List[str], threshold: float = 0.5) -> str:
    if not per_run_categories:
        return "insufficient_capability"
    from collections import Counter
    counter = Counter(per_run_categories)
    most_common_cat, most_common_count = counter.most_common(1)[0]
    if most_common_count / len(per_run_categories) > threshold:
        return most_common_cat
    return "unstable"


def build_system_prompt_sample(
    candidate_names: List[str],
    output_format: str,
    *,
    include_rule_predictions: bool = True,
) -> str:
    lines = []
    for name in candidate_names:
        lines.append(f'- "{name}": {get_rule(name).description}')
    rules_block = "\n".join(lines)
    if output_format == "json":
        output_block = (
            "Output strict JSON only:\n"
            "{\n"
            '  "hypotheses": ["rule_id_1", "rule_id_2"]\n'
            "}\n\n"
            "- No markdown. No prose outside JSON.\n"
        )
    elif output_format == "tag":
        output_block = (
            "Output format (strict):\n"
            "<hypothesis>rule_id_1, rule_id_2</hypothesis>\n\n"
            "- Inside `<hypothesis>`: comma-separated rule IDs that are still consistent with ALL active evidence.\n"
            "- Do not output anything outside the `<hypothesis>` tag.\n"
        )
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")
    if include_rule_predictions:
        turn_input_block = (
            "Each turn, the host provides:\n"
            "1. A triple and its currently recorded result: YES or NO.\n"
            "2. For non-corrected turns, what each candidate rule predicts for that triple.\n\n"
        )
    else:
        turn_input_block = (
            "Each turn, the host provides a triple and its currently recorded result: YES or NO.\n\n"
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
        f"{rules_block}\n\n"
        f"{output_block}"
    )


def _format_rule_predictions(candidate_names: List[str], triple: Tuple[int, int, int], result: str) -> str:
    lines = []
    _ = result
    for name in candidate_names:
        pred_yes = get_rule(name).validate(triple)
        pred = "YES" if pred_yes else "NO"
        lines.append(f"  - {name} → {pred}")
    return "\n".join(lines)


def build_initial_user_message_sample(
    candidate_names: List[str],
    triple: Tuple[int, int, int],
    result: str,
    *,
    include_rule_predictions: bool = True,
) -> str:
    body = (
        "Let's begin.\n\n"
        "**Turn 0 evidence:**\n"
        f"Triple ({triple[0]}, {triple[1]}, {triple[2]}): **{result}**\n\n"
    )
    if include_rule_predictions:
        body += (
            "Rule predictions for this triple:\n"
            f"{_format_rule_predictions(candidate_names, triple, result)}\n\n"
        )
    body += "Please update your hypotheses based on this evidence."
    return body


def build_turn_user_message_sample(
    *,
    candidate_names: List[str],
    turn: int,
    triple: Tuple[int, int, int],
    result: str,
    final_turn: bool,
    is_retraction: bool,
    retract_turn: Optional[int],
    include_rule_predictions: bool = True,
) -> str:
    if is_retraction:
        if retract_turn is None:
            raise ValueError("retract_turn is required for retraction prompts")
        header = (
            f"**Turn {turn} evidence:**\n"
            f"CORRECTION for Turn {retract_turn}:\n"
            f"Triple ({triple[0]}, {triple[1]}, {triple[2]}): **{result}**\n\n"
        )
    else:
        header = f"**Turn {turn} evidence:**\n"

    triple_line = (
        ""
        if is_retraction
        else f"Triple ({triple[0]}, {triple[1]}, {triple[2]}): **{result}**\n\n"
    )
    body = f"{header}{triple_line}"
    if include_rule_predictions:
        body += (
            "Rule predictions for this triple:\n"
            f"{_format_rule_predictions(candidate_names, triple, result)}\n\n"
        )
    body += "Please update your hypotheses using all currently active evidence."
    return body


def _include_rule_predictions_for_turn(
    *,
    base_include_rule_predictions: bool,
    mode: str,
    final_turn: bool,
    event_type: str,
) -> bool:
    if not base_include_rule_predictions:
        return False
    if mode == "failed_update" and final_turn and event_type == "retraction":
        return False
    return True


def run_sequence_with_local_model_sample_prompt(
    *,
    backend: Any,
    tokenizer: Any,
    sampling_params: Any,
    oracle: str,
    candidate_names: List[str],
    heldout_set: str,
    sequence: Dict[str, Any],
    parse_mode: str,
    include_rule_predictions: bool = True,
    mode: str = "failed_update",
) -> Dict[str, Any]:
    rules = candidate_rules_map(candidate_names, heldout_set)
    system_prompt = build_system_prompt_sample(
        candidate_names,
        parse_mode,
        include_rule_predictions=include_rule_predictions,
    )
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    active_evidence: List[Tuple[Tuple[int, int, int], str]] = []
    event_to_active: Dict[int, int] = {}
    turns: List[Dict[str, Any]] = []

    for turn_idx, event in enumerate(sequence["events"]):
        prompt, triple, result, event_type = apply_event_update(
            oracle=oracle,
            event_idx=turn_idx,
            event=event,
            active_evidence=active_evidence,
            event_to_active=event_to_active,
        )
        _ = prompt
        golden = compute_golden_hypotheses(rules, active_evidence)
        final_turn = turn_idx == len(sequence["events"]) - 1
        turn_include_rule_predictions = _include_rule_predictions_for_turn(
            base_include_rule_predictions=include_rule_predictions,
            mode=mode,
            final_turn=final_turn,
            event_type=event_type,
        )
        if turn_idx == 0:
            user_content = build_initial_user_message_sample(
                candidate_names,
                triple,
                result,
                include_rule_predictions=turn_include_rule_predictions,
            )
        else:
            user_content = build_turn_user_message_sample(
                candidate_names=candidate_names,
                turn=turn_idx,
                triple=triple,
                result=result,
                final_turn=final_turn,
                is_retraction=(event_type == "retraction"),
                retract_turn=event.get("retract_turn"),
                include_rule_predictions=turn_include_rule_predictions,
            )
        messages.append({"role": "user", "content": user_content})
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = backend.llm.generate(
            [rendered],
            sampling_params,
            lora_request=getattr(backend, "_lora_request", None),
            use_tqdm=False,
        )[0].outputs[0].text
        parsed = parse_agent_output_by_mode(response, parse_mode)
        model_hyp = normalize_hypotheses(parsed, candidate_names)
        turns.append(
            {
                "turn": turn_idx,
                "event_type": event_type,
                "host_triple": list(triple),
                "host_result": result,
                "agent_response": response,
                "hypotheses": sorted(model_hyp),
                "gt_survivors": sorted(golden),
                "parse_error": parsed is None,
            }
        )
        messages.append({"role": "assistant", "content": response})

    return {
        "oracle": oracle,
        "candidate_rules": candidate_names,
        "include_rule_predictions": include_rule_predictions,
        "mode": mode,
        "challenge_sequence": sequence,
        "turns": turns,
        "conversation": messages,
    }


def _generate_fixed_case_bank(args: argparse.Namespace, candidate_names: List[str]) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    total_cases = len(candidate_names) * args.num_runs
    with tqdm(total=total_cases, desc="generate cases", unit="case") as pbar:
        for rule_name in candidate_names:
            completed = 0
            max_seq_failures = args.num_runs * 20
            max_attempts = args.num_runs + max_seq_failures
            seed_base = args.seed * 1000003 + _stable_int(f"{rule_name}:{args.mode}:{args.heldout_set}")

            def generate_candidate(seq_idx: int) -> Tuple[int, Optional[Dict[str, Any]]]:
                rng = random.Random(seed_base + seq_idx)
                if args.mode == "failed_stay":
                    seq = generate_failed_stay_sequence_heldout(
                        rule_name,
                        rng,
                        candidate_names,
                        args.heldout_set,
                    )
                else:
                    seq = generate_failed_update_sequence_heldout(
                        rule_name,
                        rng,
                        candidate_names,
                        args.heldout_set,
                    )
                return seq_idx, seq

            next_seq_idx = 0
            chunk_size = max(1, args.preprocess_workers * 8)

            def consume_candidates(candidate_iter: Any) -> None:
                nonlocal completed
                for _seq_idx, seq in candidate_iter:
                    if completed >= args.num_runs:
                        break
                    if seq is None:
                        continue
                    exp_id = f"heldout_sample_prompt_{args.mode}_{rule_name}_{completed}"
                    cases.append(
                        {
                            "experiment_id": exp_id,
                            "rule_name": rule_name,
                            "challenge_sequence": seq,
                        }
                    )
                    completed += 1
                    pbar.update(1)

            if args.preprocess_workers <= 1:
                while completed < args.num_runs and next_seq_idx < max_attempts:
                    seq_ids = list(
                        range(next_seq_idx, min(max_attempts, next_seq_idx + chunk_size))
                    )
                    next_seq_idx += len(seq_ids)
                    consume_candidates(map(generate_candidate, seq_ids))
            else:
                with ThreadPoolExecutor(max_workers=args.preprocess_workers) as executor:
                    while completed < args.num_runs and next_seq_idx < max_attempts:
                        seq_ids = list(
                            range(next_seq_idx, min(max_attempts, next_seq_idx + chunk_size))
                        )
                        next_seq_idx += len(seq_ids)
                        consume_candidates(executor.map(generate_candidate, seq_ids))
    return cases


def _run_batch_inference(
    *,
    backend: Any,
    tokenizer: Any,
    sampling_params: Any,
    cases: List[Dict[str, Any]],
    candidate_names: List[str],
    heldout_set: str,
    parse_mode: str,
    include_rule_predictions: bool,
    mode: str,
    preprocess_workers: int,
) -> Dict[str, Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for case in cases:
        for rep in range(case["repeats"]):
            expanded.append(
                {
                    "case_id": case["experiment_id"],
                    "repeat_index": rep,
                    "rule_name": case["rule_name"],
                    "include_rule_predictions": include_rule_predictions,
                    "mode": mode,
                    "challenge_sequence": json.loads(json.dumps(case["challenge_sequence"])),
                }
            )

    rules_map = candidate_rules_map(candidate_names, heldout_set)
    system_prompt = build_system_prompt_sample(
        candidate_names,
        parse_mode,
        include_rule_predictions=include_rule_predictions,
    )

    def build_t0_context(row: Dict[str, Any]) -> Dict[str, Any]:
        seq = row["challenge_sequence"]
        first_prompt, first_triple, first_result, _ = apply_event_update(
            oracle=row["rule_name"],
            event_idx=0,
            event=seq["events"][0],
            active_evidence=[],
            event_to_active={},
        )
        _ = first_prompt
        t0_include_rule_predictions = _include_rule_predictions_for_turn(
            base_include_rule_predictions=include_rule_predictions,
            mode=mode,
            final_turn=False,
            event_type="evidence",
        )
        return {
            "system_prompt": system_prompt,
            "user_prompt": build_initial_user_message_sample(
                candidate_names,
                first_triple,
                first_result,
                include_rule_predictions=t0_include_rule_predictions,
            ),
            "active_evidence": [(first_triple, first_result)],
            "event_to_active": {0: 0},
        }

    t0_contexts = _ordered_thread_map(build_t0_context, expanded, preprocess_workers)

    def render_t0_prompt(ctx: Dict[str, Any]) -> str:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ctx["system_prompt"]},
                {"role": "user", "content": ctx["user_prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    prompts_t0 = _ordered_thread_map(render_t0_prompt, t0_contexts, preprocess_workers)
    outputs_t0 = backend.llm.generate(prompts_t0, sampling_params, lora_request=getattr(backend, "_lora_request", None), use_tqdm=True)
    responses_t0 = [o.outputs[0].text for o in outputs_t0]

    def build_t1_payload(idx: int) -> Dict[str, Any]:
        row = expanded[idx]
        seq = row["challenge_sequence"]
        active_evidence = list(t0_contexts[idx]["active_evidence"])
        event_to_active = dict(t0_contexts[idx]["event_to_active"])
        _, triple, result, event_type = apply_event_update(
            oracle=row["rule_name"],
            event_idx=1,
            event=seq["events"][1],
            active_evidence=active_evidence,
            event_to_active=event_to_active,
        )
        turn_include_rule_predictions = _include_rule_predictions_for_turn(
            base_include_rule_predictions=include_rule_predictions,
            mode=mode,
            final_turn=False,
            event_type=event_type,
        )
        user_prompt = build_turn_user_message_sample(
            candidate_names=candidate_names,
            turn=1,
            triple=triple,
            result=result,
            final_turn=False,
            is_retraction=(event_type == "retraction"),
            retract_turn=seq["events"][1].get("retract_turn"),
            include_rule_predictions=turn_include_rule_predictions,
        )
        return {
            "messages": [
                {"role": "system", "content": t0_contexts[idx]["system_prompt"]},
                {"role": "user", "content": t0_contexts[idx]["user_prompt"]},
                {"role": "assistant", "content": responses_t0[idx]},
                {"role": "user", "content": user_prompt},
            ],
            "golden": sorted(compute_golden_hypotheses(rules_map, active_evidence)),
            "active_evidence": active_evidence,
            "event_to_active": event_to_active,
            "user_prompt": user_prompt,
        }

    t1_payloads = _ordered_thread_map(build_t1_payload, list(range(len(expanded))), preprocess_workers)
    t1_messages = [payload["messages"] for payload in t1_payloads]
    t1_golden = [payload["golden"] for payload in t1_payloads]
    for idx, payload in enumerate(t1_payloads):
        expanded[idx]["_active_evidence_t1"] = payload["active_evidence"]
        expanded[idx]["_event_to_active_t1"] = payload["event_to_active"]
        expanded[idx]["_t1_user_prompt"] = payload["user_prompt"]

    prompts_t1 = _ordered_thread_map(
        lambda m: tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True),
        t1_messages,
        preprocess_workers,
    )
    outputs_t1 = backend.llm.generate(prompts_t1, sampling_params, lora_request=getattr(backend, "_lora_request", None), use_tqdm=True)
    responses_t1 = [o.outputs[0].text for o in outputs_t1]

    def build_t2_payload(idx: int) -> Dict[str, Any]:
        row = expanded[idx]
        seq = row["challenge_sequence"]
        active_evidence = list(row["_active_evidence_t1"])
        event_to_active = dict(row["_event_to_active_t1"])
        _, triple, result, event_type = apply_event_update(
            oracle=row["rule_name"],
            event_idx=2,
            event=seq["events"][2],
            active_evidence=active_evidence,
            event_to_active=event_to_active,
        )
        turn_include_rule_predictions = _include_rule_predictions_for_turn(
            base_include_rule_predictions=include_rule_predictions,
            mode=mode,
            final_turn=True,
            event_type=event_type,
        )
        user_prompt = build_turn_user_message_sample(
            candidate_names=candidate_names,
            turn=2,
            triple=triple,
            result=result,
            final_turn=True,
            is_retraction=(event_type == "retraction"),
            retract_turn=seq["events"][2].get("retract_turn"),
            include_rule_predictions=turn_include_rule_predictions,
        )
        return {
            "messages": t1_messages[idx]
            + [
                {"role": "assistant", "content": responses_t1[idx]},
                {"role": "user", "content": user_prompt},
            ],
            "golden": sorted(compute_golden_hypotheses(rules_map, active_evidence)),
            "user_prompt": user_prompt,
        }

    t2_payloads = _ordered_thread_map(build_t2_payload, list(range(len(expanded))), preprocess_workers)
    t2_messages = [payload["messages"] for payload in t2_payloads]
    t2_golden = [payload["golden"] for payload in t2_payloads]
    for idx, payload in enumerate(t2_payloads):
        expanded[idx]["_t2_user_prompt"] = payload["user_prompt"]

    prompts_t2 = _ordered_thread_map(
        lambda m: tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True),
        t2_messages,
        preprocess_workers,
    )
    outputs_t2 = backend.llm.generate(prompts_t2, sampling_params, lora_request=getattr(backend, "_lora_request", None), use_tqdm=True)
    responses_t2 = [o.outputs[0].text for o in outputs_t2]

    grouped: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(expanded):
        case_id = row["case_id"]
        seq = row["challenge_sequence"]
        gt0 = seq["ground_truth"][0]["survivors"]
        gt1 = t1_golden[idx]
        gt2 = t2_golden[idx]
        trajectory = {
            "oracle": row["rule_name"],
            "candidate_rules": candidate_names,
            "include_rule_predictions": include_rule_predictions,
            "mode": mode,
            "challenge_sequence": seq,
            "turns": [
                {
                    "turn": 0,
                    "event_type": seq["ground_truth"][0]["event_type"],
                    "host_triple": seq["ground_truth"][0]["triple"],
                    "host_result": seq["ground_truth"][0]["result"],
                    "agent_response": responses_t0[idx],
                    "hypotheses": sorted(normalize_hypotheses(parse_agent_output_by_mode(responses_t0[idx], parse_mode), candidate_names)),
                    "gt_survivors": gt0,
                    "parse_error": parse_agent_output_by_mode(responses_t0[idx], parse_mode) is None,
                },
                {
                    "turn": 1,
                    "event_type": seq["ground_truth"][1]["event_type"],
                    "host_triple": seq["ground_truth"][1]["triple"],
                    "host_result": seq["ground_truth"][1]["result"],
                    "agent_response": responses_t1[idx],
                    "hypotheses": sorted(normalize_hypotheses(parse_agent_output_by_mode(responses_t1[idx], parse_mode), candidate_names)),
                    "gt_survivors": gt1,
                    "parse_error": parse_agent_output_by_mode(responses_t1[idx], parse_mode) is None,
                },
                {
                    "turn": 2,
                    "event_type": seq["ground_truth"][2]["event_type"],
                    "host_triple": seq["ground_truth"][2]["triple"],
                    "host_result": seq["ground_truth"][2]["result"],
                    "agent_response": responses_t2[idx],
                    "hypotheses": sorted(normalize_hypotheses(parse_agent_output_by_mode(responses_t2[idx], parse_mode), candidate_names)),
                    "gt_survivors": gt2,
                    "parse_error": parse_agent_output_by_mode(responses_t2[idx], parse_mode) is None,
                },
            ],
            "conversation": t2_messages[idx] + [{"role": "assistant", "content": responses_t2[idx]}],
        }
        category = classify_belief_trajectory(trajectory)
        bucket = grouped.setdefault(
            case_id,
            {
                "experiment_id": case_id,
                "rule_name": row["rule_name"],
                "include_rule_predictions": include_rule_predictions,
                "mode": mode,
                "per_run_categories": [],
                "repeat_trajectories": [],
                "challenge_sequence": seq,
            },
        )
        bucket["per_run_categories"].append(category)
        bucket["repeat_trajectories"].append(
            {
                "repeat_index": row["repeat_index"],
                "category": category,
                "trajectory": trajectory,
            }
        )
    return grouped


def _print_and_save_report(
    *,
    args: argparse.Namespace,
    model_name: str,
    candidate_names: List[str],
    gpu_ids: List[int],
    all_stats: Dict[str, Dict[str, int]],
    all_results: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    from utils.io import save_json

    print(f"\n{'=' * 60}")
    print(
        "HELDOUT SAMPLE PROMPT REPORT  "
        f"(mode: {args.mode}, heldout_set: {args.heldout_set}, "
        f"include_rule_predictions: {args.include_rule_predictions}, model: {model_name})"
    )
    print(f"{'=' * 60}")

    col_w = max([len(rule_name) for rule_name in candidate_names] or [4]) + 2
    header = f"{'Rule':<{col_w}}"
    for category in CATEGORIES:
        header += f"  {category:>12}"
    print(header)
    print("-" * len(header))

    total_counts = {category: 0 for category in CATEGORIES}
    for rule_name in candidate_names:
        counts = all_stats.get(rule_name, {category: 0 for category in CATEGORIES})
        n_total = sum(counts.values())
        row = f"{rule_name:<{col_w}}"
        for category in CATEGORIES:
            n = counts[category]
            pct = n / n_total * 100 if n_total else 0.0
            row += f"  {n:>4} ({pct:>5.1f}%)"
            total_counts[category] += n
        print(row)

    grand_total = sum(total_counts.values())
    print("-" * len(header))
    row = f"{'TOTAL':<{col_w}}"
    for category in CATEGORIES:
        n = total_counts[category]
        pct = n / grand_total * 100 if grand_total else 0.0
        row += f"  {n:>4} ({pct:>5.1f}%)"
    print(row)

    report = {
        "mode": args.mode,
        "model": model_name,
        "candidate_rules": candidate_names,
        "heldout_set": args.heldout_set,
        "include_rule_predictions": args.include_rule_predictions,
        "num_runs_per_rule": args.num_runs,
        "repeats": args.repeats,
        "preprocess_workers": args.preprocess_workers,
        "gpus": gpu_ids,
        "per_rule": {
            rule_name: {
                "total": sum(all_stats[rule_name].values()),
                **{category: all_stats[rule_name][category] for category in CATEGORIES},
                **{
                    f"{category}_pct": round(
                        all_stats[rule_name][category]
                        / max(sum(all_stats[rule_name].values()), 1)
                        * 100,
                        2,
                    )
                    for category in CATEGORIES
                },
            }
            for rule_name in candidate_names
            if rule_name in all_stats
        },
        "total": {
            "total": grand_total,
            **{category: total_counts[category] for category in CATEGORIES},
            **{
                f"{category}_pct": round(
                    total_counts[category] / max(grand_total, 1) * 100,
                    2,
                )
                for category in CATEGORIES
            },
        },
    }
    save_json(os.path.join(output_dir, "stats_report.json"), report)
    save_json(os.path.join(output_dir, "all_results.json"), all_results)
    print(f"\nOutputs saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heldout-rule belief statistics with sample-aligned prompt")
    parser.add_argument("--mode", type=str, required=True, choices=["failed_update", "failed_stay"])
    parser.add_argument(
        "--agent-model-path",
        type=str,
        default=os.environ.get("AGENT_MODEL_PATH", "models/Qwen3-30B-A3B-Instruct-2507"),
    )
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-runs", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rules", nargs="+", default=None)
    parser.add_argument("--heldout-set", type=str, default="easy", choices=["easy", "hard"])
    parser.set_defaults(include_rule_predictions=True)
    parser.add_argument(
        "--include-rule-predictions",
        dest="include_rule_predictions",
        action="store_true",
        help="Include per-rule prediction lists in each user prompt. This is the default.",
    )
    parser.add_argument(
        "--no-include-rule-predictions",
        dest="include_rule_predictions",
        action="store_false",
        help="Provide evidence only in each user prompt.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent-temperature", "--temperature", dest="agent_temperature", type=float, default=0.3)
    parser.add_argument("--sampling-top-p", "--top-p", dest="sampling_top_p", type=float, default=None)
    parser.add_argument("--sampling-top-k", "--top-k", dest="sampling_top_k", type=int, default=None)
    parser.add_argument(
        "--sampling-presence-penalty",
        "--presence-penalty",
        dest="sampling_presence_penalty",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--sampling-repetition-penalty",
        "--repetition-penalty",
        dest="sampling_repetition_penalty",
        type=float,
        default=None,
    )
    parser.add_argument("--vllm-max-model-len", "--max-model-len", dest="vllm_max_model_len", type=int, default=None)
    parser.add_argument("--agent-max-tokens", "--max-output-tokens", dest="agent_max_tokens", type=int, default=512)
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=1,
        help="Thread workers for heldout sequence generation and prompt batch construction.",
    )
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--output-format", type=str, default="tag", choices=["tag", "json"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preprocess_workers < 1:
        raise ValueError("--preprocess-workers must be >= 1")
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    candidate_names = args.rules or list(resolve_heldout_rules(args.heldout_set).keys())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(args.agent_model_path)
    prediction_suffix = "with_predictions" if args.include_rule_predictions else "evidence_only"
    output_dir = args.output_dir or os.path.join(
        "task_a/outputs",
        f"heldout_sample_prompt_{args.mode}_{prediction_suffix}_stats_{model_name}_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)

    category_dirs = {
        "insufficient_capability": os.path.join(output_dir, "insufficient_capability"),
        "oracle_match": os.path.join(output_dir, "oracle_match"),
        "belief_failure": os.path.join(output_dir, "belief_failure"),
        "unstable": os.path.join(output_dir, "unstable"),
    }
    for directory in category_dirs.values():
        os.makedirs(directory, exist_ok=True)
    print(f"Preprocess workers: {args.preprocess_workers}", flush=True)
    cases = _generate_fixed_case_bank(args, candidate_names)

    from utils.llm_backend import VLLMBackend
    from utils.io import save_json

    backend = VLLMBackend(
        model_path=args.agent_model_path,
        dtype="bf16",
        max_model_len=args.vllm_max_model_len,
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    tokenizer = backend.llm.get_tokenizer()
    sampling_params = backend.SamplingParams(
        **_build_sampling_kwargs(
            temperature=args.agent_temperature,
            top_p=args.sampling_top_p,
            top_k=args.sampling_top_k,
            presence_penalty=args.sampling_presence_penalty,
            repetition_penalty=args.sampling_repetition_penalty,
            max_tokens=args.agent_max_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
    )

    for case in cases:
        case["repeats"] = args.repeats

    grouped = _run_batch_inference(
        backend=backend,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        cases=cases,
        candidate_names=candidate_names,
        heldout_set=args.heldout_set,
        parse_mode=args.output_format,
        include_rule_predictions=args.include_rule_predictions,
        mode=args.mode,
        preprocess_workers=args.preprocess_workers,
    )

    all_stats: Dict[str, Dict[str, int]] = {}
    all_results: List[Dict[str, Any]] = []
    for case_id, payload in grouped.items():
        category = aggregate_repeat_categories(payload["per_run_categories"])
        payload["category"] = category
        payload["repeats"] = args.repeats
        payload["candidate_rules"] = candidate_names
        payload["include_rule_predictions"] = args.include_rule_predictions
        save_json(os.path.join(category_dirs[category], f"{case_id}.json"), payload)
        all_results.append(
            {
                "experiment_id": case_id,
                "rule_name": payload["rule_name"],
                "category": category,
                "include_rule_predictions": args.include_rule_predictions,
                "per_run_categories": payload["per_run_categories"],
            }
        )
        if payload["rule_name"] not in all_stats:
            all_stats[payload["rule_name"]] = {c: 0 for c in CATEGORIES}
        all_stats[payload["rule_name"]][category] += 1

    _print_and_save_report(
        args=args,
        model_name=model_name,
        candidate_names=candidate_names,
        gpu_ids=gpu_ids,
        all_stats=all_stats,
        all_results=all_results,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
