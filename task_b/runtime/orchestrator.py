"""Game orchestrator for Scenario B circuit diagnosis."""

import re
from typing import Any, Dict, List, Optional, Set

from task_b.domain.rule_engine import get_topology
from task_b.runtime.agent import (
    _format_measurement_label,
    _format_measurement_value,
    build_system_prompt,
    build_turn_message,
)
from task_b.runtime.environment import (
    ChallengeSequence,
    CircuitDiagnosisEnvironment,
    NoiseChallengeSequence,
)
from utils.hypotheses_parser import parse_hypotheses_tag


_MEASURE_TAG_RE = re.compile(r"<measure>\s*([a-zA-Z0-9_\-\s]+?)\s*</measure>", re.IGNORECASE)
# Fuzzy fallback: accept common <measure> tag typos (e.g. <measrue>, <mesaure>)
_MEASURE_TAG_FUZZY_RE = re.compile(r"<me[a-z]{3,7}>\s*([a-zA-Z0-9_\-\s]+?)\s*</me[a-z]{3,7}>", re.IGNORECASE)
_TERMINATE_TAG_RE = re.compile(r"<terminate>\s*final\s*</terminate>", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_HYPOTHESIS_TAG_RE = re.compile(
    r"<hypothesis>\s*(.*?)\s*</hypothesis>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _compact_assistant_history(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Keep prior assistant turns concise when replaying history to the model."""
    compacted: List[Dict[str, str]] = []
    for message in messages:
        if message.get("role") != "assistant":
            compacted.append(message)
            continue

        content = message.get("content", "")
        matches = list(_HYPOTHESIS_TAG_RE.finditer(content))
        if matches:
            compacted.append({
                "role": "assistant",
                "content": " ".join(matches[-1].group(0).split()),
            })
            continue

        compacted.append({"role": "assistant", "content": content.strip()})
    return compacted


def parse_terminate(text: str) -> bool:
    """Return True if the agent response contains <terminate>final</terminate> outside any <think> block."""
    stripped = _THINK_BLOCK_RE.sub("", text or "")
    return bool(_TERMINATE_TAG_RE.search(stripped))


def parse_hypotheses(text: str, valid_faults: Optional[Set[str]] = None) -> Optional[List[str]]:
    """Extract valid fault-ID hypotheses from a `<hypothesis>...</hypothesis>` tag."""
    parsed = parse_hypotheses_tag(text)
    if parsed is None:
        return None

    valid = valid_faults or set(get_topology().fault_ids)
    result: List[str] = []
    seen: set[str] = set()
    for token in parsed:
        fid = token.strip().upper()
        if fid in valid and fid not in seen:
            result.append(fid)
            seen.add(fid)
    return result


def parse_measure_query(text: str) -> Optional[str]:
    """Extract one measurement query key from a `<measure>...</measure>` tag (ignores <think> blocks).

    Falls back to a fuzzy tag regex to handle common typos in the tag name.
    Normalizes the key: lowercase, hyphens/spaces → underscores.
    """
    stripped = _THINK_BLOCK_RE.sub("", text or "")
    match = _MEASURE_TAG_RE.search(stripped)
    if not match:
        match = _MEASURE_TAG_FUZZY_RE.search(stripped)
    if not match:
        return None
    raw = match.group(1).strip()
    # Normalize: lowercase, hyphens/spaces → underscores, collapse duplicates
    key = re.sub(r"[-\s]+", "_", raw.lower())
    key = re.sub(r"_+", "_", key).strip("_")
    return key if key else None


# ---------------------------------------------------------------------------
# FailedIsolation mode templates
# ---------------------------------------------------------------------------
FAILED_ISOLATION_TEMPLATES_TARGET_WRONG = [
    "I'm pretty sure `{wrong_rule}` is still likely.",
    "`{wrong_rule}` seems likely from the measurements so far.",
    "This triple looks especially diagnostic for `{wrong_rule}`.",
    "Patterns like this usually suggest `{wrong_rule}` is more likely.",
    "At this point, `{wrong_rule}` still feels like a strong candidate.",
    "I'd currently lean toward `{wrong_rule}`.",
    "If I had to guess now, I'd pick `{wrong_rule}`.",
    "So far, `{wrong_rule}` remains a plausible fault candidate.",
]

FAILED_ISOLATION_TEMPLATES_EXCLUDE_ORACLE = [
    "`{exclude_rule}` seems less likely at the moment.",
    "I'd be cautious about `{exclude_rule}` for now.",
    "Given the current measurements, `{exclude_rule}` doesn't look like the best fit.",
    "Right now, `{exclude_rule}` appears comparatively unlikely.",
    "I'd tentatively deprioritize `{exclude_rule}`.",
    "From what we've seen so far, `{exclude_rule}` is probably not the fault.",
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
    rng,
    candidate_names,
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


def _build_failed_isolation_clue_sentence(event: Optional[Dict[str, Any]]) -> str:
    """Build the first sentence for failed_isolation-mode turn prompts."""
    measurements = dict((event or {}).get("measurements") or {})
    if not measurements:
        return "Clue: no new measurements."

    pieces = [f"{key}={value}" for key, value in measurements.items()]
    return "Clue: " + ", ".join(pieces) + "."


class CircuitOrchestrator:
    """Run a complete circuit diagnosis session and return the trajectory."""

    def __init__(
        self,
        challenge: ChallengeSequence | NoiseChallengeSequence,
        backend,
        *,
        prompt_style: str = "neutral",
        temperature: float = 0.3,
        max_tokens: int = 512,
        context_turns: Optional[int] = None,
        failed_isolation_options: Optional[Dict[str, Any]] = None,
        include_fault_predictions: bool = False,
        model_type: str = "local",
    ):
        self.challenge = challenge
        self.backend = backend
        self.prompt_style = prompt_style
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_turns = context_turns
        self.failed_isolation_options = failed_isolation_options
        self.include_fault_predictions = include_fault_predictions
        self.model_type = model_type
        self.model_type = model_type

        self.is_noise = getattr(challenge, "mode", None) == "noise"
        self.env = None if self.is_noise else CircuitDiagnosisEnvironment(challenge)
        self.valid_faults = set(get_topology(challenge.circuit_type).fault_ids)

    def fault_predictions_for_measurements(
        self,
        measurements: Dict[str, str],
    ) -> Optional[Dict[str, str]]:
        if not self.include_fault_predictions or not measurements:
            return None

        topology = get_topology(self.challenge.circuit_type)
        predictions: Dict[str, List[str]] = {
            fid: [] for fid in sorted(self.valid_faults)
        }
        for key in measurements:
            presence_key = f"{key}_presence"
            absence_key = f"{key}_absence"
            presence = set(topology.abnormal_support.get(presence_key, ()))
            absence = set(topology.abnormal_support.get(absence_key, ()))
            for fid in predictions:
                if fid in presence and fid in absence:
                    value = "> 0 or = 0"
                elif fid in presence:
                    value = "> 0"
                elif fid in absence:
                    value = "= 0"
                else:
                    value = "not listed"
                predictions[fid].append(f"{_format_measurement_label(key)}: {value}")
        return {fid: "; ".join(parts) for fid, parts in predictions.items()}

    def current_matching_faults_for_measurements(
        self,
        measurements: Dict[str, str],
    ) -> Optional[List[str]]:
        if not self.include_fault_predictions or not measurements:
            return None

        topology = get_topology(self.challenge.circuit_type)
        matched: List[str] = []
        for fid in sorted(self.valid_faults):
            is_match = True
            for key, raw_value in measurements.items():
                formatted_value = _format_measurement_value(key, raw_value)
                presence = set(topology.abnormal_support.get(f"{key}_presence", ()))
                absence = set(topology.abnormal_support.get(f"{key}_absence", ()))
                if formatted_value == "> 0":
                    is_match = fid in presence
                elif formatted_value == "= 0":
                    is_match = fid in absence
                elif formatted_value == "Open":
                    is_match = fid in absence
                elif formatted_value in {"Closed", "Healthy"}:
                    is_match = fid in presence
                else:
                    is_match = False
                if not is_match:
                    break
            if is_match:
                matched.append(fid)
        return matched

    def maybe_inject_failed_isolation_comment(
        self,
        user_msg: str,
        turn_idx: int,
        event: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Rewrite failed_isolation-mode turn prompts to the A-style clue/update format."""
        opts = self.failed_isolation_options
        if not opts:
            return user_msg

        comment_start_turn = int(opts.get("comment_start_turn", 0) or 0)
        if turn_idx < comment_start_turn:
            return user_msg

        preserve_turn_message = bool(opts.get("preserve_turn_message", False))
        lines = [user_msg.strip()] if preserve_turn_message else [_build_failed_isolation_clue_sentence(event)]

        if opts.get("add_failed_isolation_comment", True):
            rng = opts.get("rng")
            oracle = opts.get("oracle") or getattr(self.challenge, "oracle", None)
            candidates = opts.get("candidate_names") or sorted(self.valid_faults)
            if rng is not None and oracle:
                line = build_failed_isolation_comment(
                    rng=rng,
                    candidate_names=candidates,
                    oracle=oracle,
                    turn_idx=turn_idx,
                    model_type=self.model_type,
                )
                if line:
                    prefix = "Host comment: " if preserve_turn_message else ""
                    lines.append(f"{prefix}{line}")

        if not preserve_turn_message:
            lines.append("Please update your hypotheses.")
        return "\n".join(lines)

    def build_context_messages(
        self, conversation: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        context_turns = self.context_turns
        if context_turns is None or context_turns < 0:
            selected = conversation
        else:
            base = conversation[:1]
            rest = conversation[1:]
            if context_turns == 0:
                selected = base + ([rest[-1]] if rest else [])
            else:
                keep_count = context_turns * 2
                if len(rest) > keep_count:
                    rest = rest[-keep_count:]
                selected = base + rest

        # Noise mode: keep full assistant messages (including <think> traces)
        # to avoid stripping reasoning context between turns.
        if self.is_noise:
            return selected
        return _compact_assistant_history(selected)

    def run(self) -> Dict[str, Any]:
        if self.is_noise:
            return self._run_noise()
        return self._run_event_sequence()

    def _run_event_sequence(self) -> Dict[str, Any]:
        conversation: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": build_system_prompt(
                    self.challenge.circuit_type,
                    self.prompt_style,
                    symptom=self.challenge.symptom,
                ),
            },
        ]

        turns: List[Dict[str, Any]] = []

        for turn_idx in range(self.challenge.total_turns):
            is_final = turn_idx == self.challenge.total_turns - 1
            env_record = self.env.step(turn_idx)
            event = env_record["event"]

            user_msg = build_turn_message(
                turn=turn_idx,
                measurements=event["measurements"],
                event_type=event.get("type", "measurement"),
                retract_turn=event.get("retract_turn"),
                final_turn=is_final,
                event=event,
                prompt_style=self.prompt_style,
                fault_predictions=self.fault_predictions_for_measurements(
                    event["measurements"]
                ),
                current_matching_faults=self.current_matching_faults_for_measurements(
                    event["measurements"]
                ),
            )
            user_msg = self.maybe_inject_failed_isolation_comment(user_msg, turn_idx, event)
            conversation.append({"role": "user", "content": user_msg})

            messages = self.build_context_messages(conversation)
            agent_response = self.backend.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            conversation.append({"role": "assistant", "content": agent_response})

            hypotheses = parse_hypotheses(agent_response, self.valid_faults)

            gt = self.env.get_ground_truth_at_step(turn_idx)
            gt_survivors: Set[str] = set(gt["survivors"])
            agent_set: Set[str] = set(hypotheses) if hypotheses else set()
            oracle = self.challenge.oracle

            belief_metrics = {
                "oracle_in_hypotheses": oracle in agent_set,
                "exact_match": agent_set == gt_survivors,
                "false_retention": sorted(agent_set - gt_survivors),
                "false_elimination": sorted(gt_survivors - agent_set),
                "agent_set_size": len(agent_set),
                "gt_set_size": len(gt_survivors),
            }

            turn_record = {
                "turn": turn_idx,
                "agent_response": agent_response,
                "hypotheses": hypotheses,
                "gt_survivors": env_record["gt_survivors"],
                "belief_metrics": belief_metrics,
                "parse_error": hypotheses is None,
            }
            turns.append(turn_record)

            print(
                f"  [Turn {turn_idx}] hyps={sorted(agent_set)}  "
                f"gt={sorted(gt_survivors)}  oracle_in={oracle in agent_set}",
                flush=True,
            )

        return {
            "mode": "failed_isolation" if self.failed_isolation_options else "failed_stay",
            "circuit_type": self.challenge.circuit_type,
            "oracle": self.challenge.oracle,
            "turns": turns,
            "conversation": conversation,
            "n_turns_played": len(turns),
        }

    def _run_noise(self) -> Dict[str, Any]:
        conversation: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": build_system_prompt(
                    self.challenge.circuit_type,
                    self.prompt_style,
                    symptom=getattr(self.challenge, "symptom", None),
                ),
            },
        ]

        turns: List[Dict[str, Any]] = []
        finished = False
        success = False
        termination_reason = "max_turns_exceeded"
        evidence = []  # list of (query_key, answer_yes) for valid queries

        pending_user_msg = build_turn_message(
            turn=0,
            measurements={},
            event_type="noise_start",
            event={"query_space": self.challenge.query_space},
            prompt_style=self.prompt_style,
        )

        for turn_idx in range(self.challenge.max_turns):
            # Dynamic ground truth: survivors consistent with evidence so far
            gt_survivors = self.challenge.compute_survivors(evidence)

            conversation.append({"role": "user", "content": pending_user_msg})
            messages = self.build_context_messages(conversation)
            agent_response = self.backend.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            conversation.append({"role": "assistant", "content": agent_response})

            hypotheses = parse_hypotheses(agent_response, self.valid_faults)
            measure_query = parse_measure_query(agent_response)
            is_terminate = parse_terminate(agent_response)

            turn_record: Dict[str, Any] = {
                "turn": turn_idx,
                "agent_response": agent_response,
                "hypotheses": hypotheses,
                "gt_survivors": gt_survivors,
                "parse_error": hypotheses is None,
            }

            if is_terminate:
                guess = hypotheses[0] if hypotheses else None
                success = hypotheses is not None and len(hypotheses) == 1 and hypotheses[0] == self.challenge.oracle
                termination_reason = "finalized"
                finished = True
                turn_record.update({
                    "action": "finalize_fault",
                    "final_guess": guess,
                    "finalize_success": success,
                })
                turns.append(turn_record)
                break

            if measure_query:
                feedback = self.challenge.evaluate_query(measure_query, turn_idx, gt_survivors=gt_survivors)
                turn_record.update({
                    "action": "ask_measure",
                    "query_key": feedback["query_key"],
                    "query_valid": feedback["query_valid"],
                    "env_answer": feedback["answer"],
                    "host_comment": feedback["host_comment"],
                })
                if feedback["query_valid"]:
                    evidence.append((feedback["query_key"], feedback["answer_yes"]))
                # Compute per-fault predictions for the queried measurement key
                fault_predictions: Optional[Dict[str, str]] = None
                if feedback["query_valid"]:
                    qkey = feedback["query_key"]
                    support_set = self.challenge._support_map.get(qkey, set())
                    fault_predictions = {
                        fid: ("YES" if fid in support_set else "NO")
                        for fid in sorted(self.valid_faults)
                    }
                pending_user_msg = build_turn_message(
                    turn=turn_idx + 1,
                    measurements={},
                    event_type="noise_feedback",
                    event=feedback,
                    prompt_style=self.prompt_style,
                    add_host_comment=self.challenge.add_host_comment,
                    fault_predictions=fault_predictions,
                )
                turns.append(turn_record)
                continue

            # No measure and no terminate: format error — drop this session
            turn_record.update({
                "action": "format_error_dropped",
                "parse_error": True,
                "query_valid": False,
            })
            turns.append(turn_record)
            termination_reason = "format_error_dropped"
            break

        if not finished and termination_reason != "format_error_dropped":
            termination_reason = "max_turns_exceeded"

        return {
            "mode": "noise",
            "circuit_type": self.challenge.circuit_type,
            "oracle": self.challenge.oracle,
            "turns": turns,
            "conversation": conversation,
            "n_turns_played": len(turns),
            "noise_success": success,
            "termination_reason": termination_reason,
        }
