import re
from typing import Any, Dict, List, Optional, Tuple

from task_a.core.rules import HiddenRule, RULES, Triple
from task_a.core.evidence_sequences import (
    RULE_NAMES,
    compute_ground_truth_beliefs,
    compute_retraction_ground_truth,
)

TurnRecord = Dict[str, Any]


def _triple_text(triple: Triple) -> str:
    return f"({triple[0]}, {triple[1]}, {triple[2]})"


def _build_rule_predictions(triple: Triple) -> Dict[str, str]:
    return {
        rule_name: "YES" if RULES[rule_name].validate(triple) else "NO"
        for rule_name in RULE_NAMES
    }


def _build_turn_record(
    *,
    turn: int,
    triple: Triple,
    formal_result: bool,
    gt_survivors: List[str],
    event_type: str | None = None,
) -> TurnRecord:
    record: TurnRecord = {
        "turn": turn,
        "triple": list(triple),
        "formal_result": formal_result,
        "formal_feedback": "YES" if formal_result else "NO",
        "rule_predictions": _build_rule_predictions(triple),
        "gt_survivors": gt_survivors,
    }
    if event_type is not None:
        record["event_type"] = event_type
    return record


def _format_rule_predictions(
    rule_predictions: Dict[str, str],
    result: str,
    label_mode: str,
) -> List[str]:
    lines: List[str] = []
    if label_mode == "grouped":
        yes_rules = [rule_name for rule_name in RULE_NAMES if rule_predictions[rule_name] == "YES"]
        no_rules = [rule_name for rule_name in RULE_NAMES if rule_predictions[rule_name] == "NO"]
        if yes_rules:
            lines.append(f"  Predicting YES: {', '.join(yes_rules)}")
        if no_rules:
            lines.append(f"  Predicting NO: {', '.join(no_rules)}")
        return lines

    for rule_name in RULE_NAMES:
        pred = rule_predictions[rule_name]
        if label_mode == "elimination":
            lines.append(f"  - {rule_name} → {pred}")
        else:
            lines.append(f"  - {rule_name} → predicts {pred}")
    return lines


class Environment:
    """Evidence-driven 2-4-6 game environment.

    In this design the environment *pushes* evidence to the agent (rather than
    the agent choosing triples). The evidence sequence is pre-designed so that
    the theoretical surviving set at each step is known a-priori.
    """

    def __init__(
        self,
        rule: HiddenRule,
        evidence_triples: List[Triple],
    ):
        self.rule = rule
        self.history: List[TurnRecord] = []

        self._evidence_triples = list(evidence_triples)

        # Pre-compute ground truth for each step
        self._ground_truth = compute_ground_truth_beliefs(
            rule.name, self._evidence_triples
        )
        self._current_step = 0

    @property
    def total_evidence_steps(self) -> int:
        return len(self._evidence_triples)

    @property
    def has_more_evidence(self) -> bool:
        return self._current_step < len(self._evidence_triples)

    def step(
        self,
        turn: int,
    ) -> TurnRecord:
        """Push the next piece of evidence to the agent.

        Returns a record with formal result and ground truth.
        """
        if not self.has_more_evidence:
            raise RuntimeError(
                f"No more evidence at step {self._current_step} "
                f"(total {self.total_evidence_steps})"
            )

        triple = self._evidence_triples[self._current_step]
        gt = self._ground_truth[self._current_step]

        formal_result = self.rule.validate(triple)

        record = _build_turn_record(
            turn=turn,
            triple=triple,
            formal_result=formal_result,
            gt_survivors=sorted(gt["survivors"]),
        )
        self.history.append(record)
        self._current_step += 1
        return record

    def format_feedback(
        self, record: TurnRecord,
        label_mode: str = "none",
        include_rule_predictions: bool = True,
    ) -> str:
        """Format environment response as text for the agent."""
        t = record["triple"]
        result = record["formal_feedback"]
        lines = [f"Triple {_triple_text(tuple(t))}: **{result}**"]
        if include_rule_predictions:
            lines.extend([
                "",
                "Rule predictions for this triple:",
            ])
            lines.extend(_format_rule_predictions(record["rule_predictions"], result, label_mode))

        return "\n".join(lines)

    def get_evidence_table(self, example_triple: Optional[Triple] = None) -> str:
        """Build a canonical text evidence table from history."""
        lines = ["| Triple | Result |", "|--------|--------|"]
        if example_triple is not None:
            lines.append(
                f"| {_triple_text(example_triple)} | YES (given) |"
            )
        for rec in self.history:
            t = rec["triple"]
            lines.append(f"| {_triple_text(tuple(t))} | {rec['formal_feedback']} |")
        if len(lines) == 2:
            lines.append("| (no tests yet) | - |")
        return "\n".join(lines)

    def get_ground_truth_at_step(self, step: int) -> Dict:
        """Get ground truth beliefs for a specific evidence step (0-indexed)."""
        if 0 <= step < len(self._ground_truth):
            return self._ground_truth[step]
        raise IndexError(f"Step {step} out of range [0, {len(self._ground_truth)})")


class RetractionEnvironment:
    """Evidence-driven environment with retraction support.

    At a designated turn, a previous recorded result is corrected.
    This can force the agent to revise a belief state that was previously
    supported by the earlier record. This tests belief failed_update.
    """

    def __init__(
        self,
        rule: HiddenRule,
        events: List[Dict],
    ):
        self.rule = rule
        self.history: List[TurnRecord] = []

        self._events = list(events)

        # Pre-compute ground truth for each event
        self._ground_truth = compute_retraction_ground_truth(
            rule.name, self._events
        )
        self._current_step = 0
        # Track which turns have been retracted (for evidence table)
        self._retracted_turns: set = set()

    @property
    def total_evidence_steps(self) -> int:
        return len(self._events)

    @property
    def has_more_evidence(self) -> bool:
        return self._current_step < len(self._events)

    def step(
        self,
        turn: int,
    ) -> TurnRecord:
        """Push the next event (evidence or retraction) to the agent."""
        if not self.has_more_evidence:
            raise RuntimeError(
                f"No more events at step {self._current_step} "
                f"(total {self.total_evidence_steps})"
            )

        event = self._events[self._current_step]
        gt = self._ground_truth[self._current_step]
        is_retraction = event["type"] == "retraction"

        if is_retraction:
            triple = event["new_triple"]
            retract_turn = event["retract_turn"]
            self._retracted_turns.add(retract_turn)
        else:
            triple = event["triple"]
            retract_turn = None

        shown_result = gt["result"]
        formal_result = shown_result == "YES"
        oracle_result = self.rule.validate(triple)

        record = _build_turn_record(
            turn=turn,
            triple=triple,
            formal_result=formal_result,
            gt_survivors=sorted(gt["survivors"]),
            event_type=event["type"],
        )
        if is_retraction:
            record["retract_turn"] = retract_turn
            # Get info about the retracted evidence
            retracted_event = self._events[retract_turn]
            retracted_triple = retracted_event["triple"]
            retracted_result = self._ground_truth[retract_turn]["result"]
            record["retracted_triple"] = list(retracted_triple)
            record["retracted_result"] = retracted_result
            record["reinstated_rules"] = sorted(gt.get("reinstated_rules", set()))
            record["pre_retraction_survivors"] = sorted(gt.get("pre_retraction_survivors", set()))
        elif shown_result != ("YES" if oracle_result else "NO"):
            record["oracle_formal_feedback"] = "YES" if oracle_result else "NO"
            record["is_misrecorded"] = True

        self.history.append(record)
        self._current_step += 1
        return record

    def format_feedback(
        self,
        record: TurnRecord,
        label_mode: str = "match",
        include_rule_predictions: bool = True,
    ) -> str:
        """Format environment response as text for the agent."""
        is_retraction = record.get("event_type") == "retraction"

        t = record["triple"]
        result = record["formal_feedback"]

        # Minimal mode: only triple + YES/NO
        if not include_rule_predictions:
            lines = []
            if is_retraction:
                rt = record["retract_turn"]
                lines.append(
                    f"Correction for Turn {rt}: "
                    f"Triple {_triple_text(tuple(t))}: **{result}**"
                )
            else:
                lines.append(f"Triple {_triple_text(tuple(t))}: **{result}**")
            return "\n".join(lines)

        lines = []

        if is_retraction:
            rt = record["retract_turn"]
            retracted_triple = record["retracted_triple"]
            retracted_result = record["retracted_result"]
            corrected_same_triple = tuple(retracted_triple) == tuple(record["triple"])
            corrected_result = record["formal_feedback"]
            lines.append(
                f"⚠️ CORRECTION: The recorded result from Turn {rt} "
                f"(triple {_triple_text(tuple(retracted_triple))} → "
                f"{retracted_result}) was wrong."
            )
            if corrected_same_triple:
                lines.append(
                    f"The correct result for that same triple is **{corrected_result}**."
                )
            else:
                lines.append("Please disregard that earlier record and use the corrected evidence below.")
            lines.append("")

        if is_retraction:
            if tuple(record["retracted_triple"]) != tuple(t):
                lines.append(f"Replacement test — Triple {_triple_text(tuple(t))}: **{result}**")
        else:
            lines.append(f"Triple {_triple_text(tuple(t))}: **{result}**")

        lines.append("")
        lines.append("Rule predictions for this triple:")
        lines.extend(_format_rule_predictions(record["rule_predictions"], result, label_mode))

        return "\n".join(lines)

    def get_evidence_table(self, example_triple: Optional[Triple] = None) -> str:
        """Build evidence table with retracted rows struck through."""
        lines = ["| Triple | Result | Status |", "|--------|--------|--------|"]
        if example_triple is not None:
            lines.append(
                f"| {_triple_text(example_triple)} "
                f"| YES (given) | active |"
            )
        for rec in self.history:
            t = rec["triple"]
            event_type = rec.get("event_type", "evidence")
            turn = rec["turn"]
            if turn in self._retracted_turns and event_type != "retraction":
                lines.append(
                    f"| ~~{_triple_text(tuple(t))}~~ "
                    f"| ~~{rec['formal_feedback']}~~ | RETRACTED |"
                )
            elif event_type == "retraction":
                lines.append(
                    f"| {_triple_text(tuple(t))} "
                    f"| {rec['formal_feedback']} | replacement |"
                )
            else:
                lines.append(
                    f"| {_triple_text(tuple(t))} | {rec['formal_feedback']} | active |"
                )
        return "\n".join(lines)

    def get_ground_truth_at_step(self, step: int) -> Dict:
        if 0 <= step < len(self._ground_truth):
            return self._ground_truth[step]
        raise IndexError(f"Step {step} out of range [0, {len(self._ground_truth)})")


# ---------------------------------------------------------------------------
# Parsing utilities (tag-based hypotheses output format)
# ---------------------------------------------------------------------------

_HYPOTHESIS_TAG_RE = re.compile(r"<hypothesis>(.*?)</hypothesis>", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ITEM_SEP_RE = re.compile(r"[,，\s]+")


def parse_hypotheses(text: str) -> Optional[List[str]]:
    """Extract hypotheses list from agent response.

    Expected format: <hypothesis>rule_a, rule_b</hypothesis>
    - Accepts `,`, `，`, or whitespace as separators.
    - Content `none` (case-insensitive) yields an empty list `[]` (valid empty set).
    - Multiple `<hypothesis>` tags: the last one wins.
    - `<think>...</think>` reasoning blocks are stripped before matching.
    Returns:
        list[str] — deduplicated rule IDs (possibly empty)
        None     — no `<hypothesis>` tag found (parse error)
    """
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
    seen: set = set()
    for token in _ITEM_SEP_RE.split(content):
        rule_id = token.strip()
        if rule_id and rule_id not in seen:
            result.append(rule_id)
            seen.add(rule_id)
    return result


def parse_example_triple(text: str) -> Triple:
    import ast
    value = ast.literal_eval(text)
    if isinstance(value, tuple) and len(value) == 3 and all(isinstance(x, int) for x in value):
        return value
    raise ValueError(f"Invalid example triple: {text}")
