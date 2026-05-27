"""Core evidence / ground-truth utilities for Scenario A."""

from typing import Dict, List, Set, Tuple

from task_a.core.rules import RULES, Triple

RULE_NAMES = sorted(RULES.keys())


def _oracle_feedback(oracle_rule: str, triple: Triple) -> str:
    return "YES" if RULES[oracle_rule].validate(triple) else "NO"


def _event_feedback_result(oracle_rule: str, event: Dict) -> str:
    """Return the result that should be shown to the agent for an event."""
    if "recorded_result" in event:
        return event["recorded_result"]
    if "new_result" in event:
        return event["new_result"]
    triple = event["new_triple"] if event["type"] == "retraction" else event["triple"]
    return _oracle_feedback(oracle_rule, triple)


def compute_surviving_set(
    evidence: List[Tuple[Triple, str]],
) -> Set[str]:
    """Given a list of (triple, 'YES'/'NO') pairs, return the set of rule names
    that are consistent with all evidence."""
    survivors = set(RULE_NAMES)
    for triple, result in evidence:
        expected = (result == "YES")
        survivors = {r for r in survivors if RULES[r].validate(triple) == expected}
    return survivors


def compute_ground_truth_beliefs(
    oracle_rule: str,
    evidence_triples: List[Triple],
) -> List[Dict]:
    """For a given oracle rule and evidence sequence, compute the ground truth
    survivor set after each step.

    Returns a list of dicts, one per evidence step:
      {"triple": (a, b, c), "result": "YES"/"NO", "survivors": {...}}
    """
    evidence_so_far: List[Tuple[Triple, str]] = []
    steps = []
    for triple in evidence_triples:
        result = _oracle_feedback(oracle_rule, triple)
        evidence_so_far.append((triple, result))
        survivors = compute_surviving_set(evidence_so_far)
        steps.append({
            "triple": triple,
            "result": result,
            "survivors": survivors,
        })
    return steps


def compute_retraction_ground_truth(
    oracle_rule: str,
    events: List[Dict],
) -> List[Dict]:
    """Compute ground truth at each step of a retraction sequence.

    Maintains an 'active evidence' list that handles retractions:
    when a retraction event occurs, the retracted evidence is removed
    and the replacement is added.

    Returns a list of dicts (one per event):
      {
        "event_type": "evidence" | "retraction",
        "triple": (a, b, c),
        "result": "YES" / "NO",
        "survivors": set of rule names,
        "retract_turn": int (only for retraction events),
        "reinstated_rules": set (only for retraction events),
        "pre_retraction_survivors": set (only for retraction events),
      }
    """
    active_evidence: List[Tuple[Triple, str]] = []
    event_to_active: Dict[int, int] = {}
    steps = []

    for i, event in enumerate(events):
        if event["type"] == "evidence":
            triple = event["triple"]
            result = _event_feedback_result(oracle_rule, event)
            event_to_active[i] = len(active_evidence)
            active_evidence.append((triple, result))
            survivors = compute_surviving_set(active_evidence)
            steps.append({
                "event_type": "evidence",
                "triple": triple,
                "result": result,
                "survivors": survivors,
            })
        elif event["type"] == "retraction":
            retract_turn = event["retract_turn"]
            new_triple = event["new_triple"]

            pre_retraction_survivors = compute_surviving_set(active_evidence)

            if retract_turn not in event_to_active:
                raise ValueError(
                    f"retract_turn={retract_turn} does not refer to an evidence event"
                )
            active_idx = event_to_active[retract_turn]
            active_evidence = [
                e for j, e in enumerate(active_evidence) if j != active_idx
            ]
            for k in list(event_to_active.keys()):
                if event_to_active[k] > active_idx:
                    event_to_active[k] -= 1
            del event_to_active[retract_turn]

            new_result = _event_feedback_result(oracle_rule, event)
            event_to_active[i] = len(active_evidence)
            active_evidence.append((new_triple, new_result))

            survivors = compute_surviving_set(active_evidence)
            reinstated = survivors - pre_retraction_survivors

            steps.append({
                "event_type": "retraction",
                "triple": new_triple,
                "result": new_result,
                "survivors": survivors,
                "retract_turn": retract_turn,
                "reinstated_rules": reinstated,
                "pre_retraction_survivors": pre_retraction_survivors,
            })
        else:
            raise ValueError(f"Unknown event type: {event['type']}")

    return steps
