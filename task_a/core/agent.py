from typing import Optional

from task_a.core.rules import list_rules


_OUTPUT_FORMAT_BLOCK = """\
Output format (strict):
<hypothesis>rule_id_1, rule_id_2</hypothesis>

- Inside `<hypothesis>`: comma-separated rule IDs that are still consistent with ALL active evidence.
- Do not output anything outside the `<hypothesis>` tag."""


def _build_rule_space_text() -> str:
    rows = []
    for rule_id, desc in list_rules().items():
        rows.append(f'- "{rule_id}": {desc}')
    return "\n".join(rows)


def build_agent_system_prompt(
    include_rule_predictions: bool = True,
    model_type: str = "local",
) -> str:
    if include_rule_predictions:
        evidence_description = (
            "Each turn, the host provides:\n"
            "1. A triple and its currently recorded result: YES or NO.\n"
            "2. What each candidate rule predicts for that triple."
        )
        correction_note = (
            "\n\nSometimes the host may correct an earlier recorded result.\n"
            "When that happens, the old record becomes inactive and the corrected one replaces it.\n"
            "Do not revise old evidence unless the host explicitly issues a correction.\n"
            "\nYour job is to maintain the set of rule IDs that are still consistent with all currently active recorded evidence.\n"
            "A rule is contradicted if its prediction differs from the currently recorded result for any active evidence item."
        )
    else:
        evidence_description = (
            "Each turn, the host provides a triple and its currently recorded result: YES or NO.\n"
            "Sometimes the host may correct an earlier recorded result."
        )
        correction_note = (
            "\n\nYour job is to maintain the set of rule IDs that are still consistent with all currently active recorded evidence."
        )
    return f"""\
You are a cautious scientist tracking candidate rules in a rule-guessing game.

{evidence_description}{correction_note}

Available rule IDs:
{_build_rule_space_text()}

{_OUTPUT_FORMAT_BLOCK}
"""


def build_initial_message(
    example_triple: str = "(2, 4, 6)",
    include_evidence_table: bool = True,
) -> str:
    """Anchor user message before the first evidence turn is appended.

    The system prompt already states that `example_triple` is accepted and that
    all candidates start consistent, so we keep this message minimal and let the
    orchestrator append the actual Turn 0 evidence.
    """
    base = "Let's begin."
    if include_evidence_table:
        base += (
            "\n\n**Active evidence table:**\n"
            "| Triple | Result |\n"
            "|--------|--------|\n"
            f"| {example_triple} | YES (given) |"
        )
    return base


def build_feedback_message(
    env_feedback_text: str,
    turn: int,
    evidence_table_text: Optional[str] = None,
) -> str:
    msg = (
        f"**Turn {turn} evidence:**\n"
        f"{env_feedback_text}\n"
    )
    if evidence_table_text:
        msg += f"\n**Active evidence table:**\n{evidence_table_text}\n"
    msg += "\nPlease update your hypotheses using all currently active evidence."
    return msg
