from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ExperimentConfig:
    experiment_id: str = "exp_default"
    seed: int = 42
    rule_name: str = "ascending_order"
    max_turns: int = 5
    example_triple: str = "(2, 4, 6)"
    agent_model: str = "unknown"
    agent_temperature: float = 0.7
    agent_max_tokens: int = 800
    output_dir: str = "outputs"


# ---------------------------------------------------------------------------
# Challenge type strings and display helpers
#
# Evidence sequences (triples, events, ground truth steps) are built in
# ``task_a.experiments.generate_sequences`` and related modules.
# ---------------------------------------------------------------------------

ChallengeSequence = Dict[str, Any]

CHALLENGE_MISRECORD_CORRECTION = "misrecord_correction"
CHALLENGE_SHRINK_THEN_HOLD = "shrink_then_hold"

# Letter / legacy aliases that may appear in older saved runs or reports.
CHALLENGE_TYPE_ALIASES: Dict[str, str] = {
    "A": CHALLENGE_MISRECORD_CORRECTION,
    "D": CHALLENGE_SHRINK_THEN_HOLD,
    "E": "stable_set_hold",
    "B": "hold_short",
    "C": "hold_long",
    "pair_hold": "stable_set_hold",
    "retraction_recover": CHALLENGE_MISRECORD_CORRECTION,
}

CHALLENGE_DISPLAY_NAMES: Dict[str, str] = {
    CHALLENGE_MISRECORD_CORRECTION: "Wrong-Convergence Correction",
    CHALLENGE_SHRINK_THEN_HOLD: "Shrink-Then-Hold",
    "stable_set_hold": "Stable-Set Hold",
    "hold_short": "Hold-Short",
    "hold_long": "Hold-Long",
    "failed_stay": "FailedStay",
}


def normalize_challenge_type(challenge_type: str) -> str:
    """Resolve legacy aliases; otherwise return the string unchanged."""
    return CHALLENGE_TYPE_ALIASES.get(challenge_type, challenge_type)


def challenge_display_name(challenge_type: str) -> str:
    """Return a short human-readable label for a challenge type string."""
    key = normalize_challenge_type(challenge_type)
    if key in CHALLENGE_DISPLAY_NAMES:
        return CHALLENGE_DISPLAY_NAMES[key]
    return key.replace("_", " ").title()
