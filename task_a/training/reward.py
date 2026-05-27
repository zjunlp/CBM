"""Reward functions for belief management RL training."""

import json
import sys
from pathlib import Path
from typing import List, Optional

from swift.rewards import ORM, orms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hypotheses_parser import parse_hypotheses


def _jaccard_from_response(response: str, gt_set: set[str]) -> float:
    hypotheses = parse_hypotheses(response)
    if hypotheses is None:
        return 0.0
    pred_set = set(hypotheses)
    union = pred_set | gt_set
    if not union:
        return 1.0
    return len(pred_set & gt_set) / len(union)


def _exact_match_from_response(response: str, gt_set: set[str]) -> float:
    hypotheses = parse_hypotheses(response)
    if hypotheses is None:
        return 0.0
    return 1.0 if set(hypotheses) == gt_set else 0.0


def belief_exact_match_reward(
    prompts: List[str],
    completions: Optional[List[str]] = None,
    gt_survivors: Optional[List[str | List[str]]] = None,
    reward_weight: Optional[List[float]] = None,
    **kwargs,
) -> List[float]:
    """Jaccard similarity reward between predicted and ground-truth hypothesis sets.

    Replaces the previous binary exact-match reward to provide partial credit and
    eliminate the incentive for the model to collapse to a single-oracle output.

    reward = |pred ∩ gt| / |pred ∪ gt|   (Jaccard index, range [0, 1])

    Special cases:
    - Parse failure (None): 0.0
    - Both sets empty:      1.0  (shouldn't occur in practice)

    Returns:
        List of reward values for the selected training turn.
    """
    if completions is None:
        completions = prompts
        prompts = kwargs.get("prompt", [""] * len(completions))
    if gt_survivors is None:
        gt_survivors = kwargs.get("gt_survivors")

    if reward_weight is None:
        reward_weight = [1.0] * len(completions)
    if gt_survivors is None:
        raise ValueError("gt_survivors is required.")
    if not (len(completions) == len(gt_survivors) == len(reward_weight)):
        raise ValueError(
            "completions, gt_survivors, and reward_weight must have the same length "
            f"got {len(completions)}, {len(gt_survivors)}, {len(reward_weight)}"
        )

    rewards = []
    for completion, gt_str, weight in zip(completions, gt_survivors, reward_weight, strict=True):
        gt_values = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
        gt_set = set(gt_values)
        jaccard = _jaccard_from_response(completion, gt_set)
        rewards.append(jaccard * float(weight))
    return rewards


def belief_binary_exact_match_reward(
    prompts: List[str],
    completions: Optional[List[str]] = None,
    gt_survivors: Optional[List[str | List[str]]] = None,
    reward_weight: Optional[List[float]] = None,
    **kwargs,
) -> List[float]:
    """0/1 exact-match reward for the predicted and golden hypothesis sets.

    reward = 1 only when the parsed hypothesis set exactly equals the golden set;
    otherwise reward = 0. Parse failure also receives 0.
    """
    if completions is None:
        completions = prompts
        prompts = kwargs.get("prompt", [""] * len(completions))
    if gt_survivors is None:
        gt_survivors = kwargs.get("gt_survivors")

    if reward_weight is None:
        reward_weight = [1.0] * len(completions)
    if gt_survivors is None:
        raise ValueError("gt_survivors is required.")
    if not (len(completions) == len(gt_survivors) == len(reward_weight)):
        raise ValueError(
            "completions, gt_survivors, and reward_weight must have the same length "
            f"got {len(completions)}, {len(gt_survivors)}, {len(reward_weight)}"
        )

    rewards = []
    for completion, gt_str, weight in zip(completions, gt_survivors, reward_weight, strict=True):
        gt_values = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
        exact = _exact_match_from_response(completion, set(gt_values))
        rewards.append(exact * float(weight))
    return rewards


class ScenarioABeliefReward(ORM):
    """Swift ORM wrapper for Scenario A hypothesis-set reward."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        return belief_exact_match_reward(completions, **kwargs)


class ScenarioABeliefExactMatchReward(ORM):
    """Swift ORM wrapper for Scenario A binary exact-match reward."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        return belief_binary_exact_match_reward(completions, **kwargs)


orms["task_a_belief_reward"] = ScenarioABeliefReward
orms["task_a_belief_exact_match_reward"] = ScenarioABeliefExactMatchReward
