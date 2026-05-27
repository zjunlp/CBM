#!/usr/bin/env python3
"""
GRPO reward function plugin for Megatron-SWIFT
Provides OrientationReward reward function class
"""
import re
from typing import List, Dict, Any
from swift.rewards import ORM, orms


def extract_facing_tags(text: str) -> List[str]:
    """Extract all GlobalFacing tags from text"""
    return re.findall(r'\[GlobalFacing=(\w+)\]', text, re.IGNORECASE)


def count_steps(text: str) -> int:
    """Count the number of steps in text"""
    steps = re.findall(r'^\d+\.\s+Actions:', text, re.MULTILINE)
    return len(steps)


class OrientationReward(ORM):
    """
    Orientation accuracy reward function - inherits from ORM base class

    Reward calculation logic:
    1. Orientation accuracy reward (weight 0.7)
       - Extract predicted [GlobalFacing=X] tags
       - Compare with expected orientations sequentially (consecutive correct)
       - Stop scoring once an error occurs
       - Reward = consecutive correct count / total count

    2. Step count penalty (weight 0.3)
       - Count steps in original and predicted text
       - Equal count: penalty 0.0
       - Unequal count: penalty 1.0

    Total reward = orientation reward * 0.7 - step penalty * 0.3
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)

    def __call__(self, completions: List[str], prompts: List[str] = None,
                 metadata: List[Dict] = None, **kwargs) -> List[float]:
        """
        Calculate rewards for batch samples

        Parameters:
        - completions: List of model generated responses
        - prompts: List of input prompts (original text)
        - metadata: List of dataset metadata (contains orientations)

        Returns:
        - rewards: List of reward scores for each sample
        """
        rewards = []

        for i, predicted_text in enumerate(completions):
            try:
                original_text = prompts[i] if prompts else ''
                sample_metadata = metadata[i] if metadata else {}
                expected_orientations = sample_metadata.get('orientations', [])

                predicted_orientations = extract_facing_tags(predicted_text)
                total_steps = len(expected_orientations)

                if total_steps == 0:
                    rewards.append(0.0)
                    continue

                correct_steps = 0
                for j, expected in enumerate(expected_orientations):
                    if j >= len(predicted_orientations):
                        break

                    if predicted_orientations[j].lower() == expected.lower():
                        correct_steps += 1
                    else:
                        break

                orientation_reward_score = correct_steps / total_steps

                original_count = count_steps(original_text)
                predicted_count = count_steps(predicted_text)

                if original_count == 0:
                    step_penalty = 0.0
                else:
                    step_penalty = 0.0 if predicted_count == original_count else 1.0

                total_reward = (orientation_reward_score * 0.7) - (step_penalty * 0.3)

                rewards.append(total_reward)

            except Exception as e:
                print(f"Error in OrientationReward for sample {i}: {e}")
                rewards.append(0.0)

        return rewards

# Register to swift.rewards.orms dictionary
orms['orientation_reward'] = OrientationReward

print(f"[grpo_plugin] Registered orientation_reward to orms. Available rewards: {list(orms.keys())}", flush=True)

