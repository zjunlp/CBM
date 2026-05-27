#!/usr/bin/env python3
"""
GRPO training script with custom reward functions
Extracts correct answers from dataset metadata.orientations for scoring
"""
import os
import sys
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict

# ============================================================================
# Reward Functions
# ============================================================================

def extract_facing_tags(text: str) -> List[str]:
    """Extract all GlobalFacing tags from text"""
    return re.findall(r'\[GlobalFacing=(\w+)\]', text, re.IGNORECASE)

def calculate_orientation_reward(predicted_text: str, expected_orientations: List[str]) -> Tuple[float, Dict]:
    """
    Calculate orientation accuracy reward

    Rules:
    - Compare tags sequentially from the first one
    - Stop scoring once an error occurs
    - Reward = consecutive correct count / total count
    """
    predicted_orientations = extract_facing_tags(predicted_text)

    total_steps = len(expected_orientations)
    if total_steps == 0:
        return 0.0, {"error": "No expected orientations"}

    correct_steps = 0
    for i, expected in enumerate(expected_orientations):
        if i >= len(predicted_orientations):
            break

        if predicted_orientations[i].lower() == expected.lower():
            correct_steps += 1
        else:
            break

    reward = correct_steps / total_steps

    details = {
        "total_steps": total_steps,
        "correct_steps": correct_steps,
        "predicted_count": len(predicted_orientations),
        "accuracy": correct_steps / total_steps,
        "reward": reward
    }

    return reward, details

def count_steps(text: str) -> int:
    """Count the number of steps in text"""
    steps = re.findall(r'^\d+\.\s+Actions:', text, re.MULTILINE)
    return len(steps)

def calculate_step_count_penalty(predicted_text: str, original_text: str) -> Tuple[float, Dict]:
    """
    Calculate step count penalty

    Rules:
    - Only check if step counts match
    - Mismatch: penalty 1.0
    - Match: penalty 0.0
    """
    original_count = count_steps(original_text)
    predicted_count = count_steps(predicted_text)

    if original_count == 0:
        return 0.0, {"error": "No original steps"}

    if predicted_count != original_count:
        penalty = 1.0
        reason = f"Step count mismatch: predicted {predicted_count}, expected {original_count}"
    else:
        penalty = 0.0
        reason = "Step count matches"

    details = {
        "original_count": original_count,
        "predicted_count": predicted_count,
        "penalty": penalty,
        "reason": reason
    }

    return penalty, details

def calculate_combined_reward(predicted_text: str, original_text: str,
                              expected_orientations: List[str],
                              orientation_weight: float = 0.7,
                              step_count_weight: float = 0.3) -> float:
    """
    Calculate combined reward for GRPO training

    Returns: total reward score (float)
    """
    orientation_reward, _ = calculate_orientation_reward(predicted_text, expected_orientations)
    step_count_penalty, _ = calculate_step_count_penalty(predicted_text, original_text)

    total_reward = (orientation_reward * orientation_weight) - (step_count_penalty * step_count_weight)

    return total_reward

# ============================================================================
# Reward Function Wrapper for megatron-swift
# ============================================================================

def reward_function(samples: List[Dict]) -> List[float]:
    """
    GRPO reward function - megatron-swift interface

    Parameters:
    - samples: List of generated samples, each containing:
        - 'prompt': Input prompt
        - 'response': Model generated response
        - 'metadata': Dataset metadata (contains orientations)

    Returns:
    - rewards: List of reward scores for each sample
    """
    rewards = []

    for sample in samples:
        try:
            original_text = sample.get('prompt', '')
            predicted_text = sample.get('response', '')
            metadata = sample.get('metadata', {})
            expected_orientations = metadata.get('orientations', [])

            reward = calculate_combined_reward(
                predicted_text=predicted_text,
                original_text=original_text,
                expected_orientations=expected_orientations,
                orientation_weight=0.7,
                rewrite_weight=0.3
            )

            rewards.append(reward)

        except Exception as e:
            print(f"Error calculating reward: {e}")
            rewards.append(0.0)

    return rewards

# ============================================================================
# Dataset Validation
# ============================================================================

def validate_dataset(dataset_path: str) -> bool:
    """Validate dataset format"""
    print(f"Validating dataset: {dataset_path}")

    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("❌ Invalid dataset format: should be a list")
            return False

        print(f"✓ Dataset contains {len(data)} samples")

        sample = data[0]

        if 'messages' not in sample:
            print("❌ Missing 'messages' field")
            return False

        if 'metadata' not in sample:
            print("❌ Missing 'metadata' field")
            return False

        metadata = sample['metadata']

        if 'orientations' not in metadata:
            print("❌ Missing 'orientations' field in metadata")
            return False

        orientations = metadata['orientations']
        print(f"✓ First sample contains {len(orientations)} orientation tags")
        print(f"  Orientation sequence: {orientations[:5]}..." if len(orientations) > 5 else f"  Orientation sequence: {orientations}")

        messages = sample['messages']
        if len(messages) < 2:
            print("❌ messages should contain at least system and user messages")
            return False

        print(f"✓ messages contains {len(messages)} messages")

        user_msg = None
        for msg in messages:
            if msg['role'] == 'user':
                user_msg = msg['content']
                break

        if user_msg:
            steps = re.findall(r'^\d+\.\s+Actions:', user_msg, re.MULTILINE)
            print(f"✓ User input contains {len(steps)} steps")

            if len(steps) != len(orientations):
                print(f"⚠️  Warning: step count({len(steps)}) does not match orientation count({len(orientations)})")

        print("\n✓ Dataset format validation passed!")
        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

# ============================================================================
# Main Function
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GRPO reward function - dataset validation')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset path')
    parser.add_argument('--test', action='store_true', help='Test reward function')

    args = parser.parse_args()

    if not validate_dataset(args.dataset):
        sys.exit(1)

    if args.test:
        print("\n" + "="*80)
        print("Testing Reward Function")
        print("="*80)

        with open(args.dataset, 'r') as f:
            data = json.load(f)

        sample = data[0]
        messages = sample['messages']
        metadata = sample['metadata']

        user_msg = None
        assistant_msg = None
        for msg in messages:
            if msg['role'] == 'user':
                user_msg = msg['content']
            elif msg['role'] == 'assistant':
                assistant_msg = msg['content']

        if user_msg and assistant_msg:
            expected_orientations = metadata['orientations']

            print(f"\nOriginal text step count: {len(re.findall(r'^\d+\.\s+Actions:', user_msg, re.MULTILINE))}")
            print(f"Expected orientation count: {len(expected_orientations)}")
            print(f"Expected orientations: {expected_orientations[:5]}...")

            reward = calculate_combined_reward(
                predicted_text=assistant_msg,
                original_text=user_msg,
                expected_orientations=expected_orientations
            )

            print(f"\n✓ Combined reward score: {reward:.4f}")

            orientation_reward, orientation_details = calculate_orientation_reward(
                assistant_msg, expected_orientations
            )
            step_count_penalty, step_count_details = calculate_step_count_penalty(
                assistant_msg, user_msg
            )

            print(f"  - Orientation accuracy reward: {orientation_reward:.4f}")
            print(f"    Correct steps: {orientation_details['correct_steps']}/{orientation_details['total_steps']}")
            print(f"    Accuracy: {orientation_details['accuracy']:.2%}")
            print(f"  - Step count penalty: {step_count_penalty:.4f}")
            print(f"    {step_count_details['reason']}")

        print("\n✓ Test completed!")
