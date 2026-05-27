#!/usr/bin/env python3
"""
GRPO training script using Megatron-SWIFT for reinforcement learning
Contains two reward functions:
1. Orientation accuracy reward - validates GlobalFacing tags step by step
2. Rewrite penalty - penalizes modifications to original text
"""
import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from difflib import SequenceMatcher

# ============================================================================
# Reward Function 1: Orientation Accuracy
# ============================================================================

def extract_facing_tags(text: str) -> List[str]:
    """Extract all GlobalFacing tags from text"""
    return re.findall(r'\[GlobalFacing=(\w+)\]', text, re.IGNORECASE)

def calculate_orientation_reward(predicted_text: str, expected_orientations: List[str]) -> Tuple[float, Dict]:
    """
    Calculate orientation accuracy reward

    Rules:
    - Score is evenly distributed across each step
    - If a step is incorrect, subsequent steps are not scored
    - More correct steps yield higher scores

    Returns: (reward score, details)
    """
    predicted_orientations = extract_facing_tags(predicted_text)

    total_steps = len(expected_orientations)
    if total_steps == 0:
        return 0.0, {"error": "No expected orientations"}

    score_per_step = 1.0 / total_steps

    correct_steps = 0
    for i, expected in enumerate(expected_orientations):
        if i >= len(predicted_orientations):
            break

        if predicted_orientations[i].lower() == expected.lower():
            correct_steps += 1
        else:
            break

    reward = correct_steps * score_per_step

    details = {
        "total_steps": total_steps,
        "correct_steps": correct_steps,
        "predicted_count": len(predicted_orientations),
        "accuracy": correct_steps / total_steps,
        "reward": reward
    }

    return reward, details

# ============================================================================
# Reward Function 2: Rewrite Penalty
# ============================================================================

def extract_steps_without_tags(text: str) -> List[str]:
    """Extract original content of each step (without GlobalFacing tags)"""
    text_without_tags = re.sub(r'\[GlobalFacing=\w+\]', '', text)

    steps = re.split(r'\n(?=\d+\.\s+Actions:)', text_without_tags.strip())
    return [step.strip() for step in steps if step.strip()]

def calculate_text_similarity(original: str, modified: str) -> float:
    """Calculate text similarity (0-1)"""
    return SequenceMatcher(None, original, modified).ratio()

def calculate_rewrite_penalty(predicted_text: str, original_text: str) -> Tuple[float, Dict]:
    """
    Calculate rewrite penalty

    Rules:
    - Penalize text modifications other than adding GlobalFacing tags
    - More modifications result in higher penalties
    - Adding text beyond GlobalFacing tags incurs penalties

    Returns: (penalty score, details)
    """
    original_steps = extract_steps_without_tags(original_text)
    predicted_steps = extract_steps_without_tags(predicted_text)

    if len(original_steps) == 0:
        return 0.0, {"error": "No original steps"}

    total_penalty = 0.0
    step_details = []

    for i, original_step in enumerate(original_steps):
        if i >= len(predicted_steps):
            penalty = 1.0
            step_details.append({
                "step": i + 1,
                "penalty": penalty,
                "reason": "Missing step"
            })
            total_penalty += penalty
            continue

        predicted_step = predicted_steps[i]

        similarity = calculate_text_similarity(original_step, predicted_step)

        penalty = 1.0 - similarity

        original_len = len(original_step)
        predicted_len = len(predicted_step)

        if predicted_len > original_len * 1.1:
            extra_penalty = 0.2
            penalty += extra_penalty
            reason = f"Text expansion (similarity: {similarity:.2f}, extra penalty: {extra_penalty:.2f})"
        else:
            reason = f"Text modification (similarity: {similarity:.2f})"

        step_details.append({
            "step": i + 1,
            "penalty": penalty,
            "similarity": similarity,
            "reason": reason
        })

        total_penalty += penalty

    normalized_penalty = total_penalty / len(original_steps)

    details = {
        "total_steps": len(original_steps),
        "total_penalty": total_penalty,
        "normalized_penalty": normalized_penalty,
        "step_details": step_details
    }

    return normalized_penalty, details

# ============================================================================
# Combined Reward Function
# ============================================================================

def calculate_combined_reward(predicted_text: str, original_text: str,
                              expected_orientations: List[str],
                              orientation_weight: float = 0.7,
                              rewrite_weight: float = 0.3) -> Tuple[float, Dict]:
    """
    Calculate combined reward

    Parameters:
    - predicted_text: Model predicted text
    - original_text: Original text (user input)
    - expected_orientations: Expected orientation list
    - orientation_weight: Orientation accuracy weight
    - rewrite_weight: Rewrite penalty weight

    Returns: (total reward, details)
    """
    orientation_reward, orientation_details = calculate_orientation_reward(
        predicted_text, expected_orientations
    )

    rewrite_penalty, rewrite_details = calculate_rewrite_penalty(
        predicted_text, original_text
    )

    total_reward = (orientation_reward * orientation_weight) - (rewrite_penalty * rewrite_weight)

    details = {
        "orientation_reward": orientation_reward,
        "orientation_details": orientation_details,
        "rewrite_penalty": rewrite_penalty,
        "rewrite_details": rewrite_details,
        "total_reward": total_reward,
        "weights": {
            "orientation": orientation_weight,
            "rewrite": rewrite_weight
        }
    }

    return total_reward, details

# ============================================================================
# GRPO Training Configuration
# ============================================================================

def create_grpo_config(args):
    """Create GRPO training configuration"""
    config = {
        "model": args.model_name,
        "dataset": args.dataset_path,

        "rl_method": "grpo",

        "tensor_model_parallel_size": args.tensor_parallel_size,
        "sequence_parallel": True,

        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,

        "lr": args.learning_rate,
        "num_train_epochs": args.num_epochs,

        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,

        "orientation_weight": args.orientation_weight,
        "rewrite_weight": args.rewrite_weight,

        "packing": True,
        "recompute_granularity": "full",

        "output_dir": args.output_dir,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,

        "eval_steps": args.eval_steps,
        "evaluation_strategy": "steps",
    }

    return config

def generate_training_command(config):
    """Generate training command"""
    cmd_parts = ["megatron grpo"]

    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key} true")
            else:
                cmd_parts.append(f"--{key} false")
        else:
            cmd_parts.append(f"--{key} {value}")

    return " \\\n    ".join(cmd_parts)

# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='GRPO training script - GlobalFacing tag addition')

    parser.add_argument('--model-name', type=str, required=True,
                       help='Model name or path')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Dataset path')

    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                       help='Tensor parallel size (number of GPUs)')

    parser.add_argument('--micro-batch-size', type=int, default=1,
                       help='Batch size per GPU')
    parser.add_argument('--global-batch-size', type=int, default=4,
                       help='Global batch size')

    parser.add_argument('--learning-rate', type=float, default=5e-6,
                       help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=3,
                       help='Number of training epochs')

    parser.add_argument('--num-generations', type=int, default=4,
                       help='Number of samples generated per prompt')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Generation temperature')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p sampling')

    parser.add_argument('--orientation-weight', type=float, default=0.7,
                       help='Orientation accuracy reward weight')
    parser.add_argument('--rewrite-weight', type=float, default=0.3,
                       help='Rewrite penalty weight')

    parser.add_argument('--output-dir', type=str, default='./output_grpo',
                       help='Output directory')
    parser.add_argument('--save-steps', type=int, default=100,
                       help='Steps to save checkpoint')
    parser.add_argument('--logging-steps', type=int, default=10,
                       help='Logging steps')
    parser.add_argument('--eval-steps', type=int, default=50,
                       help='Evaluation steps')

    parser.add_argument('--dry-run', action='store_true',
                       help='Generate command only without execution')
    parser.add_argument('--test-reward', action='store_true',
                       help='Test reward functions')

    args = parser.parse_args()

    if args.test_reward:
        test_reward_functions()
        return 0

    print(f"\n{'='*80}")
    print(f"GRPO Training Configuration - GlobalFacing Tag Addition")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Reward Weights: orientation={args.orientation_weight}, rewrite_penalty={args.rewrite_weight}")
    print(f"{'='*80}\n")

    config = create_grpo_config(args)

    training_cmd = generate_training_command(config)

    print("Training Command:")
    print("-" * 80)
    print(training_cmd)
    print("-" * 80)

    cmd_file = Path(args.output_dir) / "training_command.sh"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(cmd_file, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write(training_cmd)
        f.write("\n")

    reward_file = Path(args.output_dir) / "reward_functions.py"
    with open(reward_file, 'w') as f:
        f.write("# Reward function implementation\n")
        f.write("# Copied from train_grpo.py\n\n")
        f.write(open(__file__).read())

    print(f"\nTraining command saved to: {cmd_file}")
    print(f"Reward functions saved to: {reward_file}")

    if args.dry_run:
        print("\n[DRY RUN] Command generated only, not executing training")
        return 0

    print("\nStarting training...")
    os.system(training_cmd)

    return 0

def test_reward_functions():
    """Test reward functions"""
    print("\n" + "="*80)
    print("Testing Reward Functions")
    print("="*80 + "\n")

    original_text = """1. Actions: [Observe()]. You observe:
• rubberduck: front-right, mid distance, facing forward
• dog: front, mid distance, facing backward
2. Actions: [Rotate(90), Observe()]. You rotated clockwise 90°. You observe:
• truck: front-slight-left, slightly far, facing left"""

    expected_orientations = ["north", "east"]

    predicted_text_correct = """1. Actions: [Observe()]. You observe:
• rubberduck: front-right, mid distance, facing forward
• dog: front, mid distance, facing backward
[GlobalFacing=north]
2. Actions: [Rotate(90), Observe()]. You rotated clockwise 90°. You observe:
• truck: front-slight-left, slightly far, facing left
[GlobalFacing=east]"""

    print("Test Case 1: Completely Correct")
    print("-" * 80)
    reward, details = calculate_combined_reward(
        predicted_text_correct, original_text, expected_orientations
    )
    print(f"Total Reward: {reward:.4f}")
    print(f"Orientation Reward: {details['orientation_reward']:.4f}")
    print(f"Rewrite Penalty: {details['rewrite_penalty']:.4f}")
    print(f"Orientation Accuracy: {details['orientation_details']['accuracy']:.2%}")
    print()

    predicted_text_wrong = """1. Actions: [Observe()]. You observe:
• rubberduck: front-right, mid distance, facing forward
• dog: front, mid distance, facing backward
[GlobalFacing=south]
2. Actions: [Rotate(90), Observe()]. You rotated clockwise 90°. You observe:
• truck: front-slight-left, slightly far, facing left
[GlobalFacing=west]"""

    print("Test Case 2: Wrong Orientation")
    print("-" * 80)
    reward, details = calculate_combined_reward(
        predicted_text_wrong, original_text, expected_orientations
    )
    print(f"Total Reward: {reward:.4f}")
    print(f"Orientation Reward: {details['orientation_reward']:.4f}")
    print(f"Rewrite Penalty: {details['rewrite_penalty']:.4f}")
    print(f"Orientation Accuracy: {details['orientation_details']['accuracy']:.2%}")
    print()

    predicted_text_modified = """1. Actions: [Observe()]. You observe:
• rubberduck: front-right, mid distance
• dog: front, mid distance
[GlobalFacing=north]
2. Actions: [Rotate(90), Observe()]. You rotated clockwise 90°. You observe:
• truck: front-slight-left, slightly far
[GlobalFacing=east]"""

    print("Test Case 3: Modified Original Text")
    print("-" * 80)
    reward, details = calculate_combined_reward(
        predicted_text_modified, original_text, expected_orientations
    )
    print(f"Total Reward: {reward:.4f}")
    print(f"Orientation Reward: {details['orientation_reward']:.4f}")
    print(f"Rewrite Penalty: {details['rewrite_penalty']:.4f}")
    print(f"Orientation Accuracy: {details['orientation_details']['accuracy']:.2%}")
    print()

if __name__ == '__main__':
    sys.exit(main())
