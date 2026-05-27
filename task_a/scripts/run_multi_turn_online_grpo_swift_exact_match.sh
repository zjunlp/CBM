#!/bin/bash
# Scenario A multi-turn GRPO launcher with binary exact-match reward.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REWARD_FUNC=task_a_belief_exact_match_reward
export OUTPUT_DIR=task_a/training/checkpoints_multi_turn_online_swift_grpo_9B_thinking_exact_match_rollout_8
export LOGDIR=task_a/outputs/multi_turn_online_swift_grpo_exact_match_logs
export SWANLAB_EXP_NAME=swift_grpo_9B_zero2_exact_match

exec "$SCRIPT_DIR/run_multi_turn_online_grpo_swift.sh" "$@"
