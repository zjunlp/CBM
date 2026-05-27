from typing import Any, Dict, List, Optional, Set

from task_a.core.agent import (
    build_agent_system_prompt,
    build_feedback_message,
    build_initial_message,
)
from task_a.core.config import ExperimentConfig
from task_a.core.environment import (
    Environment,
    RetractionEnvironment,
    parse_example_triple,
    parse_hypotheses,
)
from task_a.core.evidence_sequences import RULE_NAMES


class GameOrchestrator:
    """Runs a complete evidence-driven belief tracking game session."""

    def __init__(
        self,
        config: ExperimentConfig,
        agent_backend: Any,
        env: Environment,
        label_mode: str = "none",
        include_evidence_table: bool = True,
        include_rule_predictions: bool = True,
        extra_final_suffix: Optional[str] = None,
        model_type: str = "local",
    ):
        self.config = config
        self.agent_backend = agent_backend
        self.env = env
        self.label_mode = label_mode
        self.include_evidence_table = include_evidence_table
        self.include_rule_predictions = include_rule_predictions
        self.extra_final_suffix = extra_final_suffix
        self.model_type = model_type
        self.is_retraction_env = isinstance(env, RetractionEnvironment)
        self.example_triple = parse_example_triple(self.config.example_triple)

    def _get_system_prompt(self) -> str:
        return build_agent_system_prompt(
            include_rule_predictions=self.include_rule_predictions,
            model_type=self.model_type,
        )

    def run(self) -> Dict[str, Any]:
        """Execute the full game and return the trajectory."""
        turns: List[Dict[str, Any]] = []
        conversation: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": self._get_system_prompt(),
            },
            {
                "role": "user",
                "content": build_initial_message(
                    self.config.example_triple,
                    include_evidence_table=self.include_evidence_table,
                ),
            },
        ]

        max_evidence_turns = min(self.config.max_turns, self.env.total_evidence_steps)

        for turn in range(max_evidence_turns):
            is_final_turn = (turn == max_evidence_turns - 1)

            # 1. Environment pushes the next evidence item.
            env_record = self.env.step(turn)
            evidence_table_text = None
            if self.include_evidence_table:
                evidence_table_text = self.env.get_evidence_table(
                    example_triple=self.example_triple
                )
            env_text = self.env.format_feedback(
                env_record,
                label_mode=self.label_mode,
                include_rule_predictions=self.include_rule_predictions,
            )

            # 2. Build the next user message.
            self._append_evidence_message(
                conversation=conversation,
                turn=turn,
                env_text=env_text,
                evidence_table_text=evidence_table_text,
                final_turn=is_final_turn,
            )

            # 3. Agent generates response
            print(f"  [Turn {turn}] Agent generating...", flush=True)
            messages = self._build_context_messages(conversation)
            agent_response = self.agent_backend.chat_completion(
                messages=messages,
                temperature=self.config.agent_temperature,
                max_tokens=self.config.agent_max_tokens,
            )
            conversation.append({"role": "assistant", "content": agent_response})

            # 4. Parse agent output
            hypotheses = parse_hypotheses(agent_response)

            # 5. Compute set-based belief metrics
            gt = self.env.get_ground_truth_at_step(turn)
            belief_metrics = self._compute_belief_metrics(hypotheses, gt)

            # 6. Record turn
            turn_record = self._build_turn_record(
                turn=turn,
                agent_response=agent_response,
                hypotheses=hypotheses,
                env_record=env_record,
                belief_metrics=belief_metrics,
            )
            turns.append(turn_record)

            # Log progress
            self._log_turn_summary(turn, env_record, hypotheses)

        # Compute trajectory-level metrics
        trajectory_metrics = self._compute_trajectory_metrics(turns)

        trajectory = {
            "experiment_id": self.config.experiment_id,
            "rule_name": self.config.rule_name,
            "rule_description": self.env.rule.description,
            "max_turns": self.config.max_turns,
            "agent_model": self.config.agent_model,
            "n_turns_played": len(turns),
            "trajectory_metrics": trajectory_metrics,
            "turns": turns,
            "conversation": conversation,
        }
        return trajectory

    def _build_context_messages(
        self, conversation: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Return the complete conversation history to send to the model."""
        return conversation

    def _append_evidence_message(
        self,
        *,
        conversation: List[Dict[str, str]],
        turn: int,
        env_text: str,
        evidence_table_text: Optional[str],
        final_turn: bool,
    ) -> None:
        if turn == 0:
            first_evidence = (
                f"\n\n**Turn 0 evidence:**\n{env_text}\n"
            )
            if evidence_table_text:
                first_evidence += (
                    f"\n**Active evidence table:**\n{evidence_table_text}\n"
                )
            first_evidence += "\nPlease update your hypotheses based on this evidence."
            conversation[-1]["content"] += first_evidence
            return

        feedback_msg = build_feedback_message(
            env_text,
            turn,
            evidence_table_text=evidence_table_text,
        )
        if final_turn and self.extra_final_suffix:
            feedback_msg += self.extra_final_suffix
        conversation.append({"role": "user", "content": feedback_msg})

    def _build_turn_record(
        self,
        *,
        turn: int,
        agent_response: str,
        hypotheses: Optional[List[str]],
        env_record: Dict[str, Any],
        belief_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        env_record_for_log = {
            key: value
            for key, value in env_record.items()
            if key != "gt_survivors"
        }
        return {
            "turn": turn,
            "agent_response": agent_response,
            "hypotheses": hypotheses,
            "env_record": env_record_for_log,
            "gt_survivors": env_record["gt_survivors"],
            "belief_metrics": belief_metrics,
            "parse_error": hypotheses is None,
        }

    def _log_turn_summary(
        self,
        turn: int,
        env_record: Dict[str, Any],
        hypotheses: Optional[List[str]],
    ) -> None:
        gt_survivors = set(env_record["gt_survivors"])
        agent_set = set(hypotheses) if hypotheses else set()
        oracle_in = self.config.rule_name in agent_set
        print(
            f"  [Turn {turn}] {tuple(env_record['triple'])} => "
            f"{env_record['formal_feedback']}  "
            f"gt_survivors={len(gt_survivors)}  "
            f"agent_hyps={sorted(agent_set)}  "
            f"oracle_in={oracle_in}",
            flush=True,
        )

    def _compute_belief_metrics(
        self, hypotheses: Optional[List[str]], gt: Dict,
    ) -> Dict[str, Any]:
        """Compute per-turn set-based belief quality metrics."""
        if hypotheses is None:
            return {"parse_error": True}

        gt_survivors: Set[str] = gt["survivors"]
        agent_set: Set[str] = set(hypotheses)
        oracle = self.config.rule_name

        # Core set metrics
        intersection = agent_set & gt_survivors
        precision = len(intersection) / len(agent_set) if agent_set else 0.0
        recall = len(intersection) / len(gt_survivors) if gt_survivors else 0.0
        exact_match = (agent_set == gt_survivors)
        oracle_in_hypotheses = oracle in agent_set

        # Error counts
        false_retention = sorted(agent_set - gt_survivors)   # kept but should be eliminated
        false_elimination = sorted(gt_survivors - agent_set)  # removed but should survive

        return {
            "oracle_in_hypotheses": oracle_in_hypotheses,
            "exact_match": exact_match,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "false_retention": false_retention,
            "false_retention_count": len(false_retention),
            "false_elimination": false_elimination,
            "false_elimination_count": len(false_elimination),
            "agent_set_size": len(agent_set),
            "gt_set_size": len(gt_survivors),
        }

    def _compute_trajectory_metrics(self, turns: List[Dict]) -> Dict[str, Any]:
        """Compute trajectory-level aggregate metrics."""
        if not turns:
            return {}

        valid_turns = [t for t in turns if t.get("hypotheses") is not None]
        if not valid_turns:
            return {"all_parse_errors": True}

        n_valid = len(valid_turns)

        # Oracle retention trajectory
        oracle_in_trajectory = [
            t["belief_metrics"].get("oracle_in_hypotheses", False) for t in valid_turns
        ]
        oracle_retention_rate = sum(oracle_in_trajectory) / n_valid

        # Exact match rate
        exact_matches = sum(
            1 for t in valid_turns if t["belief_metrics"].get("exact_match", False)
        )
        exact_match_rate = exact_matches / n_valid

        # Average precision / recall
        avg_precision = sum(
            t["belief_metrics"].get("precision", 0) for t in valid_turns
        ) / n_valid
        avg_recall = sum(
            t["belief_metrics"].get("recall", 0) for t in valid_turns
        ) / n_valid

        # Belief failed_update: after a rule is eliminated from gt, does the agent
        # still keep it in hypotheses on the next turn?
        failed_update_violations = 0
        failed_update_checks = 0
        for i in range(1, len(turns)):
            prev_gt = set(turns[i - 1].get("gt_survivors", RULE_NAMES))
            curr_gt = set(turns[i].get("gt_survivors", RULE_NAMES))
            newly_eliminated = prev_gt - curr_gt
            if newly_eliminated and turns[i].get("hypotheses") is not None:
                agent_set = set(turns[i]["hypotheses"])
                for r in newly_eliminated:
                    failed_update_checks += 1
                    if r in agent_set:
                        failed_update_violations += 1

        # Oracle removed after appearing
        oracle_appeared = False
        oracle_removed_after_appearing = False
        for t in valid_turns:
            if t["belief_metrics"].get("oracle_in_hypotheses", False):
                oracle_appeared = True
            elif oracle_appeared:
                oracle_removed_after_appearing = True
                break

        return {
            "oracle_retention_rate": round(oracle_retention_rate, 4),
            "oracle_in_trajectory": oracle_in_trajectory,
            "oracle_removed_after_appearing": oracle_removed_after_appearing,
            "exact_match_rate": round(exact_match_rate, 4),
            "avg_precision": round(avg_precision, 4),
            "avg_recall": round(avg_recall, 4),
            "belief_failed_update_violations": failed_update_violations,
            "belief_failed_update_checks": failed_update_checks,
            "belief_failed_update_rate": (
                round(failed_update_violations / failed_update_checks, 4)
                if failed_update_checks > 0 else 0.0
            ),
            "format_success_rate": round(n_valid / len(turns), 4) if turns else None,
            # --- JSON belief-tracking metrics ---
            "elimination_compliance": self._compute_elimination_compliance(turns),
            "cumulative_consistency": self._compute_cumulative_consistency(turns),
            # --- retraction metrics ---
            **self._compute_retraction_metrics(turns),
        }

    def _compute_elimination_compliance(self, turns: List[Dict]) -> Optional[float]:
        """Elimination Compliance (EC): when evidence contradicts a rule that
        the agent currently holds, does the agent remove it?

        EC = P(r not in H_t | r contradicted at t AND r in H_{t-1})
        """
        compliant = 0
        total = 0
        for i, t in enumerate(turns):
            hyps = t.get("hypotheses")
            if hyps is None:
                continue
            agent_set = set(hyps)
            prev_set = set(turns[i - 1]["hypotheses"]) if i > 0 and turns[i - 1].get("hypotheses") else set(RULE_NAMES)
            env = t.get("env_record", {})
            preds = env.get("rule_predictions", {})
            result = env.get("formal_feedback")
            if not preds or not result:
                continue
            for rule_name, pred in preds.items():
                if pred != result and rule_name in prev_set:
                    # This rule is contradicted AND was in previous hypotheses
                    total += 1
                    if rule_name not in agent_set:
                        compliant += 1
        if total == 0:
            return None
        return round(compliant / total, 4)

    def _compute_cumulative_consistency(self, turns: List[Dict]) -> Optional[float]:
        """Cumulative Consistency (CC): average Jaccard similarity between
        agent hypotheses and ground truth surviving set across all turns."""
        jaccards = []
        for t in turns:
            hyps = t.get("hypotheses")
            gt = t.get("gt_survivors")
            if hyps is None or gt is None:
                continue
            agent_set = set(hyps)
            gt_set = set(gt)
            union = agent_set | gt_set
            if not union:
                jaccards.append(1.0)
            else:
                jaccards.append(len(agent_set & gt_set) / len(union))
        if not jaccards:
            return None
        return round(sum(jaccards) / len(jaccards), 4)

    def _compute_retraction_metrics(self, turns: List[Dict]) -> Dict[str, Any]:
        """Compute retraction-specific belief metrics.

        Key metrics:
        - reinstatement_success: did the agent correctly broaden its hypothesis
          set after a retraction (add back rules that should be reinstated)?
        - retraction_failed_update: rules that should have been reinstated but weren't
          (agent's belief resists correction)
        - retraction_over_correction: rules that were correctly eliminated by
          non-retracted evidence but incorrectly reinstated
        """
        if not self.is_retraction_env:
            return {}

        metrics: Dict[str, Any] = {}
        for t in turns:
            env = t.get("env_record", {})
            if env.get("event_type") != "retraction":
                continue
            hyps = t.get("hypotheses")
            if hyps is None:
                metrics["retraction_parse_error"] = True
                continue

            agent_set = set(hyps)
            gt_set = set(t.get("gt_survivors", []))
            reinstated_expected = set(env.get("reinstated_rules", []))
            pre_retraction = set(env.get("pre_retraction_survivors", []))

            # Did the agent reinstate the rules it should have?
            reinstated_actual = agent_set - pre_retraction
            reinstated_correct = reinstated_expected & agent_set
            reinstated_missed = reinstated_expected - agent_set  # belief failed_update!

            # Did the agent over-correct (reinstate rules that should stay eliminated)?
            over_correction = agent_set - gt_set

            metrics["retraction_turn"] = t["turn"]
            metrics["reinstated_expected"] = sorted(reinstated_expected)
            metrics["reinstated_actual"] = sorted(reinstated_actual)
            metrics["reinstated_correct"] = sorted(reinstated_correct)
            metrics["reinstated_missed"] = sorted(reinstated_missed)
            metrics["over_correction"] = sorted(over_correction)
            metrics["reinstatement_rate"] = (
                round(len(reinstated_correct) / len(reinstated_expected), 4)
                if reinstated_expected else None
            )
            metrics["retraction_failed_update"] = len(reinstated_missed) > 0
            metrics["retraction_exact_match"] = (agent_set == gt_set)

        return metrics
