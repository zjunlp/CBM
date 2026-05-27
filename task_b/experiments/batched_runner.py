"""Batched vLLM runner for Scenario B experiment sessions."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from task_b.runtime.agent import build_system_prompt, build_turn_message
from task_b.runtime.orchestrator import parse_hypotheses


_HYPOTHESIS_TAG_RE = re.compile(
    r"<hypothesis>\s*(.*?)\s*</hypothesis>",
    flags=re.IGNORECASE | re.DOTALL,
)
_ASSISTANT_CONTEXT_CHAR_LIMIT = 4000


def _compact_assistant_history(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Keep prior assistant turns bounded when replaying chat history to vLLM.

    Full thinking traces are stored in the trajectory, but replaying them into
    later turns can exceed the model context. For belief failed_update, the relevant
    public state is the previous final hypothesis.
    """
    compacted: List[Dict[str, str]] = []
    for message in messages:
        if message.get("role") != "assistant":
            compacted.append(message)
            continue

        content = message.get("content", "")
        matches = list(_HYPOTHESIS_TAG_RE.finditer(content))
        if matches:
            hypothesis_tag = matches[-1].group(0)
            compacted.append({
                "role": "assistant",
                "content": " ".join(hypothesis_tag.split()),
            })
            continue

        if len(content) > _ASSISTANT_CONTEXT_CHAR_LIMIT:
            content = content[-_ASSISTANT_CONTEXT_CHAR_LIMIT:]
        compacted.append({"role": "assistant", "content": content.strip()})
    return compacted


def run_sessions_batched(
    *,
    run_label: str,
    sessions: List[Dict[str, Any]],
    backend: Any,
    temperature: float,
    max_tokens: int,
    checkpoint_dir: Optional[str] = None,
) -> None:
    """Run active sessions turn-by-turn in one vLLM batch per turn."""
    for session in sessions:
        orchestrator = session["orchestrator"]
        challenge = orchestrator.challenge
        session["turns"] = []
        session["conversation"] = [
            {
                "role": "system",
                "content": build_system_prompt(
                    challenge.circuit_type,
                    orchestrator.prompt_style,
                    symptom=getattr(challenge, "symptom", None),
                ),
            },
        ]
        session["max_turns"] = challenge.total_turns

    max_turns = max((session["max_turns"] for session in sessions), default=0)
    for turn_idx in range(max_turns):
        active_sessions: List[Dict[str, Any]] = []
        messages_batch: List[List[Dict[str, str]]] = []

        for session in sessions:
            if turn_idx >= session["max_turns"]:
                continue

            orchestrator = session["orchestrator"]
            challenge = orchestrator.challenge
            conversation = session["conversation"]

            env_record = orchestrator.env.step(turn_idx)
            event = env_record["event"]
            user_msg = build_turn_message(
                turn=turn_idx,
                measurements=event["measurements"],
                event_type=event.get("type", "measurement"),
                retract_turn=event.get("retract_turn"),
                final_turn=turn_idx == challenge.total_turns - 1,
                event=event,
                prompt_style=orchestrator.prompt_style,
                fault_predictions=orchestrator.fault_predictions_for_measurements(
                    event["measurements"]
                ),
                current_matching_faults=orchestrator.current_matching_faults_for_measurements(
                    event["measurements"]
                ),
            )
            user_msg = orchestrator.maybe_inject_failed_isolation_comment(user_msg, turn_idx, event)
            conversation.append({"role": "user", "content": user_msg})

            session["pending_env_record"] = env_record
            active_sessions.append(session)
            context_messages = orchestrator.build_context_messages(conversation)
            messages_batch.append(_compact_assistant_history(context_messages))

        if not messages_batch:
            continue

        print(
            f"[vLLM] {run_label}: turn {turn_idx + 1}/{max_turns}, "
            f"prompts={len(messages_batch)}",
            flush=True,
        )
        responses = backend.batch_chat_completion(
            messages_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            use_tqdm=True,
        )

        for session, agent_response in zip(active_sessions, responses):
            _append_turn_result(
                session=session,
                turn_idx=turn_idx,
                agent_response=agent_response,
            )

        if checkpoint_dir:
            _write_turn_checkpoint(
                checkpoint_dir=checkpoint_dir,
                run_label=run_label,
                turn_idx=turn_idx,
                max_turns=max_turns,
                sessions=active_sessions,
            )

    for session in sessions:
        orchestrator = session["orchestrator"]
        challenge = orchestrator.challenge
        turns = session["turns"]
        session["trajectory"] = {
            "mode": "failed_isolation" if orchestrator.failed_isolation_options else "failed_stay",
            "circuit_type": challenge.circuit_type,
            "oracle": challenge.oracle,
            "turns": turns,
            "conversation": session["conversation"],
            "n_turns_played": len(turns),
        }


def _append_turn_result(
    *,
    session: Dict[str, Any],
    turn_idx: int,
    agent_response: str,
) -> None:
    orchestrator = session["orchestrator"]
    challenge = orchestrator.challenge
    env_record = session.pop("pending_env_record")
    conversation = session["conversation"]

    conversation.append({"role": "assistant", "content": agent_response})
    hypotheses = parse_hypotheses(agent_response, orchestrator.valid_faults)
    gt = orchestrator.env.get_ground_truth_at_step(turn_idx)
    gt_survivors = set(gt["survivors"])
    agent_set = set(hypotheses) if hypotheses else set()

    session["turns"].append({
        "turn": turn_idx,
        "agent_response": agent_response,
        "hypotheses": hypotheses,
        "gt_survivors": env_record["gt_survivors"],
        "belief_metrics": {
            "oracle_in_hypotheses": challenge.oracle in agent_set,
            "exact_match": agent_set == gt_survivors,
            "false_retention": sorted(agent_set - gt_survivors),
            "false_elimination": sorted(gt_survivors - agent_set),
            "agent_set_size": len(agent_set),
            "gt_set_size": len(gt_survivors),
        },
        "parse_error": hypotheses is None,
    })


def _write_turn_checkpoint(
    *,
    checkpoint_dir: str,
    run_label: str,
    turn_idx: int,
    max_turns: int,
    sessions: List[Dict[str, Any]],
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"turn_{turn_idx + 1:02d}.jsonl")
    tmp_path = f"{path}.tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        for session in sessions:
            orchestrator = session["orchestrator"]
            challenge = orchestrator.challenge
            conversation = session.get("conversation", [])
            user_message = ""
            if len(conversation) >= 2 and conversation[-2].get("role") == "user":
                user_message = conversation[-2].get("content", "")

            row = {
                "run_label": run_label,
                "turn": turn_idx,
                "turn_number": turn_idx + 1,
                "max_turns": max_turns,
                "task_label": session.get("task_label"),
                "point_index": session.get("point_index"),
                "repeat_index": session.get("repeat_index"),
                "circuit_type": challenge.circuit_type,
                "oracle": challenge.oracle,
                "user_message": user_message,
                "turn_result": session["turns"][-1],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    os.replace(tmp_path, path)
    print(
        f"[checkpoint] saved turn {turn_idx + 1}/{max_turns}: "
        f"{path} rows={len(sessions)}",
        flush=True,
    )
