"""Quick model validation: run every (circuit_type, fault, template_idx) through 3 repeats
and check T1 oracle exclusion rate. Flag any template where exclusion < 0.6.

Usage:
    python -m task_b.experiments.validate_templates \
        --model-path models/Qwen2.5-7B-Instruct \
        --gpu 0 --repeats 3
"""

import argparse
import os
from typing import Any, Dict, List

from task_b.domain.faults import FAULT_IDS
from task_b.domain.rule_engine import CIRCUIT_TYPES
from task_b.runtime.agent import build_system_prompt, build_turn_message
from task_b.runtime.environment import ChallengeSequence
from task_b.runtime.orchestrator import parse_hypotheses
from utils.io import save_json


def _parse_hypotheses(text: str) -> List[str]:
    return parse_hypotheses(text) or []


def validate_template(backend, circuit_type: str, oracle: str, tpl_idx: int,
                      repeats: int, temperature: float = 0.3) -> Dict[str, Any]:
    challenge = ChallengeSequence(circuit_type, oracle, tpl_idx)
    ch = challenge.as_dict()
    t0_meas = ch["events"][0]["measurements"]
    t1_meas = ch["events"][1]["measurements"]
    t0_gt = sorted(ch["ground_truth"][0]["survivors"])
    t1_gt = sorted(ch["ground_truth"][1]["survivors"])

    system_prompt = build_system_prompt(circuit_type, "neutral",
                                        symptom=challenge.symptom)
    t0_msg = build_turn_message(0, t0_meas, "initial_measurement")
    t1_msg = build_turn_message(1, t1_meas, "misleading_measurement")

    oracle_excluded = 0
    t0_oracle_present = 0
    rows = []
    for _ in range(repeats):
        msgs = [{"role": "system", "content": system_prompt}]
        msgs.append({"role": "user", "content": t0_msg})
        r0 = backend.chat_completion(msgs, temperature=temperature, max_tokens=128)
        h0 = set(_parse_hypotheses(r0))
        msgs.append({"role": "assistant", "content": r0})

        msgs.append({"role": "user", "content": t1_msg})
        r1 = backend.chat_completion(msgs, temperature=temperature, max_tokens=128)
        h1 = set(_parse_hypotheses(r1))

        t0_ok = oracle in h0
        t1_excl = oracle not in h1
        t0_oracle_present += int(t0_ok)
        oracle_excluded += int(t1_excl)
        rows.append({"t0_hyps": sorted(h0), "t1_hyps": sorted(h1),
                     "t0_oracle_ok": t0_ok, "t1_excl": t1_excl})

    excl_rate = oracle_excluded / repeats
    t0_rate = t0_oracle_present / repeats
    label = f"{circuit_type}:{oracle}:tpl{tpl_idx}"
    status = "✓ OK" if excl_rate >= 0.6 else "✗ WARN"
    t1_key = list(t1_meas.keys())[0]
    t1_val = list(t1_meas.values())[0]
    print(f"  {label:<30} T0_rate={t0_rate:.0%}  T1_excl={excl_rate:.0%}  "
          f"T1={t1_key}={t1_val:<10}  {status}")
    return {
        "label": label, "circuit_type": circuit_type, "oracle": oracle,
        "tpl_idx": tpl_idx, "t0_rate": t0_rate, "t1_excl_rate": excl_rate,
        "t0_gt": t0_gt, "t1_gt": t1_gt,
        "misleading_target": ch["misleading_target"],
        "t1_measurement": t1_meas, "status": status, "details": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default=os.environ.get("MODEL_PATH", "models/Qwen2.5-7B-Instruct"))
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=str,
                        default="task_b/outputs/template_validation.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from utils.llm_backend import VLLMBackend
    print(f"Loading model: {args.model_path}")
    backend = VLLMBackend(
        model_path=args.model_path,
        dtype="bf16",
        max_model_len=8192,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        language_model_only=True,
    )

    all_results = []
    warnings = []

    for ct in CIRCUIT_TYPES:
        print(f"\n{'='*60}\n{ct}\n{'='*60}")
        for fault in FAULT_IDS:
            idxs = ChallengeSequence.get_template_indices(ct, fault)
            for i in idxs:
                r = validate_template(backend, ct, fault, i, args.repeats)
                all_results.append(r)
                if r["t1_excl_rate"] < 0.6:
                    warnings.append(r["label"])

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(all_results)} templates tested, {len(warnings)} warnings")
    if warnings:
        print("WARNING templates (T1 oracle exclusion < 60%):")
        for w in warnings:
            print(f"  {w}")
    else:
        print("All templates pass T1 oracle exclusion check!")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_json(args.output, {"warnings": warnings, "results": all_results})
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
