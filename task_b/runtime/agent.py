"""Agent prompt construction for Scenario B circuit diagnosis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from task_b.domain.rule_engine import CIRCUIT_ARCHITECTURES, get_topology


def _fault_space_text(circuit_type: str) -> str:
    topology = get_topology(circuit_type)
    return "\n".join(
        f"  {fault_id}: {description}"
        for fault_id, description in topology.fault_options.items()
    )


def _circuit_configuration_text(arch: Dict[str, Any]) -> str:
    components = ", ".join(
        str(component).upper() if str(component).startswith("r") else str(component).title()
        for component in arch.get("components", [])
    )
    topology = str(arch.get("topology", ""))
    if topology == "series":
        return f"{arch['name']} with components {components}, connected in one series loop."
    if topology == "parallel_r12_series":
        return (
            f"{arch['name']} with components {components}. "
            "The battery and switch are in the main branch, and R1 and R2 form parallel branches."
        )
    if topology == "parallel_r123_series":
        return (
            f"{arch['name']} with components {components}. "
            "The battery and switch are in the main branch, and R1/R2/R3 form three parallel branches."
        )
    if topology == "parallel_series_pairs":
        return (
            f"{arch['name']} with components {components}. "
            "The battery and switch are in the main branch. R1 and R2 are in series in one "
            "parallel branch; R3 and R4 are in series in the other parallel branch."
        )
    return f"{arch['name']} with components {components}."


def _rule_key_to_display(key: str) -> str:
    """Convert an abnormal_support key (e.g. 'ammeter_r1_absence') to the display label the
    model sees in turn messages (e.g. 'Current (R1): = 0')."""
    base, value_token = key.rsplit("_", 1)
    label_map = {
        "ammeter": "Current",
        "ammeter_main": "Current (Main)",
        "ammeter_r1": "Current (R1)",
        "ammeter_r3": "Current (R3)",
        "ammeter_r4": "Current (R4)",
        "ammeter_r2": "Current (R2)",
        "voltmeter_battery": "Voltage (Battery)",
    }
    if base in label_map:
        label = label_map[base]
    elif base.startswith("voltmeter_"):
        comp = base.split("_", 1)[1].upper()
        label = f"Voltage ({comp})"
    else:
        label = base.replace("_", " ").title()
    value_str = "= 0" if value_token == "absence" else "> 0"
    return f"{label}: {value_str}"


def _build_fault_behavior_guide(circuit_type: str) -> str:
    """Build a fault-centric reference with one reading per line.

    Unlike the consistency guide, this does not map a measurement directly to a
    survivor set. The model must still check candidate faults against the active
    readings, which is closer to a diagnostic reference than an answer table.
    The multi-line layout avoids packing similar E/F/G readings into one long
    sentence, which made the model misread fields in long thinking traces.
    """
    topology = get_topology(circuit_type)
    by_fault: dict[str, list[str]] = {fault_id: [] for fault_id in topology.fault_ids}
    for key, survivors in topology.abnormal_support.items():
        display = _rule_key_to_display(key)
        for fault_id in sorted(survivors):
            by_fault[fault_id].append(display)

    lines = [
        "Fault behavior reference — powered readings each fault can be consistent with:",
    ]
    for fault_id in topology.fault_ids:
        lines.append(f"  {fault_id}:")
        readings = by_fault[fault_id]
        if not readings:
            lines.append("    - no listed powered abnormal reading")
            continue
        for reading in readings:
            lines.append(f"    - {reading}")
    return "\n".join(lines)


def _benchmark_rule_policy_text(circuit_type: str) -> str:
    lines = [
        "Benchmark setup:",
        "- All measurements in this benchmark are modeled as non-invasive",
        "  powered-circuit observations: the circuit stays assembled while an",
        "  instrument reads a branch/component state.",
    ]
    return "\n".join(lines) + "\n"


_OUTPUT_FORMAT_THINK_BLOCK = """\
Output format (strict):
<hypothesis>fault_id_1, fault_id_2</hypothesis>

- Include exactly one `<hypothesis>...</hypothesis>` tag in every reply.
- Include exactly one `<think>...</think>` block before `<hypothesis>`.
- `<hypothesis>` must contain comma-separated fault IDs from the listed fault space, or `none`.
- If you reason before answering, put the final `<hypothesis>` tag on the last line."""


_OUTPUT_FORMAT_HYPOTHESIS_ONLY = """\
Output format (strict):
<hypothesis>fault_id_1, fault_id_2</hypothesis>

- Include exactly one `<hypothesis>...</hypothesis>` tag in every reply.
- Do not output explanatory text outside the `<hypothesis>` tag.
- `<hypothesis>` must contain comma-separated fault IDs from the listed fault space, or `none`."""


def _measurement_semantics_text(circuit_type: str) -> str:
    topology = get_topology(circuit_type)
    has_extra_branch_resistors = "r3" in set(topology.components)
    has_series_pair_branches = "r4" in set(topology.components)

    lines = [
        "Measurement semantics:",
        "- Current (Main) is current through the main branch.",
    ]
    if has_series_pair_branches:
        lines.extend([
            "- Current (R1) and Current (R2) are the same series-branch current",
            "  through the R1--R2 branch.",
            "- Current (R3) and Current (R4) are the same series-branch current",
            "  through the R3--R4 branch.",
            "- Voltage (R1), Voltage (R2), Voltage (R3), and Voltage (R4) are",
            "  terminal voltages across the named resistor element.",
        ])
    else:
        lines.extend([
            "- Current (R1) and Current (R2) are branch currents through the R1/R2",
            "  parallel branches.",
        ])
        if has_extra_branch_resistors:
            lines.extend([
                "- Current (R3) is branch current through the third parallel branch.",
            ])
        lines.extend([
            "- Voltage (R1), Voltage (R2), and Voltage (Battery) are terminal voltages",
            "  measured across the named component or branch terminals.",
        ])
        if has_extra_branch_resistors:
            lines.extend([
                "- Voltage (R3) is the terminal voltage across R3.",
            ])
    if has_series_pair_branches:
        lines.extend([
            "- Voltage (Battery) is measured across the battery terminals.",
        ])
    lines.extend([
        "- For ammeters and voltmeters: = 0 means no measurable current/voltage;",
        "  > 0 means nonzero current/voltage.",
    ])
    return "\n".join(lines)


def build_system_prompt(
    circuit_type: str,
    prompt_style: str = "neutral",
    symptom: Optional[Dict[str, Any]] = None,
) -> str:
    arch = CIRCUIT_ARCHITECTURES[circuit_type]
    circuit_text = _circuit_configuration_text(arch)
    fault_list = _fault_space_text(circuit_type)
    fault_behavior_guide = _build_fault_behavior_guide(circuit_type)
    benchmark_rule_policy = _benchmark_rule_policy_text(circuit_type)
    measurement_semantics = _measurement_semantics_text(circuit_type)

    symptom_block = ""
    if symptom and symptom.get("description"):
        symptom_block = f"Observed Condition:\n{symptom['description']}\n\n"

    if prompt_style == "noise":
        topology = get_topology(circuit_type)
        query_space = sorted(topology.abnormal_support.keys())
        allowed_keys_text = "\n".join(f"  - {key}" for key in query_space)
        noise_prompt = (
            "You are diagnosing a circuit with one hidden fault from a fixed fault space.\n\n"
            "## Background\n"
            "Each turn you choose one measurement key from the allowed list below.\n"
            "The environment tells you whether that measurement is abnormal (YES) or normal (NO) for the circuit.\n"
            "Use the answers to eliminate fault candidates until only one remains, then declare it.\n\n"
            "## Required output format\n"
            "Every reply MUST contain BOTH of the following tags (in this order):\n\n"
            "  1. <hypothesis>FAULT_ID_1, FAULT_ID_2, ...</hypothesis>\n"
            "     List every fault ID (single uppercase letter) still consistent with all evidence.\n"
            "     Start with ALL fault IDs and remove each one as evidence rules it out.\n"
            "     Use only the uppercase single-letter IDs from the fault space below.\n\n"
            "  2. ONE of the following (choose based on how many fault candidates remain):\n"
            "     a) Still narrowing down — output a measurement query:\n"
            "           <measure>measurement_key</measure>\n"
            "        measurement_key must be exactly one key from the allowed list below.\n"
            "        Choose a key whose YES/NO answer will eliminate at least one remaining fault.\n"
            "     b) Ready to commit — exactly one fault candidate remains (or you are certain):\n"
            "           <terminate>final</terminate>\n"
            "        In this case <hypothesis> must contain exactly ONE fault ID.\n\n"
            "## Concrete output examples\n"
            "Example — still narrowing down (faults D, E, F remain):\n"
            "  <hypothesis>D, E, F</hypothesis>\n"
            "  <measure>ammeter_r1_presence</measure>\n\n"
            "Example — ready to commit (only fault E remains):\n"
            "  <hypothesis>E</hypothesis>\n"
            "  <terminate>final</terminate>\n\n"
            "## Turn-message notes\n"
            "- Each environment reply contains: Turn N, one verification line, fault predictions, and optionally one host-comment line.\n"
            "- The verification line reads: Environment verification: `measurement_key` -> YES  (or NO).\n"
            "- Below the verification line, the environment shows each fault ID's prediction for your measurement:\n"
            "    Fault predictions for this measurement:\n"
            "      - D -> YES\n"
            "      - E -> NO\n"
            "      ...\n"
            "  A fault whose prediction matches the verified answer is still consistent.\n"
            "  A fault whose prediction differs from the verified answer is eliminated.\n"
            "- If your <measure> key is not in the allowed list, no measurement is recorded for that turn.\n"
            "  In that case re-read the allowed keys and output a valid key next turn.\n\n"
            f"{symptom_block}"
            f"Fault Space:\n{fault_list}\n\n"
            f"Circuit Configuration:\n{circuit_text}\n\n"
            f"{measurement_semantics}\n\n"
            f"{fault_behavior_guide}\n\n"
            f"{benchmark_rule_policy}\n"
            f"Allowed measurement keys:\n{allowed_keys_text}\n\n"
        )
        return noise_prompt

    if prompt_style in {"neutral", "neutral_no_think", "minimal_no_think"}:
        if prompt_style == "minimal_no_think":
            minimal_prompt = (
                "You are diagnosing a circuit-fault benchmark.\n"
                "Exactly one hidden fault exists.\n\n"
                "Turns are cumulative. A later turn does not restart the diagnosis.\n"
                "Keep all previous measurements active unless a correction explicitly "
                "says to discard or replace one of them.\n\n"
                f"Fault IDs:\n{fault_list}\n\n"
                "Each turn may include a fault-prediction table. Use it to find which "
                "faults match the current measurements, then maintain the cumulative "
                "set of faults consistent with all active measurements.\n\n"
                "Examples for later turns:\n"
                "- Previous <hypothesis>A, B, E, G</hypothesis>; Current matching "
                "fault IDs: C, D, E, F, G; next output is "
                "<hypothesis>E, G</hypothesis>.\n"
                "- Previous <hypothesis>A, B, D, G</hypothesis>; Current matching "
                "fault IDs: C, D, G; next output is <hypothesis>D, G</hypothesis>.\n"
                "- Previous <hypothesis>B, E, F</hypothesis>; Current matching "
                "fault IDs: C, E, F; next output is <hypothesis>E, F</hypothesis>.\n"
                "- Previous <hypothesis>B, D, G</hypothesis>; Current matching "
                "fault IDs: C, D, F; next output is <hypothesis>D</hypothesis>.\n"
                "- Previous <hypothesis>B, E, G</hypothesis>; Current matching "
                "fault IDs: C, D, G; next output is <hypothesis>G</hypothesis>.\n"
                "- Previous <hypothesis>B, E, F</hypothesis>; Current matching "
                "fault IDs: C, D, E; next output is <hypothesis>E</hypothesis>.\n"
                "- Previous <hypothesis>A, B, E, G</hypothesis>; Current matching "
                "fault IDs: C, D, G; next output is <hypothesis>G</hypothesis>.\n"
                "- Previous <hypothesis>A, B, E, G</hypothesis>; Current matching "
                "fault IDs: C, E, F; next output is <hypothesis>E</hypothesis>.\n"
                "- Previous <hypothesis>E, G</hypothesis>; Current matching fault "
                "IDs: B, E, G; next output is <hypothesis>E, G</hypothesis>.\n\n"
                "Output format (strict):\n"
                "<hypothesis>fault_id_1, fault_id_2</hypothesis>\n\n"
                "- Include exactly one `<hypothesis>...</hypothesis>` tag in every reply.\n"
                "- Do not output explanatory text outside the `<hypothesis>` tag.\n"
                "- `<hypothesis>` must contain comma-separated fault IDs from the listed "
                "fault IDs, or `none`.\n"
            )
            return minimal_prompt

        neutral_prompt = (
            "You are diagnosing a circuit-fault benchmark.\n\n"
            "Exactly one hidden fault exists, but the diagnosis may take several\n"
            "turns to narrow down.\n"
            "Each turn you receive a new measurement result. Update your assessment\n"
            "of which fault IDs are consistent with the evidence. Each new reading\n"
            "may clarify the diagnosis and call for reassessment.\n\n"
            f"{symptom_block}"
            f"Fault Space:\n{fault_list}\n\n"
            f"Circuit Configuration:\n{circuit_text}\n\n"
            f"{measurement_semantics}\n\n"
            f"{fault_behavior_guide}\n\n"
            f"{benchmark_rule_policy}\n"
            "At each turn, output every fault ID that remains consistent with the\n"
            "measurement evidence. If no fault remains\n"
            "consistent, output `none`.\n\n"
            "Important: turns are cumulative. A later turn does not restart the\n"
            "diagnosis. Keep all previous measurements active unless a correction\n"
            "explicitly says to discard or replace one of them.\n\n"
        )

        output_format = (
            _OUTPUT_FORMAT_HYPOTHESIS_ONLY
            if prompt_style == "neutral_no_think"
            else _OUTPUT_FORMAT_THINK_BLOCK
        )
        return neutral_prompt + f"{output_format}\n"

    raise ValueError(
        f"Unsupported prompt_style '{prompt_style}'. "
        "Supported styles are: neutral, neutral_no_think, minimal_no_think, noise."
    )
def _format_measurement_label(key: str) -> str:
    if key == "ammeter_main":
        return "Current (Main)"
    if key == "ammeter_r1":
        return "Current (R1)"
    if key == "ammeter_r3":
        return "Current (R3)"
    if key == "ammeter_r4":
        return "Current (R4)"
    if key == "ammeter_r2":
        return "Current (R2)"
    if key == "ammeter":
        return "Current"
    if key == "voltmeter_battery":
        return "Voltage (Battery)"
    if key.startswith("voltmeter_"):
        return f"Voltage ({key.split('_', 1)[1].upper()})"
    if key == "isolation_switch":
        return "Switch State Check"
    if key.startswith("isolation_"):
        return f"Isolation Check ({key.split('_', 1)[1].upper()})"
    if key.startswith("ohmmeter_"):
        return f"Isolation Check ({key.split('_', 1)[1].upper()})"
    return key.replace("_", " ").title()


def _format_measurement_value(key: str, value: str) -> str:
    norm = value.strip().lower()
    if key == "isolation_switch":
        if norm in ("open", "open_path", "open_circuit", "infinite"):
            return "Open"
        if norm in ("short", "short_path", "short_circuit", "0ohm"):
            return "Closed"
        if norm in ("healthy", "normal", "good"):
            return "Healthy"
    if norm in ("absence", "0", "0a", "0v", "none", "off"):
        return "= 0"
    if norm in ("presence", "battery_voltage", "high_current", "on", "ok"):
        return "> 0"
    if norm in ("open", "open_path", "open_circuit", "infinite"):
        return "Open path"
    if norm in ("short", "short_path", "short_circuit", "0ohm"):
        return "Short path"
    if norm in ("healthy", "normal", "good"):
        return "Healthy"
    return value


def _prediction_value_for_fault(
    circuit_type: str,
    fault_id: str,
    measurement_key: str,
) -> str:
    topology = get_topology(circuit_type)
    presence_key = f"{measurement_key}_presence"
    absence_key = f"{measurement_key}_absence"
    predicts_presence = fault_id in topology.abnormal_support.get(presence_key, ())
    predicts_absence = fault_id in topology.abnormal_support.get(absence_key, ())

    if predicts_presence and predicts_absence:
        return "> 0 or = 0"
    if predicts_presence:
        return "> 0"
    if predicts_absence:
        return "= 0"
    return ""


def _format_prefix_rule_prediction(event: Optional[Dict[str, Any]]) -> str:
    if not event:
        return ""

    prediction = event.get("prefix_rule_prediction")
    if not isinstance(prediction, dict):
        return ""

    circuit_type = str(prediction.get("circuit_type") or "")
    measurements = prediction.get("measurements")
    if not circuit_type or not isinstance(measurements, dict) or not measurements:
        return ""

    topology = get_topology(circuit_type)
    measured_keys = []
    for key in measurements:
        measurement_key = str(key)
        if (
            f"{measurement_key}_presence" in topology.abnormal_support
            or f"{measurement_key}_absence" in topology.abnormal_support
        ):
            measured_keys.append(measurement_key)
    if not measured_keys:
        return ""

    lines = ["- Reference table for measured keys so far:"]
    for fault_id in topology.fault_ids:
        readings = [
            f"{_format_measurement_label(key)}: "
            f"{_prediction_value_for_fault(circuit_type, fault_id, key) or 'not listed'}"
            for key in measured_keys
        ]
        lines.append(f"  - {fault_id}: " + "; ".join(readings))
    return "\n".join(lines)


def _format_event_context(event: Optional[Dict[str, Any]]) -> list[str]:
    """Return no per-turn setup text; the powered setup is fixed in the system prompt."""
    return []


def _format_evidence_block(
    measurements: Dict[str, str],
    event: Optional[Dict[str, Any]],
) -> str:
    """Combine context lines and measurements into one consistent bullet list."""
    context_lines = _format_event_context(event)
    measurement_blocks = list((event or {}).get("measurement_blocks") or [])
    if len(measurement_blocks) > 1:
        lines = ["- Measurement blocks:"]
        for index, block in enumerate(measurement_blocks, start=1):
            lines.append(f"  - Block {index}:")
            for key, value in block.items():
                lines.append(
                    f"    - {_format_measurement_label(key)}: "
                    f"{_format_measurement_value(key, value)}"
                )
        meas_block = "\n".join(lines)
    elif measurements:
        meas_lines = [
            f"  - {_format_measurement_label(key)}: {_format_measurement_value(key, value)}"
            for key, value in measurements.items()
        ]
        meas_block = "- Measurements:\n" + "\n".join(meas_lines)
    else:
        meas_block = "- Measurements:\n  - No new measurements"
    prediction_block = _format_prefix_rule_prediction(event)
    blocks = context_lines + [meas_block]
    if prediction_block:
        blocks.append(prediction_block)
    return "\n".join(blocks)


def _format_fault_predictions_block(
    fault_predictions: Optional[Dict[str, str]],
    current_matching_faults: Optional[List[str]] = None,
    prompt_style: str | None = None,
) -> str:
    if not fault_predictions:
        return ""
    pred_lines = "\n".join(
        f"  - {fid} -> {pred}"
        for fid, pred in sorted(fault_predictions.items())
    )
    matches_line = ""
    if current_matching_faults is not None:
        matches = ", ".join(current_matching_faults) if current_matching_faults else "none"
        matches_line = f"\nCurrent matching fault IDs: {matches}\n"
    return (
        "Fault predictions for these measurements:\n"
        f"{pred_lines}\n"
        f"{matches_line}"
        "For this turn, matching faults are the rows whose prediction matches "
        "these measurements."
    )


def build_turn_message(
    turn: int,
    measurements: Dict[str, str],
    event_type: str = "measurement",
    retract_turn: int | None = None,
    final_turn: bool = False,
    event: Optional[Dict[str, Any]] = None,
    prompt_style: str | None = None,
    add_host_comment: bool = True,
    fault_predictions: Optional[Dict[str, str]] = None,
    current_matching_faults: Optional[List[str]] = None,
) -> str:
    """Build the user-facing message for the agent."""

    evidence_block = _format_evidence_block(measurements, event)
    predictions_block = _format_fault_predictions_block(
        fault_predictions,
        current_matching_faults=current_matching_faults,
        prompt_style=prompt_style,
    )
    if predictions_block:
        evidence_block = f"{evidence_block}\n\n{predictions_block}"

    discard_str = (
        f"Discard the measurement from Turn {retract_turn}"
        if retract_turn is not None
        else "Discard the earlier measurement"
    )

    if event_type == "noise_start":
        if add_host_comment:
            msg = (
                "Turn 0:\n\n"
                "- Session start.\n"
                "- Host comment: Follow the task protocol in the system prompt."
            )
        else:
            msg = "Turn 0:\n\n- Session start."
    elif event_type == "noise_feedback":
        info = event or {}
        query_key = str(info.get("query_key", ""))
        answer = str(info.get("answer", "no")).upper()
        host_comment = str(info.get("host_comment", ""))
        query_valid = bool(info.get("query_valid", False))

        if query_valid:
            verification = f"- Environment verification: `{query_key}` -> {answer}"

            predictions_block = ""
            if fault_predictions:
                pred_lines = "\n".join(
                    f"  - {fid} -> {pred}"
                    for fid, pred in sorted(fault_predictions.items())
                )
                predictions_block = f"\n\nFault predictions for this measurement:\n{pred_lines}"

            if add_host_comment and host_comment:
                msg = (
                    f"Turn {turn}:\n\n"
                    f"{verification}"
                    f"{predictions_block}\n"
                    f"- {host_comment}"
                )
            else:
                msg = f"Turn {turn}:\n\n{verification}{predictions_block}"
        else:
            msg = (
                f"Turn {turn}:\n\n"
                "- Environment prompt: This question is invalid. Please rephrase your query."
            )
    elif turn == 0:
        msg = (
            f"Turn 0:\n\n"
            f"{evidence_block}\n\n"
            "Initialize the consistent fault IDs from this measurement."
        )
    elif event_type == "retraction_measurement":
        msg = (
            f"Turn {turn} (CORRECTION):\n"
            f"{discard_str} and replace it with these corrected measurements.\n\n"
            f"{evidence_block}\n\n"
            "Turn 1 is discarded. Ignore your last <hypothesis>. Use the Turn 0 "
            "matching fault IDs and the Current matching fault IDs above. Keep only "
            "IDs that appear in both."
        )
    elif event_type == "pure_retraction":
        msg = (
            f"Turn {turn} (CORRECTION):\n"
            f"{discard_str} entirely. Do not add any new measurements.\n\n"
            "Recompute the consistent fault IDs from all active measurements."
        )
    else:
        msg = (
            f"Turn {turn}:\n\n"
            f"{evidence_block}\n\n"
            "Read your last <hypothesis>. Keep only IDs that also appear in "
            "Current matching fault IDs. Remove all other IDs. Do not add new IDs."
        )

    return msg
