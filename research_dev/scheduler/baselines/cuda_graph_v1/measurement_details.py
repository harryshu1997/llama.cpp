#!/usr/bin/env python3
"""Extract reporting details from the validated, immutable reference results."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text())


def request_details(row, result):
    receipt = row["terminal_ticket"]["execution_receipt"]
    command = row["execution_command"]
    parameters = command["adapter_parameters"]
    cpu_layers = sorted(int(op["operator_id"].split(":")[1])
                        for op in command["operator_plan"]["operators"]
                        if op["operator_kind"] == "ffn"
                        and parameters["cpu_device_id"] in op["device_ids"]
                        and parameters["gpu_device_id"] not in op["device_ids"])
    events = [e for e in result["request_helper_events"]
              if e.get("request_id") == row["request_id"]]
    reasons = Counter(e.get("reason", "UNSPECIFIED") for e in events
                      if e["kind"] == "ASSISTANCE_DECISION" and e.get("selected_fraction_ppm") == 0)
    first_reasons = {}
    for event in events:
        if event["kind"] == "ASSISTANCE_DECISION" and event.get("selected_fraction_ppm") == 0:
            first_reasons.setdefault(event.get("reason", "UNSPECIFIED"), {
                key: event.get(key) for key in (
                    "observed_at_us", "token_index", "remaining_output_tokens",
                    "remaining_probe_tokens", "incumbent_fraction_ppm",
                )
            })
    return {
        "request_id": row["request_id"], "output_tokens": row["output_tokens"],
        "service_latency_us": row["actual_latency_us"],
        "arrival_to_start_us": receipt["started_us"] - row["replay_arrival_us"],
        "arrival_to_completion_us": receipt["finished_us"] - row["replay_arrival_us"],
        "arrival_to_start_definition": "Includes queueing and preparation, not pure queue wait.",
        "desktop_parent_sha256": row["execution_command"]["operator_plan"]["desktop_placement_sha256"],
        "desktop_cpu_ffn_layers": cpu_layers,
        "desktop_launch": row["execution_command"]["adapter_parameters"],
        "fraction_history": row["fraction_history"], "output_quality": row["output_quality"],
        "zero_decision_reason_counts": dict(reasons),
        "first_zero_decision_by_reason": first_reasons,
        "helper_event_counts": dict(Counter(e["kind"] for e in events)),
        "effective_decision_allowances": [
            {"maximum_latency_ppm": latency, "minimum_energy_saving_ppm": saving}
            for latency, saving in sorted({(e["maximum_latency_ppm"], e["minimum_energy_saving_ppm"])
                for e in events if e["kind"] == "ASSISTANCE_DECISION"})
        ],
        "physical_execution_proof": row.get("physical_execution_proof"),
    }


def session_details(result):
    physical = result.get("phone_residency_at_completion") or {}
    rows = []
    for stage in (result.get("offline_phone_residency") or {}).get("stages", []):
        session_id = stage["selected_session_id"]
        phases = [e for e in result["phone_residency_phase_events"] if e["session_id"] == session_id]
        calls = [e for e in result["phone_residency_call_events"] if e["session_id"] == session_id]
        publications = [e for e in result["phone_residency_events"]
                        if (e.get("session") or {}).get("session_id") == session_id]
        rows.append({
            "session_id": session_id, "model_id": stage["model_id"],
            "stage_index": stage["stage_index"], "started_at_us": stage["started_at_us"],
            "verified_at_us": stage["verified_at_us"],
            "receipt_finished_at_us": stage["ready_at_us"],
            "stage_to_verification_us": stage["verified_at_us"] - stage["started_at_us"],
            "lifecycle": [{k: e[k] for k in ("kind", "observed_at_us", "published_at_us")}
                          for e in publications],
            "native_phase_events": phases,
            "first_call_epoch_us": min((e["epoch_us"] for e in calls), default=None),
            "last_call_epoch_us": max((e["epoch_us"] for e in calls), default=None),
            "weight_source": next(w for w in physical["weight_sources"] if w["session_id"] == session_id),
            "load_count": physical["load_count_by_session"][session_id],
            "transition_receipts": stage["transition_receipts"],
        })
    return rows


def native_settings(directory):
    settings = {}
    patterns = (
        r"system_info: n_threads =.*", r"llama_context: n_ctx\s*=.*",
        r"llama_context: n_batch\s*=.*", r"llama_context: n_ubatch\s*=.*",
        r"llama_context: flash_attn\s*=.*", r"offloaded \d+/\d+ layers to GPU.*",
        r"initializing, n_slots =.*",
    )
    for path in sorted(directory.glob("*.stderr")):
        text = path.read_text(errors="replace")
        settings[path.name] = sorted({match.group(0) for pattern in patterns
                                     for match in re.finditer(pattern, text)})
    return settings


def reference_details(directory):
    result = read(directory / "RESULT.json")
    preparation_path = directory / "FIXED_RESIDENCY_PREPARATION.json"
    preparation = read(preparation_path) if preparation_path.is_file() else None
    plan = result.get("offline_phone_residency") or {}
    ready = [e["published_at_us"] for e in result["phone_residency_events"] if e["kind"] == "SESSION_READY"]
    return {
        "requests_by_model": {r["model_id"]: request_details(r, result) for r in result["request_results"]},
        "native_resolved_settings": native_settings(directory),
        "session_details": session_details(result),
        "phone_preparation": None if preparation is None else {
            k: preparation[k] for k in ("duration_us", "started_ns", "finished_ns", "energy", "energy_interval_note")
        },
        "phone_first_ready_publication_us": min(ready, default=None),
        "phone_all_ready_publication_us": max(ready, default=None),
        "phone_memory": {k: plan.get(k) for k in (
            "resident_bytes", "workspace_bytes", "phone_wide_limit_bytes", "persistent_service_reserve_bytes",
        )},
        "phone_session_event_counts": dict(Counter(e["kind"] for e in result["phone_residency_events"])),
        "desktop_execution_attempts": result["counts"]["execution_attempts"],
        "active_server_slots_peak": result["active_slots_peak"],
        "scheduler_decision_timings": result["scheduler_decision_timings"],
        "preparation_note": "Preparation energy overlaps desktop load/execution. It is included once in the paid total, not an additive marginal cost.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    comparison = read(root / "COMPARISON.json")
    if comparison.get("comparison_valid") is not True:
        raise ValueError("reference comparison is not valid")
    details = {name: reference_details(root / "references-source-v7" / name / "run")
               for name in comparison["references"]}
    for name, reference in comparison["references"].items():
        for model, assistance in reference["assistance"].items():
            cpu_layers = details[name]["requests_by_model"][model]["desktop_cpu_ffn_layers"]
            if any(not set(contract["layers"]).issubset(cpu_layers)
                   for contract in assistance["layer_column_contracts"]):
                raise ValueError("physical assistance included a GPU-resident FFN layer")
    with args.output.open("x") as output:
        json.dump({"schema": "cuda-reference-reporting-details-v1", "references": details}, output,
                  sort_keys=True, separators=(",", ":"), allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
