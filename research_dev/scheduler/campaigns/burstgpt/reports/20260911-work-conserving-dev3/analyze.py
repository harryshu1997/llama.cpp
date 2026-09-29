"""Decode the small run without changing comparison or scheduling policy."""

from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from research_dev.scheduler.campaigns.burstgpt.compare_ab import (
    ComparisonError, energy_domains, file_sha256, frozen_reference_identity,
    load_object, reference_assistance_summary, reference_phone_power_sensitivity,
    require_clean_execution, result_counts, validate_matched,
)


def differences(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        return [row for key in sorted(before.keys() | after.keys())
                for row in differences(before.get(key), after.get(key), prefix + "/" + key)]
    return [] if before == after else [{"field": prefix, "previous": before, "current": after}]


def main():
    report = Path(__file__).resolve().parent
    path = report / "physical/run/RESULT.json"
    result = load_object(path)
    previous_path = report.parent / "20260910-ncm-dev3/physical/run/RESULT.json"
    previous = load_object(previous_path)
    replay = load_object(REPO / "research_dev/scheduler/campaigns/burstgpt/data/traces/burstgpt_dev3_long_v1.json")
    require_clean_execution(result, "adaptive")
    result_counts(result, replay)
    baseline_path = REPO / "research_dev/scheduler/baselines/cuda_graph_v1/references-source-v7/desktop-matched-cuda/run/RESULT.json"
    try:
        validate_matched(load_object(baseline_path), result, expected_replay=replay)
    except ComparisonError as error:
        comparison_rejection = str(error)
    else:
        raise AssertionError("changed source and binaries must not pass matched A/B")
    requests = []
    for row in result["request_results"]:
        receipt = row["terminal_ticket"]["execution_receipt"]
        proof = row["physical_execution_proof"]
        events = [event for event in result["request_helper_events"]
                  if event.get("request_id") == row["request_id"]]
        requests.append({
            "request_index": row["combined_request_index"], "request_id": row["request_id"],
            "executor_id": row["actual_executor_id"], "output_tokens": row["output_tokens"],
            "arrival_us": row["replay_arrival_us"], "started_us": receipt["started_us"],
            "finished_us": receipt["finished_us"], "service_us": row["actual_latency_us"],
            "arrival_to_completion_us": receipt["finished_us"] - row["replay_arrival_us"],
            "first_token_us": (row["first_token_ns"] - result["paid_start_ns"]) // 1000,
            "phone_calls": proof["phone_call_count"], "proof_sha256": proof["proof_sha256"],
            "phone_sessions": proof.get("phone_calls_by_session", []),
            "quality": row["output_quality"], "recoveries": row["recoveries"],
            "helper_events": dict(Counter(event["kind"] for event in events)),
            "zero_assistance_decisions": [event for event in events
                if event["kind"] == "ASSISTANCE_DECISION" and event.get("selected_fraction_ppm") == 0],
        })
    journal = load_object(path.parent / "SCHEDULER_DECISION_LOG.json")
    alternatives = []
    for row in journal["records"]:
        if row["event_kind"] not in ("DECISION", "REPLAN"):
            continue
        for candidate in row["candidates"]:
            if "cpu-parent:" not in candidate.get("executor_id", ""):
                continue
            cost = candidate["details"]["cost_breakdown"]
            alternatives.append({
                "ticket_id": row["ticket_id"], "event_time_us": row["event_time_us"],
                "event_kind": row["event_kind"], "route_id": candidate["route_id"],
                "admitted": candidate["admitted"], "selection_status": candidate["selection_status"],
                "eligibility_reasons": candidate["executor"]["eligibility_reasons"],
                "additional_bytes_by_resource": candidate["additional_bytes_by_resource"],
                "cost": {key: cost[key] for key in (
                    "start_us", "queue_delay_us", "load_us", "service_us", "service_upper_us",
                    "fleet_energy_uj", "fleet_energy_upper_uj")},
            })
    current_power = reference_phone_power_sensitivity(result)
    previous_power = reference_phone_power_sensitivity(previous)
    summary = {
        "schema": "work-conserving-dev3-summary-v1", "execution_status": result["status"],
        "start_early_goal": "NOT_DEMONSTRATED: CPU alternative registered but not executed",
        "live_migration": "NOT_IMPLEMENTED: no verified KV/sampler handoff",
        "counts": result["counts"], "duration_us": result["duration_us"],
        "energy_uj_by_domain": energy_domains(result), "phone_power_sensitivity": current_power,
        "requests": requests, "cpu_candidate_decisions": alternatives,
        "assistance": reference_assistance_summary(path, result),
        "helper_event_count": len(result["request_helper_events"]),
        "session_state_before_cleanup": result["phone_residency_at_completion"],
        "layout_events": [e for e in result["phone_residency_events"]
                          if e["kind"] in ("PROPOSED", "PREPARING", "READY", "FAILED")],
        "phone_phase_events": result["phone_residency_phase_events"],
        "comparison": {
            "kind": "historical diagnostic only; source, binaries and catalog differ",
            "matched_validator_rejection": comparison_rejection,
            "matched_validator_reference_sha256": file_sha256(baseline_path),
            "decoded_identity_diff": differences(frozen_reference_identity(previous), frozen_reference_identity(result)),
            "previous_result_sha256": file_sha256(previous_path),
            "previous_duration_us": previous["duration_us"],
            "previous_phone_power_sensitivity": previous_power,
            "energy_difference_percent": {key: 100 * (1 - value["fleet_energy_uj"] /
                previous_power[key]["fleet_energy_uj"]) for key, value in current_power.items()},
            "attribution": "Cannot attribute difference to CPU early start; that route did not execute",
        },
        "native_graph_reuse_lines": {f.name: [line for line in f.read_text(errors="replace").splitlines()
                                    if "graphs reused" in line]
                                     for f in path.parent.glob("*.stderr")},
        "result_sha256": file_sha256(path),
        "journal_file_sha256": file_sha256(path.parent / "SCHEDULER_DECISION_LOG.json"),
        "journal_head": result["journal_final_hash"],
        "source_manifest": file_sha256(report / "physical/inputs-v3/SOURCE_MANIFEST.json"),
    }
    with (report / "SUMMARY-final.json").open("x") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"duration_s": result["duration_us"] / 1e6, "power": current_power,
                      "comparison_rejection": comparison_rejection}))


if __name__ == "__main__":
    main()
