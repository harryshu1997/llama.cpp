"""Report physical evidence and explicit historical-reference differences."""

from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from research_dev.scheduler.campaigns.burstgpt.compare_ab import (
    ComparisonError, cuda_graph_evidence, energy_domains, file_sha256,
    frozen_reference_identity, load_adaptive_windows, load_object,
    phone_preflight_identity, reference_assistance_summary,
    reference_phone_power_sensitivity, request_rows, require,
    require_clean_execution, result_counts, validate_matched,
)


def differences(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        return [row for key in sorted(before.keys() | after.keys())
                for row in differences(before.get(key), after.get(key), prefix + "/" + key)]
    return [] if before == after else [{"field": prefix, "previous": before, "current": after}]


def main():
    result_path, output = map(Path, sys.argv[1:3])
    result = load_object(result_path)
    require(result["status"] == "PASS", "three-request execution failed")
    require_clean_execution(result, "adaptive")
    replay = load_object(REPO / "research_dev/scheduler/campaigns/burstgpt/data/traces/burstgpt_dev3_long_v1.json")
    result_counts(result, replay)
    power = reference_phone_power_sensitivity(result)
    identity = frozen_reference_identity(result)
    frozen = REPO / "research_dev/scheduler/baselines/cuda_graph_v1"
    references = load_object(frozen / "COMPARISON.json")["references"]
    comparisons = {}
    for name, summary in references.items():
        path = frozen / "references-source-v7" / name / "run/RESULT.json"
        require(file_sha256(path) == summary["result_sha256"], "frozen result changed")
        reference = load_object(path)
        expected = frozen_reference_identity(reference)
        checks = {key: identity[key] == expected[key] for key in (
            "workload", "model_artifacts", "model_roles", "trace_identity", "replay_schedule",
            "initial_observation_inputs", "adaptive_controller_configuration", "preparation_accounting",
            "energy_boundary", "maximum_latency_ppm",
        )}
        if name != "desktop-default-cuda":
            checks["desktop_parents"] = identity["desktop_parents"] == expected["desktop_parents"]
            checks["native_binaries"] = identity["execution_identity"]["binaries"] == expected["execution_identity"]["binaries"]
        checks["phone_artifacts"] = phone_preflight_identity(result_path) == phone_preflight_identity(path)
        comparisons[name] = {
            "checks": checks, "decoded_identity_diff": differences(expected, identity),
            "historical_reference": summary,
            "saving_percent": ({p: 100 * (1 - power[p]["fleet_energy_uj"] /
                                summary["phone_power_sensitivity"][p]["fleet_energy_uj"])
                                for p in power} if all(checks.values()) else None),
        }
    matched = load_object(frozen / "references-source-v7/desktop-matched-cuda/run/RESULT.json")
    try:
        validate_matched(matched, result, expected_replay=replay)
    except ComparisonError as error:
        matched_rejection = str(error)
    else:
        raise AssertionError("Changed scheduler/catalog must not be labeled matched A/B")
    windows = load_adaptive_windows(result_path, result)
    helper_events = result.get("request_helper_events", [])
    requests = []
    for row in request_rows(result):
        events = [event for event in helper_events if event.get("request_id") == row["request_id"]]
        receipt = row["terminal_ticket"]["execution_receipt"]
        requests.append({
            "request_id": row["request_id"], "model_id": row["model_id"],
            "output_tokens": row["output_tokens"], "arrival_us": row["replay_arrival_us"],
            "execution_started_us": receipt["started_us"], "finished_us": receipt["finished_us"],
            "service_latency_us": row["actual_latency_us"],
            "arrival_to_completion_us": receipt["finished_us"] - row["replay_arrival_us"],
            "phone_call_count": row["physical_execution_proof"]["phone_call_count"],
            "helper_events": dict(Counter(event["kind"] for event in events)),
            "zero_assistance_events": [event for event in events if event["kind"] == "ASSISTANCE_DECISION"
                                       and event.get("selected_fraction_ppm") == 0],
            "zero_assistance_windows": [window for window in windows
                                        if window["request_id"] == row["request_id"]
                                        and window["policy"]["split_fraction_ppm"] == 0],
            "execution_adapter": row["execution_command"]["adapter_parameters"].get("execution_adapter"),
            "route_id": row["terminal_ticket"]["decision"]["route_id"],
            "output_quality": row["output_quality"],
        })
    summary = {
        "schema": "ncm-dev3-physical-summary-v1", "status": "PASS",
        "comparison_kind": "Historical reference only; catalog and scheduler differ, not matched A/B",
        "matched_validator_rejection": matched_rejection,
        "result_sha256": file_sha256(result_path), "duration_us": result["duration_us"],
        "energy_uj_by_domain": energy_domains(result), "phone_power_sensitivity": power,
        "requests": requests, "assistance": reference_assistance_summary(result_path, result),
        "cuda_graphs": cuda_graph_evidence(result_path.parent.parent / "CUDA.sqlite"),
        "comparisons": comparisons, "phone_residency": result.get("phone_residency_at_completion"),
        "layout_events": result.get("phone_layout_events"),
        "session_events": result.get("phone_residency_events"),
        "phase_events": result.get("phone_residency_phase_events"),
        "control_events": result.get("android_control_events"),
        "helper_event_count": len(helper_events),
    }
    with output.open("x") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"status": summary["status"], "duration_us": summary["duration_us"],
                      "energy": power, "requests": [
                          {key: value for key, value in row.items() if key not in (
                              "zero_assistance_events", "zero_assistance_windows", "helper_events")}
                          for row in requests]}, default=str))


if __name__ == "__main__":
    main()
