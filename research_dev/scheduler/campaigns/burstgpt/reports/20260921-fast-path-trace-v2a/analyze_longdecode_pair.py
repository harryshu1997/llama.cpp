#!/usr/bin/env python3
"""Compare the unfiltered real-window arms using exact saved output tokens."""

import argparse
import json
from pathlib import Path

from analyze_server_policy_arm import tokens
from compare_trace_energy import load_result, summarize


def analyze(baseline_dir, treatment_dir):
    baseline = load_result(baseline_dir)
    treatment = load_result(treatment_dir)
    energy = [summarize(label, result) for label, result in
              (("baseline", baseline), ("treatment", treatment))]
    rows = [{row["request_id"]: row for row in result["request_results"]}
            for result in (baseline, treatment)]
    differences = []
    input_differences = []
    identical = []
    for request_id in sorted(rows[0].keys() & rows[1].keys()):
        before, after = (arm[request_id] for arm in rows)
        for field in ("model_id", "prompt_sha256", "input_tokens", "output_tokens",
                      "seed", "source_arrival_us", "source_slo_us"):
            if before[field] != after[field]:
                input_differences.append({"request_id": request_id, "field": field,
                                          "baseline": before[field], "treatment": after[field]})
        expected = tokens(baseline_dir, before)
        actual = tokens(treatment_dir, after)
        if expected == actual:
            identical.append(request_id)
        else:
            first = next((i for i, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]),
                         min(len(expected), len(actual)))
            differences.append({"request_id": request_id, "model_id": after["model_id"],
                                "first_differing_token_zero_based": first,
                                "baseline_tokens": len(expected), "treatment_tokens": len(actual)})
    missing = sorted(rows[0].keys() - rows[1].keys())
    extra = sorted(rows[1].keys() - rows[0].keys())
    checks = {
        "both_runs_completed": all(result["status"] == "PASS" for result in (baseline, treatment)),
        "same_requests": bool(rows[0]) and not missing and not extra,
        "matched_request_inputs": not input_differences,
        "all_outputs_token_identical": len(identical) == len(rows[0]) == len(rows[1]),
    }
    by_model = {}
    for role, model_id in treatment["model_roles"].items():
        model_rows = [row for row in rows[1].values() if row["model_id"] == model_id]
        assisted = [row for row in model_rows
                    if row.get("physical_execution_proof", {}).get("phone_call_count", 0) > 0]
        by_model[role] = {
            "requests": len(model_rows), "assisted_requests": len(assisted),
            "output_tokens": sum(row["output_tokens"] for row in model_rows),
            "phone_calls": sum(row.get("physical_execution_proof", {}).get("phone_call_count", 0)
                               for row in model_rows),
        }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "energy": energy,
        "host_saving_percent": 100 * (1 - energy[1]["host_kj"] / energy[0]["host_kj"]),
        "duration_change_percent": 100 * (energy[1]["duration_s"] / energy[0]["duration_s"] - 1),
        "identical_outputs": len(identical), "output_differences": differences,
        "missing_request_ids": missing, "extra_request_ids": extra,
        "input_differences": input_differences, "treatment_by_model": by_model,
        "output_tokens": sum(row["output_tokens"] for row in rows[0].values()),
        "maximum_request_output_tokens": max(row["output_tokens"] for row in rows[0].values()),
        "requests_above_512_output_tokens": sum(row["output_tokens"] > 512 for row in rows[0].values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.baseline, args.treatment)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    with args.output.open("x") as stream:
        stream.write(encoded)
    print(encoded)
