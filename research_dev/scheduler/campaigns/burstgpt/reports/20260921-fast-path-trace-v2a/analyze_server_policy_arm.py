#!/usr/bin/env python3
"""Check the five handoff criteria against a completed trace and its saved baseline."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from statistics import median

from compare_trace_energy import load_result, summarize


def tokens(root, row):
    path = root / "streams" / f"request-{row['combined_request_index']:03d}.raw"
    output = []
    for line in path.read_text().splitlines():
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            output.extend(json.loads(line[5:]).get("tokens", []))
    if len(output) != row["output_tokens"]:
        raise ValueError(f"incomplete token stream: {path}")
    return output


def analyze(baseline_dir, run_dir):
    baseline = load_result(baseline_dir)
    run = load_result(run_dir)
    base_energy = summarize("baseline", baseline)
    energy = summarize("treatment", run)
    energy["host_saving_percent"] = 100 * (1 - energy["host_kj"] / base_energy["host_kj"])
    base_rows = {row["request_id"]: row for row in baseline["request_results"]}
    rows = {row["request_id"]: row for row in run["request_results"]}
    identical, differences = [], []
    for request_id, row in sorted(rows.items()):
        if request_id not in base_rows:
            differences.append({"request_id": request_id, "reason": "absent from baseline"})
            continue
        expected = tokens(baseline_dir, base_rows[request_id])
        actual = tokens(run_dir, row)
        if actual == expected:
            identical.append(request_id)
        else:
            first = next((i for i, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]),
                         min(len(expected), len(actual)))
            differences.append({"request_id": request_id, "first_differing_token_zero_based": first})
    qwen_rows = [row for row in rows.values() if row["model_id"] == run["model_roles"]["qwen"]]
    assisted = [row["request_id"] for row in qwen_rows
                if row.get("physical_execution_proof", {}).get("phone_call_count", 0) > 0]
    group_hashes = {row.get("physical_execution_proof", {}).get("adaptive_grouped_observation_sha256")
                    for row in qwen_rows}
    observations = json.loads((run_dir / "ADAPTIVE_DECODE_OBSERVATIONS.json").read_text())
    windows = [window for group in observations["groups"]
               if group["grouped_observation_sha256"] in group_hashes
               for window in group["windows"]
               if not window["policy"]["baseline"] and window["active_batch"] >= 2
               and (window.get("completed_phone_calls") or 0) > 0
               and window["output_valid"] and window["failure_reason"] is None]
    fleet_j = [window["energy_per_token_uj"] / 1e6 for window in windows]
    host_j = [sum(window["fleet_energy_uj_by_domain"].get(domain, 0)
                  for domain in ("cpu-package", "gpu-board"))
              / (window.get("accounting_token_count") or (window["token_end"] - window["token_start"])) / 1e6
              for window in windows]
    logs = {}
    forward_logs = {}
    for path in sorted(run_dir.glob("*physical-hot-*.stderr")):
        contents = path.read_text()
        if not re.search(r"general\.architecture[^\n]*=\s*qwen3\b", contents):
            continue
        counts = Counter(int(match[1]) for line in contents.splitlines()
                         if "S41SERVERFFNUSB " in line
                         for match in [re.search(r"\btokens=(\d+)\b", line)] if match)
        if counts:
            logs[path.name] = dict(sorted(counts.items()))
        forward_logs[path.name] = dict(sorted(Counter(
            int(value) for value in re.findall(r"S41SERVERFFNCALL[^\n]*\btokens=(\d+)\b", contents)
        ).items()))
    calls_multiple_rows = sum(count for counts in logs.values() for width, count in counts.items() if width > 1)
    checks = {
        "multi_token_qwen_phone_calls": calls_multiple_rows > 0,
        "phone_batch_ge_2_median_le_55_j_per_token": bool(fleet_j) and median(fleet_j) <= 55,
        "at_least_12_of_15_qwen_requests_assisted": len(qwen_rows) == 15 and len(assisted) >= 12,
        "host_saving_at_least_22_percent": energy["host_saving_percent"] >= 22,
        "all_24_outputs_token_identical": len(rows) == len(base_rows) == len(identical) == 24,
    }
    return {
        "status": "PASS" if run["status"] == "PASS" and all(checks.values()) else "FAIL",
        "checks": checks, "run_status": run["status"], "baseline_energy": base_energy, "energy": energy,
        "qwen_requests": len(qwen_rows), "qwen_assisted_request_ids": assisted,
        "qwen_multi_token_phone_calls": calls_multiple_rows, "qwen_calls_by_server_and_rows": logs,
        "qwen_forward_calls_by_server_and_rows": forward_logs,
        "qwen_phone_batch_ge_2_windows": len(windows),
        "qwen_phone_batch_ge_2_eligible_windows": sum(w.get("measurement_eligible", True) for w in windows),
        "qwen_phone_batch_ge_2_median_fleet_j_per_token": median(fleet_j) if fleet_j else None,
        "qwen_phone_batch_ge_2_median_host_j_per_token": median(host_j) if host_j else None,
        "identical_outputs": len(identical), "output_differences": differences,
        "missing_request_ids": sorted(set(base_rows) - set(rows)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.baseline, args.run)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    with args.output.open("x") as stream:
        stream.write(encoded)
    print(encoded)
