"""Diagnostic warm-request comparison from existing non-logprob relocation data."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.run / "REMOTE_RESIDENT_GATE.json"
    gate = json.loads(source.read_text())
    assert not gate.get("diagnostics"), "probability instrumentation is not performance evidence"
    rows, excluded = [], []
    totals = {"desktop_server_j": 0.0, "phone_path_server_j": 0.0,
              "desktop_s": 0.0, "phone_path_s": 0.0, "output_tokens": 0}
    for full, reduced in zip(gate["arms"]["full"]["requests"],
                             gate["arms"]["reduced"]["requests"], strict=True):
        assert full["request_index"] == reduced["request_index"]
        assert full["input_tokens"] == reduced["input_tokens"]
        assert full["output_tokens"] == reduced["output_tokens"]
        assert full["error"] is None and reduced["error"] is None
        command = json.loads((args.run / "phone" / (reduced["request_id"] + "-command.json")).read_text())
        if command["transitions"]:
            excluded.append({"request_index": full["request_index"],
                             "reason": "phone-arm request includes desktop preparation"})
            continue
        row = {
            "request_index": full["request_index"],
            "desktop_server_j": full["energy"]["server_compute_device_energy_j"],
            "phone_path_server_j": reduced["energy"]["server_compute_device_energy_j"],
            "desktop_s": full["duration_us"] / 1e6,
            "phone_path_s": reduced["duration_us"] / 1e6,
            "output_tokens": full["output_tokens"],
        }
        rows.append(row)
        for key in totals:
            totals[key] += row[key]
    assert rows
    baseline_fleet_j = totals["desktop_server_j"] + 0.875 * totals["desktop_s"]
    sensitivity = []
    for phone_w in (3, 4.5, 6):
        assisted_fleet_j = totals["phone_path_server_j"] + phone_w * totals["phone_path_s"]
        sensitivity.append({
            "phone_active_w": phone_w,
            "desktop_fleet_j": baseline_fleet_j,
            "phone_path_fleet_j": assisted_fleet_j,
            "saving_percent": 100 * (baseline_fleet_j - assisted_fleet_j) / baseline_fleet_j,
        })
    result = {
        "schema": "s42-remote-resident-hot-request-diagnostic-v1",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "status": "DIAGNOSTIC_NOT_MATCHED_AB", "requests": rows,
        "excluded_requests": excluded, "totals": totals,
        "duration_ratio": totals["phone_path_s"] / totals["desktop_s"],
        "phone_power_sensitivity": sensitivity,
        "accounting": {
            "server_energy": "measured RAPL package plus NVML board",
            "desktop_phone_idle_w": 0.875,
            "assisted_phone_activity": "conservatively assumed active for entire request boundary",
            "preparation_included": False,
            "cleanup_included": False,
        },
        "limits": [
            "Two short hot requests, one sample each; no statistical saving claim",
            "Existing gate boundaries and control paths are not a frozen matched performance protocol",
            "Does not establish end-to-end savings, break-even reuse, or longer-context capacity",
            "Phone power assumed, not physically measured; no token-identity requirement",
        ],
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"requests": len(rows), "duration_ratio": result["duration_ratio"],
                      "sensitivity": sensitivity}, indent=2))


if __name__ == "__main__":
    main()
