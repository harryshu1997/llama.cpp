"""Compare saved M3 outputs and measured host energy without touching hardware."""

import argparse
import json
from pathlib import Path
import statistics


def analyze(root, additional_roots=()):
    arms = {}
    for directory in sorted(path for source in (root, *additional_roots) for path in source.iterdir()):
        result_path = directory / "RESULT.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        executions = sorted((json.loads(path.read_text()) for path in directory.glob("EXECUTION-*.json")),
                            key=lambda row: row["index"])
        cohort = result["cohort"]
        arms[directory.name] = {
            "status": result["status"], "parallel": cohort["parallel"],
            "fixture_sha256": cohort["request_fixture_sha256"],
            "tokens": [row["tokens"] for row in executions],
            "prompt_tokens": [row["prompt_tokens"] for row in executions],
            "request_s": cohort["request_s"], "decode_s": cohort["decode_s"],
            "request_host_j": cohort["request_host_energy"]["server_compute_device_energy_j"],
            "decode_host_j": cohort["decode_host_energy"]["server_compute_device_energy_j"],
            "paid_s": result["paid_s"],
            "paid_host_j": result["paid_host_energy"]["server_compute_device_energy_j"],
        }
        proof = directory / "TWO_PHONE_RESULT.json"
        if proof.exists():
            raw = json.loads(proof.read_text())
            arms[directory.name].update({key: raw[key] for key in (
                "calls_by_device", "helper_stop", "helper_served_calls", "helper_assumed_j")})
    control = arms["desktop-before"]
    controls = [row for name, row in arms.items() if name.startswith("desktop-")]
    control_j = statistics.mean(row["request_host_j"] for row in controls)
    control_s = statistics.mean(row["request_s"] for row in controls)
    for row in arms.values():
        row["exact_outputs"] = [left == right for left, right in zip(control["tokens"], row["tokens"])]
        row["exact_status"] = "PASS" if (
            len(row["tokens"]) == len(control["tokens"])
            and all(row["exact_outputs"])
            and row["prompt_tokens"] == control["prompt_tokens"]
            and row["fixture_sha256"] == control["fixture_sha256"]
        ) else "FAIL"
        row["request_host_saving_pct"] = 100 * (1 - row["request_host_j"] / control_j)
        row["request_time_change_pct"] = 100 * (row["request_s"] / control_s - 1)
        if "op15-full" in arms:
            row["incremental_host_saving_vs_op15_pct"] = 100 * (
                1 - row["request_host_j"] / arms["op15-full"]["request_host_j"])
    for row in arms.values():
        del row["tokens"], row["prompt_tokens"]
    return {"schema": "s42-pixel-mechanism-comparison-v1", "arms": arms,
            "exact_status": "PASS" if all(row["exact_status"] == "PASS" for row in arms.values()) else "FAIL",
            "note": "Single observations per arm. Host RAPL/NVML only; phone values are assumptions. "
                    "Paid spans include differing startup costs. Decode intervals begin at control acknowledgement "
                    "for assisted arms and first token for desktop arms; headline uses complete requests."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--additional-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.root, args.additional_root), indent=2, sort_keys=True) + "\n")
