"""Audit captured Pixel AOA frames against the preserved CPU candidate."""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


BASE = Path(__file__).resolve().parent
ROOT = BASE / "physical/pixel10pro-aoa-v1"


def distribution(values):
    ordered = sorted(values)
    return {"count": len(ordered), "median_ms": statistics.median(ordered),
            "p90_ms": ordered[int(0.9 * len(ordered))],
            "p99_ms": ordered[int(0.99 * len(ordered))]}


def main():
    references = json.loads((BASE / "software/pixel10pro-aoa-v1/ARCHIVED_OUTPUT_HASHES.json").read_text())
    cleanup = json.loads((ROOT / "CLEANUP.json").read_text())
    assert cleanup["status"] == "PASS" and cleanup["no_probe_processes"] and cleanup["no_probe_wakelock"]
    groups = defaultdict(lambda: defaultdict(list))
    arms = []
    total_ffn = total_echo = total_rows = 0
    for directory in sorted((ROOT / "results").iterdir()):
        if not (directory / "RESULT.json").exists():
            continue
        config = json.loads((directory / "CONFIG.json").read_text())
        result = json.loads((directory / "RESULT.json").read_text())
        calls = json.loads((directory / "CALLS.json").read_text())
        assert result["status"] == "PASS" and len(calls) == result["completed"]
        assert (directory / "phone/EXIT.txt").read_text().strip() == "0"
        assert (directory / "phone/BOOT.txt").read_text().strip() == "573ce3f8-b84b-4a29-a9fd-a546747c90a7"
        measured = [call for call in calls if not call["warmup"]]
        if config["kind"] == "ffn":
            total_ffn += len(calls)
            total_rows += len(calls) * config["tokens"]
            for layer in range(18, 24):
                payload = (directory / f"output-layer{layer}.f16").read_bytes()
                assert len(payload) == config["tokens"] * 10240
                reference = references["hashes"][f"{layer}-{config['columns']}-1"]
                for row in range(config["tokens"]):
                    data = payload[row * 10240:(row + 1) * 10240]
                    assert hashlib.sha256(data).hexdigest() == reference, (directory, layer, row)
                payload_sha = hashlib.sha256(payload).hexdigest()
                assert all(call["output_sha256"] == payload_sha for call in calls if call["layer"] == layer)
        else:
            total_echo += len(calls)
        arms.append({"name": directory.name, "config": config, "result": result})
        if not directory.name.startswith(("v2-", "v3-")):
            continue
        geometry = f"B{config['tokens']}-C{config['columns']}" if config["kind"] == "ffn" else str(config["request"])
        label = f"{config['kind']}-{geometry}-gap{config['gap_ms']:g}-wake{int(config.get('wake_lock', False))}"
        groups[label][config["mode"]].append((directory.name, measured))
    comparisons = {}
    for label, transports in sorted(groups.items()):
        summary = {}
        for mode, runs in sorted(transports.items()):
            assert len(runs) == 2, (label, mode, len(runs))
            records = [record for _, rows in runs for record in rows]
            keys = [key for key in ("rpc_ms", "compute_ms", "outside_compute_ms") if key in records[0]]
            summary[mode] = {
                "arms": [name for name, _ in runs],
                "pooled": {key: distribution([r[key] for r in records]) for key in keys},
                "run_medians_ms": {key: [statistics.median(r[key] for r in rows) for _, rows in runs]
                                   for key in keys},
            }
        summary["rpc_reduction_percent"] = 100 * (1 - summary["aoa"]["pooled"]["rpc_ms"]["median_ms"] /
                                                   summary["adb"]["pooled"]["rpc_ms"]["median_ms"])
        comparisons[label] = summary
    summary = {"utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
               "milestones": {"functional_and_numerical": "PASS", "continuous_latency": "PASS",
                              "retaining_gain_with_5ms_gaps": "FAIL", "wake_lock_fix": "FAIL",
                              "cleanup": "PASS", "server_energy_and_tokens": "NOT_MEASURED"},
               "ffn_calls": total_ffn, "ffn_rows": total_rows, "echo_calls": total_echo,
               "ffn_exact_to_archived_candidate": True, "all_workers_exited_zero": True,
               "comparisons": comparisons, "arms": arms,
               "limitations": ["No server/token or energy measurement", "B2/4 duplicate the B1 input row",
                               "Shared desktop; compute and host scheduling are not fixed",
                               "Wake lock did not restore continuous-call performance"]}
    (ROOT / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS: {total_ffn} exact FFN calls / {total_rows} rows, {total_echo} validated echo calls")
    for name, comparison in comparisons.items():
        print(name, *(f"{mode}={comparison[mode]['pooled']['rpc_ms']['median_ms']:.6f}" for mode in ("adb", "aoa")),
              f"reduction={comparison['rpc_reduction_percent']:.3f}%")


if __name__ == "__main__":
    main()
