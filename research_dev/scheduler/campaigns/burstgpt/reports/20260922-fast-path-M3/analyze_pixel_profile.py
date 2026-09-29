"""Reconcile Pixel stage timers and Vulkan operation timestamps."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def logs(path):
    stages = {}
    gpu = {}
    pending = None
    ready = False
    for line in path.read_text().splitlines():
        if "[ffn-worker] ready backend=" in line:
            ready = True
            pending = None
        if not ready:
            continue
        if line == "Vulkan Timings:":
            pending = {"operations": {}, "total_us": None}
        match = re.fullmatch(r"(.+): (\d+) x ([0-9.eE+-]+) us = ([0-9.eE+-]+) us(?: \(.+\))?", line)
        if match and pending is not None:
            pending["operations"][match[1]] = {"count": int(match[2]), "total_us": float(match[4])}
        match = re.fullmatch(r"Total time: ([0-9.eE+-]+) us\.", line)
        if match and pending is not None:
            pending["total_us"] = float(match[1])
        if line.startswith("PIXEL_FFN_STAGE "):
            row = {key: int(value) for key, value in re.findall(r"(\w+)=([0-9]+)", line)}
            ident = row["request"]
            assert ident not in stages
            assert sum(row[key] for key in ("setup_us", "input_us", "execute_us", "output_us")) == row["total_us"]
            stages[ident] = row
            if pending is not None:
                assert pending["total_us"] is not None
                assert abs(sum(v["total_us"] for v in pending["operations"].values()) - pending["total_us"]) < 2
                gpu[ident] = pending
                pending = None
    return stages, gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    result = {"status": "PASS", "scope": "One-row synthetic FFN inputs, six resident layers 18-23; first two of eight repeats excluded per layer/width.",
              "arms": {}, "limitations": ["GPU profiler adds timestamps/barriers and logging; its operation times are a diagnostic measurement, not an exact decomposition of the earlier server run.",
                                            "Wall execution includes GPU dispatch, synchronization and driver overhead.",
                                            "Stage copy timers do not separate memcpy, flush/invalidate and mapping costs.",
                                            "No phone energy measurement or hardware clock control."], "source_sha256": {}}
    expected_hashes = None
    for arm in ("control-before", "stage-timers", "gpu-profile", "control-after"):
        folder = root / arm
        assert json.loads((folder / "RESULT.json").read_text())["status"] == "PASS"
        calls = [json.loads(line) for line in (folder / "CALLS.jsonl").read_text().splitlines()]
        assert len(calls) == 96
        hashes = [row["output_sha256"] for row in calls]
        if expected_hashes is None:
            expected_hashes = hashes
        assert hashes == expected_hashes
        assert all(len({r["output_sha256"] for r in calls if r["layer"] == layer and r["columns"] == columns}) == 1
                   for layer in range(18, 24) for columns in (8704, 17408))
        stages, gpu = logs(folder / "worker.log")
        if not arm.startswith("control"):
            assert len(stages) == len(calls)
        if arm == "gpu-profile":
            assert len(gpu) == len(calls)
        for call in calls:
            if call["id"] in stages:
                stage = stages[call["id"]]
                assert (stage["layer"], stage["columns"], stage["total_us"]) == (call["layer"], call["columns"], call["worker_us"])
        summaries = {}
        for columns in (8704, 17408):
            warm = [r for r in calls if r["warm"] and r["columns"] == columns]
            assert len(warm) == 36
            row = {"calls": len(warm), "rpc_mean_ms": statistics.mean(r["rpc_us"] for r in warm) / 1000,
                   "worker_mean_ms": statistics.mean(r["worker_us"] for r in warm) / 1000,
                   "worker_median_ms": statistics.median(r["worker_us"] for r in warm) / 1000,
                   "maximum_relative_l2": max(r["relative_l2"] for r in warm)}
            row["outside_worker_mean_ms"] = row["rpc_mean_ms"] - row["worker_mean_ms"]
            if stages:
                row["stage_mean_ms"] = {key.removesuffix("_us"): statistics.mean(stages[r["id"]][key] for r in warm) / 1000
                                        for key in ("setup_us", "input_us", "execute_us", "output_us")}
                assert abs(sum(row["stage_mean_ms"].values()) - row["worker_mean_ms"]) < 1e-9
                row["stage_percent_of_worker"] = {name: value / row["worker_mean_ms"] * 100
                                                  for name, value in row["stage_mean_ms"].items()}
            if gpu:
                names = set().union(*(gpu[r["id"]]["operations"] for r in warm))
                row["gpu_operation_mean_ms"] = {name: statistics.mean(gpu[r["id"]]["operations"].get(name, {}).get("total_us", 0) for r in warm) / 1000
                                                for name in sorted(names)}
                row["gpu_timestamp_total_mean_ms"] = statistics.mean(gpu[r["id"]]["total_us"] for r in warm) / 1000
                row["execute_outside_gpu_timestamps_mean_ms"] = row["stage_mean_ms"]["execute"] - row["gpu_timestamp_total_mean_ms"]
                assert row["execute_outside_gpu_timestamps_mean_ms"] >= 0
                row["gpu_group_mean_ms"] = {
                    "matrix_vector": sum(value for name, value in row["gpu_operation_mean_ms"].items() if "MUL_MAT" in name),
                    "activation": row["gpu_operation_mean_ms"].get("GLU", 0),
                    "conversion_copy": row["gpu_operation_mean_ms"].get("CPY", 0),
                }
                assert abs(sum(row["gpu_group_mean_ms"].values()) - row["gpu_timestamp_total_mean_ms"]) < 0.002
            summaries[str(columns)] = row
        result["arms"][arm] = summaries
        for name in ("CALLS.jsonl", "worker.log", "RESULT.json"):
            path = folder / name
            result["source_sha256"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["profiling_effect_pct"] = {}
    for columns in ("8704", "17408"):
        baseline = statistics.mean(result["arms"][name][columns]["worker_mean_ms"] for name in ("control-before", "control-after"))
        result["profiling_effect_pct"][columns] = {name: (result["arms"][name][columns]["worker_mean_ms"] / baseline - 1) * 100
                                                   for name in ("stage-timers", "gpu-profile")}
    assert json.loads((root / "CLEANUP.json").read_text())["status"] == "PASS"
    assert json.loads((root / "RESULT.json").read_text())["status"] == "PASS"
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
