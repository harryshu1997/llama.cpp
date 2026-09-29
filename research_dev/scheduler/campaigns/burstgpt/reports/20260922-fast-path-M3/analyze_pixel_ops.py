"""Audit synchronous per-operation FFN timings against normal-worker controls."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

import numpy as np

from analyze_pixel_profile import logs


def summarize(values):
    return {
        "samples": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p90_ms": float(np.percentile(values, 90)),
    }


def analyze(root):
    assert json.loads((root / "RESULT.json").read_text())["status"] == "PASS"
    assert json.loads((root / "CLEANUP.json").read_text())["status"] == "PASS"
    result = {
        "status": "PASS",
        "scope": "Isolated per-operation GPU timestamp intervals and dispatch-to-completion wall time; one-row Qwen FFN, six resident layers, real weights and synthetic activations.",
        "limitations": [
            "Each operation is synchronized separately; this disables down/add fusion and changes dispatch, scheduling and overlap.",
            "Times include backend dispatch and synchronization, not just GPU shader execution.",
            "GPU timestamp intervals include the existing profiler's barriers; wall time additionally includes timestamp retrieval and logging.",
            "Per-operation latencies cannot be added to decompose a normal fused FFN call.",
            "Clocks and thermals are not controlled; no energy or server-token measurement.",
        ],
        "arms": {}, "widths": {}, "source_sha256": {},
    }
    expected_outputs = None
    records = {}
    source_files = [root / "RESULT.json", root / "CLEANUP.json"]
    for arm in ("cpu", "control-before", "named-control", "op-profile", "control-after"):
        folder = root / arm
        assert json.loads((folder / "RESULT.json").read_text())["status"] == "PASS"
        rows = [json.loads(line) for line in (folder / "CALLS.jsonl").read_text().splitlines()]
        assert len(rows) == 96
        records[arm] = rows
        source_files += [folder / name for name in ("CALLS.jsonl", "RESULT.json", "worker.log")]
        if arm != "cpu":
            assert max(row["relative_l2"] for row in rows) <= 0.01
            outputs = [row["output_sha256"] for row in rows]
            if expected_outputs is None:
                expected_outputs = outputs
            assert outputs == expected_outputs
        result["arms"][arm] = {}
        for width in (8704, 17408):
            warm = [row for row in rows if row["warm"] and row["columns"] == width]
            assert len(warm) == 36
            result["arms"][arm][str(width)] = {
                "worker": summarize([row["worker_us"] / 1000 for row in warm]),
                "rpc": summarize([row["rpc_us"] / 1000 for row in warm]),
                "maximum_relative_l2": max(row.get("relative_l2", 0) for row in warm),
            }
    stage, gpu = logs(root / "op-profile/worker.log")
    assert len(stage) == len(gpu) == 96
    operations = defaultdict(list)
    pending_gpu = []
    request_gpu = {}
    ready = False
    for line in (root / "op-profile/worker.log").read_text().splitlines():
        if "[ffn-worker] ready backend=" in line:
            ready = True
        if not ready:
            continue
        match = re.fullmatch(r"Total time: ([0-9.eE+-]+) us\.", line)
        if match:
            pending_gpu.append(float(match[1]) / 1000)
        if line.startswith("PIXEL_FFN_STAGE "):
            ident = int(re.search(r"request=(\d+)", line).group(1))
            request_gpu[ident] = pending_gpu
            pending_gpu = []
        if not line.startswith("PIXEL_FFN_OP "):
            continue
        row = dict(field.split("=", 1) for field in line.split()[1:])
        for key in row.keys() - {"name", "op", "type", "src0_type", "src1_type"}:
            row[key] = int(row[key])
        row["category"] = row["name"].split("_block")[0]
        row["elapsed_ms"] = row["elapsed_ns"] / 1e6
        row["gpu_interval_ms"] = request_gpu[row["request"]][len(operations[row["request"]])]
        assert 0 < row["gpu_interval_ms"] <= row["elapsed_ms"] + 0.002
        operations[row["request"]].append(row)
    assert len(operations) == 96
    for call in records["op-profile"]:
        rows = operations[call["id"]]
        assert len(rows) == len(request_gpu[call["id"]])
        blocks = call["columns"] // 4352
        assert Counter(row["category"] for row in rows) == Counter({
            "input_cast": 1, "output_cast": 1, "gate": blocks, "up": blocks,
            "swiglu": blocks, "down": blocks, "add": blocks - 1,
        })
        assert stage[call["id"]]["total_us"] == call["worker_us"]
        assert sum(row["elapsed_ns"] for row in rows) <= stage[call["id"]]["execute_us"] * 1000 + 1000
        for row in rows:
            assert (row["layer"], row["columns"], row["rows"], row["ne1"]) == (call["layer"], call["columns"], 1, 1)
            assert row["ne0"] == (4352 if row["category"] in ("gate", "up", "swiglu") else 5120)
            assert row["type"] == ("f16" if row["category"] == "output_cast" else "f32")
            if row["category"] in ("gate", "up", "down"):
                assert row["op"] == "MUL_MAT" and row["src0_type"] == "f16" and row["src1_type"] == "f32"
                assert row["src0_ne0"] == (4352 if row["category"] == "down" else 5120)
                assert row["src0_ne1"] == row["ne0"] and row["src1_ne0"] == row["src0_ne0"]
    for width in (8704, 17408):
        calls = [row for row in records["op-profile"] if row["warm"] and row["columns"] == width]
        flat = [op for call in calls for op in operations[call["id"]]]
        categories = {}
        for category in ("input_cast", "gate", "up", "swiglu", "down", "add", "output_cast"):
            rows = [row for row in flat if row["category"] == category]
            entry = summarize([row["elapsed_ms"] for row in rows])
            entry["mean_total_per_ffn_ms"] = sum(row["elapsed_ms"] for row in rows) / len(calls)
            entry["calls_per_ffn"] = len(rows) // len(calls)
            entry["million_output_elements_per_s"] = rows[0]["ne0"] / entry["mean_ms"] / 1000
            entry["gpu_interval"] = summarize([row["gpu_interval_ms"] for row in rows])
            entry["gpu_mean_total_per_ffn_ms"] = sum(row["gpu_interval_ms"] for row in rows) / len(calls)
            entry["gpu_million_output_elements_per_s"] = rows[0]["ne0"] / entry["gpu_interval"]["mean_ms"] / 1000
            if category in ("gate", "up", "down"):
                entry["matrix_flops_per_call"] = 2 * rows[0]["ne0"] * rows[0]["src0_ne0"]
                entry["effective_gflops"] = entry["matrix_flops_per_call"] / entry["mean_ms"] / 1e6
                entry["gpu_effective_gflops"] = entry["matrix_flops_per_call"] / entry["gpu_interval"]["mean_ms"] / 1e6
            categories[category] = entry
        key = str(width)
        baseline = statistics.mean(result["arms"][arm][key]["worker"]["mean_ms"] for arm in ("control-before", "control-after"))
        worker = result["arms"]["op-profile"][key]["worker"]["mean_ms"]
        named = result["arms"]["named-control"][key]["worker"]["mean_ms"]
        result["widths"][key] = {
            "categories": categories,
            "sum_mean_op_ms": sum(entry["mean_total_per_ffn_ms"] for entry in categories.values()),
            "sum_mean_gpu_interval_ms": sum(entry["gpu_mean_total_per_ffn_ms"] for entry in categories.values()),
            "profile_worker_mean_ms": worker,
            "normal_worker_bracket_mean_ms": baseline,
            "named_control_worker_mean_ms": named,
            "profile_vs_normal_worker_pct": 100 * (worker / baseline - 1),
            "profile_vs_named_worker_pct": 100 * (worker / named - 1),
            "by_named_operation": {name: summarize([row["elapsed_ms"] for row in flat if row["name"] == name])
                                   for name in sorted({row["name"] for row in flat})},
        }
    result["all_phone_outputs_bit_exact"] = True
    result["phone_calls_checked"] = 384
    result["operator_calls_checked"] = sum(len(rows) for rows in operations.values())
    for path in source_files:
        result["source_sha256"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "widths": result["widths"]}, indent=2))


if __name__ == "__main__":
    main()
