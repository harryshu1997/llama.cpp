"""Audit named GPU timestamps from intact fused and unfused Pixel FFN graphs."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

from analyze_pixel_ops import summarize
from analyze_pixel_profile import logs


def analyze(root):
    assert json.loads((root / "RESULT.json").read_text())["status"] == "PASS"
    assert json.loads((root / "CLEANUP.json").read_text())["status"] == "PASS"
    result = {
        "status": "PASS",
        "scope": "Existing Vulkan per-operation timestamp profiler, with node labels; intact six-layer-resident FFN workload, one row, 4352-column blocks.",
        "limitations": [
            "Timestamp intervals include barriers and queue/driver scheduling effects; not pure shader arithmetic time.",
            "Profiling changes whole-worker latency; normal controls remain the end-to-end performance measurement.",
            "Down and partial addition are fused in the normal graph. The unfused arm changes execution to measure addition separately.",
            "No controlled GPU clocks, thermal state, phone energy or server token run.",
        ],
        "arms": {}, "source_sha256": {},
    }
    expected_outputs = None
    sources = [root / "RESULT.json", root / "CLEANUP.json"]
    for arm in ("cpu", "control-before", "named-profile", "unfused-profile", "control-after"):
        folder = root / arm
        assert json.loads((folder / "RESULT.json").read_text())["status"] == "PASS"
        calls = [json.loads(line) for line in (folder / "CALLS.jsonl").read_text().splitlines()]
        assert len(calls) == 96
        sources.extend(folder / name for name in ("CALLS.jsonl", "worker.log", "RESULT.json"))
        if arm != "cpu":
            outputs = [row["output_sha256"] for row in calls]
            if expected_outputs is None:
                expected_outputs = outputs
            assert outputs == expected_outputs
            assert max(row["relative_l2"] for row in calls) <= 0.01
        stage, gpu = logs(folder / "worker.log")
        if arm.endswith("profile"):
            assert len(stage) == len(gpu) == 96
            for call in calls:
                assert stage[call["id"]]["total_us"] == call["worker_us"]
                count = Counter()
                for name, entry in gpu[call["id"]]["operations"].items():
                    label = re.search(r"\[([^\]]+)\]$", name).group(1)
                    category = label.split("_block")[0]
                    if name.startswith("MUL_MAT_ADD "):
                        category = "down_fused_add"
                    assert entry["count"] == 1
                    count[category] += 1
                blocks = call["columns"] // 4352
                expected = Counter(input_cast=1, output_cast=1, gate=blocks, up=blocks, swiglu=blocks)
                if arm == "named-profile":
                    expected.update(down=1, down_fused_add=blocks - 1)
                else:
                    expected.update(down=blocks, add=blocks - 1)
                assert count == expected, (arm, call["id"], count, expected)
        result["arms"][arm] = {}
        for width in (8704, 17408):
            warm = [call for call in calls if call["warm"] and call["columns"] == width]
            assert len(warm) == 36
            row = {
                "worker": summarize([call["worker_us"] / 1000 for call in warm]),
                "rpc": summarize([call["rpc_us"] / 1000 for call in warm]),
                "maximum_relative_l2": max(call.get("relative_l2", 0) for call in warm),
            }
            if gpu:
                values = defaultdict(list)
                named = defaultdict(list)
                for call in warm:
                    for name, entry in gpu[call["id"]]["operations"].items():
                        label = re.search(r"\[([^\]]+)\]$", name).group(1)
                        category = label.split("_block")[0]
                        if name.startswith("MUL_MAT_ADD "):
                            category = "down_fused_add"
                        values[category].append(entry["total_us"] / 1000)
                        named[name].append(entry["total_us"] / 1000)
                categories = {}
                for category, times in values.items():
                    entry = summarize(times)
                    entry["count_per_ffn"] = len(times) // len(warm)
                    entry["mean_total_per_ffn_ms"] = sum(times) / len(warm)
                    if category in ("gate", "up", "down", "down_fused_add"):
                        entry["matrix_flops_per_call"] = 2 * 5120 * 4352
                        entry["effective_matrix_gflops"] = entry["matrix_flops_per_call"] / entry["mean_ms"] / 1e6
                    else:
                        elements = 4352 if category == "swiglu" else 5120
                        entry["million_output_elements_per_s"] = elements / entry["mean_ms"] / 1000
                    categories[category] = entry
                row["categories"] = categories
                row["by_named_operation"] = {name: summarize(times) for name, times in named.items()}
                row["gpu_timestamp_total_mean_ms"] = statistics.mean(gpu[call["id"]]["total_us"] / 1000 for call in warm)
                row["wall_execute_mean_ms"] = statistics.mean(stage[call["id"]]["execute_us"] / 1000 for call in warm)
                assert abs(sum(entry["mean_total_per_ffn_ms"] for entry in categories.values()) - row["gpu_timestamp_total_mean_ms"]) < 0.002
            result["arms"][arm][str(width)] = row
    result["profile_vs_bracketed_worker_pct"] = {}
    for width in ("8704", "17408"):
        baseline = statistics.mean(result["arms"][arm][width]["worker"]["mean_ms"] for arm in ("control-before", "control-after"))
        result["profile_vs_bracketed_worker_pct"][width] = {
            arm: 100 * (result["arms"][arm][width]["worker"]["mean_ms"] / baseline - 1)
            for arm in ("named-profile", "unfused-profile")
        }
    result["all_phone_outputs_bit_exact"] = True
    result["phone_calls_checked"] = 384
    for path in sources:
        result["source_sha256"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
