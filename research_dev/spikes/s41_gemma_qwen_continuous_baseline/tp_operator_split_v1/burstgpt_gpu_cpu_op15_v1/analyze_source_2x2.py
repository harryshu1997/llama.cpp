#!/usr/bin/env python3

import hashlib
import json
import math
import re
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "source_length_2x2_v1"
TRACE = HERE / "REQUESTS_SEMANTIC_SOURCE.jsonl"
OUTPUT = RESULTS / "ANALYSIS.json"

CASES = {
    "cpu_hot_absent": RESULTS / "cpu_hot_absent",
    "cpu_hot_trace": RESULTS / "cpu_hot_trace",
    "op15_hot_absent": RESULTS / "op15_hot_absent",
    "op15_hot_trace": RESULTS / "op15_hot_trace_valid_prefix",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[int(fraction * (len(values) - 1))]


def length_summary(rows: list[dict]) -> dict:
    prompt = [row["input_tokens"] for row in rows]
    output = [row["output_tokens"] for row in rows]
    total = [a + b for a, b in zip(prompt, output)]

    def stats(values: list[int]) -> dict:
        return {
            "sum": sum(values),
            "min": min(values),
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }

    return {
        "requests": len(rows),
        "prompt_tokens": stats(prompt),
        "output_tokens": stats(output),
        "total_tokens": stats(total),
    }


def cold_results_from_result(case: str) -> tuple[dict, dict[int, dict]]:
    result = json.loads((CASES[case] / "RESULT.json").read_text())
    cold = {
        row["request_index"]: row
        for row in result["request_results"]
        if row["role"] == "cold"
    }
    return result, cold


def completion_rows(case: str, role: str) -> list[dict]:
    return [
        row
        for row in read_jsonl(CASES[case] / "events.jsonl")
        if row.get("kind") == "request_complete" and row.get("role") == role
    ]


def phase_sums(rows: dict[int, dict], indices: list[int]) -> dict:
    return {
        "requests": len(indices),
        "prefill_s": sum(rows[i]["prefill_us"] for i in indices) / 1e6,
        "decode_s": sum(rows[i]["decode_us"] for i in indices) / 1e6,
        "service_s": sum(rows[i]["route_wall_us"] for i in indices) / 1e6,
    }


def reduction(before: float, after: float) -> dict:
    return {
        "before": before,
        "after": after,
        "speedup_x": before / after,
        "reduction_pct": 100.0 * (1.0 - after / before),
    }


def delta(before: float, after: float) -> dict:
    return {
        "before": before,
        "after": after,
        "delta_pct": 100.0 * (after / before - 1.0),
    }


def compare_phases(before: dict, after: dict) -> dict:
    return {
        key: reduction(before[key], after[key])
        for key in ("prefill_s", "decode_s", "service_s")
    }


def union_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def overlap_ns(start: int, end: int, intervals: list[tuple[int, int]]) -> int:
    return sum(
        max(0, min(end, interval_end) - max(start, interval_start))
        for interval_start, interval_end in intervals
    )


def hot_overlap(case: str, cold: dict[int, dict]) -> dict:
    events = read_jsonl(CASES[case] / "events.jsonl")
    trace_start = next(row["t_ns"] for row in events if row["kind"] == "trace_start")
    hot = [
        row
        for row in events
        if row.get("kind") == "request_complete" and row.get("role") == "hot"
    ]
    intervals = union_intervals(
        [(row["dispatch_ns"], row["completion_ns"]) for row in hot]
    )
    per_request = {}
    for index, row in cold.items():
        overlap = overlap_ns(row["dispatch_ns"], row["completion_ns"], intervals)
        per_request[str(index)] = {
            "overlap_s": overlap / 1e9,
            "overlap_pct": 100.0 * overlap / (row["route_wall_us"] * 1000),
        }
    return {
        "hot_requests": len(hot),
        "start_s": (intervals[0][0] - trace_start) / 1e9,
        "end_s": (intervals[-1][1] - trace_start) / 1e9,
        "union_s": sum(end - start for start, end in intervals) / 1e9,
        "per_cold_request": per_request,
    }


def parse_summary(path: Path, prefix: str) -> dict:
    match = None
    for line in path.read_text().splitlines():
        if line.startswith(prefix + " "):
            match = json.loads(line[len(prefix) + 1 :])
    if match is None:
        raise RuntimeError(f"missing {prefix} in {path}")
    return match


def last_match(path: Path, pattern: str) -> re.Match:
    matches = list(re.finditer(pattern, path.read_text()))
    if not matches:
        raise RuntimeError(f"missing {pattern} in {path}")
    return matches[-1]


def main() -> None:
    trace_rows = read_jsonl(TRACE)
    cold_trace = [row for row in trace_rows if row["model_id"] == "qwen3-14b-q4_k_m"]
    hot_trace = [row for row in trace_rows if row["model_id"] == "gemma-4-12b-it-q8_0"]
    trace_by_index = {row["request_index"]: row for row in cold_trace}

    cpu_absent_result, cpu_absent = cold_results_from_result("cpu_hot_absent")
    cpu_hot_result, cpu_hot = cold_results_from_result("cpu_hot_trace")
    op15_absent_result, op15_absent = cold_results_from_result("op15_hot_absent")
    op15_hot_all = {
        row["request_index"]: row for row in completion_rows("op15_hot_trace", "cold")
    }

    indices = sorted(cpu_absent)
    if indices != sorted(cpu_hot) or indices != sorted(op15_absent):
        raise RuntimeError("complete cold result indices do not match")

    first_failure_worker = CASES["op15_hot_trace"] / "phone-worker.log"
    first_failure_calls = int(
        last_match(first_failure_worker, r"DMA-BUF complete requests=(\d+) status=1").group(1)
    )
    cumulative_calls = 0
    valid_prefix = []
    call_boundaries = {}
    for index in indices:
        row = trace_by_index[index]
        calls = 48 * (row["output_tokens"] + math.ceil(row["input_tokens"] / 512))
        cumulative_calls += calls
        call_boundaries[str(index)] = cumulative_calls
        if cumulative_calls <= first_failure_calls:
            valid_prefix.append(index)

    if valid_prefix[-1] != 65 or len(valid_prefix) != 14:
        raise RuntimeError("unexpected valid hot OP15 prefix")

    full = {
        "cpu_hot_absent": phase_sums(cpu_absent, indices),
        "cpu_hot_trace": phase_sums(cpu_hot, indices),
        "op15_hot_absent": phase_sums(op15_absent, indices),
    }
    prefix = {
        "cpu_hot_absent": phase_sums(cpu_absent, valid_prefix),
        "cpu_hot_trace": phase_sums(cpu_hot, valid_prefix),
        "op15_hot_absent": phase_sums(op15_absent, valid_prefix),
        "op15_hot_trace": phase_sums(op15_hot_all, valid_prefix),
    }

    per_request = []
    for index in indices:
        source = trace_by_index[index]
        row = {
            "request_index": index,
            "prompt_tokens": source["input_tokens"],
            "output_tokens": source["output_tokens"],
            "cpu_hot_absent_s": cpu_absent[index]["route_wall_us"] / 1e6,
            "cpu_hot_trace_s": cpu_hot[index]["route_wall_us"] / 1e6,
            "op15_hot_absent_s": op15_absent[index]["route_wall_us"] / 1e6,
            "cpu_to_op15_speedup_x": (
                cpu_absent[index]["route_wall_us"]
                / op15_absent[index]["route_wall_us"]
            ),
        }
        if index in valid_prefix:
            row["op15_hot_trace_s"] = op15_hot_all[index]["route_wall_us"] / 1e6
            row["op15_hot_delta_pct"] = 100.0 * (
                op15_hot_all[index]["route_wall_us"]
                / op15_absent[index]["route_wall_us"]
                - 1.0
            )
        per_request.append(row)

    paired_speedups = [row["cpu_to_op15_speedup_x"] for row in per_request]
    resident_indices = valid_prefix[2:]
    op15_hot_overlap = hot_overlap("op15_hot_trace", op15_hot_all)
    cpu_hot_overlap = hot_overlap("cpu_hot_trace", cpu_hot)

    ffn = parse_summary(CASES["op15_hot_absent"] / "cold-driver.stderr", "FFNSPLIT")
    dmabuf = parse_summary(CASES["op15_hot_absent"] / "dmabuf-bridge.stderr", "FFNDMABUF")
    retry_dir = RESULTS / "op15_hot_trace_retry1_invalid"
    retry_calls = int(
        last_match(
            retry_dir / "phone-worker.log",
            r"DMA-BUF complete requests=(\d+) status=1",
        ).group(1)
    )

    cold_output_tokens = sum(row["output_tokens"] for row in cold_trace)
    complete_duration = {
        "cpu_hot_absent": cpu_absent_result["metrics"]["duration_s"],
        "cpu_hot_trace": cpu_hot_result["metrics"]["duration_s"],
        "op15_hot_absent": op15_absent_result["metrics"]["duration_s"],
    }

    analysis = {
        "schema": "s41-burstgpt-source-length-2x2-analysis-v1",
        "trace": {
            "sha256": hashlib.sha256(TRACE.read_bytes()).hexdigest(),
            "requests": len(trace_rows),
            "arrival_span_s": (
                max(row["arrival_us"] for row in trace_rows)
                - min(row["arrival_us"] for row in trace_rows)
            ) / 1e6,
            "cold": length_summary(cold_trace),
            "hot": length_summary(hot_trace),
        },
        "full_17_request_cases": {
            case: {
                **full[case],
                "duration_s": complete_duration[case],
                "cold_output_tokens_s": cold_output_tokens / complete_duration[case],
            }
            for case in full
        },
        "no_hot_cpu_to_op15": {
            "phases": compare_phases(full["cpu_hot_absent"], full["op15_hot_absent"]),
            "makespan": reduction(
                complete_duration["cpu_hot_absent"],
                complete_duration["op15_hot_absent"],
            ),
            "paired_speedup_x": {
                "min": min(paired_speedups),
                "p50": statistics.median(paired_speedups),
                "mean": statistics.mean(paired_speedups),
                "max": max(paired_speedups),
            },
        },
        "cpu_hot_effect": {
            key: delta(full["cpu_hot_absent"][key], full["cpu_hot_trace"][key])
            for key in ("prefill_s", "decode_s", "service_s")
        },
        "matched_14_request_prefix": {
            "indices": valid_prefix,
            "call_boundary_after_last_valid": call_boundaries[str(valid_prefix[-1])],
            "cases": prefix,
            "op15_hot_effect": {
                key: delta(prefix["op15_hot_absent"][key], prefix["op15_hot_trace"][key])
                for key in ("prefill_s", "decode_s", "service_s")
            },
            "op15_hot_effect_by_overlap": {
                "request_0_full_overlap": {
                    key: delta(
                        phase_sums(op15_absent, [0])[key],
                        phase_sums(op15_hot_all, [0])[key],
                    )
                    for key in ("prefill_s", "decode_s", "service_s")
                },
                "request_1_partial_overlap": {
                    key: delta(
                        phase_sums(op15_absent, [1])[key],
                        phase_sums(op15_hot_all, [1])[key],
                    )
                    for key in ("prefill_s", "decode_s", "service_s")
                },
                "requests_2_to_65_resident_only": {
                    key: delta(
                        phase_sums(op15_absent, resident_indices)[key],
                        phase_sums(op15_hot_all, resident_indices)[key],
                    )
                    for key in ("prefill_s", "decode_s", "service_s")
                },
            },
        },
        "hot_service_windows": {
            "cpu_hot_trace": cpu_hot_overlap,
            "op15_hot_trace": op15_hot_overlap,
        },
        "transport": {
            "successful_no_hot": {
                "calls": ffn["calls"],
                "decode_calls": ffn["decode_calls"],
                "prefill_calls": ffn["prefill_calls"],
                "decode_phone_compute_p50_ms": ffn["decode_phone_compute_p50_ms"],
                "decode_rpc_p50_ms": ffn["decode_rpc_p50_ms"],
                "decode_host_branch_p50_ms": ffn["host_branch_p50_ms"],
                "decode_wait_p50_ms": ffn["wait_p50_ms"],
                "prefill_phone_compute_p50_ms": ffn["prefill_phone_compute_p50_ms"],
                "prefill_rpc_p50_ms": ffn["prefill_rpc_p50_ms"],
                "prefill_overlap_p50_ms": ffn["prefill_overlap_p50_ms"],
                "usb_p50_ms": dmabuf["usb_p50_ms"],
                "decode_usb_p50_ms": dmabuf["decode_usb_p50_ms"],
                "prefill_usb_p50_ms": dmabuf["prefill_usb_p50_ms"],
                "usb_out_p50_ms": dmabuf["out_p50_ms"],
                "usb_in_p50_ms": dmabuf["in_p50_ms"],
            },
            "hot_attempt_1_failure": {
                "worker_calls_at_failure": first_failure_calls,
                "last_valid_request_index": valid_prefix[-1],
                "valid_cold_requests": len(valid_prefix),
                "bridge_error": "LIBUSB_ERROR_NO_DEVICE",
                "worker_error": "DMA-BUF output transfer failed: endpoint shutdown",
                "host_kernel_event": "reset SuperSpeed USB device",
            },
            "hot_attempt_2_failure": {
                "worker_calls_at_failure": retry_calls,
                "bridge_error": "LIBUSB_ERROR_TIMEOUT",
                "worker_error": "DMA-BUF input requeue failed: endpoint shutdown",
                "host_kernel_event": "reset SuperSpeed USB device",
            },
        },
        "per_cold_request": per_request,
    }

    OUTPUT.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
