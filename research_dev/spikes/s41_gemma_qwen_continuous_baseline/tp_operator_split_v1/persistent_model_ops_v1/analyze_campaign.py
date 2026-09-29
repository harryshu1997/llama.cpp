#!/usr/bin/env python3
"""Aggregate the S41 persistent model-operator campaign."""

import argparse
import json
import math
import statistics
from pathlib import Path


OPS = ("rmsnorm", "swiglu", "attention")
BACKENDS = (
    "HTPGraph",
    "OpenCLGraph",
    "OpenCLDispatch",
    "OpenCLPersistent",
)
BREAKDOWN_STAGES = (
    "host_prepare_ms",
    "usb_out_ms",
    "worker_validate_ms",
    "worker_decode_ms",
    "backend_set_ms",
    "backend_submit_ms",
    "backend_sync_ms",
    "backend_get_ms",
    "worker_encode_hash_ms",
    "response_path_residual_ms",
    "host_validate_ms",
)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.floor(fraction * (len(ordered) - 1)))]


def metric(run, name, statistic="median_ms"):
    return run["stages"][name][statistic]


def response_ready_summary(run):
    values = [
        sample["host_prepare_ms"]
        + sample["usb_out_ms"]
        + sample["usb_in_wait_ms"]
        for sample in run["samples"]
    ]
    return {
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.result_root)

    grouped = {op: {backend: [] for backend in BACKENDS} for op in OPS}
    sources = []
    for path in sorted(root.glob("*_rep*/*.json")):
        run = json.loads(path.read_text())
        if run.get("schema") != "s41_persistent_model_ops_acquisition_v1":
            continue
        grouped[run["op"]][run["backend"]].append(run)
        sources.append(str(path.relative_to(root)))

    rows = {}
    total_matches = 0
    for op in OPS:
        rows[op] = {}
        for backend in BACKENDS:
            runs = grouped[op][backend]
            if len(runs) != 3:
                raise ValueError(f"{op}/{backend} has {len(runs)} runs")
            crc_sets = [run["variant_output_crc32"] for run in runs]
            if any(item != crc_sets[0] for item in crc_sets[1:]):
                raise ValueError(f"non-repeatable variant output: {op}/{backend}")
            total_matches += sum(run["semantic_matches"] for run in runs)
            run_e2e = [metric(run, "e2e_ms") for run in runs]
            run_p90 = [metric(run, "e2e_ms", "p90_ms") for run in runs]
            response_ready = [response_ready_summary(run) for run in runs]
            run_backend = [
                metric(run, "backend_submit_ms")
                + metric(run, "backend_sync_ms")
                for run in runs
            ]
            rows[op][backend] = {
                "runs": len(runs),
                "e2e_median_of_run_medians_ms": statistics.median(run_e2e),
                "e2e_median_of_run_p90_ms": statistics.median(run_p90),
                "backend_median_of_run_medians_ms": statistics.median(run_backend),
                "e2e_run_medians_ms": run_e2e,
                "response_ready_median_of_run_medians_ms": statistics.median(
                    item["median_ms"] for item in response_ready
                ),
                "response_ready_median_of_run_p90_ms": statistics.median(
                    item["p90_ms"] for item in response_ready
                ),
                "response_ready_run_medians_ms": [
                    item["median_ms"] for item in response_ready
                ],
                "max_relative_l2": max(run["max_relative_l2"] for run in runs),
                "max_abs_error": max(run["max_abs_error"] for run in runs),
                "variant_output_crc32": crc_sets[0],
                "breakdown_median_of_run_medians_ms": {
                    name: statistics.median(
                        metric(run, name) for run in runs
                    )
                    for name in BREAKDOWN_STAGES
                },
            }

        dispatch = rows[op]["OpenCLDispatch"]
        persistent = rows[op]["OpenCLPersistent"]
        graph = rows[op]["OpenCLGraph"]
        htp = rows[op]["HTPGraph"]
        dispatch_e2e = dispatch["response_ready_median_of_run_medians_ms"]
        persistent_e2e = persistent["response_ready_median_of_run_medians_ms"]
        dispatch_backend = dispatch["backend_median_of_run_medians_ms"]
        persistent_backend = persistent["backend_median_of_run_medians_ms"]
        rows[op]["comparisons"] = {
            "persistent_vs_matched_dispatch_response_ready_reduction_percent":
                100.0 * (dispatch_e2e - persistent_e2e) / dispatch_e2e,
            "persistent_vs_matched_dispatch_backend_reduction_percent":
                100.0 * (dispatch_backend - persistent_backend) / dispatch_backend,
            "persistent_vs_opencl_graph_response_ready_percent": 100.0 * (
                persistent_e2e
                - graph["response_ready_median_of_run_medians_ms"]
            ) / graph["response_ready_median_of_run_medians_ms"],
            "persistent_vs_htp_graph_response_ready_percent": 100.0 * (
                persistent_e2e
                - htp["response_ready_median_of_run_medians_ms"]
            ) / htp["response_ready_median_of_run_medians_ms"],
            "direct_paths_exact_crc_match":
                dispatch["variant_output_crc32"]
                == persistent["variant_output_crc32"],
        }

    result = {
        "schema": "s41_persistent_model_ops_analysis_v1",
        "model": "Qwen3-14B",
        "n_kv": 8192,
        "repetitions": 3,
        "total_semantic_matches": total_matches,
        "rows": rows,
        "source_files": sources,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
