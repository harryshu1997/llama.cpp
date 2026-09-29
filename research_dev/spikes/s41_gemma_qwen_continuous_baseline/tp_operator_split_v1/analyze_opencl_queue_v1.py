#!/usr/bin/env python3
"""Reduce the matched OP15 OpenCL queue experiment."""

import argparse
import csv
import json
import pathlib
import re
import statistics


DISPATCH_PROFILE = re.compile(
    r"request=(\d+) queued_us=([0-9.]+) "
    r"start_us=([0-9.]+) exec_us=([0-9.]+)"
)
PERSISTENT_PROFILE = re.compile(
    r"launch_queue_us=([0-9.]+) launch_start_us=([0-9.]+) "
    r"resident_ms=([0-9.]+)"
)


def median(values):
    return statistics.median(values)


def load_group(root, pattern):
    runs = []
    for directory in sorted(root.glob(pattern)):
        paths = list(directory.glob("*.json"))
        if len(paths) != 1:
            raise ValueError(f"expected one JSON result in {directory}")
        result = json.loads(paths[0].read_text(encoding="ascii"))
        if result["iterations"] != 600 or result["semantic_matches"] != 600:
            raise ValueError(f"incomplete semantic evidence in {paths[0]}")
        stages = result["stages"]
        runs.append(
            {
                "run": directory.name,
                "e2e_median_ms": stages["e2e_ms"]["median_ms"],
                "e2e_p90_ms": stages["e2e_ms"]["p90_ms"],
                "submit_median_ms": stages["backend_submit_ms"]["median_ms"],
                "sync_median_ms": stages["backend_sync_ms"]["median_ms"],
                "backend_median_ms":
                    stages["backend_submit_ms"]["median_ms"]
                    + stages["backend_sync_ms"]["median_ms"],
            }
        )
    if len(runs) != 3:
        raise ValueError(f"expected three runs for {pattern}, found {len(runs)}")
    return {
        "runs": runs,
        "median_of_run_e2e_medians_ms": median(
            [run["e2e_median_ms"] for run in runs]
        ),
        "median_of_run_e2e_p90_ms": median(
            [run["e2e_p90_ms"] for run in runs]
        ),
        "median_of_run_backend_medians_ms": median(
            [run["backend_median_ms"] for run in runs]
        ),
        "median_of_run_submit_medians_ms": median(
            [run["submit_median_ms"] for run in runs]
        ),
        "median_of_run_sync_medians_ms": median(
            [run["sync_median_ms"] for run in runs]
        ),
    }


def load_native_csv(path):
    with path.open(newline="", encoding="ascii") as source:
        raw = list(csv.reader(source))
    names = [name.strip() for name in raw[0]]
    rows = [dict(zip(names, row)) for row in raw[1:]][50:]
    if len(rows) != 600:
        raise ValueError(f"expected 600 profiled kernels in {path}")
    return {
        name: median([float(row[name]) for row in rows])
        for name in ("queued_us", "submit_us", "exec_us")
    }


def load_dispatch_native(root):
    runs = []
    for path in sorted(root.glob("m*_dispatch/*.worker.log")):
        records = [
            (int(request), float(queued), float(start), float(execution))
            for request, queued, start, execution
            in DISPATCH_PROFILE.findall(path.read_text(encoding="ascii"))
            if int(request) > 50
        ]
        if len(records) != 600:
            raise ValueError(f"expected 600 dispatch profiles in {path}")
        runs.append(
            {
                "run": path.parent.name,
                "queued_us": median([record[1] for record in records]),
                "start_us": median([record[2] for record in records]),
                "exec_us": median([record[3] for record in records]),
            }
        )
    return {
        "runs": runs,
        "median_of_run_queued_us": median(
            [run["queued_us"] for run in runs]
        ),
        "median_of_run_start_us": median(
            [run["start_us"] for run in runs]
        ),
        "median_of_run_exec_us": median(
            [run["exec_us"] for run in runs]
        ),
    }


def load_persistent_launch(root):
    runs = []
    for path in sorted(root.glob("m*_persistent/*.worker.log")):
        matches = PERSISTENT_PROFILE.findall(path.read_text(encoding="ascii"))
        if len(matches) != 1:
            raise ValueError(f"expected one persistent profile in {path}")
        queued, start, resident = (float(value) for value in matches[0])
        runs.append(
            {
                "run": path.parent.name,
                "launch_queue_us": queued,
                "launch_start_us": start,
                "resident_ms": resident,
            }
        )
    return {"runs": runs}


def reduction(before, after):
    return {
        "reduction_percent": 100.0 * (before - after) / before,
        "speedup": before / after,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    control = load_group(args.root, "r*_control")
    flush = load_group(args.root, "r*_flush")
    dispatch = load_group(args.root, "m*_dispatch")
    persistent = load_group(args.root, "m*_persistent")
    result = {
        "schema": "s41_opencl_queue_v1",
        "device": "OP15 Adreno 840",
        "elements": 2816,
        "operation": "one F32 SQR",
        "warmup_per_run": 50,
        "samples_per_run": 600,
        "repetitions": 3,
        "primary_semantic_matches": 7200,
        "profile_semantic_matches": 1200,
        "control": control,
        "early_flush": flush,
        "matched_dispatch": dispatch,
        "persistent": persistent,
        "native_stock_profile": {
            "control": load_native_csv(
                args.root
                / "profile_control/openclrebuilt_sqr_1.cl_profiling.csv"
            ),
            "early_flush": load_native_csv(
                args.root
                / "profile_flush/openclflush_sqr_1.cl_profiling.csv"
            ),
        },
        "native_matched_dispatch": load_dispatch_native(args.root),
        "persistent_launch": load_persistent_launch(args.root),
        "comparisons": {
            "early_flush_e2e": reduction(
                control["median_of_run_e2e_medians_ms"],
                flush["median_of_run_e2e_medians_ms"],
            ),
            "early_flush_backend": reduction(
                control["median_of_run_backend_medians_ms"],
                flush["median_of_run_backend_medians_ms"],
            ),
            "persistent_vs_control_e2e": reduction(
                control["median_of_run_e2e_medians_ms"],
                persistent["median_of_run_e2e_medians_ms"],
            ),
            "persistent_vs_control_backend": reduction(
                control["median_of_run_backend_medians_ms"],
                persistent["median_of_run_backend_medians_ms"],
            ),
            "persistent_vs_matched_dispatch_e2e": reduction(
                dispatch["median_of_run_e2e_medians_ms"],
                persistent["median_of_run_e2e_medians_ms"],
            ),
            "persistent_vs_matched_dispatch_backend": reduction(
                dispatch["median_of_run_backend_medians_ms"],
                persistent["median_of_run_backend_medians_ms"],
            ),
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="ascii")
    print(encoded, end="")


if __name__ == "__main__":
    main()
