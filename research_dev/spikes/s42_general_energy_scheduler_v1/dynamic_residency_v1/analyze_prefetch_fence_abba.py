#!/usr/bin/env python3
"""Compare matched observe/prefetch OP15-fenced physical runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .analyze_prefetch_fence_run import SCHEMA, canonical, digest, require
except ImportError:
    from analyze_prefetch_fence_run import SCHEMA, canonical, digest, require


ABBA_SCHEMA = "s42-op15-fenced-gpu-prefetch-abba-v1"


def read_run(path: Path, mode: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(
        type(value) is dict
        and value.get("schema") == SCHEMA
        and value.get("status") == "PASS"
        and value.get("mode") == mode,
        f"{mode} run",
    )
    return value


def mean(rows: list[dict[str, Any]], section: str, field: str) -> float:
    return sum(row[section][field] for row in rows) / len(rows)


def change(control: float, treatment: float) -> float:
    return (treatment / control - 1) * 100


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observe-r1", type=Path, required=True)
    parser.add_argument("--prefetch-r1", type=Path, required=True)
    parser.add_argument("--prefetch-r2", type=Path, required=True)
    parser.add_argument("--observe-r2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    paths = (
        args.observe_r1,
        args.prefetch_r1,
        args.prefetch_r2,
        args.observe_r2,
    )
    try:
        runs = {
            "observe_r1": read_run(args.observe_r1, "observe"),
            "prefetch_r1": read_run(args.prefetch_r1, "prefetch"),
            "prefetch_r2": read_run(args.prefetch_r2, "prefetch"),
            "observe_r2": read_run(args.observe_r2, "observe"),
        }
        observe = [runs["observe_r1"], runs["observe_r2"]]
        prefetch = [runs["prefetch_r1"], runs["prefetch_r2"]]
        work = [row["qwen"]["work"] for row in runs.values()]
        require(all(row == work[0] for row in work[1:]), "equal Qwen work")
        require(all(
            row["gates"]["transfer_completed_inside_phone_window"]
            for row in prefetch
        ), "prefetch window upper bound")
        observe_mean = {
            "cpu_package_j": mean(observe, "energy", "cpu_package_j"),
            "duration_s": mean(observe, "qwen", "duration_s"),
            "fleet_j": mean(observe, "energy", "fleet_j"),
            "gpu_board_j": mean(observe, "energy", "gpu_board_j"),
            "phone_j": mean(observe, "energy", "phone_j"),
            "server_j": mean(observe, "energy", "server_j"),
        }
        prefetch_mean = {
            "cpu_package_j": mean(prefetch, "energy", "cpu_package_j"),
            "duration_s": mean(prefetch, "qwen", "duration_s"),
            "fleet_j": mean(prefetch, "energy", "fleet_j"),
            "gpu_board_j": mean(prefetch, "energy", "gpu_board_j"),
            "phone_j": mean(prefetch, "energy", "phone_j"),
            "server_j": mean(prefetch, "energy", "server_j"),
        }
        changes = {
            f"{field}_change_pct": change(observe_mean[field], value)
            for field, value in prefetch_mean.items()
        }
        pair_energy_changes = [
            change(
                runs[f"observe_r{index}"]["energy"]["fleet_j"],
                runs[f"prefetch_r{index}"]["energy"]["fleet_j"],
            )
            for index in (1, 2)
        ]
        output: dict[str, Any] = {
            "admission": "FENCE_TRANSFER_QUALIFIED_NO_WEIGHT_ADOPTION",
            "artifacts": {
                name + "_sha256": digest(path)
                for name, path in zip(runs, paths, strict=True)
            },
            "changes": changes,
            "gates": {
                "equal_qwen_work": True,
                "cuda_context_warmed_before_paid": all(
                    row["gates"]["cuda_context_warmed_before_paid"]
                    for row in runs.values()
                ),
                "gpu_reserve_preserved": all(
                    row["gates"]["gpu_reserve_preserved"]
                    for row in runs.values()
                ),
                "prefetch_bytes_verified": all(
                    row["gates"]["destination_bytes_verified"]
                    for row in prefetch
                ),
                "resident_pinned_source_verified": all(
                    row["gates"]["resident_pinned_source_verified"]
                    for row in runs.values()
                ),
                "prefetch_inside_phone_window": True,
                "protected_work_serialized": all(
                    row["gates"]["protected_work_serialized_after_copy"]
                    for row in runs.values()
                ),
            },
            "observe_mean": observe_mean,
            "pair_fleet_energy_change_pct": pair_energy_changes,
            "prefetch_mean": prefetch_mean,
            "run_order": [
                "observe_r1", "prefetch_r1", "prefetch_r2", "observe_r2"
            ],
            "runs": {
                name: {
                    "duration_s": row["qwen"]["duration_s"],
                    "fleet_j": row["energy"]["fleet_j"],
                    "gpu_free_min_bytes": row["helper"]["gpu_free_min_bytes"],
                    "overrun_max_ms": row["bridge"][
                        "prefetch_copied_window_overrun_max_ms"
                    ],
                    "record_sha256": row["record_sha256"],
                    "stage_bytes": row["helper"]["stage_bytes"],
                }
                for name, row in runs.items()
            },
            "schema": ABBA_SCHEMA,
            "status": "PASS",
            "weight_adoptable_by_gemma_executor": False,
        }
        require(all(output["gates"].values()), "ABBA gates")
        output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
        args.output.write_bytes(canonical(output))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"prefetch ABBA analysis failed: {exc}\n")
    print(json.dumps({
        "fleet_j_change_pct": output["changes"]["fleet_j_change_pct"],
        "output": str(args.output),
        "status": output["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
