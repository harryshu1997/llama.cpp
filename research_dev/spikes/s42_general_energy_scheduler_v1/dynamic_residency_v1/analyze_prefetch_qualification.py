#!/usr/bin/env python3
"""Bind the OP15-fenced GPU prefetch qualification receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .analyze_prefetch_fence_abba import ABBA_SCHEMA
    from .analyze_prefetch_fence_run import (
        GEMMA_MODEL_SHA256,
        SCHEMA,
        TOKEN_EMBEDDING_BYTES,
        TOKEN_EMBEDDING_OFFSET,
        canonical,
        digest,
        require,
    )
except ImportError:
    from analyze_prefetch_fence_abba import ABBA_SCHEMA
    from analyze_prefetch_fence_run import (
        GEMMA_MODEL_SHA256,
        SCHEMA,
        TOKEN_EMBEDDING_BYTES,
        TOKEN_EMBEDDING_OFFSET,
        canonical,
        digest,
        require,
    )


QUALIFICATION_SCHEMA = "s42-op15-fenced-gpu-prefetch-qualification-v1"
RUN_ORDER = (
    "pinned-abba-observe-r1",
    "pinned-abba-prefetch-r1",
    "pinned-abba-prefetch-r2",
    "pinned-abba-observe-r2",
)
MODES = ("observe", "prefetch", "prefetch", "observe")
FULL_HEAD_RUN = "full-head-prefetch-pilot"
DEFAULT_RESULT_ROOT = (
    Path(__file__).resolve().parent
    / "results/OP15_FENCED_GPU_PREFETCH_V1"
)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object: {path}")
    return value


def verify_canonical_record(value: dict[str, Any], schema: str) -> None:
    require(
        value.get("schema") == schema and value.get("status") == "PASS",
        f"{schema} status",
    )
    claimed = value.get("record_sha256")
    require(type(claimed) is str and len(claimed) == 64, "record hash")
    unhashed = dict(value)
    del unhashed["record_sha256"]
    require(
        hashlib.sha256(canonical(unhashed)).hexdigest() == claimed,
        f"{schema} canonical hash",
    )


def run_paths(root: Path, name: str) -> dict[str, Path]:
    raw = root / "raw"
    return {
        "bridge": raw / f"{name}.phone-capture/bridge.stderr",
        "helper": raw / f"{name}.prefetch-probe/helper.stdout",
        "phone": raw / f"{name}.phone-capture/PHONE_ENERGY_V3.json",
        "qwen": raw / f"{name}/RESULT.json",
        "resource": raw / f"{name}/resource-samples.jsonl",
        "run": raw / f"{name}.prefetch-probe/PREFETCH_RUN_V1.json",
        "server": raw / f"{name}.phone-capture/server.stderr",
    }


def verify_run_artifacts(run: dict[str, Any], paths: dict[str, Path]) -> None:
    expected = {
        "bridge_log_sha256": "bridge",
        "helper_log_sha256": "helper",
        "phone_energy_sha256": "phone",
        "qwen_result_sha256": "qwen",
        "server_log_sha256": "server",
    }
    for field, name in expected.items():
        require(
            run["artifacts"].get(field) == digest(paths[name]),
            f"run artifact hash: {field}",
        )


def gpu_utilization(
    path: Path,
    paid_start_ns: int,
    paid_end_ns: int,
) -> dict[str, float | int]:
    samples: list[int] = []
    with path.open("r", encoding="ascii") as stream:
        for line in stream:
            row = json.loads(line)
            require(type(row) is dict and type(row.get("gpu")) is dict,
                    "GPU resource sample")
            gpu = row["gpu"]
            sample_ns = gpu.get("sample_t_ns")
            utilization = gpu.get("utilization_pct")
            require(
                type(sample_ns) is int
                and type(utilization) is int
                and 0 <= utilization <= 100,
                "GPU utilization sample",
            )
            if paid_start_ns <= sample_ns <= paid_end_ns:
                samples.append(utilization)
    require(samples, "paid GPU utilization samples")
    count = len(samples)
    return {
        "mean_pct": sum(samples) / count,
        "nonzero_pct": 100 * sum(value > 0 for value in samples) / count,
        "samples": count,
        "samples_at_or_above_90_pct": sum(value >= 90 for value in samples),
        "sample_max_pct": max(samples),
    }


def copied_windows(path: Path) -> dict[str, float | int]:
    copied_bytes = 0
    copied_chunks = 0
    copy_ms = 0.0
    windows = 0
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith("PREFETCH_WINDOW "):
            continue
        fields = dict(item.split("=", 1) for item in line.split()[1:])
        if int(fields["armed"]) != 1 or int(fields["copied_bytes"]) == 0:
            continue
        windows += 1
        copied_bytes += int(fields["copied_bytes"])
        copied_chunks += int(fields["copied_chunks"])
        copy_ms += float(fields["copy_ms"])
    return {
        "copied_bytes": copied_bytes,
        "copied_chunks": copied_chunks,
        "copy_active_ms": copy_ms,
        "copy_windows": windows,
    }


def load_run(root: Path, name: str, mode: str) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Path]
]:
    paths = run_paths(root, name)
    run = read_object(paths["run"])
    verify_canonical_record(run, SCHEMA)
    require(run.get("mode") == mode, f"run mode: {name}")
    verify_run_artifacts(run, paths)
    utilization = gpu_utilization(
        paths["resource"],
        run["qwen"]["paid_start_ns"],
        run["qwen"]["paid_end_ns"],
    )
    return run, utilization, paths


def build(root: Path = DEFAULT_RESULT_ROOT) -> dict[str, Any]:
    abba_path = root / "PINNED_PREFETCH_ABBA_V1.json"
    abba = read_object(abba_path)
    verify_canonical_record(abba, ABBA_SCHEMA)
    require(
        abba.get("run_order")
        == ["observe_r1", "prefetch_r1", "prefetch_r2", "observe_r2"],
        "ABBA run order",
    )

    runs: dict[str, dict[str, Any]] = {}
    utilization: dict[str, dict[str, float | int]] = {}
    paths_by_run: dict[str, dict[str, Path]] = {}
    for name, mode in zip(RUN_ORDER, MODES, strict=True):
        run, gpu, paths = load_run(root, name, mode)
        short_name = name.removeprefix("pinned-abba-").replace("-r", "_r")
        runs[short_name] = run
        utilization[short_name] = gpu
        paths_by_run[short_name] = paths
        require(
            abba["artifacts"][short_name + "_sha256"]
            == digest(paths["run"]),
            f"ABBA run hash: {short_name}",
        )
        require(
            abba["runs"][short_name]["record_sha256"]
            == run["record_sha256"],
            f"ABBA record hash: {short_name}",
        )

    prefetched = [runs["prefetch_r1"], runs["prefetch_r2"]]
    copied = [
        copied_windows(paths_by_run[name]["helper"])
        for name in ("prefetch_r1", "prefetch_r2")
    ]
    for run, windows in zip(prefetched, copied, strict=True):
        require(
            windows["copied_bytes"] == run["helper"]["copied_bytes"]
            and windows["copied_chunks"] == run["helper"]["copied_chunks"]
            and windows["copy_windows"] == run["helper"]["copy_windows"],
            "copied-window accounting",
        )

    full, full_gpu, full_paths = load_run(root, FULL_HEAD_RUN, "prefetch")
    full_windows = copied_windows(full_paths["helper"])
    require(
        full["model_source"]["model_sha256"] == GEMMA_MODEL_SHA256
        and full["model_source"]["tensor"] == "token_embd.weight"
        and full["model_source"]["tensor_offset"]
            == TOKEN_EMBEDDING_OFFSET
        and full["model_source"]["tensor_bytes"]
            == TOKEN_EMBEDDING_BYTES
        and full["helper"]["stage_bytes"] == TOKEN_EMBEDDING_BYTES
        and full["helper"]["copied_bytes"] == TOKEN_EMBEDDING_BYTES
        and full["helper"]["verified"] is True
        and full["helper"]["source_fnv64"]
            == full["helper"]["destination_fnv64"],
        "full tied-output tensor verification",
    )
    require(
        full_windows["copied_bytes"] == TOKEN_EMBEDDING_BYTES
        and full_windows["copied_chunks"] == full["helper"]["copied_chunks"]
        and full_windows["copy_windows"] == full["helper"]["copy_windows"],
        "full tied-output window accounting",
    )

    observe_max_pair_delta = max(
        abs(value) for value in abba["pair_fleet_energy_change_pct"]
    )
    mean_delta = abba["changes"]["fleet_j_change_pct"]
    reserve_margin = (
        full["helper"]["gpu_free_min_bytes"]
        - full["helper"]["gpu_reserve_bytes"]
    )
    output: dict[str, Any] = {
        "admission": "FENCE_TRANSFER_QUALIFIED_NO_WEIGHT_ADOPTION",
        "artifacts": {
            "abba_sha256": digest(abba_path),
            "full_head_run_sha256": digest(full_paths["run"]),
            "raw_resource_sample_sha256": {
                name: digest(paths_by_run[name]["resource"])
                for name in paths_by_run
            } | {"full_head": digest(full_paths["resource"])},
        },
        "claim": {
            "gpu_continuously_busy": False,
            "incremental_dynamic_energy_savings_pct": None,
            "next_required_gate": (
                "SAME_PROCESS_GEMMA_WEIGHT_ADOPTION_AND_EXECUTION"
            ),
            "transfer_mechanism_qualified": True,
            "weight_adoptable_by_gemma_executor": False,
        },
        "full_tied_output_tensor": {
            "chunk_bytes": full["helper"]["chunk_bytes"],
            "chunks_per_window": full["helper"]["chunks_per_window"],
            "copied_bytes": full["helper"]["copied_bytes"],
            "copied_chunks": full["helper"]["copied_chunks"],
            "copy_active_fraction": (
                full_windows["copy_active_ms"]
                / (1000 * full["qwen"]["duration_s"])
            ),
            "copy_active_ms": full_windows["copy_active_ms"],
            "copy_p50_ms": full["helper"]["copy_p50_ms"],
            "copy_p90_ms": full["helper"]["copy_p90_ms"],
            "copy_windows": full["helper"]["copy_windows"],
            "gpu_free_min_bytes": full["helper"]["gpu_free_min_bytes"],
            "gpu_reserve_margin_bytes": reserve_margin,
            "qwen_duration_s": full["qwen"]["duration_s"],
            "tensor": "token_embd.weight",
            "verified": True,
            "window_overrun_max_ms": full["bridge"][
                "prefetch_copied_window_overrun_max_ms"
            ],
        },
        "gates": {
            "abba_artifacts_rehashed": True,
            "equal_qwen_work": abba["gates"]["equal_qwen_work"],
            "full_tensor_bytes_verified": True,
            "gpu_reserve_preserved": (
                abba["gates"]["gpu_reserve_preserved"]
                and reserve_margin >= 0
            ),
            "no_copied_window_overrun": (
                all(
                    row["bridge"][
                        "prefetch_copied_window_overrun_max_ms"
                    ] == 0
                    for row in prefetched
                )
                and full["bridge"][
                    "prefetch_copied_window_overrun_max_ms"
                ] == 0
            ),
            "resident_pinned_source_verified": (
                abba["gates"]["resident_pinned_source_verified"]
                and full["helper"]["source_pinned"] is True
                and full["helper"]["source_resident_bytes"]
                    == TOKEN_EMBEDDING_BYTES
            ),
        },
        "gpu_utilization": utilization | {"full_head": full_gpu},
        "matched_256_mib_abba": {
            "copy_active_fraction": [
                row["copy_active_ms"]
                / (1000 * run["qwen"]["duration_s"])
                for row, run in zip(copied, prefetched, strict=True)
            ],
            "copy_active_ms": [row["copy_active_ms"] for row in copied],
            "duration_change_pct": abba["changes"][
                "duration_s_change_pct"
            ],
            "fleet_energy_change_pct": mean_delta,
            "fleet_energy_pair_change_pct": abba[
                "pair_fleet_energy_change_pct"
            ],
            "mean_change_abs_below_max_abs_pair_change": (
                abs(mean_delta) < observe_max_pair_delta
            ),
            "observe_mean_fleet_j": abba["observe_mean"]["fleet_j"],
            "prefetch_mean_fleet_j": abba["prefetch_mean"]["fleet_j"],
            "stage_bytes": prefetched[0]["helper"]["stage_bytes"],
        },
        "schema": QUALIFICATION_SCHEMA,
        "status": "PASS",
    }
    require(all(output["gates"].values()), "qualification gates")
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root", type=Path, default=DEFAULT_RESULT_ROOT
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    try:
        output = build(args.result_root)
        args.output.write_bytes(canonical(output))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"prefetch qualification failed: {exc}\n")
    print(json.dumps({
        "admission": output["admission"],
        "output": str(args.output),
        "status": output["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
