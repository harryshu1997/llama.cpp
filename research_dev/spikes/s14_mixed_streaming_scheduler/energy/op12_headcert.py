#!/usr/bin/env python3
"""Measure one batch-1 Gemma head island on OP12/v75.

This wrapper reuses the frozen Stage-B measurement functions without changing
the OP15 evidence source. Each invocation produces one independently hashed
profile artifact for one layer depth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import stageb_headcert as SB


HERE = Path(__file__).resolve().parent
SERIAL = "5ae7a43d"
SOC = "SM8650"
HEXAGON = "v75"
GPU_UUID = "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f"
REMOTE_DIR = "/data/local/tmp/ls-s14-cpe"
REMOTE_BIN = "llama-layersplit"
ALLOWED_K = frozenset({6, 8})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(HERE)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def add_correctness(row: dict[str, Any], reference: list[int], requests: int) -> None:
    token_sets = row.get("token_ids_by_request")
    token_sets_valid = type(token_sets) is list and len(token_sets) == requests \
        and all(type(tokens) is list for tokens in token_sets)
    row["token_match_vs_mono"] = row.get("token_ids") == reference
    row["all_tokens_match_vs_mono"] = token_sets_valid \
        and all(tokens == reference for tokens in token_sets)


def point_passes(row: dict[str, Any], reference: list[int], requests: int,
                 thermal_start: dict[str, Any], thermal_end: dict[str, Any]) -> bool:
    add_correctness(row, reference, requests)
    return SB.row_is_eligible(row, requests) \
        and SB.thermal_ok(thermal_start, SB.THERMAL_START_MAX_MILLIC) \
        and SB.thermal_ok(thermal_end, SB.THERMAL_END_MAX_MILLIC)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True, choices=sorted(ALLOWED_K))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--n-gen", type=int, default=8)
    parser.add_argument("--driver-context", type=int, default=512)
    parser.add_argument("--driver-max-prefill", type=int, default=64)
    parser.add_argument("--prompt", default="Explain why batching improves accelerator utilization.")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in args.run_id):
        parser.error("--run-id must contain only ASCII letters, digits, '-' or '_'")
    if args.output.exists():
        parser.error("--output already exists")
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be in [1024, 65535]")
    for name in ("requests", "n_gen", "driver_context", "driver_max_prefill", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")

    log_dir = HERE / "op12_logs" / args.run_id
    if log_dir.exists():
        parser.error("run log directory already exists")
    log_dir.mkdir(parents=True)

    started_utc_us = time.time_ns() // 1000
    thermal_start = SB.thermal_snapshot(SERIAL)
    result: dict[str, Any] = {
        "schema": "s14-op12-head-placement-v1",
        "status": "OP12_HEAD_POINT_FAIL",
        "formal_claim": "NONE",
        "scope": "OP12_HTP0_PLACEMENT_CORRECTNESS_AND_LATENCY_PHONE_ENERGY_UNKNOWN",
        "run_id": args.run_id,
        "device": {"serial": SERIAL, "soc": SOC, "hexagon": HEXAGON, "backend": "HTP0"},
        "gpu_uuid": GPU_UUID,
        "model": "gemma-4-12b-it-f16",
        "model_version": "sha256:bac4e29337273d1cb41aa36f4209ebcf63baa1162e6ab4dc55f6034b3d69b35a",
        "layer_range": [0, args.k],
        "measurement_config": {
            "batch": 1,
            "requests": args.requests,
            "warmups": args.warmups,
            "n_gen": args.n_gen,
            "driver_context": args.driver_context,
            "driver_max_prefill": args.driver_max_prefill,
            "prompt": args.prompt,
        },
        "process_identity": {
            "host_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip(),
            "pid": os.getpid(),
            "started_utc_us": started_utc_us,
            "ended_utc_us": None,
        },
        "decode_fa_policy": "AUTO_WITH_V75_FUSED_FA_DISABLED_BY_BACKEND_GATE",
        "phone_binary_sha256": SB.sha256_remote(SERIAL, f"{REMOTE_DIR}/{REMOTE_BIN}"),
        "host_binary_sha256": sha256_file(Path(SB.HOST_BIN)),
        "wrapper_sha256": sha256_file(Path(__file__)),
        "stageb_source_sha256": sha256_file(Path(SB.__file__)),
        "thermal": {
            "start": thermal_start,
            "end": None,
            "start_max_millic": SB.THERMAL_START_MAX_MILLIC,
            "end_max_millic": SB.THERMAL_END_MAX_MILLIC,
        },
        "reference_token_ids": None,
        "point": None,
        "artifact_files": [],
        "error": None,
        "caveats": [
            "This is a batch-1, one-process profile point unless aggregated with independent artifacts.",
            "Phone and total-system energy are unknown.",
            "Only GET_ROWS on CPU is an allowed placement exception.",
            "The host A6000 tail and USB boundary remain part of request_wall_us.",
        ],
    }

    exit_code = 2
    try:
        if not SB.thermal_ok(thermal_start, SB.THERMAL_START_MAX_MILLIC):
            raise RuntimeError("invalid or hot start thermal sample")
        reference = SB.run_mono_reference(
            GPU_UUID, args.prompt, args.n_gen, args.timeout, batch=1,
            ctx=args.driver_context, max_prefill=args.driver_max_prefill,
        )
        row = SB.run_one_k(
            SERIAL, GPU_UUID, args.k, args.port, 1, args.n_gen, args.requests,
            args.warmups, args.driver_context, args.driver_max_prefill, args.prompt,
            REMOTE_DIR, REMOTE_BIN, log_dir, args.timeout, decode_no_fa=False,
        )
        thermal_end = SB.thermal_snapshot(SERIAL)
        result["thermal"]["end"] = thermal_end
        result["reference_token_ids"] = reference
        result["point"] = row
        result["artifact_files"] = [
            artifact_file(log_dir / f"phone_k{args.k}.log"),
            artifact_file(log_dir / f"host_k{args.k}_b1.stderr"),
        ]
        if point_passes(row, reference, args.requests, thermal_start, thermal_end):
            result["status"] = "OP12_HEAD_POINT_PASS"
            result["formal_claim"] = "OP12_BATCH1_HEAD_PLACEMENT_CORRECTNESS_AND_LATENCY_POINT"
            exit_code = 0
        else:
            result["error"] = "point failed placement, correctness, completion, or thermal gate"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if result["thermal"]["end"] is None:
            result["thermal"]["end"] = SB.thermal_snapshot(SERIAL)

    result["process_identity"]["ended_utc_us"] = time.time_ns() // 1000
    atomic_write_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "k": args.k,
        "output": str(args.output),
        "error": result["error"],
    }, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
