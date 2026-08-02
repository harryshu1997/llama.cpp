#!/usr/bin/env python3
"""Apply and certify one desktop model page-cache regime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from executor_bundle import BundleError, validate_runtime_environment


try:
    validate_runtime_environment(Path(__file__).resolve())
except (BundleError, OSError) as error:
    print(f"executor bundle failed: {error}", file=sys.stderr)
    raise SystemExit(2)

if os.environ.get("S40_EXECUTOR_BUNDLE") != "1":
    source = (
        Path(__file__).resolve().parent.parents[1]
        / "s39_desktop_swap_baseline"
    )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import cache_control


WARM_MIN_PPM = 950_000
COLD_MAX_PPM = 50_000


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def local_stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "ctime_ns": value.st_ctime_ns,
        "device_id": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "mtime_ns": value.st_mtime_ns,
        "size": value.st_size,
    }


def apply_regime(path: Path, regime: str) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file():
        raise cache_control.CacheError("model path is not an absolute file")
    if regime not in ("WARM_CACHE", "COLD_NVME"):
        raise cache_control.CacheError("unknown cache regime")
    identity_before = local_stat(path)
    before = cache_control.resident_pages(path)
    started_ns = time.monotonic_ns()
    if regime == "WARM_CACHE":
        operation = cache_control.warm_file(path)
        limit_ppm = WARM_MIN_PPM
        passed = operation["resident_ppm"] >= limit_ppm
        comparison = "at_least"
    else:
        operation = cache_control.evict_file(path)
        limit_ppm = COLD_MAX_PPM
        passed = operation["resident_ppm"] <= limit_ppm
        comparison = "at_most"
    completed_ns = time.monotonic_ns()
    identity_after = local_stat(path)
    if identity_before != identity_after:
        raise cache_control.CacheError("model identity changed during cache control")
    if not passed:
        raise cache_control.CacheError(
            f"{regime}: resident_ppm={operation['resident_ppm']} "
            f"does not satisfy {comparison} {limit_ppm}"
        )
    return {
        "after": operation,
        "before": before,
        "completed_ns": completed_ns,
        "comparison": comparison,
        "limit_ppm": limit_ppm,
        "model_path": str(path),
        "model_stat": identity_after,
        "regime": regime,
        "schema": "s40-cache-control-result-v1",
        "started_ns": started_ns,
        "success": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regime",
        choices=("WARM_CACHE", "COLD_NVME"),
        required=True,
    )
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    result = apply_regime(args.model, args.regime)
    sys.stdout.buffer.write(canonical_bytes(result))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BundleError, cache_control.CacheError, OSError) as error:
        print(f"cache control failed: {error}", file=sys.stderr)
        raise SystemExit(2)
