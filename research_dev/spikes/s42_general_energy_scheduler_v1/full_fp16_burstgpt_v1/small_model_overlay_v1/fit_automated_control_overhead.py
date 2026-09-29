#!/usr/bin/env python3
"""Fit a conservative controller-delay profile from physical run results."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path
from typing import Any


SCHEMA = "s42-automated-control-overhead-profile-v1"


class ControlOverheadFitError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlOverheadFitError(message)


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def load_result(path: Path) -> tuple[int, float, float, dict[str, object]]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"result object: {path}")
    runtime = value.get("scheduler_runtime")
    require(type(runtime) is dict, f"scheduler runtime: {path}")
    overhead = runtime.get("overhead")
    require(type(overhead) is dict, f"scheduler overhead: {path}")
    total = overhead.get("total_controller_ns")
    require(type(total) is dict, f"total controller overhead: {path}")
    samples = total.get("samples")
    mean_us = total.get("mean_us")
    maximum_us = total.get("max_us")
    require(
        type(samples) is int
        and samples > 0
        and type(mean_us) in {int, float}
        and type(maximum_us) in {int, float}
        and 0 <= mean_us <= maximum_us,
        f"controller overhead values: {path}",
    )
    return samples, float(mean_us), float(maximum_us), {
        "path": str(path),
        "sha256": digest(path),
    }


def fit(paths: tuple[Path, ...]) -> dict[str, Any]:
    require(paths, "at least one physical result is required")
    rows = [load_result(path) for path in paths]
    samples = sum(row[0] for row in rows)
    require(samples >= 2, "repeated controller observations are required")
    mean_us = ceil(sum(row[0] * row[1] for row in rows) / samples)
    upper_us = ceil(max(row[2] for row in rows))
    return {
        "mean_us": mean_us,
        "measured": True,
        "sample_count": samples,
        "schema": SCHEMA,
        "sources": [row[3] for row in rows],
        "upper_kind": "maximum_observed_controller_wall_time",
        "upper_us": upper_us,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute(), "output path must be absolute")
    require(not args.output.exists(), "output path must be new")
    result = fit(tuple(args.result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    require(not temporary.exists(), "temporary output path must be new")
    try:
        temporary.write_bytes(canonical(result))
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
