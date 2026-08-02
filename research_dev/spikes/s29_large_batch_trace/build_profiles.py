#!/usr/bin/env python3
"""Build the canonical S29 profile from a completed physical calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from large_batch_profiles import write_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    value = write_bundle(args.calibration, args.output)
    print(f"{args.output} routes={len(value['routes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
