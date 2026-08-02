#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from selected_profiles import write_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    value = write_bundle(args.calibration, args.output)
    print(f"{args.output} cut={value['selected_cut']} routes={len(value['routes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
