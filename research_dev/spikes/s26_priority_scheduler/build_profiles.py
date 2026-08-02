#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from priority_profiles import canonical_bytes, derive_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(derive_bundle()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

