#!/usr/bin/env python3
"""Compatibility CLI for the canonical scheduler profile materializer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    GIB,
    MatmulScheduleError,
    materialize_matmul_profile,
)


materialize = materialize_matmul_profile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a 4060 Ti plus OP15 matmul VQ shadow profile"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generic-family", action="store_true")
    parser.add_argument("--gpu-capacity-bytes", type=int, default=16 * GIB)
    parser.add_argument("--phone-capacity-bytes", type=int, default=10 * GIB)
    parser.add_argument("--gpu-reserved-bytes", type=int, default=0)
    parser.add_argument("--phone-reserved-bytes", type=int, default=0)
    parser.add_argument("--cpu-resource-id", default="compute:cpu")
    parser.add_argument("--gpu-resource-id", default="compute:gpu")
    parser.add_argument("--phone-resource-id", default="compute:phone")
    parser.add_argument("--pcie-resource-id", default="link:pcie")
    parser.add_argument("--usb-resource-id", default="link:usb")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output.exists():
        raise MatmulScheduleError("refusing to overwrite output")
    try:
        source = json.loads(args.input.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MatmulScheduleError(f"cannot read kernel campaign: {exc}") from exc
    profile = materialize_matmul_profile(
        source,
        generic_family=args.generic_family,
        gpu_capacity_bytes=args.gpu_capacity_bytes,
        phone_capacity_bytes=args.phone_capacity_bytes,
        gpu_reserved_bytes=args.gpu_reserved_bytes,
        phone_reserved_bytes=args.phone_reserved_bytes,
        resource_ids={
            "cpu": args.cpu_resource_id,
            "gpu": args.gpu_resource_id,
            "phone": args.phone_resource_id,
            "pcie": args.pcie_resource_id,
            "usb": args.usb_resource_id,
        },
    )
    args.output.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
