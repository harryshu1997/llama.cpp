#!/usr/bin/env python3
"""Build the sole canonical CP0-R1 V2.3 A_ONLY acquisition plan."""

from __future__ import annotations

import argparse
from pathlib import Path

import plan_common_v1 as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=common.HERE)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def run(output_dir: Path, check: bool) -> tuple[str, str]:
    output_dir = output_dir.resolve()
    _, plan_raw, manifest_raw = common.build_bundle()
    plan_path = output_dir / common.PLAN_PATH.name
    manifest_path = output_dir / common.MANIFEST_PATH.name
    if check:
        common.exact(common.read_regular(plan_path), plan_raw, "plan.bytes")
        common.exact(
            common.read_regular(manifest_path),
            manifest_raw,
            "manifest.bytes",
        )
    else:
        common.write_exclusive(plan_path, plan_raw)
        try:
            common.write_exclusive(manifest_path, manifest_raw)
        except Exception:
            plan_path.unlink(missing_ok=True)
            raise
    return common.sha256_bytes(plan_raw), common.sha256_bytes(manifest_raw)


def main() -> int:
    args = parse_args()
    try:
        plan_sha256, manifest_sha256 = run(args.output_dir, args.check)
        print(
            "A_ONLY_PLAN_BUILD_PASS "
            f"plan_sha256={plan_sha256} "
            f"manifest_sha256={manifest_sha256}"
        )
        return 0
    except (common.PlanError, OSError, ValueError) as error:
        print(f"A_ONLY_PLAN_BUILD_REFUSED: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
