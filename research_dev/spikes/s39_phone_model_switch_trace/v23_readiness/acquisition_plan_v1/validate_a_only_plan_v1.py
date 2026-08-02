#!/usr/bin/env python3
"""Validate the canonical CP0-R1 V2.3 A_ONLY plan and source manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import plan_common_v1 as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=common.PLAN_PATH)
    parser.add_argument("--manifest", type=Path, default=common.MANIFEST_PATH)
    return parser.parse_args()


def validate_outer(plan_path: Path) -> None:
    source = (
        "from pathlib import Path\n"
        "import sys\n"
        f"sys.path.insert(0, {str(common.V23)!r})\n"
        f"sys.path.insert(0, {str(common.S39)!r})\n"
        "import acquire_a_only_v23 as outer\n"
        "import cp0_r1_evidence_v23 as v23\n"
        "contract, contract_raw, _, candidate_raw = v23.validate_inputs(\n"
        "    v23.DEFAULT_CONTRACT, v23.DEFAULT_CANDIDATE)\n"
        "roles = contract['phase_protocol']['phase_roles']['A_ONLY']\n"
        "outer._load_plan(Path(sys.argv[1]), contract_raw, candidate_raw, roles)\n"
    )
    with tempfile.TemporaryDirectory(prefix="s39-v23-pycache-") as cache:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-X",
                f"pycache_prefix={cache}",
                "-c",
                source,
                str(plan_path),
            ],
            capture_output=True,
            check=False,
            timeout=60,
        )
    common.require(
        completed.returncode == 0 and completed.stderr == b"",
        "E_OUTER_COMPATIBILITY: "
        + completed.stderr.decode("ascii", errors="backslashreplace").strip(),
    )


def validate(plan_path: Path, manifest_path: Path) -> dict:
    plan_path = plan_path.resolve()
    manifest_path = manifest_path.resolve()
    expected, plan_raw, manifest_raw = common.build_bundle()
    actual = common.validate_plan(plan_path)
    common.exact(common.read_regular(plan_path), plan_raw, "plan.bytes")
    common.exact(
        common.read_regular(manifest_path),
        manifest_raw,
        "manifest.bytes",
    )
    common.exact(
        Path(actual["source_manifest_path"]),
        manifest_path,
        "source_manifest_path",
    )
    common.validate_bound_source_manifest(actual)
    acquisition_files = common.bound_flag_paths(
        actual["drivers"]["acquisition"],
        common.DRIVER_FILE_FLAGS["acquisition"],
        "drivers.acquisition",
    )
    common.validate_nested_execution_graph(
        acquisition_files["--command-plan"],
        actual["contract_sha256"],
        actual["candidate_sha256"],
    )
    readiness_files = common.bound_flag_paths(
        actual["drivers"]["artifact"],
        common.DRIVER_FILE_FLAGS["artifact"],
        "drivers.artifact",
    )
    common.validate_runtime_bundle_plan(
        readiness_files["--runtime-bundle-plan"],
        actual["contract_sha256"],
        actual["candidate_sha256"],
    )
    validate_outer(plan_path)
    common.exact(actual, expected, "plan")
    return actual


def main() -> int:
    args = parse_args()
    try:
        plan = validate(args.plan, args.manifest)
        print(
            "A_ONLY_PLAN_VALIDATE_PASS "
            f"model_id={plan['model_id']} phase={plan['phase']}"
        )
        return 0
    except (
        common.PlanError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"A_ONLY_PLAN_VALIDATE_REFUSED: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
