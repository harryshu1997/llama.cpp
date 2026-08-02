#!/usr/bin/env python3
"""Build the originator spec via the frozen MAT validators and originate.

This driver preserves every `materialize_a_only_inputs_v1.build_spec`
validation (static cross-binding, live-identity exclusion, operator and
topology binding) and the frozen originator chain. The single deviation from
`materialize_a_only_inputs_v1.materialize` is the launcher-compatibility
step: process argv[0] is pinned to the snapshot launcher (which delegates to
the frozen USB launcher after a phase-fresh boot snapshot) instead of the
frozen USB launcher itself, and every inline plan is still validated through
the FROZEN launcher stack here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SNAPSHOT_LAUNCHER = HERE / "managed_runtime_launcher_snapshot_v1.py"
SNAPSHOT_PROBE = HERE / "phone_runtime_probe_snapshot_v1.py"


class DriverError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DriverError(message)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"E_IMPORT: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> tuple[bytes, str]:
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def validate_snapshot_processes(
    mat: Any,
    phone_static: dict[str, Any],
) -> None:
    usb_module = load_module(
        "s39_driver_usb", HERE / "managed_runtime_launcher_usb_v1.py"
    )
    frozen = usb_module.load_frozen_launcher()
    launcher_raw, launcher_sha = sha256_file(SNAPSHOT_LAUNCHER)
    probe_raw, probe_sha = sha256_file(SNAPSHOT_PROBE)
    for name, process in sorted(phone_static["processes"].items()):
        argv = process["argv"]
        require(
            argv[0] == str(SNAPSHOT_LAUNCHER),
            f"E_SNAPSHOT_LAUNCHER_PATH: {name}",
        )
        require(
            process["launcher_bytes"] == len(launcher_raw)
            and process["launcher_sha256"] == launcher_sha,
            f"E_SNAPSHOT_LAUNCHER_PIN: {name}",
        )
        plan_index = argv.index("--plan-json") + 1
        digest_index = argv.index("--plan-sha256") + 1
        try:
            usb_module.validate_plan(
                argv[plan_index],
                argv[digest_index],
                frozen,
            )
        except Exception as error:
            raise DriverError(
                f"E_FROZEN_PLAN_VALIDATION: {name}: {error}"
            ) from error
    for endpoint, probe in sorted(phone_static["probes"].items()):
        for side in ("before_argv", "after_argv"):
            require(
                probe[side][0] == str(SNAPSHOT_PROBE),
                f"E_SNAPSHOT_PROBE_PATH: {endpoint}.{side}",
            )
        require(
            probe["launcher_bytes"] == len(probe_raw)
            and probe["launcher_sha256"] == probe_sha,
            f"E_SNAPSHOT_PROBE_PIN: {endpoint}",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--spec-output", type=Path, required=True)
    parser.add_argument("--cuda-route-launch", type=Path, required=True)
    parser.add_argument("--joint-capture-plan", type=Path, required=True)
    parser.add_argument("--phone-route-launch", type=Path, required=True)
    parser.add_argument("--runtime-plan", type=Path, required=True)
    parser.add_argument("--prospective-root", type=Path, required=True)
    parser.add_argument("--dry-run-report", type=Path, required=True)
    try:
        args = parser.parse_args(argv)
        mat = load_module(
            "s39_driver_mat", HERE / "materialize_a_only_inputs_v1.py"
        )
        spec = mat.build_spec(
            args.inventory.resolve(), args.inventory_sha256
        )
        validate_snapshot_processes(mat, spec["phone_route_static"])
        mat.write_new(args.spec_output.resolve(), spec)
        try:
            originator = mat._load_originator()
            originator.materialize(
                spec_path=args.spec_output.resolve(),
                output_paths={
                    "cuda_route_launch": args.cuda_route_launch.resolve(),
                    "joint_capture_plan": args.joint_capture_plan.resolve(),
                    "phone_route_launch": args.phone_route_launch.resolve(),
                    "runtime_plan": args.runtime_plan.resolve(),
                },
                prospective_root_output=args.prospective_root.resolve(),
                report_output=args.dry_run_report.resolve(),
            )
        except Exception:
            try:
                args.spec_output.resolve().unlink()
            except FileNotFoundError:
                pass
            raise
        return 0
    except (
        AttributeError,
        DriverError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"V26_ORIGINATE_DRIVER_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
