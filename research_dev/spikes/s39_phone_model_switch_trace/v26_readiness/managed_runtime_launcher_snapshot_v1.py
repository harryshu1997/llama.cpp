#!/usr/bin/python3 -I
"""Run the frozen USB launcher with a phase-fresh snapshotted boot ID.

The frozen V2.4 phone-route launch plans carry a 5-element process argv with
no `--boot-id` (the prospective plans may not embed boot identity), while the
frozen USB launcher requires one. This wrapper closes that seam exactly as
`boot_id_source=phase_fresh_snapshot` declares: it validates the plan through
the FROZEN launcher stack, snapshots the phone's live boot ID itself, and
then delegates the launch to the frozen `execute()` path unchanged. The
downstream frozen producer compares the resulting RUNTIMEPROCESS boot ID
against the phase-lock-bound plan value, so a stale or rebooted phone still
fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any


USB_SOURCE = Path(__file__).resolve().with_name(
    "managed_runtime_launcher_usb_v1.py"
)
USB_SHA256 = "52c4e1f251f4daa2c856857fcdf23fdc5a85c0d30eff93365996a8f1378611ce"
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
MAX_PLAN_BYTES = 8 * 1024 * 1024


class SnapshotLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SnapshotLaunchError(message)


def load_pinned_module(path: Path, expected_sha256: str) -> types.ModuleType:
    raw = path.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == expected_sha256,
        f"E_PINNED_SOURCE: {path.name}",
    )
    module = types.ModuleType(f"_s39_snapshot_{path.stem}")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def parse_plan(raw_text: str, expected_sha256: str) -> dict[str, Any]:
    require(
        type(raw_text) is str and 0 < len(raw_text) <= MAX_PLAN_BYTES,
        "E_PLAN_SIZE",
    )
    raw = raw_text.encode("ascii")
    require(
        hashlib.sha256(raw).hexdigest() == expected_sha256,
        "E_PLAN_DIGEST",
    )
    value = json.loads(raw_text)
    require(type(value) is dict, "E_PLAN_TYPE")
    return value


def snapshot_boot_id(android: dict[str, Any]) -> str:
    adb_path = android["adb_path"]
    adb_raw = Path(adb_path).read_bytes()
    require(
        hashlib.sha256(adb_raw).hexdigest() == android["adb_sha256"],
        "E_ADB_BINARY",
    )
    completed = subprocess.run(
        [
            adb_path,
            "-P",
            str(android["adb_port"]),
            "-s",
            android["physical_serial"],
            "shell",
            "cat /proc/sys/kernel/random/boot_id",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    require(completed.returncode == 0, "E_BOOT_PROBE")
    boot_id = completed.stdout.decode("ascii").strip()
    require(UUID_RE.fullmatch(boot_id) is not None, "E_BOOT_FORMAT")
    return boot_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--plan-sha256", required=True)
    try:
        args = parser.parse_args(argv)
        plan = parse_plan(args.plan_json, args.plan_sha256)
        android = plan.get("android")
        require(type(android) is dict, "E_ANDROID")
        require(
            android.get("boot_id_source") == "phase_fresh_snapshot",
            "E_BOOT_SOURCE",
        )
        usb = load_pinned_module(USB_SOURCE, USB_SHA256)
        boot_id = snapshot_boot_id(android)
        return usb.execute(args.plan_json, args.plan_sha256, boot_id)
    except (
        SnapshotLaunchError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"SNAPSHOT_LAUNCH_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
