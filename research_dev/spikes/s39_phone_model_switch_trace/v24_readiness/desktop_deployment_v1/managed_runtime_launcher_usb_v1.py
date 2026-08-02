#!/usr/bin/python3 -I
"""Run the frozen managed launcher with a sealed USB-only Android selector."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import types
from typing import Any


FROZEN_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "v23_readiness"
    / "a_only_acquisition_driver_v1"
    / "producers_v1"
    / "managed_runtime_launcher_v1.py"
)
FROZEN_SOURCE_SHA256 = (
    "b97941dc30399135b04e98dbdf102aaeb6c695c55a7b990402aeafd21ed4a245"
)
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_PLAN_BYTES = 8 * 1024 * 1024
ADB_PORT = 5038
ENDPOINT_SERIALS = {
    "op12": "5ae7a43d",
    "op15": "3C15AU002CL00000",
}


class UsbLauncherError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise UsbLauncherError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}",
    )


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def read_frozen_source(
    path: Path = FROZEN_SOURCE,
    expected_sha256: str = FROZEN_SOURCE_SHA256,
) -> bytes:
    require(path.is_absolute(), "E_FROZEN_SOURCE_PATH")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise UsbLauncherError(f"E_FROZEN_SOURCE_OPEN: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "E_FROZEN_SOURCE_TYPE")
        require(0 < before.st_size <= MAX_SOURCE_BYTES, "E_FROZEN_SOURCE_SIZE")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            require(bool(chunk), "E_FROZEN_SOURCE_SHORT")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"", "E_FROZEN_SOURCE_GROWTH")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        all(getattr(before, key) == getattr(after, key) for key in identity),
        "E_FROZEN_SOURCE_CHANGED",
    )
    raw = b"".join(chunks)
    exact(hashlib.sha256(raw).hexdigest(), expected_sha256, "frozen_source.sha256")
    return raw


def load_frozen_launcher(
    path: Path = FROZEN_SOURCE,
    expected_sha256: str = FROZEN_SOURCE_SHA256,
) -> types.ModuleType:
    raw = read_frozen_source(path, expected_sha256)
    module = types.ModuleType("_s39_frozen_managed_runtime_launcher_v1")
    module.__file__ = str(path)
    try:
        code = compile(raw, str(path), "exec")
        exec(code, module.__dict__)
    except (SyntaxError, UnicodeError) as error:
        raise UsbLauncherError("E_FROZEN_SOURCE_COMPILE") from error
    return module


def parse_original_plan(
    raw_text: str,
    expected_sha256: str,
    launcher: types.ModuleType,
) -> dict[str, Any]:
    require(type(raw_text) is str, "E_PLAN_TEXT")
    try:
        raw = raw_text.encode("ascii")
    except UnicodeEncodeError as error:
        raise UsbLauncherError("E_PLAN_ASCII") from error
    require(0 < len(raw) <= MAX_PLAN_BYTES, "E_PLAN_SIZE")
    require(
        type(expected_sha256) is str
        and len(expected_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_sha256),
        "E_PLAN_DIGEST",
    )
    exact(hashlib.sha256(raw).hexdigest(), expected_sha256, "plan.sha256")
    try:
        value = json.loads(
            raw_text,
            object_pairs_hook=strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                UsbLauncherError(f"E_JSON_NUMBER: {item}")
            ),
        )
    except json.JSONDecodeError as error:
        raise UsbLauncherError("E_PLAN_JSON") from error
    require(type(value) is dict, "E_PLAN_TYPE")
    try:
        canonical = launcher.canonical_compact(value)
    except launcher.LaunchError as error:
        raise UsbLauncherError(str(error)) from error
    exact(canonical, raw, "plan.canonical")
    return value


def validate_usb_identity(value: dict[str, Any]) -> tuple[str, str]:
    exact(value.get("mode"), "android", "plan.mode")
    endpoint = value.get("endpoint")
    require(type(endpoint) is str and endpoint in ENDPOINT_SERIALS, "E_ENDPOINT")
    android = value.get("android")
    require(type(android) is dict, "E_ANDROID")
    exact(android.get("adb_port"), ADB_PORT, "android.adb_port")
    serial = ENDPOINT_SERIALS[endpoint]
    exact(android.get("physical_serial"), serial, "android.physical_serial")
    exact(android.get("adb_selector"), serial, "android.adb_selector")
    return endpoint, serial


def normalize_missing_build_ids(value: dict[str, Any]) -> None:
    components = value.get("components")
    if type(components) is not list:
        return
    for component in components:
        if type(component) is not dict:
            continue
        metadata = component.get("stat")
        if type(metadata) is dict and "build_id" not in metadata:
            metadata["build_id"] = None


def usb_adb_prefix(
    launcher: types.ModuleType,
    android: dict[str, Any],
) -> list[str]:
    exact(android.get("adb_port"), ADB_PORT, "android.adb_port")
    serial = android.get("physical_serial")
    require(serial in ENDPOINT_SERIALS.values(), "E_ANDROID_SERIAL")
    exact(android.get("adb_selector"), serial, "android.adb_selector")
    adb_path = android.get("adb_path")
    require(type(adb_path) is str and bool(adb_path), "E_ADB_PATH")
    return [
        adb_path,
        "-P",
        str(ADB_PORT),
        "-s",
        serial,
    ]


def install_usb_adb_prefix(launcher: types.ModuleType) -> None:
    launcher.adb_prefix = lambda android: usb_adb_prefix(launcher, android)


def validate_plan(
    raw_text: str,
    expected_sha256: str,
    launcher: types.ModuleType,
) -> dict[str, Any]:
    value = parse_original_plan(raw_text, expected_sha256, launcher)
    _endpoint, serial = validate_usb_identity(value)
    normalize_missing_build_ids(value)

    compatibility = copy.deepcopy(value)
    compatibility["android"]["adb_selector"] = f"{serial}:1"
    try:
        plan = launcher.validate_plan(compatibility)
    except launcher.LaunchError as error:
        raise UsbLauncherError(str(error)) from error
    plan["android"]["adb_selector"] = serial
    install_usb_adb_prefix(launcher)
    exact(
        launcher.adb_prefix(plan["android"]),
        [
            plan["android"]["adb_path"],
            "-P",
            str(ADB_PORT),
            "-s",
            serial,
        ],
        "android.argv_prefix",
    )
    return plan


def execute(
    raw_text: str,
    expected_sha256: str,
    boot_id: str,
    *,
    launcher: types.ModuleType | None = None,
    runner: Any | None = None,
) -> int:
    launcher = launcher or load_frozen_launcher()
    plan = validate_plan(raw_text, expected_sha256, launcher)
    runner = runner or launcher.SubprocessRunner()
    return launcher.launch_android(plan, runner, boot_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--boot-id", required=True)
    try:
        args = parser.parse_args(argv)
        return execute(args.plan_json, args.plan_sha256, args.boot_id)
    except (
        UsbLauncherError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"MANAGED_RUNTIME_USB_LAUNCH_REFUSED: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(
            f"MANAGED_RUNTIME_USB_LAUNCH_REFUSED: "
            f"E_UNEXPECTED_{type(error).__name__}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
