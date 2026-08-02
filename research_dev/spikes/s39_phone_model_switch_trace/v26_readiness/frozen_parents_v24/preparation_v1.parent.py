#!/usr/bin/env python3
"""Capture the V2.4 reboot preparation on the bound CUDA host and phones."""

from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
import stat
import time
import types
from typing import Any, Callable


def _load_source(name: str, path: Path):
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"E_SOURCE_REGULAR: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise RuntimeError(f"E_SOURCE_CHANGED: {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(bytes(raw), str(path), "exec"), module.__dict__)
    return module


common = _load_source(
    "s39_v24_production_common",
    Path(__file__).resolve().with_name("production_common_v1.py"),
)


CONFIRMATION = "RUN_V24_REBOOT_PREPARATION_A_ONLY"
CUDA_STATUS = """\
set -eu
printf 'HOST='; hostname
printf 'BOOT_ID='; cat /proc/sys/kernel/random/boot_id
awk '/^SwapTotal:/{total=$2}/^SwapFree:/{free=$2}END{print "SWAP_USED_KB=" total-free}' /proc/meminfo
nvidia-smi --query-gpu=uuid,pci.bus_id --format=csv,noheader,nounits | awk -F', ' 'NR==1{print "GPU_UUID="$1; print "PCI_BUS_ID="$2}END{if(NR!=1)exit 41}'
"""
PHONE_STATUS = """\
set -eu
printf 'BOOT_ID='; cat /proc/sys/kernel/random/boot_id
printf 'BOOT_COMPLETED='; getprop sys.boot_completed
printf 'PRODUCT='; getprop ro.product.name
printf 'MODEL='; getprop ro.product.model
printf 'DEVICE='; getprop ro.product.device
printf 'INTERFACE=wlan0\n'
ip -4 -o addr show dev wlan0 scope global | awk 'NR==1{split($4,a,"/");print "LOCAL_IPV4="a[1]}END{if(NR!=1)exit 43}'
awk '/^MemAvailable:/{print "MEM_AVAILABLE_KB="$2}' /proc/meminfo
awk '/^SwapTotal:/{total=$2}/^SwapFree:/{free=$2}END{print "SWAP_USED_KB=" total-free}' /proc/meminfo
dumpsys thermalservice | awk -F: '/Thermal Status:/{gsub(/[[:space:]]/,"",$2); print "THERMAL_STATUS="$2; found=1; exit}END{if(!found)exit 42}'
"""


def _positive(value: str, field: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise common.ProductionError(f"E_INTEGER: {field}") from error
    return common.integer(result, field)


def _cuda_snapshot(
    runner: common.Runner,
    contract: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    raw = common.run_probe(
        runner,
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            common.CUDA_SSH_TARGET,
            "sh",
            "-c",
            CUDA_STATUS,
        ],
        timeout,
        "cuda.status",
    )
    values = common.parse_assignments(raw, "cuda.status")
    common.exact(
        set(values),
        {"BOOT_ID", "GPU_UUID", "HOST", "PCI_BUS_ID", "SWAP_USED_KB"},
        "cuda.status.keys",
    )
    expected = contract["devices"]["cuda"]
    common.exact(values["HOST"], expected["host"], "cuda.host")
    common.exact(values["GPU_UUID"], expected["uuid"], "cuda.uuid")
    common.require(common.UUID_RE.fullmatch(values["BOOT_ID"]) is not None, "E_CUDA_BOOT")
    swap = _positive(values["SWAP_USED_KB"], "cuda.swap") * 1024
    return {
        "gpu_uuid": values["GPU_UUID"],
        "host": values["HOST"],
        "host_boot_id": values["BOOT_ID"],
        "pci_bus_id": values["PCI_BUS_ID"],
        "system_swap_used_bytes": swap,
    }


def _adb(serial: str, *argv: str) -> list[str]:
    return [
        "adb",
        "-P",
        str(common.PHONE_ADB_PORT),
        "-s",
        serial,
        *argv,
    ]


def _phone_snapshot(
    runner: common.Runner,
    contract: dict[str, Any],
    phone: str,
    timeout: float,
) -> tuple[dict[str, Any], bool]:
    expected = contract["devices"][phone]
    raw = common.run_probe(
        runner,
        _adb(expected["serial"], "shell", PHONE_STATUS),
        timeout,
        f"{phone}.status",
    )
    values = common.parse_assignments(raw, f"{phone}.status")
    common.exact(
        set(values),
        {
            "BOOT_COMPLETED",
            "BOOT_ID",
            "DEVICE",
            "INTERFACE",
            "LOCAL_IPV4",
            "MEM_AVAILABLE_KB",
            "MODEL",
            "PRODUCT",
            "SWAP_USED_KB",
            "THERMAL_STATUS",
        },
        f"{phone}.status.keys",
    )
    common.require(common.UUID_RE.fullmatch(values["BOOT_ID"]) is not None, f"E_BOOT: {phone}")
    for source, key in (("DEVICE", "device"), ("MODEL", "model"), ("PRODUCT", "product")):
        common.exact(values[source], expected[key], f"{phone}.{key}")
    common.exact(values["INTERFACE"], "wlan0", f"{phone}.interface")
    try:
        address = ipaddress.IPv4Address(values["LOCAL_IPV4"])
    except ipaddress.AddressValueError as error:
        raise common.ProductionError(f"E_PHONE_IPV4: {phone}") from error
    common.require(
        not (
            address.is_unspecified
            or address.is_loopback
            or address.is_multicast
        ),
        f"E_PHONE_IPV4: {phone}",
    )
    available = _positive(values["MEM_AVAILABLE_KB"], f"{phone}.available") * 1024
    swap = _positive(values["SWAP_USED_KB"], f"{phone}.swap") * 1024
    thermal = _positive(values["THERMAL_STATUS"], f"{phone}.thermal")
    ready = (
        values["BOOT_COMPLETED"] == "1"
        and available >= contract["gates"]["phone_minimum_available_bytes"]
        and thermal == 0
    )
    return {
        "available_bytes": available,
        "boot_id": values["BOOT_ID"],
        "device": expected["device"],
        "interface": values["INTERFACE"],
        "local_ipv4": str(address),
        "model": expected["model"],
        "product": expected["product"],
        "serial": expected["serial"],
        "system_swap_used_bytes": swap,
        "thermal_status": thermal,
    }, ready


def prepare(
    *,
    output: Path,
    contract_path: Path,
    artifact_root_path: Path,
    runtime_plan_path: Path,
    confirmation: str,
    timeout_seconds: int,
    runner: common.Runner | None = None,
    clock_ns: Callable[[], int] = common.monotonic_ns,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = 1.0,
) -> dict[str, Any]:
    common.exact(confirmation, CONFIRMATION, "confirmation")
    common.require(type(timeout_seconds) is int and 60 <= timeout_seconds <= 1800, "E_TIMEOUT")
    common.require(
        type(poll_seconds) in (int, float) and 0 < poll_seconds <= 10,
        "E_POLL_SECONDS",
    )
    common.require(not output.exists(), "E_OUTPUT_EXISTS")
    contract, _ = common.read_canonical(contract_path, "contract")
    root, root_raw = common.read_canonical(artifact_root_path, "artifact_root")
    plan, plan_raw = common.read_canonical(runtime_plan_path, "runtime_plan")
    common.exact(contract.get("schema"), "s39-cp0-r1-evidence-contract-v2.4", "contract.schema")
    common.exact(root.get("schema"), "s39-cp0-r1-artifact-root-v2.4", "root.schema")
    common.exact(plan.get("schema"), "s39-cp0-r1-runtime-bundle-plan-v2.4", "plan.schema")
    common.exact(root.get("runtime_bundle_plan_sha256"), common.sha256_bytes(plan_raw), "root.plan")
    runner = runner or common.SubprocessRunner()
    deadline = monotonic() + timeout_seconds
    started_ns = clock_ns()
    common.require(root["completed_ns"] <= started_ns, "E_PREPARATION_BEFORE_ROOT")
    before_boot = {}
    for phone in ("op15", "op12"):
        serial = contract["devices"][phone]["serial"]
        raw = common.run_probe(
            runner,
            _adb(serial, "shell", "cat /proc/sys/kernel/random/boot_id"),
            timeout_seconds,
            f"{phone}.before",
        )
        value = raw.decode("ascii").strip()
        common.require(common.UUID_RE.fullmatch(value) is not None, f"E_BEFORE_BOOT: {phone}")
        before_boot[phone] = value
    reboot_started_ns = clock_ns()
    for phone in ("op15", "op12"):
        serial = contract["devices"][phone]["serial"]
        common.run_probe(
            runner,
            _adb(serial, "reboot"),
            timeout_seconds,
            f"{phone}.reboot",
            empty_stdout=True,
        )
    for phone in ("op15", "op12"):
        serial = contract["devices"][phone]["serial"]
        common.run_probe(
            runner,
            _adb(serial, "wait-for-device"),
            timeout_seconds,
            f"{phone}.wait",
            empty_stdout=True,
        )
    devices = {"cuda": _cuda_snapshot(runner, contract, timeout_seconds)}
    for phone in ("op15", "op12"):
        while True:
            common.require(monotonic() < deadline, f"E_PHONE_BOOT_TIMEOUT: {phone}")
            try:
                snapshot, ready = _phone_snapshot(
                    runner,
                    contract,
                    phone,
                    max(1.0, deadline - monotonic()),
                )
            except common.ProductionError:
                sleep(float(poll_seconds))
                continue
            if ready and snapshot["boot_id"] != before_boot[phone]:
                devices[phone] = snapshot
                break
            sleep(float(poll_seconds))
    completed_ns = clock_ns()
    common.require(started_ns <= reboot_started_ns < completed_ns, "E_PREPARATION_INTERVAL")
    result = {
        "artifact_root_sha256": common.sha256_bytes(root_raw),
        "before_boot_ids": before_boot,
        "completed_ns": completed_ns,
        "devices": devices,
        "reboot_started_ns": reboot_started_ns,
        "runtime_bundle_plan_sha256": common.sha256_bytes(plan_raw),
        "schema": "s39-cp0-r1-reboot-preparation-v2.4",
        "started_ns": started_ns,
    }
    common.write_new(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-plan", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        prepare(
            output=args.output,
            contract_path=args.contract,
            artifact_root_path=args.root,
            runtime_plan_path=args.runtime_plan,
            confirmation=args.confirm,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        return 0
    except (OSError, ValueError, common.ProductionError) as error:
        print(f"V24_PREPARATION_REFUSED: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
