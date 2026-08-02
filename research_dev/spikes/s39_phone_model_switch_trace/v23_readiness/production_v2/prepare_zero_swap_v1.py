#!/usr/bin/python3 -I
"""Reboot both phones and establish an idle zero-swap pre-phase state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Protocol


sys.dont_write_bytecode = True

CONFIRM = "S39_REBOOT_BOTH_PHONES_FOR_ZERO_SWAP"
UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


class PrepareError(ValueError):
    pass


class Runner(Protocol):
    def run(
        self,
        argv: list[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        ...


class SubprocessRunner:
    def run(
        self,
        argv: list[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            argv,
            capture_output=True,
            check=False,
            timeout=timeout,
        )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PrepareError(message)


def strict_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, f"E_DUPLICATE_KEY: {key}")
        value[key] = item
    return value


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def secure_read(path: Path) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "E_CONTRACT_REGULAR")
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
        value.st_mode,
    )
    require(identity(before) == identity(after), "E_CONTRACT_CHANGED")
    require(len(raw) == before.st_size, "E_CONTRACT_SIZE")
    return bytes(raw)


def load_contract(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = secure_read(path)
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                PrepareError(f"E_JSON_NUMBER: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrepareError(f"E_CONTRACT_JSON: {error}") from error
    require(type(value) is dict and canonical_bytes(value) == raw, "E_CONTRACT_CANONICAL")
    require(
        value.get("schema") == "s39-cp0-r1-evidence-contract-v2.3",
        "E_CONTRACT_SCHEMA",
    )
    return value, raw


def run_probe(
    runner: Runner,
    argv: list[str],
    timeout: float,
    label: str,
    *,
    require_empty: bool = False,
) -> bytes:
    try:
        result = runner.run(argv, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise PrepareError(f"E_TIMEOUT: {label}") from error
    require(type(result.returncode) is int and result.returncode == 0, f"E_EXIT: {label}")
    require(result.stderr == b"", f"E_STDERR: {label}")
    require(type(result.stdout) is bytes, f"E_STDOUT: {label}")
    if require_empty:
        require(result.stdout == b"", f"E_STDOUT_NOT_EMPTY: {label}")
    return result.stdout


def adb(port: int, serial: str, *args: str) -> list[str]:
    return ["adb", "-P", str(port), "-s", serial, *args]


def status_command() -> str:
    return """\
set -eu
printf 'BOOT_ID='; cat /proc/sys/kernel/random/boot_id
printf 'BOOT_COMPLETED='; getprop sys.boot_completed
awk '/^MemAvailable:/{print "MEM_AVAILABLE_KB="$2}' /proc/meminfo
awk '/^SwapTotal:/{print "SWAP_TOTAL_KB="$2}' /proc/meminfo
awk '/^SwapFree:/{print "SWAP_FREE_KB="$2}' /proc/meminfo
dumpsys thermalservice | awk -F: '/Thermal Status:/{gsub(/[[:space:]]/,"",$2); print "THERMAL_STATUS="$2; found=1; exit} END{if(!found) exit 42}'
"""


def parse_status(raw: bytes, serial: str) -> dict[str, Any]:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise PrepareError(f"E_STATUS_ASCII: {serial}") from error
    values = {}
    for line in lines:
        key, separator, value = line.partition("=")
        require(separator == "=" and key and key not in values, f"E_STATUS_FIELD: {serial}")
        values[key] = value
    require(
        set(values)
        == {
            "BOOT_COMPLETED",
            "BOOT_ID",
            "MEM_AVAILABLE_KB",
            "SWAP_FREE_KB",
            "SWAP_TOTAL_KB",
            "THERMAL_STATUS",
        },
        f"E_STATUS_KEYS: {serial}",
    )
    require(UUID_RE.fullmatch(values["BOOT_ID"]) is not None, f"E_BOOT_ID: {serial}")
    try:
        available = int(values["MEM_AVAILABLE_KB"]) * 1024
        swap_total = int(values["SWAP_TOTAL_KB"]) * 1024
        swap_free = int(values["SWAP_FREE_KB"]) * 1024
        thermal = int(values["THERMAL_STATUS"])
    except ValueError as error:
        raise PrepareError(f"E_STATUS_INTEGER: {serial}") from error
    require(0 <= swap_free <= swap_total, f"E_SWAP_RANGE: {serial}")
    return {
        "available_bytes": available,
        "boot_completed": values["BOOT_COMPLETED"],
        "boot_id": values["BOOT_ID"],
        "swap_used_bytes": swap_total - swap_free,
        "thermal_status": thermal,
    }


def durable_write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o644,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def prepare(
    *,
    contract_path: Path,
    output_dir: Path,
    confirmation: str,
    timeout_seconds: int,
    poll_seconds: float,
    stable_samples: int,
    runner: Runner | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    clock_ns: Callable[[], int] = lambda: time.clock_gettime_ns(
        time.CLOCK_MONOTONIC_RAW
    ),
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    require(confirmation == CONFIRM, "E_CONFIRM")
    require(type(timeout_seconds) is int and 30 <= timeout_seconds <= 1800, "E_TIMEOUT_RANGE")
    require(type(poll_seconds) in (int, float) and 0 < poll_seconds <= 10, "E_POLL_RANGE")
    require(type(stable_samples) is int and 2 <= stable_samples <= 10, "E_STABLE_RANGE")
    require(output_dir.is_absolute() and not output_dir.exists(), "E_OUTPUT")
    contract, contract_raw = load_contract(contract_path)
    runner = runner or SubprocessRunner()
    deadline = monotonic() + timeout_seconds
    started_ns = clock_ns()
    output_dir.mkdir(parents=True, exist_ok=False)
    phones = {}
    port = contract["preflight"]["phone_adb_port"]
    minimum = contract["readiness_v2_3"]["phone_minimum_available_bytes"]
    for phone in ("op15", "op12"):
        serial = contract["devices"][phone]["serial"]
        remaining = max(1.0, deadline - monotonic())
        before_raw = run_probe(
            runner,
            adb(port, serial, "shell", "cat /proc/sys/kernel/random/boot_id"),
            remaining,
            f"{phone}.before_boot",
        )
        before_boot = before_raw.decode("ascii").strip()
        require(UUID_RE.fullmatch(before_boot) is not None, f"E_BEFORE_BOOT: {phone}")
        run_probe(
            runner,
            adb(port, serial, "reboot"),
            max(1.0, deadline - monotonic()),
            f"{phone}.reboot",
            require_empty=True,
        )
        run_probe(
            runner,
            adb(port, serial, "wait-for-device"),
            max(1.0, deadline - monotonic()),
            f"{phone}.wait",
            require_empty=True,
        )
        stable = 0
        samples = []
        after_boot = None
        while stable < stable_samples:
            require(monotonic() < deadline, f"E_PREP_DEADLINE: {phone}")
            raw = run_probe(
                runner,
                adb(port, serial, "shell", status_command()),
                max(1.0, deadline - monotonic()),
                f"{phone}.status",
            )
            value = parse_status(raw, serial)
            value["observed_ns"] = clock_ns()
            samples.append(value)
            good = (
                value["boot_completed"] == "1"
                and value["boot_id"] != before_boot
                and value["available_bytes"] >= minimum
                and value["swap_used_bytes"] == 0
                and value["thermal_status"] == 0
            )
            if good and (after_boot is None or value["boot_id"] == after_boot):
                after_boot = value["boot_id"]
                stable += 1
            else:
                after_boot = value["boot_id"] if good else None
                stable = 1 if good else 0
            if stable < stable_samples:
                sleep(float(poll_seconds))
        phones[phone] = {
            "after_boot_id": after_boot,
            "before_boot_id": before_boot,
            "samples": samples[-stable_samples:],
            "serial": serial,
            "stable_samples": stable_samples,
        }
    completed_ns = clock_ns()
    require(started_ns < completed_ns, "E_PREP_INTERVAL")
    result = {
        "completed_ns": completed_ns,
        "contract_sha256": __import__("hashlib").sha256(contract_raw).hexdigest(),
        "phones": phones,
        "schema": "s39-cp0-r1-zero-swap-prepare-v1",
        "started_ns": started_ns,
        "status": "ZERO_SWAP_IDLE_PREP_PASS",
    }
    durable_write_new(output_dir / "zero_swap_prepare.json", canonical_bytes(result))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--stable-samples", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        value = prepare(
            contract_path=args.contract,
            output_dir=args.output,
            confirmation=args.confirm,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
            stable_samples=args.stable_samples,
        )
        print(
            "ZERO_SWAP_IDLE_PREP_PASS "
            + __import__("hashlib").sha256(canonical_bytes(value)).hexdigest()
        )
        return 0
    except Exception as error:
        print(f"ZERO_SWAP_IDLE_PREP_REFUSED: {type(error).__name__}: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
