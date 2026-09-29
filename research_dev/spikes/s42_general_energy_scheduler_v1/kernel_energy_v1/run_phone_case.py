#!/usr/bin/env python3
"""Run one OP15 energy case while its on-device power logger stays active."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

import energy_common


SERIAL = "3C15AU002CL00000"
LOGGER = "/data/local/tmp/s41-opoffload-dmabuf-v1/phone_power_logger.sh"
POLICY = "/data/local/tmp/s41-opoffload-dmabuf-v1/phone_power_allow.rules"


def adb(port: int, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["adb", "-P", str(port), "-s", SERIAL, *args],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
        timeout=30,
        check=check,
    )


def wait_device(port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = adb(port, "get-state", check=False)
        if result.returncode == 0 and result.stdout.strip() == "device":
            return
        time.sleep(1)
    raise energy_common.EnergyError("phone did not return to ADB")


def adb_su(
    port: int, command: str, *, exec_out: bool = False, check: bool = True
) -> subprocess.CompletedProcess:
    mode = "exec-out" if exec_out else "shell"
    return adb(port, mode, f"su -c {shlex.quote(command)}", check=check)


def exists(port: int, path: str) -> bool:
    result = adb_su(port, f"test -f {path}", check=False)
    return result.returncode == 0


def phone_uptime_s(port: int) -> float:
    result = adb(port, "shell", "cat", "/proc/uptime")
    return float(result.stdout.split()[0])


def stop_stale_logger(port: int, samples: str) -> None:
    result = adb_su(port, "ps -A -o PID,ARGS", check=False)
    pids = []
    for line in result.stdout.splitlines():
        if LOGGER not in line or samples not in line:
            continue
        fields = line.split(None, 1)
        if fields and fields[0].isdigit():
            pids.append(fields[0])
    if pids:
        adb_su(port, "kill " + " ".join(pids), check=False)
        time.sleep(0.3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--mode", choices=("adb", "external"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adb-port", type=int, default=5037)
    parser.add_argument("--adb-command")
    parser.add_argument("--external-active")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    energy_common.require(args.output.is_absolute(), "absolute output")
    energy_common.require(not args.output.exists(), "output already exists")
    energy_common.require(
        re.fullmatch(r"[A-Za-z0-9_.-]+", args.case_id) is not None,
        "case id",
    )
    if args.mode == "adb":
        energy_common.require(bool(args.adb_command), "ADB command")
        energy_common.require(not args.command, "unexpected external command")
    else:
        energy_common.require(bool(args.external_active), "external active path")
        energy_common.require(bool(args.command), "external command")

    args.output.mkdir(parents=True)
    phone_root = f"/data/local/tmp/s42-kernel-energy-v1/{args.case_id}"
    samples = f"{phone_root}/samples.tsv"
    active = (
        f"{phone_root}/active" if args.mode == "adb" else args.external_active
    )
    armed = f"{phone_root}/power.armed"
    done = f"{phone_root}/power.done"
    logger_log = f"{phone_root}/logger.log"

    wait_device(args.adb_port, 30)
    stop_stale_logger(args.adb_port, samples)
    adb_su(
        args.adb_port,
        f"mkdir -p {phone_root}; rm -f {samples} {armed} {done} {logger_log}",
    )
    adb_su(
        args.adb_port,
        f"/product/bin/magiskpolicy --live --apply {POLICY}",
    )
    launch = (
        f"nohup sh {LOGGER} {samples} {active} {armed} {done} "
        f"120 > {logger_log} 2>&1 </dev/null &"
    )
    adb_su(args.adb_port, launch)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not exists(args.adb_port, armed):
        time.sleep(0.1)
    energy_common.require(exists(args.adb_port, armed), "phone logger arm")

    start_uptime_ns = None
    end_uptime_ns = None
    if args.mode == "adb":
        adb_su(args.adb_port, f"rm -f {active}; touch {active}")
        time.sleep(0.6)
        start_uptime_ns = int(round(phone_uptime_s(args.adb_port) * 1e9))
        completed = subprocess.run(
            [
                "adb", "-P", str(args.adb_port), "-s", SERIAL,
                "shell", f"su -c {shlex.quote(args.adb_command)}",
            ],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=args.timeout_s,
            check=False,
        )
        end_uptime_ns = int(round(phone_uptime_s(args.adb_port) * 1e9))
        time.sleep(0.6)
        adb_su(args.adb_port, f"rm -f {active}")
        workload_output = completed.stdout + completed.stderr
    else:
        completed = subprocess.run(
            args.command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout_s,
            check=False,
        )
        workload_output = completed.stdout
        wait_device(args.adb_port, 120)

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not exists(args.adb_port, done):
        time.sleep(0.2)
    energy_common.require(exists(args.adb_port, done), "phone logger completion")
    pulled = adb_su(args.adb_port, f"cat {samples}", exec_out=True)
    args.output.joinpath("phone-samples.tsv").write_text(
        pulled.stdout, encoding="ascii"
    )
    args.output.joinpath("workload.log").write_text(
        workload_output, encoding="ascii"
    )
    metadata = {
        "adb_port": args.adb_port,
        "case_id": args.case_id,
        "command": args.adb_command if args.mode == "adb" else args.command,
        "end_phone_uptime_ns": end_uptime_ns,
        "mode": args.mode,
        "returncode": completed.returncode,
        "schema": "s42-phone-energy-capture-v1",
        "serial": SERIAL,
        "start_phone_uptime_ns": start_uptime_ns,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }
    args.output.joinpath("capture.json").write_bytes(
        energy_common.canonical(metadata)
    )
    return 0 if completed.returncode == 0 else completed.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except energy_common.EnergyError as error:
        print(f"S42_PHONE_CASE_ERROR: {error}")
        raise SystemExit(2)
