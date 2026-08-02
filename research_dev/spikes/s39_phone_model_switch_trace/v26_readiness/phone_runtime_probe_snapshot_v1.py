#!/usr/bin/python3 -I
"""Drive the frozen phone runtime probe with phase-fresh snapshotted identity.

The frozen V2.4 launch plans freeze the probe argv prospectively, but the
frozen v23 probe requires per-run values (`--pid`, `--start-ticks`,
`--network-pid`, `--network-start-ticks`, `--boot-id`) that cannot exist in a
prospective plan. This wrapper derives every run-time value live from the
phone named by the plan, builds the frozen probe's own plan record from the
frozen launcher's command constructor, executes the FROZEN probe in-process
with `capture_compatible`, and projects its row onto the exact V2.4
`s39-cp0-r1-v24-phone-runtime-probe-v1` key set (the frozen compatible row
lacks `thermal_status` and carries five extra `network_*` keys; the
downstream producer cross-checks the projected row against the bound plan,
the worker log, and the launcher's RUNTIMEPROCESS record).
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


HERE = Path(__file__).resolve().parent
S39_ROOT = HERE.parents[1]
USB_SOURCE = HERE / "managed_runtime_launcher_usb_v1.py"
USB_SHA256 = "52c4e1f251f4daa2c856857fcdf23fdc5a85c0d30eff93365996a8f1378611ce"
PROBE_SOURCE = (
    S39_ROOT
    / "v23_readiness"
    / "a_only_acquisition_driver_v1"
    / "producers_v1"
    / "phone_runtime_probe_v1.py"
)
PROBE_SHA256 = (
    "b46c7bde2f06cb4701a8ab85c18d07aa205d0d4958e58a37e7844c172758df55"
)
CAPTURE_SCHEMA_V24 = "s39-cp0-r1-v24-phone-runtime-probe-v1"
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
ROW_KEYS = {
    "active_sequences",
    "available_bytes",
    "boot_id",
    "device",
    "direct_peer",
    "gpu_max_millic",
    "interface",
    "loaded_shard_path",
    "loaded_shard_sha256",
    "model",
    "model_id",
    "model_sha256",
    "process_swap_bytes",
    "product",
    "schema",
    "serial",
    "system_swap_used_bytes",
    "thermal_status",
    "worker_executable_path",
    "worker_executable_sha256",
    "worker_pid",
    "worker_start_ticks",
}
MAX_PLAN_BYTES = 8 * 1024 * 1024


class SnapshotProbeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SnapshotProbeError(message)


def load_pinned_module(path: Path, expected_sha256: str) -> types.ModuleType:
    raw = path.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == expected_sha256,
        f"E_PINNED_SOURCE: {path.name}",
    )
    module = types.ModuleType(f"_s39_probe_snapshot_{path.stem}")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def parse_plan(raw_text: str, expected_sha256: str, field: str) -> dict[str, Any]:
    require(
        type(raw_text) is str and 0 < len(raw_text) <= MAX_PLAN_BYTES,
        f"E_PLAN_SIZE: {field}",
    )
    raw = raw_text.encode("ascii")
    require(
        hashlib.sha256(raw).hexdigest() == expected_sha256,
        f"E_PLAN_DIGEST: {field}",
    )
    value = json.loads(raw_text)
    require(type(value) is dict, f"E_PLAN_TYPE: {field}")
    return value


class Phone:
    def __init__(self, android: dict[str, Any]):
        self.adb_path = android["adb_path"]
        self.port = str(android["adb_port"])
        self.serial = android["physical_serial"]
        adb_raw = Path(self.adb_path).read_bytes()
        require(
            hashlib.sha256(adb_raw).hexdigest() == android["adb_sha256"],
            "E_ADB_BINARY",
        )

    def shell(self, command: str, timeout: int = 30) -> str:
        completed = subprocess.run(
            [
                self.adb_path,
                "-P",
                self.port,
                "-s",
                self.serial,
                "shell",
                command,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        require(completed.returncode == 0, f"E_ADB: {command[:40]}")
        return completed.stdout.decode("ascii", errors="strict").strip()


def start_ticks_of(phone: Phone, pid: int) -> int:
    raw = phone.shell(f"cat /proc/{pid}/stat")
    tail = raw.rsplit(")", 1)[1].split()
    require(len(tail) >= 20, "E_PROC_STAT")
    return int(tail[19])


def pid_of(phone: Phone, executable_path: str) -> int:
    name = Path(executable_path).name
    raw = phone.shell(f"pidof {name}")
    pids = raw.split()
    require(len(pids) == 1, f"E_PID_UNIQUE: {name}: {raw!r}")
    return int(pids[0])


def component_artifact(plan: dict[str, Any], component_id: str) -> dict[str, Any]:
    row = next(
        item
        for item in plan["components"]
        if item["component_id"] == component_id
    )
    return {
        "bytes": row["bytes"],
        "path": row["path"],
        "sha256": row["sha256"],
        "stat": dict(row["stat"]),
    }


def live_stat(phone: Phone, path: str) -> dict[str, int]:
    raw = phone.shell(
        "stat -c 'DEV=%d|INO=%i|SIZE=%s|MODE=%f|MTIME=%Y|CTIME=%Z' -- "
        + f"'{path}'"
    )
    values = dict(part.split("=", 1) for part in raw.split("|"))
    return {
        "ctime_ns": int(values["CTIME"]) * 1_000_000_000,
        "device_id": int(values["DEV"]),
        "inode": int(values["INO"]),
        "mode": int(values["MODE"], 16),
        "mtime_ns": int(values["MTIME"]) * 1_000_000_000,
        "size": int(values["SIZE"]),
    }


def thermal_status(phone: Phone) -> int:
    raw = phone.shell("dumpsys thermalservice")
    match = re.search(r"Thermal Status:\s*(\d+)", raw)
    require(match is not None, "E_THERMAL_STATUS")
    return int(match.group(1))


def wlan_ipv4(phone: Phone, interface: str) -> str:
    raw = phone.shell(
        f"ip -4 -o addr show {interface} | head -n 1"
        " | tr -s ' ' | cut -d ' ' -f 4"
    )
    require("/" in raw, "E_WLAN_IPV4")
    return raw.split("/")[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--when", required=True, choices=("after", "before"))
    parser.add_argument("--interface", required=True)
    parser.add_argument("--local-ipv4", required=True)
    parser.add_argument("--peer-ipv4", required=True)
    parser.add_argument("--local-port", required=True, type=int)
    parser.add_argument("--peer-port", required=True, type=int)
    parser.add_argument(
        "--network-role",
        required=True,
        choices=("direct_relay", "stagenet_worker"),
    )
    try:
        args = parser.parse_args(argv)
        usb = load_pinned_module(USB_SOURCE, USB_SHA256)
        frozen_launcher = usb.load_frozen_launcher()
        plan = usb.validate_plan(
            args.plan_json, args.plan_sha256, frozen_launcher
        )
        require(plan["route"]["kind"] == "stagenet_worker", "E_ROUTE_KIND")
        android = plan["android"]
        phone = Phone(android)

        boot_id = phone.shell("cat /proc/sys/kernel/random/boot_id")
        require(UUID_RE.fullmatch(boot_id) is not None, "E_BOOT_FORMAT")
        device = phone.shell("getprop ro.product.device")
        model = phone.shell("getprop ro.product.model")
        product = phone.shell("getprop ro.product.name")
        local_ipv4 = wlan_ipv4(phone, args.interface)
        require(local_ipv4 == args.local_ipv4, "E_LOCAL_IPV4_DRIFT")

        worker = component_artifact(plan, plan["launcher_component_id"])
        worker_argv, _worker_env, _root = frozen_launcher.build_runtime_command(
            worker["path"],
            plan["route"],
        )
        pid = pid_of(phone, worker["path"])
        ticks = start_ticks_of(phone, pid)
        if args.network_role == "direct_relay":
            network_pid = pid_of(phone, "llama-stage-direct-relay")
            network_ticks = start_ticks_of(phone, network_pid)
            relay_exe = phone.shell(f"readlink /proc/{network_pid}/exe")
            relay_cmdline = phone.shell(
                f"cat /proc/{network_pid}/cmdline | tr '\\0' '\\n'"
            ).splitlines()
            require(
                bool(relay_cmdline) and relay_cmdline[0] == relay_exe,
                "E_RELAY_CMDLINE",
            )
            relay_digest = phone.shell(
                f"sha256sum -- '{relay_exe}'"
            ).split()[0]
            relay_stat = live_stat(phone, relay_exe)
            network_process = {
                "argv": relay_cmdline,
                "artifact": {
                    "bytes": relay_stat["size"],
                    "path": relay_exe,
                    "sha256": relay_digest,
                    "stat": relay_stat,
                },
                "executable_path": relay_exe,
                "role": "direct_relay",
            }
        else:
            network_pid = pid
            network_ticks = ticks
            network_process = {
                "argv": list(worker_argv),
                "artifact": dict(worker),
                "executable_path": worker["path"],
                "role": "stagenet_worker",
            }

        shard_path = plan["route"]["model_path"]
        shard = {
            "bytes": None,
            "path": shard_path,
            "sha256": plan["route"]["model_sha256"],
            "stat": live_stat(phone, shard_path),
        }
        shard["bytes"] = shard["stat"]["size"]
        probe_plan = {
            "android": {
                "adb_path": android["adb_path"],
                "adb_port": android["adb_port"],
                "adb_selector": f"{android['physical_serial']}:1",
                "adb_sha256": android["adb_sha256"],
                "boot_id_source": "phase_fresh_snapshot",
                "device": device,
                "model": model,
                "physical_serial": android["physical_serial"],
                "product": product,
            },
            "capture_schema": CAPTURE_SCHEMA_V24,
            "model_id": "qwen3-14b-q4_k_m",
            "model_sha256": plan["route"]["model_sha256"],
            "network_process": network_process,
            "process": {
                "argv": worker_argv,
                "executable_path": worker["path"],
            },
            "schema": None,
            "shard_artifact": shard,
            "stage_v3": {
                "expected_active_sequences": 0,
                "source": "relay_owned_status",
            },
            "telemetry": {
                "direct_peer_ipv4": args.peer_ipv4,
                "direct_peer_local_port": args.local_port,
                "direct_peer_port": args.peer_port,
                "interface": args.interface,
                "local_ipv4": args.local_ipv4,
                "max_gpu_millic": 95000,
                "min_available_bytes": 536870912,
            },
            "worker_artifact": worker,
        }
        probe = load_pinned_module(PROBE_SOURCE, PROBE_SHA256)
        probe_plan["schema"] = probe.PLAN_SCHEMA
        probe.adb_prefix = lambda a: [
            android["adb_path"],
            "-P",
            str(android["adb_port"]),
            "-s",
            android["physical_serial"],
        ]
        raw_plan = json.dumps(
            probe_plan,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        validated = probe.parse_plan_json(
            raw_plan,
            hashlib.sha256(raw_plan.encode("ascii")).hexdigest(),
        )
        row = probe.run_probe(
            validated,
            probe.SubprocessRunner(),
            pid,
            ticks,
            network_pid,
            network_ticks,
            boot_id,
            True,
        )
        projected = {
            key: value
            for key, value in row.items()
            if not key.startswith("network_")
        }
        projected["thermal_status"] = thermal_status(phone)
        require(set(projected) == ROW_KEYS, "E_ROW_KEYS")
        require(projected["schema"] == CAPTURE_SCHEMA_V24, "E_ROW_SCHEMA")
        sys.stdout.buffer.write(
            (
                json.dumps(
                    projected,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("ascii")
        )
        return 0
    except (
        SnapshotProbeError,
        OSError,
        StopIteration,
        subprocess.SubprocessError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"SNAPSHOT_PROBE_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
