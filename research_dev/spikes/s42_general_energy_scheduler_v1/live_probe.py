#!/usr/bin/env python3
"""Read-only live identity probe for the first S42 physical binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "s42-live-probe-v1"


class ProbeError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def remote(host: str, command: Sequence[str], allow_failure: bool = False) -> str:
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, *command],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="ascii",
    )
    if completed.returncode != 0 and not allow_failure:
        raise ProbeError(
            f"remote command failed ({' '.join(command)}): {completed.stderr.strip()}"
        )
    return completed.stdout


def parse_gpu(line: str) -> dict[str, object]:
    fields = [field.strip() for field in line.strip().split(",")]
    if len(fields) != 7:
        raise ProbeError("unexpected nvidia-smi response")
    number = re.compile(r"^([0-9]+(?:\.[0-9]+)?)")

    def numeric(field: str) -> float:
        match = number.match(field)
        if match is None:
            raise ProbeError("invalid nvidia-smi numeric field")
        return float(match.group(1))

    return {
        "name": fields[0],
        "uuid": fields[1],
        "driver_version": fields[2],
        "pstate": fields[3],
        "power_w": numeric(fields[4]),
        "utilization_pct": int(numeric(fields[5])),
        "memory_used_mib": int(numeric(fields[6])),
    }


def parse_adb_device(output: str, serial: str) -> dict[str, object]:
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0] == serial:
            state = fields[1] if len(fields) > 1 else "unknown"
            return {"serial": serial, "state": state, "line": line}
    return {"serial": serial, "state": "missing", "line": None}


def phone_usb_speed_mbps(lsusb: str, tree: str, vendor_product: str) -> int | None:
    identity = re.search(
        rf"Bus\s+(\d+)\s+Device\s+(\d+):\s+ID\s+{re.escape(vendor_product)}",
        lsusb,
    )
    if identity is None:
        return None
    bus = int(identity.group(1))
    device = int(identity.group(2))
    current_bus: int | None = None
    for line in tree.splitlines():
        root = re.search(r"Bus\s+(\d+)\.Port", line)
        if root is not None:
            current_bus = int(root.group(1))
        child = re.search(r"Dev\s+(\d+).*,\s+([0-9]+)M(?:/|$)", line)
        if current_bus == bus and child is not None and int(child.group(1)) == device:
            return int(child.group(2))
    return None


def adb_server_ports(ss_output: str) -> list[int]:
    ports = {
        int(match.group(1))
        for line in ss_output.splitlines()
        if '"adb"' in line
        for match in [re.search(r"127\.0\.0\.1:([0-9]+)", line)]
        if match is not None
    }
    return sorted(ports)


def find_phone_adb_port(host: str, serial: str, requested_port: int) -> tuple[int | None, str]:
    if requested_port:
        ports = [requested_port]
    else:
        ports = adb_server_ports(remote(host, ["ss", "-ltnp"]))
    last_output = ""
    for port in ports:
        output = remote(
            host, ["adb", "-P", str(port), "devices", "-l"], allow_failure=True
        )
        last_output = output
        if parse_adb_device(output, serial)["state"] != "missing":
            return port, output
    return None, last_output


def probe(
    host: str,
    adb_port: int,
    expected_gpu_uuid: str,
    expected_phone_serial: str,
) -> dict[str, Any]:
    hostname = remote(host, ["hostname"]).strip()
    gpu = parse_gpu(remote(host, [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,pstate,power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]))
    cpu_model = remote(
        host, ["lscpu", "-B"], allow_failure=False
    )
    model_line = next(
        (line.split(":", 1)[1].strip() for line in cpu_model.splitlines() if line.startswith("Model name:")),
        "unknown",
    )
    lsusb = remote(host, ["lsusb"])
    usb_tree = remote(host, ["lsusb", "-t"])
    selected_adb_port, adb_output = find_phone_adb_port(
        host, expected_phone_serial, adb_port
    )
    phone = parse_adb_device(adb_output, expected_phone_serial)
    if phone["state"] == "device":
        phone["model"] = remote(host, [
            "adb", "-P", str(selected_adb_port), "-s", expected_phone_serial,
            "shell", "getprop", "ro.product.model",
        ], allow_failure=True).strip()
        phone["soc"] = remote(host, [
            "adb", "-P", str(selected_adb_port), "-s", expected_phone_serial,
            "shell", "getprop", "ro.soc.model",
        ], allow_failure=True).strip()
    speed = phone_usb_speed_mbps(lsusb, usb_tree, "22d9:2772")
    process_rows = remote(host, ["ps", "-eo", "pid,args"]).splitlines()
    processes = [
        line for line in process_rows
        if any(
            name in line
            for name in (
                "llama-server",
                "llama-layersplit",
                "phone-worker",
                "ffn_worker",
            )
        )
    ]
    identity_ready = (
        gpu["uuid"] == expected_gpu_uuid
        and phone["state"] == "device"
        and speed == 5000
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "host": host,
        "hostname": hostname,
        "desktop": {
            "gpu": gpu,
            "gpu_identity_ok": gpu["uuid"] == expected_gpu_uuid,
            "cpu_model": model_line,
        },
        "phone": {
            **phone,
            "adb_port": selected_adb_port,
            "usb_vendor_product": "22d9:2772",
            "usb_speed_mbps": speed,
            "usb_5gbps_ok": speed == 5000,
        },
        "active_inference_processes": processes,
        "identity_ready": identity_ready,
        "execution_route_ready": False,
        "execution_route_reason": (
            "IDENTITY_READY_REQUIRES_WORKER_RESIDENCY_AND_RUNTIME_LEASE"
            if identity_ready
            else "IDENTITY_OR_TRANSPORT_NOT_READY"
        ),
        "scheduler_resource_ready": {
            "cuda0": gpu["uuid"] == expected_gpu_uuid,
            "cpu-cold": True,
            "op15-usb": speed == 5000,
            "op15-htp": False,
        },
        "mutation_scope": "READ_ONLY_NO_MODEL_OR_WORKER_LAUNCHED",
    }
    result["probe_hash"] = "sha256:" + hashlib.sha256(
        canonical_bytes(result)
    ).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="zhihao@172.20.74.85")
    parser.add_argument(
        "--adb-port", type=int, default=0,
        help="existing ADB server port; 0 discovers already-running ADB servers",
    )
    parser.add_argument(
        "--expected-gpu-uuid",
        default="GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
    )
    parser.add_argument("--expected-phone-serial", default="3C15AU002CL00000")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.adb_port < 0 or args.adb_port > 65535:
        parser.error("adb port is invalid")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        result = probe(
            args.host,
            args.adb_port,
            args.expected_gpu_uuid,
            args.expected_phone_serial,
        )
        payload = canonical_bytes(result)
        if args.output is None:
            print(payload.decode("ascii"), end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(payload)
            print(json.dumps({
                "output": str(args.output),
                "probe_hash": result["probe_hash"],
            }, sort_keys=True, separators=(",", ":")))
    except ProbeError as exc:
        parser.exit(2, f"probe failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
