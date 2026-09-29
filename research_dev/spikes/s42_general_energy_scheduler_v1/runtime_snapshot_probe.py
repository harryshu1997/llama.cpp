#!/usr/bin/env python3
"""Build a fail-closed read-only runtime snapshot for the 4060 Ti plus OP15."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live_probe import (
    find_phone_adb_port,
    parse_adb_device,
    parse_gpu,
    phone_usb_speed_mbps,
    remote,
)  # noqa: E402
from research_dev.scheduler import (  # noqa: E402
    RuntimeGateError,
    RuntimeSnapshot,
)


GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
PHONE_SERIAL = "3C15AU002CL00000"
COLD_MODEL_SHA256 = "494518c2262a26e2a607af0e40bca11c4de5a0b108e7c21308906dbbfb1c6f8c"
HOT_MODEL_SHA256 = "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"
BRIDGE_SHA256 = "1c5bfb27cd263238cf2e364ee5bdb538d7cd78fc82e55dd5a6228d143f1809d7"
WORKER_SHA256 = "a7b233bf466a184205c42d90e7fb189065165b2c555bc111f5c71d0ea47f9f00"


class SnapshotProbeError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def parse_thermal_service(output: str) -> dict[str, object]:
    status_match = re.search(r"^Thermal Status:\s*([0-9]+)\s*$", output, re.MULTILINE)
    if status_match is None:
        return {"status": None, "bucket": "unqualified", "max_current_millic": None}
    current = output.split("Current temperatures from HAL:", 1)
    temperatures: list[int] = []
    if len(current) == 2:
        for match in re.finditer(
            r"Temperature\{mValue=([0-9]+(?:\.[0-9]+)?),.*?mStatus=([0-9]+)\}",
            current[1],
        ):
            value = float(match.group(1))
            sensor_status = int(match.group(2))
            if sensor_status == 0 and value > 0:
                temperatures.append(round(value * 1000))
    status = int(status_match.group(1))
    return {
        "status": status,
        "bucket": "android-none" if status == 0 else f"android-status-{status}",
        "max_current_millic": max(temperatures) if temperatures else None,
    }


def parse_nvidia_thermal(output: str) -> str:
    thermal_rows = [
        line
        for line in output.splitlines()
        if "Thermal Slowdown" in line
        and re.search(r":\s+(?:Not )?Active\s*$", line) is not None
    ]
    if len(thermal_rows) != 2:
        return "unqualified"
    return (
        "not-throttled"
        if all(line.rstrip().endswith("Not Active") for line in thermal_rows)
        else "throttled"
    )


def parse_sha256sum(output: str) -> str | None:
    fields = output.split()
    if not fields:
        return None
    digest = fields[0]
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        return None
    return digest


def process_rows(output: str, pattern: str) -> list[str]:
    return [line for line in output.splitlines() if pattern in line]


def resource_state(
    *,
    ready: bool,
    generation: int,
    temperature_millic: int | None,
    thermal_bucket: str,
    contention_bucket: str,
    qualified_contention: bool,
    residency_ids: list[str],
    heartbeat_age_us: int | None = None,
    reset_generation: int = 0,
) -> dict[str, object]:
    return {
        "circuit_open": not ready,
        "contention_bucket": contention_bucket,
        "failure_count": 0,
        "generation": generation,
        "heartbeat_age_us": heartbeat_age_us,
        "ready": ready,
        "reset_generation": reset_generation,
        "residency_ids": residency_ids,
        "slowdown_ppm": 1000000 if qualified_contention else 2000000,
        "temperature_millic": temperature_millic,
        "thermal_bucket": thermal_bucket,
    }


def probe(
    *,
    host: str,
    adb_port: int,
    epoch_key: str,
    generation: int,
    reset_generation: int | None,
    validity_us: int,
) -> dict[str, Any]:
    gpu = parse_gpu(remote(host, [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,pstate,power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]))
    gpu_temperature = int(remote(host, [
        "nvidia-smi",
        "--query-gpu=temperature.gpu",
        "--format=csv,noheader,nounits",
    ]).strip()) * 1000
    gpu_thermal = parse_nvidia_thermal(
        remote(host, ["nvidia-smi", "-q", "-d", "PERFORMANCE"])
    )

    throttle_path = "/sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count"
    throttle_before = remote(host, ["cat", throttle_path], allow_failure=True).strip()
    lsusb = remote(host, ["lsusb"])
    usb_tree = remote(host, ["lsusb", "-t"])
    selected_adb_port, adb_output = find_phone_adb_port(host, PHONE_SERIAL, adb_port)
    phone = parse_adb_device(adb_output, PHONE_SERIAL)
    desktop_processes = remote(host, ["ps", "-eo", "pid,args"])
    bridge_hash = parse_sha256sum(remote(
        host,
        ["sha256sum", "/home/zhihao/s41-dynamic-ffn-v1/ffn_dmabuf_bridge-reset-qualified"],
        allow_failure=True,
    ))

    phone_thermal = {
        "status": None,
        "bucket": "unqualified",
        "max_current_millic": None,
    }
    phone_processes = ""
    worker_hash: str | None = None
    if phone.get("state") == "device" and selected_adb_port is not None:
        adb = ["adb", "-P", str(selected_adb_port), "-s", PHONE_SERIAL, "shell"]
        phone_thermal = parse_thermal_service(
            remote(host, [*adb, "dumpsys", "thermalservice"], allow_failure=True)
        )
        phone_processes = remote(
            host,
            [*adb, "ps", "-A", "-o", "PID,ARGS"],
            allow_failure=True,
        )
        worker_hash = parse_sha256sum(remote(
            host,
            [
                *adb,
                "sha256sum",
                "/data/local/tmp/s41-opoffload-dmabuf-v1/llama-ffn-split-worker-flex-v2",
            ],
            allow_failure=True,
        ))
    throttle_after = remote(host, ["cat", throttle_path], allow_failure=True).strip()
    cpu_not_throttled = (
        throttle_before.isdigit()
        and throttle_after.isdigit()
        and throttle_before == throttle_after
    )

    hot_rows = process_rows(desktop_processes, "Qwen3-14B-Q4_K_M.gguf")
    cold_rows = process_rows(
        desktop_processes, "gemma-4-12B-it-Q4_0-op15-exact.gguf"
    )
    bridge_rows = process_rows(
        desktop_processes, "ffn_dmabuf_bridge-reset-qualified"
    )
    worker_rows = process_rows(phone_processes, "llama-ffn-split-worker-flex-v2")
    inference_rows = [
        line
        for line in desktop_processes.splitlines()
        if "llama-server" in line or "ffn_dmabuf_bridge" in line
    ]
    exact_desktop_topology = (
        len(hot_rows) == 1
        and len(cold_rows) == 1
        and len(bridge_rows) == 1
        and len(inference_rows) == 3
    )
    exact_worker = len(worker_rows) == 1 and worker_hash == WORKER_SHA256
    exact_bridge = len(bridge_rows) == 1 and bridge_hash == BRIDGE_SHA256
    usb_speed = phone_usb_speed_mbps(lsusb, usb_tree, "22d9:2772")
    reset = 1 if reset_generation is None else reset_generation

    resources = {
        "cpu-cold": resource_state(
            ready=True,
            generation=generation,
            temperature_millic=None,
            thermal_bucket="not-throttled" if cpu_not_throttled else "unqualified",
            contention_bucket=(
                "i3-hot-qwen-default-affinity"
                if exact_desktop_topology
                else "unqualified"
            ),
            qualified_contention=exact_desktop_topology,
            residency_ids=(
                [f"model:{COLD_MODEL_SHA256}"] if len(cold_rows) == 1 else []
            ),
        ),
        "cuda0": resource_state(
            ready=gpu.get("uuid") == GPU_UUID,
            generation=generation,
            temperature_millic=gpu_temperature,
            thermal_bucket=gpu_thermal,
            contention_bucket="i3-hot-qwen" if exact_desktop_topology else "unqualified",
            qualified_contention=exact_desktop_topology,
            residency_ids=(
                [f"model:{HOT_MODEL_SHA256}"] if len(hot_rows) == 1 else []
            ),
        ),
        "op15-htp": resource_state(
            ready=bool(exact_worker and phone_thermal["status"] == 0),
            generation=generation,
            temperature_millic=phone_thermal["max_current_millic"],
            thermal_bucket=str(phone_thermal["bucket"]),
            contention_bucket="exclusive-i3-worker" if exact_worker else "unqualified",
            qualified_contention=exact_worker,
            residency_ids=(
                [f"worker:{WORKER_SHA256}"] if exact_worker else []
            ),
            heartbeat_age_us=None,
            reset_generation=reset,
        ),
        "op15-usb": resource_state(
            ready=bool(exact_bridge and usb_speed == 5000),
            generation=generation,
            temperature_millic=None,
            thermal_bucket="not-applicable",
            contention_bucket=(
                "exclusive-super-speed" if exact_bridge else "unqualified"
            ),
            qualified_contention=exact_bridge,
            residency_ids=(
                [
                    f"bridge:{BRIDGE_SHA256}",
                    "transport:functionfs-dmabuf-f16",
                ]
                if exact_bridge
                else []
            ),
            heartbeat_age_us=None,
            reset_generation=reset,
        ),
    }
    captured_at_us = time.monotonic_ns() // 1000
    evidence = {
        "bridge_hash": bridge_hash,
        "cpu_throttle_count_after": throttle_after or None,
        "cpu_throttle_count_before": throttle_before or None,
        "desktop_topology_qualified": exact_desktop_topology,
        "epoch_assertion_source": "caller_not_verified_by_read_only_probe",
        "gpu": gpu,
        "gpu_temperature_millic": gpu_temperature,
        "mutation_scope": "READ_ONLY_NO_MODEL_OR_WORKER_LAUNCHED",
        "phone_state": phone.get("state"),
        "phone_thermal": phone_thermal,
        "reset_generation_verified": reset_generation is not None,
        "usb_speed_mbps": usb_speed,
        "worker_hash": worker_hash,
        "worker_topology_qualified": exact_worker,
    }
    snapshot_id = "probe-" + hashlib.sha256(canonical_bytes(evidence)).hexdigest()[:16]
    result: dict[str, Any] = {
        "cancellation_generation": 0,
        "captured_at_us": captured_at_us,
        "epoch_key": epoch_key,
        "evidence": evidence,
        "generation": generation,
        "resources": resources,
        "schema": "s42-runtime-snapshot-v1",
        "snapshot_id": snapshot_id,
        "valid_until_us": captured_at_us + validity_us,
    }
    try:
        RuntimeSnapshot.from_json(result)
    except RuntimeGateError as exc:
        raise SnapshotProbeError(str(exc)) from exc
    result["snapshot_sha256"] = "sha256:" + hashlib.sha256(
        canonical_bytes(result)
    ).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="zhihao@172.20.74.85")
    parser.add_argument("--adb-port", type=int, default=0)
    parser.add_argument("--epoch-key", required=True)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--reset-generation", type=int)
    parser.add_argument("--validity-us", type=int, default=5000000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.adb_port < 0 or args.adb_port > 65535:
        parser.error("adb port is invalid")
    if args.generation < 1:
        parser.error("generation must be positive")
    if args.reset_generation is not None and args.reset_generation < 0:
        parser.error("reset generation must be nonnegative")
    if args.validity_us < 1:
        parser.error("validity must be positive")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        result = probe(
            host=args.host,
            adb_port=args.adb_port,
            epoch_key=args.epoch_key,
            generation=args.generation,
            reset_generation=args.reset_generation,
            validity_us=args.validity_us,
        )
        payload = canonical_bytes(result)
        if args.output is None:
            print(payload.decode("ascii"), end="")
        else:
            args.output.write_bytes(payload)
    except (SnapshotProbeError, RuntimeGateError, ValueError) as exc:
        parser.exit(2, f"runtime snapshot probe failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
