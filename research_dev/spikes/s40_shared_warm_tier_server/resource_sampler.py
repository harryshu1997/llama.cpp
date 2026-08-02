#!/usr/bin/env python3
"""Sample the selected RTX board and host resources in one clock domain."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from evidence_common import EvidenceError, canonical_bytes, require
from evidence_common import digest_file, validate_digest


GPU_QUERY = (
    "uuid,memory.used,memory.free,power.draw"
)
MEMINFO_KEYS = {
    "MemAvailable": "system_mem_available_bytes",
    "SwapFree": "system_swap_free_bytes",
    "SwapTotal": "system_swap_total_bytes",
}


def parse_nvidia_csv(raw: str, expected_uuid: str) -> dict[str, int | str]:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    require(len(lines) == 1, "nvidia-smi: expected one selected GPU")
    values = [value.strip() for value in lines[0].split(",")]
    require(len(values) == 4, "nvidia-smi: field count mismatch")
    require(values[0] == expected_uuid, "nvidia-smi: GPU UUID mismatch")
    try:
        memory_used = Decimal(values[1])
        memory_free = Decimal(values[2])
        power_w = Decimal(values[3])
    except InvalidOperation as error:
        raise EvidenceError("nvidia-smi: nonnumeric sample") from error
    require(
        memory_used >= 0 and memory_free >= 0 and power_w >= 0,
        "nvidia-smi: negative sample",
    )
    used_bytes = memory_used * (1 << 20)
    free_bytes = memory_free * (1 << 20)
    power_mw = power_w * 1000
    require(
        used_bytes == used_bytes.to_integral_value()
        and free_bytes == free_bytes.to_integral_value()
        and power_mw == power_mw.to_integral_value(),
        "nvidia-smi: sample cannot be represented exactly",
    )
    return {
        "gpu_memory_free_bytes": int(free_bytes),
        "gpu_memory_used_bytes": int(used_bytes),
        "gpu_power_mw": int(power_mw),
        "gpu_uuid": values[0],
    }


def parse_meminfo(raw: str) -> dict[str, int]:
    found: dict[str, int] = {}
    for line in raw.splitlines():
        fields = line.split()
        if not fields:
            continue
        name = fields[0].removesuffix(":")
        if name not in MEMINFO_KEYS:
            continue
        require(
            len(fields) == 3
            and fields[1].isascii()
            and fields[1].isdigit()
            and fields[2] == "kB",
            f"meminfo: invalid {name}",
        )
        found[MEMINFO_KEYS[name]] = int(fields[1]) * 1024
    require(set(found) == set(MEMINFO_KEYS.values()),
            "meminfo: missing required counters")
    require(
        found["system_swap_free_bytes"]
        <= found["system_swap_total_bytes"],
        "meminfo: invalid swap counters",
    )
    return found


def parse_cpu_stat(raw: str) -> tuple[int, int]:
    first = raw.splitlines()[0].split()
    require(first and first[0] == "cpu" and len(first) >= 9,
            "proc stat: invalid aggregate row")
    require(
        all(value.isascii() and value.isdigit() for value in first[1:]),
        "proc stat: nonnumeric counter",
    )
    values = [int(value) for value in first[1:]]
    total = sum(values)
    idle = values[3] + values[4]
    require(total >= idle, "proc stat: invalid idle counter")
    return total, idle


def cpu_utilization_milli_pct(
        previous: tuple[int, int] | None,
        current: tuple[int, int]) -> int:
    if previous is None:
        return 0
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    require(total_delta > 0, "proc stat: total counter did not advance")
    require(0 <= idle_delta <= total_delta,
            "proc stat: invalid counter delta")
    return (total_delta - idle_delta) * 100_000 // total_delta


def parse_process_stat(raw: str) -> tuple[int, int]:
    open_index = raw.find("(")
    close_index = raw.rfind(")")
    require(
        open_index > 0
        and close_index > open_index
        and raw[:open_index].strip().isascii()
        and raw[:open_index].strip().isdigit(),
        "process stat: invalid prefix",
    )
    fields = raw[close_index + 1:].split()
    require(len(fields) >= 20, "process stat: missing counters")
    try:
        user_ticks = int(fields[11])
        system_ticks = int(fields[12])
        start_ticks = int(fields[19])
    except ValueError as error:
        raise EvidenceError("process stat: nonnumeric counter") from error
    require(
        user_ticks >= 0 and system_ticks >= 0 and start_ticks > 0,
        "process stat: invalid counter",
    )
    return user_ticks + system_ticks, start_ticks


def process_cpu_utilization_milli_pct(
        previous_host: tuple[int, int] | None,
        current_host: tuple[int, int],
        previous_process_ticks: int | None,
        current_process_ticks: int) -> int:
    if previous_host is None or previous_process_ticks is None:
        return 0
    total_delta = current_host[0] - previous_host[0]
    process_delta = current_process_ticks - previous_process_ticks
    require(total_delta > 0, "process CPU: host counter did not advance")
    require(0 <= process_delta <= total_delta,
            "process CPU: invalid counter delta")
    return process_delta * 100_000 // total_delta


def parse_process_status(raw: str) -> dict[str, int]:
    found: dict[str, int] = {}
    for line in raw.splitlines():
        fields = line.split()
        if not fields or fields[0] not in {"VmRSS:", "VmSwap:"}:
            continue
        require(
            len(fields) == 3
            and fields[1].isascii()
            and fields[1].isdigit()
            and fields[2] == "kB",
            "process status: invalid memory row",
        )
        key = (
            "process_rss_bytes"
            if fields[0] == "VmRSS:" else "process_swap_bytes"
        )
        found[key] = int(fields[1]) * 1024
    require(
        set(found) == {"process_rss_bytes", "process_swap_bytes"},
        "process status: missing memory counters",
    )
    return found


def collect_sample(
        run_id: str,
        sequence: int,
        gpu_uuid: str,
        controller_pid: int,
        expected_boot_id: str,
        expected_start_ticks: int,
        previous_cpu: tuple[int, int] | None,
        previous_process_ticks: int | None,
        nvidia_smi_path: Path,
        command_runner: Callable[..., subprocess.CompletedProcess[str]]
        = subprocess.run,
) -> tuple[dict[str, Any], tuple[int, int], int]:
    result = command_runner(
        [
            str(nvidia_smi_path),
            "--id",
            gpu_uuid,
            f"--query-gpu={GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=2.0,
    )
    require(result.returncode == 0, "nvidia-smi: command failed")
    gpu = parse_nvidia_csv(result.stdout, gpu_uuid)
    current_cpu = parse_cpu_stat(
        Path("/proc/stat").read_text(encoding="ascii"))
    host_boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii").strip()
    require(host_boot_id == expected_boot_id,
            "sampler: host boot identity changed")
    host = parse_meminfo(
        Path("/proc/meminfo").read_text(encoding="ascii"))
    process_ticks, start_ticks = parse_process_stat(
        Path(f"/proc/{controller_pid}/stat").read_text(encoding="ascii"))
    require(start_ticks == expected_start_ticks,
            "sampler: controller PID identity changed")
    process = parse_process_status(
        Path(f"/proc/{controller_pid}/status").read_text(encoding="ascii"))
    row = {
        "controller_pid": controller_pid,
        "controller_process_cpu_ticks": process_ticks,
        "controller_process_cpu_utilization_milli_pct":
            process_cpu_utilization_milli_pct(
                previous_cpu,
                current_cpu,
                previous_process_ticks,
                process_ticks,
            ),
        "controller_process_rss_bytes": process["process_rss_bytes"],
        "controller_process_start_ticks": start_ticks,
        "controller_process_swap_bytes": process["process_swap_bytes"],
        "cpu_utilization_milli_pct": cpu_utilization_milli_pct(
            previous_cpu, current_cpu),
        **gpu,
        "host_boot_id": host_boot_id,
        "process_metric_scope": "CONTROLLER_PROCESS_ONLY",
        "run_id": run_id,
        "schema": "s40-selected-gpu-resource-v3",
        "sequence": sequence,
        **host,
        "t_ns": time.monotonic_ns(),
    }
    return row, current_cpu, process_ticks


def sample_until_stopped(
        run_id: str,
        gpu_uuid: str,
        controller_pid: int,
        output: Path,
        stop_file: Path,
        interval_ms: int,
        nvidia_smi_path: Path,
        nvidia_smi_sha256: str) -> int:
    require(output.is_absolute() and stop_file.is_absolute(),
            "sampler: paths must be absolute")
    require(not output.exists(), "sampler: output already exists")
    require(not stop_file.exists(), "sampler: stop file already exists")
    require(interval_ms >= 50 and interval_ms <= 400,
            "sampler: interval must be 50..400 ms")
    require(controller_pid > 1, "sampler: invalid controller PID")
    require(
        nvidia_smi_path.is_absolute()
        and nvidia_smi_path.is_file()
        and os.access(nvidia_smi_path, os.X_OK),
        "sampler: nvidia-smi executable is missing",
    )
    expected_nvidia_smi_sha256 = validate_digest(
        nvidia_smi_sha256, "sampler nvidia-smi SHA-256")
    require(
        digest_file(nvidia_smi_path) == expected_nvidia_smi_sha256,
        "sampler: nvidia-smi digest mismatch",
    )
    host_boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii").strip()
    require(bool(host_boot_id) and host_boot_id.isascii(),
            "sampler: invalid host boot ID")
    _, controller_start_ticks = parse_process_stat(
        Path(f"/proc/{controller_pid}/stat").read_text(encoding="ascii"))
    previous_cpu = None
    previous_process_ticks = None
    sequence = 0
    with output.open("xb", buffering=0) as sink:
        while True:
            row, previous_cpu, previous_process_ticks = collect_sample(
                run_id,
                sequence,
                gpu_uuid,
                controller_pid,
                host_boot_id,
                controller_start_ticks,
                previous_cpu,
                previous_process_ticks,
                nvidia_smi_path,
            )
            sink.write(canonical_bytes(row))
            sink.flush()
            os.fsync(sink.fileno())
            sequence += 1
            if stop_file.exists():
                break
            time.sleep(interval_ms / 1000)
    require(
        digest_file(nvidia_smi_path) == expected_nvidia_smi_sha256,
        "sampler: nvidia-smi changed during acquisition",
    )
    directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return sequence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--controller-pid", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stop-file", required=True, type=Path)
    parser.add_argument("--interval-ms", type=int, default=200)
    parser.add_argument(
        "--nvidia-smi-path", required=True, type=Path)
    parser.add_argument("--nvidia-smi-sha256", required=True)
    args = parser.parse_args()
    try:
        count = sample_until_stopped(
            args.run_id,
            args.gpu_uuid,
            args.controller_pid,
            args.output,
            args.stop_file,
            args.interval_ms,
            args.nvidia_smi_path,
            args.nvidia_smi_sha256,
        )
    except (EvidenceError, OSError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}")
        return 2
    print(canonical_bytes({
        "record_count": count,
        "schema": "s40-resource-sampler-result-v1",
        "status": "S40_RESOURCE_SAMPLER_COMPLETE",
    }).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
