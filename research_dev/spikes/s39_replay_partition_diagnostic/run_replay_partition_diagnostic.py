#!/usr/bin/env python3
"""Acquire the frozen CUDA replay-partition matrix without reducing it."""

from __future__ import annotations

import argparse
import hashlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import replay_partition as rp


def artifact_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else rp.ROOT / path


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(command: Sequence[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode("ascii", errors="strict").strip()


def process_identity(process: subprocess.Popen[bytes]) -> dict[str, object]:
    stat = Path(f"/proc/{process.pid}/stat").read_text(encoding="ascii")
    right = stat.rfind(")")
    rp.require(right > 0, "cannot parse process stat")
    fields = stat[right + 2:].split()
    rp.require(len(fields) > 19, "process stat is truncated")
    cmdline = Path(f"/proc/{process.pid}/cmdline").read_bytes()
    return {
        "cmdline_sha256": rp.sha256(cmdline),
        "pid": process.pid,
        "start_time_ticks": int(fields[19]),
    }


def require_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise rp.DiagnosticError(f"host port is busy: {port}") from exc


def wait_log(
    path: Path,
    marker: str,
    process: subprocess.Popen[bytes],
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file() and marker in path.read_text(
            encoding="utf-8", errors="replace"
        ):
            return
        returncode = process.poll()
        if returncode is not None:
            tail = (
                path.read_text(encoding="utf-8", errors="replace")[-4000:]
                if path.is_file()
                else ""
            )
            raise rp.DiagnosticError(
                f"route process exited before readiness ({returncode}): {tail}"
            )
        time.sleep(0.1)
    raise rp.DiagnosticError(f"route readiness timed out: {path.name}")


def stop_processes(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5.0
    for process in reversed(processes):
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def wait_processes(
    processes: Sequence[subprocess.Popen[bytes]],
    timeout_s: float,
) -> list[int]:
    returncodes = []
    deadline = time.monotonic() + timeout_s
    for process in processes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stop_processes(processes)
            raise rp.DiagnosticError("route process shutdown timed out")
        try:
            returncodes.append(process.wait(timeout=remaining))
        except subprocess.TimeoutExpired as exc:
            stop_processes(processes)
            raise rp.DiagnosticError("route process shutdown timed out") from exc
    return returncodes


def start_route(
    directory: Path,
    contract: dict[str, Any],
    ports: tuple[int, int, int],
) -> tuple[list[subprocess.Popen[bytes]], list[dict[str, object]]]:
    head_port, tail_port, relay_port = ports
    for port in ports:
        require_port_free(port)
    worker = artifact_path(contract["artifacts"]["host_worker"]["path"])
    relay_binary = artifact_path(contract["artifacts"]["host_relay"]["path"])
    model = artifact_path(contract["artifacts"]["model"]["path"])
    cuda_uuid = contract["artifacts"]["cuda_uuid"]
    env = os.environ.copy()
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": cuda_uuid,
        "LAYERSPLIT_MODEL_SHA256": contract["artifacts"]["model"]["sha256"],
        "LAYERSPLIT_PLACEMENT_CERT": "1",
    })
    layer_end = contract["execution"]["layer_split"][0]
    n_layer = contract["execution"]["layer_split"][1]
    common = [
        "-m",
        str(model),
        "--driver-batch",
        str(contract["execution"]["batch"]),
        "--driver-context",
        str(contract["execution"]["n_ctx_seq"]),
        "--driver-max-prefill",
        "8",
        "--devices",
        "CUDA0",
        "-ngl",
        "99",
    ]
    specs = [
        (
            "cuda_tail.log",
            [
                str(worker),
                *common[:2],
                "--mode",
                "tailv3",
                "--port",
                str(tail_port),
                *common[2:],
            ],
            {"LLAMA_LAYER_START": str(layer_end), "LLAMA_LAYER_END": str(n_layer)},
            f"[stagenet] listening on 0.0.0.0:{tail_port}",
        ),
        (
            "cuda_head.log",
            [
                str(worker),
                *common[:2],
                "--mode",
                "stagenet",
                "--port",
                str(head_port),
                *common[2:],
            ],
            {"LLAMA_LAYER_START": "0", "LLAMA_LAYER_END": str(layer_end)},
            f"[stagenet] listening on 0.0.0.0:{head_port}",
        ),
    ]
    processes: list[subprocess.Popen[bytes]] = []
    identities: list[dict[str, object]] = []
    try:
        for log_name, command, extra_env, marker in specs:
            log_path = directory / log_name
            with log_path.open("xb") as output:
                process = subprocess.Popen(
                    command,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env={**env, **extra_env},
                )
            processes.append(process)
            identity = process_identity(process)
            identity["role"] = "tail" if "tail" in log_name else "head"
            identities.append(identity)
            wait_log(log_path, marker, process, 180.0)

        relay_log = directory / "cuda_relay.log"
        with relay_log.open("xb") as output:
            relay = subprocess.Popen(
                [
                    str(relay_binary),
                    "--listen",
                    str(relay_port),
                    "--head",
                    f"127.0.0.1:{head_port}",
                    "--tail",
                    f"127.0.0.1:{tail_port}",
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
                env=env,
            )
        processes.append(relay)
        identity = process_identity(relay)
        identity["role"] = "relay"
        identities.append(identity)
        wait_log(
            relay_log,
            f"[direct-relay] listening on 0.0.0.0:{relay_port}",
            relay,
            30.0,
        )
        return processes, identities
    except BaseException:
        stop_processes(processes)
        raise


def artifact_map(directory: Path, names: Sequence[str]) -> dict[str, str]:
    return {
        name: digest_file(directory / name)
        for name in names
        if (directory / name).is_file()
    }


def acquire_run(
    root: Path,
    contract_path: Path,
    inputs_path: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    run_spec: dict[str, Any],
    run_index: int,
) -> dict[str, object]:
    run_name = run_spec["name"]
    directory = root / run_name
    directory.mkdir()
    ports = (
        41810 + run_index * 10,
        41811 + run_index * 10,
        41812 + run_index * 10,
    )
    started_ns = time.monotonic_ns()
    processes: list[subprocess.Popen[bytes]] = []
    try:
        processes, identities = start_route(directory, contract, ports)
        stdout_path = directory / "probe.stdout"
        stderr_path = directory / "probe.stderr"
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            probe = subprocess.run(
                [
                    sys.executable,
                    str(rp.HERE / "replay_partition_probe.py"),
                    "--contract",
                    str(contract_path),
                    "--inputs",
                    str(inputs_path),
                    "--cuda-route",
                    f"127.0.0.1:{ports[2]}",
                    "--run-name",
                    run_name,
                    "--output",
                    str(directory / "raw_report.json"),
                    "--timeout",
                    "300",
                ],
                stdout=stdout,
                stderr=stderr,
                timeout=360,
            )
        returncodes = wait_processes(processes, 90.0)
        rp.require(probe.returncode == 0, f"{run_name}: probe failed")
        rp.require(returncodes == [0, 0, 0], f"{run_name}: route exit failed")
        names = [
            "cuda_head.log",
            "cuda_relay.log",
            "cuda_tail.log",
            "probe.stderr",
            "probe.stdout",
            "raw_report.json",
        ]
        files = artifact_map(directory, names)
        rp.require(set(files) == set(names), f"{run_name}: missing raw artifact")
        record = {
            "contract_sha256": contract_sha256,
            "ended_ns": time.monotonic_ns(),
            "files": files,
            "fresh_process": run_spec["fresh_process"],
            "inputs_sha256": contract["inputs"]["sha256"],
            "ports": {
                "head": ports[0],
                "relay": ports[2],
                "tail": ports[1],
            },
            "probe_returncode": probe.returncode,
            "processes": identities,
            "route_returncodes": {
                "head": returncodes[1],
                "relay": returncodes[2],
                "tail": returncodes[0],
            },
            "run_index": run_index,
            "run_name": run_name,
            "schema": rp.RUN_RECORD_SCHEMA,
            "started_ns": started_ns,
        }
        rp.write_atomic(directory / "RUN_RECORD.json", record)
        return record
    except subprocess.TimeoutExpired as exc:
        stop_processes(processes)
        raise rp.DiagnosticError(f"{run_name}: probe timed out") from exc
    except BaseException:
        stop_processes(processes)
        raise


def gpu_identity(uuid: str) -> dict[str, object]:
    output = run_text([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        rp.require(len(fields) == 4, "invalid nvidia-smi identity row")
        rows.append({
            "index": int(fields[0]),
            "memory_total_mib": int(fields[3]),
            "name": fields[2],
            "uuid": fields[1],
        })
    matches = [row for row in rows if row["uuid"] == uuid]
    rp.require(len(matches) == 1, "selected CUDA UUID is absent or ambiguous")
    return matches[0]


def require_no_compute_process(uuid: str) -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    rows = result.stdout.decode("ascii", errors="strict").splitlines()
    conflicts = [row for row in rows if row.split(",", 1)[0].strip() == uuid]
    rp.require(not conflicts, f"selected CUDA device has compute processes: {conflicts}")


def write_manifest(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS_ACQUISITION.txt":
            entries.append(
                f"{digest_file(path)}  {path.relative_to(root)}\n"
            )
    manifest = root / "SHA256SUMS_ACQUISITION.txt"
    with manifest.open("xb") as output:
        output.write("".join(entries).encode("ascii"))
        output.flush()
        os.fsync(output.fileno())
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return digest_file(manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=rp.HERE / "REPLAY_PARTITION_DIAGNOSTIC.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=rp.HERE / "results" / (
            "run_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")

    contract, contract_sha256 = rp.load_contract(args.contract)
    inputs_path = artifact_path(contract["inputs"]["path"])
    rp.load_inputs(inputs_path, contract)
    for name in ("host_worker", "host_relay", "model"):
        artifact = contract["artifacts"][name]
        path = artifact_path(artifact["path"])
        rp.require(path.is_file(), f"missing artifact: {name}")
        rp.require(
            digest_file(path) == artifact["sha256"],
            f"artifact digest mismatch: {name}",
        )
    identity = gpu_identity(contract["artifacts"]["cuda_uuid"])
    rp.require(identity["name"] == "NVIDIA RTX A6000", "CUDA device name changed")
    require_no_compute_process(contract["artifacts"]["cuda_uuid"])

    output = args.output.resolve()
    output.mkdir(parents=True)
    context = {
        "base_git_commit": run_text(["git", "-C", str(rp.ROOT), "rev-parse", "HEAD"]),
        "contract_sha256": contract_sha256,
        "cuda": identity,
        "inputs_sha256": contract["inputs"]["sha256"],
        "run_order": [run["name"] for run in contract["execution"]["runs"]],
        "schema": "s39-replay-partition-acquisition-context-v1",
        "scope": "CUDA_ONLY_DIAGNOSTIC",
        "started_unix_s": int(time.time()),
    }
    rp.write_atomic(output / "ACQUISITION_CONTEXT.json", context)
    records = []
    try:
        for index, run_spec in enumerate(contract["execution"]["runs"]):
            records.append(
                acquire_run(
                    output,
                    args.contract.resolve(),
                    inputs_path.resolve(),
                    contract,
                    contract_sha256,
                    run_spec,
                    index,
                )
            )
        acquisition = {
            "comparison_evaluated": False,
            "contract_sha256": contract_sha256,
            "inputs_sha256": contract["inputs"]["sha256"],
            "raw_reports": [
                {
                    "path": f"{record['run_name']}/raw_report.json",
                    "sha256": record["files"]["raw_report.json"],
                }
                for record in records
            ],
            "run_records": [
                {
                    "path": f"{record['run_name']}/RUN_RECORD.json",
                    "sha256": digest_file(
                        output / record["run_name"] / "RUN_RECORD.json"
                    ),
                }
                for record in records
            ],
            "schema": "s39-replay-partition-acquisition-v1",
            "status": "RAW_CAPTURE_COMPLETE_NO_EQUALITY_EVALUATED",
        }
        rp.write_atomic(output / "ACQUISITION.json", acquisition)
        manifest_sha256 = write_manifest(output)
        print(f"RAW_CAPTURE_COMPLETE {output}")
        print(f"ACQUISITION_MANIFEST_SHA256 {manifest_sha256}")
        return 0
    except BaseException as exc:
        rp.write_atomic(output / "ACQUISITION_FAILURE.json", {
            "completed_runs": [record["run_name"] for record in records],
            "contract_sha256": contract_sha256,
            "error": str(exc),
            "schema": "s39-replay-partition-acquisition-failure-v1",
            "status": "FAIL_CLOSED",
        })
        write_manifest(output)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, rp.DiagnosticError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
