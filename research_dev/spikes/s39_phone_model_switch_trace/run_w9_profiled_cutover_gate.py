#!/usr/bin/env python3
"""Acquire the exact four W9 treatment/control pairs without replacement."""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import phone_cuda_delta_probe as w6
import validate_profiled_cutover_pair as pair_validator
import w9_profiled_cutover as w9


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNTIME = "/data/local/tmp/s39-qwen25-q8-v1"
MODEL_SHA256 = "23ca481b8226b2492ba8f3eb7af41e0f99d8605c16fb6dec7bc5cf6716b673cf"
OP15_SHARD_SHA256 = "b9611440eb4901764cef6afb55e08418acb114f1dfdc5b97e2c4acafefcc5375"
OP12_SHARD_SHA256 = "b66f9f6ace28da341f21f4f0d03ffa05c31cea647d43da4023e373f6021551ac"
PHONE_WORKER_SHA256 = "45ff9eda965d9e1688776f779895388b82f8e7d6933fe453019691973997db3f"
PHONE_RELAY_SHA256 = "1c809cb50cae6aa86869d61068a05173c719e4542c851e478ee1e033c5456929"
HOST_WORKER_SHA256 = "833a7ed615a5402153a490433a773efe9b02d4447f1f9a8c72efd95f112fed70"
HOST_RELAY_SHA256 = "3b616b4f1372f138d9ca56976618a11523a90b29fb734638a8585a184d06761e"


class Campaign:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out = args.output
        self.contract = w9.load_contract(args.contract)
        self.host_processes: list[subprocess.Popen[bytes]] = []
        self.remote_started = False
        self.completed: list[str] = []
        self.thermal_starts: dict[str, list[int]] = {"op12": [], "op15": []}
        self.out.mkdir(mode=0o700, parents=False, exist_ok=False)

    def run(
        self,
        arguments: Sequence[str],
        *,
        check: bool = True,
        capture_output: bool = True,
        timeout: float = 120.0,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        process = subprocess.run(
            list(arguments),
            check=False,
            capture_output=capture_output,
            timeout=timeout,
            env=env,
        )
        if check and process.returncode != 0:
            stderr = process.stderr.decode("utf-8", errors="replace")
            raise w9.W9Error(
                f"command failed ({process.returncode}): "
                f"{' '.join(arguments)}: {stderr[-1000:]}"
            )
        return process

    def adb(
        self,
        serial: str,
        *arguments: str,
        check: bool = True,
        timeout: float = 120.0,
    ) -> subprocess.CompletedProcess[bytes]:
        return self.run(
            [
                self.args.adb,
                "-P",
                str(self.args.adb_port),
                "-s",
                serial,
                *arguments,
            ],
            check=check,
            timeout=timeout,
        )

    def remote_text(self, serial: str, command: str) -> str:
        return self.adb(serial, "shell", command).stdout.decode(
            "ascii"
        ).replace("\r", "").strip()

    def remote_hash(self, serial: str, path: str) -> str:
        output = self.remote_text(serial, f"sha256sum '{path}'")
        fields = output.split()
        w9.require(bool(fields), f"remote hash: {serial}:{path}")
        return fields[0]

    def wait_file(
        self,
        path: Path,
        process: subprocess.Popen[bytes],
        *,
        timeout: float,
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.is_file():
                return
            if process.poll() is not None:
                raise w9.W9Error(
                    f"process exited before creating {path}: rc={process.returncode}"
                )
            time.sleep(0.01)
        raise w9.W9Error(f"timed out waiting for {path}")

    def wait_host_log(
        self,
        path: Path,
        marker: str,
        process: subprocess.Popen[bytes],
        *,
        timeout: float = 120.0,
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.is_file() and marker in path.read_text(
                encoding="utf-8",
                errors="replace",
            ):
                return
            if process.poll() is not None:
                tail = (
                    path.read_text(encoding="utf-8", errors="replace")[-2000:]
                    if path.is_file()
                    else ""
                )
                raise w9.W9Error(f"host worker exited before ready: {tail}")
            time.sleep(0.1)
        raise w9.W9Error(f"host worker readiness timed out: {path}")

    def remote_pid(self, serial: str, name: str) -> int | None:
        value = self.remote_text(
            serial,
            f"cat '{RUNTIME}/{name}' 2>/dev/null || true",
        )
        return int(value) if value.isdigit() else None

    def wait_phone_log(
        self,
        serial: str,
        log: str,
        marker: str,
        pid_file: str,
        *,
        timeout: float = 180.0,
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.adb(
                serial,
                "shell",
                f"grep -Fq '{marker}' '{RUNTIME}/{log}'",
                check=False,
                timeout=10,
            ).returncode == 0:
                return
            pid = self.remote_pid(serial, pid_file)
            if pid is None or self.adb(
                serial,
                "shell",
                f"kill -0 {pid}",
                check=False,
                timeout=10,
            ).returncode != 0:
                tail = self.remote_text(
                    serial,
                    f"tail -100 '{RUNTIME}/{log}' 2>/dev/null || true",
                )
                raise w9.W9Error(f"phone worker exited before ready: {tail}")
            time.sleep(1)
        raise w9.W9Error(f"phone worker readiness timed out: {serial}/{log}")

    def stop_remote(self, serial: str, pid_file: str) -> None:
        pid = self.remote_pid(serial, pid_file)
        if pid is not None:
            self.adb(
                serial,
                "shell",
                f"kill {pid} 2>/dev/null || true",
                check=False,
            )

    def wait_remote_exit(
        self,
        serial: str,
        pid_file: str,
        *,
        timeout: float = 60.0,
    ) -> bool:
        pid = self.remote_pid(serial, pid_file)
        if pid is None:
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.adb(
                serial,
                "shell",
                f"kill -0 {pid}",
                check=False,
                timeout=10,
            ).returncode != 0:
                return True
            time.sleep(0.5)
        self.stop_remote(serial, pid_file)
        return False

    def cleanup_remote(self) -> None:
        for serial, names in (
            (self.args.op15, ("w9_relay.pid", "w9_head.pid")),
            (self.args.op12, ("w9_tail.pid",)),
        ):
            for name in names:
                try:
                    self.stop_remote(serial, name)
                except (OSError, subprocess.SubprocessError, w9.W9Error):
                    pass
        self.remote_started = False

    def cleanup_host(self) -> None:
        for process in self.host_processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 10
        for process in self.host_processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self.host_processes.clear()

    def preflight(self) -> None:
        w9.require(
            self.args.cuda_uuid == self.contract.cuda_uuid
            and self.args.op15 == self.contract.op15_serial
            and self.args.op15_wifi == self.contract.op15_wifi
            and self.args.op12 == self.contract.op12_serial
            and self.args.op12_wifi == self.contract.op12_wifi,
            "preflight: device contract mismatch",
        )
        devices = self.run(
            [self.args.adb, "-P", str(self.args.adb_port), "devices"],
        ).stdout.decode("ascii")
        for serial in (self.args.op15, self.args.op12):
            w9.require(
                any(
                    line.split()[:2] == [serial, "device"]
                    for line in devices.splitlines()
                ),
                f"device is not ready: {serial}",
            )
        artifacts = {
            self.args.model: MODEL_SHA256,
            self.args.host_worker: HOST_WORKER_SHA256,
            self.args.host_relay: HOST_RELAY_SHA256,
        }
        for path, expected in artifacts.items():
            w9.require(
                path.is_file() and self.file_sha256(path) == expected,
                f"host artifact mismatch: {path}",
            )
        for serial, path, digest in (
            (self.args.op15, f"{RUNTIME}/llama-layersplit", PHONE_WORKER_SHA256),
            (self.args.op12, f"{RUNTIME}/llama-layersplit", PHONE_WORKER_SHA256),
            (
                self.args.op15,
                f"{RUNTIME}/llama-stage-direct-relay",
                PHONE_RELAY_SHA256,
            ),
            (self.args.op15, f"{RUNTIME}/weights.gguf", OP15_SHARD_SHA256),
            (self.args.op12, f"{RUNTIME}/weights.gguf", OP12_SHARD_SHA256),
        ):
            w9.require(
                self.remote_hash(serial, path) == digest,
                f"remote artifact mismatch: {serial}:{path}",
            )
        for name, digest in self.contract.source_sha256.items():
            path = pair_validator.source_paths()[name]
            w9.require(
                w6.sha256(path.read_bytes()) == digest,
                f"source changed after freeze: {name}",
            )
        w9.require(not self.host_route_pids(), "preflight: live CUDA route")

    @staticmethod
    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while True:
                block = source.read(16 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    def host_route_pids(self) -> list[int]:
        binary = self.args.host_worker.resolve()
        result = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                argv = [
                    item.decode("utf-8")
                    for item in (entry / "cmdline").read_bytes().split(b"\0")
                    if item
                ]
                executable = Path(argv[0]).resolve() if argv else None
            except (OSError, UnicodeError):
                continue
            if (
                executable == binary
                and str(self.args.model) in argv
                and "--mode" in argv
                and any(mode in argv for mode in ("stagenet", "tailv3"))
            ):
                result.append(int(entry.name))
        return sorted(result)

    def write_campaign_context(self) -> None:
        w9.write_atomic(self.out / "CAMPAIGN_CONTEXT.json", {
            "base_git_commit": self.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"]
            ).stdout.decode("ascii").strip(),
            "contract_sha256": self.contract.raw_sha256,
            "created_utc_ns": time.time_ns(),
            "pair_ordinals": list(self.contract.pair_ordinals),
            "schema": "s39-profiled-cutover-campaign-context-v1",
        })

    def capture_gpu(
        self,
        output: Path,
        *,
        label: str,
        idle: bool,
        post: bool = False,
    ) -> None:
        arguments = [
            sys.executable,
            str(HERE / "w9_host_evidence.py"),
            "gpu",
            "--uuid",
            self.args.cuda_uuid,
            "--samples",
            str(self.contract.idle_samples if (idle or post) else 1),
            "--span-us",
            str(self.contract.idle_span_us if (idle or post) else 0),
            "--label",
            label,
            "--output",
            str(output),
        ]
        if idle:
            arguments.append("--require-idle")
        self.run(arguments, env=os.environ.copy(), timeout=60)

    def warm_model(self, output: Path, label: str) -> None:
        self.run(
            [
                sys.executable,
                str(HERE / "w9_host_evidence.py"),
                "warm",
                "--path",
                str(self.args.model),
                "--label",
                label,
                "--output",
                str(output),
            ],
            timeout=180,
        )

    def capture_thermal(self, output: Path, label: str) -> dict[str, object]:
        self.run([
            sys.executable,
            str(HERE / "capture_phone_thermal.py"),
            "--adb",
            self.args.adb,
            "--adb-port",
            str(self.args.adb_port),
            "--device",
            f"op15={self.args.op15}",
            "--device",
            f"op12={self.args.op12}",
            "--label",
            label,
            "--output",
            str(output),
        ])
        value, _ = w6.read_canonical(output, "thermal")
        return value

    def capture_thermal_start(self, pair_dir: Path, ordinal: str) -> None:
        setup = pair_dir / "setup"
        setup.mkdir(exist_ok=True)
        deadline = time.monotonic() + self.args.thermal_wait_s
        attempt = 0
        while True:
            candidate = setup / f"thermal_candidate_{attempt:03d}.json"
            value = self.capture_thermal(candidate, f"{ordinal}.T.PRE")
            current = {
                name: value["samples"][name]["gpu_max_millic"]
                for name in ("op12", "op15")
            }
            acceptable = all(
                not self.thermal_starts[name]
                or (
                    max(self.thermal_starts[name] + [current[name]])
                    - min(self.thermal_starts[name] + [current[name]])
                    <= self.contract.phone_thermal_range_millic
                )
                for name in ("op12", "op15")
            )
            if acceptable:
                target = pair_dir / "treatment/thermal_pre.json"
                os.link(candidate, target)
                candidate.unlink()
                for name in ("op12", "op15"):
                    self.thermal_starts[name].append(current[name])
                return
            if time.monotonic() >= deadline:
                raise w9.W9Error(
                    f"{ordinal}: phone thermal setup did not enter frozen range"
                )
            attempt += 1
            time.sleep(10)

    def start_phones(self, treatment: Path) -> None:
        self.remote_started = True
        self.adb(
            self.args.op12,
            "shell",
            f"""
cd '{RUNTIME}'
rm -f w9_tail.log w9_tail.pid
nohup env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  LLAMA_LAYER_START=30 LLAMA_LAYER_END=48 \
  LAYERSPLIT_MODEL_SHA256='{MODEL_SHA256}' \
  LAYERSPLIT_PLACEMENT_CERT=1 \
  ./llama-layersplit -m '{RUNTIME}/weights.gguf' \
  --mode tailv3 --port 41532 \
  --driver-batch 8 --driver-context 64 --driver-max-prefill 8 \
  --devices GPUOpenCL -ngl 99 \
  >'{RUNTIME}/w9_tail.log' 2>&1 </dev/null &
echo $! >'{RUNTIME}/w9_tail.pid'
""",
        )
        self.wait_phone_log(
            self.args.op12,
            "w9_tail.log",
            "[stagenet] listening on 0.0.0.0:41532",
            "w9_tail.pid",
        )
        self.adb(
            self.args.op15,
            "shell",
            f"""
cd '{RUNTIME}'
rm -f w9_head.log w9_head.pid
nohup env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  LLAMA_LAYER_START=0 LLAMA_LAYER_END=30 \
  LAYERSPLIT_MODEL_SHA256='{MODEL_SHA256}' \
  LAYERSPLIT_PLACEMENT_CERT=1 \
  ./llama-layersplit -m '{RUNTIME}/weights.gguf' \
  --mode stagenet --port 41515 \
  --driver-batch 8 --driver-context 64 --driver-max-prefill 8 \
  --devices GPUOpenCL -ngl 99 \
  >'{RUNTIME}/w9_head.log' 2>&1 </dev/null &
echo $! >'{RUNTIME}/w9_head.pid'
""",
        )
        self.wait_phone_log(
            self.args.op15,
            "w9_head.log",
            "[stagenet] listening on 0.0.0.0:41515",
            "w9_head.pid",
        )
        self.adb(
            self.args.op15,
            "shell",
            f"""
cd '{RUNTIME}'
rm -f w9_relay.log w9_relay.pid
nohup ./llama-stage-direct-relay \
  --listen 41525 --head 127.0.0.1:41515 \
  --tail '{self.args.op12_wifi}:41532' \
  >'{RUNTIME}/w9_relay.log' 2>&1 </dev/null &
echo $! >'{RUNTIME}/w9_relay.pid'
""",
        )
        self.wait_phone_log(
            self.args.op15,
            "w9_relay.log",
            "[direct-relay] listening on 0.0.0.0:41525",
            "w9_relay.pid",
        )

    def collect_phone_logs(self, treatment: Path) -> None:
        for serial, remote, local in (
            (self.args.op15, "w9_head.log", "op15_head.log"),
            (self.args.op15, "w9_relay.log", "phone_relay.log"),
            (self.args.op12, "w9_tail.log", "op12_tail.log"),
        ):
            process = self.adb(
                serial,
                "pull",
                f"{RUNTIME}/{remote}",
                str(treatment / local),
                check=False,
            )
            w9.require(process.returncode == 0, f"failed to collect {remote}")

    def start_host_route(
        self,
        directory: Path,
        *,
        ports: tuple[int, int, int],
    ) -> list[subprocess.Popen[bytes]]:
        head_port, tail_port, relay_port = ports
        env = os.environ.copy()
        env.update({
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": self.args.cuda_uuid,
            "LAYERSPLIT_MODEL_SHA256": MODEL_SHA256,
            "LAYERSPLIT_PLACEMENT_CERT": "1",
        })
        commands = [
            (
                directory / "cuda_tail.log",
                [
                    str(self.args.host_worker),
                    "-m",
                    str(self.args.model),
                    "--mode",
                    "tailv3",
                    "--port",
                    str(tail_port),
                    "--driver-batch",
                    "8",
                    "--driver-context",
                    "64",
                    "--driver-max-prefill",
                    "8",
                    "--devices",
                    "CUDA0",
                    "-ngl",
                    "99",
                ],
                {"LLAMA_LAYER_START": "30", "LLAMA_LAYER_END": "48"},
            ),
            (
                directory / "cuda_head.log",
                [
                    str(self.args.host_worker),
                    "-m",
                    str(self.args.model),
                    "--mode",
                    "stagenet",
                    "--port",
                    str(head_port),
                    "--driver-batch",
                    "8",
                    "--driver-context",
                    "64",
                    "--driver-max-prefill",
                    "8",
                    "--devices",
                    "CUDA0",
                    "-ngl",
                    "99",
                ],
                {"LLAMA_LAYER_START": "0", "LLAMA_LAYER_END": "30"},
            ),
        ]
        workers = []
        for log, command, extra in commands:
            output = log.open("xb")
            worker_env = {**env, **extra}
            process = subprocess.Popen(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=worker_env,
            )
            output.close()
            workers.append(process)
            self.host_processes.append(process)
        self.wait_host_log(
            directory / "cuda_tail.log",
            f"[stagenet] listening on 0.0.0.0:{tail_port}",
            workers[0],
        )
        self.wait_host_log(
            directory / "cuda_head.log",
            f"[stagenet] listening on 0.0.0.0:{head_port}",
            workers[1],
        )
        relay_output = (directory / "cuda_relay.log").open("xb")
        relay = subprocess.Popen(
            [
                str(self.args.host_relay),
                "--listen",
                str(relay_port),
                "--head",
                f"127.0.0.1:{head_port}",
                "--tail",
                f"127.0.0.1:{tail_port}",
            ],
            stdout=relay_output,
            stderr=subprocess.STDOUT,
            env=env,
        )
        relay_output.close()
        workers.append(relay)
        self.host_processes.append(relay)
        self.wait_host_log(
            directory / "cuda_relay.log",
            f"[direct-relay] listening on 0.0.0.0:{relay_port}",
            relay,
        )
        return workers

    def wait_processes(
        self,
        processes: Sequence[subprocess.Popen[bytes]],
        *,
        timeout: float = 90.0,
    ) -> bool:
        result = True
        for process in processes:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                result = False
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        return result

    def wait_until(self, deadline_ns: int) -> None:
        while True:
            remaining = deadline_ns - time.monotonic_ns()
            if remaining <= 0:
                return
            time.sleep(min(remaining / 1_000_000_000, 0.01))

    def write_launch(
        self,
        path: Path,
        *,
        phase: str,
        ordinal: str,
        run_id: str,
        request_start_ns: int,
        deadline_ns: int,
    ) -> None:
        existing = self.host_route_pids()
        w9.require(not existing, f"{phase}: preexisting CUDA route {existing}")
        self.wait_until(deadline_ns)
        w9.write_atomic(path, {
            "cuda_launch_ns": time.monotonic_ns(),
            "launch_deadline_ns": deadline_ns,
            "pair_ordinal": ordinal,
            "phase": phase,
            "preexisting_route_pids": existing,
            "request_start_ns": request_start_ns,
            "run_id": run_id,
            "schema": "s39-profiled-cutover-launch-v1",
        })

    def write_pair_context(
        self,
        pair_dir: Path,
        ordinal: str,
        run_id: str,
    ) -> None:
        pre, _ = w6.read_canonical(
            pair_dir / "treatment/gpu_pre.json",
            "gpu_pre",
        )
        identity = pre["payload"]["identity"]
        host_boot = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
        op15_boot = self.remote_text(
            self.args.op15,
            "cat /proc/sys/kernel/random/boot_id",
        )
        op12_boot = self.remote_text(
            self.args.op12,
            "cat /proc/sys/kernel/random/boot_id",
        )
        w9.write_atomic(pair_dir / "PAIR_CONTEXT.json", {
            "acquisition_unix_s": int(time.time()),
            "artifacts": {
                "host_relay_sha256": HOST_RELAY_SHA256,
                "host_worker_sha256": HOST_WORKER_SHA256,
                "model_path": str(self.args.model.resolve()),
                "model_sha256": MODEL_SHA256,
                "op12_shard_sha256": OP12_SHARD_SHA256,
                "op15_relay_sha256": PHONE_RELAY_SHA256,
                "op15_shard_sha256": OP15_SHARD_SHA256,
                "phone_worker_sha256": PHONE_WORKER_SHA256,
            },
            "base_contract_sha256": self.contract.base_contract_sha256,
            "base_git_commit": self.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"]
            ).stdout.decode("ascii").strip(),
            "contract_sha256": self.contract.raw_sha256,
            "cuda": {
                "boot_id": host_boot,
                "cuda_device_order": "PCI_BUS_ID",
                "cuda_visible_devices": self.args.cuda_uuid,
                "driver_version": identity["driver_version"],
                "logical_device": "CUDA0",
                "memory_total_mib": identity["memory_total_mib"],
                "name": identity["name"],
                "pci_bus_id": identity["pci_bus_id"],
                "physical_index": identity["index"],
                "power_limit_mw": identity["power_limit_mw"],
                "uuid": identity["uuid"],
            },
            "delta_contract_sha256": self.contract.delta_contract_sha256,
            "model_sha256": MODEL_SHA256,
            "op12": {
                "adb_target": self.args.op12,
                "boot_id": op12_boot,
                "layers": [30, 48],
                "shard_sha256": OP12_SHARD_SHA256,
                "wifi": self.args.op12_wifi,
                "worker_sha256": PHONE_WORKER_SHA256,
            },
            "op15": {
                "adb_target": self.args.op15,
                "boot_id": op15_boot,
                "layers": [0, 30],
                "relay_sha256": PHONE_RELAY_SHA256,
                "shard_sha256": OP15_SHARD_SHA256,
                "wifi": self.args.op15_wifi,
                "worker_sha256": PHONE_WORKER_SHA256,
            },
            "pair_ordinal": ordinal,
            "physical_gate_sha256": self.contract.physical_gate_sha256,
            "run_id": run_id,
            "schema": "s39-profiled-cutover-pair-context-v1",
            "sources": self.contract.source_sha256,
            "w8_contract_sha256": self.contract.w8_contract_sha256,
        })

    def treatment(self, pair_dir: Path, ordinal: str, run_id: str) -> tuple[int, bool, bool]:
        directory = pair_dir / "treatment"
        directory.mkdir()
        self.warm_model(directory / "page_cache.json", f"{ordinal}.T")
        self.capture_gpu(
            directory / "gpu_pre.json",
            label=f"{ordinal}.T.PRE",
            idle=True,
        )
        self.write_pair_context(pair_dir, ordinal, run_id)
        self.start_phones(directory)
        stdout = (directory / "probe.stdout").open("xb")
        stderr = (directory / "probe.stderr").open("xb")
        probe = subprocess.Popen(
            [
                sys.executable,
                str(HERE / "phone_cuda_profiled_cutover_probe.py"),
                "--contract",
                str(self.args.contract),
                "--w8-contract",
                str(self.args.w8_contract),
                "--base-contract",
                str(self.args.base_contract),
                "--delta-contract",
                str(self.args.delta_contract),
                "--physical-gate",
                str(self.args.physical_gate),
                "--phone-route",
                f"{self.args.op15_wifi}:41525",
                "--cuda-route",
                "127.0.0.1:41512",
                "--run-id",
                run_id,
                "--pair-ordinal",
                ordinal,
                "--prepaid-ready-marker",
                str(directory / "prepaid_ready.json"),
                "--paid-permit",
                str(directory / "paid_permit.json"),
                "--start-marker",
                str(directory / "start.json"),
                "--ready-marker",
                str(directory / "ready.json"),
                "--ledger-dir",
                str(directory / "publication_ledger"),
                "--output",
                str(directory / "report.json"),
                "--timeout",
                "300",
            ],
            stdout=stdout,
            stderr=stderr,
        )
        stdout.close()
        stderr.close()
        self.wait_file(directory / "prepaid_ready.json", probe, timeout=180)
        self.capture_thermal_start(pair_dir, ordinal)
        w9.write_atomic(directory / "paid_permit.json", {
            "pair_ordinal": ordinal,
            "run_id": run_id,
            "schema": "s39-profiled-cutover-paid-permit-v1",
        })
        self.wait_file(directory / "start.json", probe, timeout=10)
        start, _ = w6.read_canonical(directory / "start.json", "start")
        self.write_launch(
            directory / "launch.json",
            phase="TREATMENT",
            ordinal=ordinal,
            run_id=run_id,
            request_start_ns=start["request_start_ns"],
            deadline_ns=start["launch_deadline_ns"],
        )
        route = self.start_host_route(directory, ports=(41510, 41511, 41512))
        self.wait_file(directory / "ready.json", probe, timeout=120)
        self.capture_gpu(
            directory / "gpu_ready.json",
            label=f"{ordinal}.T.READY",
            idle=False,
        )
        try:
            probe_rc = probe.wait(timeout=300)
        except subprocess.TimeoutExpired:
            probe.terminate()
            probe.wait()
            probe_rc = 2
        host_ok = self.wait_processes(route)
        phone_ok = all((
            self.wait_remote_exit(self.args.op15, "w9_relay.pid"),
            self.wait_remote_exit(self.args.op15, "w9_head.pid"),
            self.wait_remote_exit(self.args.op12, "w9_tail.pid"),
        ))
        self.collect_phone_logs(directory)
        self.capture_thermal(
            directory / "thermal_post.json",
            f"{ordinal}.T.POST",
        )
        self.capture_gpu(
            directory / "gpu_post.json",
            label=f"{ordinal}.T.POST",
            idle=False,
            post=True,
        )
        self.remote_started = False
        self.host_processes.clear()
        return probe_rc, host_ok, phone_ok

    def control(self, pair_dir: Path, ordinal: str, run_id: str) -> tuple[int, bool]:
        directory = pair_dir / "control"
        directory.mkdir()
        self.warm_model(directory / "page_cache.json", f"{ordinal}.C")
        self.capture_gpu(
            directory / "gpu_pre.json",
            label=f"{ordinal}.C.PRE",
            idle=True,
        )
        request_start_ns = time.monotonic_ns()
        deadline_ns = request_start_ns + 100_000_000
        w9.write_atomic(directory / "start.json", {
            "launch_deadline_ns": deadline_ns,
            "pair_ordinal": ordinal,
            "request_start_ns": request_start_ns,
            "run_id": run_id,
            "schema": "s39-profiled-cutover-control-start-v1",
        })
        self.write_launch(
            directory / "launch.json",
            phase="CONTROL",
            ordinal=ordinal,
            run_id=run_id,
            request_start_ns=request_start_ns,
            deadline_ns=deadline_ns,
        )
        stdout = (directory / "probe.stdout").open("xb")
        stderr = (directory / "probe.stderr").open("xb")
        probe = subprocess.Popen(
            [
                sys.executable,
                str(HERE / "cuda_profiled_trace_control_probe.py"),
                "--contract",
                str(self.args.contract),
                "--w8-contract",
                str(self.args.w8_contract),
                "--base-contract",
                str(self.args.base_contract),
                "--delta-contract",
                str(self.args.delta_contract),
                "--physical-gate",
                str(self.args.physical_gate),
                "--treatment-report",
                str(pair_dir / "treatment/report.json"),
                "--launch-record",
                str(directory / "launch.json"),
                "--cuda-route",
                "127.0.0.1:41612",
                "--run-id",
                run_id,
                "--pair-ordinal",
                ordinal,
                "--ready-marker",
                str(directory / "ready.json"),
                "--output",
                str(directory / "report.json"),
                "--timeout",
                "300",
            ],
            stdout=stdout,
            stderr=stderr,
        )
        stdout.close()
        stderr.close()
        route = self.start_host_route(directory, ports=(41610, 41611, 41612))
        self.wait_file(directory / "ready.json", probe, timeout=120)
        self.capture_gpu(
            directory / "gpu_ready.json",
            label=f"{ordinal}.C.READY",
            idle=False,
        )
        try:
            probe_rc = probe.wait(timeout=300)
        except subprocess.TimeoutExpired:
            probe.terminate()
            probe.wait()
            probe_rc = 2
        host_ok = self.wait_processes(route)
        self.capture_gpu(
            directory / "gpu_post.json",
            label=f"{ordinal}.C.POST",
            idle=False,
            post=True,
        )
        self.host_processes.clear()
        return probe_rc, host_ok

    def run_validator(self, pair_dir: Path, output: Path) -> int:
        process = self.run(
            [
                sys.executable,
                str(HERE / "validate_profiled_cutover_pair.py"),
                "--pair-dir",
                str(pair_dir),
                "--contract",
                str(self.args.contract),
                "--w8-contract",
                str(self.args.w8_contract),
                "--base-contract",
                str(self.args.base_contract),
                "--delta-contract",
                str(self.args.delta_contract),
                "--physical-gate",
                str(self.args.physical_gate),
                "--output",
                str(output),
            ],
            check=False,
            timeout=120,
        )
        with (pair_dir / f"{output.stem}.stdout").open("xb") as stream:
            stream.write(process.stdout)
        with (pair_dir / f"{output.stem}.stderr").open("xb") as stream:
            stream.write(process.stderr)
        return process.returncode

    def hash_tree(self, directory: Path) -> None:
        manifest = directory / "SHA256SUMS.txt"
        w9.require(not manifest.exists(), f"manifest already exists: {manifest}")
        output = []
        for path in sorted(directory.rglob("*")):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.name != "SHA256SUMS.txt"
            ):
                output.append(
                    f"{self.file_sha256(path)}  "
                    f"{path.relative_to(directory).as_posix()}\n"
                )
        with manifest.open("xb") as stream:
            stream.write("".join(output).encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def verify_manifest(self, directory: Path) -> None:
        for line in (directory / "SHA256SUMS.txt").read_text(
            encoding="ascii"
        ).splitlines():
            digest, relative = line.split("  ", 1)
            path = directory / relative
            w9.require(
                path.is_file() and self.file_sha256(path) == digest,
                f"manifest mismatch: {path}",
            )

    def run_pair(self, ordinal: str) -> None:
        pair_dir = self.out / ordinal
        pair_dir.mkdir()
        run_id = secrets.token_hex(32)
        treatment_rc = 2
        control_rc = 2
        host_ok = False
        phone_ok = False
        try:
            treatment_rc, host_ok, phone_ok = self.treatment(
                pair_dir,
                ordinal,
                run_id,
            )
            w9.require(
                treatment_rc == 0 and host_ok and phone_ok,
                f"{ordinal}: paid treatment failed",
            )
            control_rc, control_host_ok = self.control(pair_dir, ordinal, run_id)
            host_ok = host_ok and control_host_ok
            w9.require(
                control_rc == 0 and host_ok,
                f"{ordinal}: paid control failed",
            )
            w9.write_atomic(pair_dir / "EXIT_CODES.json", {
                "control_probe_rc": control_rc,
                "host_exit_ok": host_ok,
                "phone_exit_ok": phone_ok,
                "schema": "s39-profiled-cutover-pair-exit-v1",
                "treatment_probe_rc": treatment_rc,
            })
            first = pair_dir / "pair_certificate.json"
            second = pair_dir / "pair_certificate.recomputed.json"
            w9.require(
                self.run_validator(pair_dir, first) == 0,
                f"{ordinal}: validator failed",
            )
            w9.require(
                self.run_validator(pair_dir, second) == 0,
                f"{ordinal}: validator regeneration failed",
            )
            w9.require(
                first.read_bytes() == second.read_bytes(),
                f"{ordinal}: validator output is not byte-identical",
            )
            self.hash_tree(pair_dir)
            self.verify_manifest(pair_dir)
            self.completed.append(ordinal)
        finally:
            self.cleanup_host()
            self.cleanup_remote()

    def run_series_validator(self, output: Path) -> int:
        process = self.run(
            [
                sys.executable,
                str(HERE / "validate_profiled_cutover_series.py"),
                "--series-dir",
                str(self.out),
                "--contract",
                str(self.args.contract),
                "--w8-contract",
                str(self.args.w8_contract),
                "--base-contract",
                str(self.args.base_contract),
                "--delta-contract",
                str(self.args.delta_contract),
                "--physical-gate",
                str(self.args.physical_gate),
                "--output",
                str(output),
            ],
            check=False,
            timeout=180,
        )
        with (self.out / f"{output.stem}.stdout").open("xb") as stream:
            stream.write(process.stdout)
        with (self.out / f"{output.stem}.stderr").open("xb") as stream:
            stream.write(process.stderr)
        return process.returncode

    def acquire(self) -> None:
        self.preflight()
        self.write_campaign_context()
        for ordinal in self.contract.pair_ordinals:
            self.run_pair(ordinal)
        first = self.out / "series_certificate.json"
        second = self.out / "series_certificate.recomputed.json"
        w9.require(
            self.run_series_validator(first) == 0,
            "series validator failed",
        )
        w9.require(
            self.run_series_validator(second) == 0,
            "series validator regeneration failed",
        )
        w9.require(
            first.read_bytes() == second.read_bytes(),
            "series certificate is not byte-identical",
        )
        if not (self.out / "SHA256SUMS.txt").exists():
            self.hash_tree(self.out)
        self.verify_manifest(self.out)

    def fail(self, error: BaseException) -> None:
        self.cleanup_host()
        self.cleanup_remote()
        path = self.out / "FAILURE.json"
        if not path.exists():
            w9.write_atomic(path, {
                "completed_pair_ordinals": self.completed,
                "error": str(error),
                "paid_pair_ordinals": [
                    ordinal
                    for ordinal in self.contract.pair_ordinals
                    if (self.out / ordinal / "treatment/start.json").is_file()
                ],
                "schema": "s39-profiled-cutover-campaign-failure-v1",
                "status": "FAIL",
            })
        if not (self.out / "SHA256SUMS.txt").exists():
            self.hash_tree(self.out)


def defaults(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--contract",
        type=Path,
        default=HERE / "W9_PROFILED_CUTOVER_CONTRACT.json",
    )
    parser.add_argument(
        "--w8-contract",
        type=Path,
        default=HERE / "W8_LIVE_SESSION_CONTRACT_R1.json",
    )
    parser.add_argument(
        "--base-contract",
        type=Path,
        default=HERE / "W5_HANDOFF_CONTRACT.json",
    )
    parser.add_argument(
        "--delta-contract",
        type=Path,
        default=HERE / "W6_DELTA_CONTRACT.json",
    )
    parser.add_argument(
        "--physical-gate",
        type=Path,
        default=HERE / "W6_PHYSICAL_GATE.json",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/home/myid/zs89458/Documents/models/"
            "Qwen2.5-14B-Instruct-Q8_0.gguf"
        ),
    )
    parser.add_argument(
        "--host-worker",
        type=Path,
        default=ROOT / "build-cuda/bin/llama-layersplit",
    )
    parser.add_argument(
        "--host-relay",
        type=Path,
        default=ROOT / "build-cuda/bin/llama-stage-direct-relay",
    )
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--adb-port", type=int, default=5038)
    parser.add_argument("--op15", default="3C15AU002CL00000")
    parser.add_argument("--op12", default="5ae7a43d")
    parser.add_argument("--op15-wifi", default="172.20.173.218")
    parser.add_argument("--op12-wifi", default="172.20.59.72")
    parser.add_argument(
        "--cuda-uuid",
        default="GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f",
    )
    parser.add_argument("--thermal-wait-s", type=int, default=600)


def main() -> int:
    parser = argparse.ArgumentParser()
    defaults(parser)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not 1 <= args.adb_port <= 65535
        or args.thermal_wait_s <= 0
    ):
        parser.error("invalid arguments or existing output")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_uuid
    campaign = Campaign(args)
    try:
        campaign.acquire()
        print(f"W9_PROFILED_CUTOVER_GATE=PASS output={args.output}")
        return 0
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        w6.DeltaError,
        w9.W9Error,
    ) as exc:
        campaign.fail(exc)
        print(
            f"W9_PROFILED_CUTOVER_GATE=FAIL error={exc} output={args.output}",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
