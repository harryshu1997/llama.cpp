#!/usr/bin/env python3
"""Resident GPU/CPU adapter backed by native llama-server slots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from phone_gateway import (
    COMMAND_CLEANUP,
    COMMAND_DISCARD,
    COMMAND_DRAIN,
    COMMAND_EXECUTE,
    COMMAND_LOAD,
    COMMAND_REPLAY,
    COMMAND_UNLOAD,
    GatewayError,
    MAX_COMMAND_BYTES,
    canonical_bytes,
    exact_keys,
    integer,
    make_result,
    parse_command,
    require,
    sha256_text,
    strict_json_loads,
    string,
)
from runtime_binding import (
    ControllerAuthenticator,
    RuntimeBinding,
    RuntimeBindingError,
    await_runtime_binding,
    process_start_time_ticks,
)
from readiness_v23 import (
    absolute_path,
    local_stat,
    parse_artifact_certificate,
    parse_readiness_lock,
)


@dataclass(frozen=True)
class DesktopRouteSpec:
    model_id: str
    native_model_id: str
    model_sha256: str
    model_path: str
    artifact_certificate_sha256: str
    model_stat: dict[str, int]
    backend: str
    child_argv_template: tuple[str, ...]
    child_executable: ExecutableIdentity
    device_uuid: str
    device_name: str
    device_memory_total_mib: int
    host_boot_id: str
    minimum_free_device_memory_mib: int
    n_gpu_layers: str
    phase: str
    phase_lock_sha256: str
    qualification: dict[str, Any]
    readiness_lock_sha256: str
    readiness_phase_id: str
    slot_save_path: str
    slots: tuple[int, ...]


@dataclass(frozen=True)
class ExecutableIdentity:
    path: str
    bytes: int
    sha256: str


def resolve_executor_bundle(
    *,
    executor_bundle_manifest_path: Path | None,
    executor_bundle_manifest_sha256: str | None,
) -> tuple[Path, str]:
    manifest = executor_bundle_manifest_path
    expected = executor_bundle_manifest_sha256
    if manifest is None:
        raw_manifest = os.environ.get("S40_EXECUTOR_BUNDLE_MANIFEST", "")
        require(raw_manifest, "desktop executor bundle manifest")
        manifest = Path(raw_manifest)
    if expected is None:
        expected = os.environ.get("S40_EXECUTOR_BUNDLE_SHA256", "")
    require(
        manifest.is_absolute()
        and manifest.name == "MANIFEST.json"
        and manifest.is_file(),
        "desktop executor bundle manifest",
    )
    expected = sha256_text(expected, "desktop executor bundle SHA-256")
    require(
        HostRuntimeProbe._file_sha256(manifest) == expected,
        "desktop executor bundle manifest changed",
    )
    return manifest, expected


def derive_route_qualification(
    value: Any,
    *,
    model_id: str,
    phase: str,
    slot: str,
    artifact_certificate_sha256: str,
    readiness_lock_sha256: str,
    executor_bundle_manifest_path: Path | None,
    executor_bundle_manifest_sha256: str | None,
) -> dict[str, Any]:
    from qualification_authority import validate_route_authority

    manifest, expected = resolve_executor_bundle(
        executor_bundle_manifest_path=executor_bundle_manifest_path,
        executor_bundle_manifest_sha256=executor_bundle_manifest_sha256,
    )
    return validate_route_authority(
        value,
        expected_model_id=model_id,
        expected_phase=phase,
        expected_slot=slot,
        expected_artifact_certificate_sha256=
            artifact_certificate_sha256,
        expected_readiness_lock_sha256=readiness_lock_sha256,
        executor_bundle=manifest.parent,
        executor_bundle_manifest_sha256=expected,
    )


def derive_desktop_smoke_qualification(
    value: Any,
    *,
    model_id: str,
    model_path: str,
    executor_bundle_manifest_path: Path | None,
    executor_bundle_manifest_sha256: str | None,
) -> dict[str, Any]:
    from qualification_authority import validate_desktop_smoke_authority

    manifest, expected = resolve_executor_bundle(
        executor_bundle_manifest_path=executor_bundle_manifest_path,
        executor_bundle_manifest_sha256=executor_bundle_manifest_sha256,
    )
    return validate_desktop_smoke_authority(
        value,
        expected_model_id=model_id,
        expected_model_path=model_path,
        executor_bundle=manifest.parent,
        executor_bundle_manifest_sha256=expected,
    )


@dataclass
class DesktopSession:
    model_id: str
    slot_id: int
    ownership_epoch: int
    logical_history: list[int]


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
    ) -> tuple[int, Any]:
        ...


class RuntimeProbe(Protocol):
    def inspect(
        self,
        spec: DesktopRouteSpec,
        process_id: int,
        model_path: str,
        port: int,
    ) -> dict[str, Any]:
        ...

    def process_exited(
        self,
        process_id: int,
        process_start_ticks: int,
        timeout_s: float,
    ) -> bool:
        ...


class CacheController(Protocol):
    def prepare(
        self,
        spec: DesktopRouteSpec,
        regime: str,
    ) -> dict[str, Any]:
        ...


class SubprocessCacheController:
    def __init__(self, timeout_s: float):
        require(timeout_s > 0, "cache control timeout")
        self.timeout_s = timeout_s

    def prepare(
        self,
        spec: DesktopRouteSpec,
        regime: str,
    ) -> dict[str, Any]:
        runner = Path(__file__).resolve().parent / "cache_control_runner.py"
        flags = (
            ["-B", "-s", "-P"]
            if os.environ.get("S40_EXECUTOR_BUNDLE") == "1"
            else ["-B"]
        )
        argv = [
            sys.executable,
            *flags,
            str(runner),
            "--regime",
            regime,
            "--model",
            spec.model_path,
        ]
        started_ns = time.monotonic_ns()
        process = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_s,
            check=False,
        )
        completed_ns = time.monotonic_ns()
        require(
            process.returncode == 0
            and not process.stderr
            and 0 < len(process.stdout) <= MAX_COMMAND_BYTES,
            "desktop cache control process failed",
        )
        output = strict_json_loads(
            process.stdout,
            "cache control result",
        )
        require(
            canonical_bytes(output) == process.stdout
            and type(output) is dict
            and output.get("schema") == "s40-cache-control-result-v1"
            and output.get("success") is True
            and output.get("regime") == regime
            and output.get("model_path") == spec.model_path
            and output.get("model_stat") == spec.model_stat,
            "desktop cache control result binding",
        )
        return {
            "argv": argv,
            "completed_ns": completed_ns,
            "exit_code": process.returncode,
            "output": output,
            "schema": "s40-cache-control-evidence-v1",
            "started_ns": started_ns,
            "stderr": "",
            "success": True,
        }


class HostRuntimeProbe:
    def __init__(self, nvidia_smi: ExecutableIdentity):
        self.nvidia_smi = nvidia_smi
        self._validate_nvidia_smi()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    def _validate_nvidia_smi(self) -> dict[str, Any]:
        path = Path(self.nvidia_smi.path)
        require(
            path.is_absolute()
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_mode & 0o111 != 0,
            "nvidia-smi executable",
        )
        require(
            path.stat().st_size == self.nvidia_smi.bytes
            and self._file_sha256(path) == self.nvidia_smi.sha256,
            "nvidia-smi executable changed",
        )
        return {
            "bytes": self.nvidia_smi.bytes,
            "path": self.nvidia_smi.path,
            "sha256": self.nvidia_smi.sha256,
        }

    @classmethod
    def _validate_executable(
        cls,
        executable: ExecutableIdentity,
    ) -> dict[str, Any]:
        path = Path(executable.path)
        require(
            path.is_absolute()
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_mode & 0o111 != 0,
            "child executable",
        )
        require(
            path.stat().st_size == executable.bytes
            and cls._file_sha256(path) == executable.sha256,
            "child executable changed",
        )
        return {
            "bytes": executable.bytes,
            "path": executable.path,
            "sha256": executable.sha256,
        }

    @staticmethod
    def _process_start_ticks(process_id: int) -> int:
        raw = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
        closing = raw.rfind(")")
        require(closing > 0, "child process stat framing")
        fields = raw[closing + 2:].split()
        require(len(fields) > 19, "child process stat fields")
        try:
            start_ticks = int(fields[19])
        except ValueError as error:
            raise GatewayError(
                f"child process start ticks: {error}"
            ) from error
        require(start_ticks > 0, "child process start ticks")
        return start_ticks

    @staticmethod
    def _flag_value(argv: list[str], names: tuple[str, ...], field: str) -> str:
        values = []
        for index, argument in enumerate(argv):
            if argument in names:
                require(index + 1 < len(argv), f"child {field} flag")
                values.append(argv[index + 1])
            for name in names:
                prefix = name + "="
                if argument.startswith(prefix):
                    values.append(argument[len(prefix):])
        require(len(values) == 1, f"child {field} flag")
        return values[0]

    def inspect(
        self,
        spec: DesktopRouteSpec,
        process_id: int,
        model_path: str,
        port: int,
    ) -> dict[str, Any]:
        nvidia_smi = self._validate_nvidia_smi()
        child_executable = self._validate_executable(spec.child_executable)
        require(process_id > 0, "child process id")
        require(0 < port <= 65535, "child process port")
        path = Path(model_path)
        require(path.is_absolute() and path.is_file(), "child model path")
        require(
            path.resolve() == Path(spec.model_path).resolve(),
            "child model path changed",
        )
        require(
            local_stat(path) == spec.model_stat,
            "child model stat differs from the artifact certificate",
        )
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
        require(boot_id == spec.host_boot_id, "desktop boot identity changed")
        cmdline_path = Path(f"/proc/{process_id}/cmdline")
        raw_cmdline = cmdline_path.read_bytes()
        require(raw_cmdline.endswith(b"\0"), "child command line is truncated")
        try:
            argv = [
                part.decode("utf-8")
                for part in raw_cmdline[:-1].split(b"\0")
            ]
        except UnicodeDecodeError as error:
            raise GatewayError(f"child command line encoding: {error}") from error
        expected_argv = [
            str(port) if argument == "{PORT}" else argument
            for argument in spec.child_argv_template
        ]
        require(
            argv == expected_argv,
            "child command line differs from the frozen serving template",
        )
        require(
            self._validate_executable(spec.child_executable)
            == child_executable,
            "child executable changed during the probe",
        )
        model_argument = self._flag_value(
            argv,
            ("-m", "--model"),
            "model",
        )
        require(
            Path(model_argument).resolve() == path.resolve(),
            "child command line model path mismatch",
        )
        n_gpu_layers = self._flag_value(
            argv,
            ("-ngl", "--n-gpu-layers"),
            "GPU layers",
        )
        require(
            n_gpu_layers == spec.n_gpu_layers,
            "child GPU layer allocation mismatch",
        )
        slot_save_path = self._flag_value(
            argv,
            ("--slot-save-path",),
            "slot save path",
        )
        require(
            Path(slot_save_path).resolve()
            == Path(spec.slot_save_path).resolve(),
            "child slot save path mismatch",
        )
        process_start_ticks = self._process_start_ticks(process_id)
        processes = subprocess.run(
            [
                self.nvidia_smi.path,
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
        require(
            processes.returncode == 0 and not processes.stderr,
            "nvidia-smi process query failed",
        )
        matches = []
        for line in processes.stdout.decode("ascii").splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 4 and fields[1].isdigit() and int(fields[1]) == process_id:
                matches.append(fields)
        if spec.backend == "CPU":
            require(not matches, "CPU child is bound to a GPU")
            require(spec.device_uuid == "NONE", "CPU device UUID must be NONE")
            gpu_uuid = None
            gpu_name = None
            memory_total_mib = None
            memory_free_mib = None
        else:
            require(spec.backend == "CUDA", "unsupported desktop backend")
            require(len(matches) == 1, "GPU child process binding is ambiguous")
            require(matches[0][0] == spec.device_uuid, "GPU UUID mismatch")
            gpu_uuid = matches[0][0]
            device = subprocess.run(
                [
                    self.nvidia_smi.path,
                    "--query-gpu=uuid,name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                    "--id=" + spec.device_uuid,
                ],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            )
            require(
                device.returncode == 0 and not device.stderr,
                "nvidia-smi device query failed",
            )
            rows = [
                [field.strip() for field in line.split(",")]
                for line in device.stdout.decode("ascii").splitlines()
                if line.strip()
            ]
            require(
                len(rows) == 1
                and len(rows[0]) == 4
                and rows[0][2].isdigit()
                and rows[0][3].isdigit(),
                "GPU device identity is malformed",
            )
            gpu_uuid, gpu_name = rows[0][0], rows[0][1]
            memory_total_mib = int(rows[0][2])
            memory_free_mib = int(rows[0][3])
            require(gpu_uuid == spec.device_uuid, "GPU device UUID changed")
            require(gpu_name == spec.device_name, "GPU device name changed")
            require(
                memory_total_mib == spec.device_memory_total_mib,
                "GPU device memory total changed",
            )
            require(
                memory_free_mib >= spec.minimum_free_device_memory_mib,
                "GPU device headroom is below the frozen bound",
            )
        require(
            self._validate_nvidia_smi() == nvidia_smi,
            "nvidia-smi identity changed during the probe",
        )
        result = {
            "argv": argv,
            "backend": spec.backend,
            "device_uuid": gpu_uuid,
            "device_name": gpu_name,
            "device_memory_free_mib": memory_free_mib,
            "device_memory_total_mib": memory_total_mib,
            "host_boot_id": boot_id,
            "child_argv_sha256": hashlib.sha256(raw_cmdline).hexdigest(),
            "child_executable": child_executable,
            "model_path": model_path,
            "model_sha256": spec.model_sha256,
            "minimum_free_device_memory_mib":
                spec.minimum_free_device_memory_mib,
            "native_model_id": spec.native_model_id,
            "nvidia_smi": nvidia_smi,
            "n_gpu_layers": n_gpu_layers,
            "process_id": process_id,
            "process_start_ticks": process_start_ticks,
            "port": port,
            "qualification": spec.qualification,
            "serving_envelope": {
                "batch_size": 2048,
                "cache_type_k": "f16",
                "cache_type_v": "f16",
                "continuous_batching": True,
                "context_size": 4096,
                "flash_attention": "on",
                "parallel_slots": 8,
                "split_mode": "none",
                "ubatch_size": 512,
            },
            "slot_save_path": slot_save_path,
        }
        if spec.phase == "DESKTOP_SMOKE":
            result["schema"] = "s40-desktop-smoke-runtime-probe-v1"
        else:
            result.update({
                "artifact_certificate_sha256":
                    spec.artifact_certificate_sha256,
                "readiness_lock_sha256": spec.readiness_lock_sha256,
                "readiness_phase_id": spec.readiness_phase_id,
                "schema": "s40-desktop-runtime-probe-v3",
            })
        return result

    def process_exited(
        self,
        process_id: int,
        process_start_ticks: int,
        timeout_s: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while True:
            path = Path(f"/proc/{process_id}/stat")
            if not path.exists():
                return True
            try:
                current = self._process_start_ticks(process_id)
            except FileNotFoundError:
                return True
            if current != process_start_ticks:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)


SERVING_ENVELOPE = {
    "batch_size": 2048,
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "continuous_batching": True,
    "context_size": 4096,
    "flash_attention": "on",
    "parallel_slots": 8,
    "split_mode": "none",
    "ubatch_size": 512,
}

SERVING_FORBIDDEN_FLAGS = (
    "--ui-mcp-proxy",
    "--ui_mcp_proxy",
    "--webui-mcp-proxy",
    "--webui_mcp_proxy",
    "--tools",
    "-ag",
    "--agent",
)


def validate_child_argv_template(
    value: Any,
    executable: ExecutableIdentity,
    model_path: str,
    n_gpu_layers: str,
    slot_save_path: str,
) -> tuple[str, ...]:
    require(
        type(value) is list and 16 <= len(value) <= 256,
        "desktop child argv template",
    )
    argv = []
    for index, argument in enumerate(value):
        argv.append(string(argument, f"desktop child argv[{index}]"))
    require(
        not any(
            argument in SERVING_FORBIDDEN_FLAGS
            or any(
                argument.startswith(flag + "=")
                for flag in SERVING_FORBIDDEN_FLAGS
            )
            for argument in argv
        ),
        "desktop child forbidden server feature",
    )
    require(
        argv.count("{PORT}") == 1
        and argv[0] == executable.path,
        "desktop child executable and port template",
    )

    def flag(names: tuple[str, ...], field: str) -> str:
        values = []
        for index, argument in enumerate(argv):
            if argument in names:
                require(index + 1 < len(argv), f"desktop child {field}")
                values.append(argv[index + 1])
            for name in names:
                prefix = name + "="
                if argument.startswith(prefix):
                    values.append(argument[len(prefix):])
        require(len(values) == 1, f"desktop child {field}")
        return values[0]

    require(
        flag(("-m", "--model"), "model") == model_path
        and flag(("-ngl", "--n-gpu-layers"), "GPU layers")
        == n_gpu_layers
        and flag(("--slot-save-path",), "slot save path")
        == slot_save_path
        and flag(("--port",), "port") == "{PORT}"
        and flag(("-c", "--ctx-size"), "context size") == "4096"
        and flag(("-b", "--batch-size"), "batch size") == "2048"
        and flag(("-ub", "--ubatch-size"), "ubatch size") == "512"
        and flag(("-np", "--parallel"), "parallel slots") == "8"
        and flag(("-sm", "--split-mode"), "split mode") == "none"
        and flag(("--cache-type-k",), "K cache type") == "f16"
        and flag(("--cache-type-v",), "V cache type") == "f16"
        and flag(("--flash-attn",), "flash attention") == "on"
        and argv.count("--cont-batching") == 1
        and "--no-cont-batching" not in argv,
        "desktop child serving envelope",
    )
    return tuple(argv)


class UrllibTransport:
    def __init__(self, base_url: str, timeout_s: float):
        parsed = urllib.parse.urlsplit(base_url)
        require(
            parsed.scheme == "http"
            and parsed.hostname in ("127.0.0.1", "localhost")
            and parsed.path in ("", "/")
            and not parsed.query
            and not parsed.fragment,
            "desktop executor URL must be loopback HTTP",
        )
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
    ) -> tuple[int, Any]:
        raw = None if body is None else canonical_bytes(body)
        request = urllib.request.Request(
            self.base_url + path,
            data=raw,
            method=method,
            headers={
                "Content-Type": "application/json",
                "X-Llama-Warm-Tier-Internal": "1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                output = response.read(MAX_COMMAND_BYTES + 1)
                status = response.status
        except urllib.error.HTTPError as error:
            output = error.read(MAX_COMMAND_BYTES + 1)
            status = error.code
        require(len(output) <= MAX_COMMAND_BYTES, "HTTP response too large")
        value = strict_json_loads(output, "HTTP response JSON", "utf-8")
        return status, value


def parse_desktop_config(
    path: Path,
    *,
    executor_bundle_manifest_path: Path | None = None,
    executor_bundle_manifest_sha256: str | None = None,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    dict[str, DesktopRouteSpec],
    str | None,
    ExecutableIdentity,
    str,
]:
    raw = path.read_bytes()
    require(0 < len(raw) <= MAX_COMMAND_BYTES, "desktop config size")
    value = strict_json_loads(raw, "desktop config JSON")
    require(canonical_bytes(value) == raw, "desktop config is not canonical JSON")
    value = exact_keys(
        value,
        {
            "base_url",
            "cache_regime",
            "executor_id",
            "mode",
            "nvidia_smi",
            "profile_lock_sha256",
            "role",
            "routes",
            "schema",
        },
        "desktop_config",
    )
    require(
        value["schema"] == "s40-desktop-executor-config-v4",
        "desktop config schema",
    )
    nvidia_smi_value = exact_keys(
        value["nvidia_smi"],
        {"bytes", "path", "sha256"},
        "desktop_config.nvidia_smi",
    )
    nvidia_smi = ExecutableIdentity(
        path=absolute_path(
            nvidia_smi_value["path"],
            "desktop nvidia-smi path",
        ),
        bytes=integer(
            nvidia_smi_value["bytes"],
            "desktop nvidia-smi bytes",
            1,
        ),
        sha256=sha256_text(
            nvidia_smi_value["sha256"],
            "desktop nvidia-smi SHA-256",
        ),
    )
    HostRuntimeProbe(nvidia_smi)
    executor_id = string(value["executor_id"], "desktop_config.executor_id")
    role = string(value["role"], "desktop_config.role")
    require(role in ("GPU", "CPU"), "desktop executor role")
    mode = string(value["mode"], "desktop_config.mode")
    require(
        mode in ("SINGLE_ACTIVE", "DUAL_STATIC_PARTIAL"),
        "desktop executor mode",
    )
    base_url = string(value["base_url"], "desktop_config.base_url")
    cache_regime = string(
        value["cache_regime"],
        "desktop_config.cache_regime",
    )
    require(
        cache_regime in ("WARM_CACHE", "COLD_NVME"),
        "desktop cache regime",
    )
    routes = value["routes"]
    require(type(routes) is list and 1 <= len(routes) <= 8, "desktop routes")
    result: dict[str, DesktopRouteSpec] = {}
    for index, item in enumerate(routes):
        item = exact_keys(
            item,
            {
                "artifact_certificate_path",
                "artifact_certificate_sha256",
                "backend",
                "child_argv_template",
                "child_executable",
                "device_memory_total_mib",
                "device_name",
                "device_uuid",
                "host_boot_id",
                "minimum_free_device_memory_mib",
                "model_id",
                "model_path",
                "model_sha256",
                "native_model_id",
                "n_gpu_layers",
                "phase",
                "phase_lock_sha256",
                "qualification",
                "readiness_lock_path",
                "readiness_lock_sha256",
                "route_lock_sha256",
                "slot",
                "slot_save_path",
                "slots",
            },
            f"desktop_config.routes[{index}]",
        )
        model_id = string(item["model_id"], "desktop route model")
        require(model_id not in result, "duplicate desktop route")
        native_model_id = string(
            item["native_model_id"],
            "desktop native model",
        )
        slots = item["slots"]
        require(type(slots) is list and 1 <= len(slots) <= 128, "desktop slots")
        slot_ids = tuple(integer(slot, "desktop slot") for slot in slots)
        require(len(set(slot_ids)) == len(slot_ids), "duplicate desktop slot")
        model_path = absolute_path(
            item["model_path"],
            "desktop route model path",
        )
        certificate_sha256 = sha256_text(
            item["artifact_certificate_sha256"],
            "desktop artifact certificate SHA-256",
        )
        phase = string(item["phase"], "desktop route phase")
        require(phase in ("A_ONLY", "B_ONLY"), "desktop route phase")
        slot = string(item["slot"], "desktop route slot")
        require(slot in ("A", "B"), "desktop route slot")
        route_lock_sha256 = sha256_text(
            item["route_lock_sha256"],
            "desktop route lock SHA-256",
        )
        phase_lock_sha256 = sha256_text(
            item["phase_lock_sha256"],
            "desktop phase lock SHA-256",
        )
        certificate = parse_artifact_certificate(
            Path(
                absolute_path(
                    item["artifact_certificate_path"],
                    "desktop artifact certificate path",
                )
            ),
            certificate_sha256,
            model_id,
            phase,
            slot,
            route_lock_sha256,
        )
        readiness_lock_sha256 = sha256_text(
            item["readiness_lock_sha256"],
            "desktop readiness lock SHA-256",
        )
        readiness_lock = parse_readiness_lock(
            Path(
                absolute_path(
                    item["readiness_lock_path"],
                    "desktop readiness lock path",
                )
            ),
            readiness_lock_sha256,
            phase,
            certificate,
            phase_lock_sha256,
        )
        qualification = derive_route_qualification(
            item["qualification"],
            model_id=model_id,
            phase=phase,
            slot=slot,
            artifact_certificate_sha256=certificate_sha256,
            readiness_lock_sha256=readiness_lock_sha256,
            executor_bundle_manifest_path=
                executor_bundle_manifest_path,
            executor_bundle_manifest_sha256=
                executor_bundle_manifest_sha256,
        )
        require(
            qualification["phase_id"] == readiness_lock["phase_id"]
            and qualification["phase_lock_sha256"] == phase_lock_sha256,
            "desktop qualification readiness roots",
        )
        certificate_key = ("cuda", model_path)
        require(
            certificate_key in certificate["artifacts"],
            "desktop model is absent from artifact certificate",
        )
        certificate_row = certificate["artifacts"][certificate_key]
        model_sha256 = sha256_text(
            item["model_sha256"],
            "desktop route model SHA-256",
        )
        require(
            certificate_row["sha256"] == model_sha256,
            "desktop model certificate digest mismatch",
        )
        child_executable_value = exact_keys(
            item["child_executable"],
            {"bytes", "path", "sha256"},
            "desktop child executable",
        )
        child_executable = ExecutableIdentity(
            path=absolute_path(
                child_executable_value["path"],
                "desktop child executable path",
            ),
            bytes=integer(
                child_executable_value["bytes"],
                "desktop child executable bytes",
                1,
            ),
            sha256=sha256_text(
                child_executable_value["sha256"],
                "desktop child executable SHA-256",
            ),
        )
        HostRuntimeProbe._validate_executable(child_executable)
        n_gpu_layers = string(
            item["n_gpu_layers"],
            "desktop route GPU layers",
        )
        slot_save_path = absolute_path(
            item["slot_save_path"],
            "desktop route slot save path",
        )
        child_argv_template = validate_child_argv_template(
            item["child_argv_template"],
            child_executable,
            model_path,
            n_gpu_layers,
            slot_save_path,
        )
        spec = DesktopRouteSpec(
            model_id=model_id,
            native_model_id=native_model_id,
            model_sha256=model_sha256,
            model_path=model_path,
            artifact_certificate_sha256=certificate_sha256,
            model_stat=certificate_row["stat"],
            backend=string(item["backend"], "desktop route backend"),
            child_argv_template=child_argv_template,
            child_executable=child_executable,
            device_uuid=string(item["device_uuid"], "desktop route device UUID"),
            device_name=string(item["device_name"], "desktop route device name"),
            device_memory_total_mib=integer(
                item["device_memory_total_mib"],
                "desktop route device memory total",
            ),
            host_boot_id=string(
                item["host_boot_id"],
                "desktop route host boot ID",
            ),
            minimum_free_device_memory_mib=integer(
                item["minimum_free_device_memory_mib"],
                "desktop route minimum free device memory",
            ),
            n_gpu_layers=n_gpu_layers,
            phase=phase,
            phase_lock_sha256=phase_lock_sha256,
            qualification=qualification,
            readiness_lock_sha256=readiness_lock_sha256,
            readiness_phase_id=readiness_lock["phase_id"],
            slot_save_path=slot_save_path,
            slots=slot_ids,
        )
        require(
            Path(spec.slot_save_path).is_dir(),
            "desktop route slot save directory is missing",
        )
        require(
            (
                role == "GPU"
                and spec.backend == "CUDA"
                and spec.device_uuid.startswith("GPU-")
                and spec.device_name == "NVIDIA GeForce RTX 4060 Ti"
                and spec.device_memory_total_mib > 0
            )
            or (
                role == "CPU"
                and spec.backend == "CPU"
                and spec.device_uuid == "NONE"
                and spec.device_name == "NONE"
                and spec.device_memory_total_mib == 0
                and spec.minimum_free_device_memory_mib == 0
                and spec.n_gpu_layers == "0"
            ),
            "desktop route backend placement",
        )
        result[model_id] = spec
    require(
        len({spec.native_model_id for spec in result.values()}) == len(result),
        "duplicate desktop native model",
    )
    require(
        len({Path(spec.slot_save_path).resolve() for spec in result.values()})
        == len(result),
        "desktop routes share a slot save directory",
    )
    if mode == "DUAL_STATIC_PARTIAL":
        require(
            role == "GPU"
            and len(result) == 2
            and all(
                spec.minimum_free_device_memory_mib >= 512
                for spec in result.values()
            ),
            "partial route geometry",
        )
        profile_lock_sha256 = sha256_text(
            value["profile_lock_sha256"],
            "C3 profile lock SHA-256",
        )
    else:
        require(
            value["profile_lock_sha256"] is None,
            "non-C3 profile lock must be null",
        )
        profile_lock_sha256 = None
    return (
        executor_id,
        role,
        mode,
        base_url,
        cache_regime,
        result,
        profile_lock_sha256,
        nvidia_smi,
        hashlib.sha256(raw).hexdigest(),
    )


def parse_desktop_smoke_config(
    path: Path,
    *,
    executor_bundle_manifest_path: Path | None = None,
    executor_bundle_manifest_sha256: str | None = None,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    dict[str, DesktopRouteSpec],
    None,
    ExecutableIdentity,
    str,
]:
    raw = path.read_bytes()
    require(0 < len(raw) <= MAX_COMMAND_BYTES, "desktop smoke config size")
    value = strict_json_loads(raw, "desktop smoke config JSON")
    require(
        canonical_bytes(value) == raw,
        "desktop smoke config is not canonical JSON",
    )
    value = exact_keys(
        value,
        {
            "base_url",
            "cache_regime",
            "executor_id",
            "mode",
            "nvidia_smi",
            "profile_lock_sha256",
            "role",
            "routes",
            "schema",
        },
        "desktop_smoke_config",
    )
    require(
        value["schema"] == "s40-desktop-smoke-config-v1"
        and value["role"] == "GPU"
        and value["mode"] == "SINGLE_ACTIVE"
        and value["profile_lock_sha256"] is None,
        "desktop smoke config schema",
    )
    require(
        value["cache_regime"] in ("WARM_CACHE", "COLD_NVME"),
        "desktop smoke cache regime",
    )
    nvidia_value = exact_keys(
        value["nvidia_smi"],
        {"bytes", "path", "sha256"},
        "desktop smoke nvidia-smi",
    )
    nvidia_smi = ExecutableIdentity(
        path=absolute_path(
            nvidia_value["path"],
            "desktop smoke nvidia-smi path",
        ),
        bytes=integer(
            nvidia_value["bytes"],
            "desktop smoke nvidia-smi bytes",
            1,
        ),
        sha256=sha256_text(
            nvidia_value["sha256"],
            "desktop smoke nvidia-smi SHA-256",
        ),
    )
    HostRuntimeProbe(nvidia_smi)
    routes = value["routes"]
    require(type(routes) is list and 1 <= len(routes) <= 2, "desktop smoke routes")
    result = {}
    for index, item in enumerate(routes):
        item = exact_keys(
            item,
            {
                "backend",
                "child_argv_template",
                "child_executable",
                "device_memory_total_mib",
                "device_name",
                "device_uuid",
                "host_boot_id",
                "minimum_free_device_memory_mib",
                "model_id",
                "model_path",
                "model_sha256",
                "native_model_id",
                "n_gpu_layers",
                "qualification",
                "slot_save_path",
                "slots",
            },
            f"desktop_smoke_config.routes[{index}]",
        )
        model_id = string(item["model_id"], "desktop smoke model")
        require(model_id not in result, "duplicate desktop smoke model")
        model_path = absolute_path(
            item["model_path"],
            "desktop smoke model path",
        )
        require(
            Path(model_path).is_file()
            and not Path(model_path).is_symlink(),
            "desktop smoke model file",
        )
        model_sha256 = sha256_text(
            item["model_sha256"],
            "desktop smoke model SHA-256",
        )
        qualification = derive_desktop_smoke_qualification(
            item["qualification"],
            model_id=model_id,
            model_path=model_path,
            executor_bundle_manifest_path=
                executor_bundle_manifest_path,
            executor_bundle_manifest_sha256=
                executor_bundle_manifest_sha256,
        )
        require(
            item["qualification"]["model_sha256"] == model_sha256
            and item["qualification"]["gpu_uuid"] == item["device_uuid"],
            "desktop smoke qualification binding",
        )
        executable_value = exact_keys(
            item["child_executable"],
            {"bytes", "path", "sha256"},
            "desktop smoke child executable",
        )
        child_executable = ExecutableIdentity(
            path=absolute_path(
                executable_value["path"],
                "desktop smoke child executable path",
            ),
            bytes=integer(
                executable_value["bytes"],
                "desktop smoke child executable bytes",
                1,
            ),
            sha256=sha256_text(
                executable_value["sha256"],
                "desktop smoke child executable SHA-256",
            ),
        )
        HostRuntimeProbe._validate_executable(child_executable)
        n_gpu_layers = string(
            item["n_gpu_layers"],
            "desktop smoke GPU layers",
        )
        slot_save_path = absolute_path(
            item["slot_save_path"],
            "desktop smoke slot save path",
        )
        slots = item["slots"]
        require(
            type(slots) is list and 8 <= len(slots) <= 128,
            "desktop smoke slots",
        )
        slot_ids = tuple(
            integer(slot, "desktop smoke slot")
            for slot in slots
        )
        require(len(set(slot_ids)) == len(slot_ids), "duplicate smoke slot")
        spec = DesktopRouteSpec(
            model_id=model_id,
            native_model_id=string(
                item["native_model_id"],
                "desktop smoke native model",
            ),
            model_sha256=model_sha256,
            model_path=model_path,
            artifact_certificate_sha256="",
            model_stat=local_stat(Path(model_path)),
            backend=string(item["backend"], "desktop smoke backend"),
            child_argv_template=validate_child_argv_template(
                item["child_argv_template"],
                child_executable,
                model_path,
                n_gpu_layers,
                slot_save_path,
            ),
            child_executable=child_executable,
            device_uuid=string(
                item["device_uuid"],
                "desktop smoke device UUID",
            ),
            device_name=string(
                item["device_name"],
                "desktop smoke device name",
            ),
            device_memory_total_mib=integer(
                item["device_memory_total_mib"],
                "desktop smoke device memory",
                1,
            ),
            host_boot_id=string(
                item["host_boot_id"],
                "desktop smoke host boot ID",
            ),
            minimum_free_device_memory_mib=integer(
                item["minimum_free_device_memory_mib"],
                "desktop smoke free device memory",
            ),
            n_gpu_layers=n_gpu_layers,
            phase="DESKTOP_SMOKE",
            phase_lock_sha256="",
            qualification=qualification,
            readiness_lock_sha256="",
            readiness_phase_id=qualification["phase_id"],
            slot_save_path=slot_save_path,
            slots=slot_ids,
        )
        require(
            spec.backend == "CUDA"
            and spec.device_uuid.startswith("GPU-")
            and spec.device_name == "NVIDIA GeForce RTX 4060 Ti"
            and Path(spec.slot_save_path).is_dir(),
            "desktop smoke GPU placement",
        )
        result[model_id] = spec
    return (
        string(value["executor_id"], "desktop smoke executor ID"),
        "GPU",
        "SINGLE_ACTIVE",
        string(value["base_url"], "desktop smoke base URL"),
        value["cache_regime"],
        result,
        None,
        nvidia_smi,
        hashlib.sha256(raw).hexdigest(),
    )


class DesktopExecutor:
    def __init__(
        self,
        executor_id: str,
        role: str,
        mode: str,
        routes: dict[str, DesktopRouteSpec],
        transport: HttpTransport,
        initial_model_ids: tuple[str, ...],
        cache_controller: CacheController,
        cache_regime: str,
        lifecycle_timeout_s: float = 3600.0,
        runtime_probe: RuntimeProbe | None = None,
        nvidia_smi: ExecutableIdentity | None = None,
    ):
        require(role in ("GPU", "CPU"), "desktop role")
        require(
            mode in ("SINGLE_ACTIVE", "DUAL_STATIC_PARTIAL"),
            "desktop mode",
        )
        require(lifecycle_timeout_s > 0, "desktop lifecycle timeout")
        self.executor_id = executor_id
        self.role = role
        self.mode = mode
        self.routes = dict(routes)
        self.transport = transport
        self.cache_controller = cache_controller
        self.cache_regime = cache_regime
        require(
            cache_regime in ("WARM_CACHE", "COLD_NVME"),
            "desktop cache regime",
        )
        require(
            runtime_probe is not None or nvidia_smi is not None,
            "desktop runtime probe identity",
        )
        self.runtime_probe = (
            runtime_probe
            if runtime_probe is not None
            else HostRuntimeProbe(nvidia_smi)
        )
        self.lifecycle_timeout_s = lifecycle_timeout_s
        self._condition = threading.Condition()
        self._active_models: set[str] = set()
        self._instance_ids: dict[str, str] = {}
        self._runtime_probes: dict[str, dict[str, Any]] = {}
        self._sessions: dict[str, DesktopSession] = {}
        self._free_slots: dict[str, list[int]] = {}
        self._busy: set[str] = set()
        self._draining: set[str] = set()
        self._execute_evidence: dict[int, dict[str, Any]] = {}
        self._lifecycle_evidence: dict[int, dict[str, Any]] = {}
        self._bootstrap_preflight: dict[str, Any] | None = None
        require(
            (
                mode == "SINGLE_ACTIVE"
                and len(initial_model_ids) <= 1
            )
            or (
                mode == "DUAL_STATIC_PARTIAL"
                and (
                    not initial_model_ids
                    or set(initial_model_ids) == set(routes)
                )
            ),
            "desktop initial model set",
        )
        for model_id in initial_model_ids:
            self._install_model(model_id)
        if mode == "DUAL_STATIC_PARTIAL" and initial_model_ids:
            process_ids = {
                probe["process_id"] for probe in self._runtime_probes.values()
            }
            instance_ids = set(self._instance_ids.values())
            require(
                len(process_ids) == len(routes)
                and len(instance_ids) == len(routes),
                "partial routes do not identify distinct native children",
            )

    def _models(self) -> list[dict[str, Any]]:
        status, value = self.transport.request("GET", "/models", None)
        require(status == 200 and type(value) is dict, "router model query failed")
        data = value.get("data")
        require(type(data) is list, "router model list")
        return data

    def _preflight_empty_router(self) -> dict[str, Any]:
        rows = self._models()
        result = []
        for model_id, spec in sorted(self.routes.items()):
            matches = [
                row
                for row in rows
                if type(row) is dict
                and row.get("id") == spec.native_model_id
            ]
            require(
                len(matches) == 1,
                "router preflight model identity is ambiguous",
            )
            status = matches[0].get("status")
            require(
                type(status) is dict
                and status.get("value") == "unloaded",
                "router preflight found a resident model",
            )
            require(
                matches[0].get("warm_tier_runtime") in (None, {}),
                "router preflight found a live child identity",
            )
            result.append({
                "logical_model_id": model_id,
                "native_model_id": spec.native_model_id,
                "status": "unloaded",
            })
        return {
            "models": result,
            "schema": "s40-desktop-empty-router-preflight-v1",
        }

    def _route_row(self, model_id: str) -> dict[str, Any]:
        native_model_id = self.routes[model_id].native_model_id
        matches = [
            row for row in self._models()
            if type(row) is dict and row.get("id") == native_model_id
        ]
        require(len(matches) == 1, "router model identity is ambiguous")
        return matches[0]

    def _ready_row(self, model_id: str) -> dict[str, Any]:
        row = self._route_row(model_id)
        status = row.get("status")
        require(
            type(status) is dict and status.get("value") == "loaded",
            "router model is not loaded",
        )
        runtime = row.get("warm_tier_runtime")
        require(type(runtime) is dict, "router omitted warm-tier runtime")
        runtime = exact_keys(
            runtime,
            {
                "instance_id",
                "port",
                "process_id",
                "schema",
            },
            "warm_tier_runtime",
        )
        require(
            runtime["schema"] == "llama-server-warm-tier-runtime-identity-v1",
            "router runtime schema",
        )
        process_id = integer(runtime["process_id"], "router process id", 1)
        port = integer(runtime["port"], "router child port", 1)
        require(port <= 65535, "router child port")
        instance_id = string(runtime["instance_id"], "router instance id")
        require(
            instance_id.startswith(f"{process_id}:{port}:"),
            "router instance identity mismatch",
        )
        props_path = (
            "/props?model="
            + urllib.parse.quote(self.routes[model_id].native_model_id, safe="")
        )
        props_status, props = self.transport.request("GET", props_path, None)
        require(
            props_status == 200 and type(props) is dict,
            "router props query failed",
        )
        model_path = props.get("model_path")
        require(type(model_path) is str and model_path, "router model path")
        probe = self.runtime_probe.inspect(
            self.routes[model_id],
            process_id,
            model_path,
            port,
        )
        return {
            "instance_id": instance_id,
            "port": port,
            "probe": probe,
            "process_id": process_id,
        }

    def _wait_loaded(self, model_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.lifecycle_timeout_s
        while True:
            row = self._route_row(model_id)
            status = row.get("status")
            require(type(status) is dict, "router model status")
            value = status.get("value")
            if value == "loaded":
                return self._ready_row(model_id)
            require(value == "loading", "router load entered a terminal failure")
            require(time.monotonic() < deadline, "router load timed out")
            time.sleep(0.1)

    def _wait_unloaded(self, model_id: str) -> None:
        deadline = time.monotonic() + self.lifecycle_timeout_s
        while True:
            row = self._route_row(model_id)
            status = row.get("status")
            require(type(status) is dict, "router model status")
            value = status.get("value")
            if value == "unloaded":
                return
            require(
                value in ("loaded", "loading", "sleeping"),
                "router unload entered a terminal failure",
            )
            require(time.monotonic() < deadline, "router unload timed out")
            time.sleep(0.1)

    def _validate_ready(self, model_id: str) -> dict[str, Any]:
        require(model_id in self.routes, "unknown desktop model")
        return self._ready_row(model_id)

    def _install_model(
        self,
        model_id: str,
        identity: dict[str, Any] | None = None,
    ) -> None:
        if identity is None:
            identity = self._validate_ready(model_id)
        with self._condition:
            require(model_id not in self._active_models, "desktop model already installed")
            if self.mode == "SINGLE_ACTIVE":
                require(not self._active_models, "desktop route already has a model")
            self._active_models.add(model_id)
            self._instance_ids[model_id] = identity["instance_id"]
            self._runtime_probes[model_id] = identity["probe"]
            self._free_slots[model_id] = sorted(self.routes[model_id].slots)
            self._draining.discard(model_id)

    def _require_active(self, model_id: str) -> DesktopRouteSpec:
        with self._condition:
            require(model_id in self._active_models, "desktop model is not active")
            return self.routes[model_id]

    def _allocate(self, request: dict[str, Any]) -> DesktopSession:
        with self._condition:
            model_id = request["model_id"]
            require(model_id not in self._draining, "desktop route is draining")
            free_slots = self._free_slots.get(model_id)
            require(free_slots, "desktop route has no slot credit")
            session = DesktopSession(
                model_id,
                free_slots.pop(0),
                request["ownership_epoch"],
                [],
            )
            self._sessions[request["request_id"]] = session
            return session

    def _erase(self, request_id: str) -> None:
        with self._condition:
            session = self._sessions.pop(request_id, None)
        if session is None:
            return
        status, value = self.transport.request(
            "POST",
            f"/slots/{session.slot_id}?action=erase",
            {
                "model":
                    self.routes[session.model_id].native_model_id
            },
        )
        require(
            status == 200
            and type(value) is dict
            and value.get("id_slot") == session.slot_id,
            "desktop slot erase failed",
        )
        with self._condition:
            free_slots = self._free_slots.get(session.model_id)
            require(free_slots is not None, "desktop slot pool missing")
            free_slots.append(session.slot_id)
            free_slots.sort()

    @staticmethod
    def _completion_body(
        model_id: str,
        slot_id: int,
        history: list[int],
        n_predict: int,
    ) -> dict[str, Any]:
        return {
            "cache_prompt": True,
            "id_slot": slot_id,
            "ignore_eos": True,
            "model": model_id,
            "n_predict": n_predict,
            "prompt": history,
            "return_tokens": True,
            "seed": 0,
            "temperature": 0.0,
        }

    def _complete(
        self,
        model_id: str,
        session: DesktopSession,
        history: list[int],
        n_predict: int,
        reused: bool,
    ) -> tuple[int | None, int, int]:
        status, value = self.transport.request(
            "POST",
            "/completion",
            self._completion_body(
                self.routes[model_id].native_model_id,
                session.slot_id,
                history,
                n_predict,
            ),
        )
        require(status == 200 and type(value) is dict, "desktop completion failed")
        require(value.get("id_slot") == session.slot_id, "desktop slot changed")
        raw_tokens = value.get("tokens")
        require(type(raw_tokens) is list, "desktop tokens missing")
        require(len(raw_tokens) == n_predict, "desktop token quantum mismatch")
        for token in raw_tokens:
            require(type(token) is int and 0 <= token < (1 << 31), "desktop token")
        tokens_cached = value.get("tokens_cached")
        tokens_evaluated = value.get("tokens_evaluated")
        require(
            type(tokens_cached) is int
            and type(tokens_evaluated) is int
            and 0 <= tokens_cached <= tokens_evaluated,
            "desktop cache counters",
        )
        if reused:
            require(tokens_cached > 0, "desktop session was fully re-prefilled")
            require(
                tokens_evaluated - tokens_cached <= 1,
                "desktop session evaluated more than one new token",
            )
        return (raw_tokens[0] if raw_tokens else None), tokens_cached, tokens_evaluated

    def _replay(self, command: dict[str, Any]) -> dict[str, Any]:
        request = command["request"]
        self._erase(request["request_id"])
        session = self._allocate(request)
        history = request["prompt_tokens"] + request["committed_output_tokens"]
        try:
            token, _, _ = self._complete(
                command["model_id"],
                session,
                history,
                0,
                False,
            )
            require(token is None, "replay generated a token")
            session.logical_history = list(history)
        except BaseException:
            self._erase(request["request_id"])
            raise
        return make_result(
            command,
            success=True,
            detail="desktop request replayed",
            replay_snapshot=request,
        )

    def _execute(self, command: dict[str, Any]) -> dict[str, Any]:
        request = command["request"]
        with self._condition:
            require(
                command["model_id"] not in self._draining,
                "desktop route is draining",
            )
            require(request["request_id"] not in self._busy, "request already busy")
            self._busy.add(request["request_id"])
            session = self._sessions.get(request["request_id"])
        try:
            reused = session is not None
            if session is None:
                session = self._allocate(request)
            require(
                session.ownership_epoch == request["ownership_epoch"],
                "stale desktop ownership epoch",
            )
            history = request["prompt_tokens"] + request["committed_output_tokens"]
            if reused:
                require(session.logical_history == history, "desktop frontier mismatch")
            token, cached, evaluated = self._complete(
                command["model_id"],
                session,
                history,
                1,
                reused,
            )
            require(token is not None, "desktop token missing")
            session.logical_history = history + [token]
            publication = {
                "owner_id": command["executor_id"],
                "ownership_epoch": request["ownership_epoch"],
                "position": request["position"],
                "publication_index": request["publication_index"],
                "token": token,
            }
            with self._condition:
                self._execute_evidence[command["command_id"]] = {
                    "execute_quantum_tokens": 1,
                    "full_history_per_token_reprefill": False,
                    "initial_history_replay": not reused,
                    "instance_id": self._instance_ids[command["model_id"]],
                    "publication_count": 1,
                    "resident_session_reused": reused,
                    "runtime_probe": self._runtime_probes[command["model_id"]],
                    "sampler": {
                        "seed": 0,
                        "temperature": 0.0,
                        "type": "greedy",
                    },
                    "slot_id": session.slot_id,
                    "tokens_cached": cached,
                    "tokens_evaluated": evaluated,
                }
            return make_result(
                command,
                success=True,
                detail="desktop route executed",
                publications=[publication],
                request_complete=(
                    len(request["committed_output_tokens"]) + 1
                    == command["total_output_tokens"]
                ),
            )
        finally:
            with self._condition:
                self._busy.discard(request["request_id"])

    def take_execute_evidence(self, command_id: int) -> dict[str, Any] | None:
        with self._condition:
            return self._execute_evidence.pop(command_id, None)

    def take_lifecycle_evidence(self, command_id: int) -> dict[str, Any] | None:
        with self._condition:
            return self._lifecycle_evidence.pop(command_id, None)

    def runtime_inventory(self) -> list[dict[str, Any]]:
        with self._condition:
            return [
                {
                    "instance_id": self._instance_ids[model_id],
                    "logical_model_id": model_id,
                    "native_model_id": self.routes[model_id].native_model_id,
                    "n_gpu_layers": self.routes[model_id].n_gpu_layers,
                    "runtime_probe": self._runtime_probes[model_id],
                }
                for model_id in sorted(self._active_models)
            ]

    def close(self) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        with self._condition:
            busy = sorted(self._busy)
            request_ids = sorted(self._sessions)
            active_models = sorted(self._active_models)
        problems = []
        if busy:
            problems.append("busy requests remained at shutdown")
        for request_id in request_ids:
            try:
                self._erase(request_id)
            except BaseException as error:
                problems.append(
                    f"slot cleanup {request_id}: {type(error).__name__}: {error}"
                )
        unloaded = []
        for model_id in active_models:
            with self._condition:
                probe = dict(self._runtime_probes[model_id])
                instance_id = self._instance_ids[model_id]
            try:
                status, value = self.transport.request(
                    "POST",
                    "/models/unload",
                    {"model": self.routes[model_id].native_model_id},
                )
                require(
                    status == 200
                    and type(value) is dict
                    and value.get("success") is True,
                    "desktop shutdown unload failed",
                )
                self._wait_unloaded(model_id)
                exited = self.runtime_probe.process_exited(
                    probe["process_id"],
                    probe["process_start_ticks"],
                    self.lifecycle_timeout_s,
                )
                require(exited, "desktop child did not exit")
                unloaded.append({
                    "instance_id": instance_id,
                    "logical_model_id": model_id,
                    "native_model_id":
                        self.routes[model_id].native_model_id,
                    "process_id": probe["process_id"],
                    "process_start_ticks": probe["process_start_ticks"],
                    "process_exited": True,
                })
                with self._condition:
                    self._active_models.remove(model_id)
                    self._instance_ids.pop(model_id, None)
                    self._runtime_probes.pop(model_id, None)
                    self._free_slots.pop(model_id, None)
                    self._draining.discard(model_id)
            except BaseException as error:
                problems.append(
                    f"model cleanup {model_id}: {type(error).__name__}: {error}"
                )
        with self._condition:
            remaining_sessions = sorted(self._sessions)
            remaining_models = sorted(self._active_models)
        if remaining_sessions:
            problems.append("request sessions remained after cleanup")
        if remaining_models:
            problems.append("models remained after cleanup")
        return {
            "completed_ns": time.monotonic_ns(),
            "initial_active_models": active_models,
            "initial_busy_requests": busy,
            "initial_request_sessions": request_ids,
            "problems": problems,
            "remaining_active_models": remaining_models,
            "remaining_request_sessions": remaining_sessions,
            "schema": "s40-desktop-cleanup-evidence-v1",
            "started_ns": started_ns,
            "success": not problems,
            "unloaded": unloaded,
        }

    def handle(self, command: dict[str, Any]) -> dict[str, Any]:
        require(command["executor_id"] == self.executor_id, "desktop executor mismatch")
        kind = command["kind"]
        if self.mode == "DUAL_STATIC_PARTIAL" and kind in (
            COMMAND_DISCARD,
            COMMAND_DRAIN,
            COMMAND_UNLOAD,
        ):
            raise GatewayError("static partial route rejects model lifecycle mutation")
        if kind in (COMMAND_EXECUTE, COMMAND_REPLAY, COMMAND_CLEANUP, COMMAND_DISCARD, COMMAND_DRAIN, COMMAND_UNLOAD):
            self._require_active(command["model_id"])
        if kind == COMMAND_EXECUTE:
            return self._execute(command)
        if kind == COMMAND_REPLAY:
            with self._condition:
                require(
                    command["model_id"] not in self._draining,
                    "desktop route is draining",
                )
            return self._replay(command)
        if kind == COMMAND_CLEANUP:
            self._erase(command["request_id"])
            return make_result(command, success=True, detail="desktop slot erased")
        if kind == COMMAND_DISCARD:
            with self._condition:
                request_ids = list(self._sessions)
            for request_id in request_ids:
                self._erase(request_id)
            return make_result(command, success=True, detail="desktop state discarded")
        if kind == COMMAND_DRAIN:
            with self._condition:
                require(not self._busy, "desktop route is busy")
                self._draining.add(command["model_id"])
            return make_result(command, success=True, detail="desktop route drained")
        if kind == COMMAND_UNLOAD:
            with self._condition:
                require(
                    command["model_id"] in self._draining
                    and not any(
                        session.model_id == command["model_id"]
                        for session in self._sessions.values()
                    ),
                    "desktop route not clean",
                )
                unload_identity = {
                    "instance_id":
                        self._instance_ids[command["model_id"]],
                    "runtime_probe":
                        self._runtime_probes[command["model_id"]],
                }
            status, value = self.transport.request(
                "POST",
                "/models/unload",
                {
                    "model":
                        self.routes[command["model_id"]].native_model_id
                },
            )
            require(
                status == 200
                and type(value) is dict
                and value.get("success") is True,
                "desktop unload failed",
            )
            self._wait_unloaded(command["model_id"])
            probe = unload_identity["runtime_probe"]
            require(
                self.runtime_probe.process_exited(
                    probe["process_id"],
                    probe["process_start_ticks"],
                    self.lifecycle_timeout_s,
                ),
                "desktop child did not exit",
            )
            with self._condition:
                self._active_models.remove(command["model_id"])
                self._instance_ids.pop(command["model_id"], None)
                self._runtime_probes.pop(command["model_id"], None)
                self._free_slots.pop(command["model_id"], None)
                self._draining.discard(command["model_id"])
                self._lifecycle_evidence[command["command_id"]] = {
                    **unload_identity,
                    "operation": "UNLOAD",
                    "process_exited": True,
                    "router_status": "unloaded",
                }
            return make_result(command, success=True, detail="desktop route unloaded")
        if kind == COMMAND_LOAD:
            require(command["model_id"] in self.routes, "unknown desktop model")
            with self._condition:
                require(
                    command["model_id"] not in self._active_models,
                    "desktop route already loaded",
                )
                if self.mode == "SINGLE_ACTIVE":
                    require(
                        not self._active_models,
                        "desktop route already has another model",
                    )
                needs_preflight = self._bootstrap_preflight is None
            preflight = (
                self._preflight_empty_router()
                if needs_preflight
                else self._bootstrap_preflight
            )
            with self._condition:
                if self._bootstrap_preflight is None:
                    self._bootstrap_preflight = preflight
            cache_control = self.cache_controller.prepare(
                self.routes[command["model_id"]],
                self.cache_regime,
            )
            status, value = self.transport.request(
                "POST",
                "/models/load",
                {
                    "model":
                        self.routes[command["model_id"]].native_model_id
                },
            )
            require(
                status == 200
                and type(value) is dict
                and value.get("success") is True,
                "desktop load failed",
            )
            identity = self._wait_loaded(command["model_id"])
            self._install_model(command["model_id"], identity)
            with self._condition:
                self._lifecycle_evidence[command["command_id"]] = {
                    "cache_control": cache_control,
                    "instance_id": identity["instance_id"],
                    "operation": "LOAD",
                    "preflight": preflight,
                    "runtime_probe": identity["probe"],
                }
                if (
                    self.mode == "DUAL_STATIC_PARTIAL"
                    and self._active_models == set(self.routes)
                ):
                    process_ids = {
                        (
                            probe["process_id"],
                            probe["process_start_ticks"],
                        )
                        for probe in self._runtime_probes.values()
                    }
                    instance_ids = set(self._instance_ids.values())
                    require(
                        len(process_ids) == len(self.routes)
                        and len(instance_ids) == len(self.routes),
                        "partial routes do not identify distinct native children",
                    )
            return make_result(command, success=True, detail="desktop route loaded")
        raise GatewayError("unsupported desktop command")


class DesktopGateway:
    def __init__(
        self,
        socket_path: Path,
        evidence_path: Path,
        executor: DesktopExecutor,
        timeout_s: float,
        *,
        controller_authenticator: ControllerAuthenticator,
        executor_config_sha256: str,
        profile_lock_sha256: str | None,
        run_id: str,
        runtime_binding: RuntimeBinding,
    ):
        require(socket_path.is_absolute(), "desktop socket path")
        require(evidence_path.is_absolute(), "desktop evidence path")
        require(
            (
                executor.mode == "DUAL_STATIC_PARTIAL"
                and profile_lock_sha256 is not None
            )
            or (
                executor.mode != "DUAL_STATIC_PARTIAL"
                and profile_lock_sha256 is None
            ),
            "desktop gateway profile lock",
        )
        self.socket_path = socket_path
        self.executor = executor
        self.timeout_s = timeout_s
        self.run_id = string(run_id, "desktop gateway run ID")
        require(
            runtime_binding.executor_id == executor.executor_id
            and runtime_binding.gateway_pid == os.getpid()
            and runtime_binding.gateway_start_time_ticks
            == process_start_time_ticks(),
            "desktop gateway runtime executor identity",
        )
        self.runtime_binding = runtime_binding
        self.controller_authenticator = controller_authenticator
        self.executor_instance_id = runtime_binding.executor_instance_id
        self.runtime_config_sha256 = runtime_binding.runtime_config_sha256
        self._evidence = evidence_path.open("x", encoding="ascii")
        self._evidence_lock = threading.Lock()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._stopping = threading.Event()
        self._threads: set[threading.Thread] = set()
        self._threads_lock = threading.Lock()
        self._fatal_error: BaseException | None = None
        startup = {
            "cache_regime": executor.cache_regime,
            "executor_config_sha256": sha256_text(
                executor_config_sha256,
                "desktop executor config SHA-256",
            ),
            "executor_id": executor.executor_id,
            "executor_instance_id": self.executor_instance_id,
            "gateway_pid": runtime_binding.gateway_pid,
            "gateway_start_time_ticks":
                runtime_binding.gateway_start_time_ticks,
            "mode": executor.mode,
            "profile_lock_sha256": profile_lock_sha256,
            "role": executor.role,
            "routes": executor.runtime_inventory(),
            "run_id": self.run_id,
            "runtime_config_device": runtime_binding.runtime_config_device,
            "runtime_config_inode": runtime_binding.runtime_config_inode,
            "runtime_config_path": runtime_binding.runtime_config_path,
            "runtime_config_sha256": self.runtime_config_sha256,
            "schema": "s40-desktop-startup-evidence-v2",
        }
        self._evidence.write(canonical_bytes(startup).decode("ascii"))
        self._evidence.flush()
        os.fsync(self._evidence.fileno())

    @staticmethod
    def _read_one(connection: socket.socket) -> bytes:
        raw = bytearray()
        while len(raw) <= MAX_COMMAND_BYTES:
            chunk = connection.recv(min(65536, MAX_COMMAND_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
            if raw.endswith(b"\n"):
                break
        require(raw.endswith(b"\n"), "desktop gateway framing")
        require(len(raw) <= MAX_COMMAND_BYTES, "desktop command too large")
        require(connection.recv(1) == b"", "desktop command trailing bytes")
        return bytes(raw)

    def _serve_one(self, connection: socket.socket) -> None:
        try:
            with connection:
                connection.settimeout(self.timeout_s)
                self.controller_authenticator.authenticate(connection)
                command = parse_command(
                    self._read_one(connection),
                    self.executor.executor_id,
                    self.executor_instance_id,
                )
                started_ns = time.monotonic_ns()
                try:
                    result = self.executor.handle(command)
                except BaseException as error:
                    result = make_result(
                        command,
                        success=False,
                        detail=f"{type(error).__name__}: {error}",
                    )
                completed_ns = time.monotonic_ns()
                execute = self.executor.take_execute_evidence(
                    command["command_id"]
                )
                lifecycle = self.executor.take_lifecycle_evidence(
                    command["command_id"]
                )
                row = {
                    "command": command,
                    "command_id": command["command_id"],
                    "completed_ns": completed_ns,
                    "controller_epoch": command["controller_epoch"],
                    "durability": "fsync_each_record",
                    "execute": execute,
                    "executor_id": command["executor_id"],
                    "executor_instance_id": command[
                        "executor_instance_id"
                    ],
                    "kind": command["kind"],
                    "lifecycle": lifecycle,
                    "model_id": command["model_id"],
                    "request_id": command["request_id"] or None,
                    "result": result,
                    "role": self.executor.role,
                    "run_id": self.run_id,
                    "runtime_config_sha256": self.runtime_config_sha256,
                    "schema": "s40-desktop-command-evidence-v4",
                    "started_ns": started_ns,
                    "success": result["success"],
                }
                with self._evidence_lock:
                    self._evidence.write(canonical_bytes(row).decode("ascii"))
                    self._evidence.flush()
                    os.fsync(self._evidence.fileno())
                connection.sendall(canonical_bytes(result))
        except BaseException as error:
            with self._threads_lock:
                if self._fatal_error is None:
                    self._fatal_error = error
            self._stopping.set()
        finally:
            with self._threads_lock:
                self._threads.discard(threading.current_thread())

    def request_stop(self) -> None:
        self._stopping.set()

    def serve_forever(self) -> None:
        try:
            self._server.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            self._server.listen(128)
            self._server.settimeout(0.2)
            while not self._stopping.is_set():
                try:
                    connection, _ = self._server.accept()
                except socket.timeout:
                    continue
                thread = threading.Thread(
                    target=self._serve_one,
                    args=(connection,),
                    daemon=False,
                )
                with self._threads_lock:
                    self._threads.add(thread)
                thread.start()
        finally:
            self._stopping.set()
            self._server.close()
            while True:
                with self._threads_lock:
                    threads = list(self._threads)
                if not threads:
                    break
                for thread in threads:
                    thread.join(self.timeout_s)
                    require(
                        not thread.is_alive(),
                        "desktop gateway request did not drain",
                    )
            cleanup_error = None
            cleanup = None
            try:
                cleanup = {
                    **self.executor.close(),
                    "executor_id": self.executor.executor_id,
                    "executor_instance_id": self.executor_instance_id,
                    "run_id": self.run_id,
                    "runtime_config_sha256": self.runtime_config_sha256,
                    "schema": "s40-desktop-cleanup-evidence-v2",
                }
                with self._evidence_lock:
                    self._evidence.write(
                        canonical_bytes(cleanup).decode("ascii")
                    )
                    self._evidence.flush()
                    os.fsync(self._evidence.fileno())
            except BaseException as error:
                cleanup_error = error
            try:
                self._evidence.close()
            finally:
                self.socket_path.unlink(missing_ok=True)
            if cleanup_error is not None:
                raise cleanup_error
            require(
                cleanup is not None and cleanup["success"] is True,
                "desktop cleanup failed",
            )
            with self._threads_lock:
                fatal_error = self._fatal_error
            if fatal_error is not None:
                raise fatal_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--initial-model", action="append", default=[])
    parser.add_argument("--executor-instance-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--controller-identity", type=Path, required=True)
    parser.add_argument(
        "--controller-binding-evidence",
        type=Path,
        required=True,
    )
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    require(
        args.config.is_absolute()
        and args.evidence.is_absolute()
        and args.runtime_config.is_absolute()
        and args.controller_identity.is_absolute()
        and args.controller_binding_evidence.is_absolute()
        and args.socket.is_absolute()
        and args.timeout > 0,
        "desktop gateway arguments",
    )
    (
        executor_id,
        role,
        mode,
        base_url,
        cache_regime,
        routes,
        profile_lock_sha256,
        nvidia_smi,
        executor_config_sha256,
    ) = parse_desktop_config(args.config)
    require(
        not args.initial_model,
        "desktop gateway must start empty; use controller LOAD",
    )
    runtime_binding = await_runtime_binding(
        args.runtime_config,
        executor_id=executor_id,
        executor_instance_id=args.executor_instance_id,
        run_id=args.run_id,
        socket_path=args.socket,
        timeout_s=args.timeout,
    )
    controller_authenticator = ControllerAuthenticator(
        args.controller_identity,
        args.controller_binding_evidence,
        run_id=args.run_id,
        runtime_binding=runtime_binding,
        timeout_s=args.timeout,
    )
    executor = DesktopExecutor(
        executor_id,
        role,
        mode,
        routes,
        UrllibTransport(base_url, args.timeout),
        (),
        SubprocessCacheController(args.timeout),
        cache_regime,
        args.timeout,
        nvidia_smi=nvidia_smi,
    )
    gateway = DesktopGateway(
        args.socket,
        args.evidence,
        executor,
        args.timeout,
        controller_authenticator=controller_authenticator,
        executor_config_sha256=executor_config_sha256,
        profile_lock_sha256=profile_lock_sha256,
        run_id=args.run_id,
        runtime_binding=runtime_binding,
    )
    previous_term = signal.signal(
        signal.SIGTERM,
        lambda _signum, _frame: gateway.request_stop(),
    )
    previous_int = signal.signal(
        signal.SIGINT,
        lambda _signum, _frame: gateway.request_stop(),
    )
    try:
        gateway.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GatewayError, OSError, RuntimeBindingError, TimeoutError) as error:
        print(f"desktop gateway failed: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)
