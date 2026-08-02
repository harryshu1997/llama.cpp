#!/usr/bin/env python3
"""Materialize the V2.6 A_ONLY production runtime inventory and phase lock."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import time
import types
from typing import Any


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
V24 = S39 / "v24_readiness"

LOCK_SCHEMA = "s39-cp0-r1-v26-phase-lock-v1"
RECORD_SCHEMA = "s39-cp0-r1-v26-production-materialization-v1"
VALIDATION_SCHEMA = "s39-cp0-r1-v26-production-validation-v1"
CONFIRMATION = "RUN_CP0_R1_V26_PRODUCTION_MATERIALIZATION"
PHASE = "A_ONLY"
MODEL_ID = "qwen3-14b-q4_k_m"
PHASE_ID_PREFIX = "cp0-r1-v26-a-only-"
CLOCK_ID = "HOST_MONOTONIC_RAW"
PRODUCER_RELATIVE_PATH = "v26_readiness/materialize_production_v26.py"

CONTRACT_PATH = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2_6.json"
V24_CONTRACT_PATH = V24 / "CP0_R1_EVIDENCE_CONTRACT_V2_4.json"
TOPOLOGY_PATH = HERE / "results" / "no_model_topology_v241.json"
MONO_LAUNCH_PATH = (
    V24 / "results" / "prephase_20260726T0915Z" / "cuda-monolithic-launch.json"
)
TOPOLOGY_VERIFIER_PATH = V24 / "desktop_deployment_v1" / "verify_topology_v1.py"
TOPOLOGY_VERIFIER_SHA256 = (
    "602dc531001cea06fc30161879c1bcf454b06f92c6c9942bfc33571ed3885c66"
)
USB_LAUNCHER_SOURCE = (
    V24 / "desktop_deployment_v1" / "managed_runtime_launcher_usb_v1.py"
)
USB_LAUNCHER_SHA256 = (
    "52c4e1f251f4daa2c856857fcdf23fdc5a85c0d30eff93365996a8f1378611ce"
)
FROZEN_LAUNCHER_SOURCE = (
    S39
    / "v23_readiness"
    / "a_only_acquisition_driver_v1"
    / "producers_v1"
    / "managed_runtime_launcher_v1.py"
)
FROZEN_LAUNCHER_SHA256 = (
    "b97941dc30399135b04e98dbdf102aaeb6c695c55a7b990402aeafd21ed4a245"
)

SSH_TARGET = "zhihao@172.20.74.85"
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")
DESKTOP_ROOT = "/home/zhihao/s39-v26-a-only"
CUDA_MONO_ROOT = DESKTOP_ROOT + "/runtime-v1/cuda-monolithic"
CUDA_ROUTE_ROOT = DESKTOP_ROOT + "/runtime-v1/cuda-route"
LAUNCHER_ROOT = DESKTOP_ROOT + "/launcher-v1"
DESKTOP_USB_LAUNCHER = (
    LAUNCHER_ROOT
    + "/v24_readiness/desktop_deployment_v1/managed_runtime_launcher_usb_v1.py"
)
DESKTOP_FROZEN_LAUNCHER = (
    LAUNCHER_ROOT
    + "/v23_readiness/a_only_acquisition_driver_v1"
    + "/producers_v1/managed_runtime_launcher_v1.py"
)
DESKTOP_ADB_PATH = "/usr/lib/android-sdk/platform-tools/adb"
DESKTOP_MODEL_PATH = "/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf"
LOCAL_ADB_PATH = "/usr/lib/android-sdk/platform-tools/adb"
ADB_PORT = 5038

OP15_STAGE_ROOT = "/data/local/tmp/s39-v24-a-only/runtime-v1/op15-stagenet"
OP15_RELAY_ROOT = "/data/local/tmp/s39-v24-a-only/runtime-v1/op15-direct-relay"
OP12_STAGE_ROOT = "/data/local/tmp/s39-v24-a-only/runtime-v1/op12-stagenet"
SHARD_PATH = (
    "/data/local/tmp/s39-active-warm/v1/models/qwen3-14b-q4_k_m/weights.gguf"
)

UNBOUND_BOOT_IDS = {
    "cuda": "00000000-0000-0000-0000-000000000000",
    "op12": "00000000-0000-0000-0000-000000000001",
    "op15": "00000000-0000-0000-0000-000000000002",
}
UNBOUND_RELAY_HEAD_HOST = "0.0.0.12"
PORTS = {
    "cuda_route": 39125,
    "op12_stage": 39126,
    "op15_stage": 39127,
    "relay": 39128,
    "relay_tail_source": 39129,
}

MONO_FILES = {
    "artifact_root_capture_v1.py": (
        "cuda-mono.capture-artifact-root",
        "capture_entrypoint",
    ),
    "cuda_monolithic_v1.py": (
        "cuda-mono.capture-cuda-monolithic",
        "capture_entrypoint",
    ),
    "libggml-base.so.0.15.3": ("cuda-mono.libggml-base", "shared_library"),
    "libggml-cpu.so.0.15.3": ("cuda-mono.libggml-cpu", "shared_library"),
    "libggml-cuda.so.0.15.3": ("cuda-mono.libggml-cuda", "backend_library"),
    "libggml.so.0.15.3": ("cuda-mono.libggml", "shared_library"),
    "libllama-common.so.0.0.9875": (
        "cuda-mono.libllama-common",
        "shared_library",
    ),
    "libllama.so.0.0.9875": ("cuda-mono.libllama", "shared_library"),
    "llama-layersplit": ("cuda-mono.bin", "executable"),
}
ROUTE_FILES = {
    "artifact_root_capture_v1.py": (
        "cuda-route.capture-artifact-root",
        "capture_entrypoint",
    ),
    "cuda_monolithic_v1.py": (
        "cuda-route.capture-cuda-monolithic",
        "capture_entrypoint",
    ),
    "fast_fresh_capture_v1.py": (
        "cuda-route.capture-fast-fresh-readiness",
        "capture_entrypoint",
    ),
    "joint_phone_cuda_v1.py": (
        "cuda-route.capture-joint-phone-cuda",
        "capture_entrypoint",
    ),
    "libggml-base.so.0": ("cuda-route.libggml-base", "shared_library"),
    "libggml-cpu.so.0": ("cuda-route.libggml-cpu", "shared_library"),
    "libggml-cuda.so.0": ("cuda-route.libggml-cuda", "backend_library"),
    "libggml.so.0": ("cuda-route.libggml", "shared_library"),
    "libllama-common.so.0": ("cuda-route.libllama-common", "shared_library"),
    "libllama.so.0": ("cuda-route.libllama", "shared_library"),
    "llama-layersplit": ("cuda-route.bin", "executable"),
    "llama-token-codec": ("cuda-tokenize", "executable"),
}
PHONE_STAGE_FILES = {
    "libggml-base.so": ("libggml-base", "shared_library"),
    "libggml-cpu.so": ("libggml-cpu", "shared_library"),
    "libggml-hexagon.so": ("libggml-hexagon", "backend_library"),
    "libggml-opencl.so": ("libggml-opencl", "backend_library"),
    "libggml.so": ("libggml", "shared_library"),
    "libllama-common.so": ("libllama-common", "shared_library"),
    "libllama.so": ("libllama", "shared_library"),
    "llama-layersplit": ("bin", "executable"),
}
PHONE_STAGE_HVX = {
    "op12_stagenet": ("libggml-htp-v75.so", "libggml-htp-v75"),
    "op15_stagenet": ("libggml-htp-v81.so", "libggml-htp-v81"),
}
RELAY_FILES = {
    "llama-stage-direct-relay": ("op15_direct_relay.bin", "executable"),
}
MONO_PIN_IDS = {
    "libggml-base.so.0.15.3": "cuda-mono.libggml-base",
    "libggml-cpu.so.0.15.3": "cuda-mono.libggml-cpu",
    "libggml-cuda.so.0.15.3": "cuda-mono.libggml-cuda",
    "libggml.so.0.15.3": "cuda-mono.libggml",
    "libllama-common.so.0.0.9875": "cuda-mono.libllama-common",
    "libllama.so.0.0.9875": "cuda-mono.libllama",
    "llama-layersplit": "cuda-mono.bin",
}
CAPTURE_ENTRYPOINT_IDS = {
    "artifact_root": "cuda-mono.capture-artifact-root",
    "cuda_monolithic": "cuda-mono.capture-cuda-monolithic",
    "fast_fresh_readiness": "cuda-route.capture-fast-fresh-readiness",
    "joint_phone_cuda": "cuda-route.capture-joint-phone-cuda",
}
BUNDLE_ROOTS = {
    "cuda_monolithic": CUDA_MONO_ROOT,
    "cuda_route": CUDA_ROUTE_ROOT,
    "op12_stagenet": OP12_STAGE_ROOT,
    "op15_direct_relay": OP15_RELAY_ROOT,
    "op15_stagenet": OP15_STAGE_ROOT,
}
BUNDLE_ENDPOINTS = {
    "cuda_monolithic": "cuda",
    "cuda_route": "cuda",
    "op12_stagenet": "op12",
    "op15_direct_relay": "op15",
    "op15_stagenet": "op15",
}
BUNDLE_LAUNCHERS = {
    "cuda_monolithic": "cuda-mono.bin",
    "cuda_route": "cuda-route.bin",
    "op12_stagenet": "op12_stagenet.bin",
    "op15_direct_relay": "op15_direct_relay.bin",
    "op15_stagenet": "op15_stagenet.bin",
}
MANAGED_BUNDLES = (
    "cuda_route",
    "op12_stagenet",
    "op15_direct_relay",
    "op15_stagenet",
)
LOCK_KEYS = {
    "clock_id",
    "completed_ns",
    "contract_sha256",
    "cuda_monolithic_launch_sha256",
    "device_boot_ids",
    "device_identities",
    "event_ns",
    "event_utc_ns",
    "inventory_sha256",
    "materialization_host",
    "model_artifacts",
    "model_id",
    "phase",
    "phase_id",
    "producer",
    "schema",
    "spec_sha256",
    "started_ns",
    "topology_receipt_sha256",
}
PUBLISHED_FILES = (
    "RUNTIME_INVENTORY_SPEC_V2_6.json",
    "RUNTIME_INVENTORY_V2_6.json",
    "PHASE_LOCK_V2_6.json",
    "PRODUCTION_MATERIALIZATION_V2_6.json",
)
MANIFEST_NAME = "SHA256SUMS.txt"


class ProductionError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionError(message)


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"E_IMPORT: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inv = _load_module("s39_v26_prod_inventory", HERE / "runtime_inventory_v26.py")
authority = _load_module("s39_v26_prod_authority", HERE / "cp0_r1_evidence_v26.py")


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}",
    )


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    try:
        return inv.canonical_bytes(value)
    except inv.InventoryError as error:
        raise ProductionError(str(error)) from error


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def monotonic_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def utc_ns() -> int:
    return time.time_ns()


def _load_pinned_module(name: str, path: Path, expected_sha256: str) -> Any:
    raw = path.read_bytes()
    exact(sha256(raw), expected_sha256, f"pinned_source.{path.name}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


class Capture:
    """Ordered evidence log of every probe command and its output."""

    def __init__(self, runner: Any, clock_ns: Any):
        self.runner = runner
        self.clock_ns = clock_ns
        self.rows: list[dict[str, Any]] = []

    def run(self, capture_id: str, argv: list[str], timeout: int) -> bytes:
        started = self.clock_ns()
        raw = self.runner.run(argv, timeout)
        completed = self.clock_ns()
        require(completed > started, f"E_CAPTURE_INTERVAL: {capture_id}")
        self.rows.append(
            {
                "argv": list(argv),
                "capture_id": capture_id,
                "completed_ns": completed,
                "started_ns": started,
                "stdout_sha256": sha256(raw),
            }
        )
        return raw


def ssh_argv(script: str, target: str = SSH_TARGET) -> list[str]:
    return ["ssh", *SSH_OPTIONS, target, "sh -c " + shlex.quote(script)]


def adb_argv(serial: str, *arguments: str) -> list[str]:
    return [LOCAL_ADB_PATH, "-P", str(ADB_PORT), "-s", serial, *arguments]


def _remote_stat_script(path: str) -> str:
    quoted = shlex.quote(path)
    format_string = "DEV=%d|INO=%i|SIZE=%s|MODE=%f|MTIME=%Y|CTIME=%Z"
    return "; ".join(
        (
            "set -eu",
            f"test -f {quoted}",
            f"test ! -L {quoted}",
            f"stat -c {shlex.quote(format_string)} -- {quoted}",
        )
    )


class SshDirectoryObserver:
    """Observe desktop-local bundle closures over hardened SSH.

    Implements the same exact-closure and read-once TOCTOU discipline as
    AdbDirectoryObserver: ancestor no-symlink checks, LC_ALL=C listing,
    double sha256sum, and byte-identical before/after stat identity.
    """

    def __init__(self, target: str = SSH_TARGET, runner: Any = None):
        self.target = target
        self.runner = runner or inv.SubprocessRunner()

    def _run(self, script: str, timeout: int) -> bytes:
        return self.runner.run(ssh_argv(script, self.target), timeout)

    def _stat(self, path: str, field: str) -> dict[str, Any]:
        raw = self._run(_remote_stat_script(path), 30)
        return inv._parse_remote_stat(raw, field)

    def _pin(self, path: str, executable: bool) -> dict[str, Any]:
        before = self._stat(path, f"{self.target}:{path}.before")
        digest_script = "; ".join(
            ("set -eu", "sha256sum -- " + shlex.quote(path))
        )
        first = self._run(digest_script, 600)
        second = self._run(digest_script, 600)
        exact(second, first, f"desktop.digest_mutation.{path}")
        after = self._stat(path, f"{self.target}:{path}.after")
        exact(after, before, f"desktop.stat_mutation.{path}")
        try:
            fields = first.decode("ascii").strip().split()
        except UnicodeDecodeError as error:
            raise ProductionError(f"E_DESKTOP_DIGEST: {path}") from error
        require(
            len(fields) == 2
            and len(fields[0]) == 64
            and all(character in "0123456789abcdef" for character in fields[0]),
            f"E_DESKTOP_DIGEST: {path}",
        )
        if executable:
            require(before["mode"] & 0o111 != 0, f"E_DESKTOP_EXECUTABLE: {path}")
        return {
            "bytes": before["size"],
            "path": path,
            "sha256": fields[0],
            "stat": before,
        }

    def observe(
        self,
        *,
        endpoint: str,
        root: str,
        files: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        require(endpoint == "cuda", f"E_DESKTOP_ENDPOINT: {endpoint}")
        checks = []
        for ancestor in inv._remote_ancestors(root):
            quoted = shlex.quote(ancestor)
            checks.extend((f"test -d {quoted}", f"test ! -L {quoted}"))
        checks.append(f"LC_ALL=C ls -1A -- {shlex.quote(root)}")
        raw = self._run("; ".join(("set -eu", *checks)), 30)
        try:
            actual = raw.decode("ascii").splitlines()
        except UnicodeDecodeError as error:
            raise ProductionError(f"E_DESKTOP_LIST: {root}") from error
        require(
            actual == sorted(set(actual))
            and all(
                item and "/" not in item and item not in {".", ".."}
                for item in actual
            ),
            f"E_DESKTOP_NAMES: {root}",
        )
        exact(
            actual,
            sorted(item["filename"] for item in files),
            f"closure.desktop.{root}",
        )
        return [
            self._pin(
                f"{root}/{item['filename']}",
                item["role"] in {"capture_entrypoint", "executable"},
            )
            for item in files
        ]

    def pin_launcher(self, path: str) -> dict[str, Any]:
        return self._pin(path, True)


def read_file_once(path: Path, field: str) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProductionError(f"E_READ: {field}: {path}") from error
    try:
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    require(bool(raw), f"E_EMPTY: {field}: {path}")
    return raw


def verify_static_inputs(contract: dict[str, Any]) -> None:
    for name, descriptor in sorted(contract["static_inputs"].items()):
        raw = read_file_once(S39 / descriptor["path"], f"static_inputs.{name}")
        exact(len(raw), descriptor["bytes"], f"static_inputs.{name}.bytes")
        exact(sha256(raw), descriptor["sha256"], f"static_inputs.{name}.sha256")


def load_contract(path: Path = CONTRACT_PATH) -> tuple[dict[str, Any], bytes]:
    try:
        contract, raw = authority.load_contract(path)
        authority.validate_composition(contract)
    except authority.EvidenceError as error:
        raise ProductionError(f"E_CONTRACT: {error}") from error
    verify_static_inputs(contract)
    return contract, raw


def load_v24_contract(
    contract: dict[str, Any],
    path: Path = V24_CONTRACT_PATH,
) -> dict[str, Any]:
    try:
        value, raw = authority.read_canonical(path)
    except authority.EvidenceError as error:
        raise ProductionError(f"E_V24_CONTRACT: {error}") from error
    exact(
        sha256(raw),
        contract["raw_predicate"]["v24_contract_sha256"],
        "v24_contract.sha256",
    )
    return value


def load_topology(
    contract_v24: dict[str, Any],
    path: Path = TOPOLOGY_PATH,
) -> tuple[dict[str, Any], str]:
    try:
        value, raw = authority.read_canonical(path)
    except authority.EvidenceError as error:
        raise ProductionError(f"E_TOPOLOGY: {error}") from error
    verifier = _load_pinned_module(
        "s39_v26_topology_verifier",
        TOPOLOGY_VERIFIER_PATH,
        TOPOLOGY_VERIFIER_SHA256,
    )
    try:
        verifier.validate_topology(value, contract_v24, require_fresh=False)
    except Exception as error:
        raise ProductionError(f"E_TOPOLOGY_REPLAY: {error}") from error
    return value, sha256(raw)


def load_mono_launch(path: Path = MONO_LAUNCH_PATH) -> tuple[dict[str, Any], str]:
    try:
        value, raw = authority.read_canonical(path)
    except authority.EvidenceError as error:
        raise ProductionError(f"E_MONO_LAUNCH: {error}") from error
    exact(
        value.get("schema"),
        "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
        "mono_launch.schema",
    )
    exact(value.get("model_id"), MODEL_ID, "mono_launch.model_id")
    return value, sha256(raw)


def mono_component_pins(mono_launch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pins = {}
    for component in mono_launch["required_components"]:
        pins[component["component_id"]] = {
            "bytes": component["stat"]["size"],
            "sha256": component["sha256"],
        }
    exact(sorted(pins), sorted(MONO_PIN_IDS.values()), "mono_launch.components")
    return pins


def _parse_key_value(raw: bytes, field: str) -> str:
    try:
        return raw.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ProductionError(f"E_PROBE_TEXT: {field}") from error


def probe_adb_devices(
    capture: Capture,
    contract: dict[str, Any],
) -> None:
    raw = capture.run(
        "adb_devices",
        [LOCAL_ADB_PATH, "-P", str(ADB_PORT), "devices", "-l"],
        30,
    )
    text = _parse_key_value(raw, "adb_devices")
    for endpoint in ("op12", "op15"):
        devices = contract["devices"][endpoint]
        rows = [
            line
            for line in text.splitlines()
            if line.split() and line.split()[0] == devices["serial"]
        ]
        require(len(rows) == 1, f"E_ADB_DEVICE: {endpoint}")
        row = rows[0]
        require(" device " in row, f"E_ADB_STATE: {endpoint}")
        for key in ("product", "model", "device"):
            require(
                f"{key}:{devices[key]}" in row.split(),
                f"E_ADB_IDENTITY: {endpoint}.{key}",
            )


def probe_phone(
    capture: Capture,
    endpoint: str,
    contract: dict[str, Any],
    topology: dict[str, Any],
) -> dict[str, str]:
    serial = contract["devices"][endpoint]["serial"]
    boot_id = _parse_key_value(
        capture.run(
            f"{endpoint}.boot_id",
            adb_argv(serial, "shell", "cat /proc/sys/kernel/random/boot_id"),
            30,
        ),
        f"{endpoint}.boot_id",
    )
    observed = topology["observed"]["phones"][endpoint]
    exact(boot_id, observed["boot_id"], f"{endpoint}.boot_id.topology")
    for prop, key in (
        ("ro.product.name", "product"),
        ("ro.product.model", "model"),
        ("ro.product.device", "device"),
    ):
        value = _parse_key_value(
            capture.run(
                f"{endpoint}.{key}",
                adb_argv(serial, "shell", f"getprop {prop}"),
                30,
            ),
            f"{endpoint}.{key}",
        )
        exact(value, contract["devices"][endpoint][key], f"{endpoint}.{key}")
    address = _parse_key_value(
        capture.run(
            f"{endpoint}.wifi_ipv4",
            adb_argv(
                serial,
                "shell",
                "ip -4 -o addr show wlan0 | head -n 1"
                " | tr -s ' ' | cut -d ' ' -f 4",
            ),
            30,
        ),
        f"{endpoint}.wifi_ipv4",
    )
    require("/" in address, f"E_PHONE_IPV4: {endpoint}")
    ipv4 = address.split("/")[0]
    exact(ipv4, observed["wifi_ipv4"], f"{endpoint}.wifi_ipv4.topology")
    return {"boot_id": boot_id, "wifi_ipv4": ipv4}


def pin_phone_shard(
    capture: Capture,
    endpoint: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    serial = contract["devices"][endpoint]["serial"]
    known = contract["model_route_lock"]["geometry"]["known_shards"][endpoint]
    exact(known["path"], SHARD_PATH, f"{endpoint}.shard.path")
    stat_script = "sh -c " + shlex.quote(_remote_stat_script(SHARD_PATH))
    before = inv._parse_remote_stat(
        capture.run(
            f"{endpoint}.shard.stat_before",
            adb_argv(serial, "shell", stat_script),
            30,
        ),
        f"{endpoint}.shard.before",
    )
    digest_raw = capture.run(
        f"{endpoint}.shard.sha256",
        adb_argv(serial, "shell", "sha256sum -- " + shlex.quote(SHARD_PATH)),
        1800,
    )
    fields = _parse_key_value(digest_raw, f"{endpoint}.shard.sha256").split()
    require(len(fields) == 2 and len(fields[0]) == 64, f"E_SHARD_DIGEST: {endpoint}")
    after = inv._parse_remote_stat(
        capture.run(
            f"{endpoint}.shard.stat_after",
            adb_argv(serial, "shell", stat_script),
            30,
        ),
        f"{endpoint}.shard.after",
    )
    exact(after, before, f"{endpoint}.shard.stat_mutation")
    exact(before["size"], known["bytes"], f"{endpoint}.shard.bytes")
    exact(fields[0], known["sha256"], f"{endpoint}.shard.sha256")
    return {
        "bytes": before["size"],
        "path": SHARD_PATH,
        "sha256": fields[0],
    }


def probe_desktop(
    capture: Capture,
    contract: dict[str, Any],
    topology: dict[str, Any],
) -> dict[str, Any]:
    cuda = contract["devices"]["cuda"]
    hostname = _parse_key_value(
        capture.run("cuda.hostname", ssh_argv("hostname"), 30),
        "cuda.hostname",
    )
    exact(hostname, cuda["host"], "cuda.hostname")
    boot_id = _parse_key_value(
        capture.run(
            "cuda.boot_id",
            ssh_argv("cat /proc/sys/kernel/random/boot_id"),
            30,
        ),
        "cuda.boot_id",
    )
    exact(boot_id, topology["observed"]["cuda"]["boot_id"], "cuda.boot_id.topology")
    gpu = _parse_key_value(
        capture.run(
            "cuda.gpu",
            ssh_argv(
                "nvidia-smi --id " + shlex.quote(cuda["uuid"])
                + " --query-gpu=uuid,name,memory.total"
                + " --format=csv,noheader,nounits"
            ),
            60,
        ),
        "cuda.gpu",
    )
    parts = [part.strip() for part in gpu.split(",")]
    require(len(parts) == 3, "E_GPU_QUERY")
    exact(parts[0], cuda["uuid"], "cuda.gpu.uuid")
    exact(parts[1], cuda["name"], "cuda.gpu.name")
    require(parts[2].isdigit(), "E_GPU_MEMORY")
    exact(
        int(parts[2]) * 1024 * 1024,
        cuda["memory_total_bytes"],
        "cuda.gpu.memory_total_bytes",
    )
    return {"boot_id": boot_id, "hostname": hostname}


def pin_desktop_model(
    capture: Capture,
    contract: dict[str, Any],
) -> dict[str, Any]:
    artifact = contract["candidate_lock"]["model"]["artifact"]
    geometry = contract["model_route_lock"]["geometry"]
    exact(geometry["cuda_model_path"], DESKTOP_MODEL_PATH, "cuda_model.path")
    before = inv._parse_remote_stat(
        capture.run(
            "cuda.model.stat_before",
            ssh_argv(_remote_stat_script(DESKTOP_MODEL_PATH)),
            30,
        ),
        "cuda.model.before",
    )
    digest_raw = capture.run(
        "cuda.model.sha256",
        ssh_argv(
            "sha256sum -- " + shlex.quote(DESKTOP_MODEL_PATH)
        ),
        1800,
    )
    fields = _parse_key_value(digest_raw, "cuda.model.sha256").split()
    require(len(fields) == 2 and len(fields[0]) == 64, "E_MODEL_DIGEST")
    after = inv._parse_remote_stat(
        capture.run(
            "cuda.model.stat_after",
            ssh_argv(_remote_stat_script(DESKTOP_MODEL_PATH)),
            30,
        ),
        "cuda.model.after",
    )
    exact(after, before, "cuda.model.stat_mutation")
    exact(before["size"], artifact["bytes"], "cuda.model.bytes")
    exact(fields[0], artifact["sha256"], "cuda.model.sha256")
    return {
        "bytes": before["size"],
        "path": DESKTOP_MODEL_PATH,
        "sha256": fields[0],
    }


def pin_desktop_adb(capture: Capture) -> dict[str, Any]:
    before = inv._parse_remote_stat(
        capture.run(
            "cuda.adb.stat",
            ssh_argv(_remote_stat_script(DESKTOP_ADB_PATH)),
            30,
        ),
        "cuda.adb.before",
    )
    digest_raw = capture.run(
        "cuda.adb.sha256",
        ssh_argv("sha256sum -- " + shlex.quote(DESKTOP_ADB_PATH)),
        120,
    )
    fields = _parse_key_value(digest_raw, "cuda.adb.sha256").split()
    require(len(fields) == 2 and len(fields[0]) == 64, "E_ADB_DIGEST")
    require(before["mode"] & 0o111 != 0, "E_ADB_EXECUTABLE")
    return {"path": DESKTOP_ADB_PATH, "sha256": fields[0]}


def ensure_desktop_launcher(
    capture: Capture,
    local_source: Path,
    expected_sha256: str,
    remote_path: str,
    label: str,
) -> None:
    local_pin = inv.secure_local_pin(local_source, executable=False)
    exact(local_pin["sha256"], expected_sha256, f"launcher.local.{label}")
    quoted = shlex.quote(remote_path)
    presence = _parse_key_value(
        capture.run(
            f"launcher.{label}.presence",
            ssh_argv(
                f"if test -e {quoted}; then echo PRESENT; else echo ABSENT; fi"
            ),
            30,
        ),
        f"launcher.{label}.presence",
    )
    require(presence in {"PRESENT", "ABSENT"}, f"E_LAUNCHER_PRESENCE: {label}")
    if presence == "ABSENT":
        parent = shlex.quote(str(Path(remote_path).parent))
        capture.run(
            f"launcher.{label}.mkdir",
            ssh_argv(f"mkdir -p -- {parent}"),
            30,
        )
        capture.run(
            f"launcher.{label}.copy",
            [
                "scp",
                "-o",
                "BatchMode=yes",
                str(local_source),
                f"{SSH_TARGET}:{remote_path}",
            ],
            120,
        )
        capture.run(
            f"launcher.{label}.chmod",
            ssh_argv(f"chmod 0755 -- {quoted}"),
            30,
        )
    digest_raw = capture.run(
        f"launcher.{label}.sha256",
        ssh_argv(
            "; ".join(
                (
                    "set -eu",
                    f"test -f {quoted}",
                    f"test ! -L {quoted}",
                    f"test -x {quoted}",
                    f"sha256sum -- {quoted}",
                )
            )
        ),
        120,
    )
    fields = _parse_key_value(digest_raw, f"launcher.{label}.sha256").split()
    require(len(fields) == 2, f"E_LAUNCHER_DIGEST: {label}")
    exact(fields[0], expected_sha256, f"launcher.remote.{label}")


def _spec_files(
    table: dict[str, tuple[str, str]],
    bundle_id: str,
    pins: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for filename, (suffix, role) in table.items():
        if bundle_id in {"cuda_monolithic", "cuda_route"}:
            component_id = suffix
        elif suffix.startswith(bundle_id):
            component_id = suffix
        else:
            component_id = f"{bundle_id}.{suffix}"
        pin = pins[filename]
        rows.append(
            {
                "bytes": pin["bytes"],
                "component_id": component_id,
                "filename": filename,
                "role": role,
                "sha256": pin["sha256"],
            }
        )
    rows.sort(key=lambda row: row["component_id"])
    return rows


def _phone_table(bundle_id: str) -> dict[str, tuple[str, str]]:
    if bundle_id == "op15_direct_relay":
        return dict(RELAY_FILES)
    table = dict(PHONE_STAGE_FILES)
    htp_name, htp_component = PHONE_STAGE_HVX[bundle_id]
    table[htp_name] = (htp_component, "backend_library")
    return table


def bundle_tables() -> dict[str, dict[str, tuple[str, str]]]:
    return {
        "cuda_monolithic": dict(MONO_FILES),
        "cuda_route": dict(ROUTE_FILES),
        "op12_stagenet": _phone_table("op12_stagenet"),
        "op15_direct_relay": _phone_table("op15_direct_relay"),
        "op15_stagenet": _phone_table("op15_stagenet"),
    }


def observe_bundles(
    local_observer: Any,
    android_observer: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    observed = {}
    for bundle_id, table in bundle_tables().items():
        endpoint = BUNDLE_ENDPOINTS[bundle_id]
        observer = local_observer if endpoint == "cuda" else android_observer
        files = [
            {"filename": filename, "role": role}
            for filename, (_suffix, role) in sorted(table.items())
        ]
        pins = observer.observe(
            endpoint=endpoint,
            root=BUNDLE_ROOTS[bundle_id],
            files=files,
        )
        exact(len(pins), len(files), f"observe.{bundle_id}")
        observed[bundle_id] = {
            entry["filename"]: pin for entry, pin in zip(files, pins)
        }
    return observed


def verify_pinned_components(
    observed: dict[str, dict[str, dict[str, Any]]],
    contract: dict[str, Any],
    contract_v24: dict[str, Any],
    mono_launch: dict[str, Any],
) -> None:
    producers = contract["composition"]["capture_producers"]
    v24_programs = contract_v24["producer_requirements"]["source_programs"]
    entrypoint_pins = {
        "artifact_root_capture_v1.py": producers["artifact_root"],
        "cuda_monolithic_v1.py": v24_programs["cuda_monolithic"],
        "fast_fresh_capture_v1.py": producers["fast_fresh_readiness"],
        "joint_phone_cuda_v1.py": v24_programs["joint_phone_cuda"],
    }
    for bundle_id in ("cuda_monolithic", "cuda_route"):
        for filename, pin in observed[bundle_id].items():
            expected = entrypoint_pins.get(filename)
            if expected is not None:
                exact(pin["sha256"], expected["sha256"], f"entrypoint.{filename}")
                exact(pin["bytes"], expected["bytes"], f"entrypoint.{filename}.bytes")
    launch_pins = mono_component_pins(mono_launch)
    for filename, component_id in MONO_PIN_IDS.items():
        pin = observed["cuda_monolithic"][filename]
        expected = launch_pins[component_id]
        exact(pin["sha256"], expected["sha256"], f"mono.{component_id}")
        exact(pin["bytes"], expected["bytes"], f"mono.{component_id}.bytes")


def _managed_expectation_components(
    spec_bundle_files: list[dict[str, Any]],
    observed: dict[str, dict[str, Any]],
    root: str,
) -> list[dict[str, Any]]:
    rows = []
    for entry in spec_bundle_files:
        pin = observed[entry["filename"]]
        rows.append(
            {
                "bytes": pin["bytes"],
                "component_id": entry["component_id"],
                "path": f"{root}/{entry['filename']}",
                "sha256": pin["sha256"],
                "stat": dict(pin["stat"]),
            }
        )
    return rows


def _cuda_route_definition(contract: dict[str, Any]) -> dict[str, Any]:
    model_sha = contract["candidate_lock"]["model"]["artifact"]["sha256"]
    flags = {
        "--backend": "CUDA0",
        "--driver-batch": "64",
        "--driver-context": "512",
        "--driver-max-prefill": "64",
        "--layer-end": "40",
        "--layer-start": "0",
        "--mode": "monov3",
        "--model": DESKTOP_MODEL_PATH,
        "--port": str(PORTS["cuda_route"]),
    }
    environment = {
        "CUDA_VISIBLE_DEVICES": contract["devices"]["cuda"]["uuid"],
        "LAYERSPLIT_MEMORY_CERT": "1",
        "LAYERSPLIT_MODEL_SHA256": model_sha,
        "LAYERSPLIT_PLACEMENT_CERT": "1",
        "LD_LIBRARY_PATH": CUDA_ROUTE_ROOT,
    }
    argv = [
        CUDA_ROUTE_ROOT + "/llama-layersplit",
        "--model",
        flags["--model"],
        "--mode",
        "monov3",
        "--backend",
        "CUDA0",
        "--layer-start",
        "0",
        "--layer-end",
        "40",
        "--port",
        flags["--port"],
        "--driver-batch",
        "64",
        "--driver-context",
        "512",
        "--driver-max-prefill",
        "64",
    ]
    route = {
        "argv": argv,
        "cwd": CUDA_ROUTE_ROOT,
        "environment": environment,
        "kind": "local_exec",
    }
    return {"environment": environment, "flags": flags, "route": route}


def _phone_routes(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    model_sha = contract["candidate_lock"]["model"]["artifact"]["sha256"]
    shards = contract["model_route_lock"]["geometry"]["known_shards"]
    return {
        "op12_stagenet": {
            "devices": "GPUOpenCL",
            "driver_batch": 64,
            "driver_context": 512,
            "driver_max_prefill": 64,
            "dynamic_cut": True,
            "kind": "stagenet_worker",
            "kv_unified": True,
            "layer_end": 40,
            "layer_start": 30,
            "mode": "tailv3",
            "model_path": shards["op12"]["path"],
            "model_sha256": shards["op12"]["sha256"],
            "n_gpu_layers": 999,
            "placement_cert": True,
            "port": PORTS["op12_stage"],
            "runtime_root": OP12_STAGE_ROOT,
        },
        "op15_direct_relay": {
            "emit_direct_frames": True,
            "head_host": UNBOUND_RELAY_HEAD_HOST,
            "head_port": PORTS["op15_stage"],
            "kind": "direct_relay",
            "listen_port": PORTS["relay"],
            "runtime_root": OP15_RELAY_ROOT,
            "tail_host": "127.0.0.1",
            "tail_port": PORTS["op12_stage"],
            "tail_source_port": PORTS["relay_tail_source"],
        },
        "op15_stagenet": {
            "devices": "GPUOpenCL",
            "driver_batch": 64,
            "driver_context": 512,
            "driver_max_prefill": 64,
            "dynamic_cut": True,
            "kind": "stagenet_worker",
            "kv_unified": True,
            "layer_end": 30,
            "layer_start": 0,
            "mode": "stagenet",
            "model_path": shards["op15"]["path"],
            "model_sha256": shards["op15"]["sha256"],
            "n_gpu_layers": 999,
            "placement_cert": True,
            "port": PORTS["op15_stage"],
            "runtime_root": OP15_STAGE_ROOT,
        },
    }


def _inline_managed(
    launcher_path: str,
    plan: dict[str, Any],
    boot_id: str,
) -> list[str]:
    raw = compact_json(plan)
    return [
        launcher_path,
        "--plan-json",
        raw,
        "--plan-sha256",
        sha256(raw.encode("ascii")),
        "--boot-id",
        boot_id,
    ]


def build_managed_processes(
    contract: dict[str, Any],
    spec_bundles: dict[str, list[dict[str, Any]]],
    observed: dict[str, dict[str, dict[str, Any]]],
    launcher_pins: dict[str, dict[str, Any]],
    desktop_adb: dict[str, Any],
    bound_boot_ids: dict[str, str],
    phone_ipv4: dict[str, str],
) -> list[dict[str, Any]]:
    cuda_definition = _cuda_route_definition(contract)
    phone_routes = _phone_routes(contract)
    processes = []
    for bundle_id in sorted(MANAGED_BUNDLES):
        endpoint = BUNDLE_ENDPOINTS[bundle_id]
        root = BUNDLE_ROOTS[bundle_id]
        components = _managed_expectation_components(
            spec_bundles[bundle_id],
            observed[bundle_id],
            root,
        )
        launcher_component_id = BUNDLE_LAUNCHERS[bundle_id]
        runtime = next(
            row for row in components
            if row["component_id"] == launcher_component_id
        )
        if endpoint == "cuda":
            managed = launcher_pins["frozen"]
            prospective_route = json.loads(
                canonical_bytes(cuda_definition["route"])
            )
            bound_route = json.loads(canonical_bytes(cuda_definition["route"]))
            expectation = {
                "android": None,
                "bound_route": bound_route,
                "bundle_id": bundle_id,
                "components": components,
                "cuda_environment": dict(cuda_definition["environment"]),
                "cuda_flags": dict(cuda_definition["flags"]),
                "endpoint": endpoint,
                "launcher_component_id": launcher_component_id,
                "managed_launcher": {
                    "path": managed["path"],
                    "sha256": managed["sha256"],
                },
                "mode": "local_cuda",
                "prospective_route": prospective_route,
                "runtime_launcher": {
                    "component_id": launcher_component_id,
                    "path": runtime["path"],
                    "sha256": runtime["sha256"],
                },
            }
        else:
            managed = launcher_pins["usb"]
            serial = contract["devices"][endpoint]["serial"]
            prospective_route = json.loads(
                canonical_bytes(phone_routes[bundle_id])
            )
            bound_route = json.loads(canonical_bytes(phone_routes[bundle_id]))
            if bundle_id == "op15_direct_relay":
                bound_route["head_host"] = phone_ipv4["op12"]
            expectation = {
                "android": {
                    "adb_path": desktop_adb["path"],
                    "adb_port": ADB_PORT,
                    "adb_selector": serial,
                    "adb_sha256": desktop_adb["sha256"],
                    "boot_id_source": "phase_fresh_snapshot",
                    "physical_serial": serial,
                    "shutdown_timeout_ms": 30_000,
                    "startup_timeout_ms": 120_000,
                },
                "bound_route": bound_route,
                "bundle_id": bundle_id,
                "components": components,
                "cuda_environment": None,
                "cuda_flags": None,
                "endpoint": endpoint,
                "launcher_component_id": launcher_component_id,
                "managed_launcher": {
                    "path": managed["path"],
                    "sha256": managed["sha256"],
                },
                "mode": "android",
                "prospective_route": prospective_route,
                "runtime_launcher": {
                    "component_id": launcher_component_id,
                    "path": runtime["path"],
                    "sha256": runtime["sha256"],
                },
            }
        plan_common = {
            "android": expectation["android"],
            "bundle_id": bundle_id,
            "components": components,
            "endpoint": endpoint,
            "launcher_component_id": launcher_component_id,
            "mode": expectation["mode"],
            "schema": "s39-managed-runtime-launch-plan-v1",
            "ssh": None,
        }
        prospective_plan = dict(plan_common)
        prospective_plan["route"] = prospective_route
        bound_plan = dict(plan_common)
        bound_plan["route"] = bound_route
        processes.append(
            {
                "bound": {
                    "argv": _inline_managed(
                        managed["path"],
                        bound_plan,
                        bound_boot_ids[endpoint],
                    ),
                    "boot_id": bound_boot_ids[endpoint],
                },
                "bundle_id": bundle_id,
                "expectation": expectation,
                "process_id": bundle_id,
                "prospective": {
                    "argv": _inline_managed(
                        managed["path"],
                        prospective_plan,
                        UNBOUND_BOOT_IDS[endpoint],
                    ),
                    "boot_id": UNBOUND_BOOT_IDS[endpoint],
                },
            }
        )
    return processes


def build_spec(
    phase_id: str,
    observed: dict[str, dict[str, dict[str, Any]]],
    managed_processes: list[dict[str, Any]],
) -> dict[str, Any]:
    tables = bundle_tables()
    bundles = []
    spec_bundle_files = {}
    for bundle_id in sorted(BUNDLE_ROOTS):
        files = _spec_files(
            tables[bundle_id],
            bundle_id,
            observed[bundle_id],
        )
        spec_bundle_files[bundle_id] = files
        bundles.append(
            {
                "bundle_id": bundle_id,
                "endpoint": BUNDLE_ENDPOINTS[bundle_id],
                "files": files,
                "launcher_component_id": BUNDLE_LAUNCHERS[bundle_id],
                "managed_plan_required": bundle_id in MANAGED_BUNDLES,
                "process_role": bundle_id,
                "root": BUNDLE_ROOTS[bundle_id],
                "transport": (
                    "local"
                    if BUNDLE_ENDPOINTS[bundle_id] == "cuda"
                    else "android"
                ),
            }
        )
    captures = [
        {
            "bundle_id": inv.CAPTURE_BUNDLES[kind],
            "component_id": CAPTURE_ENTRYPOINT_IDS[kind],
            "kind": kind,
        }
        for kind in sorted(inv.CAPTURE_KINDS)
    ]
    return {
        "bundles": bundles,
        "capture_entrypoints": captures,
        "managed_processes": managed_processes,
        "phase": PHASE,
        "phase_id": phase_id,
        "schema": inv.SPEC_SCHEMA,
    }


def _spec_bundle_files(spec: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {bundle["bundle_id"]: bundle["files"] for bundle in spec["bundles"]}


def generate_phase_id(suffix: str | None = None) -> str:
    if suffix is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        suffix = f"{stamp}-{os.urandom(4).hex()}"
    phase_id = PHASE_ID_PREFIX + suffix
    require(
        0 < len(phase_id) <= 128
        and all(
            character.isalnum() or character in "._-" for character in phase_id
        ),
        "E_PHASE_ID",
    )
    return phase_id


def producer_self_pin() -> dict[str, Any]:
    raw = read_file_once(Path(__file__).resolve(), "producer")
    return {
        "bytes": len(raw),
        "path": PRODUCER_RELATIVE_PATH,
        "sha256": sha256(raw),
    }


def write_exclusive(path: Path, raw: bytes) -> None:
    require(not path.is_symlink(), f"E_PUBLISH_SYMLINK: {path}")
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


def publish(root: Path, records: dict[str, bytes]) -> None:
    require(not root.exists(), f"E_OUTPUT_EXISTS: {root}")
    root.mkdir(mode=0o755, parents=False)
    try:
        manifest_lines = []
        for name in PUBLISHED_FILES:
            raw = records[name]
            write_exclusive(root / name, raw)
            manifest_lines.append(f"{sha256(raw)}  {name}\n")
        write_exclusive(
            root / MANIFEST_NAME,
            "".join(manifest_lines).encode("ascii"),
        )
        directory = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        for child in sorted(root.iterdir()):
            child.unlink()
        root.rmdir()
        raise


def materialize(
    *,
    confirmation: str,
    output_root: Path | None = None,
    contract_path: Path = CONTRACT_PATH,
    topology_path: Path = TOPOLOGY_PATH,
    runner: Any = None,
    local_observer: Any = None,
    android_observer: Any = None,
    clock_ns: Any = monotonic_ns,
    wall_ns: Any = utc_ns,
    phase_suffix: str | None = None,
) -> dict[str, Any]:
    exact(confirmation, CONFIRMATION, "confirmation")
    contract, contract_raw = load_contract(contract_path)
    contract_v24 = load_v24_contract(contract)
    topology, topology_sha256 = load_topology(contract_v24, topology_path)
    mono_launch, mono_launch_sha256 = load_mono_launch()
    producer = producer_self_pin()
    phase_id = generate_phase_id(phase_suffix)

    runner = runner or inv.SubprocessRunner()
    capture = Capture(runner, clock_ns)
    started_ns = clock_ns()

    probe_adb_devices(capture, contract)
    phone_state = {
        endpoint: probe_phone(capture, endpoint, contract, topology)
        for endpoint in ("op12", "op15")
    }
    shard_pins = {
        endpoint: pin_phone_shard(capture, endpoint, contract)
        for endpoint in ("op12", "op15")
    }
    desktop = probe_desktop(capture, contract, topology)
    model_pin = pin_desktop_model(capture, contract)
    desktop_adb = pin_desktop_adb(capture)
    ensure_desktop_launcher(
        capture,
        USB_LAUNCHER_SOURCE,
        USB_LAUNCHER_SHA256,
        DESKTOP_USB_LAUNCHER,
        "usb",
    )
    ensure_desktop_launcher(
        capture,
        FROZEN_LAUNCHER_SOURCE,
        FROZEN_LAUNCHER_SHA256,
        DESKTOP_FROZEN_LAUNCHER,
        "frozen",
    )

    local_observer = local_observer or SshDirectoryObserver(runner=runner)
    android_observer = android_observer or inv.AdbDirectoryObserver(
        {
            "op12": contract["devices"]["op12"]["serial"],
            "op15": contract["devices"]["op15"]["serial"],
        },
        runner=runner,
    )
    observed = observe_bundles(local_observer, android_observer)
    verify_pinned_components(observed, contract, contract_v24, mono_launch)

    bound_boot_ids = {
        "cuda": desktop["boot_id"],
        "op12": phone_state["op12"]["boot_id"],
        "op15": phone_state["op15"]["boot_id"],
    }
    launcher_pins = {
        "frozen": {
            "path": DESKTOP_FROZEN_LAUNCHER,
            "sha256": FROZEN_LAUNCHER_SHA256,
        },
        "usb": {"path": DESKTOP_USB_LAUNCHER, "sha256": USB_LAUNCHER_SHA256},
    }
    spec_probe = build_spec(phase_id, observed, [])
    managed_processes = build_managed_processes(
        contract,
        _spec_bundle_files(spec_probe),
        observed,
        launcher_pins,
        desktop_adb,
        bound_boot_ids,
        {
            "op12": phone_state["op12"]["wifi_ipv4"],
            "op15": phone_state["op15"]["wifi_ipv4"],
        },
    )
    spec = build_spec(phase_id, observed, managed_processes)
    record = inv.materialize_runtime_inventory(
        spec,
        local_observer=local_observer,
        android_observer=android_observer,
    )
    completed_ns = clock_ns()
    event_ns = clock_ns()
    require(started_ns < completed_ns <= event_ns, "E_MATERIALIZE_INTERVAL")

    spec_raw = canonical_bytes(spec)
    inventory_raw = canonical_bytes(record)
    lock = {
        "clock_id": CLOCK_ID,
        "completed_ns": completed_ns,
        "contract_sha256": sha256(contract_raw),
        "cuda_monolithic_launch_sha256": mono_launch_sha256,
        "device_boot_ids": bound_boot_ids,
        "device_identities": json.loads(canonical_bytes(contract["devices"])),
        "event_ns": event_ns,
        "event_utc_ns": wall_ns(),
        "inventory_sha256": sha256(inventory_raw),
        "materialization_host": {
            "boot_id": read_file_once(
                Path("/proc/sys/kernel/random/boot_id"), "host_boot_id"
            )
            .decode("ascii")
            .strip(),
            "hostname": os.uname().nodename,
        },
        "model_artifacts": {
            "cuda": model_pin,
            "op12_shard": shard_pins["op12"],
            "op15_shard": shard_pins["op15"],
        },
        "model_id": MODEL_ID,
        "phase": PHASE,
        "phase_id": phase_id,
        "producer": producer,
        "schema": LOCK_SCHEMA,
        "spec_sha256": sha256(spec_raw),
        "started_ns": started_ns,
        "topology_receipt_sha256": topology_sha256,
    }
    exact(set(lock), LOCK_KEYS, "lock.keys")
    lock_raw = canonical_bytes(lock)
    materialization = {
        "captures": capture.rows,
        "completed_ns": completed_ns,
        "inventory_sha256": lock["inventory_sha256"],
        "lock_sha256": sha256(lock_raw),
        "phase": PHASE,
        "phase_id": phase_id,
        "schema": RECORD_SCHEMA,
        "spec_sha256": lock["spec_sha256"],
        "started_ns": started_ns,
        "status": "V2_6_PRODUCTION_MATERIALIZATION_COMPLETE",
    }
    materialization_raw = canonical_bytes(materialization)

    if output_root is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        output_root = HERE / "results" / f"production_materialization_{stamp}"
    publish(
        output_root,
        {
            "RUNTIME_INVENTORY_SPEC_V2_6.json": spec_raw,
            "RUNTIME_INVENTORY_V2_6.json": inventory_raw,
            "PHASE_LOCK_V2_6.json": lock_raw,
            "PRODUCTION_MATERIALIZATION_V2_6.json": materialization_raw,
        },
    )
    validation = validate_production(
        output_root,
        contract_path=contract_path,
        topology_path=topology_path,
    )
    return {
        "output_root": str(output_root),
        "phase_id": phase_id,
        "validation": validation,
    }


def _read_published(root: Path, name: str) -> tuple[dict[str, Any], bytes]:
    try:
        value, raw = authority.read_canonical(root / name)
    except authority.EvidenceError as error:
        raise ProductionError(f"E_PUBLISHED: {name}: {error}") from error
    return value, raw


def validate_production(
    root: Path,
    *,
    contract_path: Path = CONTRACT_PATH,
    topology_path: Path = TOPOLOGY_PATH,
) -> dict[str, Any]:
    require(root.is_dir(), f"E_ROOT: {root}")
    manifest_raw = read_file_once(root / MANIFEST_NAME, "manifest")
    entries = {}
    for line in manifest_raw.decode("ascii").splitlines():
        require("  " in line, "E_MANIFEST_FORMAT")
        digest, name = line.split("  ", 1)
        require(name not in entries, f"E_MANIFEST_DUPLICATE: {name}")
        entries[name] = digest
    exact(sorted(entries), sorted(PUBLISHED_FILES), "manifest.names")
    raws = {}
    values = {}
    for name in PUBLISHED_FILES:
        value, raw = _read_published(root, name)
        exact(sha256(raw), entries[name], f"manifest.{name}")
        raws[name] = raw
        values[name] = value
    extras = sorted(
        item.name
        for item in root.iterdir()
        if item.name not in set(PUBLISHED_FILES) | {MANIFEST_NAME}
    )
    exact(extras, [], "root.extras")

    contract, contract_raw = load_contract(contract_path)
    contract_v24 = load_v24_contract(contract)
    _topology, topology_sha256 = load_topology(contract_v24, topology_path)
    mono_launch, mono_launch_sha256 = load_mono_launch()

    try:
        inventory, inventory_raw = authority.validate_inventory(
            root / "RUNTIME_INVENTORY_V2_6.json",
            contract,
        )
    except authority.EvidenceError as error:
        raise ProductionError(f"E_INVENTORY: {error}") from error
    body = inventory["inventory"]

    spec = values["RUNTIME_INVENTORY_SPEC_V2_6.json"]
    try:
        inv._validate_spec(json.loads(canonical_bytes(spec)))
    except inv.InventoryError as error:
        raise ProductionError(f"E_SPEC: {error}") from error
    spec_raw = raws["RUNTIME_INVENTORY_SPEC_V2_6.json"]
    exact(sha256(spec_raw), body["spec_sha256"], "spec.sha256")
    exact(spec["phase_id"], body["phase_id"], "spec.phase_id")

    lock = values["PHASE_LOCK_V2_6.json"]
    lock_raw = raws["PHASE_LOCK_V2_6.json"]
    require(type(lock) is dict, "E_LOCK_TYPE")
    exact(set(lock), LOCK_KEYS, "lock.keys")
    exact(lock["schema"], LOCK_SCHEMA, "lock.schema")
    exact(lock["phase"], PHASE, "lock.phase")
    exact(lock["model_id"], MODEL_ID, "lock.model_id")
    exact(lock["clock_id"], CLOCK_ID, "lock.clock_id")
    phase_id = lock["phase_id"]
    require(
        type(phase_id) is str and phase_id.startswith(PHASE_ID_PREFIX),
        "E_LOCK_PHASE_ID",
    )
    exact(phase_id, body["phase_id"], "lock.inventory_phase_id")
    exact(lock["contract_sha256"], sha256(contract_raw), "lock.contract_sha256")
    exact(lock["spec_sha256"], sha256(spec_raw), "lock.spec_sha256")
    exact(
        lock["inventory_sha256"],
        sha256(inventory_raw),
        "lock.inventory_sha256",
    )
    exact(
        lock["topology_receipt_sha256"],
        topology_sha256,
        "lock.topology_receipt_sha256",
    )
    exact(
        lock["cuda_monolithic_launch_sha256"],
        mono_launch_sha256,
        "lock.cuda_monolithic_launch_sha256",
    )
    exact(
        lock["device_identities"],
        json.loads(canonical_bytes(contract["devices"])),
        "lock.device_identities",
    )

    started_ns = lock["started_ns"]
    completed_ns = lock["completed_ns"]
    event_ns = lock["event_ns"]
    for name, value in (
        ("started_ns", started_ns),
        ("completed_ns", completed_ns),
        ("event_ns", event_ns),
        ("event_utc_ns", lock["event_utc_ns"]),
    ):
        require(type(value) is int and value > 0, f"E_LOCK_TIME: {name}")
    require(
        started_ns < completed_ns <= event_ns,
        "E_LOCK_ORDER",
    )

    boot_ids = lock["device_boot_ids"]
    require(
        type(boot_ids) is dict and set(boot_ids) == {"cuda", "op12", "op15"},
        "E_LOCK_BOOT_KEYS",
    )
    require(
        len(set(boot_ids.values())) == 3
        and not set(boot_ids.values()) & set(UNBOUND_BOOT_IDS.values()),
        "E_LOCK_BOOT_REUSE",
    )
    endpoint_boots = {}
    for process in body["managed_processes"]:
        endpoint = BUNDLE_ENDPOINTS[process["bundle_id"]]
        exact(
            process["bound"]["boot_id"],
            boot_ids[endpoint],
            f"lock.bound_boot.{process['bundle_id']}",
        )
        exact(
            process["prospective"]["boot_id"],
            UNBOUND_BOOT_IDS[endpoint],
            f"lock.prospective_boot.{process['bundle_id']}",
        )
        endpoint_boots[endpoint] = process["bound"]["boot_id"]
    exact(sorted(endpoint_boots), ["cuda", "op12", "op15"], "lock.managed_endpoints")

    artifacts = lock["model_artifacts"]
    require(
        type(artifacts) is dict
        and set(artifacts) == {"cuda", "op12_shard", "op15_shard"},
        "E_LOCK_ARTIFACT_KEYS",
    )
    candidate_artifact = contract["candidate_lock"]["model"]["artifact"]
    geometry = contract["model_route_lock"]["geometry"]
    exact(
        artifacts["cuda"],
        {
            "bytes": candidate_artifact["bytes"],
            "path": geometry["cuda_model_path"],
            "sha256": candidate_artifact["sha256"],
        },
        "lock.model_artifacts.cuda",
    )
    for endpoint in ("op12", "op15"):
        known = geometry["known_shards"][endpoint]
        exact(
            artifacts[f"{endpoint}_shard"],
            {
                "bytes": known["bytes"],
                "path": known["path"],
                "sha256": known["sha256"],
            },
            f"lock.model_artifacts.{endpoint}",
        )

    producer = lock["producer"]
    exact(producer.get("path"), PRODUCER_RELATIVE_PATH, "lock.producer.path")
    producer_raw = read_file_once(S39 / PRODUCER_RELATIVE_PATH, "producer")
    exact(producer.get("bytes"), len(producer_raw), "lock.producer.bytes")
    exact(producer.get("sha256"), sha256(producer_raw), "lock.producer.sha256")

    host = lock["materialization_host"]
    require(
        type(host) is dict and set(host) == {"boot_id", "hostname"},
        "E_LOCK_HOST",
    )

    verify_inventory_pins(body, contract, contract_v24, mono_launch)
    verify_managed_semantics(body, spec, contract, lock, _topology)

    record = values["PRODUCTION_MATERIALIZATION_V2_6.json"]
    exact(record.get("schema"), RECORD_SCHEMA, "record.schema")
    exact(record.get("phase"), PHASE, "record.phase")
    exact(record.get("phase_id"), phase_id, "record.phase_id")
    exact(record.get("spec_sha256"), lock["spec_sha256"], "record.spec_sha256")
    exact(
        record.get("inventory_sha256"),
        lock["inventory_sha256"],
        "record.inventory_sha256",
    )
    exact(record.get("lock_sha256"), sha256(lock_raw), "record.lock_sha256")
    exact(record.get("started_ns"), started_ns, "record.started_ns")
    exact(record.get("completed_ns"), completed_ns, "record.completed_ns")
    exact(
        record.get("status"),
        "V2_6_PRODUCTION_MATERIALIZATION_COMPLETE",
        "record.status",
    )
    captures = record.get("captures")
    require(type(captures) is list and bool(captures), "E_RECORD_CAPTURES")
    previous = started_ns
    for index, row in enumerate(captures):
        require(
            type(row) is dict
            and set(row)
            == {"argv", "capture_id", "completed_ns", "started_ns", "stdout_sha256"},
            f"E_RECORD_CAPTURE: {index}",
        )
        require(
            started_ns <= row["started_ns"] < row["completed_ns"] <= completed_ns,
            f"E_RECORD_CAPTURE_INTERVAL: {index}",
        )
        require(previous <= row["started_ns"], f"E_RECORD_CAPTURE_ORDER: {index}")
        previous = row["started_ns"]

    return {
        "inventory_sha256": lock["inventory_sha256"],
        "lock_sha256": sha256(lock_raw),
        "phase_id": phase_id,
        "runtime_component_count": len(body["components"]),
        "schema": VALIDATION_SCHEMA,
        "spec_sha256": lock["spec_sha256"],
        "status": "V2_6_PRODUCTION_MATERIALIZATION_PASS",
    }


def verify_inventory_pins(
    body: dict[str, Any],
    contract: dict[str, Any],
    contract_v24: dict[str, Any],
    mono_launch: dict[str, Any],
) -> None:
    components = {row["component_id"]: row for row in body["components"]}
    producers = contract["composition"]["capture_producers"]
    v24_programs = contract_v24["producer_requirements"]["source_programs"]
    entrypoint_pins = {
        "cuda-mono.capture-artifact-root": producers["artifact_root"],
        "cuda-mono.capture-cuda-monolithic": v24_programs["cuda_monolithic"],
        "cuda-route.capture-fast-fresh-readiness": producers[
            "fast_fresh_readiness"
        ],
        "cuda-route.capture-joint-phone-cuda": v24_programs["joint_phone_cuda"],
    }
    for component_id, pin in entrypoint_pins.items():
        row = components.get(component_id)
        require(row is not None, f"E_PIN_MISSING: {component_id}")
        exact(row["sha256"], pin["sha256"], f"pin.{component_id}")
        exact(row["bytes"], pin["bytes"], f"pin.{component_id}.bytes")
        exact(row["role"], "capture_entrypoint", f"pin.{component_id}.role")
    launch_pins = mono_component_pins(mono_launch)
    for component_id, pin in launch_pins.items():
        row = components.get(component_id)
        require(row is not None, f"E_PIN_MISSING: {component_id}")
        exact(row["sha256"], pin["sha256"], f"pin.{component_id}")
        exact(row["bytes"], pin["bytes"], f"pin.{component_id}.bytes")
    for endpoint, root in (
        ("op12", OP12_STAGE_ROOT),
        ("op15", OP15_STAGE_ROOT),
    ):
        bundle = next(
            row
            for row in body["bundles"]
            if row["bundle_id"] == f"{endpoint}_stagenet"
        )
        exact(bundle["root"], root, f"pin.root.{endpoint}")


def verify_managed_semantics(
    body: dict[str, Any],
    spec: dict[str, Any],
    contract: dict[str, Any],
    lock: dict[str, Any],
    topology: dict[str, Any],
) -> None:
    """Recompute every managed process from external anchors and compare.

    Anchors the exact argv arrays, launch plans, and routes to the frozen
    contract, the validated topology receipt, the recorded bound boot IDs,
    and the digest-bound observed components. Only android.adb_sha256 has no
    external anchor; it is forced to one shared value across all android
    processes.
    """

    observed = {bundle_id: {} for bundle_id in BUNDLE_ROOTS}
    for row in body["components"]:
        observed[row["bundle_id"]][Path(row["path"]).name] = {
            "bytes": row["bytes"],
            "path": row["path"],
            "sha256": row["sha256"],
            "stat": row["stat"],
        }
    android_shas = {
        process["expectation"]["android"]["adb_sha256"]
        for process in body["managed_processes"]
        if process["expectation"]["mode"] == "android"
    }
    require(len(android_shas) == 1, "E_MANAGED_ADB_SHA")
    expected = build_managed_processes(
        contract,
        _spec_bundle_files(spec),
        observed,
        {
            "frozen": {
                "path": DESKTOP_FROZEN_LAUNCHER,
                "sha256": FROZEN_LAUNCHER_SHA256,
            },
            "usb": {
                "path": DESKTOP_USB_LAUNCHER,
                "sha256": USB_LAUNCHER_SHA256,
            },
        },
        {"path": DESKTOP_ADB_PATH, "sha256": android_shas.pop()},
        lock["device_boot_ids"],
        {
            endpoint: topology["observed"]["phones"][endpoint]["wifi_ipv4"]
            for endpoint in ("op12", "op15")
        },
    )
    recorded = [
        {
            "bound": process["bound"],
            "bundle_id": process["bundle_id"],
            "expectation": process["expectation"],
            "process_id": process["process_id"],
            "prospective": process["prospective"],
        }
        for process in body["managed_processes"]
    ]
    exact(
        json.loads(canonical_bytes(recorded)),
        json.loads(canonical_bytes(expected)),
        "managed.recompute",
    )
    for process in body["managed_processes"]:
        pin = process["managed_launcher_pin"]
        expected_launcher = (
            FROZEN_LAUNCHER_SHA256
            if process["expectation"]["mode"] == "local_cuda"
            else USB_LAUNCHER_SHA256
        )
        exact(
            pin["sha256"],
            expected_launcher,
            f"managed.launcher.{process['process_id']}",
        )


def live_check(
    root: Path,
    *,
    runner: Any = None,
    clock_ns: Any = monotonic_ns,
) -> dict[str, Any]:
    lock, _raw = _read_published(root, "PHASE_LOCK_V2_6.json")
    exact(set(lock), LOCK_KEYS, "lock.keys")
    runner = runner or inv.SubprocessRunner()
    capture = Capture(runner, clock_ns)
    serials = {
        endpoint: lock["device_identities"][endpoint]["serial"]
        for endpoint in ("op12", "op15")
    }
    for endpoint, serial in sorted(serials.items()):
        boot_id = _parse_key_value(
            capture.run(
                f"live.{endpoint}.boot_id",
                adb_argv(
                    serial, "shell", "cat /proc/sys/kernel/random/boot_id"
                ),
                30,
            ),
            f"live.{endpoint}.boot_id",
        )
        exact(
            boot_id,
            lock["device_boot_ids"][endpoint],
            f"live.{endpoint}.boot_id",
        )
    desktop_boot = _parse_key_value(
        capture.run(
            "live.cuda.boot_id",
            ssh_argv("cat /proc/sys/kernel/random/boot_id"),
            30,
        ),
        "live.cuda.boot_id",
    )
    exact(desktop_boot, lock["device_boot_ids"]["cuda"], "live.cuda.boot_id")
    return {
        "phase_id": lock["phase_id"],
        "schema": "s39-cp0-r1-v26-production-live-check-v1",
        "status": "V2_6_PRODUCTION_BOOT_IDENTITY_LIVE",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("materialize")
    run.add_argument("--output-root", type=Path, default=None)
    run.add_argument("--topology", type=Path, default=TOPOLOGY_PATH)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--confirm", default="")
    check = commands.add_parser("validate")
    check.add_argument("--root", type=Path, required=True)
    live = commands.add_parser("live-check")
    live.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "materialize":
            require(args.execute, "E_EXECUTE_REQUIRED")
            result = materialize(
                confirmation=args.confirm,
                output_root=args.output_root,
                topology_path=args.topology.resolve(),
            )
        elif args.command == "validate":
            result = validate_production(args.root)
        else:
            result = live_check(args.root)
    except (ProductionError, inv.InventoryError, OSError) as error:
        print(
            f"V2_6_PRODUCTION_MATERIALIZATION_REFUSED: {error}",
            file=sys.stderr,
        )
        return 2
    sys.stdout.write(canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
