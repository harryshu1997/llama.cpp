#!/usr/bin/env python3
"""Capture the desktop A_ONLY runtime inventory without publishing unsafe input."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
from typing import Any, Protocol


RUNTIME_SCHEMA = "s39-cp0-r1-runtime-bundle-closure-input-v1"
OPERATOR_SCHEMA = "s39-cp0-r1-v24-desktop-operator-input-v1"
INVENTORY_SCHEMA = "s39-cp0-r1-v24-a-only-desktop-inventory-v1"
MANAGED_PLAN_SCHEMA = "s39-managed-runtime-launch-plan-v1"
MODEL_ID = "qwen3-14b-q4_k_m"
MODEL_PATH = "/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf"
MODEL_SHA256 = (
    "500a8806e85ee9c83f3ae084202955924"
    "51379b4f8cf2d0f41c15dffeb6b81f0"
)
MODEL_BYTES = 9_001_752_960
CUDA_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
CONTROLLER_HOST = "zhihao-Z690-C-ac"
ADB_PATH = "/usr/lib/android-sdk/platform-tools/adb"
ADB_PORT = 5038
CUDA_MONOLITHIC_ROOT = "/home/zhihao/llama.cpp-s40/build-s40-cuda/bin"
CUDA_ROUTE_ROOT = "/home/zhihao/s39-v24-a-only/runtime-v1/cuda-route"
OP15_STAGE_ROOT = "/data/local/tmp/s39-v24-a-only/runtime-v1/op15-stagenet"
OP15_RELAY_ROOT = "/data/local/tmp/s39-v24-a-only/runtime-v1/op15-direct-relay"
OP12_STAGE_ROOT = "/data/local/tmp/s39-v24-a-only/runtime-v1/op12-stagenet"
SHARD_PATH = (
    "/data/local/tmp/s39-active-warm/v1/models/"
    "qwen3-14b-q4_k_m/weights.gguf"
)
UNBOUND_BOOT_IDS = {
    "cuda": "00000000-0000-0000-0000-000000000000",
    "op12": "00000000-0000-0000-0000-000000000001",
    "op15": "00000000-0000-0000-0000-000000000002",
}
UNBOUND_PHONE_NETWORK = {
    "op12": {"interface": "UNBOUND_AFTER_REBOOT", "local_ipv4": "0.0.0.12"},
    "op15": {"interface": "UNBOUND_AFTER_REBOOT", "local_ipv4": "0.0.0.15"},
}
PHONES = {
    "op12": {
        "device": "OP595DL1",
        "executed_layers": [30, 40],
        "model": "CPH2583",
        "product": "CPH2583",
        "serial": "5ae7a43d",
        "shard_sha256": (
            "72e312af745160dc33a0ba39ba94fbbc"
            "e6112950d0409d39c42ddc3b25e756ab"
        ),
        "stored_layers": [24, 40],
    },
    "op15": {
        "device": "OP611FL1",
        "executed_layers": [0, 30],
        "model": "CPH2749",
        "product": "CPH2749",
        "serial": "3C15AU002CL00000",
        "shard_sha256": (
            "ba56b9c5e19b3a4512777e6a47803cc"
            "03261c2d3c2734965cd5ec96b7c6c59fb"
        ),
        "stored_layers": [0, 32],
    },
}
ROOTS = {
    "cuda_monolithic": CUDA_MONOLITHIC_ROOT,
    "cuda_route": CUDA_ROUTE_ROOT,
    "op12_stagenet": OP12_STAGE_ROOT,
    "op15_direct_relay": OP15_RELAY_ROOT,
    "op15_stagenet": OP15_STAGE_ROOT,
}
CUDA_ROUTE_FILES = {
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
PORTS = {
    "adb_server": 5038,
    "cuda_monolithic": 39124,
    "cuda_route": 39125,
    "op12_stage": 39126,
    "op15_stage": 39127,
    "relay": 39128,
    "relay_tail_source": 39129,
}
INPUT_SCHEMAS = {
    "candidate": "s39-cp0-r1-candidate-v1",
    "contract": "s39-cp0-r1-evidence-contract-v2.4",
    "cuda_monolithic_launch": "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
    "operator_input": OPERATOR_SCHEMA,
    "runtime_bundle_inventory": RUNTIME_SCHEMA,
    "token_history": "s39-cp0-r1-token-history-v2.4",
    "tokenizer_plan": "s39-cp0-r1-a-only-tokenizer-plan-v2",
    "topology_receipt": "s39-cp0-r1-v24-desktop-topology-v1",
}


class CaptureError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CaptureError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}: expected {expected!r}, got {value!r}",
    )


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stat_record(metadata: os.stat_result) -> dict[str, int]:
    return {
        "ctime_ns": metadata.st_ctime_ns,
        "device_id": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "mtime_ns": metadata.st_mtime_ns,
        "size": metadata.st_size,
    }


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return stat_record(left) == stat_record(right)


def secure_local_directory(path: Path) -> None:
    require(path.is_absolute(), f"E_LOCAL_DIRECTORY_PATH: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        metadata = os.lstat(current)
        require(
            stat.S_ISDIR(metadata.st_mode),
            f"E_LOCAL_DIRECTORY_TYPE: {current}",
        )


def secure_local_pin(path: Path, *, executable: bool = False) -> dict[str, Any]:
    require(path.is_absolute(), f"E_LOCAL_PATH: {path}")
    before = os.lstat(path)
    require(stat.S_ISREG(before.st_mode), f"E_LOCAL_NOT_REGULAR: {path}")
    if executable:
        require(before.st_mode & 0o111 != 0, f"E_LOCAL_NOT_EXECUTABLE: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        require(_same_stat(before, opened), f"E_LOCAL_OPEN_RACE: {path}")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.lstat(path)
    require(
        _same_stat(opened, after_fd) and _same_stat(after_fd, after_path),
        f"E_LOCAL_MUTATION: {path}",
    )
    exact(total, opened.st_size, f"local.size.{path}")
    return {
        "bytes": total,
        "path": str(path),
        "sha256": digest.hexdigest(),
        "stat": stat_record(opened),
    }


def read_canonical(path: Path, schema: str | None = None) -> tuple[dict[str, Any], bytes]:
    pin = secure_local_pin(path)
    require(pin["bytes"] <= 64 * 1024 * 1024, f"E_JSON_SIZE: {path}")
    raw = path.read_bytes()
    after = os.lstat(path)
    exact(stat_record(after), pin["stat"], f"json.mutation.{path}")
    exact(len(raw), pin["bytes"], f"json.bytes.{path}")
    exact(sha256_bytes(raw), pin["sha256"], f"json.sha256.{path}")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                CaptureError(f"E_JSON_NUMBER: {item}")
            ),
            parse_float=lambda item: (_ for _ in ()).throw(
                CaptureError(f"E_JSON_FLOAT: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"E_JSON: {path}") from error
    require(type(value) is dict, f"E_JSON_TYPE: {path}")
    exact(canonical_bytes(value), raw, f"canonical.{path}")
    if schema is not None:
        exact(value.get("schema"), schema, f"schema.{path}")
    return value, raw


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"E_JSON_DUPLICATE: {key}")
        result[key] = value
    return result


class Runner(Protocol):
    def run(self, argv: list[str], timeout: int) -> bytes:
        ...


class SubprocessRunner:
    def run(self, argv: list[str], timeout: int) -> bytes:
        result = subprocess.run(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if result.returncode != 0:
            message = result.stderr.decode("ascii", errors="replace").strip()
            raise CaptureError(
                f"E_COMMAND: rc={result.returncode}: {argv!r}: {message}"
            )
        return result.stdout


def _adb(serial: str, *arguments: str) -> list[str]:
    return [ADB_PATH, "-P", str(ADB_PORT), "-s", serial, *arguments]


def _android_time(value: str, seconds: int, field: str) -> int:
    try:
        base, zone = value.rsplit(" ", 1)
        whole, fraction = base.rsplit(".", 1)
        timestamp = datetime.datetime.strptime(
            f"{whole} {zone}",
            "%Y-%m-%d %H:%M:%S %z",
        )
        result = int(timestamp.timestamp()) * 1_000_000_000 + int(
            fraction.ljust(9, "0")
        )
    except (ValueError, OverflowError) as error:
        raise CaptureError(f"E_ANDROID_TIME: {field}") from error
    exact(result // 1_000_000_000, seconds, f"{field}.seconds")
    return result


def parse_android_stat(raw: bytes, field: str) -> dict[str, int]:
    try:
        line = raw.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise CaptureError(f"E_ANDROID_STAT_ASCII: {field}") from error
    keys = (
        "DEV",
        "INO",
        "SIZE",
        "MODE",
        "MTIME_S",
        "MTIME",
        "CTIME_S",
        "CTIME",
    )
    parts = line.split("|")
    require(len(parts) == len(keys), f"E_ANDROID_STAT_FIELDS: {field}")
    values: dict[str, str] = {}
    for key, part in zip(keys, parts):
        prefix = f"{key}="
        require(part.startswith(prefix), f"E_ANDROID_STAT_FIELD: {field}.{key}")
        values[key] = part[len(prefix):]
    try:
        seconds_mtime = int(values["MTIME_S"])
        seconds_ctime = int(values["CTIME_S"])
        result = {
            "ctime_ns": _android_time(values["CTIME"], seconds_ctime, field),
            "device_id": int(values["DEV"]),
            "inode": int(values["INO"]),
            "mode": int(values["MODE"], 16),
            "mtime_ns": _android_time(values["MTIME"], seconds_mtime, field),
            "size": int(values["SIZE"]),
        }
    except ValueError as error:
        raise CaptureError(f"E_ANDROID_STAT_INTEGER: {field}") from error
    require(
        result["inode"] > 0
        and result["size"] > 0
        and stat.S_ISREG(result["mode"]),
        f"E_REMOTE_NOT_REGULAR: {field}",
    )
    return result


def _remote_stat_command(serial: str, path: str) -> list[str]:
    quoted = shlex.quote(path)
    stat_format = (
        "DEV=%d|INO=%i|SIZE=%s|MODE=%f|"
        "MTIME_S=%Y|MTIME=%y|CTIME_S=%Z|CTIME=%z"
    )
    script = "; ".join(
        (
            "set -eu",
            f"test -f {quoted}",
            f"test ! -L {quoted}",
            f"stat -c {shlex.quote(stat_format)} -- {quoted}",
        )
    )
    return _adb(serial, "shell", "sh -c " + shlex.quote(script))


def remote_pin(
    runner: Runner,
    serial: str,
    path: str,
    *,
    executable: bool = False,
) -> dict[str, Any]:
    before = parse_android_stat(
        runner.run(_remote_stat_command(serial, path), 30),
        f"{serial}:{path}.before",
    )
    digest_raw = runner.run(
        _adb(
            serial,
            "shell",
            "sha256sum -- " + shlex.quote(path),
        ),
        600,
    )
    digest_check_raw = runner.run(
        _adb(
            serial,
            "shell",
            "sha256sum -- " + shlex.quote(path),
        ),
        600,
    )
    exact(digest_check_raw, digest_raw, f"remote.digest_mutation.{serial}:{path}")
    after = parse_android_stat(
        runner.run(_remote_stat_command(serial, path), 30),
        f"{serial}:{path}.after",
    )
    exact(after, before, f"remote.mutation.{serial}:{path}")
    try:
        digest_fields = digest_raw.decode("ascii").strip().split()
    except UnicodeDecodeError as error:
        raise CaptureError(f"E_REMOTE_DIGEST_ASCII: {path}") from error
    require(
        len(digest_fields) == 2
        and len(digest_fields[0]) == 64
        and all(character in "0123456789abcdef" for character in digest_fields[0]),
        f"E_REMOTE_DIGEST: {path}",
    )
    if executable:
        require(before["mode"] & 0o111 != 0, f"E_REMOTE_NOT_EXECUTABLE: {path}")
    return {
        "bytes": before["size"],
        "path": path,
        "sha256": digest_fields[0],
        "stat": before,
    }


def _phone_identity(runner: Runner, endpoint: str) -> None:
    expected = PHONES[endpoint]
    serial = expected["serial"]
    probes = {
        "serial": "ro.serialno",
        "product": "ro.product.name",
        "model": "ro.product.model",
        "device": "ro.product.device",
    }
    for field, property_name in probes.items():
        raw = runner.run(_adb(serial, "shell", "getprop", property_name), 30)
        try:
            observed = raw.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise CaptureError(f"E_PHONE_ASCII: {endpoint}.{field}") from error
        exact(observed, expected[field], f"phone.{endpoint}.{field}")


def _remote_names(runner: Runner, endpoint: str, root: str) -> list[str]:
    serial = PHONES[endpoint]["serial"]
    quoted = shlex.quote(root)
    raw = runner.run(
        _adb(
            serial,
            "shell",
            "sh -c "
            + shlex.quote(
                f"set -eu; test -d {quoted}; "
                f"test ! -L {quoted}; LC_ALL=C ls -1A -- {quoted}"
            ),
        ),
        30,
    )
    try:
        names = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise CaptureError(f"E_REMOTE_LIST_ASCII: {endpoint}:{root}") from error
    require(
        all(
            name
            and "/" not in name
            and name not in {".", ".."}
            and name.isascii()
            for name in names
        ),
        f"E_REMOTE_LIST_NAME: {endpoint}:{root}",
    )
    require(names == sorted(set(names)), f"E_REMOTE_LIST_ORDER: {endpoint}:{root}")
    return names


def _local_components(
    root: Path,
    bundle_id: str,
    endpoint: str,
    files: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    secure_local_directory(root)
    actual = sorted(item.name for item in root.iterdir())
    exact(actual, sorted(files), f"closure.{bundle_id}")
    result = []
    for filename, (component_id, role) in sorted(files.items()):
        pin = secure_local_pin(root / filename, executable=role == "executable")
        result.append(
            {
                "bundle_id": bundle_id,
                **pin,
                "component_id": component_id,
                "endpoint": endpoint,
                "role": role,
            }
        )
    return result


def _phone_components(
    runner: Runner,
    endpoint: str,
    root: str,
    bundle_id: str,
    files: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    exact(_remote_names(runner, endpoint, root), sorted(files), f"closure.{bundle_id}")
    result = []
    serial = PHONES[endpoint]["serial"]
    for filename, (suffix, role) in sorted(files.items()):
        component_id = (
            suffix if suffix.startswith(bundle_id) else f"{bundle_id}.{suffix}"
        )
        pin = remote_pin(
            runner,
            serial,
            f"{root}/{filename}",
            executable=role == "executable",
        )
        result.append(
            {
                "bundle_id": bundle_id,
                **pin,
                "component_id": component_id,
                "endpoint": endpoint,
                "role": role,
            }
        )
    return result


def capture_runtime_inventory(
    monolithic_launch_path: Path,
    *,
    runner: Runner | None = None,
) -> dict[str, Any]:
    runner = runner or SubprocessRunner()
    monolithic, _raw = read_canonical(
        monolithic_launch_path,
        "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
    )
    exact(monolithic["bundle_root"], CUDA_MONOLITHIC_ROOT, "monolithic.root")
    exact(monolithic["model_sha256"], MODEL_SHA256, "monolithic.model")
    _phone_identity(runner, "op12")
    _phone_identity(runner, "op15")

    components: list[dict[str, Any]] = []
    monolithic_root = Path(monolithic["bundle_root"])
    secure_local_directory(monolithic_root)
    monolithic_names = sorted(
        Path(expected["path"]).name
        for expected in monolithic["required_components"]
    )
    exact(
        sorted(item.name for item in monolithic_root.iterdir()),
        monolithic_names,
        "closure.cuda_monolithic",
    )
    for expected in monolithic["required_components"]:
        pin = secure_local_pin(
            Path(expected["path"]),
            executable=expected["component_id"] == monolithic["launcher_component_id"],
        )
        exact(pin["sha256"], expected["sha256"], f"monolithic.{expected['component_id']}")
        exact(pin["stat"], expected["stat"], f"monolithic.stat.{expected['component_id']}")
        components.append(
            {
                "bundle_id": "cuda_monolithic",
                **pin,
                "component_id": expected["component_id"],
                "endpoint": "cuda",
                "role": (
                    "executable"
                    if expected["component_id"] == monolithic["launcher_component_id"]
                    else "backend_library"
                    if "cuda" in expected["component_id"]
                    else "shared_library"
                ),
            }
        )

    components.extend(
        _local_components(
            Path(CUDA_ROUTE_ROOT),
            "cuda_route",
            "cuda",
            CUDA_ROUTE_FILES,
        )
    )
    for endpoint, bundle_id, root in (
        ("op12", "op12_stagenet", OP12_STAGE_ROOT),
        ("op15", "op15_stagenet", OP15_STAGE_ROOT),
    ):
        files = dict(PHONE_STAGE_FILES)
        htp_name, htp_component = PHONE_STAGE_HVX[bundle_id]
        files[htp_name] = (htp_component, "backend_library")
        components.extend(
            _phone_components(runner, endpoint, root, bundle_id, files)
        )
    components.extend(
        _phone_components(
            runner,
            "op15",
            OP15_RELAY_ROOT,
            "op15_direct_relay",
            RELAY_FILES,
        )
    )
    components.sort(key=lambda item: item["component_id"])
    require(
        len({item["component_id"] for item in components}) == len(components),
        "E_COMPONENT_ID_REUSE",
    )

    launchers = {
        "cuda_monolithic": monolithic["launcher_component_id"],
        "cuda_route": "cuda-route.bin",
        "op12_stagenet": "op12_stagenet.bin",
        "op15_direct_relay": "op15_direct_relay.bin",
        "op15_stagenet": "op15_stagenet.bin",
    }
    endpoints = {
        "cuda_monolithic": "cuda",
        "cuda_route": "cuda",
        "op12_stagenet": "op12",
        "op15_direct_relay": "op15",
        "op15_stagenet": "op15",
    }
    bundles = []
    for bundle_id in sorted(ROOTS):
        required = sorted(
            item["component_id"]
            for item in components
            if item["bundle_id"] == bundle_id
        )
        require(launchers[bundle_id] in required, f"E_LAUNCHER: {bundle_id}")
        if bundle_id == "op15_direct_relay":
            exact(required, [launchers[bundle_id]], "relay.closure")
        else:
            require(
                any(
                    item["role"] in {"backend_library", "shared_library"}
                    for item in components
                    if item["bundle_id"] == bundle_id
                ),
                f"E_LIBRARY: {bundle_id}",
            )
        bundles.append(
            {
                "bundle_id": bundle_id,
                "endpoint": endpoints[bundle_id],
                "launcher_component_id": launchers[bundle_id],
                "process_role": bundle_id,
                "required_component_ids": required,
            }
        )
    return {
        "bundle_roots": dict(sorted(ROOTS.items())),
        "bundles": bundles,
        "closure_complete": True,
        "components": components,
        "schema": RUNTIME_SCHEMA,
    }


def _component_map(runtime: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["component_id"]: item for item in runtime["components"]}


def _bundle(runtime: dict[str, Any], bundle_id: str) -> dict[str, Any]:
    return next(item for item in runtime["bundles"] if item["bundle_id"] == bundle_id)


def _managed_components(
    runtime: dict[str, Any],
    bundle_id: str,
) -> list[dict[str, Any]]:
    components = _component_map(runtime)
    return [
        {
            "bytes": components[component_id]["bytes"],
            "component_id": component_id,
            "path": components[component_id]["path"],
            "sha256": components[component_id]["sha256"],
            "stat": {
                "build_id": None,
                **components[component_id]["stat"],
            },
        }
        for component_id in _bundle(runtime, bundle_id)["required_component_ids"]
    ]


def _inline_managed(
    launcher: dict[str, Any],
    plan: dict[str, Any],
    boot_id: str,
) -> list[str]:
    raw = compact_json(plan)
    return [
        launcher["path"],
        "--plan-json",
        raw,
        "--plan-sha256",
        sha256_bytes(raw.encode("ascii")),
        "--boot-id",
        boot_id,
    ]


def build_local_cuda_plan(
    runtime: dict[str, Any],
    cuda_launcher: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    bundle = _bundle(runtime, "cuda_route")
    components = _component_map(runtime)
    runtime_executable = components[bundle["launcher_component_id"]]
    target_argv = [
        runtime_executable["path"],
        "--model",
        MODEL_PATH,
        "--mode",
        "monov3",
        "--backend",
        "CUDA0",
        "--layer-start",
        "0",
        "--layer-end",
        "40",
        "--port",
        str(PORTS["cuda_route"]),
        "--driver-batch",
        "64",
        "--driver-context",
        "512",
        "--driver-max-prefill",
        "64",
    ]
    environment = {
        "CUDA_VISIBLE_DEVICES": CUDA_UUID,
        "LAYERSPLIT_MEMORY_CERT": "1",
        "LAYERSPLIT_MODEL_SHA256": MODEL_SHA256,
        "LAYERSPLIT_PLACEMENT_CERT": "1",
        "LD_LIBRARY_PATH": CUDA_ROUTE_ROOT,
    }
    plan = {
        "android": None,
        "bundle_id": "cuda_route",
        "components": _managed_components(runtime, "cuda_route"),
        "endpoint": "cuda",
        "launcher_component_id": bundle["launcher_component_id"],
        "mode": "local_cuda",
        "route": {
            "argv": target_argv,
            "cwd": CUDA_ROUTE_ROOT,
            "environment": environment,
            "kind": "local_exec",
        },
        "schema": MANAGED_PLAN_SCHEMA,
        "ssh": None,
    }
    return plan, _inline_managed(cuda_launcher, plan, UNBOUND_BOOT_IDS["cuda"])


def _android_plan(
    runtime: dict[str, Any],
    bundle_id: str,
    endpoint: str,
    route: dict[str, Any],
    adb_pin: dict[str, Any],
) -> dict[str, Any]:
    bundle = _bundle(runtime, bundle_id)
    serial = PHONES[endpoint]["serial"]
    return {
        "android": {
            "adb_path": adb_pin["path"],
            "adb_port": ADB_PORT,
            "adb_selector": serial,
            "adb_sha256": adb_pin["sha256"],
            "boot_id_source": "phase_fresh_snapshot",
            "physical_serial": serial,
            "shutdown_timeout_ms": 30_000,
            "startup_timeout_ms": 120_000,
        },
        "bundle_id": bundle_id,
        "components": _managed_components(runtime, bundle_id),
        "endpoint": endpoint,
        "launcher_component_id": bundle["launcher_component_id"],
        "mode": "android",
        "route": route,
        "schema": MANAGED_PLAN_SCHEMA,
        "ssh": None,
    }


def build_phone_processes(
    runtime: dict[str, Any],
    usb_launcher: dict[str, Any],
    adb_pin: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    routes = {
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
            "model_path": SHARD_PATH,
            "model_sha256": MODEL_SHA256,
            "n_gpu_layers": 999,
            "placement_cert": True,
            "port": PORTS["op12_stage"],
            "runtime_root": OP12_STAGE_ROOT,
        },
        "op15_direct_relay": {
            "emit_direct_frames": True,
            "head_host": UNBOUND_PHONE_NETWORK["op12"]["local_ipv4"],
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
            "model_path": SHARD_PATH,
            "model_sha256": MODEL_SHA256,
            "n_gpu_layers": 999,
            "placement_cert": True,
            "port": PORTS["op15_stage"],
            "runtime_root": OP15_STAGE_ROOT,
        },
    }
    endpoints = {
        "op12_stagenet": "op12",
        "op15_direct_relay": "op15",
        "op15_stagenet": "op15",
    }
    components = _component_map(runtime)
    result = {}
    for name in sorted(routes):
        endpoint = endpoints[name]
        plan = _android_plan(
            runtime,
            name,
            endpoint,
            routes[name],
            adb_pin,
        )
        bundle = _bundle(runtime, name)
        runtime_executable = components[bundle["launcher_component_id"]]
        result[name] = {
            "argv": _inline_managed(
                usb_launcher,
                plan,
                UNBOUND_BOOT_IDS[endpoint],
            ),
            "cwd": str(Path(usb_launcher["path"]).parent),
            "environment": {},
            "launcher_bytes": usb_launcher["bytes"],
            "launcher_sha256": usb_launcher["sha256"],
            "runtime_component_ids": bundle["required_component_ids"],
            "runtime_executable_path": runtime_executable["path"],
            "runtime_executable_sha256": runtime_executable["sha256"],
            "shutdown_timeout_ms": 30_000,
            "startup_timeout_ms": 120_000,
        }
    return result


def current_authority_blockers(
    *,
    materializer: Any,
    cuda_static: dict[str, Any],
    phone_static: dict[str, Any],
) -> list[str]:
    blockers = []
    try:
        materializer._validate_cuda_static(cuda_static, phone_static)
    except Exception as error:
        if "E_CUDA_RUNTIME_ARGV" in str(error):
            blockers.append("E_V24_FLATTENED_CUDA_ARGV")
        else:
            raise
    else:
        raise CaptureError("E_EXPECTED_CUDA_AUTHORITY_REFUSAL_MISSING")
    phone_plan = json.loads(
        phone_static["processes"]["op12_stagenet"]["argv"][
            phone_static["processes"]["op12_stagenet"]["argv"].index("--plan-json")
            + 1
        ]
    )
    try:
        materializer.exact_keys(
            phone_plan,
            {
                "android",
                "bundle_id",
                "components",
                "endpoint",
                "launcher_component_id",
                "mode",
                "route",
                "schema",
            },
            "authority.phone.plan",
        )
    except Exception as error:
        if "E_KEYS" in str(error):
            blockers.append("E_V24_PHONE_PLAN_OMITS_SSH")
        else:
            raise
    else:
        raise CaptureError("E_EXPECTED_PHONE_AUTHORITY_REFUSAL_MISSING")
    return blockers


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-monolithic-launch", type=Path, required=True)
    parser.add_argument("--cuda-launcher", type=Path, required=True)
    parser.add_argument("--usb-launcher", type=Path, required=True)
    parser.add_argument("--artifact-root-entrypoint", type=Path, required=True)
    parser.add_argument("--cuda-monolithic-entrypoint", type=Path, required=True)
    parser.add_argument("--fast-fresh-entrypoint", type=Path, required=True)
    parser.add_argument("--joint-phone-cuda-entrypoint", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        require(1 <= args.timeout_seconds <= 7200, "E_TIMEOUT")
        for path in (
            args.cuda_launcher,
            args.usb_launcher,
            args.artifact_root_entrypoint,
            args.cuda_monolithic_entrypoint,
            args.fast_fresh_entrypoint,
            args.joint_phone_cuda_entrypoint,
        ):
            secure_local_pin(path, executable=True)
        runtime = capture_runtime_inventory(args.cuda_monolithic_launch)
        cuda_launcher = secure_local_pin(args.cuda_launcher, executable=True)
        local_plan, outer_argv = build_local_cuda_plan(runtime, cuda_launcher)
        del local_plan
        require(
            outer_argv[1] == "--plan-json"
            and outer_argv[3] == "--plan-sha256"
            and outer_argv[5] == "--boot-id"
            and len(outer_argv) == 7,
            "E_LOCAL_MANAGED_LAYERING",
        )
        digest = sha256_bytes(canonical_bytes(runtime))
        raise CaptureError(
            "E_V24_AUTHORITY_BLOCKED: "
            "E_V24_FLATTENED_CUDA_ARGV,E_V24_PHONE_PLAN_OMITS_SSH; "
            f"verified_runtime_inventory_sha256={digest}"
        )
    except Exception as error:
        print(
            f"V24_A_ONLY_INVENTORY_REFUSED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
