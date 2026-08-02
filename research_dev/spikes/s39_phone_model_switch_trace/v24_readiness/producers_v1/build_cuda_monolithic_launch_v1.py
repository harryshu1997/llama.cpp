#!/usr/bin/python3 -I
"""Build the exact RTX 4060 Ti monolithic launch plan before acquisition."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable


sys.dont_write_bytecode = True

SCHEMA = "s39-cp0-r1-v24-cuda-monolithic-launch-v1"
BUNDLE_ID = "cuda_monolithic"
ENDPOINT = "cuda"
PROCESS_ROLE = "cuda_monolithic"
MODEL_ID = "qwen3-14b-q4_k_m"
MODEL_SHA256 = "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"
MODEL_BYTES = 9001752960
MODEL_PATH = Path("/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf")
BUNDLE_ROOT = Path("/home/zhihao/llama.cpp-s40/build-s40-cuda/bin")
CWD = Path("/home/zhihao/llama.cpp-s40")
PORT = 39124
LAUNCHER_COMPONENT_ID = "cuda-mono.bin"
ALLOWED_SYSTEM_ROOTS = (
    "/mnt/storage/s21_deps/cuda-13.2.1/lib/",
    "/usr/lib/x86_64-linux-gnu/",
)
MAX_OUTPUT_BYTES = 1024 * 1024


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class LaunchConfig:
    allowed_system_roots: tuple[str, ...]
    bundle_root: Path
    components: tuple[ComponentSpec, ...]
    cwd: Path
    environment: tuple[tuple[str, str], ...]
    expected_file_type: int
    expected_n_embd: int
    launcher_component_id: str
    model_bytes: int
    model_path: Path
    model_sha256: str
    port: int


COMPONENTS = (
    ComponentSpec(
        "cuda-mono.bin",
        BUNDLE_ROOT / "llama-layersplit",
        "b6e3fd647a869f1b00dc6b01460810541872b456c3d671a745ba272683a9787d",
    ),
    ComponentSpec(
        "cuda-mono.libggml",
        BUNDLE_ROOT / "libggml.so.0.15.3",
        "879447caeae3a8c028c796c1bf9787fa61f86e7902144e601041154ae69429fb",
    ),
    ComponentSpec(
        "cuda-mono.libggml-base",
        BUNDLE_ROOT / "libggml-base.so.0.15.3",
        "aba0cb48199ba5d918221f6fa0cee13fec74f13189b67aa8afffe8bc6e4920ba",
    ),
    ComponentSpec(
        "cuda-mono.libggml-cpu",
        BUNDLE_ROOT / "libggml-cpu.so.0.15.3",
        "08fbf4f55bd37af7fc0eeb7d4947eb1f49ae1b5ba1b1e7e591c0cbab4f3657d0",
    ),
    ComponentSpec(
        "cuda-mono.libggml-cuda",
        BUNDLE_ROOT / "libggml-cuda.so.0.15.3",
        "f239d3e1ad76893b0c1fe47105ec1c9ff638a507283e55e8c4c3ccc086ceb528",
    ),
    ComponentSpec(
        "cuda-mono.libllama",
        BUNDLE_ROOT / "libllama.so.0.0.9875",
        "c39d348d373ad65665035687de39fd3a434b57306eb5dc53694de8aaf47c7b45",
    ),
    ComponentSpec(
        "cuda-mono.libllama-common",
        BUNDLE_ROOT / "libllama-common.so.0.0.9875",
        "43e52457d8c0f6b292e132ee19704e2a67534ba3ad77655003dde8b8c72a513b",
    ),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def canonical_bytes(value: Any) -> bytes:
    try:
        raw = (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise BuildError("E_CANONICAL") from error
    require(len(raw) <= MAX_OUTPUT_BYTES, "E_OUTPUT_SIZE")
    return raw


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def valid_digest(value: str, field: str) -> str:
    require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"E_DIGEST: {field}",
    )
    return value


def stat_record(value: os.stat_result) -> dict[str, int]:
    return {
        "ctime_ns": value.st_ctime_ns,
        "device_id": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "mtime_ns": value.st_mtime_ns,
        "size": value.st_size,
    }


def stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )


def snapshot_file(
    path: Path,
    expected_sha256: str,
    expected_bytes: int | None = None,
    after_hash: Callable[[], None] | None = None,
) -> dict[str, Any]:
    require(path.is_absolute(), f"E_PATH: {path}")
    valid_digest(expected_sha256, f"expected:{path}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BuildError(f"E_OPEN: {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"E_FILE_TYPE: {path}")
        require(before.st_size > 0, f"E_FILE_SIZE: {path}")
        digest = hashlib.sha256()
        consumed = 0
        while block := os.read(descriptor, 4 * 1024 * 1024):
            digest.update(block)
            consumed += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        stat_identity(before) == stat_identity(after)
        and consumed == before.st_size,
        f"E_HASH_MUTATION: {path}",
    )
    require(digest.hexdigest() == expected_sha256, f"E_FILE_SHA256: {path}")
    if expected_bytes is not None:
        require(before.st_size == expected_bytes, f"E_FILE_BYTES: {path}")
    if after_hash is not None:
        after_hash()
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BuildError(f"E_REOPEN: {path}: {error}") from error
    try:
        reopened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        stat_identity(before) == stat_identity(reopened),
        f"E_REOPEN_MUTATION: {path}",
    )
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "stat": stat_record(before),
    }


def bundle_digest(components: list[dict[str, Any]], launcher_id: str) -> str:
    identity = {
        "bundle_id": BUNDLE_ID,
        "components": components,
        "endpoint": ENDPOINT,
        "launcher_component_id": launcher_id,
        "process_role": PROCESS_ROLE,
        "schema": "s39-cp0-r1-runtime-bundle-root-identity-v2.4",
    }
    return sha256_bytes(canonical_bytes(identity))


def build_launch(config: LaunchConfig, route_epoch: int) -> dict[str, Any]:
    require(type(route_epoch) is int and route_epoch > 0, "E_ROUTE_EPOCH")
    require(
        config.bundle_root.is_absolute()
        and config.bundle_root.is_dir()
        and not config.bundle_root.is_symlink(),
        "E_BUNDLE_ROOT",
    )
    require(config.cwd.is_absolute() and config.cwd.is_dir(), "E_CWD")
    component_ids = [component.component_id for component in config.components]
    require(
        component_ids == sorted(set(component_ids))
        and config.launcher_component_id in component_ids,
        "E_COMPONENT_IDS",
    )
    components = []
    for component in config.components:
        try:
            component.path.relative_to(config.bundle_root)
        except ValueError as error:
            raise BuildError(f"E_COMPONENT_ROOT: {component.component_id}") from error
        snapshot = snapshot_file(component.path, component.sha256)
        components.append({
            "component_id": component.component_id,
            **snapshot,
        })
    model = snapshot_file(
        config.model_path,
        config.model_sha256,
        config.model_bytes,
    )
    environment = dict(config.environment)
    require(
        len(environment) == len(config.environment)
        and environment.get("LAYERSPLIT_MODEL_SHA256") == config.model_sha256
        and environment.get("LAYERSPLIT_PLACEMENT_CERT") == "1",
        "E_ENVIRONMENT",
    )
    require(
        config.allowed_system_roots == ALLOWED_SYSTEM_ROOTS,
        "E_SYSTEM_ROOTS",
    )
    require(1 <= config.port <= 65535, "E_PORT")
    launcher = next(
        component["path"]
        for component in components
        if component["component_id"] == config.launcher_component_id
    )
    command = [
        launcher,
        "-m",
        str(config.model_path),
        "--mode",
        "monov3",
        "--port",
        str(config.port),
        "--devices",
        "CUDA0",
        "--driver-batch",
        "8",
        "--driver-context",
        "512",
        "--driver-max-prefill",
        "8",
    ]
    return {
        "allowed_system_roots": list(config.allowed_system_roots),
        "bundle_id": BUNDLE_ID,
        "bundle_root": str(config.bundle_root),
        "bundle_sha256": bundle_digest(components, config.launcher_component_id),
        "command": command,
        "cwd": str(config.cwd),
        "endpoint": ENDPOINT,
        "env": environment,
        "expected_capabilities": 0x3F,
        "expected_file_type": config.expected_file_type,
        "expected_max_streams": 8,
        "expected_n_batch": 64,
        "expected_n_ctx_seq": 512,
        "expected_n_embd": config.expected_n_embd,
        "expected_n_layer": 40,
        "expected_n_ubatch": 64,
        "host": "127.0.0.1",
        "io_timeout_ms": 300000,
        "launcher_component_id": config.launcher_component_id,
        "model_artifact": model,
        "model_id": MODEL_ID,
        "model_sha256": config.model_sha256,
        "port": config.port,
        "required_components": components,
        "route_epoch": route_epoch,
        "schema": SCHEMA,
        "shutdown_timeout_ms": 30000,
        "startup_timeout_ms": 300000,
    }


def production_config() -> LaunchConfig:
    return LaunchConfig(
        allowed_system_roots=ALLOWED_SYSTEM_ROOTS,
        bundle_root=BUNDLE_ROOT,
        components=COMPONENTS,
        cwd=CWD,
        environment=(
            ("CUDA_VISIBLE_DEVICES", "0"),
            ("HOME", "/home/zhihao"),
            ("LAYERSPLIT_MEMORY_CERT", "1"),
            ("LAYERSPLIT_MODEL_SHA256", MODEL_SHA256),
            ("LAYERSPLIT_PLACEMENT_CERT", "1"),
            ("LC_ALL", "C"),
            ("LD_LIBRARY_PATH", str(BUNDLE_ROOT)),
        ),
        expected_file_type=15,
        expected_n_embd=5120,
        launcher_component_id=LAUNCHER_COMPONENT_ID,
        model_bytes=MODEL_BYTES,
        model_path=MODEL_PATH,
        model_sha256=MODEL_SHA256,
        port=PORT,
    )


def write_exclusive(path: Path, raw: bytes) -> None:
    require(path.is_absolute() and not path.exists(), "E_OUTPUT")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o644,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, "E_OUTPUT_WRITE")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--route-epoch", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        output = Path(args.output)
        launch = build_launch(production_config(), args.route_epoch)
        write_exclusive(output, canonical_bytes(launch))
        return 0
    except (BuildError, OSError, ValueError) as error:
        print(
            f"CUDA_MONOLITHIC_LAUNCH_REFUSED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
