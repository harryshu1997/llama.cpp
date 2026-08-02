#!/usr/bin/env python3
"""Materialize an exact-bound V2.3 A_ONLY runtime input bundle."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any


INPUT_SCHEMA = "s39-cp0-r1-a-only-runtime-materializer-input-v1"
MANIFEST_SCHEMA = "s39-cp0-r1-a-only-runtime-input-manifest-v1"
SPECS_SCHEMA = "s39-cp0-r1-a-only-readiness-acquisition-specs-v1"
COMMAND_SCHEMA = "s39-cp0-r1-a-only-runtime-command-plan-v1"
JOINT_SCHEMA = "s39-cp0-r1-a-only-joint-capture-plan-v1"
PHONE_SCHEMA = "s39-cp0-r1-a-only-phone-route-launch-v1"
CUDA_SCHEMA = "s39-cp0-r1-a-only-cuda-route-launch-v1"
MONOLITHIC_SCHEMA = "s39-cp0-r1-a-only-cuda-monolithic-launch-v1"
HISTORY_SCHEMA = "s39-cp0-r1-a-only-b8-histories-v1"
RUNTIME_SCHEMA = "s39-cp0-r1-runtime-bundle-plan-v1"
RUNTIME_INPUT_SCHEMA = "s39-cp0-r1-runtime-bundle-closure-input-v1"

MODEL_ID = "qwen3-14b-q4_k_m"
MODEL_SHA256 = (
    "500a8806e85ee9c83f3ae084202955924"
    "51379b4f8cf2d0f41c15dffeb6b81f0"
)
MODEL_BYTES = 9001752960
MODEL_PATH = "/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf"
PHASE = "A_ONLY"
N_LAYER = 40
N_EMBD = 5120
MAX_STREAMS = 8
N_CTX_SEQ = 256
N_BATCH = 64
N_UBATCH = 64
FILE_TYPE = 15
CAPABILITIES = 0x3F
CUDA_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
PHONE_ADB_PORT = 5038
PHONE_SHARD_PATH = (
    "/data/local/tmp/s39-active-warm/v1/models/"
    "qwen3-14b-q4_k_m/weights.gguf"
)
OP15_SHARD_SHA256 = (
    "ba56b9c5e19b3a4512777e6a47803cc"
    "03261c2d3c2734965cd5ec96b7c6c59fb"
)
OP12_SHARD_SHA256 = (
    "72e312af745160dc33a0ba39ba94fbbc"
    "e6112950d0409d39c42ddc3b25e756ab"
)

HISTORIES_NAME = "A_ONLY_B8_HISTORIES_V1.json"
RUNTIME_NAME = "RUNTIME_BUNDLE_PLAN_V1.json"
PHONE_NAME = "PHONE_ROUTE_LAUNCH_V1.json"
CUDA_NAME = "CUDA_ROUTE_LAUNCH_V1.json"
MONOLITHIC_NAME = "CUDA_MONOLITHIC_LAUNCH_V1.json"
JOINT_NAME = "JOINT_CAPTURE_PLAN_V1.json"
COMMAND_NAME = "A_ONLY_COMMAND_PLAN_V1.json"
SPECS_NAME = "READINESS_ACQUISITION_SPECS_V1.json"
MANIFEST_NAME = "INPUT_MANIFEST_V1.json"

SOURCE_KEYS = {
    "acquisition_driver",
    "artifact_driver",
    "base_support",
    "cuda_producer",
    "entry_support",
    "fresh_driver",
    "joint_producer",
    "monolithic_producer",
    "phone_launcher",
    "phone_probe",
    "phone_producer",
    "relay_process_probe",
    "runtime_support",
}
EXECUTABLE_SOURCE_KEYS = {
    "acquisition_driver",
    "artifact_driver",
    "cuda_producer",
    "fresh_driver",
    "joint_producer",
    "monolithic_producer",
    "phone_launcher",
    "phone_probe",
    "phone_producer",
    "relay_process_probe",
}
RUNTIME_KEYS = {
    "adb_path",
    "codec_path",
    "cuda_launcher_path",
    "cuda_runtime_path",
    "model_path",
    "monolithic_launcher_path",
    "monolithic_runtime_path",
    "nvidia_smi_path",
}
ROUTE_KEYS = {
    "cuda_environment",
    "op12_telemetry",
    "op12_worker",
    "op15_telemetry",
    "op15_worker",
    "relay",
}
PHONE_KEYS = {
    "adb_selector",
    "device",
    "interface",
    "local_ipv4",
    "model",
    "product",
    "relay_path",
    "serial",
    "shard_bytes",
    "shard_stat",
    "worker_path",
}
NETWORK_KEYS = {
    "cuda_monolithic_port",
    "cuda_route_port",
    "op12_stage_port",
    "op15_stage_port",
    "relay_host",
    "relay_port",
    "relay_tail_source_port",
}
BUNDLE_IDS = {
    "cuda_monolithic",
    "cuda_route",
    "op12_stagenet",
    "op15_direct_relay",
    "op15_stagenet",
}
PAYLOAD_ROLES = {
    f"model.{MODEL_ID}.bridge": "bridge.jsonl",
    f"model.{MODEL_ID}.cuda_memory": "cuda-memory.jsonl",
    f"model.{MODEL_ID}.mechanics.phone": "mechanics-phone.jsonl",
    f"model.{MODEL_ID}.oracle.cuda_monolithic": "oracle-cuda-monolithic.jsonl",
    f"model.{MODEL_ID}.oracle.cuda_route": "oracle-cuda-route.jsonl",
    f"model.{MODEL_ID}.placement.op12": "placement-op12.jsonl",
    f"model.{MODEL_ID}.placement.op15": "placement-op15.jsonl",
    f"model.{MODEL_ID}.quality.cuda": "quality-cuda.jsonl",
    f"model.{MODEL_ID}.quality.phone": "quality-phone.jsonl",
    f"model.{MODEL_ID}.route_transfer": "route-transfer.jsonl",
}


class MaterializeError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializeError(message)


def exact_keys(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_INPUT_TYPE: {field}")
    require(set(value) == keys, f"E_INPUT_KEYS: {field}")
    return value


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        require(key not in value, f"E_DUPLICATE_KEY: {key}")
        value[key] = item
    return value


def reject_constant(value: str) -> None:
    raise MaterializeError(f"E_JSON_NUMBER: {value}")


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise MaterializeError("E_CANONICAL") from error


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest(value: Any, field: str) -> str:
    require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"E_INPUT_DIGEST: {field}",
    )
    return value


def text(value: Any, field: str) -> str:
    require(
        type(value) is str
        and bool(value)
        and all(0x20 <= ord(character) <= 0x7E for character in value),
        f"E_INPUT_TEXT: {field}",
    )
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"E_INPUT_INTEGER: {field}")
    return value


def absolute(value: Any, field: str) -> Path:
    path = Path(text(value, field))
    require(path.is_absolute(), f"E_INPUT_ABSOLUTE: {field}")
    require(".." not in path.parts, f"E_INPUT_PATH: {field}")
    return path


def no_symlink_chain(path: Path, include_leaf: bool = True) -> None:
    require(path.is_absolute(), f"E_INPUT_ABSOLUTE: {path}")
    parts = path.parts
    current = Path(parts[0])
    end = len(parts) if include_leaf else len(parts) - 1
    for part in parts[1:end]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise MaterializeError(f"E_INPUT_UNAVAILABLE: {current}") from error
        require(not stat.S_ISLNK(metadata.st_mode), f"E_INPUT_SYMLINK: {current}")


def read_regular(path: Path, field: str) -> tuple[bytes, os.stat_result]:
    no_symlink_chain(path)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise MaterializeError(f"E_INPUT_UNAVAILABLE: {field}: {path}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"E_INPUT_FILE_TYPE: {field}")
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
        item.st_mode,
    )
    require(identity(before) == identity(after), f"E_INPUT_CHANGED: {field}")
    require(len(raw) == before.st_size and bool(raw), f"E_INPUT_SIZE: {field}")
    return bytes(raw), before


def read_canonical(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    raw, _ = read_regular(path, field)
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializeError(f"E_INPUT_JSON: {field}") from error
    require(type(value) is dict, f"E_INPUT_TYPE: {field}")
    require(canonical_bytes(value) == raw, f"E_INPUT_CANONICAL: {field}")
    return value, raw


def stat_record(metadata: os.stat_result) -> dict[str, int]:
    return {
        "ctime_ns": metadata.st_ctime_ns,
        "device_id": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "mtime_ns": metadata.st_mtime_ns,
        "size": metadata.st_size,
    }


def validate_managed_stat(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "build_id",
            "ctime_ns",
            "device_id",
            "inode",
            "mode",
            "mtime_ns",
            "size",
        },
        field,
    )
    result = dict(value)
    for key in ("ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"):
        integer(result[key], f"{field}.{key}")
    require(
        result["inode"] > 0
        and result["size"] > 0
        and stat.S_ISREG(result["mode"]),
        f"E_INPUT_STAT: {field}",
    )
    build_id = result["build_id"]
    require(
        build_id is None or type(build_id) is str and bool(build_id),
        f"E_INPUT_BUILD_ID: {field}",
    )
    return result


def probe_stat(value: Any, field: str) -> dict[str, int]:
    value = exact_keys(
        value,
        {"ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"},
        field,
    )
    result = dict(value)
    for key in result:
        integer(result[key], f"{field}.{key}")
    require(
        result["inode"] > 0
        and result["size"] > 0
        and stat.S_ISREG(result["mode"]),
        f"E_INPUT_STAT: {field}",
    )
    return result


def local_artifact(path: Path, field: str, executable: bool) -> dict[str, Any]:
    raw, metadata = read_regular(path, field)
    if executable:
        require(metadata.st_mode & 0o111 != 0, f"E_INPUT_EXECUTABLE: {field}")
    return {
        "bytes": len(raw),
        "path": str(path),
        "sha256": sha256(raw),
        "stat": stat_record(metadata),
    }


def source_record(path: Path, field: str, executable: bool = False) -> dict[str, Any]:
    raw, metadata = read_regular(path, field)
    if executable:
        require(metadata.st_mode & 0o111 != 0, f"E_INPUT_EXECUTABLE: {field}")
    return {
        "bytes": len(raw),
        "mode": stat.S_IMODE(metadata.st_mode),
        "path": str(path),
        "sha256": sha256(raw),
    }


def executed_file(path: Path, argv_index: int) -> dict[str, Any]:
    raw, _ = read_regular(path, f"executed[{argv_index}]")
    return {
        "argv_index": argv_index,
        "bytes": len(raw),
        "path": str(path),
        "sha256": sha256(raw),
    }


def validate_argv(value: Any, field: str) -> list[str]:
    require(
        type(value) is list
        and bool(value)
        and all(
            type(item) is str
            and bool(item)
            and "\x00" not in item
            and "\n" not in item
            for item in value
        ),
        f"E_INPUT_ARGV: {field}",
    )
    require(Path(value[0]).is_absolute(), f"E_INPUT_ARGV_PATH: {field}")
    require(
        Path(value[0]).name not in {"bash", "dash", "sh", "zsh"}
        and "-c" not in value,
        f"E_INPUT_SHELL: {field}",
    )
    return list(value)


def validate_environment(value: Any, field: str) -> dict[str, str]:
    require(type(value) is dict, f"E_INPUT_ENV: {field}")
    for key, item in value.items():
        text(key, f"{field}.key")
        text(item, f"{field}.{key}")
        require("=" not in key, f"E_INPUT_ENV_KEY: {field}")
    return dict(value)


def validate_worker_route(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "devices",
            "driver_batch",
            "driver_context",
            "driver_max_prefill",
            "dynamic_cut",
            "kv_unified",
            "n_gpu_layers",
            "placement_cert",
        },
        field,
    )
    result = dict(value)
    text(result["devices"], f"{field}.devices")
    for key in (
        "driver_batch",
        "driver_context",
        "driver_max_prefill",
        "n_gpu_layers",
    ):
        integer(result[key], f"{field}.{key}", 0 if key == "n_gpu_layers" else 1)
    require(
        result["driver_max_prefill"] <= result["driver_context"],
        f"E_INPUT_ROUTE_CAPACITY: {field}",
    )
    for key in ("dynamic_cut", "kv_unified", "placement_cert"):
        require(type(result[key]) is bool, f"E_INPUT_BOOL: {field}.{key}")
    return result


def validate_telemetry(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "direct_peer_local_port",
            "direct_peer_port",
            "max_gpu_millic",
            "min_available_bytes",
        },
        field,
    )
    result = dict(value)
    for key in ("direct_peer_local_port", "direct_peer_port"):
        require(
            1 <= integer(result[key], f"{field}.{key}", 1) <= 65535,
            f"E_INPUT_PORT: {field}.{key}",
        )
    integer(result["min_available_bytes"], f"{field}.min_available_bytes", 1)
    require(
        1 <= integer(result["max_gpu_millic"], f"{field}.max_gpu_millic", 1)
        <= 200_000,
        f"E_INPUT_THERMAL: {field}",
    )
    return result


def validate_routes(value: Any) -> dict[str, Any]:
    value = exact_keys(value, ROUTE_KEYS, "routes")
    relay = exact_keys(
        value["relay"],
        {"head_host", "tail_host"},
        "routes.relay",
    )
    return {
        "cuda_environment": validate_environment(
            value["cuda_environment"],
            "routes.cuda_environment",
        ),
        "op12_telemetry": validate_telemetry(
            value["op12_telemetry"],
            "routes.op12_telemetry",
        ),
        "op12_worker": validate_worker_route(
            value["op12_worker"],
            "routes.op12_worker",
        ),
        "op15_telemetry": validate_telemetry(
            value["op15_telemetry"],
            "routes.op15_telemetry",
        ),
        "op15_worker": validate_worker_route(
            value["op15_worker"],
            "routes.op15_worker",
        ),
        "relay": {
            "head_host": text(relay["head_host"], "routes.relay.head_host"),
            "tail_host": text(relay["tail_host"], "routes.relay.tail_host"),
        },
    }


def quality_content_sha256(path: Path) -> tuple[str, str]:
    raw, _ = read_regular(path, "quality_corpus")
    lines = raw.splitlines(keepends=True)
    require(len(lines) == 64, "E_INPUT_CORPUS_ROWS")
    wrappers = {
        "acquisition_id",
        "phase",
        "phase_id",
        "role",
    }
    body_keys = {
        "choices",
        "dataset",
        "dataset_revision",
        "expected_answer",
        "item_index",
        "kind",
        "question",
        "source_row",
        "subject",
    }
    content = bytearray()
    for index, line in enumerate(lines):
        try:
            row = json.loads(
                line.decode("ascii"),
                object_pairs_hook=strict_object,
                parse_constant=reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MaterializeError(f"E_INPUT_CORPUS_JSON: {index}") from error
        require(type(row) is dict, f"E_INPUT_CORPUS_TYPE: {index}")
        require(canonical_bytes(row) == line, f"E_INPUT_CORPUS_CANONICAL: {index}")
        require(wrappers.issubset(row), f"E_INPUT_CORPUS_WRAPPER: {index}")
        require(row["phase"] == PHASE, f"E_INPUT_CORPUS_PHASE: {index}")
        require(
            row["acquisition_id"] == row["phase_id"],
            f"E_INPUT_CORPUS_ACQUISITION: {index}",
        )
        require(row["role"] == "quality.corpus", f"E_INPUT_CORPUS_ROLE: {index}")
        body = {key: item for key, item in row.items() if key not in wrappers}
        require(set(body) == body_keys, f"E_INPUT_CORPUS_BODY_KEYS: {index}")
        require(body["kind"] == "item", f"E_INPUT_CORPUS_KIND: {index}")
        del body["kind"]
        require(body.get("item_index") == index, f"E_INPUT_CORPUS_INDEX: {index}")
        require(
            type(body["choices"]) is list
            and len(body["choices"]) == 4
            and all(type(choice) is str for choice in body["choices"]),
            f"E_INPUT_CORPUS_CHOICES: {index}",
        )
        require(
            body["expected_answer"] in (0, 1, 2, 3),
            f"E_INPUT_CORPUS_ANSWER: {index}",
        )
        content.extend(canonical_bytes(body))
    return sha256(bytes(content)), sha256(raw)


def validate_histories(path: Path, route_epoch: int) -> tuple[dict[str, Any], bytes]:
    value, raw = read_canonical(path, "histories")
    exact_keys(
        value,
        {
            "histories",
            "history_width",
            "model_id",
            "model_sha256",
            "request_ids",
            "route_epoch",
            "schema",
        },
        "histories",
    )
    require(value["schema"] == HISTORY_SCHEMA, "E_INPUT_HISTORY_SCHEMA")
    require(value["model_id"] == MODEL_ID, "E_INPUT_HISTORY_MODEL")
    require(value["model_sha256"] == MODEL_SHA256, "E_INPUT_HISTORY_DIGEST")
    require(value["request_ids"] == list(range(8)), "E_INPUT_HISTORY_REQUESTS")
    require(value["history_width"] == 2, "E_INPUT_HISTORY_WIDTH")
    require(value["route_epoch"] == route_epoch, "E_INPUT_HISTORY_EPOCH")
    histories = value["histories"]
    require(type(histories) is list and len(histories) == 8, "E_INPUT_HISTORY_BATCH")
    for index, history in enumerate(histories):
        require(
            type(history) is list
            and len(history) == 2
            and all(type(token) is int and token >= 0 for token in history),
            f"E_INPUT_HISTORY_ROW: {index}",
        )
    return value, raw


def find_candidate_model(candidate: dict[str, Any]) -> dict[str, Any]:
    models = candidate.get("models")
    require(type(models) is list, "E_INPUT_CANDIDATE_MODELS")
    matches = [
        item
        for item in models
        if type(item) is dict and item.get("model_id") == MODEL_ID
    ]
    require(len(matches) == 1, "E_INPUT_CANDIDATE_MODEL")
    model = matches[0]
    require(model.get("slot") == "A", "E_INPUT_CANDIDATE_SLOT")
    artifact = model.get("artifact")
    require(type(artifact) is dict, "E_INPUT_CANDIDATE_ARTIFACT")
    require(artifact.get("sha256") == MODEL_SHA256, "E_INPUT_MODEL_SHA256")
    require(artifact.get("bytes") == MODEL_BYTES, "E_INPUT_MODEL_BYTES")
    return model


def validate_contract_candidate(contract_path: Path, candidate_path: Path
                                ) -> tuple[str, str]:
    contract, contract_raw = read_canonical(contract_path, "contract")
    candidate, candidate_raw = read_canonical(candidate_path, "candidate")
    require(
        contract.get("schema") == "s39-cp0-r1-evidence-contract-v2.3",
        "E_INPUT_CONTRACT_SCHEMA",
    )
    find_candidate_model(candidate)
    contract_digest = sha256(contract_raw)
    candidate_digest = sha256(candidate_raw)
    require(
        candidate.get("contract_sha256") == contract_digest,
        "E_INPUT_CONTRACT_CANDIDATE_BINDING",
    )
    return contract_digest, candidate_digest


def validate_phone(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, PHONE_KEYS, field)
    result = {
        key: text(value[key], f"{field}.{key}")
        for key in PHONE_KEYS - {"shard_bytes", "shard_stat"}
    }
    result["shard_bytes"] = integer(
        value["shard_bytes"],
        f"{field}.shard_bytes",
        1,
    )
    result["shard_stat"] = probe_stat(
        value["shard_stat"],
        f"{field}.shard_stat",
    )
    require(
        result["shard_stat"]["size"] == result["shard_bytes"],
        f"E_INPUT_SHARD_SIZE: {field}",
    )
    for key in ("relay_path", "worker_path"):
        require(Path(result[key]).is_absolute(), f"E_INPUT_ABSOLUTE: {field}.{key}")
    require(
        ":" in result["adb_selector"]
        and result["adb_selector"] != result["serial"],
        f"E_INPUT_WIFI_SELECTOR: {field}",
    )
    return result


def validate_runtime_closure(value: Any, launcher_paths: dict[str, str],
                             contract_sha256: str,
                             candidate_sha256: str
                             ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value = exact_keys(
        value,
        {"bundle_roots", "bundles", "closure_complete", "components", "schema"},
        "runtime_bundles",
    )
    require(value["schema"] == RUNTIME_INPUT_SCHEMA, "E_INPUT_RUNTIME_SCHEMA")
    require(value["closure_complete"] is True, "E_INPUT_RUNTIME_CLOSURE")
    roots = exact_keys(value["bundle_roots"], BUNDLE_IDS,
                      "runtime_bundles.bundle_roots")
    root_paths = {key: absolute(item, f"bundle_roots.{key}") for key, item in roots.items()}
    for left, left_path in root_paths.items():
        for right, right_path in root_paths.items():
            if left == right:
                continue
            require(
                left_path not in right_path.parents,
                f"E_INPUT_RUNTIME_ROOT_OVERLAP: {left}:{right}",
            )
    bundles = value["bundles"]
    require(type(bundles) is list, "E_INPUT_RUNTIME_BUNDLES")
    by_bundle = {}
    for index, bundle in enumerate(bundles):
        field = f"runtime_bundles.bundles[{index}]"
        exact_keys(
            bundle,
            {
                "bundle_id",
                "endpoint",
                "launcher_component_id",
                "process_role",
                "required_component_ids",
            },
            field,
        )
        bundle_id = text(bundle["bundle_id"], f"{field}.bundle_id")
        require(bundle_id in BUNDLE_IDS and bundle_id not in by_bundle,
                f"E_INPUT_RUNTIME_BUNDLE_ID: {bundle_id}")
        endpoint = text(bundle["endpoint"], f"{field}.endpoint")
        expected_endpoint = (
            "cuda" if bundle_id.startswith("cuda_")
            else "op12" if bundle_id.startswith("op12_") else "op15"
        )
        require(endpoint == expected_endpoint, f"E_INPUT_RUNTIME_ENDPOINT: {bundle_id}")
        required = bundle["required_component_ids"]
        require(
            type(required) is list
            and len(required) >= 2
            and required == sorted(set(required))
            and bundle["launcher_component_id"] in required,
            f"E_INPUT_RUNTIME_REQUIRED: {bundle_id}",
        )
        by_bundle[bundle_id] = dict(bundle)
    require(set(by_bundle) == BUNDLE_IDS, "E_INPUT_RUNTIME_BUNDLE_SET")
    components = value["components"]
    require(type(components) is list and bool(components), "E_INPUT_RUNTIME_COMPONENTS")
    by_component = {}
    paths_by_endpoint: dict[str, set[str]] = {"cuda": set(), "op12": set(), "op15": set()}
    for index, component in enumerate(components):
        field = f"runtime_bundles.components[{index}]"
        exact_keys(
            component,
            {
                "bundle_id",
                "bytes",
                "component_id",
                "endpoint",
                "path",
                "role",
                "sha256",
                "stat",
            },
            field,
        )
        component_id = text(component["component_id"], f"{field}.component_id")
        require(component_id not in by_component, f"E_INPUT_RUNTIME_COMPONENT_REUSE: {component_id}")
        bundle_id = text(component["bundle_id"], f"{field}.bundle_id")
        require(bundle_id in by_bundle, f"E_INPUT_RUNTIME_COMPONENT_BUNDLE: {component_id}")
        endpoint = text(component["endpoint"], f"{field}.endpoint")
        require(endpoint == by_bundle[bundle_id]["endpoint"],
                f"E_INPUT_RUNTIME_COMPONENT_ENDPOINT: {component_id}")
        path = absolute(component["path"], f"{field}.path")
        try:
            path.relative_to(root_paths[bundle_id])
        except ValueError as error:
            raise MaterializeError(f"E_INPUT_RUNTIME_ROOT_ESCAPE: {component_id}") from error
        require(str(path) not in paths_by_endpoint[endpoint],
                f"E_INPUT_RUNTIME_PATH_REUSE: {path}")
        paths_by_endpoint[endpoint].add(str(path))
        integer(component["bytes"], f"{field}.bytes", 1)
        digest(component["sha256"], f"{field}.sha256")
        component_stat = validate_managed_stat(component["stat"], f"{field}.stat")
        require(
            component_stat["size"] == component["bytes"],
            f"E_INPUT_RUNTIME_STAT_SIZE: {component_id}",
        )
        require(component["role"] in {"backend_library", "executable", "shared_library"},
                f"E_INPUT_RUNTIME_ROLE: {component_id}")
        record = dict(component)
        by_component[component_id] = record
        if endpoint == "cuda":
            live = local_artifact(path, field, component["role"] == "executable")
            require(live["bytes"] == component["bytes"], f"E_INPUT_RUNTIME_BYTES: {component_id}")
            require(live["sha256"] == component["sha256"], f"E_INPUT_RUNTIME_SHA256: {component_id}")
            require(
                live["stat"] == {
                    key: component_stat[key]
                    for key in live["stat"]
                },
                f"E_INPUT_RUNTIME_STAT: {component_id}",
            )
    referenced = []
    for bundle_id, bundle in by_bundle.items():
        required = bundle["required_component_ids"]
        require(all(item in by_component for item in required),
                f"E_INPUT_RUNTIME_DEPENDENCY: {bundle_id}")
        require(
            all(by_component[item]["bundle_id"] == bundle_id for item in required),
            f"E_INPUT_RUNTIME_OWNER: {bundle_id}",
        )
        launcher = by_component[bundle["launcher_component_id"]]
        require(launcher["role"] == "executable", f"E_INPUT_RUNTIME_LAUNCHER_ROLE: {bundle_id}")
        require(launcher["path"] == launcher_paths[bundle_id],
                f"E_INPUT_RUNTIME_LAUNCHER_PATH: {bundle_id}")
        require(
            any(by_component[item]["role"] in {"backend_library", "shared_library"}
                for item in required),
            f"E_INPUT_RUNTIME_LIBRARY: {bundle_id}",
        )
        referenced.extend(required)
    require(sorted(referenced) == sorted(by_component),
            "E_INPUT_RUNTIME_UNOWNED_COMPONENT")
    plan = {
        "bundle_roots": {key: str(root_paths[key]) for key in sorted(root_paths)},
        "bundles": [by_bundle[key] for key in sorted(by_bundle)],
        "candidate_sha256": candidate_sha256,
        "components": [
            {
                key: item
                for key, item in by_component[component_id].items()
                if key != "stat"
            }
            for component_id in sorted(by_component)
        ],
        "contract_sha256": contract_sha256,
        "model_id": MODEL_ID,
        "phase": PHASE,
        "schema": RUNTIME_SCHEMA,
    }
    return plan, by_component


def command_identity(argv: list[str], runtime_ids: list[str],
                     runtime_path: str, runtime_sha256: str,
                     source: dict[str, Any]) -> dict[str, Any]:
    return {
        "argv": argv,
        "cwd": str(Path(argv[0]).parent),
        "environment": {},
        "launcher_bytes": source["bytes"],
        "launcher_sha256": source["sha256"],
        "runtime_component_ids": runtime_ids,
        "runtime_executable_path": runtime_path,
        "runtime_executable_sha256": runtime_sha256,
        "shutdown_timeout_ms": 30_000,
        "startup_timeout_ms": 120_000,
    }


def find_bundle(runtime_plan: dict[str, Any], bundle_id: str
                ) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = next(item for item in runtime_plan["bundles"] if item["bundle_id"] == bundle_id)
    components = {
        item["component_id"]: item
        for item in runtime_plan["components"]
        if item["bundle_id"] == bundle_id
    }
    return bundle, components


def compact_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, UnicodeEncodeError) as error:
        raise MaterializeError("E_CANONICAL_INLINE") from error


def managed_components(bundle: dict[str, Any],
                       component_inputs: dict[str, dict[str, Any]]
                       ) -> list[dict[str, Any]]:
    return [
        {
            "bytes": component_inputs[component_id]["bytes"],
            "component_id": component_id,
            "path": component_inputs[component_id]["path"],
            "sha256": component_inputs[component_id]["sha256"],
            "stat": component_inputs[component_id]["stat"],
        }
        for component_id in bundle["required_component_ids"]
    ]


def managed_plan(runtime_plan: dict[str, Any],
                 component_inputs: dict[str, dict[str, Any]],
                 bundle_id: str, route: dict[str, Any],
                 android: dict[str, Any] | None) -> dict[str, Any]:
    bundle, _ = find_bundle(runtime_plan, bundle_id)
    return {
        "android": android,
        "bundle_id": bundle_id,
        "components": managed_components(bundle, component_inputs),
        "endpoint": bundle["endpoint"],
        "launcher_component_id": bundle["launcher_component_id"],
        "mode": "android" if android is not None else "local_cuda",
        "route": route,
        "schema": "s39-managed-runtime-launch-plan-v1",
    }


def inline_plan_argv(launcher: Path, plan: dict[str, Any]) -> list[str]:
    inline = compact_json(plan)
    return [
        str(launcher),
        "--plan-json",
        inline,
        "--plan-sha256",
        sha256(inline.encode("ascii")),
    ]


def android_plan(phone: dict[str, Any], adb: dict[str, Any]) -> dict[str, Any]:
    return {
        "adb_path": adb["path"],
        "adb_port": PHONE_ADB_PORT,
        "adb_selector": phone["adb_selector"],
        "adb_sha256": adb["sha256"],
        "boot_id_source": "phase_fresh_snapshot",
        "physical_serial": phone["serial"],
        "shutdown_timeout_ms": 30_000,
        "startup_timeout_ms": 120_000,
    }


def worker_route(route: dict[str, Any], runtime_root: str,
                 model_sha256: str, port: int, mode: str,
                 layer_start: int, layer_end: int) -> dict[str, Any]:
    return {
        **route,
        "kind": "stagenet_worker",
        "layer_end": layer_end,
        "layer_start": layer_start,
        "mode": mode,
        "model_path": PHONE_SHARD_PATH,
        "model_sha256": model_sha256,
        "port": port,
        "runtime_root": runtime_root,
    }


def worker_runtime_argv(path: str, route: dict[str, Any]) -> list[str]:
    return [
        path,
        "-m",
        route["model_path"],
        "--mode",
        route["mode"],
        "--port",
        str(route["port"]),
        "--driver-batch",
        str(route["driver_batch"]),
        "--driver-context",
        str(route["driver_context"]),
        "--driver-max-prefill",
        str(route["driver_max_prefill"]),
        "--devices",
        route["devices"],
        "-ngl",
        str(route["n_gpu_layers"]),
    ]


def relay_runtime_argv(path: str, route: dict[str, Any]) -> list[str]:
    argv = [
        path,
        "--listen",
        str(route["listen_port"]),
        "--head",
        f"{route['head_host']}:{route['head_port']}",
        "--tail",
        f"{route['tail_host']}:{route['tail_port']}",
        "--tail-source-port",
        str(route["tail_source_port"]),
    ]
    if route["emit_direct_frames"]:
        argv.append("--emit-direct-frames")
    return argv


def component_probe_artifact(component: dict[str, Any]) -> dict[str, Any]:
    return {
        "bytes": component["bytes"],
        "path": component["path"],
        "sha256": component["sha256"],
        "stat": {
            key: component["stat"][key]
            for key in (
                "ctime_ns",
                "device_id",
                "inode",
                "mode",
                "mtime_ns",
                "size",
            )
        },
    }


def probe_plan(phone: dict[str, Any], adb: dict[str, Any],
               worker: dict[str, Any], process_argv: list[str],
               telemetry: dict[str, Any],
               shard_sha256: str,
               network_role: str,
               network_component: dict[str, Any],
               network_argv: list[str]) -> dict[str, Any]:
    return {
        "android": {
            "adb_path": adb["path"],
            "adb_port": PHONE_ADB_PORT,
            "adb_selector": phone["adb_selector"],
            "adb_sha256": adb["sha256"],
            "boot_id_source": "phase_fresh_snapshot",
            "device": phone["device"],
            "model": phone["model"],
            "physical_serial": phone["serial"],
            "product": phone["product"],
        },
        "capture_schema": "s39-cp0-r1-phone-runtime-probe-v1",
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "network_process": {
            "argv": network_argv,
            "artifact": component_probe_artifact(network_component),
            "executable_path": network_component["path"],
            "role": network_role,
        },
        "process": {
            "argv": process_argv,
            "executable_path": worker["path"],
        },
        "schema": "s39-phone-runtime-probe-plan-v1",
        "shard_artifact": {
            "bytes": phone["shard_bytes"],
            "path": PHONE_SHARD_PATH,
            "sha256": shard_sha256,
            "stat": phone["shard_stat"],
        },
        "stage_v3": {
            "expected_active_sequences": 0,
            "source": "relay_owned_status",
        },
        "telemetry": {
            **telemetry,
            "direct_peer_ipv4": phone["direct_peer_ipv4"],
            "interface": phone["interface"],
            "local_ipv4": phone["local_ipv4"],
        },
        "worker_artifact": component_probe_artifact(worker),
    }


def inline_probe_argv(probe: Path, plan: dict[str, Any]) -> list[str]:
    inline = compact_json(plan)
    return [
        str(probe),
        "--plan-json",
        inline,
        "--plan-sha256",
        sha256(inline.encode("ascii")),
        "--capture-compatible",
    ]


def validate_inline_plan_argv(argv: Any, plan: dict[str, Any],
                              probe: bool = False) -> None:
    argv = validate_argv(argv, "inline_plan_argv")
    require("--plan" not in argv, "E_INLINE_PLAN_PATH")
    require(argv.count("--plan-json") == 1, "E_INLINE_PLAN_JSON_COUNT")
    plan_index = argv.index("--plan-json") + 1
    require(plan_index < len(argv), "E_INLINE_PLAN_JSON_VALUE")
    expected = compact_json(plan)
    require(argv[plan_index] == expected, "E_INLINE_PLAN_CONTENT")
    try:
        parsed = json.loads(
            argv[plan_index],
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise MaterializeError("E_INLINE_PLAN_JSON") from error
    require(parsed == plan, "E_INLINE_PLAN_VALUE")
    require(
        argv.count("--plan-sha256") == 1,
        "E_INLINE_PLAN_SHA256_COUNT",
    )
    digest_index = argv.index("--plan-sha256") + 1
    require(digest_index < len(argv), "E_INLINE_PLAN_SHA256_VALUE")
    require(
        argv[digest_index] == sha256(expected.encode("ascii")),
        "E_INLINE_PLAN_SHA256",
    )
    if probe:
        require(argv.count("--capture-compatible") == 1,
                "E_CAPTURE_COMPATIBLE")


def mechanism_matrix(commands: dict[str, list[str]], nvidia: dict[str, Any],
                     monolithic: list[str]) -> dict[str, list[list[str]]]:
    return {
        "desktop": [
            commands["codec"],
            commands["cuda_route"],
            nvidia["device_argv"],
            nvidia["process_argv"],
            nvidia["device_argv"],
            nvidia["process_argv"],
            nvidia["device_argv"],
            nvidia["process_argv"],
            monolithic,
        ],
        "op12": [
            commands["op12_stagenet"],
            commands["op12_probe_before"],
            commands["op12_probe_after"],
        ],
        "op15": [
            commands["op15_stagenet"],
            commands["op15_direct_relay"],
            [],  # Filled with the exact relay process probe below.
            commands["op15_probe_before"],
            commands["op15_probe_after"],
        ],
    }


def bound_file(path: Path, argv_index: int,
               virtual_files: dict[str, bytes]) -> dict[str, Any]:
    raw = virtual_files.get(str(path))
    if raw is None:
        return executed_file(path, argv_index)
    return {
        "argv_index": argv_index,
        "bytes": len(raw),
        "path": str(path),
        "sha256": sha256(raw),
    }


def producer_spec(argv: list[str], file_flags: set[str], result: str,
                  timeout: int, virtual_files: dict[str, bytes]) -> dict[str, Any]:
    indexes = {0}
    for flag in file_flags:
        require(argv.count(flag) == 1, f"E_INTERNAL_FLAG: {flag}")
        index = argv.index(flag) + 1
        require(index < len(argv), f"E_INTERNAL_FLAG_VALUE: {flag}")
        indexes.add(index)
    return {
        "argv_template": argv,
        "executed_files": [
            bound_file(Path(argv[index]), index, virtual_files)
            for index in sorted(indexes)
        ],
        "result_filename": result,
        "timeout_seconds": timeout,
    }


def driver_spec(argv: list[str], file_flags: set[str], timeout: int,
                virtual_files: dict[str, bytes]
                ) -> dict[str, Any]:
    indexes = {0}
    for flag in file_flags:
        require(argv.count(flag) == 1, f"E_INTERNAL_DRIVER_FLAG: {flag}")
        indexes.add(argv.index(flag) + 1)
    return {
        "argv_template": argv,
        "executed_files": [
            bound_file(Path(argv[index]), index, virtual_files)
            for index in sorted(indexes)
        ],
        "timeout_seconds": timeout,
    }


def build_bundle(input_path: Path, output_dir: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    spec, spec_raw = read_canonical(input_path, "input_spec")
    exact_keys(
        spec,
        {
            "contract_path",
            "candidate_path",
            "histories_path",
            "network",
            "phones",
            "quality_corpus_path",
            "routes",
            "route_epoch",
            "runtime",
            "runtime_bundles",
            "schema",
            "sources",
        },
        "input_spec",
    )
    require(spec["schema"] == INPUT_SCHEMA, "E_INPUT_SCHEMA")
    require(output_dir.is_absolute(), "E_OUTPUT_ABSOLUTE")
    require(not output_dir.exists(), "E_OUTPUT_EXISTS")
    no_symlink_chain(output_dir, include_leaf=False)

    contract_path = absolute(spec["contract_path"], "contract_path")
    candidate_path = absolute(spec["candidate_path"], "candidate_path")
    corpus_path = absolute(spec["quality_corpus_path"], "quality_corpus_path")
    histories_input = absolute(spec["histories_path"], "histories_path")
    route_epoch = integer(spec["route_epoch"], "route_epoch", 1)
    contract_digest, candidate_digest = validate_contract_candidate(
        contract_path, candidate_path
    )
    quality_digest, corpus_file_digest = quality_content_sha256(corpus_path)
    _, histories_raw = validate_histories(histories_input, route_epoch)

    source_values = exact_keys(spec["sources"], SOURCE_KEYS, "sources")
    sources = {
        key: absolute(source_values[key], f"sources.{key}")
        for key in sorted(SOURCE_KEYS)
    }
    source_records = {
        key: source_record(
            path,
            f"sources.{key}",
            key in EXECUTABLE_SOURCE_KEYS,
        )
        for key, path in sources.items()
    }
    runtime_values = exact_keys(spec["runtime"], RUNTIME_KEYS, "runtime")
    runtime = {
        key: absolute(runtime_values[key], f"runtime.{key}")
        for key in sorted(RUNTIME_KEYS)
    }
    require(str(runtime["model_path"]) == MODEL_PATH, "E_INPUT_MODEL_PATH")
    local = {
        key: local_artifact(path, f"runtime.{key}", key != "model_path")
        for key, path in runtime.items()
    }
    require(local["model_path"]["bytes"] == MODEL_BYTES, "E_INPUT_MODEL_BYTES")
    require(local["model_path"]["sha256"] == MODEL_SHA256, "E_INPUT_MODEL_SHA256")
    routes = validate_routes(spec["routes"])

    phones_value = exact_keys(spec["phones"], {"op12", "op15"}, "phones")
    phones = {
        key: validate_phone(phones_value[key], f"phones.{key}")
        for key in ("op12", "op15")
    }
    require(phones["op12"]["local_ipv4"] != phones["op15"]["local_ipv4"],
            "E_INPUT_PHONE_IP_REUSE")
    require(phones["op12"]["serial"] != phones["op15"]["serial"],
            "E_INPUT_PHONE_SERIAL_REUSE")
    phones["op12"]["direct_peer_ipv4"] = phones["op15"]["local_ipv4"]
    phones["op15"]["direct_peer_ipv4"] = phones["op12"]["local_ipv4"]

    network = exact_keys(spec["network"], NETWORK_KEYS, "network")
    relay_host = text(network["relay_host"], "network.relay_host")
    require(relay_host in {"127.0.0.1", "::1"}, "E_INPUT_RELAY_HOST")
    ports = {
        key: integer(network[key], f"network.{key}", 1)
        for key in NETWORK_KEYS - {"relay_host"}
    }
    require(all(port <= 65535 for port in ports.values()), "E_INPUT_PORT")
    require(len(set(ports.values())) == len(ports), "E_INPUT_PORT_REUSE")

    launcher_paths = {
        "cuda_monolithic": str(runtime["monolithic_runtime_path"]),
        "cuda_route": str(runtime["cuda_runtime_path"]),
        "op12_stagenet": phones["op12"]["worker_path"],
        "op15_direct_relay": phones["op15"]["relay_path"],
        "op15_stagenet": phones["op15"]["worker_path"],
    }
    runtime_plan, component_inputs = validate_runtime_closure(
        spec["runtime_bundles"],
        launcher_paths,
        contract_digest,
        candidate_digest,
    )

    def bundle_launcher(bundle_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        bundle, _ = find_bundle(runtime_plan, bundle_id)
        return bundle, component_inputs[bundle["launcher_component_id"]]

    op12_bundle, op12_worker = bundle_launcher("op12_stagenet")
    op15_bundle, op15_worker = bundle_launcher("op15_stagenet")
    relay_bundle, relay_runtime = bundle_launcher("op15_direct_relay")
    cuda_bundle, cuda_runtime = bundle_launcher("cuda_route")
    mono_bundle, mono_runtime = bundle_launcher("cuda_monolithic")

    op12_route = worker_route(
        routes["op12_worker"],
        runtime_plan["bundle_roots"]["op12_stagenet"],
        MODEL_SHA256,
        ports["op12_stage_port"],
        "tailv3",
        30,
        40,
    )
    op15_route = worker_route(
        routes["op15_worker"],
        runtime_plan["bundle_roots"]["op15_stagenet"],
        MODEL_SHA256,
        ports["op15_stage_port"],
        "stagenet",
        0,
        30,
    )
    relay_route = {
        "emit_direct_frames": True,
        "head_host": routes["relay"]["head_host"],
        "head_port": ports["op15_stage_port"],
        "kind": "direct_relay",
        "listen_port": ports["relay_port"],
        "runtime_root": runtime_plan["bundle_roots"]["op15_direct_relay"],
        "tail_host": routes["relay"]["tail_host"],
        "tail_port": ports["op12_stage_port"],
        "tail_source_port": ports["relay_tail_source_port"],
    }

    def cuda_target(path: str, port: int) -> list[str]:
        return [
            path,
            "--model",
            str(runtime["model_path"]),
            "--mode",
            "monov3",
            "--backend",
            "CUDA0",
            "--layer-start",
            "0",
            "--layer-end",
            str(N_LAYER),
            "--port",
            str(port),
            "--driver-batch",
            str(N_BATCH),
            "--driver-context",
            str(N_CTX_SEQ),
            "--driver-max-prefill",
            str(N_BATCH),
        ]

    cuda_environment = {
        **routes["cuda_environment"],
        "LAYERSPLIT_MEMORY_CERT": "1",
        "LAYERSPLIT_MODEL_SHA256": MODEL_SHA256,
        "LAYERSPLIT_PLACEMENT_CERT": "1",
    }
    cuda_target_argv = cuda_target(
        cuda_runtime["path"],
        ports["cuda_route_port"],
    )
    mono_target_argv = cuda_target(
        mono_runtime["path"],
        ports["cuda_monolithic_port"],
    )
    managed_plans = {
        "cuda_route": managed_plan(
            runtime_plan,
            component_inputs,
            "cuda_route",
            {
                "argv": cuda_target_argv,
                "cwd": runtime_plan["bundle_roots"]["cuda_route"],
                "environment": cuda_environment,
                "kind": "local_exec",
            },
            None,
        ),
        "cuda_monolithic": managed_plan(
            runtime_plan,
            component_inputs,
            "cuda_monolithic",
            {
                "argv": mono_target_argv,
                "cwd": runtime_plan["bundle_roots"]["cuda_monolithic"],
                "environment": cuda_environment,
                "kind": "local_exec",
            },
            None,
        ),
        "op12_stagenet": managed_plan(
            runtime_plan,
            component_inputs,
            "op12_stagenet",
            op12_route,
            android_plan(phones["op12"], local["adb_path"]),
        ),
        "op15_direct_relay": managed_plan(
            runtime_plan,
            component_inputs,
            "op15_direct_relay",
            relay_route,
            android_plan(phones["op15"], local["adb_path"]),
        ),
        "op15_stagenet": managed_plan(
            runtime_plan,
            component_inputs,
            "op15_stagenet",
            op15_route,
            android_plan(phones["op15"], local["adb_path"]),
        ),
    }
    probe_plans = {
        "op12": probe_plan(
            phones["op12"],
            local["adb_path"],
            op12_worker,
            worker_runtime_argv(op12_worker["path"], op12_route),
            routes["op12_telemetry"],
            OP12_SHARD_SHA256,
            "stagenet_worker",
            op12_worker,
            worker_runtime_argv(op12_worker["path"], op12_route),
        ),
        "op15": probe_plan(
            phones["op15"],
            local["adb_path"],
            op15_worker,
            worker_runtime_argv(op15_worker["path"], op15_route),
            routes["op15_telemetry"],
            OP15_SHARD_SHA256,
            "direct_relay",
            relay_runtime,
            relay_runtime_argv(relay_runtime["path"], relay_route),
        ),
    }
    commands = {
        "codec": [
            str(runtime["codec_path"]),
            "--model",
            str(runtime["model_path"]),
            "--model-sha256",
            MODEL_SHA256,
        ],
        "cuda_monolithic": inline_plan_argv(
            runtime["monolithic_launcher_path"],
            managed_plans["cuda_monolithic"],
        ),
        "cuda_route": inline_plan_argv(
            runtime["cuda_launcher_path"],
            managed_plans["cuda_route"],
        ),
        "op12_probe_after": inline_probe_argv(
            sources["phone_probe"],
            probe_plans["op12"],
        ),
        "op12_probe_before": inline_probe_argv(
            sources["phone_probe"],
            probe_plans["op12"],
        ),
        "op12_stagenet": inline_plan_argv(
            sources["phone_launcher"],
            managed_plans["op12_stagenet"],
        ),
        "op15_direct_relay": inline_plan_argv(
            sources["phone_launcher"],
            managed_plans["op15_direct_relay"],
        ),
        "op15_probe_after": inline_probe_argv(
            sources["phone_probe"],
            probe_plans["op15"],
        ),
        "op15_probe_before": inline_probe_argv(
            sources["phone_probe"],
            probe_plans["op15"],
        ),
        "op15_stagenet": inline_plan_argv(
            sources["phone_launcher"],
            managed_plans["op15_stagenet"],
        ),
    }
    for name, plan in managed_plans.items():
        validate_inline_plan_argv(commands[name], plan)
    for phone in ("op12", "op15"):
        for moment in ("before", "after"):
            validate_inline_plan_argv(
                commands[f"{phone}_probe_{moment}"],
                probe_plans[phone],
                probe=True,
            )

    histories_path = output_dir / HISTORIES_NAME
    runtime_path = output_dir / RUNTIME_NAME
    phone_path = output_dir / PHONE_NAME
    cuda_path = output_dir / CUDA_NAME
    monolithic_path = output_dir / MONOLITHIC_NAME
    joint_path = output_dir / JOINT_NAME
    command_path = output_dir / COMMAND_NAME

    nvidia = {
        "device_argv": [
            str(runtime["nvidia_smi_path"]),
            f"--id={CUDA_UUID}",
            "--query-gpu=name,uuid,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        "executable": local["nvidia_smi_path"],
        "process_argv": [
            str(runtime["nvidia_smi_path"]),
            f"--id={CUDA_UUID}",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        "timeout_ms": 30_000,
    }
    matrix = mechanism_matrix(commands, nvidia, commands["cuda_monolithic"])

    relay_expected = relay_runtime_argv(relay_runtime["path"], relay_route)
    require(
        relay_expected.count("--listen") == 1
        and relay_expected[relay_expected.index("--listen") + 1]
        == str(ports["relay_port"]),
        "E_INPUT_RELAY_LISTEN",
    )
    relay_probe_argv = [
        str(sources["relay_process_probe"]),
        "--adb",
        str(runtime["adb_path"]),
        "--adb-port",
        str(PHONE_ADB_PORT),
        "--adb-selector",
        phones["op15"]["adb_selector"],
        "--adb-sha256",
        local["adb_path"]["sha256"],
        "--expected-executable",
        phones["op15"]["relay_path"],
        "--expected-argv-json",
        json.dumps(relay_expected, ensure_ascii=True, separators=(",", ":")),
        "--expected-port",
        str(ports["relay_port"]),
    ]
    matrix["op15"][2] = relay_probe_argv
    mechanism_digest = sha256(canonical_bytes(matrix))

    def phone_bundle_command(bundle_id: str, argv: list[str]) -> dict[str, Any]:
        bundle, components = find_bundle(runtime_plan, bundle_id)
        launcher = components[bundle["launcher_component_id"]]
        return command_identity(
            argv,
            bundle["required_component_ids"],
            launcher["path"],
            launcher["sha256"],
            source_records["phone_launcher"],
        )

    phone_plan = {
        "codec": {
            "argv": commands["codec"],
            "cwd": str(runtime["codec_path"].parent),
            "environment": {},
            "executable_bytes": local["codec_path"]["bytes"],
            "executable_sha256": local["codec_path"]["sha256"],
            "timeout_ms": 600_000,
        },
        "expected_file_type": FILE_TYPE,
        "expected_max_streams": MAX_STREAMS,
        "expected_n_batch": N_BATCH,
        "expected_n_ctx_seq": N_CTX_SEQ,
        "expected_n_embd": N_EMBD,
        "expected_n_layer": N_LAYER,
        "expected_n_ubatch": N_UBATCH,
        "history_path": str(histories_path),
        "history_sha256": sha256(histories_raw),
        "mechanism_commands": matrix,
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "phones": {
            "op12": {
                "adb_selector": phones["op12"]["adb_selector"],
                "boot_id_source": "phase_fresh_snapshot",
                "device": phones["op12"]["device"],
                "direct_peer_ipv4": phones["op15"]["local_ipv4"],
                "executed_layers": [30, 40],
                "expected_worker_executable_path": phones["op12"]["worker_path"],
                "expected_worker_executable_sha256": find_bundle(
                    runtime_plan, "op12_stagenet"
                )[1][find_bundle(runtime_plan, "op12_stagenet")[0]["launcher_component_id"]]["sha256"],
                "interface": phones["op12"]["interface"],
                "loaded_shard_path": PHONE_SHARD_PATH,
                "loaded_shard_sha256": OP12_SHARD_SHA256,
                "local_ipv4": phones["op12"]["local_ipv4"],
                "model": phones["op12"]["model"],
                "product": phones["op12"]["product"],
                "serial": phones["op12"]["serial"],
                "stored_layers": [24, 40],
            },
            "op15": {
                "adb_selector": phones["op15"]["adb_selector"],
                "boot_id_source": "phase_fresh_snapshot",
                "device": phones["op15"]["device"],
                "direct_peer_ipv4": phones["op12"]["local_ipv4"],
                "executed_layers": [0, 30],
                "expected_worker_executable_path": phones["op15"]["worker_path"],
                "expected_worker_executable_sha256": find_bundle(
                    runtime_plan, "op15_stagenet"
                )[1][find_bundle(runtime_plan, "op15_stagenet")[0]["launcher_component_id"]]["sha256"],
                "interface": phones["op15"]["interface"],
                "loaded_shard_path": PHONE_SHARD_PATH,
                "loaded_shard_sha256": OP15_SHARD_SHA256,
                "local_ipv4": phones["op15"]["local_ipv4"],
                "model": phones["op15"]["model"],
                "product": phones["op15"]["product"],
                "serial": phones["op15"]["serial"],
                "stored_layers": [0, 32],
            },
        },
        "probes": {
            phone: {
                "after_argv": commands[f"{phone}_probe_after"],
                "before_argv": commands[f"{phone}_probe_before"],
                "cwd": str(sources["phone_probe"].parent),
                "environment": {},
                "launcher_bytes": source_records["phone_probe"]["bytes"],
                "launcher_sha256": source_records["phone_probe"]["sha256"],
                "timeout_ms": 60_000,
            }
            for phone in ("op12", "op15")
        },
        "processes": {
            "op12_stagenet": phone_bundle_command(
                "op12_stagenet", commands["op12_stagenet"]
            ),
            "op15_direct_relay": phone_bundle_command(
                "op15_direct_relay", commands["op15_direct_relay"]
            ),
            "op15_stagenet": phone_bundle_command(
                "op15_stagenet", commands["op15_stagenet"]
            ),
        },
        "quality_corpus_content_sha256": quality_digest,
        "relay_process_probe": {
            "argv": relay_probe_argv,
            "cwd": str(sources["relay_process_probe"].parent),
            "environment": {},
            "expected_argv": relay_expected,
            "expected_executable_path": phones["op15"]["relay_path"],
            "expected_port": ports["relay_port"],
            "launcher_bytes": source_records["relay_process_probe"]["bytes"],
            "launcher_sha256": source_records["relay_process_probe"]["sha256"],
            "timeout_ms": 60_000,
        },
        "relay_host": relay_host,
        "relay_port": ports["relay_port"],
        "route_epoch": route_epoch,
        "schema": PHONE_SCHEMA,
    }

    cuda_bundle, cuda_components = find_bundle(runtime_plan, "cuda_route")
    cuda_launcher = cuda_components[cuda_bundle["launcher_component_id"]]
    cuda_plan = {
        "codec": {
            "argv": commands["codec"],
            "cwd": str(runtime["codec_path"].parent),
            "environment": {},
            "executable": local["codec_path"],
            "timeout_ms": 600_000,
        },
        "expected_capabilities": CAPABILITIES,
        "expected_file_type": FILE_TYPE,
        "expected_max_streams": MAX_STREAMS,
        "expected_n_batch": N_BATCH,
        "expected_n_ctx_seq": N_CTX_SEQ,
        "expected_n_embd": N_EMBD,
        "expected_n_layer": N_LAYER,
        "expected_n_ubatch": N_UBATCH,
        "history_path": str(histories_path),
        "history_sha256": sha256(histories_raw),
        "host": "127.0.0.1",
        "io_timeout_ms": 60_000,
        "mechanism_commands": matrix,
        "model_artifact": local["model_path"],
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "nvidia_smi": nvidia,
        "port": ports["cuda_route_port"],
        "quality_corpus_content_sha256": quality_digest,
        "route_epoch": route_epoch,
        "schema": CUDA_SCHEMA,
        "worker": {
            "argv": commands["cuda_route"],
            "cwd": str(runtime["cuda_launcher_path"].parent),
            "environment": cuda_environment,
            "executable": local["cuda_launcher_path"],
            "runtime_component_ids": cuda_bundle["required_component_ids"],
            "runtime_executable": local["cuda_runtime_path"],
            "shutdown_timeout_ms": 30_000,
            "startup_timeout_ms": 120_000,
        },
    }

    monolithic_plan = {
        "command": commands["cuda_monolithic"],
        "cwd": str(runtime["monolithic_launcher_path"].parent),
        "env": cuda_environment,
        "expected_capabilities": CAPABILITIES,
        "expected_file_type": FILE_TYPE,
        "expected_max_streams": MAX_STREAMS,
        "expected_n_batch": N_BATCH,
        "expected_n_ctx_seq": N_CTX_SEQ,
        "expected_n_embd": N_EMBD,
        "expected_n_layer": N_LAYER,
        "expected_n_ubatch": N_UBATCH,
        "host": "127.0.0.1",
        "io_timeout_ms": 60_000,
        "mechanism_commands": matrix,
        "mechanism_commands_sha256": mechanism_digest,
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "port": ports["cuda_monolithic_port"],
        "schema": MONOLITHIC_SCHEMA,
        "shutdown_timeout_ms": 30_000,
        "startup_timeout_ms": 120_000,
    }
    runtime_raw = canonical_bytes(runtime_plan)
    phone_raw = canonical_bytes(phone_plan)
    cuda_raw = canonical_bytes(cuda_plan)
    monolithic_raw = canonical_bytes(monolithic_plan)
    nested_virtual = {
        str(histories_path): histories_raw,
        str(phone_path): phone_raw,
        str(cuda_path): cuda_raw,
        str(monolithic_path): monolithic_raw,
        str(runtime_path): runtime_raw,
    }

    phone_argv = [
        str(sources["phone_producer"]),
        "--output", "{output_path}",
        "--phase-id", "{phase_id}",
        "--pre-dir", "{pre_dir}",
        "--started", "{acquisition_started_ns}",
        "--plan", "{command_plan_sha256}",
        "--mechanism-commands-sha256", mechanism_digest,
        "--model-sha256", MODEL_SHA256,
        "--histories", str(histories_path),
        "--launch-plan", str(phone_path),
        "--execute",
        "--confirm", "RUN_PHONE_ROUTE_A_ONLY",
    ]
    cuda_argv = [
        str(sources["cuda_producer"]),
        "--output", "{output_path}",
        "--phase-id", "{phase_id}",
        "--pre-dir", "{pre_dir}",
        "--started", "{acquisition_started_ns}",
        "--plan", "{command_plan_sha256}",
        "--mechanism-commands-sha256", mechanism_digest,
        "--model-sha256", MODEL_SHA256,
        "--histories", str(histories_path),
        "--launch-plan", str(cuda_path),
        "--execute",
        "--confirm", "RUN_CUDA_ROUTE_A_ONLY",
    ]
    joint_plan = {
        "commands": {
            "cuda": producer_spec(
                cuda_argv,
                {"--histories", "--launch-plan"},
                "cuda.json",
                1800,
                nested_virtual,
            ),
            "phone": producer_spec(
                phone_argv,
                {"--histories", "--launch-plan"},
                "phone.json",
                1800,
                nested_virtual,
            ),
        },
        "mechanism_commands": matrix,
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "phase": PHASE,
        "schema": JOINT_SCHEMA,
    }
    joint_raw = canonical_bytes(joint_plan)
    nested_virtual[str(joint_path)] = joint_raw

    joint_argv = [
        str(sources["joint_producer"]),
        "--capture-plan", str(joint_path),
        "--output", "{output_path}",
        "--phase-id", "{phase_id}",
        "--pre-dir", "{pre_dir}",
        "--acquisition-started-ns", "{acquisition_started_ns}",
        "--command-plan-sha256", "{command_plan_sha256}",
    ]
    mono_argv = [
        str(sources["monolithic_producer"]),
        "--output", "{output_path}",
        "--phase-id", "{phase_id}",
        "--pre-dir", "{pre_dir}",
        "--started", "{acquisition_started_ns}",
        "--plan", "{command_plan_sha256}",
        "--mechanism-commands-sha256", mechanism_digest,
        "--model-sha256", MODEL_SHA256,
        "--histories", str(histories_path),
        "--launch-plan", str(monolithic_path),
    ]
    command_plan = {
        "candidate_sha256": candidate_digest,
        "contract_sha256": contract_digest,
        "mechanism_commands": matrix,
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "outputs": {
            **PAYLOAD_ROLES,
            "runtime_bundle_identity": "runtime_bundle_identity.json",
            "runtime_identity": "runtime_identity.json",
        },
        "phase": PHASE,
        "producers": {
            "cuda_monolithic": producer_spec(
                mono_argv,
                {"--histories", "--launch-plan"},
                "cuda-monolithic.json",
                1800,
                nested_virtual,
            ),
            "joint_phone_cuda": producer_spec(
                joint_argv,
                {"--capture-plan"},
                "joint-phone-cuda.json",
                1800,
                nested_virtual,
            ),
        },
        "schema": COMMAND_SCHEMA,
    }
    command_raw = canonical_bytes(command_plan)
    nested_virtual[str(command_path)] = command_raw

    common_driver_tail = [
        "--entry-support", str(sources["entry_support"]),
        "--base-support", str(sources["base_support"]),
        "--support", str(sources["runtime_support"]),
        "--runtime-bundle-plan", str(runtime_path),
        "--contract", str(contract_path),
        "--candidate", str(candidate_path),
        "--op15-worker", phones["op15"]["worker_path"],
        "--op12-worker", phones["op12"]["worker_path"],
        "--phase-id", "{phase_id}",
        "--pre", "{pre_dir}",
        "--output", "{output_dir}",
    ]
    file_flags = {
        "--entry-support",
        "--base-support",
        "--support",
        "--runtime-bundle-plan",
        "--contract",
        "--candidate",
    }
    acquisition_argv = [
        str(sources["acquisition_driver"]),
        "--command-plan", str(command_path),
        "--contract", str(contract_path),
        "--candidate", str(candidate_path),
        "--output-dir", "{output_dir}",
        "--phase-id", "{phase_id}",
        "--pre-dir", "{pre_dir}",
        "--acquisition-started-ns", "{acquisition_started_ns}",
    ]
    specs = {
        "acquisition": driver_spec(
            acquisition_argv,
            {"--command-plan", "--contract", "--candidate"},
            3600,
            nested_virtual,
        ),
        "artifact": driver_spec(
            [str(sources["artifact_driver"]), *common_driver_tail],
            file_flags,
            1800,
            nested_virtual,
        ),
        "fresh": driver_spec(
            [str(sources["fresh_driver"]), *common_driver_tail],
            file_flags,
            1800,
            nested_virtual,
        ),
        "schema": SPECS_SCHEMA,
    }

    outputs = {
        HISTORIES_NAME: histories_raw,
        RUNTIME_NAME: runtime_raw,
        PHONE_NAME: phone_raw,
        CUDA_NAME: cuda_raw,
        MONOLITHIC_NAME: monolithic_raw,
        JOINT_NAME: joint_raw,
        COMMAND_NAME: command_raw,
        SPECS_NAME: canonical_bytes(specs),
    }
    manifest = {
        "bindings": {
            "candidate_sha256": candidate_digest,
            "contract_sha256": contract_digest,
            "histories_sha256": sha256(histories_raw),
            "mechanism_commands_sha256": mechanism_digest,
            "model_sha256": MODEL_SHA256,
            "quality_corpus_content_sha256": quality_digest,
            "quality_corpus_file_sha256": corpus_file_digest,
            "typed_plan_sha256": {
                **{
                    f"managed.{name}": sha256(
                        compact_json(plan).encode("ascii")
                    )
                    for name, plan in sorted(managed_plans.items())
                },
                **{
                    f"probe.{name}": sha256(
                        compact_json(plan).encode("ascii")
                    )
                    for name, plan in sorted(probe_plans.items())
                },
            },
        },
        "generated_files": [
            {
                "bytes": len(raw),
                "name": name,
                "sha256": sha256(raw),
            }
            for name, raw in sorted(outputs.items())
        ],
        "input_spec": {
            "bytes": len(spec_raw),
            "path": str(input_path),
            "sha256": sha256(spec_raw),
        },
        "inputs": [
            {
                "bytes": record["bytes"],
                "path": record["path"],
                "role": f"runtime.{role}",
                "sha256": record["sha256"],
                "stat": record["stat"],
            }
            for role, record in sorted(local.items())
        ] + [
            {
                "bytes": record["bytes"],
                "mode": record["mode"],
                "path": record["path"],
                "role": role,
                "sha256": record["sha256"],
            }
            for role, record in (
                ("candidate", source_record(candidate_path, "candidate")),
                ("contract", source_record(contract_path, "contract")),
                ("histories", source_record(histories_input, "histories")),
                ("quality_corpus", source_record(corpus_path, "quality_corpus")),
            )
        ],
        "model_id": MODEL_ID,
        "phase": PHASE,
        "schema": MANIFEST_SCHEMA,
        "sources": [
            {"role": role, **record}
            for role, record in sorted(source_records.items())
        ],
    }
    outputs[MANIFEST_NAME] = canonical_bytes(manifest)
    return outputs, manifest


def write_new(path: Path, raw: bytes) -> None:
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


def rename_no_replace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "E_ATOMIC_PUBLICATION_UNAVAILABLE")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise MaterializeError(
            f"E_OUTPUT_PUBLISH: {os.strerror(error_number)}"
        )


def publish(outputs: dict[str, bytes], output_dir: Path) -> None:
    parent = output_dir.parent
    require(parent.is_dir(), "E_OUTPUT_PARENT")
    no_symlink_chain(parent)
    temporary = parent / f".{output_dir.name}.tmp.{os.getpid()}"
    require(not temporary.exists(), "E_OUTPUT_TEMP_EXISTS")
    temporary.mkdir(mode=0o700)
    try:
        for name, raw in sorted(outputs.items()):
            write_new(temporary / name, raw)
        directory = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        rename_no_replace(temporary, output_dir)
        parent_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-inputs", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require(args.input_spec.is_absolute(), "E_INPUT_SPEC_ABSOLUTE")
        require(args.output_dir.is_absolute(), "E_OUTPUT_ABSOLUTE")
        outputs, manifest = build_bundle(args.input_spec, args.output_dir)
        if args.validate_inputs:
            print(
                "A_ONLY_RUNTIME_INPUTS_VALID "
                + manifest["bindings"]["mechanism_commands_sha256"]
            )
            return 0
        publish(outputs, args.output_dir)
        print(
            "A_ONLY_RUNTIME_INPUTS_MATERIALIZED "
            + sha256(outputs[MANIFEST_NAME])
        )
        return 0
    except (MaterializeError, OSError, ValueError) as error:
        print(
            f"A_ONLY_RUNTIME_INPUTS_REFUSED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
