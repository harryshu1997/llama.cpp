#!/usr/bin/env python3
"""Materialize a V2.4 A_ONLY prospective spec from a pinned desktop inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import types
from typing import Any


HERE = Path(__file__).resolve().parent
V24 = HERE.parent
ORIGINATOR_PATH = V24 / "production_plan_v1" / "originate_runtime_v1.py"
TOPOLOGY_VERIFIER_PATH = HERE / "verify_topology_v1.py"
USB_LAUNCHER_PATH = HERE / "managed_runtime_launcher_usb_v1.py"
ORIGINATOR_SHA256 = (
    "0f8caa48aa9e3a294d0677de3afc6568"
    "af4888205b2b9a32d8f7c5c536da8b1a"
)
TOPOLOGY_VERIFIER_SHA256 = (
    "0f90228fda3acf37e9e751bc65a882df1"
    "9d971a8589062c85df6f5664998c129"
)
USB_LAUNCHER_SHA256 = (
    "52c4e1f251f4daa2c856857fcdf23fdc"
    "5a85c0d30eff93365996a8f1378611ce"
)

INVENTORY_SCHEMA = "s39-cp0-r1-v24-a-only-desktop-inventory-v1"
SPEC_SCHEMA = "s39-cp0-r1-v24-prospective-runtime-spec-v1"
OPERATOR_SCHEMA = "s39-cp0-r1-v24-desktop-operator-input-v1"
CONTRACT_SCHEMA = "s39-cp0-r1-evidence-contract-v2.4"
CONTRACT_SHA256 = (
    "20eea0fd1fb3aba6be9265a1cf84aaddeb867a9c0277db1109273faa4731b455"
)
CANDIDATE_SHA256 = (
    "ee3196ca660fa7eb7ea293260dc98dd6fdbf14571a4d4c5aece5343fb29b28d8"
)
MODEL_ID = "qwen3-14b-q4_k_m"
MODEL_SHA256 = (
    "500a8806e85ee9c83f3ae084202955924"
    "51379b4f8cf2d0f41c15dffeb6b81f0"
)
MODEL_BYTES = 9_001_752_960
MODEL_PATH = "/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf"
PHASE = "A_ONLY"
PHONE_ADB_PORT = 5038
CONTROLLER_HOST = "zhihao-Z690-C-ac"
CUDA_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
EXPECTED_PHONES = {
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
INPUT_SCHEMAS = {
    "candidate": "s39-cp0-r1-candidate-v1",
    "contract": CONTRACT_SCHEMA,
    "cuda_monolithic_launch": "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
    "operator_input": OPERATOR_SCHEMA,
    "runtime_bundle_inventory": "s39-cp0-r1-runtime-bundle-closure-input-v1",
    "token_history": "s39-cp0-r1-token-history-v2.4",
    "tokenizer_plan": "s39-cp0-r1-a-only-tokenizer-plan-v2",
    "topology_receipt": "s39-cp0-r1-v24-desktop-topology-v1",
}
RUNTIME_BUNDLE_IDS = {
    "cuda_monolithic",
    "cuda_route",
    "op12_stagenet",
    "op15_direct_relay",
    "op15_stagenet",
}
AUTHORITY_PROCESS_ROLES = {
    "cuda_monolithic": "cuda_monolithic",
    "cuda_route": "cuda_route",
    "op12_stagenet": "stagenet_worker",
    "op15_direct_relay": "direct_relay",
    "op15_stagenet": "stagenet_worker",
}
CAPTURE_KINDS = {
    "artifact_root",
    "cuda_monolithic",
    "fast_fresh_readiness",
    "joint_phone_cuda",
}
WORKER_ROUTE_KEYS = {
    "devices",
    "driver_batch",
    "driver_context",
    "driver_max_prefill",
    "dynamic_cut",
    "kind",
    "kv_unified",
    "layer_end",
    "layer_start",
    "mode",
    "model_path",
    "model_sha256",
    "n_gpu_layers",
    "placement_cert",
    "port",
    "runtime_root",
}
RELAY_ROUTE_KEYS = {
    "emit_direct_frames",
    "head_host",
    "head_port",
    "kind",
    "listen_port",
    "runtime_root",
    "tail_host",
    "tail_port",
    "tail_source_port",
}
UNBOUND_BOOT_IDS = {
    "cuda": "00000000-0000-0000-0000-000000000000",
    "op12": "00000000-0000-0000-0000-000000000001",
    "op15": "00000000-0000-0000-0000-000000000002",
}
UNBOUND_PHONE_NETWORK = {
    "op12": {
        "interface": "UNBOUND_AFTER_REBOOT",
        "local_ipv4": "0.0.0.12",
    },
    "op15": {
        "interface": "UNBOUND_AFTER_REBOOT",
        "local_ipv4": "0.0.0.15",
    },
}
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
MAX_INPUT_BYTES = 64 * 1024 * 1024


class MaterializeError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializeError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}: expected {expected!r}, got {value!r}",
    )


def exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    actual = set(value)
    require(
        actual == expected,
        f"E_KEYS: {field}: missing={sorted(expected - actual)}, "
        f"unknown={sorted(actual - expected)}",
    )
    return value


def text(value: Any, field: str) -> str:
    require(
        type(value) is str
        and bool(value)
        and value.isascii()
        and "\x00" not in value
        and "\n" not in value,
        f"E_TEXT: {field}",
    )
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(
        type(value) is int and minimum <= value <= (1 << 63) - 1,
        f"E_INTEGER: {field}",
    )
    return value


def digest(value: Any, field: str) -> str:
    value = text(value, field)
    require(DIGEST_RE.fullmatch(value) is not None, f"E_DIGEST: {field}")
    return value


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise MaterializeError(f"E_JSON_NUMBER: {value}")


def reject_float(value: str) -> None:
    raise MaterializeError(f"E_JSON_FLOAT: {value}")


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


def parse_json(raw: bytes, field: str) -> Any:
    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
            parse_float=reject_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializeError(f"E_JSON: {field}") from error


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def read_regular(path: Path, field: str) -> bytes:
    require(path.is_absolute(), f"E_ABSOLUTE: {field}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise MaterializeError(f"E_READ: {field}: {path}") from error
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode)
            and 0 < before.st_size <= MAX_INPUT_BYTES,
            f"E_REGULAR: {field}",
        )
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
            require(len(raw) <= MAX_INPUT_BYTES, f"E_SIZE: {field}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    exact(_identity(after), _identity(before), f"E_TOCTOU: {field}")
    exact(len(raw), before.st_size, f"E_READ_SIZE: {field}")
    return bytes(raw)


def read_canonical(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path, field)
    value = parse_json(raw, field)
    require(type(value) is dict, f"E_TYPE: {field}")
    exact(canonical_bytes(value), raw, f"E_CANONICAL: {field}")
    return value, raw


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _absolute(value: Any, field: str) -> Path:
    path = Path(text(value, field))
    require(path.is_absolute() and ".." not in path.parts, f"E_PATH: {field}")
    return path


def _read_pin(value: Any, field: str) -> tuple[Path, dict[str, Any], bytes]:
    value = exact_keys(value, {"bytes", "path", "sha256"}, field)
    path = _absolute(value["path"], f"{field}.path")
    expected_bytes = integer(value["bytes"], f"{field}.bytes", 1)
    expected_sha256 = digest(value["sha256"], f"{field}.sha256")
    parsed, raw = read_canonical(path, field)
    exact(len(raw), expected_bytes, f"{field}.bytes")
    exact(sha256(raw), expected_sha256, f"{field}.sha256")
    return path, parsed, raw


def _pin_identity(value: Any, field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    require(
        {"bytes", "path", "sha256"} <= set(value),
        f"E_PIN_FIELDS: {field}",
    )
    return {
        "bytes": integer(value["bytes"], f"{field}.bytes", 1),
        "path": str(_absolute(value["path"], f"{field}.path")),
        "sha256": digest(value["sha256"], f"{field}.sha256"),
    }


def _find_model(candidate: dict[str, Any]) -> dict[str, Any]:
    models = candidate.get("models")
    require(type(models) is list, "E_CANDIDATE_MODELS")
    matches = [
        value
        for value in models
        if type(value) is dict
        and value.get("slot") == "A"
        and value.get("model_id") == MODEL_ID
    ]
    require(len(matches) == 1, "E_CANDIDATE_MODEL_A")
    model = matches[0]
    exact(model.get("n_layer"), 40, "candidate.model.n_layer")
    artifact = exact_keys(
        model.get("artifact"),
        {"bytes", "file_name", "sha256"},
        "candidate.model.artifact",
    )
    exact(artifact["bytes"], MODEL_BYTES, "candidate.model.bytes")
    exact(artifact["sha256"], MODEL_SHA256, "candidate.model.sha256")
    return model


def _strip_stat(component: Any, field: str) -> dict[str, Any]:
    component = exact_keys(
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
    integer(component["bytes"], f"{field}.bytes", 1)
    digest(component["sha256"], f"{field}.sha256")
    _absolute(component["path"], f"{field}.path")
    metadata = _validate_stat_record(component["stat"], f"{field}.stat")
    exact(metadata["size"], component["bytes"], f"{field}.stat.size")
    return {key: value for key, value in component.items() if key != "stat"}


def _validate_stat_record(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(
        value,
        {"ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"},
        field,
    )
    for key, item in value.items():
        integer(item, f"{field}.{key}")
    require(stat.S_ISREG(value["mode"]), f"E_STAT_MODE: {field}")
    return value


def _artifact_identity(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, {"bytes", "path", "sha256", "stat"}, field)
    byte_count = integer(value["bytes"], f"{field}.bytes", 1)
    _absolute(value["path"], f"{field}.path")
    digest(value["sha256"], f"{field}.sha256")
    metadata = _validate_stat_record(value["stat"], f"{field}.stat")
    exact(metadata["size"], byte_count, f"{field}.stat.size")
    return value


def _validate_runtime_static(
    runtime_static: Any,
    inventory: dict[str, Any],
    monolithic: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    inventory = exact_keys(
        inventory,
        {
            "bundle_roots",
            "bundles",
            "closure_complete",
            "components",
            "schema",
        },
        "runtime_inventory",
    )
    exact(
        inventory["schema"],
        INPUT_SCHEMAS["runtime_bundle_inventory"],
        "runtime_inventory.schema",
    )
    exact(inventory["closure_complete"], True, "runtime_inventory.closure_complete")
    roots = exact_keys(
        inventory["bundle_roots"],
        RUNTIME_BUNDLE_IDS,
        "runtime_inventory.bundle_roots",
    )
    for bundle_id, root in roots.items():
        _absolute(root, f"runtime_inventory.bundle_roots.{bundle_id}")
    bundles = inventory["bundles"]
    require(type(bundles) is list and len(bundles) == len(RUNTIME_BUNDLE_IDS),
            "E_RUNTIME_BUNDLES")
    bundle_ids = []
    normalized_bundles = []
    for index, bundle in enumerate(bundles):
        field = f"runtime_inventory.bundles[{index}]"
        bundle = exact_keys(
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
        require(bundle_id in RUNTIME_BUNDLE_IDS, f"E_RUNTIME_BUNDLE: {bundle_id}")
        required = bundle["required_component_ids"]
        require(
            type(required) is list
            and bool(required)
            and required == sorted(set(required))
            and all(type(item) is str and bool(item) for item in required),
            f"E_RUNTIME_REQUIRED: {bundle_id}",
        )
        require(
            bundle["launcher_component_id"] in required,
            f"E_RUNTIME_LAUNCHER: {bundle_id}",
        )
        exact(
            bundle["process_role"],
            bundle_id,
            f"{field}.verifier_process_role",
        )
        normalized_bundles.append(
            {
                **bundle,
                "process_role": AUTHORITY_PROCESS_ROLES[bundle_id],
            }
        )
        bundle_ids.append(bundle_id)
    exact(bundle_ids, sorted(RUNTIME_BUNDLE_IDS), "runtime_inventory.bundle_order")
    inventory_bundle_map = {
        value["bundle_id"]: value for value in bundles
    }
    runtime_static = exact_keys(
        runtime_static,
        {
            "bundle_roots",
            "bundles",
            "capture_entrypoints",
            "components",
            "tokenizer_component_id",
        },
        "static.runtime",
    )
    exact(
        runtime_static["bundle_roots"],
        inventory["bundle_roots"],
        "runtime.bundle_roots",
    )
    exact(
        runtime_static["bundles"],
        normalized_bundles,
        "runtime.bundles",
    )
    base_components = {}
    base_component_stats = {}
    for index, value in enumerate(inventory["components"]):
        field = f"runtime_inventory.components[{index}]"
        stripped = _strip_stat(value, field)
        component_id = stripped["component_id"]
        require(component_id not in base_components, f"E_COMPONENT_REUSE: {component_id}")
        base_components[component_id] = stripped
        base_component_stats[component_id] = value["stat"]
    monolithic_bundle = inventory_bundle_map["cuda_monolithic"]
    exact(
        roots["cuda_monolithic"],
        monolithic["bundle_root"],
        "runtime_inventory.cuda_monolithic.root",
    )
    exact(
        monolithic_bundle["launcher_component_id"],
        monolithic["launcher_component_id"],
        "runtime_inventory.cuda_monolithic.launcher",
    )
    monolithic_components = {
        value["component_id"]: value
        for value in monolithic["required_components"]
    }
    exact(
        monolithic_bundle["required_component_ids"],
        sorted(monolithic_components),
        "runtime_inventory.cuda_monolithic.required",
    )
    for component_id, required in monolithic_components.items():
        require(component_id in base_components, f"E_MONOLITHIC_COMPONENT: {component_id}")
        expected = {
            key: base_components[component_id][key]
            for key in ("component_id", "path", "sha256")
        }
        exact(
            expected,
            {
                key: required[key]
                for key in ("component_id", "path", "sha256")
            },
            f"runtime_inventory.cuda_monolithic.component.{component_id}",
        )
        exact(
            base_component_stats[component_id],
            required["stat"],
            f"runtime_inventory.cuda_monolithic.stat.{component_id}",
        )
    values = runtime_static["components"]
    require(type(values) is list and bool(values), "E_RUNTIME_COMPONENTS")
    components = {}
    for index, value in enumerate(values):
        field = f"static.runtime.components[{index}]"
        value = exact_keys(
            value,
            {
                "bundle_id",
                "bytes",
                "component_id",
                "endpoint",
                "path",
                "role",
                "sha256",
            },
            field,
        )
        component_id = text(value["component_id"], f"{field}.component_id")
        require(component_id not in components, f"E_COMPONENT_REUSE: {component_id}")
        components[component_id] = value
    exact(
        [value["component_id"] for value in values],
        sorted(components),
        "runtime.component_order",
    )
    for component_id, expected in base_components.items():
        require(component_id in components, f"E_COMPONENT_MISSING: {component_id}")
        exact(components[component_id], expected, f"runtime.component.{component_id}")
    captures = runtime_static["capture_entrypoints"]
    require(type(captures) is list and len(captures) == 4, "E_CAPTURE_ENTRIES")
    by_kind = {}
    for index, value in enumerate(captures):
        field = f"static.runtime.capture_entrypoints[{index}]"
        value = exact_keys(
            value,
            {
                "component_id",
                "execution_mode",
                "kind",
                "nested_capture_entrypoint_component_ids",
            },
            field,
        )
        kind = text(value["kind"], f"{field}.kind")
        require(kind in CAPTURE_KINDS and kind not in by_kind, f"E_CAPTURE_KIND: {kind}")
        exact(
            value["execution_mode"],
            "SELF_CONTAINED_PHYSICAL_CAPTURE",
            f"{field}.execution_mode",
        )
        component_id = text(value["component_id"], f"{field}.component_id")
        require(component_id in components, f"E_CAPTURE_COMPONENT: {kind}")
        require(component_id not in base_components, f"E_CAPTURE_RUNTIME_ALIAS: {kind}")
        nested = value["nested_capture_entrypoint_component_ids"]
        require(
            type(nested) is list
            and nested == sorted(set(nested))
            and all(
                type(item) is str and item in components and item != component_id
                for item in nested
            ),
            f"E_CAPTURE_NESTED: {kind}",
        )
        by_kind[kind] = value
    exact([value["kind"] for value in captures], sorted(CAPTURE_KINDS), "capture.order")
    exact(set(by_kind), CAPTURE_KINDS, "capture.kinds")
    extra = set(components) - set(base_components)
    exact(
        extra,
        {value["component_id"] for value in captures},
        "runtime.capture_component_closure",
    )
    launchers = {
        value["bundle_id"]: value["launcher_component_id"]
        for value in normalized_bundles
    }
    expected_nested = {
        "artifact_root": [],
        "cuda_monolithic": [launchers["cuda_monolithic"]],
        "fast_fresh_readiness": [],
        "joint_phone_cuda": sorted(
            launchers[bundle_id]
            for bundle_id in (
                "cuda_route",
                "op12_stagenet",
                "op15_direct_relay",
                "op15_stagenet",
            )
        ),
    }
    for kind, nested in expected_nested.items():
        exact(
            by_kind[kind]["nested_capture_entrypoint_component_ids"],
            nested,
            f"capture.{kind}.nested",
        )
    source_programs = contract["producer_requirements"]["source_programs"]
    for kind in ("cuda_monolithic", "joint_phone_cuda"):
        component = components[by_kind[kind]["component_id"]]
        source = source_programs[kind]
        exact(component["bytes"], source["bytes"], f"capture.{kind}.bytes")
        exact(component["sha256"], source["sha256"], f"capture.{kind}.sha256")
    tokenizer = text(
        runtime_static["tokenizer_component_id"],
        "runtime.tokenizer_component_id",
    )
    require(tokenizer in components, "E_TOKENIZER_COMPONENT")
    return runtime_static


def _inline_plan(argv: Any, field: str) -> dict[str, Any]:
    require(
        type(argv) is list
        and all(
            type(item) is str
            and item.isascii()
            and "\x00" not in item
            and "\n" not in item
            for item in argv
        )
        and argv.count("--plan-json") == 1
        and argv.count("--plan-sha256") == 1,
        f"E_INLINE_PLAN: {field}",
    )
    index = argv.index("--plan-json") + 1
    digest_index = argv.index("--plan-sha256") + 1
    require(index < len(argv) and digest_index < len(argv), f"E_INLINE_PLAN: {field}")
    try:
        value = json.loads(
            argv[index],
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
            parse_float=reject_float,
        )
    except json.JSONDecodeError as error:
        raise MaterializeError(f"E_INLINE_JSON: {field}") from error
    require(type(value) is dict, f"E_INLINE_TYPE: {field}")
    exact(
        argv[digest_index],
        sha256(argv[index].encode("ascii")),
        f"{field}.plan_sha256",
    )
    return value


def _validate_phone_static(value: Any, contract: dict[str, Any]) -> dict[str, Any]:
    required = {
        "codec",
        "expected_file_type",
        "expected_max_streams",
        "expected_n_batch",
        "expected_n_ctx_seq",
        "expected_n_embd",
        "expected_n_layer",
        "expected_n_ubatch",
        "mechanism_commands",
        "phones",
        "probes",
        "processes",
        "relay_host",
        "relay_port",
        "route_epoch",
    }
    value = exact_keys(value, required, "static.phone_route")
    for key, expected in (
        ("expected_file_type", 15),
        ("expected_max_streams", 8),
        ("expected_n_batch", 64),
        ("expected_n_ctx_seq", 512),
        ("expected_n_embd", 5120),
        ("expected_n_layer", 40),
        ("expected_n_ubatch", 64),
    ):
        exact(value[key], expected, f"static.phone_route.{key}")
    exact(value["relay_host"], "127.0.0.1", "static.phone_route.relay_host")
    integer(value["relay_port"], "static.phone_route.relay_port", 1)
    integer(value["route_epoch"], "static.phone_route.route_epoch", 1)
    codec = exact_keys(
        value["codec"],
        {
            "argv",
            "cwd",
            "environment",
            "executable_bytes",
            "executable_sha256",
            "timeout_ms",
        },
        "static.phone_route.codec",
    )
    _option(codec["argv"], "--model", MODEL_PATH, "static.phone_route.codec")
    _option(
        codec["argv"],
        "--model-sha256",
        MODEL_SHA256,
        "static.phone_route.codec",
    )
    phones = exact_keys(value["phones"], {"op12", "op15"}, "static.phone_route.phones")
    geometry = contract["model_geometry"][MODEL_ID]["known_shards"]
    phone_keys = {
        "device",
        "executed_layers",
        "expected_worker_executable_path",
        "expected_worker_executable_sha256",
        "loaded_shard_path",
        "loaded_shard_sha256",
        "model",
        "product",
        "serial",
        "stored_layers",
    }
    for endpoint, expected in EXPECTED_PHONES.items():
        phone = exact_keys(phones[endpoint], phone_keys, f"phones.{endpoint}")
        for key in ("device", "model", "product", "serial"):
            exact(phone[key], expected[key], f"phones.{endpoint}.{key}")
        exact(
            phone["executed_layers"],
            expected["executed_layers"],
            f"phones.{endpoint}.executed_layers",
        )
        exact(
            phone["stored_layers"],
            expected["stored_layers"],
            f"phones.{endpoint}.stored_layers",
        )
        exact(
            phone["loaded_shard_sha256"],
            expected["shard_sha256"],
            f"phones.{endpoint}.loaded_shard_sha256",
        )
        exact(
            phone["loaded_shard_path"],
            geometry[endpoint]["path"],
            f"phones.{endpoint}.loaded_shard_path",
        )
        digest(
            phone["expected_worker_executable_sha256"],
            f"phones.{endpoint}.worker_sha256",
        )
    processes = exact_keys(
        value["processes"],
        {"op12_stagenet", "op15_direct_relay", "op15_stagenet"},
        "static.phone_route.processes",
    )
    expected_process = {
        "op12_stagenet": ("op12", "op12_stagenet"),
        "op15_direct_relay": ("op15", "op15_direct_relay"),
        "op15_stagenet": ("op15", "op15_stagenet"),
    }
    for name, (endpoint, bundle_id) in expected_process.items():
        process = exact_keys(
            processes[name],
            {
                "argv",
                "cwd",
                "environment",
                "launcher_bytes",
                "launcher_sha256",
                "runtime_component_ids",
                "runtime_executable_path",
                "runtime_executable_sha256",
                "shutdown_timeout_ms",
                "startup_timeout_ms",
            },
            f"processes.{name}",
        )
        plan = _inline_plan(process["argv"], f"processes.{name}")
        plan = exact_keys(
            plan,
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
            f"{name}.plan",
        )
        exact(plan["schema"], "s39-managed-runtime-launch-plan-v1", f"{name}.schema")
        exact(plan["bundle_id"], bundle_id, f"{name}.bundle")
        exact(plan["endpoint"], endpoint, f"{name}.endpoint")
        android = exact_keys(
            plan["android"],
            {
                "adb_path",
                "adb_port",
                "adb_selector",
                "adb_sha256",
                "boot_id_source",
                "physical_serial",
                "shutdown_timeout_ms",
                "startup_timeout_ms",
            },
            f"{name}.android",
        )
        exact(android["adb_port"], PHONE_ADB_PORT, f"{name}.adb_port")
        exact(
            android["adb_selector"],
            EXPECTED_PHONES[endpoint]["serial"],
            f"{name}.adb_selector",
        )
        exact(
            android["physical_serial"],
            EXPECTED_PHONES[endpoint]["serial"],
            f"{name}.physical_serial",
        )
        route = plan["route"]
        if name.endswith("stagenet"):
            route = exact_keys(route, WORKER_ROUTE_KEYS, f"{name}.route")
            exact(route["devices"], "GPUOpenCL", f"{name}.backend")
            exact(
                [route["layer_start"], route["layer_end"]],
                EXPECTED_PHONES[endpoint]["executed_layers"],
                f"{name}.layers",
            )
            exact(route["model_sha256"], MODEL_SHA256, f"{name}.model_sha256")
            for key, expected in (
                ("driver_batch", 64),
                ("driver_context", 512),
                ("driver_max_prefill", 64),
                ("kind", "stagenet_worker"),
                ("n_gpu_layers", 999),
                ("placement_cert", True),
            ):
                exact(route[key], expected, f"{name}.{key}")
        else:
            exact_keys(route, RELAY_ROUTE_KEYS, f"{name}.route")
            exact(route["kind"], "direct_relay", f"{name}.kind")
            exact(route["emit_direct_frames"], True, f"{name}.emit_direct_frames")
    probes = exact_keys(
        value["probes"],
        {"op12", "op15"},
        "static.phone_route.probes",
    )
    for endpoint, probe in probes.items():
        exact_keys(
            probe,
            {
                "after_argv",
                "before_argv",
                "cwd",
                "environment",
                "launcher_bytes",
                "launcher_sha256",
                "timeout_ms",
            },
            f"probes.{endpoint}",
        )
        require(
            type(probe["before_argv"]) is list
            and bool(probe["before_argv"])
            and type(probe["after_argv"]) is list
            and bool(probe["after_argv"]),
            f"E_PROBE_ARGV: {endpoint}",
        )
        exact(
            probe["before_argv"][0],
            probe["after_argv"][0],
            f"probes.{endpoint}.launcher",
        )
    mechanism = exact_keys(
        value["mechanism_commands"],
        {"desktop", "op12", "op15"},
        "static.phone_route.mechanism_commands",
    )
    exact(
        mechanism["op12"],
        [
            processes["op12_stagenet"]["argv"],
            probes["op12"]["before_argv"],
            probes["op12"]["after_argv"],
        ],
        "mechanism.op12",
    )
    exact(
        mechanism["op15"],
        [
            processes["op15_stagenet"]["argv"],
            processes["op15_direct_relay"]["argv"],
            probes["op15"]["before_argv"],
            probes["op15"]["after_argv"],
        ],
        "mechanism.op15",
    )
    require(type(mechanism["desktop"]) is list and len(mechanism["desktop"]) == 9,
            "E_MECHANISM_DESKTOP")
    return value


def _option(argv: Any, option: str, expected: str, field: str) -> None:
    require(
        type(argv) is list
        and argv.count(option) == 1
        and argv.index(option) + 1 < len(argv),
        f"E_OPTION: {field}.{option}",
    )
    exact(argv[argv.index(option) + 1], expected, f"{field}.{option}")


def _validate_cuda_static(value: Any, phone: dict[str, Any]) -> dict[str, Any]:
    required = {
        "codec",
        "expected_capabilities",
        "expected_file_type",
        "expected_max_streams",
        "expected_n_batch",
        "expected_n_ctx_seq",
        "expected_n_embd",
        "expected_n_layer",
        "expected_n_ubatch",
        "host",
        "io_timeout_ms",
        "mechanism_commands",
        "model_artifact",
        "nvidia_smi",
        "port",
        "route_epoch",
        "worker",
    }
    value = exact_keys(value, required, "static.cuda_route")
    for key, expected in (
        ("expected_capabilities", 0x3F),
        ("expected_file_type", 15),
        ("expected_max_streams", 8),
        ("expected_n_batch", 64),
        ("expected_n_ctx_seq", 512),
        ("expected_n_embd", 5120),
        ("expected_n_layer", 40),
        ("expected_n_ubatch", 64),
    ):
        exact(value[key], expected, f"static.cuda_route.{key}")
    exact(value["host"], "127.0.0.1", "static.cuda_route.host")
    exact(value["mechanism_commands"], phone["mechanism_commands"], "mechanism.cuda")
    model = exact_keys(
        value["model_artifact"],
        {"bytes", "path", "sha256", "stat"},
        "static.cuda_route.model_artifact",
    )
    exact(model["bytes"], MODEL_BYTES, "model.bytes")
    exact(model["path"], MODEL_PATH, "model.path")
    exact(model["sha256"], MODEL_SHA256, "model.sha256")
    worker = exact_keys(
        value["worker"],
        {
            "argv",
            "cwd",
            "environment",
            "executable",
            "runtime_component_ids",
            "runtime_executable",
            "shutdown_timeout_ms",
            "startup_timeout_ms",
        },
        "static.cuda_route.worker",
    )
    for name in ("executable", "runtime_executable"):
        _artifact_identity(
            worker[name],
            f"static.cuda_route.worker.{name}",
        )
    argv = worker["argv"]
    exact(
        argv[0],
        worker["executable"]["path"],
        "static.cuda_route.worker.argv0",
    )
    require(
        worker["runtime_executable"]["path"] in argv,
        "E_CUDA_RUNTIME_ARGV",
    )
    environment = worker["environment"]
    require(type(environment) is dict, "E_CUDA_ENV")
    for key, expected in (
        ("LAYERSPLIT_MEMORY_CERT", "1"),
        ("LAYERSPLIT_MODEL_SHA256", MODEL_SHA256),
        ("LAYERSPLIT_PLACEMENT_CERT", "1"),
    ):
        exact(environment.get(key), expected, f"static.cuda_route.worker.env.{key}")
    for option, expected in (
        ("--mode", "monov3"),
        ("--backend", "CUDA0"),
        ("--layer-start", "0"),
        ("--layer-end", "40"),
        ("--model", MODEL_PATH),
    ):
        _option(argv, option, expected, "static.cuda_route.worker")
    codec = exact_keys(
        value["codec"],
        {"argv", "cwd", "environment", "executable", "timeout_ms"},
        "static.cuda_route.codec",
    )
    codec_executable = _artifact_identity(
        codec["executable"],
        "static.cuda_route.codec.executable",
    )
    require(type(codec["argv"]) is list and bool(codec["argv"]), "E_CUDA_CODEC_ARGV")
    exact(codec["argv"][0], codec_executable["path"], "static.cuda_route.codec.argv0")
    _option(codec["argv"], "--model", MODEL_PATH, "static.cuda_route.codec")
    _option(
        codec["argv"],
        "--model-sha256",
        MODEL_SHA256,
        "static.cuda_route.codec",
    )
    nvidia = exact_keys(
        value["nvidia_smi"],
        {"device_argv", "executable", "process_argv", "timeout_ms"},
        "static.cuda_route.nvidia_smi",
    )
    nvidia_executable = _artifact_identity(
        nvidia["executable"],
        "static.cuda_route.nvidia_smi.executable",
    )
    expected_device = [
        nvidia_executable["path"],
        f"--id={CUDA_UUID}",
        "--query-gpu=name,uuid,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    expected_process = [
        nvidia_executable["path"],
        f"--id={CUDA_UUID}",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    exact(nvidia["device_argv"], expected_device, "static.cuda_route.nvidia.device")
    exact(nvidia["process_argv"], expected_process, "static.cuda_route.nvidia.process")
    return value


def _validate_no_live_identity(
    static: dict[str, Any],
    topology: dict[str, Any],
) -> None:
    forbidden = set()
    observed = topology.get("observed", {})
    cuda = observed.get("cuda", {}) if type(observed) is dict else {}
    if type(cuda) is dict and type(cuda.get("boot_id")) is str:
        forbidden.add(cuda["boot_id"])
    phones = observed.get("phones", {}) if type(observed) is dict else {}
    if type(phones) is dict:
        for value in phones.values():
            if type(value) is dict:
                for key in ("boot_id", "wifi_ipv4", "wifi_selector"):
                    if type(value.get(key)) is str:
                        forbidden.add(value[key])
    serialized = canonical_bytes(static)
    for item in forbidden:
        require(
            item.encode("ascii") not in serialized,
            f"E_LIVE_IDENTITY: {item}",
        )
    for boot_id in UNBOUND_BOOT_IDS.values():
        require(boot_id not in serialized.decode("ascii"), "E_PREBOUND_BOOT_ID")


def _cross_bind_static(
    *,
    cuda: dict[str, Any],
    phone: dict[str, Any],
    runtime_inventory: dict[str, Any],
    operator: dict[str, Any],
    operator_files: dict[str, Any],
) -> None:
    for field, file_name in (
        ("model_artifact", "model"),
    ):
        exact(
            cuda[field],
            _artifact_identity(
                operator_files[file_name],
                f"operator.files.{file_name}",
            ),
            f"cuda.{field}",
        )
    for field, file_name in (
        ("executable", "cuda_launcher"),
        ("runtime_executable", "cuda_runtime"),
    ):
        exact(
            cuda["worker"][field],
            _artifact_identity(
                operator_files[file_name],
                f"operator.files.{file_name}",
            ),
            f"cuda.worker.{field}",
        )
    for container, file_name in (
        ("codec", "codec"),
        ("nvidia_smi", "nvidia_smi"),
    ):
        exact(
            cuda[container]["executable"],
            _artifact_identity(
                operator_files[file_name],
                f"operator.files.{file_name}",
            ),
            f"cuda.{container}.executable",
        )
    codec_pin = _artifact_identity(
        operator_files["codec"],
        "operator.files.codec",
    )
    exact(phone["codec"]["argv"][0], codec_pin["path"], "phone.codec.argv0")
    exact(phone["codec"]["executable_bytes"], codec_pin["bytes"], "phone.codec.bytes")
    exact(
        phone["codec"]["executable_sha256"],
        codec_pin["sha256"],
        "phone.codec.sha256",
    )
    cuda_bundle = next(
        value
        for value in runtime_inventory["bundles"]
        if value["bundle_id"] == "cuda_route"
    )
    exact(
        cuda["worker"]["runtime_component_ids"],
        cuda_bundle["required_component_ids"],
        "cuda.worker.runtime_component_ids",
    )
    cuda_launcher = next(
        value
        for value in runtime_inventory["components"]
        if value["component_id"] == cuda_bundle["launcher_component_id"]
    )
    exact(
        cuda["worker"]["runtime_executable"]["path"],
        cuda_launcher["path"],
        "cuda.worker.runtime_executable.path",
    )
    exact(
        cuda["worker"]["runtime_executable"]["sha256"],
        cuda_launcher["sha256"],
        "cuda.worker.runtime_executable.sha256",
    )
    tokenizer = next(
        (
            value
            for value in runtime_inventory["components"]
            if value["component_id"] == "cuda-tokenize"
        ),
        None,
    )
    require(type(tokenizer) is dict, "E_TOKENIZER_COMPONENT")
    exact(codec_pin["path"], tokenizer["path"], "runtime.tokenizer.path")
    exact(codec_pin["sha256"], tokenizer["sha256"], "runtime.tokenizer.sha256")
    exact(cuda["port"], operator["ports"]["cuda_route"], "cuda.port")
    exact(phone["relay_port"], operator["ports"]["relay"], "phone.relay_port")
    exact(
        cuda["worker"]["cwd"],
        operator["directories"]["cuda_bundle_root"],
        "cuda.worker.cwd",
    )

    components = {
        value["component_id"]: value
        for value in runtime_inventory["components"]
    }
    bundles = {
        value["bundle_id"]: value
        for value in runtime_inventory["bundles"]
    }
    process_specs = {
        "op12_stagenet": ("op12", "op12_stage"),
        "op15_direct_relay": ("op15", "relay"),
        "op15_stagenet": ("op15", "op15_stage"),
    }
    for name, (endpoint, port_name) in process_specs.items():
        process = phone["processes"][name]
        plan = _inline_plan(process["argv"], f"processes.{name}")
        bundle = bundles[name]
        required_ids = bundle["required_component_ids"]
        exact(
            process["runtime_component_ids"],
            required_ids,
            f"processes.{name}.runtime_component_ids",
        )
        launcher = components[bundle["launcher_component_id"]]
        exact(
            process["runtime_executable_path"],
            launcher["path"],
            f"processes.{name}.runtime_executable_path",
        )
        exact(
            process["runtime_executable_sha256"],
            launcher["sha256"],
            f"processes.{name}.runtime_executable_sha256",
        )
        expected_components = [
            {
                key: components[component_id][key]
                for key in ("bytes", "component_id", "path", "sha256", "stat")
            }
            for component_id in required_ids
        ]
        exact(plan["components"], expected_components, f"{name}.components")
        exact(
            plan["launcher_component_id"],
            bundle["launcher_component_id"],
            f"{name}.launcher_component_id",
        )
        exact(plan["mode"], "android", f"{name}.mode")
        android = plan["android"]
        adb = _artifact_identity(operator_files["adb"], "operator.files.adb")
        exact(android["adb_path"], adb["path"], f"{name}.android.adb_path")
        exact(android["adb_sha256"], adb["sha256"], f"{name}.android.adb_sha256")
        route = plan["route"]
        if name.endswith("stagenet"):
            exact(route["port"], operator["ports"][port_name], f"{name}.port")
            exact(
                process["runtime_executable_path"],
                phone["phones"][endpoint]["expected_worker_executable_path"],
                f"{name}.expected_worker_path",
            )
            exact(
                process["runtime_executable_sha256"],
                phone["phones"][endpoint]["expected_worker_executable_sha256"],
                f"{name}.expected_worker_sha256",
            )
        else:
            exact(
                route["listen_port"],
                operator["ports"]["relay"],
                f"{name}.listen_port",
            )
            exact(
                route["tail_source_port"],
                operator["ports"]["relay_tail_source"],
                f"{name}.tail_source_port",
            )


def build_spec(inventory_path: Path, inventory_sha256: str) -> dict[str, Any]:
    inventory, inventory_raw = read_canonical(inventory_path, "inventory")
    exact(sha256(inventory_raw), digest(inventory_sha256, "inventory_sha256"),
          "inventory.sha256")
    inventory = exact_keys(
        inventory,
        {
            "captured_on",
            "inputs",
            "model_id",
            "phase",
            "route_epoch",
            "schema",
            "static",
        },
        "inventory",
    )
    exact(inventory["schema"], INVENTORY_SCHEMA, "inventory.schema")
    exact(inventory["phase"], PHASE, "inventory.phase")
    exact(inventory["model_id"], MODEL_ID, "inventory.model_id")
    route_epoch = integer(inventory["route_epoch"], "inventory.route_epoch", 1)
    captured = exact_keys(
        inventory["captured_on"],
        {
            "controller_host",
            "cuda_uuid",
            "phone_adb_port",
            "physical_usb_selectors",
        },
        "inventory.captured_on",
    )
    exact(captured["controller_host"], CONTROLLER_HOST, "captured.controller")
    exact(captured["cuda_uuid"], CUDA_UUID, "captured.cuda_uuid")
    exact(captured["phone_adb_port"], PHONE_ADB_PORT, "captured.adb_port")
    exact(
        captured["physical_usb_selectors"],
        {name: value["serial"] for name, value in EXPECTED_PHONES.items()},
        "captured.physical_usb_selectors",
    )

    pins = exact_keys(inventory["inputs"], set(INPUT_SCHEMAS), "inventory.inputs")
    paths = {}
    values = {}
    raws = {}
    for name, schema in sorted(INPUT_SCHEMAS.items()):
        path, value, raw = _read_pin(pins[name], f"inputs.{name}")
        exact(value.get("schema"), schema, f"inputs.{name}.schema")
        paths[name] = path
        values[name] = value
        raws[name] = raw
    exact(sha256(raws["contract"]), CONTRACT_SHA256, "contract.sha256")
    exact(sha256(raws["candidate"]), CANDIDATE_SHA256, "candidate.sha256")
    contract = values["contract"]
    candidate = values["candidate"]
    _find_model(candidate)
    exact(
        contract["candidate_lock"],
        {
            "bytes": len(raws["candidate"]),
            "model_id": MODEL_ID,
            "sha256": sha256(raws["candidate"]),
            "slot": "A",
        },
        "contract.candidate_lock",
    )
    route_lock = contract["incumbent_route_lock"]
    exact(route_lock["backend"], "GPUOpenCL", "contract.route.backend")
    exact(route_lock["cut_layer"], 30, "contract.route.cut")
    exact(route_lock["op15_stored_layers"], [0, 32], "contract.route.op15")
    exact(route_lock["op12_stored_layers"], [24, 40], "contract.route.op12")
    history = values["token_history"]
    exact(history["candidate_sha256"], CANDIDATE_SHA256, "history.candidate")
    exact(history["model_sha256"], MODEL_SHA256, "history.model")
    exact(history["n_ctx_seq"], 512, "history.n_ctx_seq")
    exact(history["continuation_tokens_per_request"], 8, "history.continuation")
    exact(
        values["tokenizer_plan"]["model"]["sha256"],
        MODEL_SHA256,
        "tokenizer.model",
    )
    exact(
        values["cuda_monolithic_launch"]["model_sha256"],
        MODEL_SHA256,
        "monolithic.model",
    )
    topology_verifier = _load_topology_verifier()
    observed = topology_verifier.validate_topology(
        values["topology_receipt"],
        contract,
        require_fresh=False,
    )
    exact(observed["controller"]["host"], CONTROLLER_HOST, "topology.controller")
    exact(observed["cuda"]["uuid"], CUDA_UUID, "topology.cuda_uuid")
    for endpoint, expected in EXPECTED_PHONES.items():
        exact(
            observed["phones"][endpoint]["usb_selector"],
            expected["serial"],
            f"topology.{endpoint}.usb_selector",
        )

    operator = exact_keys(
        values["operator_input"],
        {
            "controller_host",
            "directories",
            "files",
            "model_id",
            "phase",
            "ports",
            "route_epoch",
            "schema",
            "topology",
        },
        "operator",
    )
    exact(operator["controller_host"], CONTROLLER_HOST, "operator.controller")
    exact(operator["phase"], PHASE, "operator.phase")
    exact(operator["model_id"], MODEL_ID, "operator.model")
    exact(operator["route_epoch"], route_epoch, "operator.route_epoch")
    ports = exact_keys(
        operator["ports"],
        {
            "adb_server",
            "cuda_monolithic",
            "cuda_route",
            "op12_stage",
            "op15_stage",
            "relay",
            "relay_tail_source",
        },
        "operator.ports",
    )
    exact(ports["adb_server"], PHONE_ADB_PORT, "operator.adb_port")
    topology = exact_keys(
        operator["topology"],
        {"adb_host", "adb_port", "cuda_uuid", "physical_serials"},
        "operator.topology",
    )
    exact(topology["adb_host"], "127.0.0.1", "operator.adb_host")
    exact(topology["adb_port"], PHONE_ADB_PORT, "operator.adb_port")
    exact(topology["cuda_uuid"], CUDA_UUID, "operator.cuda_uuid")
    exact(
        topology["physical_serials"],
        captured["physical_usb_selectors"],
        "operator.physical_serials",
    )
    operator_files = exact_keys(
        operator["files"],
        {
            "adb",
            "candidate",
            "codec",
            "contract",
            "cuda_launcher",
            "cuda_monolithic_launch",
            "cuda_runtime",
            "model",
            "monolithic_launcher",
            "monolithic_runtime",
            "nvidia_smi",
            "python",
            "quality_corpus",
            "runtime_bundle_inventory",
            "ssh",
            "token_history",
            "tokenizer_plan",
            "topology_receipt",
        },
        "operator.files",
    )
    for name in (
        "candidate",
        "contract",
        "cuda_monolithic_launch",
        "runtime_bundle_inventory",
        "token_history",
        "tokenizer_plan",
        "topology_receipt",
    ):
        exact(
            _pin_identity(
                operator_files[name],
                f"operator.files.{name}",
            ),
            pins[name],
            f"operator.files.{name}",
        )

    static = exact_keys(
        inventory["static"],
        {"cuda_route", "joint_cwd", "phone_route", "runtime"},
        "inventory.static",
    )
    phone = _validate_phone_static(static["phone_route"], contract)
    exact(phone["route_epoch"], route_epoch, "phone.route_epoch")
    cuda = _validate_cuda_static(static["cuda_route"], phone)
    exact(cuda["route_epoch"], route_epoch, "cuda.route_epoch")
    runtime = _validate_runtime_static(
        static["runtime"],
        values["runtime_bundle_inventory"],
        values["cuda_monolithic_launch"],
        contract,
    )
    joint_cwd = _absolute(static["joint_cwd"], "static.joint_cwd")
    require(joint_cwd.is_dir(), "E_JOINT_CWD")
    exact(
        str(joint_cwd),
        operator["directories"]["joint_cwd"],
        "static.joint_cwd",
    )
    _cross_bind_static(
        cuda=cuda,
        phone=phone,
        runtime_inventory=values["runtime_bundle_inventory"],
        operator=operator,
        operator_files=operator_files,
    )
    _validate_no_live_identity(static, values["topology_receipt"])

    return {
        "candidate_path": str(paths["candidate"]),
        "contract_path": str(paths["contract"]),
        "cuda_monolithic_launch_path": str(paths["cuda_monolithic_launch"]),
        "cuda_route_static": cuda,
        "identity_placeholders": UNBOUND_BOOT_IDS,
        "joint": {"cwd": str(joint_cwd)},
        "network_placeholders": UNBOUND_PHONE_NETWORK,
        "phone_route_static": phone,
        "runtime_static": runtime,
        "schema": SPEC_SCHEMA,
        "token_history_path": str(paths["token_history"]),
        "tokenizer_plan_path": str(paths["tokenizer_plan"]),
    }


def write_new(path: Path, value: Any) -> None:
    require(path.is_absolute() and not path.exists(), "E_OUTPUT_EXISTS")
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
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
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _load_originator():
    raw = read_regular(ORIGINATOR_PATH, "originator")
    exact(sha256(raw), ORIGINATOR_SHA256, "originator.sha256")
    module = types.ModuleType("s39_v24_originator_for_desktop")
    module.__file__ = str(ORIGINATOR_PATH)
    exec(compile(raw, str(ORIGINATOR_PATH), "exec"), module.__dict__)
    return module


def _load_topology_verifier():
    raw = read_regular(TOPOLOGY_VERIFIER_PATH, "topology_verifier")
    exact(
        sha256(raw),
        TOPOLOGY_VERIFIER_SHA256,
        "topology_verifier.sha256",
    )
    module = types.ModuleType("s39_v24_topology_verifier_for_desktop")
    module.__file__ = str(TOPOLOGY_VERIFIER_PATH)
    exec(compile(raw, str(TOPOLOGY_VERIFIER_PATH), "exec"), module.__dict__)
    return module


def _require_launcher_compatible(phone_static: dict[str, Any]) -> None:
    raw = read_regular(USB_LAUNCHER_PATH, "usb_launcher")
    exact(sha256(raw), USB_LAUNCHER_SHA256, "usb_launcher.sha256")
    require(os.access(USB_LAUNCHER_PATH, os.X_OK), "E_USB_LAUNCHER_EXECUTABLE")
    module = types.ModuleType("s39_v24_usb_launcher_compatibility")
    module.__file__ = str(USB_LAUNCHER_PATH)
    exec(compile(raw, str(USB_LAUNCHER_PATH), "exec"), module.__dict__)
    frozen = module.load_frozen_launcher()
    for name in (
        "op12_stagenet",
        "op15_direct_relay",
        "op15_stagenet",
    ):
        process = phone_static["processes"][name]
        argv = process["argv"]
        exact(argv[0], str(USB_LAUNCHER_PATH), f"launcher_compatibility.{name}.path")
        exact(
            process["launcher_bytes"],
            len(raw),
            f"launcher_compatibility.{name}.bytes",
        )
        exact(
            process["launcher_sha256"],
            USB_LAUNCHER_SHA256,
            f"launcher_compatibility.{name}.sha256",
        )
        _inline_plan(argv, f"launcher_compatibility.{name}")
        plan_index = argv.index("--plan-json") + 1
        digest_index = argv.index("--plan-sha256") + 1
        try:
            module.validate_plan(
                argv[plan_index],
                argv[digest_index],
                frozen,
            )
        except Exception as error:
            raise MaterializeError(
                f"E_USB_LAUNCHER_COMPATIBILITY: {name}: {error}"
            ) from error


def materialize(
    *,
    inventory_path: Path,
    inventory_sha256: str,
    spec_output: Path,
    output_paths: dict[str, Path],
    prospective_root_output: Path,
    report_output: Path,
) -> None:
    spec = build_spec(inventory_path, inventory_sha256)
    _require_launcher_compatible(spec["phone_route_static"])
    write_new(spec_output, spec)
    try:
        originator = _load_originator()
        originator.materialize(
            spec_path=spec_output,
            output_paths=output_paths,
            prospective_root_output=prospective_root_output,
            report_output=report_output,
        )
    except Exception:
        try:
            spec_output.unlink()
        except FileNotFoundError:
            pass
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--spec-output", type=Path, required=True)
    parser.add_argument("--cuda-route-launch", type=Path, required=True)
    parser.add_argument("--joint-capture-plan", type=Path, required=True)
    parser.add_argument("--phone-route-launch", type=Path, required=True)
    parser.add_argument("--runtime-plan", type=Path, required=True)
    parser.add_argument("--prospective-root", type=Path, required=True)
    parser.add_argument("--dry-run-report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        materialize(
            inventory_path=args.inventory.resolve(),
            inventory_sha256=args.inventory_sha256,
            spec_output=args.spec_output.resolve(),
            output_paths={
                "cuda_route_launch": args.cuda_route_launch.resolve(),
                "joint_capture_plan": args.joint_capture_plan.resolve(),
                "phone_route_launch": args.phone_route_launch.resolve(),
                "runtime_plan": args.runtime_plan.resolve(),
            },
            prospective_root_output=args.prospective_root.resolve(),
            report_output=args.dry_run_report.resolve(),
        )
        return 0
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"V24_A_ONLY_INPUTS_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
