#!/usr/bin/env python3
"""Validate the prospective-to-bound managed launcher boundary."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SCHEMA = "s39-v26-managed-plan-gate-result-v1"
PLAN_SCHEMA = "s39-managed-runtime-launch-plan-v1"
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

PLAN_KEYS = {
    "android",
    "bundle_id",
    "components",
    "endpoint",
    "launcher_component_id",
    "mode",
    "route",
    "schema",
    "ssh",
}
EXPECTATION_KEYS = {
    "android",
    "bound_route",
    "bundle_id",
    "components",
    "cuda_environment",
    "cuda_flags",
    "endpoint",
    "launcher_component_id",
    "managed_launcher",
    "mode",
    "prospective_route",
    "runtime_launcher",
}
COMPONENT_KEYS = {
    "bytes",
    "component_id",
    "path",
    "sha256",
    "stat",
}
STAT_KEYS = {
    "build_id",
    "ctime_ns",
    "device_id",
    "inode",
    "mode",
    "mtime_ns",
    "size",
}
ANDROID_KEYS = {
    "adb_path",
    "adb_port",
    "adb_selector",
    "adb_sha256",
    "boot_id_source",
    "physical_serial",
    "shutdown_timeout_ms",
    "startup_timeout_ms",
}
LOCAL_ROUTE_KEYS = {"argv", "cwd", "environment", "kind"}
CUDA_FLAG_KEYS = {
    "--backend",
    "--driver-batch",
    "--driver-context",
    "--driver-max-prefill",
    "--layer-end",
    "--layer-start",
    "--mode",
    "--model",
    "--port",
}
CUDA_ENV_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "LAYERSPLIT_MEMORY_CERT",
    "LAYERSPLIT_MODEL_SHA256",
    "LAYERSPLIT_PLACEMENT_CERT",
    "LD_LIBRARY_PATH",
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


class ManagedPlanGateError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ManagedPlanGateError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}",
    )


def exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    require(set(value) == expected, f"E_KEYS: {field}")
    return value


def text(value: Any, field: str, maximum: int = 4096) -> str:
    require(
        type(value) is str
        and 0 < len(value) <= maximum
        and value.isascii()
        and "\x00" not in value
        and "\n" not in value,
        f"E_TEXT: {field}",
    )
    return value


def digest(value: Any, field: str) -> str:
    value = text(value, field, 64)
    require(DIGEST_RE.fullmatch(value) is not None, f"E_DIGEST: {field}")
    return value


def absolute(value: Any, field: str) -> str:
    value = text(value, field)
    require(Path(value).is_absolute(), f"E_PATH: {field}")
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"E_INTEGER: {field}")
    return value


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ManagedPlanGateError(f"E_JSON_NUMBER: {value}")


def reject_float(value: str) -> None:
    raise ManagedPlanGateError(f"E_JSON_FLOAT: {value}")


def canonical_compact(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise ManagedPlanGateError("E_CANONICAL") from error


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_argv(value: Any, field: str) -> list[str]:
    require(
        type(value) is list
        and bool(value)
        and all(
            type(item) is str
            and 0 < len(item) <= 8192
            and item.isascii()
            and "\x00" not in item
            and "\n" not in item
            for item in value
        ),
        f"E_ARGV: {field}",
    )
    absolute(value[0], f"{field}[0]")
    return value


def _validate_stat(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, STAT_KEYS, field)
    exact(value["build_id"], None, f"{field}.build_id")
    for key in STAT_KEYS - {"build_id"}:
        integer(value[key], f"{field}.{key}")
    return value


def _validate_component(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, COMPONENT_KEYS, field)
    size = integer(value["bytes"], f"{field}.bytes", 1)
    component_id = text(value["component_id"], f"{field}.component_id", 128)
    require(
        all(character.isalnum() or character in "._-" for character in component_id),
        f"E_COMPONENT_ID: {field}",
    )
    absolute(value["path"], f"{field}.path")
    digest(value["sha256"], f"{field}.sha256")
    stat_record = _validate_stat(value["stat"], f"{field}.stat")
    exact(stat_record["size"], size, f"{field}.stat.size")
    return value


def _validate_expectation(value: Any) -> dict[str, Any]:
    value = exact_keys(value, EXPECTATION_KEYS, "expectation")
    mode = text(value["mode"], "expectation.mode", 32)
    require(mode in {"android", "local_cuda"}, "E_MODE")
    endpoint = text(value["endpoint"], "expectation.endpoint", 32)
    require(
        endpoint in ({"op12", "op15"} if mode == "android" else {"cuda"}),
        "E_ENDPOINT",
    )
    text(value["bundle_id"], "expectation.bundle_id", 128)

    managed = exact_keys(
        value["managed_launcher"],
        {"path", "sha256"},
        "expectation.managed_launcher",
    )
    absolute(managed["path"], "expectation.managed_launcher.path")
    digest(managed["sha256"], "expectation.managed_launcher.sha256")

    components = value["components"]
    require(type(components) is list and bool(components), "E_COMPONENTS")
    component_map = {}
    paths = set()
    for index, component in enumerate(components):
        component = _validate_component(component, f"expectation.components[{index}]")
        component_id = component["component_id"]
        require(component_id not in component_map, "E_COMPONENT_REUSE")
        require(component["path"] not in paths, "E_COMPONENT_PATH_REUSE")
        component_map[component_id] = component
        paths.add(component["path"])
    exact(list(component_map), sorted(component_map), "expectation.components.order")

    launcher_id = text(
        value["launcher_component_id"],
        "expectation.launcher_component_id",
        128,
    )
    require(launcher_id in component_map, "E_LAUNCHER_COMPONENT")
    runtime = exact_keys(
        value["runtime_launcher"],
        {"component_id", "path", "sha256"},
        "expectation.runtime_launcher",
    )
    exact(runtime["component_id"], launcher_id, "expectation.runtime.component_id")
    exact(runtime["path"], component_map[launcher_id]["path"], "expectation.runtime.path")
    exact(
        runtime["sha256"],
        component_map[launcher_id]["sha256"],
        "expectation.runtime.sha256",
    )

    if mode == "local_cuda":
        exact(value["android"], None, "expectation.android")
        flags = exact_keys(value["cuda_flags"], CUDA_FLAG_KEYS, "expectation.cuda_flags")
        for flag, item in flags.items():
            text(item, f"expectation.cuda_flags.{flag}")
        exact(flags["--mode"], "monov3", "expectation.cuda_flags.--mode")
        exact(flags["--backend"], "CUDA0", "expectation.cuda_flags.--backend")
        environment = exact_keys(
            value["cuda_environment"],
            CUDA_ENV_KEYS,
            "expectation.cuda_environment",
        )
        for key, item in environment.items():
            text(key, "expectation.cuda_environment.key", 128)
            text(item, f"expectation.cuda_environment.{key}")
            require("=" not in key, "E_CUDA_ENV_KEY")
        exact(
            environment["LAYERSPLIT_MEMORY_CERT"],
            "1",
            "expectation.cuda_environment.LAYERSPLIT_MEMORY_CERT",
        )
        exact(
            environment["LAYERSPLIT_PLACEMENT_CERT"],
            "1",
            "expectation.cuda_environment.LAYERSPLIT_PLACEMENT_CERT",
        )
        digest(
            environment["LAYERSPLIT_MODEL_SHA256"],
            "expectation.cuda_environment.LAYERSPLIT_MODEL_SHA256",
        )
        require(
            environment["CUDA_VISIBLE_DEVICES"].startswith("GPU-"),
            "E_CUDA_VISIBLE_DEVICE",
        )
        absolute(
            environment["LD_LIBRARY_PATH"],
            "expectation.cuda_environment.LD_LIBRARY_PATH",
        )
        for field in ("prospective_route", "bound_route"):
            _validate_cuda_route(
                value[field],
                component_map[launcher_id]["path"],
                flags,
                environment,
                f"expectation.{field}",
            )
    else:
        exact(value["cuda_flags"], None, "expectation.cuda_flags")
        exact(value["cuda_environment"], None, "expectation.cuda_environment")
        android = exact_keys(value["android"], ANDROID_KEYS, "expectation.android")
        absolute(android["adb_path"], "expectation.android.adb_path")
        digest(android["adb_sha256"], "expectation.android.adb_sha256")
        integer(android["adb_port"], "expectation.android.adb_port", 1)
        serial = text(android["physical_serial"], "expectation.android.physical_serial", 255)
        exact(android["adb_selector"], serial, "expectation.android.adb_selector")
        exact(
            android["boot_id_source"],
            "phase_fresh_snapshot",
            "expectation.android.boot_id_source",
        )
        for key in ("startup_timeout_ms", "shutdown_timeout_ms"):
            integer(android[key], f"expectation.android.{key}", 1)
        for field in ("prospective_route", "bound_route"):
            _validate_android_route(
                value[field],
                endpoint,
                component_map[launcher_id]["path"],
                f"expectation.{field}",
            )
    return value


def _validate_cuda_route(
    value: Any,
    launcher_path: str,
    expected_flags: dict[str, str],
    expected_environment: dict[str, str],
    field: str,
) -> dict[str, Any]:
    value = exact_keys(value, LOCAL_ROUTE_KEYS, field)
    exact(value["kind"], "local_exec", f"{field}.kind")
    absolute(value["cwd"], f"{field}.cwd")
    exact(value["environment"], expected_environment, f"{field}.environment")
    argv = validate_argv(value["argv"], f"{field}.argv")
    exact(argv[0], launcher_path, f"{field}.argv0")
    exact(len(argv), 1 + 2 * len(CUDA_FLAG_KEYS), f"{field}.length")
    observed_flags = argv[1::2]
    exact(set(observed_flags), CUDA_FLAG_KEYS, f"{field}.flags")
    exact(len(observed_flags), len(set(observed_flags)), f"{field}.flag_reuse")
    for flag, expected in expected_flags.items():
        exact(argv.count(flag), 1, f"{field}.{flag}.count")
        index = argv.index(flag)
        require(index + 1 < len(argv), f"E_CUDA_FLAG_VALUE: {field}.{flag}")
        exact(argv[index + 1], expected, f"{field}.{flag}")
    exact(value["cwd"], expected_environment["LD_LIBRARY_PATH"], f"{field}.cwd")
    return value


def _validate_android_route(
    value: Any,
    endpoint: str,
    launcher_path: str,
    field: str,
) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    kind = value.get("kind")
    if kind == "stagenet_worker":
        value = exact_keys(value, WORKER_ROUTE_KEYS, field)
        exact(value["devices"], "GPUOpenCL", f"{field}.devices")
        exact(value["dynamic_cut"], True, f"{field}.dynamic_cut")
        exact(value["kv_unified"], True, f"{field}.kv_unified")
        exact(value["n_gpu_layers"], 999, f"{field}.n_gpu_layers")
        exact(value["placement_cert"], True, f"{field}.placement_cert")
        for key in (
            "driver_batch",
            "driver_context",
            "driver_max_prefill",
            "layer_end",
            "port",
        ):
            integer(value[key], f"{field}.{key}", 1)
        layer_start = integer(value["layer_start"], f"{field}.layer_start")
        require(layer_start < value["layer_end"], f"E_LAYER_RANGE: {field}")
        expected_mode = "stagenet" if endpoint == "op15" else "tailv3"
        exact(value["mode"], expected_mode, f"{field}.mode")
        absolute(value["model_path"], f"{field}.model_path")
        digest(value["model_sha256"], f"{field}.model_sha256")
        runtime_root = absolute(value["runtime_root"], f"{field}.runtime_root")
    elif kind == "direct_relay":
        exact(endpoint, "op15", f"{field}.endpoint")
        value = exact_keys(value, RELAY_ROUTE_KEYS, field)
        exact(value["emit_direct_frames"], True, f"{field}.emit_direct_frames")
        for key in ("head_host", "tail_host"):
            text(value[key], f"{field}.{key}", 255)
        for key in ("head_port", "listen_port", "tail_port", "tail_source_port"):
            port = integer(value[key], f"{field}.{key}", 1)
            require(port <= 65535, f"E_PORT: {field}.{key}")
        runtime_root = absolute(value["runtime_root"], f"{field}.runtime_root")
    else:
        raise ManagedPlanGateError(f"E_ANDROID_ROUTE_KIND: {field}")
    exact(
        str(Path(launcher_path).parent),
        runtime_root,
        f"{field}.runtime_root",
    )
    return value


def _parse_managed_argv(
    value: Any,
    expected_launcher_path: str,
    expected_boot_id: str,
    field: str,
) -> tuple[dict[str, Any], bytes, str]:
    argv = validate_argv(value, field)
    exact(len(argv), 7, f"{field}.length")
    exact(
        [argv[0], argv[1], argv[3], argv[5]],
        [
            expected_launcher_path,
            "--plan-json",
            "--plan-sha256",
            "--boot-id",
        ],
        f"{field}.shape",
    )
    raw = argv[2].encode("ascii")
    exact(argv[4], sha256(raw), f"{field}.plan_sha256")
    exact(argv[6], expected_boot_id, f"{field}.boot_id")
    try:
        plan = json.loads(
            raw,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
            parse_float=reject_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManagedPlanGateError(f"E_PLAN_JSON: {field}") from error
    require(type(plan) is dict, f"E_PLAN_TYPE: {field}")
    exact(canonical_compact(plan), raw, f"{field}.canonical")
    require(
        expected_boot_id.encode("ascii") not in raw,
        f"E_BOOT_INSIDE_PLAN: {field}",
    )
    return plan, raw, argv[4]


def _validate_plan(
    plan: Any,
    expectation: dict[str, Any],
    expected_route: dict[str, Any],
    field: str,
) -> None:
    plan = exact_keys(plan, PLAN_KEYS, field)
    exact(plan["schema"], PLAN_SCHEMA, f"{field}.schema")
    for key in ("mode", "endpoint", "bundle_id", "launcher_component_id"):
        exact(plan[key], expectation[key], f"{field}.{key}")
    exact(plan["components"], expectation["components"], f"{field}.components")
    exact(plan["ssh"], None, f"{field}.ssh")
    exact(plan["route"], expected_route, f"{field}.route")
    if expectation["mode"] == "local_cuda":
        exact(plan["android"], None, f"{field}.android")
        _validate_cuda_route(
            plan["route"],
            expectation["runtime_launcher"]["path"],
            expectation["cuda_flags"],
            expectation["cuda_environment"],
            f"{field}.route",
        )
    else:
        exact(plan["android"], expectation["android"], f"{field}.android")
        _validate_android_route(
            plan["route"],
            expectation["endpoint"],
            expectation["runtime_launcher"]["path"],
            f"{field}.route",
        )


def validate_managed_plan_pair(
    prospective_argv: Any,
    bound_argv: Any,
    *,
    expectation: Any,
    prospective_boot_id: Any,
    bound_boot_id: Any,
) -> dict[str, Any]:
    """Validate one managed plan before and after post-reboot binding."""

    expectation = _validate_expectation(copy.deepcopy(expectation))
    prospective_boot_id = text(
        prospective_boot_id,
        "prospective_boot_id",
        64,
    )
    bound_boot_id = text(bound_boot_id, "bound_boot_id", 64)
    require(
        UUID_RE.fullmatch(prospective_boot_id) is not None,
        "E_PROSPECTIVE_BOOT_ID",
    )
    require(UUID_RE.fullmatch(bound_boot_id) is not None, "E_BOUND_BOOT_ID")
    require(bound_boot_id != prospective_boot_id, "E_STALE_BOOT")

    launcher_path = expectation["managed_launcher"]["path"]
    prospective, prospective_raw, prospective_digest = _parse_managed_argv(
        prospective_argv,
        launcher_path,
        prospective_boot_id,
        "prospective.argv",
    )
    bound, bound_raw, bound_digest = _parse_managed_argv(
        bound_argv,
        launcher_path,
        bound_boot_id,
        "bound.argv",
    )
    for field, raw in (
        ("prospective", prospective_raw),
        ("bound", bound_raw),
    ):
        for boot_id in (prospective_boot_id, bound_boot_id):
            require(
                boot_id.encode("ascii") not in raw,
                f"E_BOOT_INSIDE_PLAN: {field}",
            )
    _validate_plan(
        prospective,
        expectation,
        expectation["prospective_route"],
        "prospective.plan",
    )
    _validate_plan(
        bound,
        expectation,
        expectation["bound_route"],
        "bound.plan",
    )

    prospective_fixed = copy.deepcopy(prospective)
    bound_fixed = copy.deepcopy(bound)
    prospective_fixed["route"] = expectation["bound_route"]
    exact(prospective_fixed, bound_fixed, "pair.unapproved_change")

    return {
        "bound_boot_id": bound_boot_id,
        "bound_plan_sha256": bound_digest,
        "bundle_id": expectation["bundle_id"],
        "endpoint": expectation["endpoint"],
        "managed_launcher_sha256": expectation["managed_launcher"]["sha256"],
        "mode": expectation["mode"],
        "prospective_boot_id": prospective_boot_id,
        "prospective_plan_sha256": prospective_digest,
        "runtime_launcher_sha256": expectation["runtime_launcher"]["sha256"],
        "schema": SCHEMA,
        "status": "MANAGED_PLAN_BOUNDARY_PASS",
    }
