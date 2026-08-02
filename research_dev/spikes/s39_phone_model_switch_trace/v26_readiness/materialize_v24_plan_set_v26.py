#!/usr/bin/env python3
"""Materialize the five V2.4 prephase plan artifacts from V2.6 evidence.

This is the bounded V2.6-to-V2.4 adapter. It derives the desktop inventory,
operator input, and runtime-bundle closure input from the materialized V2.6
production evidence, mirrors the byte-verified frozen V2.4 tool closure to an
isolated tree on the acquisition desktop, and drives the UNMODIFIED frozen
`materialize_a_only_inputs_v1.py` -> `originate_runtime_v1.py` chain there.
Every frozen validator (launcher compatibility, live-identity exclusion,
originator revalidation, authority validate_runtime_plan) stays in the path.
No model process is launched.
"""

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
V23 = S39 / "v23_readiness"
PREPHASE = V24 / "results" / "prephase_20260726T0915Z"

RECORD_SCHEMA = "s39-cp0-r1-v26-v24-plan-set-materialization-v1"
VALIDATION_SCHEMA = "s39-cp0-r1-v26-v24-plan-set-validation-v1"
CONFIRMATION = "RUN_CP0_R1_V26_V24_PLAN_SET"
MODEL_ID = "qwen3-14b-q4_k_m"
MODEL_SHA256 = (
    "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"
)
PHASE = "A_ONLY"
ROUTE_EPOCH = 1

DEFAULT_V26_ROOT = HERE / "results" / "production_materialization_20260726T170521Z"

MIRROR_ROOT = "/home/zhihao/s39-v26-a-only/repo-v1/s39"
INPUTS_ROOT = "/home/zhihao/s39-v26-a-only/v24-plan-inputs-v1"
DESKTOP_WORK_DIR = "/home/zhihao/s39-v26-a-only"
DESKTOP_PYTHON = "/usr/bin/python3"
DESKTOP_NVIDIA_SMI = "/usr/bin/nvidia-smi"
DESKTOP_SSH = "/usr/bin/ssh"

PLAN_NAMES = (
    "cuda-route-launch.json",
    "joint-capture-plan.json",
    "phone-route-launch.json",
    "runtime-bundle-plan.json",
    "prospective-runtime-root.json",
)
INPUT_NAMES = (
    "runtime-bundle-inventory.json",
    "topology-receipt.json",
    "operator-input.json",
    "desktop-inventory.json",
)
SIDE_NAMES = ("prospective-runtime-spec.json", "dry-run-report.json")

MIRROR_EXECUTABLES = {
    "v24_readiness/desktop_deployment_v1/managed_runtime_launcher_usb_v1.py",
    "v24_readiness/desktop_deployment_v1/"
    "managed_runtime_launcher_snapshot_v1.py",
    "v24_readiness/desktop_deployment_v1/phone_runtime_probe_snapshot_v1.py",
}
PROBE_SOURCE = (
    V23 / "a_only_acquisition_driver_v1" / "producers_v1"
    / "phone_runtime_probe_v1.py"
)
PROBE_MIRROR_RELATIVE = (
    "v24_readiness/desktop_deployment_v1/phone_runtime_probe_v1.py"
)
SNAPSHOT_LAUNCHER_RELATIVE = (
    "v24_readiness/desktop_deployment_v1/"
    "managed_runtime_launcher_snapshot_v1.py"
)
SNAPSHOT_PROBE_RELATIVE = (
    "v24_readiness/desktop_deployment_v1/phone_runtime_probe_snapshot_v1.py"
)
DRIVER_RELATIVE = (
    "v24_readiness/desktop_deployment_v1/originate_from_inventory_v26.py"
)
EXTRA_MIRROR_SOURCES = {
    PROBE_MIRROR_RELATIVE: PROBE_SOURCE,
    SNAPSHOT_LAUNCHER_RELATIVE: (
        HERE / "managed_runtime_launcher_snapshot_v1.py"
    ),
    SNAPSHOT_PROBE_RELATIVE: HERE / "phone_runtime_probe_snapshot_v1.py",
    DRIVER_RELATIVE: HERE / "originate_from_inventory_v26.py",
}
PROBE_WIRING = {
    "op12": {"local_port": 39126, "peer_port": 39129, "role": "stagenet_worker"},
    "op15": {"local_port": 39129, "peer_port": 39126, "role": "direct_relay"},
}
UNBOUND_NETWORK = {"op12": "0.0.0.12", "op15": "0.0.0.15"}
UNBOUND_INTERFACE = "UNBOUND_AFTER_REBOOT"
MIRROR_FILES = (
    "CP0_R1_CANDIDATE.json",
    "CP0_R1_EVIDENCE_CONTRACT_V2.json",
    "CP0_R1_EVIDENCE_CONTRACT_V2_2.json",
    "CP0_R1_MMLU64_CORPUS_V2_2.jsonl",
    "build_cp0_r1_mmlu64_v22.py",
    "build_cp0_r1_v21.py",
    "build_cp0_r1_v22.py",
    "cp0_r1_evidence_v2.py",
    "cp0_r1_evidence_v21.py",
    "cp0_r1_evidence_v22.py",
    "v23_readiness/CP0_R1_EVIDENCE_CONTRACT_V2_3.json",
    "v23_readiness/a_only_acquisition_driver_v1/producers_v1/"
    "managed_runtime_launcher_v1.py",
    "v23_readiness/production_v2/artifact_snapshot_driver_v2.py",
    "v23_readiness/production_v2/driver_common_v2.py",
    "v23_readiness/production_v2/fresh_readiness_driver_v2.py",
    "v23_readiness/production_v2/prepare_zero_swap_v1.py",
    "v23_readiness/production_v2/runtime_bundle_overlay_v1.py",
    "v23_readiness/production_v2/source_entry_v2.py",
    "v24_readiness/CP0_R1_EVIDENCE_CONTRACT_V2_4.json",
    "v24_readiness/build_contract_v24.py",
    "v24_readiness/cp0_r1_evidence_v24.py",
    "v24_readiness/v24_common.py",
    "v24_readiness/desktop_deployment_v1/artifact_root_capture_v1.py",
    "v24_readiness/desktop_deployment_v1/fast_fresh_capture_v1.py",
    "v24_readiness/desktop_deployment_v1/managed_runtime_launcher_usb_v1.py",
    "v24_readiness/desktop_deployment_v1/materialize_a_only_inputs_v1.py",
    "v24_readiness/desktop_deployment_v1/verify_topology_v1.py",
    "v24_readiness/orchestration_v1/build_a_only_plan_v1.py",
    "v24_readiness/orchestration_v1/orchestration_v1.py",
    "v24_readiness/orchestration_v1/run_a_only_v1.py",
    "v24_readiness/producers_v1/build_cuda_monolithic_launch_v1.py",
    "v24_readiness/producers_v1/build_cuda_route_launch_v1.py",
    "v24_readiness/producers_v1/build_joint_capture_plan_v1.py",
    "v24_readiness/producers_v1/build_phone_route_launch_v1.py",
    "v24_readiness/producers_v1/cuda_monolithic_v1.py",
    "v24_readiness/producers_v1/cuda_route_v1.py",
    "v24_readiness/producers_v1/joint_phone_cuda_v1.py",
    "v24_readiness/producers_v1/phone_route_v1.py",
    "v24_readiness/production_plan_v1/fan_in_v1.py",
    "v24_readiness/production_plan_v1/identity_binding_v1.py",
    "v24_readiness/production_plan_v1/materialize_config_v1.py",
    "v24_readiness/production_plan_v1/originate_runtime_v1.py",
    "v24_readiness/production_plan_v1/phase_lock_v1.py",
    "v24_readiness/production_plan_v1/preparation_v1.py",
    "v24_readiness/production_plan_v1/production_common_v1.py",
    "v24_readiness/production_plan_v1/readiness_projection_v1.py",
    "v24_readiness/results/prephase_20260726T0915Z/cuda-monolithic-launch.json",
    "v24_readiness/results/prephase_20260726T0915Z/token-history.json",
    "v24_readiness/results/prephase_20260726T0915Z/tokenizer-plan.json",
)
MIRROR_PINS = {
    "CP0_R1_CANDIDATE.json": (
        "ee3196ca660fa7eb7ea293260dc98dd6fdbf14571a4d4c5aece5343fb29b28d8"
    ),
    "v23_readiness/a_only_acquisition_driver_v1/producers_v1/"
    "managed_runtime_launcher_v1.py": (
        "b97941dc30399135b04e98dbdf102aaeb6c695c55a7b990402aeafd21ed4a245"
    ),
    "v24_readiness/CP0_R1_EVIDENCE_CONTRACT_V2_4.json": (
        "264d16b33d56176ee6d3ac84471b3616d4b17e5bea785e08d03723ceb73a439f"
    ),
    "v24_readiness/desktop_deployment_v1/"
    "managed_runtime_launcher_usb_v1.py": (
        "52c4e1f251f4daa2c856857fcdf23fdc5a85c0d30eff93365996a8f1378611ce"
    ),
    "v24_readiness/desktop_deployment_v1/verify_topology_v1.py": (
        "602dc531001cea06fc30161879c1bcf454b06f92c6c9942bfc33571ed3885c66"
    ),
    "v24_readiness/production_plan_v1/originate_runtime_v1.py": (
        "0f8caa48aa9e3a294d0677de3afc6568af4888205b2b9a32d8f7c5c536da8b1a"
    ),
    PROBE_MIRROR_RELATIVE: (
        "b46c7bde2f06cb4701a8ab85c18d07aa205d0d4958e58a37e7844c172758df55"
    ),
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
CAPTURE_COMPONENT_FILES = {
    "artifact_root": "artifact_root_capture_v1.py",
    "cuda_monolithic": "cuda_monolithic_v1.py",
    "fast_fresh_readiness": "fast_fresh_capture_v1.py",
    "joint_phone_cuda": "joint_phone_cuda_v1.py",
}
CAPTURE_COMPONENT_PATHS = {
    "artifact_root": (
        "v24_readiness/desktop_deployment_v1/artifact_root_capture_v1.py"
    ),
    "cuda_monolithic": "v24_readiness/producers_v1/cuda_monolithic_v1.py",
    "fast_fresh_readiness": (
        "v24_readiness/desktop_deployment_v1/fast_fresh_capture_v1.py"
    ),
    "joint_phone_cuda": "v24_readiness/producers_v1/joint_phone_cuda_v1.py",
}


class PlanSetError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanSetError(message)


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"E_IMPORT: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prod = _load_module("s39_v26_planset_prod", HERE / "materialize_production_v26.py")


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}",
    )


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return prod.canonical_bytes(value)


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def strip_build_id(metadata: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in metadata.items() if key != "build_id"}
    exact(
        set(result),
        {"ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"},
        "stat.keys",
    )
    return result


def load_v26_evidence(
    root: Path,
    topology_path: Path | None = None,
) -> dict[str, Any]:
    validation = prod.validate_production(
        root,
        topology_path=(
            topology_path if topology_path is not None else prod.TOPOLOGY_PATH
        ),
    )
    exact(
        validation["status"],
        "V2_6_PRODUCTION_MATERIALIZATION_PASS",
        "v26.validation",
    )
    inventory, inventory_raw = prod.authority.read_canonical(
        root / "RUNTIME_INVENTORY_V2_6.json"
    )
    lock, lock_raw = prod.authority.read_canonical(root / "PHASE_LOCK_V2_6.json")
    return {
        "body": inventory["inventory"],
        "lock": lock,
        "lock_sha256": sha256(lock_raw),
        "inventory_sha256": sha256(inventory_raw),
        "validation": validation,
    }


def record_component_pins(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pins: dict[str, dict[str, Any]] = {}
    for row in body["components"]:
        pins[row["component_id"]] = {
            "bundle_id": row["bundle_id"],
            "bytes": row["bytes"],
            "filename": Path(row["path"]).name,
            "path": row["path"],
            "role": row["role"],
            "sha256": row["sha256"],
            "stat": strip_build_id(row["stat"]),
        }
    return pins


def mono_components(mono_launch: dict[str, Any]) -> list[dict[str, Any]]:
    launcher = mono_launch["launcher_component_id"]
    rows = []
    for item in mono_launch["required_components"]:
        rows.append(
            {
                "bundle_id": "cuda_monolithic",
                "bytes": item["stat"]["size"],
                "component_id": item["component_id"],
                "endpoint": "cuda",
                "path": item["path"],
                "role": (
                    "executable"
                    if item["component_id"] == launcher
                    else "backend_library"
                    if "cuda" in item["component_id"]
                    else "shared_library"
                ),
                "sha256": item["sha256"],
                "stat": dict(item["stat"]),
            }
        )
    return rows


def build_closure_input(
    body: dict[str, Any],
    mono_launch: dict[str, Any],
) -> dict[str, Any]:
    pins = record_component_pins(body)
    components = mono_components(mono_launch)
    for component_id in sorted(pins):
        pin = pins[component_id]
        if pin["bundle_id"] == "cuda_monolithic":
            continue
        if pin["role"] == "capture_entrypoint":
            continue
        components.append(
            {
                "bundle_id": pin["bundle_id"],
                "bytes": pin["bytes"],
                "component_id": component_id,
                "endpoint": prod.BUNDLE_ENDPOINTS[pin["bundle_id"]],
                "path": pin["path"],
                "role": pin["role"],
                "sha256": pin["sha256"],
                "stat": pin["stat"],
            }
        )
    components.sort(key=lambda row: row["component_id"])
    bundle_required: dict[str, list[str]] = {}
    for row in components:
        bundle_required.setdefault(row["bundle_id"], []).append(
            row["component_id"]
        )
    bundles = []
    roots = {
        "cuda_monolithic": mono_launch["bundle_root"],
        "cuda_route": prod.DESKTOP_ROOT,
        "op12_stagenet": prod.OP12_STAGE_ROOT,
        "op15_direct_relay": prod.OP15_RELAY_ROOT,
        "op15_stagenet": prod.OP15_STAGE_ROOT,
    }
    launchers = {
        "cuda_monolithic": mono_launch["launcher_component_id"],
        "cuda_route": "cuda-route.bin",
        "op12_stagenet": "op12_stagenet.bin",
        "op15_direct_relay": "op15_direct_relay.bin",
        "op15_stagenet": "op15_stagenet.bin",
    }
    for bundle_id in sorted(roots):
        required = sorted(bundle_required[bundle_id])
        require(launchers[bundle_id] in required, f"E_LAUNCHER: {bundle_id}")
        bundles.append(
            {
                "bundle_id": bundle_id,
                "endpoint": prod.BUNDLE_ENDPOINTS[bundle_id],
                "launcher_component_id": launchers[bundle_id],
                "process_role": bundle_id,
                "required_component_ids": required,
            }
        )
    return {
        "bundle_roots": roots,
        "bundles": bundles,
        "closure_complete": True,
        "components": components,
        "schema": "s39-cp0-r1-runtime-bundle-closure-input-v1",
    }


def _bundle(closure: dict[str, Any], bundle_id: str) -> dict[str, Any]:
    return next(
        row for row in closure["bundles"] if row["bundle_id"] == bundle_id
    )


def _component(closure: dict[str, Any], component_id: str) -> dict[str, Any]:
    return next(
        row
        for row in closure["components"]
        if row["component_id"] == component_id
    )


def _artifact_pin(closure_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "bytes": closure_row["bytes"],
        "path": closure_row["path"],
        "sha256": closure_row["sha256"],
        "stat": dict(closure_row["stat"]),
    }


def _inline(usb_launcher_path: str, plan: dict[str, Any]) -> list[str]:
    raw = compact_json(plan)
    return [
        usb_launcher_path,
        "--plan-json",
        raw,
        "--plan-sha256",
        sha256(raw.encode("ascii")),
    ]


def build_phone_static(
    closure: dict[str, Any],
    contract_v24: dict[str, Any],
    desktop_pins: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    usb_launcher = desktop_pins["usb_launcher"]
    probe = desktop_pins["probe"]
    adb = desktop_pins["adb"]
    codec_row = desktop_pins["codec"]
    model_path = contract_v24["model_geometry"][MODEL_ID]["cuda_model_path"]
    model_sha = MODEL_SHA256
    shards = contract_v24["model_geometry"][MODEL_ID]["known_shards"]
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
            "model_path": shards["op12"]["path"],
            "model_sha256": model_sha,
            "n_gpu_layers": 999,
            "placement_cert": True,
            "port": PORTS["op12_stage"],
            "runtime_root": prod.OP12_STAGE_ROOT,
        },
        "op15_direct_relay": {
            "emit_direct_frames": True,
            "head_host": prod.UNBOUND_RELAY_HEAD_HOST,
            "head_port": PORTS["op15_stage"],
            "kind": "direct_relay",
            "listen_port": PORTS["relay"],
            "runtime_root": prod.OP15_RELAY_ROOT,
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
            "model_sha256": model_sha,
            "n_gpu_layers": 999,
            "placement_cert": True,
            "port": PORTS["op15_stage"],
            "runtime_root": prod.OP15_STAGE_ROOT,
        },
    }
    endpoints = {
        "op12_stagenet": "op12",
        "op15_direct_relay": "op15",
        "op15_stagenet": "op15",
    }
    process_environment = {
        "ADB_SERVER_PORT": str(PORTS["adb_server"]),
        "ANDROID_ADB_SERVER_PORT": str(PORTS["adb_server"]),
        "HOME": "/home/zhihao",
        "PATH": "/usr/bin:/bin",
    }
    processes = {}
    for name in sorted(routes):
        endpoint = endpoints[name]
        serial = contract_v24["devices"][endpoint]["serial"]
        bundle = _bundle(closure, name)
        components = [
            {
                key: _component(closure, component_id)[key]
                for key in ("bytes", "component_id", "path", "sha256", "stat")
            }
            for component_id in bundle["required_component_ids"]
        ]
        launcher = _component(closure, bundle["launcher_component_id"])
        plan = {
            "android": {
                "adb_path": adb["path"],
                "adb_port": PORTS["adb_server"],
                "adb_selector": serial,
                "adb_sha256": adb["sha256"],
                "boot_id_source": "phase_fresh_snapshot",
                "physical_serial": serial,
                "shutdown_timeout_ms": 30000,
                "startup_timeout_ms": 120000,
            },
            "bundle_id": name,
            "components": components,
            "endpoint": endpoint,
            "launcher_component_id": bundle["launcher_component_id"],
            "mode": "android",
            "route": routes[name],
            "schema": "s39-managed-runtime-launch-plan-v1",
        }
        processes[name] = {
            "argv": _inline(usb_launcher["path"], plan),
            "cwd": DESKTOP_WORK_DIR,
            "environment": dict(process_environment),
            "launcher_bytes": usb_launcher["bytes"],
            "launcher_sha256": usb_launcher["sha256"],
            "runtime_component_ids": bundle["required_component_ids"],
            "runtime_executable_path": launcher["path"],
            "runtime_executable_sha256": launcher["sha256"],
            "shutdown_timeout_ms": 30000,
            "startup_timeout_ms": 120000,
        }
    probes = {}
    for endpoint in ("op12", "op15"):
        stage_argv = processes[f"{endpoint}_stagenet"]["argv"]
        peer = "op15" if endpoint == "op12" else "op12"
        wiring = PROBE_WIRING[endpoint]
        common_argv = [
            probe["path"],
            "--plan-json",
            stage_argv[2],
            "--plan-sha256",
            stage_argv[4],
            "--interface",
            UNBOUND_INTERFACE,
            "--local-ipv4",
            UNBOUND_NETWORK[endpoint],
            "--peer-ipv4",
            UNBOUND_NETWORK[peer],
            "--local-port",
            str(wiring["local_port"]),
            "--peer-port",
            str(wiring["peer_port"]),
            "--network-role",
            wiring["role"],
        ]
        probes[endpoint] = {
            "after_argv": common_argv[:5] + ["--when", "after"] + common_argv[5:],
            "before_argv": common_argv[:5] + ["--when", "before"] + common_argv[5:],
            "cwd": DESKTOP_WORK_DIR,
            "environment": dict(process_environment),
            "launcher_bytes": probe["bytes"],
            "launcher_sha256": probe["sha256"],
            "timeout_ms": 60000,
        }
    codec = {
        "argv": [
            codec_row["path"],
            "--model",
            model_path,
            "--model-sha256",
            model_sha,
        ],
        "cwd": prod.CUDA_ROUTE_ROOT,
        "environment": {
            "LC_ALL": "C",
            "LD_LIBRARY_PATH": prod.CUDA_ROUTE_ROOT,
        },
        "executable_bytes": codec_row["bytes"],
        "executable_sha256": codec_row["sha256"],
        "timeout_ms": 600000,
    }
    worker_argv = _cuda_worker_argv(closure, model_path)
    nvidia = _nvidia_block(contract_v24, desktop_pins["nvidia_smi"])
    mechanism = {
        "desktop": [
            list(codec["argv"]),
            list(worker_argv),
            list(nvidia["device_argv"]),
            list(nvidia["process_argv"]),
            list(nvidia["device_argv"]),
            list(nvidia["process_argv"]),
            list(nvidia["device_argv"]),
            list(nvidia["process_argv"]),
            list(desktop_pins["mono_command"]),
        ],
        "op12": [
            processes["op12_stagenet"]["argv"],
            probes["op12"]["before_argv"],
            probes["op12"]["after_argv"],
        ],
        "op15": [
            processes["op15_stagenet"]["argv"],
            processes["op15_direct_relay"]["argv"],
            probes["op15"]["before_argv"],
            probes["op15"]["after_argv"],
        ],
    }
    phones = {}
    for endpoint in ("op12", "op15"):
        expected = contract_v24["devices"][endpoint]
        stage_bundle = _bundle(closure, f"{endpoint}_stagenet")
        stage_launcher = _component(
            closure, stage_bundle["launcher_component_id"]
        )
        route_lock = contract_v24["incumbent_route_lock"]
        phones[endpoint] = {
            "device": expected["device"],
            "executed_layers": [30, 40] if endpoint == "op12" else [0, 30],
            "expected_worker_executable_path": stage_launcher["path"],
            "expected_worker_executable_sha256": stage_launcher["sha256"],
            "loaded_shard_path": shards[endpoint]["path"],
            "loaded_shard_sha256": route_lock[f"{endpoint}_shard_sha256"],
            "model": expected["model"],
            "product": expected["product"],
            "serial": expected["serial"],
            "stored_layers": route_lock[f"{endpoint}_stored_layers"],
        }
    return {
        "codec": codec,
        "expected_file_type": 15,
        "expected_max_streams": 8,
        "expected_n_batch": 64,
        "expected_n_ctx_seq": 512,
        "expected_n_embd": 5120,
        "expected_n_layer": 40,
        "expected_n_ubatch": 64,
        "mechanism_commands": mechanism,
        "phones": phones,
        "probes": probes,
        "processes": processes,
        "relay_host": "127.0.0.1",
        "relay_port": PORTS["relay"],
        "route_epoch": ROUTE_EPOCH,
    }


def _cuda_worker_argv(closure: dict[str, Any], model_path: str) -> list[str]:
    runtime = _component(closure, "cuda-route.bin")
    return [
        runtime["path"],
        "--model",
        model_path,
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


def _nvidia_block(
    contract_v24: dict[str, Any],
    nvidia_pin: dict[str, Any],
) -> dict[str, Any]:
    uuid = contract_v24["devices"]["cuda"]["uuid"]
    return {
        "device_argv": [
            nvidia_pin["path"],
            f"--id={uuid}",
            "--query-gpu=name,uuid,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        "executable": dict(nvidia_pin),
        "process_argv": [
            nvidia_pin["path"],
            f"--id={uuid}",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        "timeout_ms": 60000,
    }


def build_cuda_static(
    closure: dict[str, Any],
    contract_v24: dict[str, Any],
    desktop_pins: dict[str, dict[str, Any]],
    phone_static: dict[str, Any],
) -> dict[str, Any]:
    model_path = contract_v24["model_geometry"][MODEL_ID]["cuda_model_path"]
    runtime_pin = dict(desktop_pins["cuda_runtime"])
    codec_pin = dict(desktop_pins["codec"])
    bundle = _bundle(closure, "cuda_route")
    return {
        "codec": {
            "argv": [
                codec_pin["path"],
                "--model",
                model_path,
                "--model-sha256",
                MODEL_SHA256,
            ],
            "cwd": prod.CUDA_ROUTE_ROOT,
            "environment": {
                "LC_ALL": "C",
                "LD_LIBRARY_PATH": prod.CUDA_ROUTE_ROOT,
            },
            "executable": codec_pin,
            "timeout_ms": 600000,
        },
        "expected_capabilities": 63,
        "expected_file_type": 15,
        "expected_max_streams": 8,
        "expected_n_batch": 64,
        "expected_n_ctx_seq": 512,
        "expected_n_embd": 5120,
        "expected_n_layer": 40,
        "expected_n_ubatch": 64,
        "host": "127.0.0.1",
        "io_timeout_ms": 60000,
        "mechanism_commands": phone_static["mechanism_commands"],
        "model_artifact": dict(desktop_pins["model"]),
        "nvidia_smi": _nvidia_block(contract_v24, desktop_pins["nvidia_smi"]),
        "port": PORTS["cuda_route"],
        "route_epoch": ROUTE_EPOCH,
        "worker": {
            "argv": _cuda_worker_argv(closure, model_path),
            "cwd": prod.CUDA_ROUTE_ROOT,
            "environment": {
                "CUDA_VISIBLE_DEVICES": contract_v24["devices"]["cuda"]["uuid"],
                "LAYERSPLIT_MEMORY_CERT": "1",
                "LAYERSPLIT_MODEL_SHA256": MODEL_SHA256,
                "LAYERSPLIT_PLACEMENT_CERT": "1",
                "LD_LIBRARY_PATH": prod.CUDA_ROUTE_ROOT,
            },
            "executable": dict(runtime_pin),
            "runtime_component_ids": bundle["required_component_ids"],
            "runtime_executable": dict(runtime_pin),
            "shutdown_timeout_ms": 30000,
            "startup_timeout_ms": 120000,
        },
    }


def build_runtime_static(
    closure: dict[str, Any],
    body: dict[str, Any],
    contract_v24: dict[str, Any],
) -> dict[str, Any]:
    pins = record_component_pins(body)
    components = [
        {key: value for key, value in row.items() if key != "stat"}
        for row in closure["components"]
    ]
    producers = contract_v24["producer_requirements"]["source_programs"]
    launchers = {
        row["bundle_id"]: row["launcher_component_id"]
        for row in closure["bundles"]
    }
    captures = []
    for kind in sorted(CAPTURE_COMPONENT_FILES):
        filename = CAPTURE_COMPONENT_FILES[kind]
        pin = next(
            row
            for row in pins.values()
            if row["filename"] == filename
            and row["role"] == "capture_entrypoint"
        )
        component_id = f"capture.{kind}"
        components.append(
            {
                "bundle_id": "cuda_route",
                "bytes": pin["bytes"],
                "component_id": component_id,
                "endpoint": "cuda",
                "path": f"{MIRROR_ROOT}/{CAPTURE_COMPONENT_PATHS[kind]}",
                "role": "executable",
                "sha256": pin["sha256"],
            }
        )
        if kind in {"cuda_monolithic", "joint_phone_cuda"}:
            exact(pin["sha256"], producers[kind]["sha256"], f"capture.{kind}")
            exact(pin["bytes"], producers[kind]["bytes"], f"capture.{kind}.bytes")
        nested: list[str] = []
        if kind == "cuda_monolithic":
            nested = [launchers["cuda_monolithic"]]
        elif kind == "joint_phone_cuda":
            nested = sorted(
                launchers[name]
                for name in (
                    "cuda_route",
                    "op12_stagenet",
                    "op15_direct_relay",
                    "op15_stagenet",
                )
            )
        captures.append(
            {
                "component_id": component_id,
                "execution_mode": "SELF_CONTAINED_PHYSICAL_CAPTURE",
                "kind": kind,
                "nested_capture_entrypoint_component_ids": nested,
            }
        )
    return {
        "bundle_roots": dict(closure["bundle_roots"]),
        "bundles": [
            {
                **row,
                "process_role": {
                    "cuda_monolithic": "cuda_monolithic",
                    "cuda_route": "cuda_route",
                    "op12_stagenet": "stagenet_worker",
                    "op15_direct_relay": "direct_relay",
                    "op15_stagenet": "stagenet_worker",
                }[row["bundle_id"]],
            }
            for row in closure["bundles"]
        ],
        "capture_entrypoints": sorted(captures, key=lambda row: row["kind"]),
        "components": sorted(components, key=lambda row: row["component_id"]),
        "tokenizer_component_id": "cuda-tokenize",
    }


def build_operator_input(
    input_pins: dict[str, dict[str, Any]],
    desktop_pins: dict[str, dict[str, Any]],
    contract_v24: dict[str, Any],
) -> dict[str, Any]:
    files = {
        name: dict(input_pins[name])
        for name in (
            "candidate",
            "contract",
            "cuda_monolithic_launch",
            "runtime_bundle_inventory",
            "token_history",
            "tokenizer_plan",
            "topology_receipt",
        )
    }
    files.update(
        {
            "adb": dict(desktop_pins["adb"]),
            "codec": dict(desktop_pins["codec"]),
            "cuda_launcher": dict(desktop_pins["cuda_runtime"]),
            "cuda_runtime": dict(desktop_pins["cuda_runtime"]),
            "model": dict(desktop_pins["model"]),
            "monolithic_launcher": dict(desktop_pins["mono_launcher"]),
            "monolithic_runtime": dict(desktop_pins["mono_launcher"]),
            "nvidia_smi": dict(desktop_pins["nvidia_smi"]),
            "python": dict(desktop_pins["python"]),
            "quality_corpus": dict(desktop_pins["quality_corpus"]),
            "ssh": dict(desktop_pins["ssh"]),
        }
    )
    return {
        "controller_host": contract_v24["devices"]["cuda"]["host"],
        "directories": {
            "cuda_bundle_root": prod.CUDA_ROUTE_ROOT,
            "joint_cwd": DESKTOP_WORK_DIR,
        },
        "files": files,
        "model_id": MODEL_ID,
        "phase": PHASE,
        "ports": dict(PORTS),
        "route_epoch": ROUTE_EPOCH,
        "schema": "s39-cp0-r1-v24-desktop-operator-input-v1",
        "topology": {
            "adb_host": "127.0.0.1",
            "adb_port": PORTS["adb_server"],
            "cuda_uuid": contract_v24["devices"]["cuda"]["uuid"],
            "physical_serials": {
                "op12": contract_v24["devices"]["op12"]["serial"],
                "op15": contract_v24["devices"]["op15"]["serial"],
            },
        },
    }


def build_desktop_inventory(
    input_pins: dict[str, dict[str, Any]],
    operator: dict[str, Any],
    statics: dict[str, Any],
    contract_v24: dict[str, Any],
) -> dict[str, Any]:
    return {
        "captured_on": {
            "controller_host": contract_v24["devices"]["cuda"]["host"],
            "cuda_uuid": contract_v24["devices"]["cuda"]["uuid"],
            "phone_adb_port": PORTS["adb_server"],
            "physical_usb_selectors": {
                "op12": contract_v24["devices"]["op12"]["serial"],
                "op15": contract_v24["devices"]["op15"]["serial"],
            },
        },
        "inputs": {
            name: {
                "bytes": pin["bytes"],
                "path": pin["path"],
                "sha256": pin["sha256"],
            }
            for name, pin in sorted(input_pins.items())
        },
        "model_id": MODEL_ID,
        "phase": PHASE,
        "route_epoch": ROUTE_EPOCH,
        "schema": "s39-cp0-r1-v24-a-only-desktop-inventory-v1",
        "static": {
            "cuda_route": statics["cuda_route"],
            "joint_cwd": DESKTOP_WORK_DIR,
            "phone_route": statics["phone_route"],
            "runtime": statics["runtime"],
        },
    }


REMOTE_STAT_SOURCE = (
    "import json,os,sys\n"
    "rows={}\n"
    "for path in sys.argv[1:]:\n"
    "    value=os.stat(path,follow_symlinks=False)\n"
    "    rows[path]={'ctime_ns':value.st_ctime_ns,'device_id':value.st_dev,"
    "'inode':value.st_ino,'mode':value.st_mode,'mtime_ns':value.st_mtime_ns,"
    "'size':value.st_size}\n"
    "print(json.dumps(rows,sort_keys=True,separators=(',',':')))\n"
)


def desktop_ns_stats(
    capture: Any,
    paths: list[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    """Exact-nanosecond remote stats via python os.stat on the desktop."""

    argv = prod.ssh_argv(
        " ".join(
            (
                DESKTOP_PYTHON,
                "-c",
                shlex.quote(REMOTE_STAT_SOURCE),
                *(shlex.quote(path) for path in sorted(paths)),
            )
        )
    )
    raw = capture.run(f"stat.{label}", argv, 60)
    try:
        rows = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlanSetError(f"E_REMOTE_STAT: {label}") from error
    require(
        type(rows) is dict and set(rows) == set(paths),
        f"E_REMOTE_STAT_PATHS: {label}",
    )
    for path, row in rows.items():
        exact(
            set(row),
            {"ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"},
            f"stat.{label}.{path}",
        )
    return rows


def desktop_stat_pin(
    capture: Any,
    path: str,
    label: str,
    *,
    executable: bool,
) -> dict[str, Any]:
    before = desktop_ns_stats(capture, [path], label)[path]
    digest_raw = capture.run(
        f"pin.{label}.sha256",
        prod.ssh_argv("sha256sum -- " + shlex.quote(path)),
        600,
    )
    fields = digest_raw.decode("ascii").strip().split()
    require(len(fields) == 2 and len(fields[0]) == 64, f"E_PIN_DIGEST: {label}")
    after = desktop_ns_stats(capture, [path], f"{label}.after")[path]
    exact(after, before, f"pin.{label}.stat_mutation")
    if executable:
        require(before["mode"] & 0o111 != 0, f"E_PIN_EXECUTABLE: {label}")
    return {
        "bytes": before["size"],
        "path": path,
        "sha256": fields[0],
        "stat": dict(before),
    }


def verify_mono_live(
    capture: Any,
    mono_launch: dict[str, Any],
) -> None:
    paths = [item["path"] for item in mono_launch["required_components"]]
    live = desktop_ns_stats(capture, paths, "mono")
    for item in mono_launch["required_components"]:
        exact(
            live[item["path"]],
            dict(item["stat"]),
            f"mono.live_stat.{item['component_id']}",
        )


def sync_mirror(capture: Any) -> dict[str, str]:
    digests: dict[str, str] = {}
    for relative in sorted(set(MIRROR_FILES) | set(EXTRA_MIRROR_SOURCES)):
        source = EXTRA_MIRROR_SOURCES.get(relative, S39 / relative)
        raw = prod.read_file_once(source, f"mirror.{relative}")
        digest = sha256(raw)
        pinned = MIRROR_PINS.get(relative)
        if pinned is not None:
            exact(digest, pinned, f"mirror.pin.{relative}")
        digests[relative] = digest
        remote = f"{MIRROR_ROOT}/{relative}"
        quoted = shlex.quote(remote)
        presence = capture.run(
            f"mirror.{relative}.presence",
            prod.ssh_argv(
                f"if test -e {quoted}; then sha256sum -- {quoted};"
                f" else echo ABSENT; fi"
            ),
            120,
        ).decode("ascii").strip()
        if presence != "ABSENT":
            fields = presence.split()
            require(
                len(fields) == 2 and fields[0] == digest,
                f"E_MIRROR_CONFLICT: {relative}",
            )
        else:
            parent = shlex.quote(str(Path(remote).parent))
            capture.run(
                f"mirror.{relative}.mkdir",
                prod.ssh_argv(f"mkdir -p -- {parent}"),
                30,
            )
            capture.run(
                f"mirror.{relative}.copy",
                [
                    "scp",
                    "-o",
                    "BatchMode=yes",
                    str(source),
                    f"{prod.SSH_TARGET}:{remote}",
                ],
                300,
            )
            verify = capture.run(
                f"mirror.{relative}.verify",
                prod.ssh_argv(f"sha256sum -- {quoted}"),
                120,
            ).decode("ascii").strip().split()
            require(
                len(verify) == 2 and verify[0] == digest,
                f"E_MIRROR_VERIFY: {relative}",
            )
        if relative in MIRROR_EXECUTABLES:
            capture.run(
                f"mirror.{relative}.chmod",
                prod.ssh_argv(f"chmod 0755 -- {quoted}"),
                30,
            )
    return digests


def push_input(
    capture: Any,
    local_path: Path,
    remote_path: str,
    label: str,
) -> None:
    raw = prod.read_file_once(local_path, f"input.{label}")
    digest = sha256(raw)
    quoted = shlex.quote(remote_path)
    capture.run(
        f"input.{label}.mkdir",
        prod.ssh_argv(
            "mkdir -p -- " + shlex.quote(str(Path(remote_path).parent))
        ),
        30,
    )
    presence = capture.run(
        f"input.{label}.presence",
        prod.ssh_argv(
            f"if test -e {quoted}; then echo PRESENT; else echo ABSENT; fi"
        ),
        30,
    ).decode("ascii").strip()
    require(presence == "ABSENT", f"E_INPUT_EXISTS: {label}")
    capture.run(
        f"input.{label}.copy",
        [
            "scp",
            "-o",
            "BatchMode=yes",
            str(local_path),
            f"{prod.SSH_TARGET}:{remote_path}",
        ],
        120,
    )
    verify = capture.run(
        f"input.{label}.verify",
        prod.ssh_argv(f"sha256sum -- {quoted}"),
        120,
    ).decode("ascii").strip().split()
    require(len(verify) == 2 and verify[0] == digest, f"E_INPUT_VERIFY: {label}")


def fetch_remote(
    capture: Any,
    remote_path: str,
    local_path: Path,
    label: str,
) -> bytes:
    require(not local_path.exists(), f"E_FETCH_EXISTS: {label}")
    capture.run(
        f"fetch.{label}.copy",
        [
            "scp",
            "-o",
            "BatchMode=yes",
            f"{prod.SSH_TARGET}:{remote_path}",
            str(local_path),
        ],
        120,
    )
    raw = prod.read_file_once(local_path, f"fetch.{label}")
    remote_digest = capture.run(
        f"fetch.{label}.verify",
        prod.ssh_argv(f"sha256sum -- {shlex.quote(remote_path)}"),
        120,
    ).decode("ascii").strip().split()
    require(
        len(remote_digest) == 2 and remote_digest[0] == sha256(raw),
        f"E_FETCH_VERIFY: {label}",
    )
    return raw


def load_v24_authority() -> types.ModuleType:
    return _load_module("s39_v24_authority_for_plan_set", V24 / "cp0_r1_evidence_v24.py")


def validate_plan_set(
    prephase: Path = PREPHASE,
    *,
    contract_v24_path: Path = V24 / "CP0_R1_EVIDENCE_CONTRACT_V2_4.json",
    candidate_path: Path = S39 / "CP0_R1_CANDIDATE.json",
) -> dict[str, Any]:
    contract_v24, contract_raw = prod.authority.read_canonical(contract_v24_path)
    candidate_raw = prod.read_file_once(candidate_path, "candidate")
    plan, plan_raw = prod.authority.read_canonical(
        prephase / "runtime-bundle-plan.json"
    )
    authority = load_v24_authority()
    try:
        authority.validate_runtime_plan(
            json.loads(canonical_bytes(plan)),
            contract_v24,
            contract_raw,
            candidate_raw,
        )
    except Exception as error:
        raise PlanSetError(f"E_RUNTIME_PLAN: {error}") from error
    cuda_launch, cuda_raw = prod.authority.read_canonical(
        prephase / "cuda-route-launch.json"
    )
    phone_launch, phone_raw = prod.authority.read_canonical(
        prephase / "phone-route-launch.json"
    )
    exact(
        cuda_launch.get("schema"),
        "s39-cp0-r1-v24-cuda-route-launch-v1",
        "cuda_launch.schema",
    )
    exact(
        phone_launch.get("schema"),
        "s39-cp0-r1-v24-phone-route-launch-v1",
        "phone_launch.schema",
    )
    exact(
        cuda_launch.get("mechanism_commands"),
        phone_launch.get("mechanism_commands"),
        "mechanism.matrix",
    )
    joint, joint_raw = prod.authority.read_canonical(
        prephase / "joint-capture-plan.json"
    )
    root_value, root_raw = prod.authority.read_canonical(
        prephase / "prospective-runtime-root.json"
    )
    exact(
        root_value.get("schema"),
        "s39-cp0-r1-v24-prospective-runtime-root-v1",
        "root.schema",
    )
    exact(
        root_value.get("status"),
        "POST_REBOOT_IDENTITY_BINDING_REQUIRED",
        "root.status",
    )
    exact(root_value.get("acquisition_ready"), False, "root.acquisition_ready")
    exact(
        joint.get("schema"),
        "s39-cp0-r1-v24-joint-capture-plan-v1",
        "joint.schema",
    )
    for name, raw in (
        ("cuda-route-launch.json", cuda_raw),
        ("joint-capture-plan.json", joint_raw),
        ("phone-route-launch.json", phone_raw),
        ("runtime-bundle-plan.json", plan_raw),
        ("prospective-runtime-root.json", root_raw),
    ):
        require(b"v23_readiness" not in raw, f"E_V23_REFERENCE: {name}")
    return {
        "artifacts": {
            name: sha256(prod.read_file_once(prephase / name, name))
            for name in PLAN_NAMES
        },
        "schema": VALIDATION_SCHEMA,
        "status": "V24_PLAN_SET_VALIDATION_PASS",
    }


def materialize(
    *,
    confirmation: str,
    v26_root: Path = DEFAULT_V26_ROOT,
    output_root: Path | None = None,
    prephase: Path = PREPHASE,
    topology_path: Path | None = None,
    runner: Any = None,
    clock_ns: Any = prod.monotonic_ns,
) -> dict[str, Any]:
    exact(confirmation, CONFIRMATION, "confirmation")
    for name in PLAN_NAMES:
        require(
            not (prephase / name).exists(),
            f"E_PLAN_EXISTS: {name}",
        )
    contract_v26, contract_v26_raw = prod.load_contract()
    contract_v24 = prod.load_v24_contract(contract_v26)
    topology, topology_sha256 = prod.load_topology(
        contract_v24,
        topology_path if topology_path is not None else prod.TOPOLOGY_PATH,
    )
    mono_launch, mono_launch_sha256 = prod.load_mono_launch()
    evidence = load_v26_evidence(v26_root, topology_path)
    body = evidence["body"]

    runner = runner or prod.inv.SubprocessRunner()
    capture = prod.Capture(runner, clock_ns)
    started_ns = clock_ns()
    live = prod.live_check(v26_root, runner=runner, clock_ns=clock_ns)
    exact(
        live["status"],
        "V2_6_PRODUCTION_BOOT_IDENTITY_LIVE",
        "v26.live_check",
    )

    verify_mono_live(capture, mono_launch)
    producers = contract_v24["producer_requirements"]["source_programs"]
    prod.ensure_desktop_launcher(
        capture,
        V24 / "desktop_deployment_v1" / "artifact_root_capture_v1.py",
        contract_v26["composition"]["capture_producers"]["artifact_root"][
            "sha256"
        ],
        f"{prod.CUDA_ROUTE_ROOT}/artifact_root_capture_v1.py",
        "route-artifact-root",
    )
    prod.ensure_desktop_launcher(
        capture,
        V24 / "producers_v1" / "cuda_monolithic_v1.py",
        producers["cuda_monolithic"]["sha256"],
        f"{prod.CUDA_ROUTE_ROOT}/cuda_monolithic_v1.py",
        "route-cuda-monolithic",
    )
    model_pin = desktop_stat_pin(
        capture,
        contract_v24["model_geometry"][MODEL_ID]["cuda_model_path"],
        "model",
        executable=False,
    )
    exact(
        {key: model_pin[key] for key in ("bytes", "sha256")},
        {
            "bytes": evidence["lock"]["model_artifacts"]["cuda"]["bytes"],
            "sha256": evidence["lock"]["model_artifacts"]["cuda"]["sha256"],
        },
        "model.lock_binding",
    )
    mirror_digests = sync_mirror(capture)

    pins = record_component_pins(body)
    codec_pin = desktop_stat_pin(
        capture, pins["cuda-tokenize"]["path"], "codec", executable=True
    )
    runtime_pin = desktop_stat_pin(
        capture, pins["cuda-route.bin"]["path"], "cuda_runtime", executable=True
    )
    for label, fresh, recorded in (
        ("codec", codec_pin, pins["cuda-tokenize"]),
        ("cuda_runtime", runtime_pin, pins["cuda-route.bin"]),
    ):
        exact(fresh["bytes"], recorded["bytes"], f"pin.{label}.bytes")
        exact(fresh["sha256"], recorded["sha256"], f"pin.{label}.sha256")
    desktop_pins = {
        "adb": desktop_stat_pin(
            capture, prod.DESKTOP_ADB_PATH, "adb", executable=True
        ),
        "codec": codec_pin,
        "cuda_runtime": runtime_pin,
        "model": model_pin,
        "mono_command": list(mono_launch["command"]),
        "mono_launcher": next(
            {
                "bytes": item["stat"]["size"],
                "path": item["path"],
                "sha256": item["sha256"],
                "stat": dict(item["stat"]),
            }
            for item in mono_launch["required_components"]
            if item["component_id"] == mono_launch["launcher_component_id"]
        ),
        "nvidia_smi": desktop_stat_pin(
            capture, DESKTOP_NVIDIA_SMI, "nvidia_smi", executable=True
        ),
        "probe": {
            "bytes": len(
                prod.read_file_once(
                    EXTRA_MIRROR_SOURCES[SNAPSHOT_PROBE_RELATIVE],
                    "snapshot_probe",
                )
            ),
            "path": f"{MIRROR_ROOT}/{SNAPSHOT_PROBE_RELATIVE}",
            "sha256": sha256(
                prod.read_file_once(
                    EXTRA_MIRROR_SOURCES[SNAPSHOT_PROBE_RELATIVE],
                    "snapshot_probe",
                )
            ),
        },
        "python": desktop_stat_pin(
            capture, DESKTOP_PYTHON, "python", executable=True
        ),
        "quality_corpus": {
            "bytes": contract_v26["quality"]["corpus"]["bytes"],
            "path": f"{MIRROR_ROOT}/CP0_R1_MMLU64_CORPUS_V2_2.jsonl",
            "sha256": contract_v26["quality"]["corpus"]["sha256"],
        },
        "ssh": desktop_stat_pin(capture, DESKTOP_SSH, "ssh", executable=True),
        "usb_launcher": {
            "bytes": len(
                prod.read_file_once(
                    EXTRA_MIRROR_SOURCES[SNAPSHOT_LAUNCHER_RELATIVE],
                    "snapshot_launcher",
                )
            ),
            "path": f"{MIRROR_ROOT}/{SNAPSHOT_LAUNCHER_RELATIVE}",
            "sha256": sha256(
                prod.read_file_once(
                    EXTRA_MIRROR_SOURCES[SNAPSHOT_LAUNCHER_RELATIVE],
                    "snapshot_launcher",
                )
            ),
        },
    }

    closure = build_closure_input(body, mono_launch)
    phone_static = build_phone_static(closure, contract_v24, desktop_pins)
    cuda_static = build_cuda_static(
        closure, contract_v24, desktop_pins, phone_static
    )
    runtime_static = build_runtime_static(closure, body, contract_v24)

    if output_root is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        output_root = HERE / "results" / f"v24_plan_set_{stamp}"
    require(not output_root.exists(), f"E_OUTPUT_EXISTS: {output_root}")
    output_root.mkdir(mode=0o755, parents=False)

    closure_raw = canonical_bytes(closure)
    topology_raw = canonical_bytes(topology)
    input_pins = {
        "candidate": {
            "bytes": contract_v26["static_inputs"]["candidate"]["bytes"],
            "path": f"{MIRROR_ROOT}/CP0_R1_CANDIDATE.json",
            "sha256": contract_v26["static_inputs"]["candidate"]["sha256"],
        },
        "contract": {
            "bytes": len(
                prod.read_file_once(prod.V24_CONTRACT_PATH, "v24_contract")
            ),
            "path": (
                f"{MIRROR_ROOT}/v24_readiness/CP0_R1_EVIDENCE_CONTRACT_V2_4.json"
            ),
            "sha256": contract_v26["raw_predicate"]["v24_contract_sha256"],
        },
        "cuda_monolithic_launch": {
            "bytes": len(prod.read_file_once(prod.MONO_LAUNCH_PATH, "mono")),
            "path": (
                f"{MIRROR_ROOT}/v24_readiness/results/"
                "prephase_20260726T0915Z/cuda-monolithic-launch.json"
            ),
            "sha256": mono_launch_sha256,
        },
        "operator_input": {},
        "runtime_bundle_inventory": {
            "bytes": len(closure_raw),
            "path": f"{INPUTS_ROOT}/runtime-bundle-inventory.json",
            "sha256": sha256(closure_raw),
        },
        "token_history": {
            "bytes": contract_v26["static_inputs"]["token_history"]["bytes"],
            "path": (
                f"{MIRROR_ROOT}/v24_readiness/results/"
                "prephase_20260726T0915Z/token-history.json"
            ),
            "sha256": contract_v26["static_inputs"]["token_history"]["sha256"],
        },
        "tokenizer_plan": {
            "bytes": contract_v26["static_inputs"]["tokenizer_plan"]["bytes"],
            "path": (
                f"{MIRROR_ROOT}/v24_readiness/results/"
                "prephase_20260726T0915Z/tokenizer-plan.json"
            ),
            "sha256": contract_v26["static_inputs"]["tokenizer_plan"]["sha256"],
        },
        "topology_receipt": {
            "bytes": len(topology_raw),
            "path": f"{INPUTS_ROOT}/topology-receipt.json",
            "sha256": topology_sha256,
        },
    }
    operator = build_operator_input(input_pins, desktop_pins, contract_v24)
    operator_raw = canonical_bytes(operator)
    input_pins["operator_input"] = {
        "bytes": len(operator_raw),
        "path": f"{INPUTS_ROOT}/operator-input.json",
        "sha256": sha256(operator_raw),
    }
    inventory = build_desktop_inventory(
        input_pins,
        operator,
        {
            "cuda_route": cuda_static,
            "phone_route": phone_static,
            "runtime": runtime_static,
        },
        contract_v24,
    )
    inventory_raw = canonical_bytes(inventory)

    local_inputs = {
        "runtime-bundle-inventory.json": closure_raw,
        "topology-receipt.json": topology_raw,
        "operator-input.json": operator_raw,
        "desktop-inventory.json": inventory_raw,
    }
    for name, raw in local_inputs.items():
        prod.write_exclusive(output_root / name, raw)
        push_input(
            capture,
            output_root / name,
            f"{INPUTS_ROOT}/{name}",
            name,
        )

    mat_argv = [
        DESKTOP_PYTHON,
        "-B",
        f"{MIRROR_ROOT}/{DRIVER_RELATIVE}",
        "--inventory",
        f"{INPUTS_ROOT}/desktop-inventory.json",
        "--inventory-sha256",
        sha256(inventory_raw),
        "--spec-output",
        f"{INPUTS_ROOT}/prospective-runtime-spec.json",
        "--cuda-route-launch",
        f"{MIRROR_ROOT}/v24_readiness/results/prephase_20260726T0915Z/"
        "cuda-route-launch.json",
        "--joint-capture-plan",
        f"{MIRROR_ROOT}/v24_readiness/results/prephase_20260726T0915Z/"
        "joint-capture-plan.json",
        "--phone-route-launch",
        f"{MIRROR_ROOT}/v24_readiness/results/prephase_20260726T0915Z/"
        "phone-route-launch.json",
        "--runtime-plan",
        f"{MIRROR_ROOT}/v24_readiness/results/prephase_20260726T0915Z/"
        "runtime-bundle-plan.json",
        "--prospective-root",
        f"{MIRROR_ROOT}/v24_readiness/results/prephase_20260726T0915Z/"
        "prospective-runtime-root.json",
        "--dry-run-report",
        f"{INPUTS_ROOT}/dry-run-report.json",
    ]
    capture.run(
        "mat.materialize",
        ["ssh", *prod.SSH_OPTIONS, prod.SSH_TARGET, "--", *mat_argv],
        900,
    )

    remote_prephase = (
        f"{MIRROR_ROOT}/v24_readiness/results/prephase_20260726T0915Z"
    )
    fetched: dict[str, bytes] = {}
    for name in PLAN_NAMES:
        fetched[name] = fetch_remote(
            capture,
            f"{remote_prephase}/{name}",
            output_root / name,
            name,
        )
    for name, remote in (
        ("prospective-runtime-spec.json", f"{INPUTS_ROOT}/prospective-runtime-spec.json"),
        ("dry-run-report.json", f"{INPUTS_ROOT}/dry-run-report.json"),
    ):
        fetched[name] = fetch_remote(
            capture, remote, output_root / name, name
        )

    published: list[Path] = []
    try:
        for name in PLAN_NAMES:
            prod.write_exclusive(prephase / name, fetched[name])
            published.append(prephase / name)
    except BaseException:
        for path in published:
            path.unlink()
        raise

    completed_ns = clock_ns()
    record = {
        "captures": capture.rows,
        "completed_ns": completed_ns,
        "contract_v26_sha256": sha256(contract_v26_raw),
        "inputs": {
            name: sha256(raw) for name, raw in sorted(local_inputs.items())
        },
        "mirror_root": MIRROR_ROOT,
        "mirror_sources": mirror_digests,
        "outputs": {
            name: sha256(raw) for name, raw in sorted(fetched.items())
        },
        "phase_id": evidence["lock"]["phase_id"],
        "schema": RECORD_SCHEMA,
        "started_ns": started_ns,
        "status": "V24_PLAN_SET_MATERIALIZED",
        "v26_inventory_sha256": evidence["inventory_sha256"],
        "v26_lock_sha256": evidence["lock_sha256"],
        "v26_root": str(v26_root),
    }
    record_raw = canonical_bytes(record)
    prod.write_exclusive(
        output_root / "V24_PLAN_SET_MATERIALIZATION_V2_6.json", record_raw
    )
    manifest_lines = [
        f"{sha256(raw)}  {name}\n"
        for name, raw in sorted(
            {**local_inputs, **fetched}.items()
        )
    ]
    manifest_lines.append(
        f"{sha256(record_raw)}  V24_PLAN_SET_MATERIALIZATION_V2_6.json\n"
    )
    prod.write_exclusive(
        output_root / "SHA256SUMS.txt",
        "".join(manifest_lines).encode("ascii"),
    )

    validation = validate_plan_set(prephase)
    return {
        "output_root": str(output_root),
        "phase_id": record["phase_id"],
        "validation": validation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("materialize")
    run.add_argument("--v26-root", type=Path, default=DEFAULT_V26_ROOT)
    run.add_argument("--output-root", type=Path, default=None)
    run.add_argument("--topology", type=Path, default=None)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--confirm", default="")
    check = commands.add_parser("validate")
    check.add_argument("--prephase", type=Path, default=PREPHASE)
    args = parser.parse_args(argv)
    try:
        if args.command == "materialize":
            require(args.execute, "E_EXECUTE_REQUIRED")
            result = materialize(
                confirmation=args.confirm,
                v26_root=args.v26_root.resolve(),
                output_root=args.output_root,
                topology_path=(
                    args.topology.resolve()
                    if args.topology is not None
                    else prod.TOPOLOGY_PATH
                ),
            )
        else:
            result = validate_plan_set(args.prephase.resolve())
    except (PlanSetError, prod.ProductionError, prod.inv.InventoryError, OSError) as error:
        print(f"V24_PLAN_SET_REFUSED: {error}", file=sys.stderr)
        return 2
    sys.stdout.write(canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
