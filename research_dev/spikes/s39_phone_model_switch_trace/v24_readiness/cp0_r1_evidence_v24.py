#!/usr/bin/env python3
"""Validate the CP0-R1 V2.4 pre-reboot artifact and live readiness chain."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import sys
import types
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(S39) not in sys.path:
    sys.path.insert(0, str(S39))


def _execute_source(
    name: str,
    path: Path,
    dependencies: dict[str, types.ModuleType] | None = None,
) -> tuple[types.ModuleType, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"E_AUTHORITY_SOURCE_READ: {path}: {error}") from error
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    replacements = {name: module, **(dependencies or {})}
    previous = {
        key: sys.modules.get(key)
        for key in replacements
    }
    try:
        sys.modules.update(replacements)
        exec(compile(raw, str(path), "exec"), module.__dict__)
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module, raw


common, _COMMON_SOURCE = _execute_source(
    "v24_common",
    HERE / "v24_common.py",
)
builder, _BUILDER_SOURCE = _execute_source(
    "build_contract_v24",
    HERE / "build_contract_v24.py",
    {"v24_common": common},
)
sys.modules["v24_common"] = common
sys.modules["build_contract_v24"] = builder


sys.dont_write_bytecode = True

DEFAULT_CONTRACT = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2_4.json"
DEFAULT_CANDIDATE = S39 / "CP0_R1_CANDIDATE.json"
V22_CONTRACT = S39 / "CP0_R1_EVIDENCE_CONTRACT_V2_2.json"
RAW_MANIFEST_NAME = "EVIDENCE_BUNDLE_V2_4.json"
PHASE = "A_ONLY"
MODEL_ID = "qwen3-14b-q4_k_m"
ORCHESTRATION_PLAN_SCHEMA = "s39-cp0-r1-v24-a-only-orchestration-plan-v2"
ORCHESTRATION_RECEIPT_SCHEMA = "s39-cp0-r1-v24-stage-receipt-v1"
ORCHESTRATION_SOURCE_STAGES = (
    "preparation",
    "phase_lock",
    "identity_binding",
    "readiness_projection",
    "fan_in",
)
BOUND_ARTIFACT_SCHEMAS = {
    "cuda_route_launch": "s39-cp0-r1-v24-cuda-route-launch-v1",
    "joint_capture_plan": "s39-cp0-r1-v24-joint-capture-plan-v1",
    "phone_route_launch": "s39-cp0-r1-v24-phone-route-launch-v1",
    "runtime_plan": "s39-cp0-r1-runtime-bundle-plan-v2.4",
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
CUDA_SSH_TARGET = "zhihao@172.20.74.85"
PHONE_ADB_PORT = 5038
REQUIRED_BUNDLES = {
    "cuda_monolithic": ("cuda", "cuda_monolithic"),
    "cuda_route": ("cuda", "cuda_route"),
    "op12_stagenet": ("op12", "stagenet_worker"),
    "op15_direct_relay": ("op15", "direct_relay"),
    "op15_stagenet": ("op15", "stagenet_worker"),
}
CAPTURE_KINDS = {
    "artifact_root",
    "cuda_monolithic",
    "fast_fresh_readiness",
    "joint_phone_cuda",
}
PROCESS_EVIDENCE_ROLES = {
    "cuda_monolithic": "capture.cuda_monolithic",
    "cuda_route": "capture.joint_phone_cuda",
    "op12_stagenet": "capture.joint_phone_cuda",
    "op15_direct_relay": "capture.joint_phone_cuda",
    "op15_stagenet": "capture.joint_phone_cuda",
}
CAPTURE_RECEIPT_ROLES = {
    "capture.cuda_monolithic",
    "capture.joint_phone_cuda",
}
PRE_ACQUISITION_ROLES = {
    "phase.lock",
    "phase.preflight",
    "quality.corpus",
    f"model.{MODEL_ID}.route_lock",
}
JOINT_ROLE_FIELDS = {
    f"model.{MODEL_ID}.bridge": "bridge_rows",
    f"model.{MODEL_ID}.cuda_memory": "cuda_memory_rows",
    f"model.{MODEL_ID}.mechanics.phone": "mechanics_rows",
    f"model.{MODEL_ID}.oracle.cuda_route": "cuda_route_rows",
    f"model.{MODEL_ID}.placement.op12": "placement_op12_rows",
    f"model.{MODEL_ID}.placement.op15": "placement_op15_rows",
    f"model.{MODEL_ID}.quality.cuda": "quality_cuda_rows",
    f"model.{MODEL_ID}.quality.phone": "quality_phone_rows",
    f"model.{MODEL_ID}.route_transfer": "route_transfer_rows",
}


def _verified_helpers(
    contract: dict[str, Any],
) -> dict[str, types.ModuleType]:
    support = contract["exit_authority"]["support"]

    def load(
        key: str,
        module_name: str,
        dependencies: dict[str, types.ModuleType] | None = None,
    ) -> types.ModuleType:
        record = support[key]
        path = S39 / record["path"]
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise common.EvidenceError(
                f"E_AUTHORITY_SOURCE_READ: {key}: {error}"
            ) from error
        common.exact(len(raw), record["bytes"], f"authority.{key}.bytes")
        common.exact(
            hashlib.sha256(raw).hexdigest(),
            record["sha256"],
            f"authority.{key}.sha256",
        )
        module, executed_raw = _execute_source(
            module_name,
            path,
            dependencies,
        )
        common.exact(executed_raw, raw, f"E_AUTHORITY_SOURCE_TOCTOU: {key}")
        return module

    build_v21 = load("builder_v21", "build_cp0_r1_v21")
    build_v22 = load("builder_v22", "build_cp0_r1_v22")
    mmlu_v22 = load("mmlu_builder_v22", "build_cp0_r1_mmlu64_v22")
    v2 = load("evaluator_v2", "cp0_r1_evidence_v2")
    v21 = load(
        "evaluator_v21",
        "cp0_r1_evidence_v21",
        {
            "build_cp0_r1_v21": build_v21,
            "cp0_r1_evidence_v2": v2,
        },
    )
    v22 = load(
        "evaluator_v22",
        "cp0_r1_evidence_v22",
        {
            "build_cp0_r1_mmlu64_v22": mmlu_v22,
            "build_cp0_r1_v22": build_v22,
            "cp0_r1_evidence_v2": v2,
            "cp0_r1_evidence_v21": v21,
        },
    )
    return {"v2": v2, "v21": v21, "v22": v22}


def _relative_path(value: Any, field: str) -> str:
    value = common.text(value, field)
    path = Path(value)
    common.require(
        not path.is_absolute()
        and value not in (".", "..")
        and ".." not in path.parts,
        f"E_PATH: {field}",
    )
    return value


def validate_inputs(
    contract_path: Path,
    candidate_path: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    contract, contract_raw = common.read_canonical(contract_path)
    common.exact(contract, builder.build_contract(), "contract")
    authority_sources = {
        "entrypoint": contract["exit_authority"]["entrypoint"],
        **contract["exit_authority"]["support"],
    }
    for name, record in authority_sources.items():
        path = S39 / record["path"]
        common.exact(path.stat().st_size, record["bytes"], f"authority.{name}.bytes")
        common.exact(
            common.sha256_file(path),
            record["sha256"],
            f"authority.{name}.sha256",
        )
    common.exact(
        common.sha256_bytes(_COMMON_SOURCE),
        contract["exit_authority"]["support"]["common"]["sha256"],
        "authority.common.executed_sha256",
    )
    common.exact(
        common.sha256_bytes(_BUILDER_SOURCE),
        contract["exit_authority"]["support"]["contract_builder"]["sha256"],
        "authority.contract_builder.executed_sha256",
    )
    candidate, candidate_raw = common.read_canonical(candidate_path)
    common.exact(
        common.sha256_bytes(candidate_raw),
        contract["candidate_lock"]["sha256"],
        "candidate.sha256",
    )
    common.exact(candidate["schema"], "s39-cp0-r1-candidate-v1", "candidate.schema")
    model = [row for row in candidate["models"] if row["slot"] == "A"]
    common.require(len(model) == 1, "E_CANDIDATE_A")
    common.exact(model[0]["model_id"], MODEL_ID, "candidate.model_id")
    return contract, contract_raw, candidate, candidate_raw


def raw_predicate_inputs(
    contract: dict[str, Any],
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    raw_contract, raw_parent = builder.build_raw_predicate_contract()
    raw = common.canonical_bytes(raw_contract)
    parent_raw = common.canonical_bytes(raw_parent)
    binding = contract["raw_predicate_contract"]
    common.exact(
        binding,
        {
            "historical_v2_2_sha256": contract["parent"][
                "v2_2_contract_sha256"
            ],
            "manifest_name": RAW_MANIFEST_NAME,
            "parent_sha256": common.sha256_bytes(parent_raw),
            "schema": "s39-cp0-r1-raw-predicate-contract-v2.4",
            "sha256": common.sha256_bytes(raw),
        },
        "raw_predicate_contract",
    )
    common.exact(
        raw_contract["serving_envelope"],
        contract["serving_envelope"],
        "raw_predicate_contract.serving_envelope",
    )
    common.exact(
        raw_parent["serving_envelope"],
        contract["serving_envelope"],
        "raw_predicate_parent.serving_envelope",
    )
    common.exact(
        raw_contract["serving_envelope"]["n_ctx_seq"],
        512,
        "raw_predicate_contract.n_ctx_seq",
    )
    return raw_contract, raw, raw_parent


def validate_route_envelope_digest(
    digest_value: Any,
    contract: dict[str, Any],
) -> None:
    v2 = _verified_helpers(contract)["v2"]

    common.exact(
        digest_value,
        v2.digest_json(contract["serving_envelope"]),
        "E_V2_4_ROUTE_ENVELOPE",
    )


def _path_under(path: str, root: str, field: str) -> None:
    path_value = Path(path)
    root_value = Path(root)
    common.require(
        path_value.is_relative_to(root_value),
        f"E_RUNTIME_ROOT: {field}",
    )


def validate_runtime_plan(
    plan: dict[str, Any],
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate_raw: bytes,
) -> dict[str, Any]:
    common.exact_keys(
        plan,
        {
            "bundle_roots",
            "bundles",
            "candidate_sha256",
            "capture_entrypoints",
            "components",
            "contract_sha256",
            "cuda_monolithic_launch",
            "model_id",
            "phase",
            "schema",
            "token_history",
        },
        "runtime_plan",
    )
    common.exact(
        plan["schema"],
        "s39-cp0-r1-runtime-bundle-plan-v2.4",
        "runtime_plan.schema",
    )
    common.exact(plan["phase"], PHASE, "runtime_plan.phase")
    common.exact(plan["model_id"], MODEL_ID, "runtime_plan.model")
    common.exact(
        plan["contract_sha256"],
        common.sha256_bytes(contract_raw),
        "runtime_plan.contract",
    )
    common.exact(
        plan["candidate_sha256"],
        common.sha256_bytes(candidate_raw),
        "runtime_plan.candidate",
    )

    roots = common.exact_keys(
        plan["bundle_roots"],
        set(REQUIRED_BUNDLES),
        "runtime_plan.bundle_roots",
    )
    for bundle_id, root in roots.items():
        common.absolute_path(root, f"runtime_plan.bundle_roots.{bundle_id}")
    root_paths = {bundle_id: Path(value) for bundle_id, value in roots.items()}
    for left_id, left in root_paths.items():
        for right_id, right in root_paths.items():
            if left_id != right_id:
                common.require(
                    not left.is_relative_to(right),
                    f"E_RUNTIME_ROOT_OVERLAP: {left_id}:{right_id}",
                )

    values = plan["components"]
    common.require(type(values) is list and bool(values), "E_RUNTIME_COMPONENTS")
    components: dict[str, dict[str, Any]] = {}
    locations: set[tuple[str, str]] = set()
    previous = None
    for index, value in enumerate(values):
        field = f"runtime_plan.components[{index}]"
        common.exact_keys(
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
        component_id = common.text(value["component_id"], f"{field}.component_id")
        common.require(component_id not in components, f"E_COMPONENT_REUSE: {component_id}")
        if previous is not None:
            common.require(previous < component_id, "E_COMPONENT_ORDER")
        previous = component_id
        bundle_id = common.text(value["bundle_id"], f"{field}.bundle_id")
        common.require(bundle_id in REQUIRED_BUNDLES, f"E_COMPONENT_BUNDLE: {field}")
        endpoint = common.text(value["endpoint"], f"{field}.endpoint")
        common.exact(
            endpoint,
            REQUIRED_BUNDLES[bundle_id][0],
            f"{field}.endpoint",
        )
        path = common.absolute_path(value["path"], f"{field}.path")
        _path_under(path, roots[bundle_id], field)
        common.require((endpoint, path) not in locations, f"E_COMPONENT_PATH_REUSE: {field}")
        locations.add((endpoint, path))
        common.require(
            value["role"] in ("backend_library", "executable", "shared_library"),
            f"E_COMPONENT_ROLE: {field}",
        )
        common.integer(value["bytes"], f"{field}.bytes", 1)
        common.digest(value["sha256"], f"{field}.sha256")
        components[component_id] = value

    values = plan["bundles"]
    common.require(
        type(values) is list and len(values) == len(REQUIRED_BUNDLES),
        "E_RUNTIME_BUNDLES",
    )
    bundles: dict[str, dict[str, Any]] = {}
    referenced: set[str] = set()
    previous = None
    for index, value in enumerate(values):
        field = f"runtime_plan.bundles[{index}]"
        common.exact_keys(
            value,
            {
                "bundle_id",
                "endpoint",
                "launcher_component_id",
                "process_role",
                "required_component_ids",
            },
            field,
        )
        bundle_id = common.text(value["bundle_id"], f"{field}.bundle_id")
        common.require(bundle_id in REQUIRED_BUNDLES, f"E_BUNDLE_ID: {field}")
        common.require(bundle_id not in bundles, f"E_BUNDLE_REUSE: {field}")
        if previous is not None:
            common.require(previous < bundle_id, "E_BUNDLE_ORDER")
        previous = bundle_id
        endpoint, process_role = REQUIRED_BUNDLES[bundle_id]
        common.exact(value["endpoint"], endpoint, f"{field}.endpoint")
        common.exact(value["process_role"], process_role, f"{field}.process_role")
        required = value["required_component_ids"]
        common.require(
            type(required) is list
            and bool(required)
            and required == sorted(set(required))
            and all(type(item) is str for item in required),
            f"E_BUNDLE_COMPONENTS: {field}",
        )
        launcher = common.text(
            value["launcher_component_id"],
            f"{field}.launcher_component_id",
        )
        common.require(launcher in required, f"E_BUNDLE_LAUNCHER: {field}")
        for component_id in required:
            common.require(component_id in components, f"E_BUNDLE_COMPONENT: {component_id}")
            common.exact(
                components[component_id]["bundle_id"],
                bundle_id,
                f"E_BUNDLE_COMPONENT_OWNER: {component_id}",
            )
        common.exact(
            components[launcher]["role"],
            "executable",
            f"E_BUNDLE_LAUNCHER_ROLE: {field}",
        )
        referenced.update(required)
        bundles[bundle_id] = value
    common.exact(set(bundles), set(REQUIRED_BUNDLES), "runtime_plan.bundle_ids")

    values = plan["capture_entrypoints"]
    common.require(
        type(values) is list and len(values) == len(CAPTURE_KINDS),
        "E_CAPTURE_ENTRYPOINTS",
    )
    captures: dict[str, dict[str, Any]] = {}
    used_capture_components: set[str] = set()
    previous = None
    for index, value in enumerate(values):
        field = f"runtime_plan.capture_entrypoints[{index}]"
        common.exact_keys(
            value,
            {
                "component_id",
                "execution_mode",
                "kind",
                "nested_capture_entrypoint_component_ids",
            },
            field,
        )
        kind = common.text(value["kind"], f"{field}.kind")
        common.require(kind in CAPTURE_KINDS and kind not in captures, f"E_CAPTURE_KIND: {field}")
        if previous is not None:
            common.require(previous < kind, "E_CAPTURE_ORDER")
        previous = kind
        common.exact(
            value["execution_mode"],
            "SELF_CONTAINED_PHYSICAL_CAPTURE",
            f"{field}.execution_mode",
        )
        component_id = common.text(value["component_id"], f"{field}.component_id")
        common.require(component_id in components, f"E_CAPTURE_COMPONENT: {field}")
        common.require(component_id not in used_capture_components, f"E_CAPTURE_COMPONENT_REUSE: {field}")
        common.require(
            component_id not in referenced,
            f"E_CAPTURE_COMPONENT_IS_RUNTIME: {field}",
        )
        used_capture_components.add(component_id)
        common.exact(components[component_id]["role"], "executable", f"{field}.role")
        nested = value["nested_capture_entrypoint_component_ids"]
        common.require(
            type(nested) is list
            and nested == sorted(set(nested))
            and all(type(item) is str and item in components for item in nested),
            f"E_CAPTURE_NESTED: {field}",
        )
        captures[kind] = value
    common.exact(set(captures), CAPTURE_KINDS, "runtime_plan.capture_kinds")
    expected_joint = sorted(
        bundles[bundle_id]["launcher_component_id"]
        for bundle_id in (
            "cuda_route",
            "op12_stagenet",
            "op15_direct_relay",
            "op15_stagenet",
        )
    )
    common.exact(
        captures["joint_phone_cuda"]["nested_capture_entrypoint_component_ids"],
        expected_joint,
        "runtime_plan.joint_nested_entrypoints",
    )
    common.exact(
        captures["cuda_monolithic"]["nested_capture_entrypoint_component_ids"],
        [bundles["cuda_monolithic"]["launcher_component_id"]],
        "runtime_plan.cuda_monolithic_nested_entrypoint",
    )
    producer_pins = contract["producer_requirements"]["source_programs"]
    for kind, producer_name in (
        ("cuda_monolithic", "cuda_monolithic"),
        ("joint_phone_cuda", "joint_phone_cuda"),
    ):
        component = components[captures[kind]["component_id"]]
        common.exact(
            component["bytes"],
            producer_pins[producer_name]["bytes"],
            f"E_CAPTURE_SOURCE_BYTES: {kind}",
        )
        common.exact(
            component["sha256"],
            producer_pins[producer_name]["sha256"],
            f"E_CAPTURE_SOURCE_SHA256: {kind}",
        )
    for kind in ("artifact_root", "fast_fresh_readiness"):
        common.exact(
            captures[kind]["nested_capture_entrypoint_component_ids"],
            [],
            f"runtime_plan.{kind}.nested_entrypoints",
        )
    common.exact(
        referenced | used_capture_components,
        set(components),
        "runtime_plan.unused_components",
    )
    token_history = common.exact_keys(
        plan["token_history"],
        {
            "artifact_path",
            "batch",
            "continuation_tokens_per_request",
            "corpus_sha256",
            "decode_calls_after_prefill",
            "model_sha256",
            "n_batch",
            "n_ctx_seq",
            "n_ubatch",
            "prefill_chunking",
            "prefill_row_order",
            "mechanics_item_indices",
            "quality_group_count",
            "quality_group_size",
            "quality_items",
            "tokenizer_component_id",
            "tokenizer_plan_bytes",
            "tokenizer_plan_path",
            "tokenizer_plan_sha256",
        },
        "runtime_plan.token_history",
    )
    protocol = contract["token_history_protocol"]
    for key in (
        "batch",
        "continuation_tokens_per_request",
        "decode_calls_after_prefill",
        "n_batch",
        "n_ctx_seq",
        "n_ubatch",
        "prefill_chunking",
        "prefill_row_order",
        "mechanics_item_indices",
        "quality_group_count",
        "quality_group_size",
        "quality_items",
    ):
        common.exact(
            token_history[key],
            protocol[key],
            f"runtime_plan.token_history.{key}",
        )
    common.exact(
        token_history["corpus_sha256"],
        contract["quality_corpus"]["sha256"],
        "runtime_plan.token_history.corpus",
    )
    common.exact(
        token_history["model_sha256"],
        next(
            row["artifact"]["sha256"]
            for row in common.parse_json(candidate_raw, "candidate")["models"]
            if row["slot"] == "A"
        ),
        "runtime_plan.token_history.model",
    )
    tokenizer_component_id = common.text(
        token_history["tokenizer_component_id"],
        "runtime_plan.token_history.tokenizer_component_id",
    )
    common.require(
        tokenizer_component_id in components,
        "E_TOKENIZER_COMPONENT",
    )
    tokenizer = components[tokenizer_component_id]
    common.exact(tokenizer["endpoint"], "cuda", "E_TOKENIZER_ENDPOINT")
    common.exact(tokenizer["role"], "executable", "E_TOKENIZER_ROLE")
    artifact_path = common.absolute_path(
        token_history["artifact_path"],
        "runtime_plan.token_history.artifact_path",
    )
    common.require(
        all(
            artifact_path != component["path"]
            for component in components.values()
        ),
        "E_TOKEN_HISTORY_PATH_REUSE",
    )
    tokenizer_plan_path = common.absolute_path(
        token_history["tokenizer_plan_path"],
        "runtime_plan.token_history.tokenizer_plan_path",
    )
    common.require(
        tokenizer_plan_path != artifact_path
        and all(
            tokenizer_plan_path != component["path"]
            for component in components.values()
        ),
        "E_TOKENIZER_PLAN_PATH_REUSE",
    )
    common.integer(
        token_history["tokenizer_plan_bytes"],
        "runtime_plan.token_history.tokenizer_plan_bytes",
        1,
    )
    common.digest(
        token_history["tokenizer_plan_sha256"],
        "runtime_plan.token_history.tokenizer_plan_sha256",
    )
    cuda_launch = common.exact_keys(
        plan["cuda_monolithic_launch"],
        {
            "allowed_system_roots",
            "bundle_id",
            "bundle_root",
            "bundle_sha256",
            "command",
            "cwd",
            "endpoint",
            "env",
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
            "launcher_component_id",
            "model_artifact",
            "model_id",
            "model_sha256",
            "port",
            "required_components",
            "route_epoch",
            "schema",
            "shutdown_timeout_ms",
            "startup_timeout_ms",
        },
        "runtime_plan.cuda_monolithic_launch",
    )
    mono_bundle = bundles["cuda_monolithic"]
    mono_launcher = components[mono_bundle["launcher_component_id"]]
    common.exact(
        cuda_launch["schema"],
        "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
        "runtime_plan.cuda_monolithic_launch.schema",
    )
    common.exact(
        cuda_launch["bundle_id"],
        "cuda_monolithic",
        "runtime_plan.cuda_monolithic_launch.bundle",
    )
    common.exact(
        cuda_launch["bundle_root"],
        roots["cuda_monolithic"],
        "runtime_plan.cuda_monolithic_launch.root",
    )
    common.exact(
        cuda_launch["endpoint"],
        "cuda",
        "runtime_plan.cuda_monolithic_launch.endpoint",
    )
    common.exact(
        cuda_launch["launcher_component_id"],
        mono_bundle["launcher_component_id"],
        "runtime_plan.cuda_monolithic_launch.launcher",
    )
    common.digest(
        cuda_launch["bundle_sha256"],
        "runtime_plan.cuda_monolithic_launch.bundle_sha256",
    )
    argv = cuda_launch["command"]
    common.require(
        type(argv) is list
        and len(argv) >= 3
        and all(type(value) is str and bool(value) and value.isascii() for value in argv)
        and all("{" not in value and "}" not in value for value in argv),
        "E_CUDA_MONOLITHIC_ARGV",
    )
    common.exact(argv[0], mono_launcher["path"], "E_CUDA_MONOLITHIC_ARGV0")
    model_path = contract["model_geometry"][MODEL_ID]["cuda_model_path"]
    port = common.integer(
        cuda_launch["port"],
        "runtime_plan.cuda_monolithic_launch.port",
        1,
    )
    common.require(port <= 65535, "E_CUDA_MONOLITHIC_PORT")
    common.exact(
        argv,
        [
            mono_launcher["path"],
            "-m",
            model_path,
            "--mode",
            "monov3",
            "--port",
            str(port),
            "--devices",
            "CUDA0",
            "--driver-batch",
            "8",
            "--driver-context",
            "512",
            "--driver-max-prefill",
            "8",
        ],
        "E_CUDA_MONOLITHIC_COMMAND",
    )
    environment = common.exact_keys(
        cuda_launch["env"],
        {
            "CUDA_VISIBLE_DEVICES",
            "HOME",
            "LAYERSPLIT_MEMORY_CERT",
            "LAYERSPLIT_MODEL_SHA256",
            "LAYERSPLIT_PLACEMENT_CERT",
            "LC_ALL",
            "LD_LIBRARY_PATH",
        },
        "runtime_plan.cuda_monolithic_launch.environment",
    )
    common.exact(environment, {
        "CUDA_VISIBLE_DEVICES": "0",
        "HOME": "/home/zhihao",
        "LAYERSPLIT_MEMORY_CERT": "1",
        "LAYERSPLIT_MODEL_SHA256": token_history["model_sha256"],
        "LAYERSPLIT_PLACEMENT_CERT": "1",
        "LC_ALL": "C",
        "LD_LIBRARY_PATH": roots["cuda_monolithic"],
    }, "E_CUDA_MONOLITHIC_ENV")
    common.exact(
        cuda_launch["cwd"],
        "/home/zhihao/llama.cpp-s40",
        "E_CUDA_MONOLITHIC_CWD",
    )
    common.exact(
        cuda_launch["allowed_system_roots"],
        [
            "/mnt/storage/s21_deps/cuda-13.2.1/lib/",
            "/usr/lib/x86_64-linux-gnu/",
        ],
        "E_CUDA_MONOLITHIC_SYSTEM_ROOTS",
    )
    common.exact(cuda_launch["host"], "127.0.0.1", "E_CUDA_MONOLITHIC_HOST")
    for key, expected in (
        ("expected_capabilities", 0x3F),
        ("expected_max_streams", 8),
        ("expected_n_batch", 64),
        ("expected_n_ctx_seq", 512),
        ("expected_n_embd", 5120),
        ("expected_n_layer", 40),
        ("expected_n_ubatch", 64),
        ("io_timeout_ms", 300000),
        ("shutdown_timeout_ms", 30000),
        ("startup_timeout_ms", 300000),
    ):
        common.exact(
            cuda_launch[key],
            expected,
            f"runtime_plan.cuda_monolithic_launch.{key}",
        )
    common.exact(
        cuda_launch["model_id"],
        MODEL_ID,
        "runtime_plan.cuda_monolithic_launch.model_id",
    )
    common.exact(
        cuda_launch["model_sha256"],
        token_history["model_sha256"],
        "runtime_plan.cuda_monolithic_launch.model_sha256",
    )
    common.exact(
        cuda_launch["expected_file_type"],
        contract["cuda_monolithic_identity"]["expected_file_type"],
        "runtime_plan.cuda_monolithic_launch.file_type",
    )
    common.integer(
        cuda_launch["route_epoch"],
        "runtime_plan.cuda_monolithic_launch.route_epoch",
        1,
    )
    launch_components = cuda_launch["required_components"]
    common.require(
        type(launch_components) is list
        and len(launch_components) == len(mono_bundle["required_component_ids"]),
        "E_CUDA_MONOLITHIC_LAUNCH_COMPONENTS",
    )
    for index, (component, component_id) in enumerate(
        zip(launch_components, mono_bundle["required_component_ids"])
    ):
        field = f"runtime_plan.cuda_monolithic_launch.required_components[{index}]"
        common.exact_keys(
            component,
            {"component_id", "path", "sha256", "stat"},
            field,
        )
        common.exact(component["component_id"], component_id, f"{field}.id")
        for key in ("path", "sha256"):
            common.exact(
                component[key],
                components[component_id][key],
                f"{field}.{key}",
            )
        common.stat_record(component["stat"], f"{field}.stat")
    model_artifact = common.exact_keys(
        cuda_launch["model_artifact"],
        {"path", "sha256", "stat"},
        "runtime_plan.cuda_monolithic_launch.model_artifact",
    )
    common.exact(model_artifact["path"], model_path, "E_CUDA_MONOLITHIC_MODEL_PATH")
    common.exact(
        model_artifact["sha256"],
        token_history["model_sha256"],
        "E_CUDA_MONOLITHIC_MODEL_SHA",
    )
    common.stat_record(model_artifact["stat"], "cuda_monolithic_launch.model.stat")
    return {
        "bundles": bundles,
        "captures": captures,
        "components": components,
        "cuda_monolithic_launch": cuda_launch,
        "roots": roots,
        "token_history": token_history,
    }


def _expected_weight_components(
    contract: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    model = next(row for row in candidate["models"] if row["slot"] == "A")
    geometry = contract["model_geometry"][MODEL_ID]
    return {
        "model.cuda": {
            "bytes": model["artifact"]["bytes"],
            "endpoint": "cuda",
            "kind": "model_weight",
            "path": geometry["cuda_model_path"],
            "sha256": model["artifact"]["sha256"],
        },
        "model.op12_shard": {
            "bytes": geometry["known_shards"]["op12"]["bytes"],
            "endpoint": "op12",
            "kind": "model_shard",
            "path": geometry["known_shards"]["op12"]["path"],
            "sha256": geometry["known_shards"]["op12"]["sha256"],
        },
        "model.op15_shard": {
            "bytes": geometry["known_shards"]["op15"]["bytes"],
            "endpoint": "op15",
            "kind": "model_shard",
            "path": geometry["known_shards"]["op15"]["path"],
            "sha256": geometry["known_shards"]["op15"]["sha256"],
        },
    }


def _load_corpus(contract: dict[str, Any]) -> dict[int, dict[str, Any]]:
    path = S39 / contract["quality_corpus"]["path"]
    raw = path.read_bytes()
    common.exact(len(raw), contract["quality_corpus"]["bytes"], "quality_corpus.bytes")
    common.exact(
        common.sha256_bytes(raw),
        contract["quality_corpus"]["sha256"],
        "quality_corpus.sha256",
    )
    rows = {}
    for index, line in enumerate(raw.splitlines(keepends=True)):
        common.require(line.endswith(b"\n"), f"E_CORPUS_LINE: {index}")
        row = common.parse_json(line, f"quality_corpus[{index}]")
        common.require(type(row) is dict, f"E_CORPUS_ROW: {index}")
        common.require(
            common.canonical_bytes(row) == line,
            f"E_CORPUS_CANONICAL: {index}",
        )
        item_index = common.integer(row.get("item_index"), f"quality_corpus[{index}].item_index")
        common.require(item_index not in rows, f"E_CORPUS_ITEM_REUSE: {item_index}")
        rows[item_index] = row
    common.exact(len(rows), contract["quality_corpus"]["items"], "quality_corpus.items")
    return rows


def _prompt(row: dict[str, Any], candidate: dict[str, Any]) -> str:
    choices = row["choices"]
    common.require(
        type(choices) is list
        and len(choices) == 4
        and all(type(value) is str for value in choices),
        "E_CORPUS_CHOICES",
    )
    try:
        return candidate["task_suite"]["prompt_format"].format(
            question=row["question"],
            choice0=choices[0],
            choice1=choices[1],
            choice2=choices[2],
            choice3=choices[3],
        )
    except (KeyError, IndexError, ValueError) as error:
        raise common.EvidenceError("E_PROMPT_FORMAT") from error


def validate_tokenizer_plan(
    tokenizer_plan: dict[str, Any],
    tokenizer_plan_raw: bytes,
    contract: dict[str, Any],
    candidate: dict[str, Any],
    plan_derived: dict[str, Any],
) -> None:
    common.exact_keys(
        tokenizer_plan,
        {
            "command_template",
            "component_id",
            "cwd",
            "environment",
            "executable",
            "model",
            "protocol",
            "schema",
            "timeout_seconds",
        },
        "tokenizer_plan",
    )
    common.exact(
        tokenizer_plan["schema"],
        "s39-cp0-r1-a-only-tokenizer-plan-v2",
        "tokenizer_plan.schema",
    )
    spec = plan_derived["token_history"]
    common.exact(
        len(tokenizer_plan_raw),
        spec["tokenizer_plan_bytes"],
        "tokenizer_plan.bytes",
    )
    common.exact(
        common.sha256_bytes(tokenizer_plan_raw),
        spec["tokenizer_plan_sha256"],
        "tokenizer_plan.sha256",
    )
    common.exact(
        tokenizer_plan["component_id"],
        spec["tokenizer_component_id"],
        "tokenizer_plan.component_id",
    )
    component = plan_derived["components"][spec["tokenizer_component_id"]]
    executable = common.exact_keys(
        tokenizer_plan["executable"],
        {"bytes", "path", "sha256"},
        "tokenizer_plan.executable",
    )
    for key in ("bytes", "path", "sha256"):
        common.exact(
            executable[key],
            component[key],
            f"tokenizer_plan.executable.{key}",
        )
    model_a = next(row for row in candidate["models"] if row["slot"] == "A")
    model = common.exact_keys(
        tokenizer_plan["model"],
        {"bytes", "model_id", "path", "sha256", "vocab_size"},
        "tokenizer_plan.model",
    )
    common.exact(model["model_id"], MODEL_ID, "tokenizer_plan.model.id")
    common.exact(
        model["path"],
        contract["model_geometry"][MODEL_ID]["cuda_model_path"],
        "tokenizer_plan.model.path",
    )
    common.exact(model["bytes"], model_a["artifact"]["bytes"], "tokenizer_plan.model.bytes")
    common.exact(model["sha256"], model_a["artifact"]["sha256"], "tokenizer_plan.model.sha256")
    common.exact(
        model["vocab_size"],
        contract["token_history_protocol"]["vocab_size"],
        "tokenizer_plan.model.vocab_size",
    )
    expected_command = [
        component["path"],
        "-m",
        model["path"],
        "--ids",
        "-f",
        "{PROMPT_FILE}",
        "--log-disable",
    ]
    common.exact(
        tokenizer_plan["command_template"],
        expected_command,
        "tokenizer_plan.command_template",
    )
    common.exact(
        tokenizer_plan["cwd"],
        str(Path(component["path"]).parent),
        "tokenizer_plan.cwd",
    )
    common.exact(
        tokenizer_plan["environment"],
        {
            "LC_ALL": "C",
            "LD_LIBRARY_PATH": str(Path(component["path"]).parent),
        },
        "tokenizer_plan.environment",
    )
    common.exact(
        tokenizer_plan["protocol"],
        {
            "add_bos": "MODEL_DEFAULT",
            "escape": True,
            "output_format": "BRACKETED_DECIMAL_IDS",
            "parse_special": True,
            "prompt_file_placeholder": "{PROMPT_FILE}",
        },
        "tokenizer_plan.protocol",
    )
    common.exact(tokenizer_plan["timeout_seconds"], 300, "tokenizer_plan.timeout")


def validate_token_history(
    history: dict[str, Any],
    history_raw: bytes,
    contract: dict[str, Any],
    candidate: dict[str, Any],
    plan_derived: dict[str, Any],
) -> None:
    common.exact_keys(
        history,
        {
            "batch",
            "candidate_sha256",
            "continuation_tokens_per_request",
            "corpus_sha256",
            "mechanics_b8",
            "model_id",
            "model_sha256",
            "n_batch",
            "n_ctx_seq",
            "n_ubatch",
            "prefill_chunking",
            "prefill_row_order",
            "quality_groups",
            "requests",
            "schema",
            "tokenizer",
        },
        "token_history",
    )
    common.exact(
        history["schema"],
        "s39-cp0-r1-token-history-v2.4",
        "token_history.schema",
    )
    spec = plan_derived["token_history"]
    common.exact(history["model_id"], MODEL_ID, "token_history.model_id")
    common.exact(
        history["candidate_sha256"],
        contract["candidate_lock"]["sha256"],
        "token_history.candidate",
    )
    for key in (
        "batch",
        "continuation_tokens_per_request",
        "corpus_sha256",
        "model_sha256",
        "n_batch",
        "n_ctx_seq",
        "n_ubatch",
        "prefill_chunking",
        "prefill_row_order",
    ):
        common.exact(history[key], spec[key], f"token_history.{key}")
    tokenizer = common.exact_keys(
        history["tokenizer"],
        {"component_id", "path", "plan_sha256", "sha256"},
        "token_history.tokenizer",
    )
    tokenizer_component = plan_derived["components"][spec["tokenizer_component_id"]]
    common.exact(tokenizer["component_id"], spec["tokenizer_component_id"], "token_history.tokenizer.id")
    common.exact(tokenizer["path"], tokenizer_component["path"], "token_history.tokenizer.path")
    common.exact(tokenizer["sha256"], tokenizer_component["sha256"], "token_history.tokenizer.sha256")
    common.exact(
        tokenizer["plan_sha256"],
        spec["tokenizer_plan_sha256"],
        "token_history.tokenizer.plan",
    )

    corpus = _load_corpus(contract)
    values = history["requests"]
    common.require(
        type(values) is list and len(values) == spec["quality_items"],
        "E_TOKEN_REQUESTS",
    )
    requests: dict[int, dict[str, Any]] = {}
    for request_index, value in enumerate(values):
        field = f"token_history.requests[{request_index}]"
        common.exact_keys(
            value,
            {
                "item_index",
                "prompt_sha256",
                "prompt_utf8_base64",
                "prompt_utf8_bytes",
                "request_id",
                "seq_id",
                "token_ids",
            },
            field,
        )
        item_index = request_index
        common.exact(value["item_index"], item_index, f"{field}.item_index")
        common.exact(
            value["request_id"],
            item_index % spec["batch"] + 1,
            f"{field}.request_id",
        )
        common.exact(
            value["seq_id"],
            item_index % spec["batch"],
            f"{field}.seq_id",
        )
        common.require(item_index in corpus, f"E_TOKEN_CORPUS_ITEM: {item_index}")
        prompt_raw = _prompt(corpus[item_index], candidate).encode(
            contract["token_history_protocol"]["prompt_hash_encoding"].lower()
        )
        prompt_sha256 = common.sha256_bytes(prompt_raw)
        common.exact(value["prompt_sha256"], prompt_sha256, f"{field}.prompt_sha256")
        common.exact(value["prompt_utf8_bytes"], len(prompt_raw), f"{field}.prompt_utf8_bytes")
        common.exact(
            value["prompt_utf8_base64"],
            base64.b64encode(prompt_raw).decode("ascii"),
            f"{field}.prompt_utf8_base64",
        )
        tokens = value["token_ids"]
        common.require(
            type(tokens) is list
            and 0 < len(tokens)
            <= (
                spec["n_ctx_seq"]
                - spec["continuation_tokens_per_request"]
            ),
            f"E_TOKEN_IDS: {field}",
        )
        for position, token_id in enumerate(tokens):
            common.require(
                common.integer(token_id, f"{field}.token_ids[{position}]")
                < contract["token_history_protocol"]["vocab_size"],
                f"E_TOKEN_RANGE: {field}.token_ids[{position}]",
            )
        requests[item_index] = value

    def expected_group(group_index: int) -> dict[str, Any]:
        item_indices = list(
            range(
                group_index * spec["quality_group_size"],
                (group_index + 1) * spec["quality_group_size"],
            )
        )
        token_rows: list[tuple[int, int, int, int, int]] = []
        for item_index in item_indices:
            request = requests[item_index]
            for position, token_id in enumerate(request["token_ids"]):
                token_rows.append(
                    (
                        position,
                        item_index,
                        request["request_id"],
                        request["seq_id"],
                        token_id,
                    )
                )
        token_rows.sort(key=lambda row: (row[0], row[1]))
        waves = [
            [row for row in token_rows if row[0] == position]
            for position in sorted({row[0] for row in token_rows})
        ]
        partitions: list[list[tuple[int, int, int, int, int]]] = []
        current: list[tuple[int, int, int, int, int]] = []
        for wave in waves:
            common.require(len(wave) <= spec["n_ubatch"], "E_TOKEN_POSITION_WAVE")
            if current and len(current) + len(wave) > spec["n_ubatch"]:
                partitions.append(current)
                current = []
            current.extend(wave)
        if current:
            partitions.append(current)
        prefill_partitions = [
            {
                "call_index": call_index,
                "rows": [
                    {
                        "item_index": item_index,
                        "position": position,
                        "request_id": request_id,
                        "seq_id": seq_id,
                        "token_id": token_id,
                    }
                    for position, item_index, request_id, seq_id, token_id
                    in partition
                ],
            }
            for call_index, partition in enumerate(partitions)
        ]
        decode_calls = [
            {
                "call_index": len(prefill_partitions) + call_index,
                "continuation_input_ordinal": call_index,
                "continuation_output_ordinal": call_index + 1,
                "rows": [
                    {
                        "item_index": item_index,
                        "position": (
                            len(requests[item_index]["token_ids"]) + call_index
                        ),
                        "request_id": requests[item_index]["request_id"],
                        "seq_id": requests[item_index]["seq_id"],
                    }
                    for item_index in item_indices
                ],
            }
            for call_index in range(spec["decode_calls_after_prefill"])
        ]
        return {
            "decode_calls": decode_calls,
            "group_index": group_index,
            "item_indices": item_indices,
            "prefill_partitions": prefill_partitions,
        }

    expected_groups = [
        expected_group(group_index)
        for group_index in range(spec["quality_group_count"])
    ]
    common.exact(
        history["quality_groups"],
        expected_groups,
        "token_history.quality_groups",
    )
    common.exact(
        history["mechanics_b8"],
        expected_groups[0],
        "token_history.mechanics_b8",
    )
    common.exact(
        history["mechanics_b8"],
        history["quality_groups"][0],
        "E_MECHANICS_GROUP_ZERO",
    )
    common.require(
        common.canonical_bytes(history) == history_raw,
        "E_TOKEN_HISTORY_CANONICAL",
    )


def _root_component_map(
    root: dict[str, Any],
    contract: dict[str, Any],
    candidate: dict[str, Any],
    plan_derived: dict[str, Any],
    history_raw: bytes,
    tokenizer_plan_raw: bytes,
) -> dict[str, dict[str, Any]]:
    values = root["components"]
    expected_count = len(plan_derived["components"]) + 5
    common.require(type(values) is list and len(values) == expected_count, "E_ROOT_COMPONENTS")
    result: dict[str, dict[str, Any]] = {}
    locations: set[tuple[str, str]] = set()
    previous = None
    expected = _expected_weight_components(contract, candidate)
    expected["token_history.mmlu64"] = {
        "bytes": len(history_raw),
        "endpoint": "cuda",
        "kind": "token_history",
        "path": plan_derived["token_history"]["artifact_path"],
        "sha256": common.sha256_bytes(history_raw),
    }
    expected["tokenizer.plan"] = {
        "bytes": len(tokenizer_plan_raw),
        "endpoint": "cuda",
        "kind": "tokenizer_plan",
        "path": plan_derived["token_history"]["tokenizer_plan_path"],
        "sha256": common.sha256_bytes(tokenizer_plan_raw),
    }
    for component_id, value in plan_derived["components"].items():
        expected[component_id] = {
            "bytes": value["bytes"],
            "endpoint": value["endpoint"],
            "kind": "runtime_component",
            "path": value["path"],
            "sha256": value["sha256"],
        }
    for index, value in enumerate(values):
        field = f"artifact_root.components[{index}]"
        common.exact_keys(
            value,
            {
                "bytes",
                "component_id",
                "endpoint",
                "kind",
                "path",
                "sha256",
                "stat",
            },
            field,
        )
        component_id = common.text(value["component_id"], f"{field}.component_id")
        common.require(component_id not in result, f"E_ROOT_COMPONENT_REUSE: {field}")
        if previous is not None:
            common.require(previous < component_id, "E_ROOT_COMPONENT_ORDER")
        previous = component_id
        common.require(component_id in expected, f"E_ROOT_COMPONENT_UNKNOWN: {component_id}")
        for key in ("bytes", "endpoint", "kind", "path", "sha256"):
            common.exact(value[key], expected[component_id][key], f"{field}.{key}")
        common.stat_record(value["stat"], f"{field}.stat")
        common.exact(value["stat"]["size"], value["bytes"], f"{field}.stat.size")
        location = (value["endpoint"], value["path"])
        common.require(location not in locations, f"E_ROOT_PATH_REUSE: {field}")
        locations.add(location)
        result[component_id] = value
    common.exact(set(result), set(expected), "artifact_root.component_ids")
    return result


def _validate_inventories(
    values: Any,
    plan_derived: dict[str, Any],
    field: str,
) -> dict[str, dict[str, Any]]:
    common.require(
        type(values) is list and len(values) == len(REQUIRED_BUNDLES),
        f"E_INVENTORIES: {field}",
    )
    result: dict[str, dict[str, Any]] = {}
    previous = None
    for index, value in enumerate(values):
        item = f"{field}[{index}]"
        common.exact_keys(
            value,
            {"bundle_id", "endpoint", "paths", "root"},
            item,
        )
        bundle_id = common.text(value["bundle_id"], f"{item}.bundle_id")
        common.require(bundle_id in REQUIRED_BUNDLES and bundle_id not in result, f"E_INVENTORY_ID: {item}")
        if previous is not None:
            common.require(previous < bundle_id, f"E_INVENTORY_ORDER: {field}")
        previous = bundle_id
        bundle = plan_derived["bundles"][bundle_id]
        common.exact(value["endpoint"], bundle["endpoint"], f"{item}.endpoint")
        common.exact(value["root"], plan_derived["roots"][bundle_id], f"{item}.root")
        expected_paths = sorted(
            plan_derived["components"][component_id]["path"]
            for component_id in bundle["required_component_ids"]
        )
        common.exact(value["paths"], expected_paths, f"{item}.paths")
        result[bundle_id] = value
    common.exact(set(result), set(REQUIRED_BUNDLES), f"{field}.bundle_ids")
    return result


def validate_artifact_root(
    root: dict[str, Any],
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate: dict[str, Any],
    candidate_raw: bytes,
    plan_raw: bytes,
    plan_derived: dict[str, Any],
    history_raw: bytes,
    tokenizer_plan_raw: bytes,
) -> dict[str, Any]:
    common.exact_keys(
        root,
        {
            "candidate_sha256",
            "completed_ns",
            "components",
            "contract_sha256",
            "inventories",
            "model_id",
            "phase_scope",
            "runtime_bundle_plan_sha256",
            "schema",
            "started_ns",
        },
        "artifact_root",
    )
    common.exact(root["schema"], "s39-cp0-r1-artifact-root-v2.4", "artifact_root.schema")
    common.exact(root["phase_scope"], "PRE_REBOOT_OUTSIDE_PHASE", "artifact_root.scope")
    common.exact(root["model_id"], MODEL_ID, "artifact_root.model")
    common.exact(root["contract_sha256"], common.sha256_bytes(contract_raw), "artifact_root.contract")
    common.exact(root["candidate_sha256"], common.sha256_bytes(candidate_raw), "artifact_root.candidate")
    common.exact(
        root["runtime_bundle_plan_sha256"],
        common.sha256_bytes(plan_raw),
        "artifact_root.runtime_plan",
    )
    started = common.integer(root["started_ns"], "artifact_root.started", 1)
    completed = common.integer(root["completed_ns"], "artifact_root.completed", 1)
    common.require(started < completed, "E_ARTIFACT_ROOT_INTERVAL")
    components = _root_component_map(
        root,
        contract,
        candidate,
        plan_derived,
        history_raw,
        tokenizer_plan_raw,
    )
    inventories = _validate_inventories(
        root["inventories"],
        plan_derived,
        "artifact_root.inventories",
    )
    launch = plan_derived["cuda_monolithic_launch"]
    for index, value in enumerate(launch["required_components"]):
        component = components[value["component_id"]]
        common.exact(
            value["stat"],
            component["stat"],
            f"E_CUDA_LAUNCH_COMPONENT_STAT: {index}",
        )
    common.exact(
        launch["model_artifact"]["stat"],
        components["model.cuda"]["stat"],
        "E_CUDA_LAUNCH_MODEL_STAT",
    )
    common.exact(
        launch["bundle_sha256"],
        _bundle_digest(
            "cuda_monolithic",
            plan_derived,
            components,
        ),
        "E_CUDA_LAUNCH_BUNDLE_DIGEST",
    )
    return {
        "completed_ns": completed,
        "components": components,
        "inventories": inventories,
    }


def _validate_device_snapshot(
    value: Any,
    contract: dict[str, Any],
    field: str,
    *,
    phone_network: bool = False,
) -> dict[str, dict[str, Any]]:
    value = common.exact_keys(value, {"cuda", "op12", "op15"}, field)
    cuda = common.exact_keys(
        value["cuda"],
        {
            "gpu_uuid",
            "host",
            "host_boot_id",
            "pci_bus_id",
            "system_swap_used_bytes",
        },
        f"{field}.cuda",
    )
    common.exact(cuda["gpu_uuid"], contract["devices"]["cuda"]["uuid"], f"{field}.cuda.uuid")
    common.exact(cuda["host"], contract["devices"]["cuda"]["host"], f"{field}.cuda.host")
    common.uuid(cuda["host_boot_id"], f"{field}.cuda.boot_id")
    common.text(cuda["pci_bus_id"], f"{field}.cuda.pci_bus_id")
    common.integer(
        cuda["system_swap_used_bytes"],
        f"{field}.cuda.swap",
    )
    result = {"cuda": cuda}
    for phone in ("op12", "op15"):
        keys = {
            "available_bytes",
            "boot_id",
            "device",
            "model",
            "product",
            "serial",
            "system_swap_used_bytes",
            "thermal_status",
        }
        if phone_network:
            keys |= {"interface", "local_ipv4"}
        record = common.exact_keys(
            value[phone],
            keys,
            f"{field}.{phone}",
        )
        for key in ("device", "model", "product", "serial"):
            common.exact(
                record[key],
                contract["devices"][phone][key],
                f"{field}.{phone}.{key}",
            )
        common.uuid(record["boot_id"], f"{field}.{phone}.boot_id")
        common.require(
            common.integer(record["available_bytes"], f"{field}.{phone}.available")
            >= contract["gates"]["phone_minimum_available_bytes"],
            f"E_PHONE_MEMORY: {phone}",
        )
        common.integer(
            record["system_swap_used_bytes"],
            f"{field}.{phone}.system_swap",
        )
        common.exact(record["thermal_status"], 0, f"{field}.{phone}.thermal")
        if phone_network:
            common.exact(
                record["interface"],
                "wlan0",
                f"{field}.{phone}.interface",
            )
            try:
                address = ipaddress.IPv4Address(
                    common.text(
                        record["local_ipv4"],
                        f"{field}.{phone}.local_ipv4",
                    )
                )
            except ipaddress.AddressValueError as error:
                raise common.EvidenceError(
                    f"E_PHONE_IPV4: {phone}"
                ) from error
            common.require(
                not (
                    address.is_unspecified
                    or address.is_loopback
                    or address.is_multicast
                ),
                f"E_PHONE_IPV4: {phone}",
            )
        result[phone] = record
    return result


def validate_preparation(
    preparation: dict[str, Any],
    preparation_raw: bytes,
    contract: dict[str, Any],
    root_raw: bytes,
    root_completed_ns: int,
    plan_raw: bytes,
) -> dict[str, Any]:
    del preparation_raw
    common.exact_keys(
        preparation,
        {
            "artifact_root_sha256",
            "before_boot_ids",
            "completed_ns",
            "devices",
            "reboot_started_ns",
            "runtime_bundle_plan_sha256",
            "schema",
            "started_ns",
        },
        "preparation",
    )
    common.exact(
        preparation["schema"],
        "s39-cp0-r1-reboot-preparation-v2.4",
        "preparation.schema",
    )
    common.exact(preparation["artifact_root_sha256"], common.sha256_bytes(root_raw), "preparation.root")
    common.exact(
        preparation["runtime_bundle_plan_sha256"],
        common.sha256_bytes(plan_raw),
        "preparation.runtime_plan",
    )
    started = common.integer(preparation["started_ns"], "preparation.started", 1)
    reboot = common.integer(preparation["reboot_started_ns"], "preparation.reboot", 1)
    completed = common.integer(preparation["completed_ns"], "preparation.completed", 1)
    common.require(
        root_completed_ns <= started <= reboot < completed,
        "E_PREPARATION_ORDER",
    )
    devices = _validate_device_snapshot(
        preparation["devices"],
        contract,
        "preparation.devices",
        phone_network=True,
    )
    before_boot_ids = common.exact_keys(
        preparation["before_boot_ids"],
        {"op12", "op15"},
        "preparation.before_boot_ids",
    )
    for phone in ("op12", "op15"):
        common.uuid(
            before_boot_ids[phone],
            f"preparation.before_boot_ids.{phone}",
        )
        common.require(
            before_boot_ids[phone] != devices[phone]["boot_id"],
            f"E_STALE_BOOT_REUSE: {phone}",
        )
    common.require(
        len(set(before_boot_ids.values())) == 2,
        "E_BEFORE_BOOT_ID_REUSE",
    )
    common.require(
        devices["op12"]["local_ipv4"] != devices["op15"]["local_ipv4"],
        "E_PHONE_IPV4_REUSE",
    )
    return {
        "before_boot_ids": before_boot_ids,
        "completed_ns": completed,
        "devices": devices,
    }


def validate_phase_lock(
    lock: dict[str, Any],
    lock_raw: bytes,
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate_raw: bytes,
    root_raw: bytes,
    root_completed_ns: int,
    preparation_raw: bytes,
    preparation_derived: dict[str, Any],
    plan_raw: bytes,
) -> dict[str, Any]:
    del lock_raw
    common.exact_keys(
        lock,
        {
            "artifact_root_sha256",
            "candidate_sha256",
            "contract_sha256",
            "device_boot_ids",
            "event_ns",
            "model_id",
            "phase",
            "phase_id",
            "preparation_sha256",
            "quality_corpus_sha256",
            "runtime_bundle_plan_sha256",
            "schema",
        },
        "phase_lock",
    )
    common.exact(lock["schema"], "s39-cp0-r1-phase-lock-v2.4", "phase_lock.schema")
    common.exact(lock["phase"], PHASE, "phase_lock.phase")
    common.exact(lock["model_id"], MODEL_ID, "phase_lock.model")
    common.exact(
        lock["quality_corpus_sha256"],
        contract["quality_corpus"]["sha256"],
        "phase_lock.quality_corpus_sha256",
    )
    phase_id = common.text(lock["phase_id"], "phase_lock.phase_id")
    common.require(
        phase_id.startswith("cp0-r1-v24-a-only-") and len(phase_id) <= 128,
        "E_PHASE_ID",
    )
    common.exact(lock["contract_sha256"], common.sha256_bytes(contract_raw), "phase_lock.contract")
    common.exact(lock["candidate_sha256"], common.sha256_bytes(candidate_raw), "phase_lock.candidate")
    common.exact(lock["artifact_root_sha256"], common.sha256_bytes(root_raw), "phase_lock.root")
    common.exact(
        lock["preparation_sha256"],
        common.sha256_bytes(preparation_raw),
        "phase_lock.preparation",
    )
    common.exact(
        lock["runtime_bundle_plan_sha256"],
        common.sha256_bytes(plan_raw),
        "phase_lock.runtime_plan",
    )
    event = common.integer(lock["event_ns"], "phase_lock.event", 1)
    common.require(preparation_derived["completed_ns"] <= event, "E_PHASE_BEFORE_PREPARATION")
    common.require(
        event - root_completed_ns
        <= contract["gates"]["artifact_root_maximum_age_ns"],
        "E_ARTIFACT_ROOT_STALE",
    )
    boot_ids = common.exact_keys(
        lock["device_boot_ids"],
        {"cuda", "op12", "op15"},
        "phase_lock.device_boot_ids",
    )
    for endpoint in ("cuda", "op12", "op15"):
        common.exact(
            boot_ids[endpoint],
            preparation_derived["devices"][endpoint][
                "host_boot_id" if endpoint == "cuda" else "boot_id"
            ],
            f"phase_lock.boot_id.{endpoint}",
        )
    return {"event_ns": event, "phase_id": phase_id, "boot_ids": boot_ids}


def _parse_fast_stats(
    values: Any,
    root_components: dict[str, dict[str, Any]],
) -> None:
    common.require(
        type(values) is list and len(values) == len(root_components),
        "E_FAST_COMPONENT_STATS",
    )
    previous = None
    seen = set()
    for index, value in enumerate(values):
        field = f"fresh.component_stats[{index}]"
        common.exact_keys(
            value,
            {"component_id", "endpoint", "path", "stat"},
            field,
        )
        component_id = common.text(value["component_id"], f"{field}.component_id")
        common.require(component_id in root_components and component_id not in seen, f"E_FAST_COMPONENT: {field}")
        if previous is not None:
            common.require(previous < component_id, "E_FAST_COMPONENT_ORDER")
        previous = component_id
        seen.add(component_id)
        expected = root_components[component_id]
        common.exact(value["endpoint"], expected["endpoint"], f"{field}.endpoint")
        common.exact(value["path"], expected["path"], f"{field}.path")
        common.stat_record(value["stat"], f"{field}.stat")
        common.exact(value["stat"], expected["stat"], f"E_POST_HASH_MUTATION: {component_id}")
    common.exact(seen, set(root_components), "fresh.component_ids")


def validate_fresh(
    fresh: dict[str, Any],
    fresh_raw: bytes,
    contract: dict[str, Any],
    lock_raw: bytes,
    lock_derived: dict[str, Any],
    root_raw: bytes,
    root_derived: dict[str, Any],
    preparation_raw: bytes,
    plan_raw: bytes,
    plan_derived: dict[str, Any],
    acquisition_started_ns: int,
) -> dict[str, Any]:
    del fresh_raw
    common.exact_keys(
        fresh,
        {
            "artifact_root_sha256",
            "component_stats",
            "completed_ns",
            "devices",
            "inventories",
            "phase",
            "phase_id",
            "phase_lock_sha256",
            "preparation_sha256",
            "runtime_bundle_plan_sha256",
            "schema",
            "started_ns",
        },
        "fresh",
    )
    common.exact(fresh["schema"], "s39-cp0-r1-fast-fresh-readiness-v2.4", "fresh.schema")
    common.exact(fresh["phase"], PHASE, "fresh.phase")
    common.exact(fresh["phase_id"], lock_derived["phase_id"], "fresh.phase_id")
    common.exact(fresh["phase_lock_sha256"], common.sha256_bytes(lock_raw), "fresh.lock")
    common.exact(fresh["artifact_root_sha256"], common.sha256_bytes(root_raw), "fresh.root")
    common.exact(
        fresh["preparation_sha256"],
        common.sha256_bytes(preparation_raw),
        "fresh.preparation",
    )
    common.exact(
        fresh["runtime_bundle_plan_sha256"],
        common.sha256_bytes(plan_raw),
        "fresh.runtime_plan",
    )
    started = common.integer(fresh["started_ns"], "fresh.started", 1)
    completed = common.integer(fresh["completed_ns"], "fresh.completed", 1)
    common.require(lock_derived["event_ns"] <= started < completed, "E_FRESH_BEFORE_LOCK")
    common.require(
        completed - started <= contract["gates"]["fast_check_maximum_duration_ns"],
        "E_FAST_CHECK_SLOW",
    )
    common.require(completed <= acquisition_started_ns, "E_FRESH_AFTER_ACQUISITION")
    common.require(
        acquisition_started_ns - completed
        <= contract["gates"]["fresh_snapshot_maximum_age_ns"],
        "E_FRESH_STALE",
    )
    _parse_fast_stats(fresh["component_stats"], root_derived["components"])
    _validate_inventories(fresh["inventories"], plan_derived, "fresh.inventories")
    common.exact(fresh["inventories"], list(root_derived["inventories"].values()), "E_FAST_INVENTORY_DRIFT")
    devices = _validate_device_snapshot(fresh["devices"], contract, "fresh.devices")
    for endpoint in ("cuda", "op12", "op15"):
        common.exact(
            devices[endpoint]["host_boot_id" if endpoint == "cuda" else "boot_id"],
            lock_derived["boot_ids"][endpoint],
            f"E_FRESH_BOOT: {endpoint}",
        )
    return {"completed_ns": completed, "devices": devices}


def _bundle_digest(
    bundle_id: str,
    plan_derived: dict[str, Any],
    root_components: dict[str, dict[str, Any]],
) -> str:
    bundle = plan_derived["bundles"][bundle_id]
    identity = {
        "bundle_id": bundle_id,
        "components": [
            {
                "component_id": component_id,
                "path": root_components[component_id]["path"],
                "sha256": root_components[component_id]["sha256"],
                "stat": root_components[component_id]["stat"],
            }
            for component_id in bundle["required_component_ids"]
        ],
        "endpoint": bundle["endpoint"],
        "launcher_component_id": bundle["launcher_component_id"],
        "process_role": bundle["process_role"],
        "schema": "s39-cp0-r1-runtime-bundle-root-identity-v2.4",
    }
    return common.sha256_bytes(common.canonical_bytes(identity))


def _artifact_digests(
    artifacts: Any,
    required_roles: set[str],
) -> dict[str, dict[str, Any]]:
    common.require(
        type(artifacts) is list and len(artifacts) == len(required_roles),
        "E_ACQUISITION_ARTIFACTS",
    )
    result = {}
    previous = None
    paths = set()
    for index, value in enumerate(artifacts):
        field = f"acquisition.artifacts[{index}]"
        common.exact_keys(value, {"bytes", "path", "role", "sha256"}, field)
        role = common.text(value["role"], f"{field}.role")
        common.require(role in required_roles and role not in result, f"E_ACQUISITION_ROLE: {field}")
        if previous is not None:
            common.require(previous < role, "E_ACQUISITION_ROLE_ORDER")
        previous = role
        path = _relative_path(value["path"], f"{field}.path")
        common.require(path not in paths, f"E_ACQUISITION_PATH_REUSE: {field}")
        paths.add(path)
        common.integer(value["bytes"], f"{field}.bytes", 1)
        result[role] = {
            "bytes": common.integer(value["bytes"], f"{field}.bytes", 1),
            "path": path,
            "sha256": common.digest(value["sha256"], f"{field}.sha256"),
        }
    common.exact(set(result), required_roles, "acquisition.roles")
    return result


def validate_acquisition(
    acquisition: dict[str, Any],
    acquisition_raw: bytes,
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate_raw: bytes,
    root_raw: bytes,
    preparation_raw: bytes,
    lock_raw: bytes,
    lock_derived: dict[str, Any],
    fresh_raw: bytes,
    plan_raw: bytes,
    required_roles: set[str],
) -> dict[str, Any]:
    del acquisition_raw
    common.exact_keys(
        acquisition,
        {
            "artifact_root_sha256",
            "artifacts",
            "candidate_sha256",
            "completed_ns",
            "contract_sha256",
            "fresh_readiness_sha256",
            "phase",
            "phase_id",
            "phase_lock_sha256",
            "preparation_sha256",
            "runtime_bundle_plan_sha256",
            "runtime_identity_sha256",
            "raw_manifest_name",
            "raw_manifest_sha256",
            "raw_predicate_contract_sha256",
            "schema",
            "started_ns",
            "status",
        },
        "acquisition",
    )
    common.exact(acquisition["schema"], "s39-cp0-r1-a-only-acquisition-v2.4", "acquisition.schema")
    common.exact(acquisition["status"], "RAW_CAPTURE_COMPLETE_UNEVALUATED", "acquisition.status")
    common.exact(acquisition["phase"], PHASE, "acquisition.phase")
    common.exact(acquisition["phase_id"], lock_derived["phase_id"], "acquisition.phase_id")
    common.exact(acquisition["contract_sha256"], common.sha256_bytes(contract_raw), "acquisition.contract")
    common.exact(acquisition["candidate_sha256"], common.sha256_bytes(candidate_raw), "acquisition.candidate")
    common.exact(acquisition["artifact_root_sha256"], common.sha256_bytes(root_raw), "acquisition.root")
    common.exact(acquisition["preparation_sha256"], common.sha256_bytes(preparation_raw), "acquisition.preparation")
    common.exact(acquisition["phase_lock_sha256"], common.sha256_bytes(lock_raw), "acquisition.lock")
    common.exact(acquisition["fresh_readiness_sha256"], common.sha256_bytes(fresh_raw), "acquisition.fresh")
    common.exact(
        acquisition["runtime_bundle_plan_sha256"],
        common.sha256_bytes(plan_raw),
        "acquisition.runtime_plan",
    )
    common.digest(acquisition["runtime_identity_sha256"], "acquisition.runtime_identity")
    common.exact(
        acquisition["raw_manifest_name"],
        RAW_MANIFEST_NAME,
        "acquisition.raw_manifest_name",
    )
    common.digest(
        acquisition["raw_manifest_sha256"],
        "acquisition.raw_manifest_sha256",
    )
    common.exact(
        acquisition["raw_predicate_contract_sha256"],
        contract["raw_predicate_contract"]["sha256"],
        "acquisition.raw_predicate_contract_sha256",
    )
    started = common.integer(acquisition["started_ns"], "acquisition.started", 1)
    completed = common.integer(acquisition["completed_ns"], "acquisition.completed", 1)
    common.require(lock_derived["event_ns"] < started < completed, "E_ACQUISITION_INTERVAL")
    artifacts = _artifact_digests(acquisition["artifacts"], required_roles)
    return {
        "artifacts": artifacts,
        "completed_ns": completed,
        "digests": {
            role: value["sha256"] for role, value in artifacts.items()
        },
        "started_ns": started,
    }


def validate_runtime_identity(
    runtime: dict[str, Any],
    runtime_raw: bytes,
    acquisition: dict[str, Any],
    acquisition_derived: dict[str, Any],
    lock_derived: dict[str, Any],
    root_raw: bytes,
    root_derived: dict[str, Any],
    fresh_raw: bytes,
    fresh_derived: dict[str, Any],
    plan_raw: bytes,
    plan_derived: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    common.exact(
        acquisition["runtime_identity_sha256"],
        common.sha256_bytes(runtime_raw),
        "acquisition.runtime_identity",
    )
    common.exact_keys(
        runtime,
        {
            "artifact_root_sha256",
            "completed_ns",
            "fresh_readiness_sha256",
            "phase",
            "phase_id",
            "phone_after",
            "processes",
            "runtime_bundle_plan_sha256",
            "schema",
            "started_ns",
        },
        "runtime",
    )
    common.exact(runtime["schema"], "s39-cp0-r1-runtime-identity-v2.4", "runtime.schema")
    common.exact(runtime["phase"], PHASE, "runtime.phase")
    common.exact(runtime["phase_id"], lock_derived["phase_id"], "runtime.phase_id")
    common.exact(runtime["artifact_root_sha256"], common.sha256_bytes(root_raw), "runtime.root")
    common.exact(runtime["fresh_readiness_sha256"], common.sha256_bytes(fresh_raw), "runtime.fresh")
    common.exact(
        runtime["runtime_bundle_plan_sha256"],
        common.sha256_bytes(plan_raw),
        "runtime.runtime_plan",
    )
    started = common.integer(runtime["started_ns"], "runtime.started", 1)
    completed = common.integer(runtime["completed_ns"], "runtime.completed", 1)
    common.require(
        acquisition_derived["started_ns"] <= started < completed
        <= acquisition_derived["completed_ns"],
        "E_RUNTIME_INTERVAL",
    )
    phone_after = common.exact_keys(
        runtime["phone_after"],
        {"op12", "op15"},
        "runtime.phone_after",
    )
    for phone in ("op12", "op15"):
        value = common.exact_keys(
            phone_after[phone],
            {"boot_id", "observed_ns", "system_swap_used_bytes"},
            f"runtime.phone_after.{phone}",
        )
        common.exact(
            value["boot_id"],
            lock_derived["boot_ids"][phone],
            f"E_SWAP_BOOT: {phone}",
        )
        observed = common.integer(
            value["observed_ns"],
            f"runtime.phone_after.{phone}.observed_ns",
            1,
        )
        common.require(
            started <= observed <= completed,
            f"E_SWAP_AFTER_INTERVAL: {phone}",
        )
        after = common.integer(
            value["system_swap_used_bytes"],
            f"runtime.phone_after.{phone}.system_swap",
        )
        before = fresh_derived["devices"][phone]["system_swap_used_bytes"]
        common.require(after >= before, f"E_SWAP_COUNTER_RESET: {phone}")
        common.exact(
            after - before,
            contract["gates"]["maximum_system_swap_growth_bytes"],
            f"E_SYSTEM_SWAP_GROWTH: {phone}",
        )
    values = runtime["processes"]
    common.require(
        type(values) is list and len(values) == len(REQUIRED_BUNDLES),
        "E_RUNTIME_PROCESSES",
    )
    seen = set()
    previous = None
    for index, value in enumerate(values):
        field = f"runtime.processes[{index}]"
        common.exact_keys(
            value,
            {
                "boot_id",
                "bundle_id",
                "bundle_sha256",
                "endpoint",
                "evidence_role",
                "evidence_sha256",
                "launcher_component_id",
                "launcher_path",
                "loaded_repo_component_ids",
                "model_mapping",
                "observed_ns",
                "pid",
                "process_swap_bytes",
                "start_ticks",
            },
            field,
        )
        bundle_id = common.text(value["bundle_id"], f"{field}.bundle_id")
        common.require(bundle_id in REQUIRED_BUNDLES and bundle_id not in seen, f"E_RUNTIME_BUNDLE: {field}")
        if previous is not None:
            common.require(previous < bundle_id, "E_RUNTIME_PROCESS_ORDER")
        previous = bundle_id
        seen.add(bundle_id)
        bundle = plan_derived["bundles"][bundle_id]
        common.exact(value["endpoint"], bundle["endpoint"], f"{field}.endpoint")
        common.exact(value["boot_id"], lock_derived["boot_ids"][bundle["endpoint"]], f"E_RUNTIME_BOOT: {bundle_id}")
        common.exact(
            value["bundle_sha256"],
            _bundle_digest(bundle_id, plan_derived, root_derived["components"]),
            f"E_RUNTIME_BUNDLE_DIGEST: {bundle_id}",
        )
        common.exact(
            value["launcher_component_id"],
            bundle["launcher_component_id"],
            f"{field}.launcher_component_id",
        )
        launcher = root_derived["components"][bundle["launcher_component_id"]]
        common.exact(value["launcher_path"], launcher["path"], f"{field}.launcher_path")
        common.exact(
            value["loaded_repo_component_ids"],
            bundle["required_component_ids"],
            f"E_RUNTIME_COMPONENTS: {bundle_id}",
        )
        role = PROCESS_EVIDENCE_ROLES[bundle_id]
        common.exact(value["evidence_role"], role, f"{field}.evidence_role")
        common.exact(
            value["evidence_sha256"],
            acquisition_derived["digests"][role],
            f"E_RUNTIME_EVIDENCE: {bundle_id}",
        )
        common.integer(value["pid"], f"{field}.pid", 1)
        common.integer(value["start_ticks"], f"{field}.start_ticks", 1)
        observed = common.integer(value["observed_ns"], f"{field}.observed_ns", 1)
        common.require(started <= observed <= completed, f"E_RUNTIME_OBSERVED: {bundle_id}")
        common.exact(
            value["process_swap_bytes"],
            contract["gates"]["maximum_process_swap_bytes"],
            f"{field}.swap",
        )
        if bundle_id != "cuda_monolithic":
            common.exact(value["model_mapping"], None, f"{field}.model_mapping")
            continue
        mapping = common.exact_keys(
            value["model_mapping"],
            {
                "argv",
                "environment",
                "model_mapping_rows",
                "model_file_type",
                "model_path",
                "model_sha256",
                "other_gguf_mapping_paths",
                "post_stat",
                "pre_stat",
            },
            f"{field}.model_mapping",
        )
        launch = plan_derived["cuda_monolithic_launch"]
        model_component = root_derived["components"]["model.cuda"]
        common.exact(mapping["argv"], launch["command"], "E_CUDA_MONOLITHIC_LIVE_ARGV")
        common.exact(
            mapping["environment"],
            launch["env"],
            "E_CUDA_MONOLITHIC_LIVE_ENV",
        )
        common.exact(mapping["model_path"], model_component["path"], "E_CUDA_MONOLITHIC_MODEL_PATH")
        common.exact(mapping["model_sha256"], model_component["sha256"], "E_CUDA_MONOLITHIC_MODEL_SHA")
        common.exact(
            mapping["model_file_type"],
            contract["cuda_monolithic_identity"]["expected_file_type"],
            "E_CUDA_MONOLITHIC_FILE_TYPE",
        )
        common.stat_record(mapping["pre_stat"], f"{field}.model_mapping.pre_stat")
        common.stat_record(mapping["post_stat"], f"{field}.model_mapping.post_stat")
        common.exact(mapping["pre_stat"], model_component["stat"], "E_CUDA_MONOLITHIC_PRE_STAT")
        common.exact(mapping["post_stat"], model_component["stat"], "E_CUDA_MONOLITHIC_POST_STAT")
        common.exact(
            mapping["other_gguf_mapping_paths"],
            [],
            "E_CUDA_MONOLITHIC_OTHER_GGUF",
        )
        mapping_rows = mapping["model_mapping_rows"]
        expected_rows = contract["cuda_monolithic_identity"]["maps_exact_rows"]
        common.require(
            type(mapping_rows) is list
            and len(mapping_rows) == len(expected_rows),
            "E_CUDA_MONOLITHIC_MAPS_COUNT",
        )
        normalized_rows = []
        seen_rows = set()
        for row_index, row in enumerate(mapping_rows):
            map_field = f"{field}.model_mapping.model_mapping_rows[{row_index}]"
            common.exact_keys(
                row,
                {
                    "address_range",
                    "device_major",
                    "device_minor",
                    "inode",
                    "offset_bytes",
                    "path",
                    "permissions",
                },
                map_field,
            )
            address_match = re.fullmatch(
                r"([0-9a-f]+)-([0-9a-f]+)",
                common.text(
                    row["address_range"],
                    f"{map_field}.address_range",
                ),
            )
            common.require(
                address_match is not None
                and int(address_match.group(1), 16)
                < int(address_match.group(2), 16),
                f"E_CUDA_MONOLITHIC_MAPS_ADDRESS: {row_index}",
            )
            common.exact(
                row["path"],
                model_component["path"],
                f"E_CUDA_MONOLITHIC_MAPS_PATH: {row_index}",
            )
            common.exact(
                row["device_major"],
                os.major(model_component["stat"]["device_id"]),
                f"E_CUDA_MONOLITHIC_MAPS_DEVICE_MAJOR: {row_index}",
            )
            common.exact(
                row["device_minor"],
                os.minor(model_component["stat"]["device_id"]),
                f"E_CUDA_MONOLITHIC_MAPS_DEVICE_MINOR: {row_index}",
            )
            common.exact(
                row["inode"],
                model_component["stat"]["inode"],
                f"E_CUDA_MONOLITHIC_MAPS_INODE: {row_index}",
            )
            identity = (
                row["device_major"],
                row["device_minor"],
                row["inode"],
                common.integer(
                    row["offset_bytes"],
                    f"{map_field}.offset_bytes",
                ),
                common.text(row["permissions"], f"{map_field}.permissions"),
                row["path"],
            )
            common.require(
                identity not in seen_rows,
                f"E_CUDA_MONOLITHIC_MAPS_DUPLICATE: {row_index}",
            )
            seen_rows.add(identity)
            normalized_rows.append(
                {
                    "offset_bytes": identity[3],
                    "permissions": identity[4],
                }
            )
        common.exact(
            normalized_rows,
            expected_rows,
            "E_CUDA_MONOLITHIC_MAPS_ROWS",
        )
    common.exact(seen, set(REQUIRED_BUNDLES), "runtime.bundle_ids")
    return {"completed_ns": completed}


def validate_chain(
    *,
    contract_path: Path,
    candidate_path: Path,
    runtime_plan_path: Path,
    tokenizer_plan_path: Path,
    token_history_path: Path,
    artifact_root_path: Path,
    preparation_path: Path,
    phase_lock_path: Path,
    fresh_path: Path,
    runtime_identity_path: Path,
    acquisition_path: Path,
) -> dict[str, Any]:
    contract, contract_raw, candidate, candidate_raw = validate_inputs(
        contract_path,
        candidate_path,
    )
    plan, plan_raw = common.read_canonical(runtime_plan_path)
    plan_derived = validate_runtime_plan(
        plan,
        contract,
        contract_raw,
        candidate_raw,
    )
    tokenizer_plan, tokenizer_plan_raw = common.read_canonical(
        tokenizer_plan_path
    )
    validate_tokenizer_plan(
        tokenizer_plan,
        tokenizer_plan_raw,
        contract,
        candidate,
        plan_derived,
    )
    history, history_raw = common.read_canonical(token_history_path)
    validate_token_history(
        history,
        history_raw,
        contract,
        candidate,
        plan_derived,
    )
    root, root_raw = common.read_canonical(artifact_root_path)
    root_derived = validate_artifact_root(
        root,
        contract,
        contract_raw,
        candidate,
        candidate_raw,
        plan_raw,
        plan_derived,
        history_raw,
        tokenizer_plan_raw,
    )
    preparation, preparation_raw = common.read_canonical(preparation_path)
    preparation_derived = validate_preparation(
        preparation,
        preparation_raw,
        contract,
        root_raw,
        root_derived["completed_ns"],
        plan_raw,
    )
    lock, lock_raw = common.read_canonical(phase_lock_path)
    lock_derived = validate_phase_lock(
        lock,
        lock_raw,
        contract,
        contract_raw,
        candidate_raw,
        root_raw,
        root_derived["completed_ns"],
        preparation_raw,
        preparation_derived,
        plan_raw,
    )
    acquisition, acquisition_raw = common.read_canonical(acquisition_path)
    raw_predicate_contract, _, _ = raw_predicate_inputs(contract)
    required_roles = set(
        raw_predicate_contract["phase_protocol"]["phase_roles"][PHASE]
    )
    dynamic_roles = (
        required_roles - PRE_ACQUISITION_ROLES
    ) | CAPTURE_RECEIPT_ROLES
    acquisition_started = common.integer(
        acquisition.get("started_ns"),
        "acquisition.started",
        1,
    )
    fresh, fresh_raw = common.read_canonical(fresh_path)
    fresh_derived = validate_fresh(
        fresh,
        fresh_raw,
        contract,
        lock_raw,
        lock_derived,
        root_raw,
        root_derived,
        preparation_raw,
        plan_raw,
        plan_derived,
        acquisition_started,
    )
    acquisition_derived = validate_acquisition(
        acquisition,
        acquisition_raw,
        contract,
        contract_raw,
        candidate_raw,
        root_raw,
        preparation_raw,
        lock_raw,
        lock_derived,
        fresh_raw,
        plan_raw,
        dynamic_roles,
    )
    runtime, runtime_raw = common.read_canonical(runtime_identity_path)
    runtime_derived = validate_runtime_identity(
        runtime,
        runtime_raw,
        acquisition,
        acquisition_derived,
        lock_derived,
        root_raw,
        root_derived,
        fresh_raw,
        fresh_derived,
        plan_raw,
        plan_derived,
        contract,
    )
    return {
        "artifact_root_sha256": common.sha256_bytes(root_raw),
        "fresh_readiness_sha256": common.sha256_bytes(fresh_raw),
        "phase": PHASE,
        "phase_id": lock_derived["phase_id"],
        "runtime_identity_sha256": common.sha256_bytes(runtime_raw),
        "schema": "s39-cp0-r1-v2.4-chain-result-v1",
        "status": "V2_4_READINESS_CHAIN_PASS_RAW_QUALIFICATION_NOT_EVALUATED",
        "timing": {
            "acquisition_completed_ns": acquisition_derived["completed_ns"],
            "artifact_root_completed_ns": root_derived["completed_ns"],
            "fresh_completed_ns": fresh_derived["completed_ns"],
            "phase_lock_ns": lock_derived["event_ns"],
            "runtime_completed_ns": runtime_derived["completed_ns"],
        },
    }


def evaluate_raw_predicates(
    bundle_root: Path,
    contract: dict[str, Any],
    candidate: dict[str, Any],
    candidate_raw: bytes,
    history: dict[str, Any],
) -> dict[str, Any]:
    raw_contract, raw_contract_raw, raw_parent = raw_predicate_inputs(contract)
    helpers = _verified_helpers(contract)
    v21 = helpers["v21"]
    v22 = helpers["v22"]

    helper_errors = tuple(
        {
            error_type
            for module in helpers.values()
            if isinstance(
                error_type := getattr(module, "EvidenceError", None),
                type,
            )
        }
    )
    try:
        frozen_corpus = v22.load_frozen_corpus(raw_contract)
        (
            loaded_manifest,
            loaded_manifest_raw,
            rows_by_role,
            artifact_digests,
        ) = v21.load_bundle(
            bundle_root,
            RAW_MANIFEST_NAME,
            raw_contract,
            raw_contract_raw,
            candidate_raw,
        )
        projected_executions = validate_path_matched_history(
            rows_by_role,
            history,
        )
        return evaluate_v24_model_phase(
            raw_contract,
            raw_contract_raw,
            candidate,
            candidate_raw,
            raw_parent,
            frozen_corpus,
            loaded_manifest,
            loaded_manifest_raw,
            rows_by_role,
            artifact_digests,
            projected_executions,
            helpers,
        )
    except helper_errors as error:
        raise common.EvidenceError(
            f"E_VERIFIED_HELPER_REFUSAL: {error}"
        ) from error


def validate_path_matched_history(
    rows_by_role: dict[str, list[dict[str, Any]]],
    history: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    group = history["mechanics_b8"]
    expected_calls = [
        {
            "call_index": partition["call_index"],
            "n_seqs": len({row["seq_id"] for row in partition["rows"]}),
            "n_tokens": len(partition["rows"]),
            "phase": "prefill",
            "positions": [row["position"] for row in partition["rows"]],
            "request_ids": [row["request_id"] for row in partition["rows"]],
            "seq_ids": [row["seq_id"] for row in partition["rows"]],
        }
        for partition in group["prefill_partitions"]
    ] + [
        {
            "call_index": call["call_index"],
            "n_seqs": len({row["seq_id"] for row in call["rows"]}),
            "n_tokens": len(call["rows"]),
            "phase": "decode",
            "positions": [row["position"] for row in call["rows"]],
            "request_ids": [row["request_id"] for row in call["rows"]],
            "seq_ids": [row["seq_id"] for row in call["rows"]],
        }
        for call in group["decode_calls"]
    ]
    requests = {
        value["item_index"]: value
        for value in history["requests"]
    }
    roles = (
        f"model.{MODEL_ID}.mechanics.phone",
        f"model.{MODEL_ID}.oracle.cuda_route",
        f"model.{MODEL_ID}.oracle.cuda_monolithic",
    )
    projected: dict[str, list[dict[str, Any]]] = {}
    for role in roles:
        rows = rows_by_role[role]
        meta = [row for row in rows if row.get("kind") == "meta"]
        common.require(len(meta) == 1, f"E_PATH_HISTORY_META: {role}")
        common.exact(
            meta[0].get("call_shapes"),
            expected_calls,
            f"E_PATH_HISTORY_CALLS: {role}",
        )
        common.exact(
            sum(
                call["n_tokens"]
                for call in expected_calls
                if call["phase"] == "prefill"
            ),
            sum(len(requests[item]["token_ids"]) for item in group["item_indices"]),
            f"E_PATH_HISTORY_PREFILL_ROWS: {role}",
        )
        common.exact(
            sum(
                call["n_tokens"]
                for call in expected_calls
                if call["phase"] == "decode"
            ),
            7 * 8,
            f"E_PATH_HISTORY_DECODE_ROWS: {role}",
        )
        request_rows = {
            row.get("request_id"): row
            for row in rows
            if row.get("kind") == "request"
        }
        common.exact(
            set(request_rows),
            set(range(1, 9)),
            f"E_PATH_HISTORY_REQUESTS: {role}",
        )
        for local_id, item_index in enumerate(group["item_indices"]):
            request = request_rows[local_id + 1]
            tokens = requests[item_index]["token_ids"]
            common.exact(
                request.get("input_tokens"),
                tokens,
                f"E_PATH_HISTORY_TOKENS: {role}[{local_id}]",
            )
            common.exact(
                request.get("positions"),
                list(range(len(tokens))),
                f"E_PATH_HISTORY_POSITIONS: {role}[{local_id}]",
            )
            common.require(
                type(request.get("continuation_tokens")) is list
                and len(request["continuation_tokens"]) == 8,
                f"E_PATH_HISTORY_CONTINUATIONS: {role}[{local_id}]",
            )
        projected_rows = copy.deepcopy(rows)
        projected_rows[0]["call_shapes"] = [
            {
                key: call[key]
                for key in ("call_index", "n_seqs", "n_tokens", "phase")
            }
            for call in expected_calls
        ]
        for row in projected_rows[1:]:
            common.require(row.get("kind") == "request", f"E_PATH_HISTORY_ROW: {role}")
            row["request_id"] -= 1
        projected[role] = projected_rows
    return projected


def _validate_cuda_placement_certificate(
    value: Any,
    runtime_process: dict[str, Any],
) -> None:
    common.exact_keys(
        value,
        {
            "compute_by_buffer_type",
            "compute_by_op",
            "compute_by_op_and_buffer",
            "compute_nodes",
            "copy_by_buffer_type",
            "copy_nodes",
            "layer_end",
            "layer_start",
            "metadata_nodes",
            "missing_buffer_compute_nodes",
            "mode",
            "n_layer",
            "pid",
            "role",
            "run_rc",
            "schema",
            "status",
        },
        "cuda_receipt.placement",
    )
    common.exact(value["schema"], "layersplit-scheduled-placement-v2", "E_PLACEMENT_SCHEMA")
    for key in ("role", "mode"):
        common.exact(value[key], "monov3", f"E_PLACEMENT_{key.upper()}")
    for key, expected in (
        ("layer_start", 0),
        ("layer_end", 40),
        ("n_layer", 40),
        ("pid", runtime_process["pid"]),
        ("run_rc", 0),
        ("missing_buffer_compute_nodes", 0),
        ("copy_nodes", 0),
    ):
        common.exact(value[key], expected, f"E_PLACEMENT_{key.upper()}")
    common.exact(
        value["status"],
        "SCHEDULED_PLACEMENT_OK",
        "E_PLACEMENT_STATUS",
    )
    compute_nodes = common.integer(
        value["compute_nodes"],
        "cuda_receipt.placement.compute_nodes",
        1,
    )
    common.integer(
        value["metadata_nodes"],
        "cuda_receipt.placement.metadata_nodes",
    )
    common.exact(value["copy_by_buffer_type"], {}, "E_PLACEMENT_COPY_BUFFERS")
    buffers = value["compute_by_buffer_type"]
    common.require(
        type(buffers) is dict
        and bool(buffers)
        and set(buffers) <= {"CUDA0", "CUDA_Host"}
        and common.integer(buffers.get("CUDA0"), "placement.CUDA0", 1) > 0
        and all(type(count) is int and count > 0 for count in buffers.values())
        and sum(buffers.values()) == compute_nodes,
        "E_PLACEMENT_BUFFER_TOTAL",
    )
    by_op = value["compute_by_op"]
    nested = value["compute_by_op_and_buffer"]
    common.require(
        type(by_op) is dict
        and bool(by_op)
        and type(nested) is dict
        and set(nested) == set(by_op)
        and all(
            type(operation) is str
            and bool(operation)
            and operation.isascii()
            and type(count) is int
            and count > 0
            for operation, count in by_op.items()
        )
        and sum(by_op.values()) == compute_nodes,
        "E_PLACEMENT_OPS",
    )
    derived_buffers: dict[str, int] = {}
    for operation, count in by_op.items():
        operation_buffers = nested[operation]
        common.require(
            type(operation_buffers) is dict
            and bool(operation_buffers)
            and all(
                backend in {"CUDA0", "CUDA_Host"}
                and type(backend_count) is int
                and backend_count > 0
                for backend, backend_count in operation_buffers.items()
            )
            and sum(operation_buffers.values()) == count,
            f"E_PLACEMENT_OP_TOTAL: {operation}",
        )
        if operation == "GET_ROWS":
            common.require(
                set(operation_buffers) <= {"CUDA0", "CUDA_Host"},
                "E_PLACEMENT_GET_ROWS",
            )
        else:
            common.exact(
                set(operation_buffers),
                {"CUDA0"},
                f"E_PLACEMENT_HOST_OP: {operation}",
            )
        for backend, count_for_backend in operation_buffers.items():
            derived_buffers[backend] = (
                derived_buffers.get(backend, 0) + count_for_backend
            )
    common.exact(derived_buffers, buffers, "E_PLACEMENT_BUFFER_REDUCTION")


def _validate_protocol_identity(
    value: Any,
    contract: dict[str, Any],
    model_sha256: str,
) -> None:
    common.exact(
        value,
        {
            "capabilities": 0x3F,
            "file_type": contract["cuda_monolithic_identity"][
                "expected_file_type"
            ],
            "layer_end": 40,
            "layer_start": 0,
            "max_streams": 8,
            "model_sha256": model_sha256,
            "n_batch": 64,
            "n_ctx_seq": 512,
            "n_embd": 5120,
            "n_layer": 40,
            "n_ubatch": 64,
            "schema": "layersplit-stage-v3-identity-v1",
            "stage_identity_version": 1,
            "stage_protocol_version": 3,
        },
        "E_CUDA_PROTOCOL_IDENTITY",
    )


def _validate_memory_certificate(
    value: Any,
    runtime_process: dict[str, Any],
    placement: dict[str, Any],
    cuda_memory_rows: list[dict[str, Any]] | None = None,
) -> None:
    common.exact_keys(
        value,
        {
            "compute_buffer_bytes",
            "host_compute_buffer_bytes",
            "host_context_buffer_bytes",
            "host_model_buffer_bytes",
            "kv_buffer_bytes",
            "model_buffer_bytes",
            "pid",
            "role",
            "schema",
        },
        "cuda_receipt.memory",
    )
    common.exact(value["schema"], "layersplit-memory-breakdown-v1", "E_MEMORY_SCHEMA")
    common.exact(value["role"], "monov3", "E_MEMORY_ROLE")
    common.exact(value["pid"], runtime_process["pid"], "E_MEMORY_PID")
    for key in (
        "compute_buffer_bytes",
        "host_compute_buffer_bytes",
        "host_context_buffer_bytes",
        "host_model_buffer_bytes",
        "kv_buffer_bytes",
        "model_buffer_bytes",
    ):
        common.integer(value[key], f"cuda_receipt.memory.{key}")
    common.require(
        value["model_buffer_bytes"] > 0
        and value["kv_buffer_bytes"] > 0,
        "E_MEMORY_DEVICE_BYTES",
    )
    if cuda_memory_rows is None:
        return
    ready = [row for row in cuda_memory_rows if row.get("kind") == "ready"]
    common.require(len(ready) == 1, "E_MEMORY_READY_ROW")
    common.exact(
        ready[0].get("model_buffer_bytes"),
        value["model_buffer_bytes"],
        "E_MEMORY_MODEL_LINK",
    )
    common.exact(
        ready[0].get("kv_buffer_bytes"),
        value["kv_buffer_bytes"],
        "E_MEMORY_KV_LINK",
    )
    common.exact(
        ready[0].get("placement_compute_nodes"),
        placement["compute_nodes"],
        "E_MEMORY_PLACEMENT_LINK",
    )
    common.exact(
        ready[0].get("process_pid"),
        runtime_process["pid"],
        "E_MEMORY_PROCESS_LINK",
    )
    device_bytes = (
        value["model_buffer_bytes"]
        + value["kv_buffer_bytes"]
        + value["compute_buffer_bytes"]
    )
    common.require(
        ready[0].get("process_used_bytes", -1) >= device_bytes,
        "E_MEMORY_PROCESS_BYTES",
    )


def _read_bound_artifact(
    bundle_root: Path,
    acquisition: dict[str, Any],
    role: str,
) -> tuple[dict[str, Any], bytes]:
    matches = [
        value
        for value in acquisition["artifacts"]
        if value["role"] == role
    ]
    common.require(len(matches) == 1, f"E_CAPTURE_RECEIPT_ROLE: {role}")
    record = matches[0]
    relative = _relative_path(record["path"], f"capture.{role}.path")
    path = (bundle_root / relative).resolve()
    root = bundle_root.resolve()
    common.require(path.is_relative_to(root), f"E_CAPTURE_RECEIPT_PATH: {role}")
    value, raw = common.read_canonical(path)
    common.exact(len(raw), record["bytes"], f"E_CAPTURE_RECEIPT_BYTES: {role}")
    common.exact(
        common.sha256_bytes(raw),
        record["sha256"],
        f"E_CAPTURE_RECEIPT_SHA256: {role}",
    )
    return value, raw


def _read_stable_regular(path: Path, field: str) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise common.EvidenceError(f"E_CAPTURE_OPEN: {field}: {error}") from error
    try:
        before = os.fstat(descriptor)
        common.require(stat.S_ISREG(before.st_mode), f"E_CAPTURE_FILE_TYPE: {field}")
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )
    common.exact(identity(after), identity(before), f"E_CAPTURE_TOCTOU: {field}")
    common.exact(len(raw), before.st_size, f"E_CAPTURE_READ_SIZE: {field}")
    return bytes(raw)


def _read_stable_regular_with_stat(
    path: Path,
    field: str,
) -> tuple[bytes, dict[str, int]]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise common.EvidenceError(f"E_CAPTURE_OPEN: {field}: {error}") from error
    try:
        before = os.fstat(descriptor)
        common.require(stat.S_ISREG(before.st_mode), f"E_CAPTURE_FILE_TYPE: {field}")
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )
    common.exact(identity(after), identity(before), f"E_CAPTURE_TOCTOU: {field}")
    common.exact(len(raw), before.st_size, f"E_CAPTURE_READ_SIZE: {field}")
    return bytes(raw), {
        "ctime_ns": before.st_ctime_ns,
        "device_id": before.st_dev,
        "inode": before.st_ino,
        "mode": before.st_mode,
        "mtime_ns": before.st_mtime_ns,
        "size": before.st_size,
    }


def _validate_identity_attestation(
    value: Any,
    field: str,
) -> dict[str, Any]:
    value = common.exact_keys(
        value,
        {
            "bound_root_sha256",
            "identity_binding_receipt_sha256",
            "schema",
            "status",
        },
        field,
    )
    common.exact(
        value["schema"],
        "s39-cp0-r1-v24-identity-binding-attestation-v1",
        f"{field}.schema",
    )
    common.exact(
        value["status"],
        "POST_REBOOT_IDENTITY_BINDING_PASS",
        f"{field}.status",
    )
    common.digest(
        value["bound_root_sha256"],
        f"{field}.bound_root_sha256",
    )
    common.digest(
        value["identity_binding_receipt_sha256"],
        f"{field}.identity_binding_receipt_sha256",
    )
    return value


def validate_orchestration_provenance(
    *,
    orchestration_plan_path: Path,
    bundle_root: Path,
    contract_path: Path,
    candidate_path: Path,
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate_raw: bytes,
) -> dict[str, Any]:
    plan_raw = _read_stable_regular(
        orchestration_plan_path,
        "orchestration.plan",
    )
    plan = common.parse_json(plan_raw, "orchestration.plan")
    common.require(type(plan) is dict, "E_ORCHESTRATION_PLAN_TYPE")
    common.exact(
        common.canonical_bytes(plan),
        plan_raw,
        "E_ORCHESTRATION_PLAN_CANONICAL",
    )
    common.exact_keys(
        plan,
        {
            "inputs",
            "mechanism_commands_sha256",
            "model_id",
            "model_sha256",
            "phase",
            "phase_id",
            "python",
            "run_root",
            "schema",
            "stages",
        },
        "orchestration.plan",
    )
    common.exact(
        plan["schema"],
        ORCHESTRATION_PLAN_SCHEMA,
        "orchestration.plan.schema",
    )
    common.exact(plan["phase"], PHASE, "orchestration.plan.phase")
    common.exact(plan["model_id"], MODEL_ID, "orchestration.plan.model")
    phase_id = common.text(plan["phase_id"], "orchestration.plan.phase_id")
    run_root = Path(
        common.absolute_path(plan["run_root"], "orchestration.plan.run_root")
    ).resolve()
    common.exact(
        bundle_root.resolve(),
        run_root / "raw-bundle",
        "E_ORCHESTRATION_BUNDLE_ROOT",
    )
    inputs = plan["inputs"]
    common.require(type(inputs) is dict, "E_ORCHESTRATION_INPUTS")
    for name, path, raw in (
        ("contract", contract_path.resolve(), contract_raw),
        ("candidate", candidate_path.resolve(), candidate_raw),
    ):
        record = common.exact_keys(
            inputs.get(name),
            {"bytes", "path", "sha256", "stat"},
            f"orchestration.inputs.{name}",
        )
        common.exact(
            Path(common.absolute_path(record["path"], f"orchestration.inputs.{name}.path")),
            path,
            f"E_ORCHESTRATION_INPUT_PATH: {name}",
        )
        common.exact(record["bytes"], len(raw), f"E_ORCHESTRATION_INPUT_BYTES: {name}")
        common.exact(
            record["sha256"],
            common.sha256_bytes(raw),
            f"E_ORCHESTRATION_INPUT_SHA256: {name}",
        )

    requirements = common.exact_keys(
        contract["orchestration_requirements"],
        {"source_programs", "stage_support", "support"},
        "orchestration.requirements",
    )
    source_pins = common.exact_keys(
        requirements["source_programs"],
        set(ORCHESTRATION_SOURCE_STAGES),
        "orchestration.source_programs",
    )
    support_pins = requirements["support"]
    common.require(
        type(support_pins) is dict and bool(support_pins),
        "E_ORCHESTRATION_SUPPORT",
    )
    stage_support = common.exact_keys(
        requirements["stage_support"],
        set(ORCHESTRATION_SOURCE_STAGES),
        "orchestration.stage_support",
    )
    stages = plan["stages"]
    common.require(type(stages) is dict, "E_ORCHESTRATION_STAGES")
    contract_root = contract_path.resolve().parent.parent

    def verify_source(
        record: Any,
        pin: Any,
        expected_path: Path,
        captured_path: Path,
        field: str,
    ) -> str:
        record = common.exact_keys(
            record,
            {"bytes", "path", "sha256", "stat"},
            field,
        )
        pin = common.exact_keys(pin, {"bytes", "path", "sha256"}, f"{field}.pin")
        common.integer(pin["bytes"], f"{field}.pin.bytes", 1)
        common.digest(pin["sha256"], f"{field}.pin.sha256")
        common.stat_record(record["stat"], f"{field}.stat")
        source_path = Path(common.absolute_path(record["path"], f"{field}.path"))
        common.exact(source_path, expected_path, f"E_ORCHESTRATION_SOURCE_PATH: {field}")
        raw, source_stat = _read_stable_regular_with_stat(source_path, field)
        common.exact(record["stat"], source_stat, f"E_ORCHESTRATION_SOURCE_STAT: {field}")
        common.exact(record["bytes"], pin["bytes"], f"E_ORCHESTRATION_SOURCE_BYTES: {field}")
        common.exact(len(raw), pin["bytes"], f"E_ORCHESTRATION_LIVE_BYTES: {field}")
        common.exact(record["sha256"], pin["sha256"], f"E_ORCHESTRATION_SOURCE_SHA256: {field}")
        common.exact(
            common.sha256_bytes(raw),
            pin["sha256"],
            f"E_ORCHESTRATION_LIVE_SHA256: {field}",
        )
        captured_raw = _read_stable_regular(captured_path, f"{field}.captured")
        common.exact(captured_raw, raw, f"E_ORCHESTRATION_EXECUTED_SOURCE: {field}")
        return str(source_path)

    source_digests = {}
    stage_intervals = {}
    identity_attestation = None
    previous_completed = None
    for stage in ORCHESTRATION_SOURCE_STAGES:
        field = f"orchestration.stages.{stage}"
        stage_plan = stages.get(stage)
        common.require(type(stage_plan) is dict, f"E_ORCHESTRATION_STAGE: {stage}")
        pin = source_pins[stage]
        source_relative = _relative_path(pin["path"], f"{field}.pin.path")
        source_path = (contract_root / source_relative).resolve()
        source_argv = verify_source(
            stage_plan.get("entrypoint"),
            pin,
            source_path,
            run_root / "executed" / stage / "entrypoint.py",
            f"{field}.entrypoint",
        )
        source_digests[stage] = pin["sha256"]

        support_names = stage_support[stage]
        common.require(
            type(support_names) is list
            and bool(support_names)
            and support_names == sorted(set(support_names))
            and set(support_names) <= set(support_pins),
            f"E_ORCHESTRATION_SUPPORT_NAMES: {stage}",
        )
        support_records = stage_plan.get("support_files")
        common.require(type(support_records) is list, f"E_ORCHESTRATION_SUPPORT: {stage}")
        common.exact(
            len(support_records),
            len(support_names),
            f"E_ORCHESTRATION_SUPPORT_COUNT: {stage}",
        )
        expected_support = sorted(
            ((name, support_pins[name]) for name in support_names),
            key=lambda item: item[1]["path"],
        )
        for index, ((name, support_pin), support_record) in enumerate(
            zip(expected_support, support_records)
        ):
            support_relative = _relative_path(
                support_pin["path"],
                f"{field}.support.{name}.pin.path",
            )
            verify_source(
                support_record,
                support_pin,
                (contract_root / support_relative).resolve(),
                run_root
                / "executed"
                / stage
                / "support"
                / f"{index:03d}-{Path(support_relative).name}",
                f"{field}.support.{name}",
            )

        receipt_raw = _read_stable_regular(
            run_root / "receipts" / stage / "receipt.json",
            f"{field}.receipt",
        )
        receipt = common.parse_json(receipt_raw, f"{field}.receipt")
        common.require(type(receipt) is dict, f"E_ORCHESTRATION_RECEIPT: {stage}")
        common.exact(
            common.canonical_bytes(receipt),
            receipt_raw,
            f"E_ORCHESTRATION_RECEIPT_CANONICAL: {stage}",
        )
        common.exact_keys(
            receipt,
            {
                "argv",
                "completed_ns",
                "returncode",
                "schema",
                "stage",
                "started_ns",
            },
            f"{field}.receipt",
        )
        common.exact(
            receipt["schema"],
            ORCHESTRATION_RECEIPT_SCHEMA,
            f"{field}.receipt.schema",
        )
        common.exact(receipt["stage"], stage, f"{field}.receipt.stage")
        common.exact(receipt["returncode"], 0, f"E_ORCHESTRATION_STAGE_EXIT: {stage}")
        started = common.integer(receipt["started_ns"], f"{field}.receipt.started", 1)
        completed = common.integer(
            receipt["completed_ns"],
            f"{field}.receipt.completed",
            1,
        )
        common.require(started < completed, f"E_ORCHESTRATION_STAGE_INTERVAL: {stage}")
        if previous_completed is not None:
            common.require(
                previous_completed <= started,
                f"E_ORCHESTRATION_STAGE_ORDER: {stage}",
            )
        previous_completed = completed
        stage_intervals[stage] = {
            "completed_ns": completed,
            "started_ns": started,
        }
        argv = receipt["argv"]
        common.require(
            type(argv) is list and len(argv) >= 3,
            f"E_ORCHESTRATION_STAGE_ARGV: {stage}",
        )
        python_record = plan["python"]
        common.require(type(python_record) is dict, "E_ORCHESTRATION_PYTHON")
        common.exact(
            argv[0],
            python_record.get("path"),
            f"E_ORCHESTRATION_STAGE_PYTHON: {stage}",
        )
        common.exact(argv[1], "-B", f"E_ORCHESTRATION_STAGE_NO_BYTECODE: {stage}")
        common.exact(argv[2], source_argv, f"E_ORCHESTRATION_STAGE_SOURCE: {stage}")
        stdout = _read_stable_regular(
            run_root / "receipts" / stage / "stdout",
            f"{field}.stdout",
        )
        if stage == "identity_binding":
            identity_attestation = common.parse_json(
                stdout,
                f"{field}.stdout",
            )
            common.require(
                type(identity_attestation) is dict,
                "E_IDENTITY_ATTESTATION_TYPE",
            )
            common.exact(
                common.canonical_bytes(identity_attestation),
                stdout,
                "E_IDENTITY_ATTESTATION_CANONICAL",
            )
            identity_attestation = _validate_identity_attestation(
                identity_attestation,
                "identity.attestation",
            )
        elif stage == "readiness_projection":
            projection = common.parse_json(stdout, f"{field}.stdout")
            common.require(type(projection) is dict, "E_ORCHESTRATION_PROJECTION")
            common.exact(
                common.canonical_bytes(projection),
                stdout,
                "E_ORCHESTRATION_PROJECTION_CANONICAL",
            )
            common.exact(
                projection.get("schema"),
                "s39-cp0-r1-v24-pre-acquisition-projection-v1",
                "E_ORCHESTRATION_PROJECTION_SCHEMA",
            )
            common.exact(
                projection.get("status"),
                "V2_4_PRE_ACQUISITION_READINESS_PASS",
                "E_ORCHESTRATION_PROJECTION_STATUS",
            )
        else:
            common.exact(stdout, b"", f"E_ORCHESTRATION_STDOUT: {stage}")

    return {
        "identity_attestation": identity_attestation,
        "inputs": {
            "prospective_root": plan["inputs"]["prospective_root"],
        },
        "mechanism_commands_sha256": common.digest(
            plan["mechanism_commands_sha256"],
            "orchestration.plan.mechanism_commands_sha256",
        ),
        "phase_id": phase_id,
        "plan_sha256": common.sha256_bytes(plan_raw),
        "run_root": str(run_root),
        "schema": "s39-cp0-r1-v24-orchestration-provenance-v1",
        "source_sha256s": source_digests,
        "stage_intervals": stage_intervals,
        "status": "V2_4_ORCHESTRATION_PROVENANCE_PASS",
    }


def _read_stable_canonical(
    path: Path,
    field: str,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_stable_regular(path, field)
    value = common.parse_json(raw, field)
    common.require(type(value) is dict, f"E_TYPE: {field}")
    common.exact(common.canonical_bytes(value), raw, f"E_CANONICAL: {field}")
    return value, raw


def _read_identity_record(
    record: Any,
    field: str,
    expected_path: Path | None = None,
) -> tuple[dict[str, Any], bytes, Path]:
    record = common.exact_keys(record, {"bytes", "path", "sha256"}, field)
    path = Path(common.absolute_path(record["path"], f"{field}.path"))
    if expected_path is not None:
        common.exact(path, expected_path, f"E_IDENTITY_PATH: {field}")
    value, raw = _read_stable_canonical(path, field)
    common.exact(record["bytes"], len(raw), f"E_IDENTITY_BYTES: {field}")
    common.exact(
        record["sha256"],
        common.sha256_bytes(raw),
        f"E_IDENTITY_SHA256: {field}",
    )
    return value, raw, path


def _reject_unbound_identity(value: Any, field: str) -> None:
    sentinels = {
        *UNBOUND_BOOT_IDS.values(),
        *(
            item
            for values in UNBOUND_PHONE_NETWORK.values()
            for item in values.values()
        ),
    }
    if type(value) is str:
        common.require(
            not any(sentinel in value for sentinel in sentinels),
            f"E_SENTINEL_LEAKAGE: {field}",
        )
    elif type(value) is list:
        for index, item in enumerate(value):
            _reject_unbound_identity(item, f"{field}[{index}]")
    elif type(value) is dict:
        for key, item in value.items():
            _reject_unbound_identity(item, f"{field}.{key}")


def _replace_identity_strings(
    value: Any,
    replacements: dict[str, str],
) -> Any:
    if type(value) is str:
        if value in replacements:
            return replacements[value]
        for old, new in replacements.items():
            prefix = f"{old}:"
            if value.startswith(prefix):
                port = value[len(prefix):]
                if port.isdecimal() and 0 < int(port) <= 65535:
                    return f"{new}:{port}"
        return value
    if type(value) is list:
        return [_replace_identity_strings(item, replacements) for item in value]
    if type(value) is dict:
        return {
            key: _replace_identity_strings(item, replacements)
            for key, item in value.items()
        }
    return value


def _rebind_structured_identity_argv(
    value: Any,
    replacements: dict[str, str],
    field: str = "phone",
) -> None:
    if type(value) is dict:
        for key, item in value.items():
            _rebind_structured_identity_argv(
                item,
                replacements,
                f"{field}.{key}",
            )
        return
    if type(value) is not list:
        return
    for index, item in enumerate(value):
        _rebind_structured_identity_argv(
            item,
            replacements,
            f"{field}[{index}]",
        )
    if not value or not all(type(item) is str for item in value):
        return
    for option in ("--plan-json", "--expected-argv-json"):
        count = value.count(option)
        common.require(
            count in (0, 1),
            f"E_STRUCTURED_ARGV_OPTION: {field}.{option}",
        )
        if count == 0:
            continue
        index = value.index(option) + 1
        common.require(
            index < len(value),
            f"E_STRUCTURED_ARGV_OPTION: {field}.{option}",
        )
        try:
            parsed = json.loads(value[index])
        except json.JSONDecodeError as error:
            raise common.EvidenceError(
                f"E_STRUCTURED_ARGV_JSON: {field}.{option}"
            ) from error
        rebound = _replace_identity_strings(parsed, replacements)
        _rebind_structured_identity_argv(
            rebound,
            replacements,
            f"{field}.{option}",
        )
        value[index] = json.dumps(
            rebound,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )


def _validate_inline_identity_plan_digests(
    value: Any,
    *,
    rewrite: bool,
    field: str = "phone",
) -> None:
    if type(value) is dict:
        for key, item in value.items():
            _validate_inline_identity_plan_digests(
                item,
                rewrite=rewrite,
                field=f"{field}.{key}",
            )
        return
    if type(value) is not list:
        return
    for index, item in enumerate(value):
        _validate_inline_identity_plan_digests(
            item,
            rewrite=rewrite,
            field=f"{field}[{index}]",
        )
    if not value or not all(type(item) is str for item in value):
        return
    plan_count = value.count("--plan-json")
    digest_count = value.count("--plan-sha256")
    common.require(
        plan_count == digest_count,
        f"E_INLINE_PLAN_OPTIONS: {field}",
    )
    if plan_count == 0:
        return
    common.require(plan_count == 1, f"E_INLINE_PLAN_OPTIONS: {field}")
    plan_index = value.index("--plan-json") + 1
    digest_index = value.index("--plan-sha256") + 1
    common.require(
        plan_index < len(value) and digest_index < len(value),
        f"E_INLINE_PLAN_OPTIONS: {field}",
    )
    try:
        parsed = json.loads(value[plan_index])
    except json.JSONDecodeError as error:
        raise common.EvidenceError(
            f"E_INLINE_PLAN_JSON: {field}"
        ) from error
    compact = json.dumps(
        parsed,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    common.exact(value[plan_index], compact, f"{field}.plan_json")
    expected = common.sha256_bytes(compact.encode("ascii"))
    if rewrite:
        value[digest_index] = expected
    else:
        common.exact(value[digest_index], expected, f"{field}.plan_sha256")


def _phone_mechanism_matrix(plan: dict[str, Any]) -> dict[str, list[list[str]]]:
    return {
        "desktop": [
            list(argv) for argv in plan["mechanism_commands"]["desktop"]
        ],
        "op12": [
            list(plan["processes"]["op12_stagenet"]["argv"]),
            list(plan["probes"]["op12"]["before_argv"]),
            list(plan["probes"]["op12"]["after_argv"]),
        ],
        "op15": [
            list(plan["processes"]["op15_stagenet"]["argv"]),
            list(plan["processes"]["op15_direct_relay"]["argv"]),
            list(plan["probes"]["op15"]["before_argv"]),
            list(plan["probes"]["op15"]["after_argv"]),
        ],
    }


def validate_identity_binding(
    *,
    prospective_root_path: Path,
    bound_root_path: Path,
    identity_binding_receipt_path: Path,
    identity_binding_stage_receipt_path: Path,
    contract_path: Path,
    candidate_path: Path,
    runtime_plan_path: Path,
    token_history_path: Path,
    tokenizer_plan_path: Path,
    preparation_path: Path,
    phase_lock_path: Path,
    fresh_path: Path,
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate_raw: bytes,
    orchestration: dict[str, Any],
) -> dict[str, Any]:
    run_root = Path(orchestration["run_root"])
    common.exact(
        prospective_root_path,
        Path(orchestration["inputs"]["prospective_root"]["path"]),
        "E_PROSPECTIVE_PLAN_PATH",
    )
    prospective, prospective_raw = _read_stable_canonical(
        prospective_root_path,
        "prospective_root",
    )
    plan_root_record = orchestration["inputs"]["prospective_root"]
    common.exact(
        plan_root_record["bytes"],
        len(prospective_raw),
        "E_PROSPECTIVE_PLAN_BYTES",
    )
    common.exact(
        plan_root_record["sha256"],
        common.sha256_bytes(prospective_raw),
        "E_PROSPECTIVE_PLAN_SHA256",
    )
    common.exact_keys(
        prospective,
        {
            "acquisition_ready",
            "artifacts",
            "candidate",
            "contract",
            "cuda_monolithic_launch",
            "desktop_control",
            "identity_placeholders",
            "model_id",
            "model_sha256",
            "network_placeholders",
            "phase",
            "schema",
            "spec",
            "status",
            "token_history",
            "tokenizer_plan",
        },
        "prospective_root",
    )
    common.exact(
        prospective["schema"],
        "s39-cp0-r1-v24-prospective-runtime-root-v1",
        "prospective_root.schema",
    )
    common.exact(prospective["phase"], PHASE, "prospective_root.phase")
    common.exact(prospective["model_id"], MODEL_ID, "prospective_root.model")
    common.exact(prospective["acquisition_ready"], False, "prospective_root.ready")
    common.exact(
        prospective["status"],
        "POST_REBOOT_IDENTITY_BINDING_REQUIRED",
        "prospective_root.status",
    )
    common.exact(
        prospective["desktop_control"],
        {
            "cuda_ssh_target": CUDA_SSH_TARGET,
            "phone_adb_port": PHONE_ADB_PORT,
        },
        "prospective_root.desktop_control",
    )
    common.exact(
        prospective["identity_placeholders"],
        UNBOUND_BOOT_IDS,
        "prospective_root.identity_placeholders",
    )
    common.exact(
        prospective["network_placeholders"],
        UNBOUND_PHONE_NETWORK,
        "prospective_root.network_placeholders",
    )

    external = (
        ("contract", contract_path, contract_raw),
        ("candidate", candidate_path, candidate_raw),
    )
    for name, path, expected_raw in external:
        _, raw, _ = _read_identity_record(
            prospective[name],
            f"prospective_root.{name}",
            path,
        )
        common.exact(raw, expected_raw, f"E_PROSPECTIVE_{name.upper()}")
    candidate = common.parse_json(candidate_raw, "identity.candidate")
    common.require(type(candidate) is dict, "E_IDENTITY_CANDIDATE_TYPE")
    model = next(
        (
            value
            for value in candidate.get("models", [])
            if value.get("slot") == "A"
        ),
        None,
    )
    common.require(type(model) is dict, "E_IDENTITY_MODEL_A")
    common.exact(
        prospective["model_sha256"],
        model["artifact"]["sha256"],
        "E_PROSPECTIVE_MODEL_SHA256",
    )
    history, history_raw, _ = _read_identity_record(
        prospective["token_history"],
        "prospective_root.token_history",
        token_history_path,
    )
    _, tokenizer_raw, _ = _read_identity_record(
        prospective["tokenizer_plan"],
        "prospective_root.tokenizer_plan",
        tokenizer_plan_path,
    )
    mono, _, _ = _read_identity_record(
        prospective["cuda_monolithic_launch"],
        "prospective_root.cuda_monolithic_launch",
    )
    _read_identity_record(prospective["spec"], "prospective_root.spec")
    common.exact(
        history["model_sha256"],
        prospective["model_sha256"],
        "prospective_root.history.model",
    )
    common.exact(
        history["candidate_sha256"],
        common.sha256_bytes(candidate_raw),
        "prospective_root.history.candidate",
    )
    artifacts = common.exact_keys(
        prospective["artifacts"],
        set(BOUND_ARTIFACT_SCHEMAS),
        "prospective_root.artifacts",
    )
    prospective_values = {}
    prospective_raws = {}
    prospective_paths = {}
    for name, schema in BOUND_ARTIFACT_SCHEMAS.items():
        value, raw, path = _read_identity_record(
            artifacts[name],
            f"prospective_root.artifacts.{name}",
        )
        common.exact(value.get("schema"), schema, f"prospective.{name}.schema")
        prospective_values[name] = value
        prospective_raws[name] = raw
        prospective_paths[name] = path
    common.exact(
        prospective_values["runtime_plan"]["cuda_monolithic_launch"],
        mono,
        "E_PROSPECTIVE_CUDA_MONOLITHIC",
    )
    common.exact(
        prospective_values["runtime_plan"]["contract_sha256"],
        common.sha256_bytes(contract_raw),
        "E_PROSPECTIVE_RUNTIME_CONTRACT",
    )
    common.exact(
        prospective_values["runtime_plan"]["candidate_sha256"],
        common.sha256_bytes(candidate_raw),
        "E_PROSPECTIVE_RUNTIME_CANDIDATE",
    )
    common.exact(
        prospective_values["runtime_plan"]["token_history"]["artifact_path"],
        str(token_history_path),
        "E_PROSPECTIVE_RUNTIME_HISTORY_PATH",
    )
    common.exact(
        prospective_values["runtime_plan"]["token_history"][
            "tokenizer_plan_sha256"
        ],
        common.sha256_bytes(tokenizer_raw),
        "E_PROSPECTIVE_RUNTIME_TOKENIZER",
    )
    common.exact(
        prospective_values["joint_capture_plan"]["history"]["path"],
        str(token_history_path),
        "E_PROSPECTIVE_JOINT_HISTORY_PATH",
    )
    common.exact(
        prospective_values["joint_capture_plan"]["history"]["sha256"],
        common.sha256_bytes(history_raw),
        "E_PROSPECTIVE_JOINT_HISTORY_SHA256",
    )
    prospective_phone = prospective_values["phone_route_launch"]["phones"]
    _validate_inline_identity_plan_digests(
        prospective_values["phone_route_launch"],
        rewrite=False,
        field="prospective.phone",
    )
    for phone in ("op12", "op15"):
        common.exact(
            prospective_phone[phone]["boot_id"],
            UNBOUND_BOOT_IDS[phone],
            f"E_PROSPECTIVE_BOOT_SENTINEL: {phone}",
        )
        for key, expected in UNBOUND_PHONE_NETWORK[phone].items():
            common.exact(
                prospective_phone[phone][key],
                expected,
                f"E_PROSPECTIVE_NETWORK_SENTINEL: {phone}.{key}",
            )
        peer = "op15" if phone == "op12" else "op12"
        common.exact(
            prospective_phone[phone]["direct_peer_ipv4"],
            UNBOUND_PHONE_NETWORK[peer]["local_ipv4"],
            f"E_PROSPECTIVE_PEER_SENTINEL: {phone}",
        )
    prospective_mechanism = _phone_mechanism_matrix(
        prospective_values["phone_route_launch"]
    )
    common.exact(
        prospective_values["phone_route_launch"]["mechanism_commands"],
        prospective_mechanism,
        "E_PROSPECTIVE_PHONE_MECHANISM",
    )
    common.exact(
        prospective_values["cuda_route_launch"]["mechanism_commands"],
        prospective_mechanism,
        "E_PROSPECTIVE_CUDA_MECHANISM",
    )
    common.exact(
        prospective_values["joint_capture_plan"]["mechanism_commands"],
        prospective_mechanism,
        "E_PROSPECTIVE_JOINT_MECHANISM",
    )
    prospective_mechanism_sha256 = common.sha256_bytes(
        common.canonical_bytes(prospective_mechanism)
    )
    common.exact(
        prospective_mechanism_sha256,
        orchestration["mechanism_commands_sha256"],
        "E_PROSPECTIVE_ORCHESTRATION_MECHANISM",
    )

    preparation, preparation_raw = _read_stable_canonical(
        preparation_path,
        "identity.preparation",
    )
    lock, lock_raw = _read_stable_canonical(
        phase_lock_path,
        "identity.phase_lock",
    )
    fresh, _ = _read_stable_canonical(fresh_path, "identity.fresh")
    receipt, receipt_raw = _read_stable_canonical(
        identity_binding_receipt_path,
        "identity.receipt",
    )
    common.exact_keys(
        receipt,
        {
            "completed_ns",
            "desktop_identity",
            "device_boot_ids",
            "mechanism_commands_sha256",
            "outputs",
            "phase",
            "phase_id",
            "phase_lock_sha256",
            "preparation_sha256",
            "prospective_root_sha256",
            "schema",
            "started_ns",
        },
        "identity.receipt",
    )
    common.exact(
        receipt["schema"],
        "s39-cp0-r1-v24-identity-binding-receipt-v1",
        "identity.receipt.schema",
    )
    common.exact(receipt["phase"], PHASE, "identity.receipt.phase")
    common.exact(receipt["phase_id"], lock["phase_id"], "identity.receipt.phase_id")
    common.exact(
        receipt["phase_lock_sha256"],
        common.sha256_bytes(lock_raw),
        "identity.receipt.phase_lock",
    )
    common.exact(
        receipt["preparation_sha256"],
        common.sha256_bytes(preparation_raw),
        "identity.receipt.preparation",
    )
    common.exact(
        receipt["prospective_root_sha256"],
        common.sha256_bytes(prospective_raw),
        "identity.receipt.prospective_root",
    )
    started = common.integer(receipt["started_ns"], "identity.receipt.started", 1)
    completed = common.integer(
        receipt["completed_ns"],
        "identity.receipt.completed",
        1,
    )
    common.require(
        common.integer(lock["event_ns"], "identity.phase_lock.event", 1)
        <= started
        <= completed
        < common.integer(fresh["started_ns"], "identity.fresh.started", 1),
        "E_IDENTITY_BINDING_ORDER",
    )
    boot_ids = common.exact_keys(
        receipt["device_boot_ids"],
        {"cuda", "op12", "op15"},
        "identity.receipt.device_boot_ids",
    )
    common.exact(boot_ids, lock["device_boot_ids"], "E_IDENTITY_BOOT_IDS")
    common.require(
        len(set(boot_ids.values())) == 3
        and not set(boot_ids.values()).intersection(UNBOUND_BOOT_IDS.values()),
        "E_IDENTITY_BOOT_REUSE",
    )
    before_boot_ids = common.exact_keys(
        preparation["before_boot_ids"],
        {"op12", "op15"},
        "identity.preparation.before_boot_ids",
    )
    common.require(
        len(set(before_boot_ids.values())) == 2,
        "E_IDENTITY_BEFORE_BOOT_REUSE",
    )
    for phone in ("op12", "op15"):
        common.require(
            before_boot_ids[phone] != boot_ids[phone],
            f"E_STALE_BOOT_REUSE: {phone}",
        )
    desktop_identity = {
        "cuda_boot_id": boot_ids["cuda"],
        "cuda_host": contract["devices"]["cuda"]["host"],
        "cuda_ssh_target": CUDA_SSH_TARGET,
        "cuda_uuid": contract["devices"]["cuda"]["uuid"],
        "phone_adb_port": PHONE_ADB_PORT,
    }
    common.exact(
        receipt["desktop_identity"],
        desktop_identity,
        "E_IDENTITY_DESKTOP",
    )

    expected_stage_path = (
        run_root / "receipts" / "identity_binding" / "receipt.json"
    )
    common.exact(
        identity_binding_stage_receipt_path,
        expected_stage_path,
        "E_IDENTITY_STAGE_RECEIPT_PATH",
    )
    stage_receipt, _ = _read_stable_canonical(
        identity_binding_stage_receipt_path,
        "identity.stage_receipt",
    )
    stage_interval = orchestration["stage_intervals"]["identity_binding"]
    common.exact(
        stage_receipt["started_ns"],
        stage_interval["started_ns"],
        "E_IDENTITY_STAGE_STARTED",
    )
    common.exact(
        stage_receipt["completed_ns"],
        stage_interval["completed_ns"],
        "E_IDENTITY_STAGE_COMPLETED",
    )
    common.require(
        stage_receipt["started_ns"]
        <= started
        <= completed
        <= stage_receipt["completed_ns"],
        "E_IDENTITY_STAGE_INTERVAL",
    )

    expected_bound_paths = {
        "cuda_route_launch": run_root / "bound" / "cuda-route-launch.json",
        "joint_capture_plan": run_root / "bound" / "joint-capture-plan.json",
        "phone_route_launch": run_root / "bound" / "phone-route-launch.json",
        "runtime_plan": run_root / "bound" / "runtime-bundle-plan.json",
    }
    bound_values = {}
    bound_raws = {}
    outputs = common.exact_keys(
        receipt["outputs"],
        set(BOUND_ARTIFACT_SCHEMAS),
        "identity.receipt.outputs",
    )
    for name, schema in BOUND_ARTIFACT_SCHEMAS.items():
        value, raw, _ = _read_identity_record(
            outputs[name],
            f"identity.receipt.outputs.{name}",
            expected_bound_paths[name],
        )
        common.exact(value.get("schema"), schema, f"bound.{name}.schema")
        _reject_unbound_identity(value, f"bound.{name}")
        bound_values[name] = value
        bound_raws[name] = raw
    common.exact(
        expected_bound_paths["runtime_plan"],
        runtime_plan_path,
        "E_BOUND_RUNTIME_PATH",
    )
    common.exact(
        bound_raws["runtime_plan"],
        prospective_raws["runtime_plan"],
        "E_BOUND_RUNTIME_PROJECTION",
    )
    bound_phone = bound_values["phone_route_launch"]["phones"]
    for phone in ("op12", "op15"):
        common.exact(
            bound_phone[phone]["boot_id"],
            boot_ids[phone],
            f"E_BOUND_PHONE_BOOT: {phone}",
        )
        for key in ("interface", "local_ipv4"):
            common.exact(
                bound_phone[phone][key],
                preparation["devices"][phone][key],
                f"E_BOUND_PHONE_NETWORK: {phone}.{key}",
            )
    common.exact(
        bound_phone["op12"]["direct_peer_ipv4"],
        bound_phone["op15"]["local_ipv4"],
        "E_BOUND_PHONE_PEER: op12",
    )
    common.exact(
        bound_phone["op15"]["direct_peer_ipv4"],
        bound_phone["op12"]["local_ipv4"],
        "E_BOUND_PHONE_PEER: op15",
    )
    realized_mechanism = _phone_mechanism_matrix(
        bound_values["phone_route_launch"]
    )
    common.exact(
        bound_values["phone_route_launch"]["mechanism_commands"],
        realized_mechanism,
        "E_BOUND_PHONE_MECHANISM",
    )
    common.exact(
        bound_values["cuda_route_launch"]["mechanism_commands"],
        realized_mechanism,
        "E_BOUND_CUDA_MECHANISM",
    )
    common.exact(
        bound_values["joint_capture_plan"]["mechanism_commands"],
        realized_mechanism,
        "E_BOUND_JOINT_MECHANISM",
    )
    realized_mechanism_sha256 = common.sha256_bytes(
        common.canonical_bytes(realized_mechanism)
    )
    common.exact(
        receipt["mechanism_commands_sha256"],
        realized_mechanism_sha256,
        "E_IDENTITY_REALIZED_MECHANISM",
    )
    expected_cuda = copy.deepcopy(prospective_values["cuda_route_launch"])
    expected_cuda["mechanism_commands"] = copy.deepcopy(realized_mechanism)
    common.exact(
        bound_values["cuda_route_launch"],
        expected_cuda,
        "E_BOUND_CUDA_PROJECTION",
    )
    replacements = {
        UNBOUND_PHONE_NETWORK["op12"]["interface"]:
            preparation["devices"]["op12"]["interface"],
        UNBOUND_PHONE_NETWORK["op12"]["local_ipv4"]:
            preparation["devices"]["op12"]["local_ipv4"],
        UNBOUND_PHONE_NETWORK["op15"]["local_ipv4"]:
            preparation["devices"]["op15"]["local_ipv4"],
    }
    expected_phone = _replace_identity_strings(
        copy.deepcopy(prospective_values["phone_route_launch"]),
        replacements,
    )
    _rebind_structured_identity_argv(
        expected_phone,
        replacements,
        "expected.phone",
    )
    _validate_inline_identity_plan_digests(
        expected_phone,
        rewrite=True,
        field="expected.phone",
    )
    for phone in ("op12", "op15"):
        expected_phone["phones"][phone]["boot_id"] = boot_ids[phone]
    common.exact(
        bound_values["phone_route_launch"],
        expected_phone,
        "E_BOUND_PHONE_PROJECTION",
    )
    _validate_inline_identity_plan_digests(
        bound_values["phone_route_launch"],
        rewrite=False,
        field="bound.phone",
    )
    bound_joint = bound_values["joint_capture_plan"]["commands"]
    for name, artifact_name in (
        ("cuda", "cuda_route_launch"),
        ("phone", "phone_route_launch"),
    ):
        command = bound_joint[name]
        index = common.integer(
            command["launch_plan_argv_index"],
            f"bound.joint.{name}.launch_index",
            1,
        )
        common.exact(
            command["argv_template"][index],
            str(expected_bound_paths[artifact_name]),
            f"E_BOUND_JOINT_PATH: {name}",
        )
        common.exact(
            command["launch_plan_sha256"],
            common.sha256_bytes(bound_raws[artifact_name]),
            f"E_BOUND_JOINT_SHA256: {name}",
        )
        records = [
            record
            for record in command["executed_files"]
            if record["argv_index"] == index
        ]
        common.require(len(records) == 1, f"E_BOUND_JOINT_RECORD: {name}")
        common.exact(
            records[0],
            {
                "argv_index": index,
                "bytes": len(bound_raws[artifact_name]),
                "path": str(expected_bound_paths[artifact_name]),
                "sha256": common.sha256_bytes(bound_raws[artifact_name]),
            },
            f"E_BOUND_JOINT_EXECUTED: {name}",
        )
        common.exact(
            _argv_value(
                command["argv_template"],
                "--mechanism-commands-sha256",
                f"bound.joint.{name}.argv",
            ),
            realized_mechanism_sha256,
            f"E_BOUND_JOINT_MECHANISM_ARGV: {name}",
        )
    expected_joint = copy.deepcopy(
        prospective_values["joint_capture_plan"]
    )
    expected_joint["mechanism_commands"] = copy.deepcopy(realized_mechanism)
    for name, artifact_name in (
        ("cuda", "cuda_route_launch"),
        ("phone", "phone_route_launch"),
    ):
        command = expected_joint["commands"][name]
        mechanism_index = command["argv_template"].index(
            "--mechanism-commands-sha256"
        ) + 1
        common.exact(
            command["argv_template"][mechanism_index],
            prospective_mechanism_sha256,
            f"E_PROSPECTIVE_JOINT_MECHANISM_ARGV: {name}",
        )
        command["argv_template"][mechanism_index] = realized_mechanism_sha256
        launch_index = command["launch_plan_argv_index"]
        command["argv_template"][launch_index] = str(
            expected_bound_paths[artifact_name]
        )
        command["launch_plan_sha256"] = common.sha256_bytes(
            bound_raws[artifact_name]
        )
        matching = [
            record
            for record in command["executed_files"]
            if record["argv_index"] == launch_index
        ]
        common.require(
            len(matching) == 1,
            f"E_PROSPECTIVE_JOINT_RECORD: {name}",
        )
        command["executed_files"][
            command["executed_files"].index(matching[0])
        ] = {
            "argv_index": launch_index,
            "bytes": len(bound_raws[artifact_name]),
            "path": str(expected_bound_paths[artifact_name]),
            "sha256": common.sha256_bytes(bound_raws[artifact_name]),
        }
        command["executed_files"].sort(
            key=lambda record: record["argv_index"]
        )
    common.exact(
        bound_values["joint_capture_plan"],
        expected_joint,
        "E_BOUND_JOINT_PROJECTION",
    )

    common.exact(
        identity_binding_receipt_path,
        run_root / "bound" / "identity-binding-receipt.json",
        "E_IDENTITY_RECEIPT_PATH",
    )
    common.exact(
        bound_root_path,
        run_root / "bound" / "bound-runtime-root.json",
        "E_BOUND_ROOT_PATH",
    )
    bound_root, bound_root_raw = _read_stable_canonical(
        bound_root_path,
        "bound_root",
    )
    attestation = _validate_identity_attestation(
        orchestration["identity_attestation"],
        "identity.attestation",
    )
    common.exact(
        attestation["identity_binding_receipt_sha256"],
        common.sha256_bytes(receipt_raw),
        "E_IDENTITY_ATTESTED_RECEIPT",
    )
    common.exact(
        attestation["bound_root_sha256"],
        common.sha256_bytes(bound_root_raw),
        "E_IDENTITY_ATTESTED_ROOT",
    )
    common.exact_keys(
        bound_root,
        {
            "artifacts",
            "desktop_identity",
            "device_boot_ids",
            "identity_binding_receipt",
            "mechanism_commands_sha256",
            "phase",
            "phase_id",
            "phase_lock_sha256",
            "preparation_sha256",
            "prospective_root_sha256",
            "schema",
        },
        "bound_root",
    )
    common.exact(
        bound_root["schema"],
        "s39-cp0-r1-v24-bound-runtime-root-v1",
        "bound_root.schema",
    )
    common.exact(bound_root["phase"], PHASE, "bound_root.phase")
    common.exact(bound_root["phase_id"], lock["phase_id"], "bound_root.phase_id")
    for key in (
        "desktop_identity",
        "device_boot_ids",
        "phase_lock_sha256",
        "preparation_sha256",
        "prospective_root_sha256",
    ):
        common.exact(bound_root[key], receipt[key], f"E_BOUND_ROOT_PROJECTION: {key}")
    common.exact(
        bound_root["mechanism_commands_sha256"],
        realized_mechanism_sha256,
        "E_BOUND_ROOT_REALIZED_MECHANISM",
    )
    common.exact(
        bound_root["artifacts"],
        outputs,
        "E_BOUND_ROOT_ARTIFACTS",
    )
    _, projected_receipt_raw, _ = _read_identity_record(
        bound_root["identity_binding_receipt"],
        "bound_root.identity_binding_receipt",
        identity_binding_receipt_path,
    )
    common.exact(
        projected_receipt_raw,
        receipt_raw,
        "E_BOUND_ROOT_RECEIPT",
    )

    stage_argv = stage_receipt["argv"]
    for flag, expected in (
        ("--prospective-root", prospective_root_path),
        ("--contract", contract_path),
        ("--preparation", preparation_path),
        ("--phase-lock", phase_lock_path),
        ("--receipt", identity_binding_receipt_path),
        ("--bound-root", bound_root_path),
        ("--bound-runtime-plan", runtime_plan_path),
    ):
        common.exact(
            _argv_value(stage_argv, flag, "identity.stage.argv"),
            str(expected),
            f"E_IDENTITY_STAGE_ARG: {flag}",
        )
    for name, path in prospective_paths.items():
        common.exact(
            _argv_value(
                stage_argv,
                f"--prospective-{name.replace('_', '-')}",
                "identity.stage.argv",
            ),
            str(path),
            f"E_IDENTITY_STAGE_PROSPECTIVE: {name}",
        )
    for name, path in expected_bound_paths.items():
        common.exact(
            _argv_value(
                stage_argv,
                f"--bound-{name.replace('_', '-')}",
                "identity.stage.argv",
            ),
            str(path),
            f"E_IDENTITY_STAGE_BOUND: {name}",
        )

    return {
        "artifact_sha256s": {
            name: common.sha256_bytes(raw)
            for name, raw in bound_raws.items()
        },
        "completed_ns": completed,
        "bound_root_sha256": common.sha256_bytes(bound_root_raw),
        "identity_binding_receipt_sha256": common.sha256_bytes(receipt_raw),
        "mechanism_commands_sha256": realized_mechanism_sha256,
        "phase_id": receipt["phase_id"],
        "prospective_root_sha256": common.sha256_bytes(prospective_raw),
        "schema": "s39-cp0-r1-v24-identity-binding-result-v1",
        "started_ns": started,
        "status": "V2_4_IDENTITY_BINDING_PASS",
    }


def _read_nested_artifact(
    bundle_root: Path,
    record: Any,
    field: str,
    *,
    canonical: bool,
) -> tuple[Any, bytes]:
    record = common.exact_keys(record, {"bytes", "path", "sha256"}, field)
    expected_bytes = common.integer(record["bytes"], f"{field}.bytes", 1)
    expected_sha256 = common.digest(record["sha256"], f"{field}.sha256")
    path = Path(common.absolute_path(record["path"], f"{field}.path"))
    root = bundle_root.resolve()
    resolved = path.resolve()
    common.require(resolved.is_relative_to(root), f"E_CAPTURE_PATH: {field}")
    raw = _read_stable_regular(resolved, field)
    common.exact(len(raw), expected_bytes, f"E_CAPTURE_BYTES: {field}")
    common.exact(
        common.sha256_bytes(raw),
        expected_sha256,
        f"E_CAPTURE_SHA256: {field}",
    )
    if not canonical:
        return None, raw
    value = common.parse_json(raw, field)
    common.require(type(value) is dict, f"E_CAPTURE_JSON_TYPE: {field}")
    common.exact(common.canonical_bytes(value), raw, f"E_CAPTURE_CANONICAL: {field}")
    return value, raw


def _validate_evidence_artifacts(
    bundle_root: Path,
    records: Any,
    field: str,
) -> None:
    common.require(type(records) is list and bool(records), f"E_CAPTURE_ARTIFACTS: {field}")
    paths = set()
    for index, record in enumerate(records):
        item = f"{field}[{index}]"
        path = record.get("path") if type(record) is dict else None
        common.require(path not in paths, f"E_CAPTURE_ARTIFACT_PATH_REUSE: {item}")
        paths.add(path)
        _read_nested_artifact(
            bundle_root,
            record,
            item,
            canonical=False,
        )


def _exact_role_projection(
    rows_by_role: dict[str, list[dict[str, Any]]],
    role: str,
    values: Any,
    phase_id: str,
) -> None:
    common.require(type(values) is list and bool(values), f"E_CAPTURE_ROWS: {role}")
    projected = [
        {
            "acquisition_id": phase_id,
            **row,
            "phase": PHASE,
            "phase_id": phase_id,
            "role": role,
        }
        for row in values
    ]
    common.exact(
        rows_by_role[role],
        projected,
        f"E_CAPTURE_RAW_PROJECTION: {role}",
    )


def _runtime_processes_by_id(runtime: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        value["bundle_id"]: value
        for value in runtime["processes"]
    }


def _validate_captured_runtime_process(
    value: Any,
    runtime: dict[str, Any],
    bundle_id: str,
    started_ns: int,
    completed_ns: int,
) -> None:
    common.exact_keys(
        value,
        {
            "boot_id",
            "bundle_id",
            "endpoint",
            "launcher_path",
            "loaded_repo_component_ids",
            "observed_ns",
            "pid",
            "start_ticks",
            "system_dependencies",
        },
        f"joint.runtime_process.{bundle_id}",
    )
    expected = _runtime_processes_by_id(runtime)[bundle_id]
    for key in (
        "boot_id",
        "bundle_id",
        "endpoint",
        "launcher_path",
        "loaded_repo_component_ids",
        "pid",
        "start_ticks",
    ):
        common.exact(
            value[key],
            expected[key],
            f"E_JOINT_RUNTIME_PROCESS: {bundle_id}.{key}",
        )
    observed = common.integer(
        value["observed_ns"],
        f"joint.runtime_process.{bundle_id}.observed_ns",
        1,
    )
    common.require(
        started_ns <= observed <= completed_ns,
        f"E_JOINT_RUNTIME_INTERVAL: {bundle_id}",
    )
    dependencies = value["system_dependencies"]
    common.require(
        type(dependencies) is list and bool(dependencies),
        f"E_JOINT_RUNTIME_DEPENDENCIES: {bundle_id}",
    )
    paths = []
    for index, dependency in enumerate(dependencies):
        item = f"joint.runtime_process.{bundle_id}.dependencies[{index}]"
        common.exact_keys(
            dependency,
            {
                "build_id",
                "ctime_ns",
                "device_id",
                "inode",
                "mode",
                "mtime_ns",
                "path",
                "size",
            },
            item,
        )
        paths.append(common.absolute_path(dependency["path"], f"{item}.path"))
        for key in ("ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"):
            common.integer(dependency[key], f"{item}.{key}")
        build_id = dependency["build_id"]
        common.require(
            build_id is None or (type(build_id) is str and bool(build_id)),
            f"E_JOINT_RUNTIME_BUILD_ID: {item}",
        )
    common.exact(paths, sorted(set(paths)), f"E_JOINT_RUNTIME_DEPENDENCY_ORDER: {bundle_id}")


def _argv_value(argv: list[str], flag: str, field: str) -> str:
    common.exact(argv.count(flag), 1, f"{field}.{flag}.count")
    index = argv.index(flag)
    common.require(index + 1 < len(argv), f"E_CAPTURE_ARGV_VALUE: {field}.{flag}")
    return common.text(argv[index + 1], f"{field}.{flag}")


def _validate_subproducer_receipt(
    *,
    bundle_root: Path,
    name: str,
    evidence: dict[str, Any],
    outer: dict[str, Any],
    contract: dict[str, Any],
    model_sha256: str,
    acquisition_started_ns: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt, _ = _read_nested_artifact(
        bundle_root,
        evidence["receipt_artifact"],
        f"joint.{name}.receipt_artifact",
        canonical=True,
    )
    common.exact(receipt, evidence["receipt"], f"E_JOINT_RECEIPT_PROJECTION: {name}")
    common.exact_keys(
        receipt,
        {
            "argv",
            "completed_ns",
            "cwd",
            "environment",
            "launch_plan_sha256",
            "producer_sha256",
            "returncode",
            "schema",
            "started_ns",
        },
        f"joint.{name}.receipt",
    )
    common.exact(
        receipt["schema"],
        "s39-cp0-r1-v24-subproducer-receipt-v1",
        f"joint.{name}.receipt.schema",
    )
    common.exact(receipt["returncode"], 0, f"E_JOINT_SUBPRODUCER_EXIT: {name}")
    started = common.integer(receipt["started_ns"], f"joint.{name}.started", 1)
    completed = common.integer(receipt["completed_ns"], f"joint.{name}.completed", 1)
    common.require(
        outer["started_ns"] <= started < completed <= outer["completed_ns"],
        f"E_JOINT_SUBPRODUCER_INTERVAL: {name}",
    )
    argv = receipt["argv"]
    common.require(
        type(argv) is list
        and len(argv) >= 20
        and all(type(item) is str and bool(item) for item in argv),
        f"E_JOINT_SUBPRODUCER_ARGV: {name}",
    )
    common.absolute_path(argv[0], f"joint.{name}.argv0")
    source_record = contract["producer_requirements"]["source_programs"][
        "phone_route" if name == "phone" else "cuda_route"
    ]
    source_path = Path(argv[0]).resolve()
    common.require(
        source_path.is_relative_to(bundle_root.resolve()),
        f"E_JOINT_SUBPRODUCER_SOURCE_PATH: {name}",
    )
    source_raw = _read_stable_regular(source_path, f"joint.{name}.source")
    common.exact(len(source_raw), source_record["bytes"], f"E_JOINT_SOURCE_BYTES: {name}")
    common.exact(
        common.sha256_bytes(source_raw),
        source_record["sha256"],
        f"E_JOINT_SOURCE_SHA256: {name}",
    )
    common.exact(
        receipt["producer_sha256"],
        source_record["sha256"],
        f"E_JOINT_RECEIPT_SOURCE: {name}",
    )
    common.exact(
        outer["subproducer_bindings"][name]["producer_sha256"],
        source_record["sha256"],
        f"E_JOINT_OUTER_SOURCE: {name}",
    )
    fragment, _ = _read_nested_artifact(
        bundle_root,
        evidence["fragment"],
        f"joint.{name}.fragment",
        canonical=True,
    )
    common.exact(
        evidence["fragment"]["sha256"],
        outer["fragment_sha256"][name],
        f"E_JOINT_FRAGMENT_SHA256: {name}",
    )
    common.exact(
        _argv_value(argv, "--output", f"joint.{name}.argv"),
        evidence["fragment"]["path"],
        f"E_JOINT_OUTPUT_PATH: {name}",
    )
    common.exact(
        _argv_value(argv, "--phase-id", f"joint.{name}.argv"),
        outer["phase_id"],
        f"E_JOINT_ARG_PHASE: {name}",
    )
    common.exact(
        _argv_value(argv, "--started", f"joint.{name}.argv"),
        str(acquisition_started_ns),
        f"E_JOINT_ARG_STARTED: {name}",
    )
    common.exact(
        _argv_value(argv, "--plan", f"joint.{name}.argv"),
        outer["command_plan_sha256"],
        f"E_JOINT_ARG_COMMAND_PLAN: {name}",
    )
    common.exact(
        _argv_value(
            argv,
            "--mechanism-commands-sha256",
            f"joint.{name}.argv",
        ),
        outer["mechanism_commands_sha256"],
        f"E_JOINT_ARG_MECHANISM: {name}",
    )
    common.exact(
        _argv_value(argv, "--model-sha256", f"joint.{name}.argv"),
        model_sha256,
        f"E_JOINT_ARG_MODEL: {name}",
    )
    common.exact(argv.count("--execute"), 1, f"E_JOINT_ARG_EXECUTE: {name}")
    expected_confirm = (
        "RUN_V24_PHONE_ROUTE_A_ONLY"
        if name == "phone"
        else "RUN_V24_CUDA_ROUTE_A_ONLY"
    )
    common.exact(
        _argv_value(argv, "--confirm", f"joint.{name}.argv"),
        expected_confirm,
        f"E_JOINT_ARG_CONFIRM: {name}",
    )
    common.absolute_path(
        _argv_value(argv, "--pre-dir", f"joint.{name}.argv"),
        f"joint.{name}.pre_dir",
    )
    launch_path = Path(
        common.absolute_path(
            _argv_value(argv, "--launch-plan", f"joint.{name}.argv"),
            f"joint.{name}.launch_path",
        )
    ).resolve()
    common.require(
        launch_path.is_relative_to(bundle_root.resolve()),
        f"E_JOINT_LAUNCH_PATH: {name}",
    )
    launch_raw = _read_stable_regular(launch_path, f"joint.{name}.launch")
    launch = common.parse_json(launch_raw, f"joint.{name}.launch")
    common.require(type(launch) is dict, f"E_JOINT_LAUNCH_TYPE: {name}")
    common.exact(
        common.canonical_bytes(launch),
        launch_raw,
        f"E_JOINT_LAUNCH_CANONICAL: {name}",
    )
    launch_sha256 = common.sha256_bytes(launch_raw)
    for value, field in (
        (receipt["launch_plan_sha256"], "receipt"),
        (evidence["launch_plan_sha256"], "evidence"),
        (outer["subproducer_bindings"][name]["launch_plan_sha256"], "outer"),
    ):
        common.exact(
            value,
            launch_sha256,
            f"E_JOINT_LAUNCH_SHA256: {name}.{field}",
        )
    environment = receipt["environment"]
    common.require(
        type(environment) is dict
        and all(
            type(key) is str
            and bool(key)
            and type(value) is str
            and key.isascii()
            and value.isascii()
            and "=" not in key
            and "\x00" not in value
            for key, value in environment.items()
        ),
        f"E_JOINT_SUBPRODUCER_ENV: {name}",
    )
    common.absolute_path(receipt["cwd"], f"joint.{name}.cwd")
    return fragment, launch


def _validate_joint_capture_artifacts(
    *,
    receipt: dict[str, Any],
    bundle_root: Path,
    contract: dict[str, Any],
    candidate: dict[str, Any],
    history_raw: bytes,
    runtime_plan: dict[str, Any],
    acquisition_started_ns: int,
) -> dict[str, Any]:
    capture_plan, capture_plan_raw = _read_nested_artifact(
        bundle_root,
        receipt["capture_plan_artifact"],
        "joint.capture_plan_artifact",
        canonical=True,
    )
    common.exact(
        common.sha256_bytes(capture_plan_raw),
        receipt["capture_plan_sha256"],
        "E_JOINT_CAPTURE_PLAN_SHA256",
    )
    producer_source = contract["producer_requirements"]["source_programs"][
        "joint_phone_cuda"
    ]
    _, producer_raw = _read_nested_artifact(
        bundle_root,
        receipt["joint_producer_artifact"],
        "joint.joint_producer_artifact",
        canonical=False,
    )
    common.exact(
        len(producer_raw),
        producer_source["bytes"],
        "E_JOINT_PRODUCER_ARTIFACT_BYTES",
    )
    common.exact(
        common.sha256_bytes(producer_raw),
        producer_source["sha256"],
        "E_JOINT_PRODUCER_ARTIFACT_SHA256",
    )
    common.exact(
        receipt["joint_producer_sha256"],
        producer_source["sha256"],
        "E_JOINT_PRODUCER_ARTIFACT_BINDING",
    )
    common.exact_keys(
        capture_plan,
        {
            "commands",
            "history",
            "mechanism_commands",
            "model_id",
            "model_sha256",
            "phase",
            "schema",
        },
        "joint.capture_plan",
    )
    common.exact(
        capture_plan["schema"],
        "s39-cp0-r1-v24-joint-capture-plan-v1",
        "E_JOINT_CAPTURE_PLAN_SCHEMA",
    )
    common.exact(capture_plan["phase"], PHASE, "E_JOINT_CAPTURE_PLAN_PHASE")
    common.exact(
        capture_plan["model_id"],
        MODEL_ID,
        "E_JOINT_CAPTURE_PLAN_MODEL",
    )
    model = next(value for value in candidate["models"] if value["slot"] == "A")
    common.exact(
        capture_plan["model_sha256"],
        model["artifact"]["sha256"],
        "E_JOINT_CAPTURE_PLAN_MODEL_SHA256",
    )
    history_record = common.exact_keys(
        capture_plan["history"],
        {"bytes", "path", "sha256"},
        "joint.capture_plan.history",
    )
    common.absolute_path(
        history_record["path"],
        "joint.capture_plan.history.path",
    )
    common.exact(
        history_record["bytes"],
        len(history_raw),
        "E_JOINT_CAPTURE_PLAN_HISTORY_BYTES",
    )
    common.exact(
        history_record["sha256"],
        common.sha256_bytes(history_raw),
        "E_JOINT_CAPTURE_PLAN_HISTORY_SHA256",
    )
    common.exact(
        receipt["command_plan_sha256"],
        common.sha256_bytes(common.canonical_bytes(runtime_plan)),
        "E_JOINT_RUNTIME_PLAN_SHA256",
    )
    common.exact(
        common.sha256_bytes(
            common.canonical_bytes(capture_plan["mechanism_commands"])
        ),
        receipt["mechanism_commands_sha256"],
        "E_JOINT_CAPTURE_PLAN_MECHANISM",
    )
    commands = common.exact_keys(
        capture_plan["commands"],
        {"cuda", "phone"},
        "joint.capture_plan.commands",
    )
    common.exact_keys(
        receipt["executed_file_artifacts"],
        {"cuda", "phone"},
        "joint.executed_file_artifacts",
    )
    placeholders = {
        "{acquisition_started_ns}",
        "{command_plan_sha256}",
        "{output_path}",
        "{phase_id}",
        "{pre_dir}",
    }
    for name in ("cuda", "phone"):
        field = f"joint.capture_plan.commands.{name}"
        command = common.exact_keys(
            commands[name],
            {
                "argv_template",
                "cwd",
                "environment",
                "executed_files",
                "launch_plan_argv_index",
                "launch_plan_sha256",
                "producer_sha256",
                "result_filename",
                "timeout_seconds",
            },
            field,
        )
        common.exact(
            receipt["executed_file_artifacts"][name],
            command["executed_files"],
            f"E_JOINT_EXECUTED_FILE_PROJECTION: {name}",
        )
        argv_template = command["argv_template"]
        common.require(
            type(argv_template) is list
            and bool(argv_template)
            and all(type(value) is str and bool(value) for value in argv_template),
            f"E_JOINT_COMMAND_TEMPLATE: {name}",
        )
        used_placeholders = {
            value
            for value in argv_template
            if value.startswith("{") and value.endswith("}")
        }
        common.exact(
            used_placeholders,
            placeholders,
            f"E_JOINT_COMMAND_PLACEHOLDERS: {name}",
        )
        evidence = receipt[f"{name}_evidence"]
        subreceipt = evidence["receipt"]
        actual_argv = subreceipt["argv"]
        common.require(
            type(actual_argv) is list
            and len(actual_argv) == len(argv_template),
            f"E_JOINT_COMMAND_ARGV_LENGTH: {name}",
        )
        common.exact(
            subreceipt["cwd"],
            command["cwd"],
            f"E_JOINT_COMMAND_CWD: {name}",
        )
        common.exact(
            subreceipt["environment"],
            command["environment"],
            f"E_JOINT_COMMAND_ENVIRONMENT: {name}",
        )
        common.integer(
            command["timeout_seconds"],
            f"{field}.timeout_seconds",
            1,
        )
        result_filename = common.text(
            command["result_filename"],
            f"{field}.result_filename",
        )
        common.require(
            Path(result_filename).name == result_filename,
            f"E_JOINT_COMMAND_RESULT_FILENAME: {name}",
        )
        records = command["executed_files"]
        common.require(
            type(records) is list and bool(records),
            f"E_JOINT_EXECUTED_FILES: {name}",
        )
        by_index: dict[int, dict[str, Any]] = {}
        for offset, record in enumerate(records):
            item = f"{field}.executed_files[{offset}]"
            common.exact_keys(
                record,
                {"argv_index", "bytes", "path", "sha256"},
                item,
            )
            index = common.integer(record["argv_index"], f"{item}.argv_index")
            common.require(
                index < len(argv_template) and index not in by_index,
                f"E_JOINT_EXECUTED_FILE_INDEX: {name}.{index}",
            )
            by_index[index] = record
            common.exact(
                argv_template[index],
                common.absolute_path(record["path"], f"{item}.path"),
                f"E_JOINT_EXECUTED_FILE_TEMPLATE: {name}.{index}",
            )
            copied_path = Path(
                common.absolute_path(
                    actual_argv[index],
                    f"joint.{name}.executed_argv[{index}]",
                )
            ).resolve()
            common.require(
                copied_path.is_relative_to(bundle_root.resolve()),
                f"E_JOINT_EXECUTED_FILE_PATH: {name}.{index}",
            )
            copied_raw = _read_stable_regular(
                copied_path,
                f"joint.{name}.executed_file[{index}]",
            )
            common.exact(
                len(copied_raw),
                common.integer(record["bytes"], f"{item}.bytes", 1),
                f"E_JOINT_EXECUTED_FILE_BYTES: {name}.{index}",
            )
            common.exact(
                common.sha256_bytes(copied_raw),
                common.digest(record["sha256"], f"{item}.sha256"),
                f"E_JOINT_EXECUTED_FILE_SHA256: {name}.{index}",
            )
        common.require(0 in by_index, f"E_JOINT_EXECUTED_ENTRYPOINT: {name}")
        source_name = "cuda_route" if name == "cuda" else "phone_route"
        common.exact(
            by_index[0]["sha256"],
            contract["producer_requirements"]["source_programs"][source_name][
                "sha256"
            ],
            f"E_JOINT_EXECUTED_ENTRYPOINT_SHA256: {name}",
        )
        launch_index = common.integer(
            command["launch_plan_argv_index"],
            f"{field}.launch_plan_argv_index",
        )
        common.require(
            launch_index in by_index,
            f"E_JOINT_EXECUTED_LAUNCH_INDEX: {name}",
        )
        common.exact(
            by_index[launch_index]["sha256"],
            command["launch_plan_sha256"],
            f"E_JOINT_EXECUTED_LAUNCH_SHA256: {name}",
        )
        common.exact(
            command["producer_sha256"],
            by_index[0]["sha256"],
            f"E_JOINT_EXECUTED_PRODUCER_SHA256: {name}",
        )
        replacements = {
            "{acquisition_started_ns}": str(acquisition_started_ns),
            "{command_plan_sha256}": receipt["command_plan_sha256"],
            "{output_path}": evidence["fragment"]["path"],
            "{phase_id}": receipt["phase_id"],
            "{pre_dir}": _argv_value(
                actual_argv,
                "--pre-dir",
                f"joint.{name}.argv",
            ),
        }
        expected_argv = [
            (
                actual_argv[index]
                if index in by_index
                else replacements.get(value, value)
            )
            for index, value in enumerate(argv_template)
        ]
        common.exact(
            actual_argv,
            expected_argv,
            f"E_JOINT_EXECUTED_ARGV: {name}",
        )
    return capture_plan


def _validate_capture_execution_groups(
    values: Any,
    history: dict[str, Any],
    route_epoch: int,
    field: str,
) -> dict[int, list[int]]:
    groups = history["quality_groups"]
    common.require(
        type(values) is list and len(values) == len(groups),
        f"E_CAPTURE_EXECUTION_GROUPS: {field}",
    )
    requests = {
        request["item_index"]: request
        for request in history["requests"]
    }
    all_continuations: dict[int, list[int]] = {}
    next_frame = 0
    for group_index, (value, expected_group) in enumerate(zip(values, groups)):
        group_field = f"{field}[{group_index}]"
        common.exact_keys(
            value,
            {
                "call_receipts",
                "continuations",
                "group_index",
                "item_indices",
                "wire_request_ids",
            },
            group_field,
        )
        common.exact(value["group_index"], group_index, f"{group_field}.group_index")
        common.exact(
            value["item_indices"],
            expected_group["item_indices"],
            f"{group_field}.item_indices",
        )
        wires = value["wire_request_ids"]
        common.require(
            type(wires) is list
            and len(wires) == 8
            and len(set(wires)) == 8
            and all(type(wire) is int and wire > 0 for wire in wires),
            f"E_CAPTURE_WIRE_IDS: {group_field}",
        )
        expected_calls = [
            ("prefill", call)
            for call in expected_group["prefill_partitions"]
        ] + [
            ("decode", call)
            for call in expected_group["decode_calls"]
        ]
        receipts = value["call_receipts"]
        common.require(
            type(receipts) is list and len(receipts) == len(expected_calls),
            f"E_CAPTURE_CALL_RECEIPTS: {group_field}",
        )
        current: dict[int, int] = {}
        continuations = [[] for _ in range(8)]
        for call_offset, (receipt, expected_call) in enumerate(
            zip(receipts, expected_calls)
        ):
            phase, call = expected_call
            call_field = f"{group_field}.calls[{call_offset}]"
            common.exact_keys(
                receipt,
                {"call_index", "frame_call_index", "phase", "rows"},
                call_field,
            )
            common.exact(receipt["call_index"], call["call_index"], f"{call_field}.call_index")
            common.exact(receipt["frame_call_index"], next_frame, f"{call_field}.frame_call_index")
            next_frame += 1
            common.exact(receipt["phase"], phase, f"{call_field}.phase")
            rows = receipt["rows"]
            common.require(
                type(rows) is list and len(rows) == len(call["rows"]),
                f"E_CAPTURE_CALL_ROWS: {call_field}",
            )
            next_tokens: dict[int, int] = {}
            for row_index, (row, expected_row) in enumerate(zip(rows, call["rows"])):
                row_field = f"{call_field}.rows[{row_index}]"
                common.exact_keys(
                    row,
                    {
                        "input_token",
                        "item_index",
                        "output_token",
                        "position",
                        "request_id",
                        "route_epoch",
                        "seq_id",
                        "wire_request_id",
                    },
                    row_field,
                )
                for key in ("item_index", "position", "request_id", "seq_id"):
                    common.exact(row[key], expected_row[key], f"{row_field}.{key}")
                sequence = row["seq_id"]
                common.exact(
                    row["wire_request_id"],
                    wires[sequence],
                    f"{row_field}.wire_request_id",
                )
                common.exact(row["route_epoch"], route_epoch, f"{row_field}.route_epoch")
                output = common.integer(row["output_token"], f"{row_field}.output")
                if phase == "prefill":
                    common.exact(
                        row["input_token"],
                        expected_row["token_id"],
                        f"{row_field}.prefill_input",
                    )
                    request = requests[row["item_index"]]
                    if row["position"] == len(request["token_ids"]) - 1:
                        common.require(
                            sequence not in current,
                            f"E_CAPTURE_FINAL_PREFILL_REUSE: {row_field}",
                        )
                        current[sequence] = output
                else:
                    common.require(sequence in current, f"E_CAPTURE_DECODE_STATE: {row_field}")
                    common.exact(
                        row["input_token"],
                        current[sequence],
                        f"{row_field}.decode_input",
                    )
                    common.require(
                        sequence not in next_tokens,
                        f"E_CAPTURE_DECODE_SEQUENCE_REUSE: {row_field}",
                    )
                    next_tokens[sequence] = output
            if phase == "decode":
                common.exact(set(next_tokens), set(range(8)), f"{call_field}.decode_outputs")
                current = next_tokens
                for sequence in range(8):
                    continuations[sequence].append(current[sequence])
        first_outputs: dict[int, int] = {}
        for receipt in receipts:
            if receipt["phase"] != "prefill":
                continue
            for row in receipt["rows"]:
                request = requests[row["item_index"]]
                if row["position"] == len(request["token_ids"]) - 1:
                    first_outputs[row["seq_id"]] = row["output_token"]
        common.exact(set(first_outputs), set(range(8)), f"{group_field}.first_outputs")
        for sequence in range(8):
            continuations[sequence].insert(0, first_outputs[sequence])
        common.exact(value["continuations"], continuations, f"{group_field}.continuations")
        common.require(
            all(len(tokens) == 8 for tokens in continuations),
            f"E_CAPTURE_CONTINUATION_LENGTH: {group_field}",
        )
        for sequence, item_index in enumerate(expected_group["item_indices"]):
            all_continuations[item_index] = continuations[sequence]
    common.exact(set(all_continuations), set(range(64)), f"{field}.items")
    return all_continuations


def _validate_fragment_event_intervals(
    fragment: dict[str, Any],
    scalar_fields: tuple[str, ...],
    row_fields: tuple[str, ...],
    field: str,
) -> None:
    started = common.integer(fragment["started_ns"], f"{field}.started_ns", 1)
    completed = common.integer(
        fragment["completed_ns"],
        f"{field}.completed_ns",
        1,
    )
    common.require(started < completed, f"E_JOINT_EVENT_INTERVAL: {field}")
    for key in scalar_fields:
        row = fragment[key]
        common.require(type(row) is dict, f"E_JOINT_EVENT_ROW: {field}.{key}")
        event = common.integer(
            row.get("event_ns"),
            f"{field}.{key}.event_ns",
            1,
        )
        common.require(
            started <= event <= completed,
            f"E_JOINT_EVENT_INTERVAL: {field}.{key}",
        )
    for key in row_fields:
        rows = fragment[key]
        common.require(
            type(rows) is list and bool(rows),
            f"E_JOINT_EVENT_ROWS: {field}.{key}",
        )
        previous = None
        for index, row in enumerate(rows):
            item = f"{field}.{key}[{index}]"
            common.require(type(row) is dict, f"E_JOINT_EVENT_ROW: {item}")
            event = common.integer(row.get("event_ns"), f"{item}.event_ns", 1)
            common.require(
                started <= event <= completed,
                f"E_JOINT_EVENT_INTERVAL: {item}",
            )
            if previous is not None:
                common.require(
                    previous <= event,
                    f"E_JOINT_EVENT_ORDER: {field}.{key}",
                )
            previous = event


def _validate_cuda_route_model_binding(
    value: Any,
    process: dict[str, Any],
    model_component: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    common.exact_keys(
        value,
        {
            "model_mapping_rows",
            "model_path",
            "model_sha256",
            "other_gguf_mapping_paths",
            "pid",
            "start_ticks",
        },
        "joint.cuda.runtime_model_binding",
    )
    common.exact(value["pid"], process["pid"], "E_JOINT_CUDA_MODEL_PID")
    common.exact(value["start_ticks"], process["start_ticks"], "E_JOINT_CUDA_MODEL_START")
    common.exact(value["model_path"], model_component["path"], "E_JOINT_CUDA_MODEL_PATH")
    common.exact(value["model_sha256"], model_component["sha256"], "E_JOINT_CUDA_MODEL_SHA")
    common.exact(value["other_gguf_mapping_paths"], [], "E_JOINT_CUDA_OTHER_GGUF")
    rows = value["model_mapping_rows"]
    expected_rows = contract["cuda_monolithic_identity"]["maps_exact_rows"]
    common.require(
        type(rows) is list and len(rows) == len(expected_rows),
        "E_JOINT_CUDA_MODEL_MAP_COUNT",
    )
    normalized = []
    identities = set()
    for index, row in enumerate(rows):
        item = f"joint.cuda.runtime_model_binding.rows[{index}]"
        common.exact_keys(
            row,
            {
                "address_range",
                "device_major",
                "device_minor",
                "inode",
                "offset_bytes",
                "path",
                "permissions",
            },
            item,
        )
        match = re.fullmatch(
            r"([0-9a-f]+)-([0-9a-f]+)",
            common.text(row["address_range"], f"{item}.address_range"),
        )
        common.require(
            match is not None
            and int(match.group(1), 16) < int(match.group(2), 16),
            f"E_JOINT_CUDA_MODEL_ADDRESS: {index}",
        )
        common.exact(row["path"], model_component["path"], f"{item}.path")
        common.exact(
            row["device_major"],
            os.major(model_component["stat"]["device_id"]),
            f"{item}.device_major",
        )
        common.exact(
            row["device_minor"],
            os.minor(model_component["stat"]["device_id"]),
            f"{item}.device_minor",
        )
        common.exact(row["inode"], model_component["stat"]["inode"], f"{item}.inode")
        identity = (
            row["device_major"],
            row["device_minor"],
            row["inode"],
            common.integer(row["offset_bytes"], f"{item}.offset_bytes"),
            common.text(row["permissions"], f"{item}.permissions"),
            row["path"],
        )
        common.require(identity not in identities, f"E_JOINT_CUDA_MODEL_MAP_DUPLICATE: {index}")
        identities.add(identity)
        normalized.append(
            {
                "offset_bytes": identity[3],
                "permissions": identity[4],
            }
        )
    common.exact(normalized, expected_rows, "E_JOINT_CUDA_MODEL_MAP_ROWS")


def _validate_phone_certificates(
    evidence: dict[str, Any],
    runtime: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    sessions = common.exact_keys(
        evidence["session_certificates"],
        {"op12", "op15"},
        "joint.phone.sessions",
    )
    placements = common.exact_keys(
        evidence["placement_certificates"],
        {"op12", "op15"},
        "joint.phone.placements",
    )
    processes = _runtime_processes_by_id(runtime)
    expected = {
        "op15": {
            "bundle_id": "op15_stagenet",
            "executed_layers": [0, 30],
            "mode": "stagenet",
            "role": "phone_stage",
        },
        "op12": {
            "bundle_id": "op12_stagenet",
            "executed_layers": [30, 40],
            "mode": "tailv3",
            "role": "host_tail_v3",
        },
    }
    for phone in ("op15", "op12"):
        session = common.exact_keys(
            sessions[phone],
            {
                "compute_by_op_and_buffer",
                "device_boot_id",
                "expected_backend",
                "layer_end",
                "layer_start",
                "missing_buffer_compute_nodes",
                "n_layer",
                "placement_status",
                "proto_version",
                "reset_applied",
                "schema",
                "session_end",
                "session_id",
                "steps_session",
                "steps_total",
                "worker_boot_nonce",
                "worker_pid",
            },
            f"joint.phone.session.{phone}",
        )
        process = processes[expected[phone]["bundle_id"]]
        layer_start, layer_end = expected[phone]["executed_layers"]
        for key, value in (
            ("schema", "ls-stagenet-session-v2"),
            ("device_boot_id", process["boot_id"]),
            ("expected_backend", "GPUOpenCL"),
            ("layer_start", layer_start),
            ("layer_end", layer_end),
            ("n_layer", 40),
            ("worker_pid", process["pid"]),
            ("missing_buffer_compute_nodes", 0),
            ("placement_status", "SCHEDULED_PLACEMENT_OK"),
            ("proto_version", 2),
            ("reset_applied", False),
            ("session_end", "STOP"),
            ("session_id", 1),
        ):
            common.exact(
                session[key],
                value,
                f"E_PHONE_SESSION: {phone}.{key}",
            )
        common.require(
            common.integer(session["steps_session"], f"phone.{phone}.steps_session", 1)
            == common.integer(session["steps_total"], f"phone.{phone}.steps_total", 1),
            f"E_PHONE_SESSION_STEPS: {phone}",
        )
        nonce = common.text(session["worker_boot_nonce"], f"phone.{phone}.nonce")
        common.require(
            len(nonce) == 16
            and all(character in "0123456789abcdef" for character in nonce),
            f"E_PHONE_SESSION_NONCE: {phone}",
        )
        op_map = session["compute_by_op_and_buffer"]
        common.require(type(op_map) is dict and bool(op_map), f"E_PHONE_SESSION_OPS: {phone}")
        session_count = 0
        for operation, backends in op_map.items():
            common.require(
                type(operation) is str
                and bool(operation)
                and type(backends) is dict
                and bool(backends),
                f"E_PHONE_SESSION_OP_MAP: {phone}",
            )
            for backend, count in backends.items():
                common.require(
                    backend == "OpenCL"
                    or (
                        phone == "op15"
                        and operation == "GET_ROWS"
                        and backend == "CPU"
                    ),
                    f"E_PHONE_SESSION_FALLBACK: {phone}.{operation}.{backend}",
                )
                session_count += common.integer(
                    count,
                    f"phone.{phone}.session.{operation}.{backend}",
                    1,
                )
        placement = common.exact_keys(
            placements[phone],
            {
                "compute_by_buffer_type",
                "compute_by_op",
                "compute_by_op_and_buffer",
                "compute_nodes",
                "copy_by_buffer_type",
                "copy_nodes",
                "layer_end",
                "layer_start",
                "metadata_nodes",
                "missing_buffer_compute_nodes",
                "mode",
                "n_layer",
                "pid",
                "role",
                "run_rc",
                "schema",
                "status",
            },
            f"joint.phone.placement.{phone}",
        )
        for key, value in (
            ("schema", "layersplit-scheduled-placement-v2"),
            ("layer_start", layer_start),
            ("layer_end", layer_end),
            ("n_layer", 40),
            ("pid", process["pid"]),
            ("missing_buffer_compute_nodes", 0),
            ("mode", expected[phone]["mode"]),
            ("role", expected[phone]["role"]),
            ("run_rc", 0),
            ("status", "SCHEDULED_PLACEMENT_OK"),
        ):
            common.exact(
                placement[key],
                value,
                f"E_PHONE_PLACEMENT: {phone}.{key}",
            )
        common.exact(
            placement["compute_by_op_and_buffer"],
            session["compute_by_op_and_buffer"],
            f"E_PHONE_PLACEMENT_SESSION_OPS: {phone}",
        )
        common.exact(
            placement["compute_nodes"],
            session_count,
            f"E_PHONE_PLACEMENT_NODE_COUNT: {phone}",
        )


def _validate_phone_direct_evidence(
    evidence: dict[str, Any],
    model_sha256: str,
    route_transfer_rows: list[dict[str, Any]],
) -> None:
    frames = evidence["direct_frames"]
    common.require(type(frames) is list and bool(frames), "E_PHONE_DIRECT_FRAMES")
    total_bytes = 0
    total_rows = 0
    for index, frame in enumerate(frames):
        item = f"joint.phone.direct_frames[{index}]"
        common.exact_keys(
            frame,
            {
                "activation_payload_bytes",
                "call_index",
                "hidden_width",
                "payload_sha256",
                "positions",
                "request_ids",
                "route_epochs",
                "rows",
                "schema",
                "seq_ids",
            },
            item,
        )
        common.exact(frame["schema"], "ls-stage-direct-frame-v1", f"{item}.schema")
        common.exact(frame["call_index"], index, f"{item}.call_index")
        rows = common.integer(frame["rows"], f"{item}.rows", 1)
        common.exact(frame["hidden_width"], 5120, f"{item}.hidden_width")
        common.exact(
            frame["activation_payload_bytes"],
            rows * 5120 * 4,
            f"{item}.payload_bytes",
        )
        common.digest(frame["payload_sha256"], f"{item}.payload_sha256")
        for key in ("positions", "request_ids", "route_epochs", "seq_ids"):
            common.require(
                type(frame[key]) is list and len(frame[key]) == rows,
                f"E_PHONE_DIRECT_VECTOR: {item}.{key}",
            )
        total_bytes += frame["activation_payload_bytes"]
        total_rows += rows
    certificate = common.exact_keys(
        evidence["direct_certificate"],
        {
            "activation_payload_bytes",
            "batches",
            "cut_layer",
            "file_type",
            "head_endpoint",
            "host_activation_payload_bytes",
            "layer_end",
            "layer_start",
            "model_sha256",
            "n_embd",
            "n_layer",
            "rows",
            "run_rc",
            "schema",
            "status",
            "tail_endpoint",
        },
        "joint.phone.direct_certificate",
    )
    for key, value in (
        ("activation_payload_bytes", total_bytes),
        ("batches", len(frames)),
        ("cut_layer", 30),
        ("file_type", 15),
        ("host_activation_payload_bytes", 0),
        ("layer_end", 40),
        ("layer_start", 0),
        ("model_sha256", model_sha256),
        ("n_embd", 5120),
        ("n_layer", 40),
        ("rows", total_rows),
        ("run_rc", 0),
        ("schema", "ls-stage-direct-relay-v1"),
        ("status", "DIRECT_RELAY_OK"),
    ):
        common.exact(
            certificate[key],
            value,
            f"E_PHONE_DIRECT_CERTIFICATE: {key}",
        )
    for key in ("head_endpoint", "tail_endpoint"):
        common.text(certificate[key], f"joint.phone.direct_certificate.{key}")
    common.require(
        type(route_transfer_rows) is list and len(route_transfer_rows) > 1,
        "E_PHONE_ROUTE_TRANSFER_ROWS",
    )
    transfer_meta = route_transfer_rows[0]
    common.exact(
        {
            key: transfer_meta.get(key)
            for key in (
                "batch",
                "cut_layer",
                "kind",
                "model_id",
                "model_sha256",
                "request_ids",
            )
        },
        {
            "batch": 8,
            "cut_layer": 30,
            "kind": "meta",
            "model_id": MODEL_ID,
            "model_sha256": model_sha256,
            "request_ids": list(range(8)),
        },
        "E_PHONE_ROUTE_TRANSFER_META",
    )
    transfer_rows = route_transfer_rows[1:]
    common.require(
        len(transfer_rows) <= len(frames),
        "E_PHONE_ROUTE_TRANSFER_COUNT",
    )
    for index, (row, frame) in enumerate(zip(transfer_rows, frames)):
        item = f"joint.phone.route_transfer[{index}]"
        common.exact_keys(
            row,
            {
                "call_index",
                "event_ns",
                "host_payload_bytes",
                "kind",
                "path",
                "payload_bytes",
                "payload_sha256",
                "receiver",
                "row_count",
                "sender",
            },
            item,
        )
        for key, value in (
            ("call_index", index),
            ("host_payload_bytes", 0),
            ("kind", "transfer"),
            ("path", "WIFI_TCP_DIRECT"),
            ("payload_bytes", frame["activation_payload_bytes"]),
            ("payload_sha256", frame["payload_sha256"]),
            ("receiver", "op12"),
            ("row_count", frame["rows"]),
            ("sender", "op15"),
        ):
            common.exact(row[key], value, f"E_PHONE_ROUTE_TRANSFER: {index}.{key}")


def _validate_phone_raw_probes(
    evidence: dict[str, Any],
    launch: dict[str, Any],
    runtime: dict[str, Any],
    contract: dict[str, Any],
    started_ns: int,
    completed_ns: int,
) -> None:
    probes = common.exact_keys(
        evidence["raw_probes"],
        {"op12", "op15"},
        "joint.phone.raw_probes",
    )
    phones = common.exact_keys(
        launch["phones"],
        {"op12", "op15"},
        "joint.phone.launch.phones",
    )
    processes = _runtime_processes_by_id(runtime)
    bundles = {"op12": "op12_stagenet", "op15": "op15_stagenet"}
    payload_bytes = evidence["direct_certificate"]["activation_payload_bytes"]
    checked: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    probe_keys = {
        "active_sequences",
        "available_bytes",
        "boot_id",
        "device",
        "direct_peer",
        "gpu_max_millic",
        "interface",
        "loaded_shard_path",
        "loaded_shard_sha256",
        "model",
        "model_id",
        "model_sha256",
        "process_swap_bytes",
        "product",
        "schema",
        "serial",
        "system_swap_used_bytes",
        "thermal_status",
        "worker_executable_path",
        "worker_executable_sha256",
        "worker_pid",
        "worker_start_ticks",
    }
    for phone in ("op12", "op15"):
        bundle = common.exact_keys(
            probes[phone],
            {"after", "after_ns", "before", "before_ns"},
            f"joint.phone.raw_probes.{phone}",
        )
        before_ns = common.integer(
            bundle["before_ns"],
            f"joint.phone.raw_probes.{phone}.before_ns",
            1,
        )
        after_ns = common.integer(
            bundle["after_ns"],
            f"joint.phone.raw_probes.{phone}.after_ns",
            1,
        )
        common.require(
            started_ns <= before_ns < after_ns <= completed_ns,
            f"E_PHONE_PROBE_INTERVAL: {phone}",
        )
        expected = phones[phone]
        process = processes[bundles[phone]]
        values = []
        for when in ("before", "after"):
            value = common.exact_keys(
                bundle[when],
                probe_keys,
                f"joint.phone.raw_probes.{phone}.{when}",
            )
            for key, expected_value in (
                ("schema", "s39-cp0-r1-v24-phone-runtime-probe-v1"),
                ("boot_id", process["boot_id"]),
                ("device", contract["devices"][phone]["device"]),
                ("model", contract["devices"][phone]["model"]),
                ("product", contract["devices"][phone]["product"]),
                ("serial", contract["devices"][phone]["serial"]),
                ("model_id", MODEL_ID),
                ("model_sha256", evidence["direct_certificate"]["model_sha256"]),
                ("worker_pid", process["pid"]),
                ("worker_start_ticks", process["start_ticks"]),
                (
                    "worker_executable_path",
                    expected["expected_worker_executable_path"],
                ),
                (
                    "worker_executable_sha256",
                    expected["expected_worker_executable_sha256"],
                ),
                ("loaded_shard_path", expected["loaded_shard_path"]),
                ("loaded_shard_sha256", expected["loaded_shard_sha256"]),
                ("active_sequences", 0),
                ("process_swap_bytes", 0),
                ("system_swap_used_bytes", 0),
                ("thermal_status", 0),
            ):
                common.exact(
                    value[key],
                    expected_value,
                    f"E_PHONE_PROBE: {phone}.{when}.{key}",
                )
            common.require(
                common.integer(
                    value["available_bytes"],
                    f"joint.phone.raw_probes.{phone}.{when}.available_bytes",
                )
                >= contract["gates"]["phone_minimum_available_bytes"],
                f"E_PHONE_PROBE_HEADROOM: {phone}.{when}",
            )
            common.integer(
                value["gpu_max_millic"],
                f"joint.phone.raw_probes.{phone}.{when}.gpu_max_millic",
                1,
            )
            interface = common.exact_keys(
                value["interface"],
                {"ipv4", "name", "rx_bytes", "tx_bytes"},
                f"joint.phone.raw_probes.{phone}.{when}.interface",
            )
            for key, expected_value in (
                ("ipv4", expected["local_ipv4"]),
                ("name", expected["interface"]),
            ):
                common.exact(
                    interface[key],
                    expected_value,
                    f"E_PHONE_PROBE_INTERFACE: {phone}.{when}.{key}",
                )
            for key in ("rx_bytes", "tx_bytes"):
                common.integer(
                    interface[key],
                    f"joint.phone.raw_probes.{phone}.{when}.interface.{key}",
                )
            peer = common.exact_keys(
                value["direct_peer"],
                {
                    "interface",
                    "local_ipv4",
                    "peer_ipv4",
                    "socket_peer_observed",
                },
                f"joint.phone.raw_probes.{phone}.{when}.direct_peer",
            )
            for key, expected_value in (
                ("interface", expected["interface"]),
                ("local_ipv4", expected["local_ipv4"]),
                ("peer_ipv4", expected["direct_peer_ipv4"]),
                ("socket_peer_observed", True),
            ):
                common.exact(
                    peer[key],
                    expected_value,
                    f"E_PHONE_PROBE_PEER: {phone}.{when}.{key}",
                )
            values.append(value)
        before, after = values
        for key in ("rx_bytes", "tx_bytes"):
            common.require(
                after["interface"][key] >= before["interface"][key],
                f"E_PHONE_PROBE_COUNTER_RESET: {phone}.{key}",
            )
        checked[phone] = (before, after)
    common.require(
        checked["op15"][1]["interface"]["tx_bytes"]
        - checked["op15"][0]["interface"]["tx_bytes"]
        >= payload_bytes,
        "E_PHONE_PROBE_OP15_TX",
    )
    common.require(
        checked["op12"][1]["interface"]["rx_bytes"]
        - checked["op12"][0]["interface"]["rx_bytes"]
        >= payload_bytes,
        "E_PHONE_PROBE_OP12_RX",
    )


def _validate_joint_cuda_launch(
    launch: dict[str, Any],
    model_component: dict[str, Any],
    history_raw: bytes,
    plan: dict[str, Any],
    runtime: dict[str, Any],
    outer: dict[str, Any],
) -> None:
    common.exact(
        launch.get("schema"),
        "s39-cp0-r1-v24-cuda-route-launch-v1",
        "E_JOINT_CUDA_LAUNCH_SCHEMA",
    )
    for key, value in (
        ("model_id", MODEL_ID),
        ("model_sha256", model_component["sha256"]),
        ("expected_file_type", 15),
        ("expected_n_layer", 40),
        ("expected_n_embd", 5120),
        ("expected_max_streams", 8),
        ("expected_n_ctx_seq", 512),
        ("expected_n_batch", 64),
        ("expected_n_ubatch", 64),
        ("expected_capabilities", 0x3F),
        ("history_path", plan["token_history"]["artifact_path"]),
        ("history_sha256", common.sha256_bytes(history_raw)),
        ("route_epoch", outer["route_epoch"]),
    ):
        common.exact(launch.get(key), value, f"E_JOINT_CUDA_LAUNCH: {key}")
    common.exact(
        launch.get("model_artifact"),
        {
            "bytes": model_component["bytes"],
            "path": model_component["path"],
            "sha256": model_component["sha256"],
            "stat": model_component["stat"],
        },
        "E_JOINT_CUDA_LAUNCH_MODEL",
    )
    common.exact(
        common.sha256_bytes(common.canonical_bytes(launch.get("mechanism_commands"))),
        outer["mechanism_commands_sha256"],
        "E_JOINT_CUDA_LAUNCH_MECHANISM",
    )
    worker = launch.get("worker")
    common.require(type(worker) is dict, "E_JOINT_CUDA_WORKER")
    processes = _runtime_processes_by_id(runtime)
    process = processes["cuda_route"]
    runtime_executable = worker.get("runtime_executable")
    common.require(type(runtime_executable) is dict, "E_JOINT_CUDA_RUNTIME_EXECUTABLE")
    common.exact(
        runtime_executable.get("path"),
        process["launcher_path"],
        "E_JOINT_CUDA_WORKER_PATH",
    )
    common.exact(
        worker.get("runtime_component_ids"),
        process["loaded_repo_component_ids"],
        "E_JOINT_CUDA_WORKER_COMPONENTS",
    )
    environment = worker.get("environment")
    common.require(type(environment) is dict, "E_JOINT_CUDA_WORKER_ENV")
    for key, value in (
        ("LAYERSPLIT_MODEL_SHA256", model_component["sha256"]),
        ("LAYERSPLIT_MEMORY_CERT", "1"),
        ("LAYERSPLIT_PLACEMENT_CERT", "1"),
    ):
        common.exact(environment.get(key), value, f"E_JOINT_CUDA_WORKER_ENV: {key}")
    argv = worker.get("argv")
    common.require(type(argv) is list and bool(argv), "E_JOINT_CUDA_WORKER_ARGV")
    for flag, value in (
        ("--mode", "monov3"),
        ("--backend", "CUDA0"),
        ("--layer-start", "0"),
        ("--layer-end", "40"),
        ("--model", model_component["path"]),
    ):
        common.exact(
            _argv_value(argv, flag, "joint.cuda.launch.worker"),
            value,
            f"E_JOINT_CUDA_WORKER_ARG: {flag}",
        )


def _validate_joint_phone_launch(
    launch: dict[str, Any],
    contract: dict[str, Any],
    history_raw: bytes,
    plan: dict[str, Any],
    runtime: dict[str, Any],
    outer: dict[str, Any],
) -> None:
    common.exact(
        launch.get("schema"),
        "s39-cp0-r1-v24-phone-route-launch-v1",
        "E_JOINT_PHONE_LAUNCH_SCHEMA",
    )
    model_sha256 = outer["model_sha256"]
    for key, value in (
        ("model_id", MODEL_ID),
        ("model_sha256", model_sha256),
        ("expected_file_type", 15),
        ("expected_n_layer", 40),
        ("expected_n_embd", 5120),
        ("expected_max_streams", 8),
        ("expected_n_ctx_seq", 512),
        ("expected_n_batch", 64),
        ("expected_n_ubatch", 64),
        ("history_path", plan["token_history"]["artifact_path"]),
        ("history_sha256", common.sha256_bytes(history_raw)),
        ("route_epoch", outer["route_epoch"]),
    ):
        common.exact(launch.get(key), value, f"E_JOINT_PHONE_LAUNCH: {key}")
    phones = common.exact_keys(
        launch.get("phones"),
        {"op12", "op15"},
        "joint.phone.launch.phones",
    )
    expected = {
        "op15": {
            "executed_layers": [0, 30],
            "stored_layers": [0, 32],
            "shard": contract["incumbent_route_lock"]["op15_shard_sha256"],
            "bundle": "op15_stagenet",
        },
        "op12": {
            "executed_layers": [30, 40],
            "stored_layers": [24, 40],
            "shard": contract["incumbent_route_lock"]["op12_shard_sha256"],
            "bundle": "op12_stagenet",
        },
    }
    processes = _runtime_processes_by_id(runtime)
    for phone in ("op15", "op12"):
        value = phones[phone]
        for key in ("device", "model", "product", "serial"):
            common.exact(
                value.get(key),
                contract["devices"][phone][key],
                f"E_JOINT_PHONE_LAUNCH_IDENTITY: {phone}.{key}",
            )
        common.exact(
            value.get("boot_id"),
            processes[expected[phone]["bundle"]]["boot_id"],
            f"E_JOINT_PHONE_LAUNCH_BOOT: {phone}",
        )
        common.exact(
            value.get("executed_layers"),
            expected[phone]["executed_layers"],
            f"E_JOINT_PHONE_LAUNCH_EXECUTED: {phone}",
        )
        common.exact(
            value.get("stored_layers"),
            expected[phone]["stored_layers"],
            f"E_JOINT_PHONE_LAUNCH_STORED: {phone}",
        )
        common.exact(
            value.get("loaded_shard_sha256"),
            expected[phone]["shard"],
            f"E_JOINT_PHONE_LAUNCH_SHARD: {phone}",
        )
        common.exact(
            value.get("expected_worker_executable_path"),
            processes[expected[phone]["bundle"]]["launcher_path"],
            f"E_JOINT_PHONE_LAUNCH_WORKER: {phone}",
        )
    common.exact(
        phones["op15"].get("direct_peer_ipv4"),
        phones["op12"].get("local_ipv4"),
        "E_JOINT_PHONE_DIRECT_PEER: op15",
    )
    common.exact(
        phones["op12"].get("direct_peer_ipv4"),
        phones["op15"].get("local_ipv4"),
        "E_JOINT_PHONE_DIRECT_PEER: op12",
    )
    process_specs = common.exact_keys(
        launch.get("processes"),
        {"op12_stagenet", "op15_direct_relay", "op15_stagenet"},
        "joint.phone.launch.processes",
    )
    for bundle_id, spec in process_specs.items():
        process = processes[bundle_id]
        common.exact(
            spec.get("runtime_executable_path"),
            process["launcher_path"],
            f"E_JOINT_PHONE_PROCESS_PATH: {bundle_id}",
        )
        common.exact(
            spec.get("runtime_component_ids"),
            process["loaded_repo_component_ids"],
            f"E_JOINT_PHONE_PROCESS_COMPONENTS: {bundle_id}",
        )


def _artifact_root_components(root: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        value["component_id"]: value
        for value in root["components"]
    }


def _validate_cuda_monolithic_producer_identity(
    receipt: dict[str, Any],
    bundle_root: Path,
    contract: dict[str, Any],
    plan: dict[str, Any],
    runtime_process: dict[str, Any],
) -> None:
    entrypoints = [
        value
        for value in plan["capture_entrypoints"]
        if value["kind"] == "cuda_monolithic"
    ]
    common.require(
        len(entrypoints) == 1,
        "E_CUDA_PRODUCER_ENTRYPOINT",
    )
    component_id = entrypoints[0]["component_id"]
    components = [
        value
        for value in plan["components"]
        if value["component_id"] == component_id
    ]
    common.require(
        len(components) == 1,
        "E_CUDA_PRODUCER_COMPONENT",
    )
    component = components[0]
    source_pin = contract["producer_requirements"]["source_programs"][
        "cuda_monolithic"
    ]
    artifact = common.exact_keys(
        receipt["producer_artifact"],
        {"bytes", "path", "sha256", "stat"},
        "cuda_receipt.producer_artifact",
    )
    source_path = Path(
        common.absolute_path(
            artifact["path"],
            "cuda_receipt.producer_artifact.path",
        )
    ).resolve()
    common.exact(
        str(source_path),
        component["path"],
        "E_CUDA_PRODUCER_COMPONENT_PATH",
    )
    for value, expected, field in (
        (artifact["bytes"], component["bytes"], "component.bytes"),
        (artifact["sha256"], component["sha256"], "component.sha256"),
        (artifact["bytes"], source_pin["bytes"], "contract.bytes"),
        (artifact["sha256"], source_pin["sha256"], "contract.sha256"),
        (receipt["producer_sha256"], artifact["sha256"], "receipt.sha256"),
    ):
        common.exact(value, expected, f"E_CUDA_PRODUCER_BINDING: {field}")
    source_raw, source_stat = _read_stable_regular_with_stat(
        source_path,
        "cuda_receipt.producer_source",
    )
    common.exact(
        len(source_raw),
        common.integer(
            artifact["bytes"],
            "cuda_receipt.producer_artifact.bytes",
            1,
        ),
        "E_CUDA_PRODUCER_SOURCE_BYTES",
    )
    common.exact(
        common.sha256_bytes(source_raw),
        common.digest(
            artifact["sha256"],
            "cuda_receipt.producer_artifact.sha256",
        ),
        "E_CUDA_PRODUCER_SOURCE_SHA256",
    )
    common.exact(
        artifact["stat"],
        source_stat,
        "E_CUDA_PRODUCER_SOURCE_STAT",
    )
    process = common.exact_keys(
        receipt["producer_process_receipt"],
        {
            "argv",
            "boot_id",
            "cwd",
            "pid",
            "schema",
            "source_path",
            "source_sha256",
            "start_ticks",
        },
        "cuda_receipt.producer_process_receipt",
    )
    common.exact(
        process["schema"],
        "s39-cp0-r1-v24-producer-process-receipt-v1",
        "E_CUDA_PRODUCER_PROCESS_SCHEMA",
    )
    common.exact(
        process["source_path"],
        str(source_path),
        "E_CUDA_PRODUCER_PROCESS_SOURCE_PATH",
    )
    common.exact(
        process["source_sha256"],
        artifact["sha256"],
        "E_CUDA_PRODUCER_PROCESS_SOURCE_SHA256",
    )
    common.exact(
        process["boot_id"],
        runtime_process["boot_id"],
        "E_CUDA_PRODUCER_PROCESS_BOOT",
    )
    producer_pid = common.integer(
        process["pid"],
        "cuda_receipt.producer_process_receipt.pid",
        1,
    )
    producer_start = common.integer(
        process["start_ticks"],
        "cuda_receipt.producer_process_receipt.start_ticks",
        1,
    )
    common.require(
        producer_pid != runtime_process["pid"],
        "E_CUDA_PRODUCER_PROCESS_PID_REUSE",
    )
    common.require(
        producer_start <= runtime_process["start_ticks"],
        "E_CUDA_PRODUCER_PROCESS_START",
    )
    common.exact(
        process["cwd"],
        plan["bundle_roots"]["cuda_monolithic"],
        "E_CUDA_PRODUCER_PROCESS_CWD",
    )
    argv = process["argv"]
    flags = (
        "--output",
        "--phase-id",
        "--pre-dir",
        "--started",
        "--plan",
        "--mechanism-commands-sha256",
        "--model-sha256",
        "--histories",
        "--launch-plan",
    )
    common.require(
        type(argv) is list
        and len(argv) == 1 + 2 * len(flags)
        and all(type(value) is str and bool(value) for value in argv),
        "E_CUDA_PRODUCER_PROCESS_ARGV",
    )
    common.exact(argv[0], str(source_path), "E_CUDA_PRODUCER_PROCESS_ARGV0")
    common.exact(
        [argv[index] for index in range(1, len(argv), 2)],
        list(flags),
        "E_CUDA_PRODUCER_PROCESS_FLAGS",
    )
    output_path = Path(
        common.absolute_path(
            _argv_value(argv, "--output", "cuda_receipt.producer.argv"),
            "cuda_receipt.producer.output",
        )
    ).resolve()
    common.require(
        output_path.is_relative_to(bundle_root.resolve()),
        "E_CUDA_PRODUCER_OUTPUT_PATH",
    )
    common.exact(
        receipt["worker_log_path"],
        str(output_path) + ".worker.log",
        "E_CUDA_PRODUCER_OUTPUT_LOG",
    )
    common.exact(
        _argv_value(argv, "--phase-id", "cuda_receipt.producer.argv"),
        receipt["phase_id"],
        "E_CUDA_PRODUCER_ARG_PHASE",
    )
    pre_dir = Path(
        common.absolute_path(
            _argv_value(argv, "--pre-dir", "cuda_receipt.producer.argv"),
            "cuda_receipt.producer.pre_dir",
        )
    )
    common.require(pre_dir.is_dir(), "E_CUDA_PRODUCER_PRE_DIR")
    acquisition_started_text = _argv_value(
        argv,
        "--started",
        "cuda_receipt.producer.argv",
    )
    common.require(
        re.fullmatch(r"[1-9][0-9]*", acquisition_started_text) is not None,
        "E_CUDA_PRODUCER_ARG_STARTED",
    )
    acquisition_started = common.integer(
        int(acquisition_started_text),
        "cuda_receipt.producer.started",
        1,
    )
    common.require(
        acquisition_started <= receipt["started_ns"],
        "E_CUDA_PRODUCER_ACQUISITION_ORDER",
    )
    common.exact(
        _argv_value(argv, "--plan", "cuda_receipt.producer.argv"),
        common.sha256_bytes(common.canonical_bytes(plan)),
        "E_CUDA_PRODUCER_ARG_PLAN",
    )
    common.exact(
        _argv_value(
            argv,
            "--mechanism-commands-sha256",
            "cuda_receipt.producer.argv",
        ),
        receipt["mechanism_commands_sha256"],
        "E_CUDA_PRODUCER_ARG_MECHANISM",
    )
    common.exact(
        _argv_value(argv, "--model-sha256", "cuda_receipt.producer.argv"),
        receipt["model_sha256"],
        "E_CUDA_PRODUCER_ARG_MODEL",
    )
    common.exact(
        _argv_value(argv, "--histories", "cuda_receipt.producer.argv"),
        plan["token_history"]["artifact_path"],
        "E_CUDA_PRODUCER_ARG_HISTORY",
    )
    launch_path = Path(
        common.absolute_path(
            _argv_value(argv, "--launch-plan", "cuda_receipt.producer.argv"),
            "cuda_receipt.producer.launch_plan",
        )
    )
    launch, launch_raw = common.read_canonical(launch_path)
    common.exact(
        launch,
        plan["cuda_monolithic_launch"],
        "E_CUDA_PRODUCER_ARG_LAUNCH",
    )
    common.exact(
        common.sha256_bytes(launch_raw),
        receipt["launch_binding"]["launch_plan_sha256"],
        "E_CUDA_PRODUCER_ARG_LAUNCH_SHA256",
    )


def validate_cuda_monolithic_receipt(
    receipt: dict[str, Any],
    bundle_root: Path,
    contract: dict[str, Any],
    candidate: dict[str, Any],
    history: dict[str, Any],
    history_raw: bytes,
    plan: dict[str, Any],
    runtime: dict[str, Any],
    rows_by_role: dict[str, list[dict[str, Any]]],
) -> None:
    common.exact_keys(
        receipt,
        {
            "completed_ns",
            "history_binding",
            "launch_binding",
            "mechanism_commands_sha256",
            "memory_certificate",
            "model_id",
            "model_sha256",
            "oracle_cuda_monolithic_rows",
            "phase_id",
            "placement_certificate",
            "producer_artifact",
            "producer_process_receipt",
            "producer_sha256",
            "protocol_identity",
            "runtime_model_binding",
            "runtime_process",
            "schema",
            "started_ns",
            "worker_log_bytes",
            "worker_log_path",
            "worker_log_sha256",
        },
        "cuda_receipt",
    )
    common.exact(
        receipt["schema"],
        "s39-cp0-r1-v24-cuda-monolithic-raw-v1",
        "cuda_receipt.schema",
    )
    common.exact(receipt["model_id"], MODEL_ID, "cuda_receipt.model_id")
    model = next(row for row in candidate["models"] if row["slot"] == "A")
    common.exact(
        receipt["model_sha256"],
        model["artifact"]["sha256"],
        "cuda_receipt.model_sha256",
    )
    common.exact(
        receipt["phase_id"],
        runtime["phase_id"],
        "cuda_receipt.phase_id",
    )
    started = common.integer(receipt["started_ns"], "cuda_receipt.started", 1)
    completed = common.integer(receipt["completed_ns"], "cuda_receipt.completed", 1)
    common.require(
        runtime["started_ns"] <= started < completed <= runtime["completed_ns"],
        "E_CUDA_RECEIPT_INTERVAL",
    )
    launch_binding = common.exact_keys(
        receipt["launch_binding"],
        {
            "launch",
            "launch_plan_sha256",
            "runtime_boot_id",
            "runtime_pid",
            "runtime_start_ticks",
        },
        "cuda_receipt.launch_binding",
    )
    common.exact(
        launch_binding["launch"],
        plan["cuda_monolithic_launch"],
        "E_CUDA_RECEIPT_LAUNCH",
    )
    common.exact(
        launch_binding["launch_plan_sha256"],
        common.sha256_bytes(
            common.canonical_bytes(plan["cuda_monolithic_launch"])
        ),
        "E_CUDA_RECEIPT_LAUNCH_SHA256",
    )
    processes = {
        value["bundle_id"]: value for value in runtime["processes"]
    }
    runtime_identity = processes["cuda_monolithic"]
    captured_process = receipt["runtime_process"]
    for key in (
        "boot_id",
        "bundle_id",
        "bundle_sha256",
        "endpoint",
        "launcher_component_id",
        "launcher_path",
        "loaded_repo_component_ids",
        "pid",
        "start_ticks",
    ):
        common.exact(
            captured_process.get(key),
            runtime_identity[key],
            f"E_CUDA_RECEIPT_PROCESS: {key}",
        )
    _validate_cuda_monolithic_producer_identity(
        receipt,
        bundle_root,
        contract,
        plan,
        captured_process,
    )
    common.exact(
        launch_binding["runtime_boot_id"],
        captured_process["boot_id"],
        "E_CUDA_RECEIPT_LAUNCH_BOOT",
    )
    common.exact(
        launch_binding["runtime_pid"],
        captured_process["pid"],
        "E_CUDA_RECEIPT_LAUNCH_PID",
    )
    common.exact(
        launch_binding["runtime_start_ticks"],
        captured_process["start_ticks"],
        "E_CUDA_RECEIPT_LAUNCH_START",
    )
    model_binding = receipt["runtime_model_binding"]
    live_mapping = runtime_identity["model_mapping"]
    common.exact(model_binding.get("argv"), live_mapping["argv"], "E_CUDA_RECEIPT_MODEL_ARGV")
    common.exact(
        model_binding.get("model_mapping_rows"),
        live_mapping["model_mapping_rows"],
        "E_CUDA_RECEIPT_MODEL_MAPS",
    )
    common.exact(model_binding.get("model_path"), live_mapping["model_path"], "E_CUDA_RECEIPT_MODEL_PATH")
    common.exact(model_binding.get("model_sha256"), live_mapping["model_sha256"], "E_CUDA_RECEIPT_MODEL_SHA")
    common.exact(model_binding.get("model_stat"), live_mapping["pre_stat"], "E_CUDA_RECEIPT_MODEL_STAT")
    common.exact(model_binding.get("other_gguf_mapping_paths"), [], "E_CUDA_RECEIPT_OTHER_GGUF")
    for key in ("pid", "start_ticks"):
        common.exact(
            model_binding.get(key),
            runtime_identity[key],
            f"E_CUDA_RECEIPT_MODEL_PROCESS: {key}",
        )
    _validate_protocol_identity(
        receipt["protocol_identity"],
        contract,
        model["artifact"]["sha256"],
    )
    _validate_cuda_placement_certificate(
        receipt["placement_certificate"],
        captured_process,
    )
    _validate_memory_certificate(
        receipt["memory_certificate"],
        captured_process,
        receipt["placement_certificate"],
    )
    raw_rows = rows_by_role[f"model.{MODEL_ID}.oracle.cuda_monolithic"]
    stripped_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"acquisition_id", "phase", "phase_id", "role"}
        }
        for row in raw_rows
    ]
    common.exact(
        receipt["oracle_cuda_monolithic_rows"],
        stripped_rows,
        "E_CUDA_RECEIPT_RAW_ROWS",
    )
    binding = common.exact_keys(
        receipt["history_binding"],
        {
            "corpus_item_indices",
            "histories_sha256",
            "prompt_sha256s",
            "quality_corpus_sha256",
            "token_history_corpus_sha256",
        },
        "cuda_receipt.history_binding",
    )
    common.exact(binding["corpus_item_indices"], list(range(8)), "E_CUDA_RECEIPT_HISTORY_ITEMS")
    common.exact(binding["histories_sha256"], common.sha256_bytes(history_raw), "E_CUDA_RECEIPT_HISTORY_SHA")
    common.exact(binding["quality_corpus_sha256"], contract["quality_corpus"]["sha256"], "E_CUDA_RECEIPT_CORPUS_SHA")
    common.exact(binding["token_history_corpus_sha256"], history["corpus_sha256"], "E_CUDA_RECEIPT_TOKEN_CORPUS")
    common.exact(
        binding["prompt_sha256s"],
        [history["requests"][index]["prompt_sha256"] for index in range(8)],
        "E_CUDA_RECEIPT_PROMPTS",
    )
    common.digest(receipt["mechanism_commands_sha256"], "cuda_receipt.mechanism")
    common.integer(receipt["worker_log_bytes"], "cuda_receipt.worker_log_bytes", 1)
    common.digest(receipt["worker_log_sha256"], "cuda_receipt.worker_log_sha256")
    worker_log_path = Path(
        common.absolute_path(
            receipt["worker_log_path"],
            "cuda_receipt.worker_log_path",
        )
    ).resolve()
    common.require(
        worker_log_path.is_relative_to(bundle_root.resolve()),
        "E_CUDA_RECEIPT_WORKER_LOG_PATH",
    )
    try:
        worker_log_raw = worker_log_path.read_bytes()
    except OSError as error:
        raise common.EvidenceError(
            f"E_CUDA_RECEIPT_WORKER_LOG_READ: {error}"
        ) from error
    common.exact(
        len(worker_log_raw),
        receipt["worker_log_bytes"],
        "E_CUDA_RECEIPT_WORKER_LOG_BYTES",
    )
    common.exact(
        common.sha256_bytes(worker_log_raw),
        receipt["worker_log_sha256"],
        "E_CUDA_RECEIPT_WORKER_LOG_SHA256",
    )


def validate_joint_phone_cuda_receipt(
    receipt: dict[str, Any],
    bundle_root: Path,
    contract: dict[str, Any],
    candidate: dict[str, Any],
    history: dict[str, Any],
    history_raw: bytes,
    plan: dict[str, Any],
    runtime: dict[str, Any],
    artifact_root: dict[str, Any],
    rows_by_role: dict[str, list[dict[str, Any]]],
    acquisition_started_ns: int,
    expected_command_plan_sha256: str | None = None,
) -> None:
    common.exact_keys(
        receipt,
        {
            "bridge_rows",
            "capture_plan_artifact",
            "capture_plan_sha256",
            "command_plan_sha256",
            "completed_ns",
            "cuda_evidence",
            "cuda_memory_rows",
            "cuda_route_rows",
            "fragment_sha256",
            "gpu_runtime",
            "history_sha256",
            "joint_producer_artifact",
            "joint_producer_sha256",
            "mechanics_rows",
            "mechanism_commands_sha256",
            "model_id",
            "model_sha256",
            "op12_runtime",
            "op15_runtime",
            "phase_id",
            "phone_evidence",
            "placement_op12_rows",
            "placement_op15_rows",
            "quality_cuda_rows",
            "quality_phone_rows",
            "route_epoch",
            "route_transfer_rows",
            "runtime_processes",
            "schema",
            "started_ns",
            "subproducer_bindings",
            "executed_file_artifacts",
        },
        "joint_receipt",
    )
    common.exact(
        receipt["schema"],
        "s39-cp0-r1-v24-joint-phone-cuda-raw-v1",
        "joint_receipt.schema",
    )
    common.exact(receipt["phase_id"], runtime["phase_id"], "joint_receipt.phase_id")
    common.exact(receipt["model_id"], MODEL_ID, "joint_receipt.model_id")
    model = next(value for value in candidate["models"] if value["slot"] == "A")
    common.exact(
        receipt["model_sha256"],
        model["artifact"]["sha256"],
        "joint_receipt.model_sha256",
    )
    common.exact(
        receipt["history_sha256"],
        common.sha256_bytes(history_raw),
        "E_JOINT_HISTORY_SHA256",
    )
    started = common.integer(receipt["started_ns"], "joint_receipt.started_ns", 1)
    completed = common.integer(receipt["completed_ns"], "joint_receipt.completed_ns", 1)
    common.require(
        runtime["started_ns"] <= started < completed <= runtime["completed_ns"],
        "E_JOINT_RECEIPT_INTERVAL",
    )
    common.require(
        acquisition_started_ns <= started,
        "E_JOINT_RECEIPT_ACQUISITION_INTERVAL",
    )
    source = contract["producer_requirements"]["source_programs"][
        "joint_phone_cuda"
    ]
    common.exact(
        receipt["joint_producer_sha256"],
        source["sha256"],
        "E_JOINT_PRODUCER_SHA256",
    )
    common.digest(receipt["capture_plan_sha256"], "joint_receipt.capture_plan_sha256")
    command_plan_sha256 = common.digest(
        receipt["command_plan_sha256"],
        "joint_receipt.command_plan_sha256",
    )
    if expected_command_plan_sha256 is not None:
        common.exact(
            command_plan_sha256,
            expected_command_plan_sha256,
            "E_JOINT_ORCHESTRATION_PLAN_SHA256",
        )
    common.digest(
        receipt["mechanism_commands_sha256"],
        "joint_receipt.mechanism_commands_sha256",
    )
    common.integer(receipt["route_epoch"], "joint_receipt.route_epoch", 1)
    common.exact_keys(
        receipt["subproducer_bindings"],
        {"cuda", "phone"},
        "joint_receipt.subproducer_bindings",
    )
    for name in ("cuda", "phone"):
        common.exact_keys(
            receipt["subproducer_bindings"][name],
            {"launch_plan_sha256", "producer_sha256"},
            f"joint_receipt.subproducer_bindings.{name}",
        )
    cuda_evidence = common.exact_keys(
        receipt["cuda_evidence"],
        {
            "fragment",
            "launch_plan_sha256",
            "memory_certificate",
            "placement_certificate",
            "protocol_identity",
            "raw_memory_samples",
            "receipt",
            "receipt_artifact",
            "runtime_model_binding",
            "runtime_process",
        },
        "joint_receipt.cuda_evidence",
    )
    phone_evidence = common.exact_keys(
        receipt["phone_evidence"],
        {
            "direct_certificate",
            "direct_frames",
            "fragment",
            "launch_plan_sha256",
            "placement_certificates",
            "raw_probes",
            "receipt",
            "receipt_artifact",
            "runtime_processes",
            "session_certificates",
        },
        "joint_receipt.phone_evidence",
    )
    _validate_joint_capture_artifacts(
        receipt=receipt,
        bundle_root=bundle_root,
        contract=contract,
        candidate=candidate,
        history_raw=history_raw,
        runtime_plan=plan,
        acquisition_started_ns=acquisition_started_ns,
    )
    cuda_fragment, cuda_launch = _validate_subproducer_receipt(
        bundle_root=bundle_root,
        name="cuda",
        evidence=cuda_evidence,
        outer=receipt,
        contract=contract,
        model_sha256=model["artifact"]["sha256"],
        acquisition_started_ns=acquisition_started_ns,
    )
    phone_fragment, phone_launch = _validate_subproducer_receipt(
        bundle_root=bundle_root,
        name="phone",
        evidence=phone_evidence,
        outer=receipt,
        contract=contract,
        model_sha256=model["artifact"]["sha256"],
        acquisition_started_ns=acquisition_started_ns,
    )
    common.exact_keys(
        cuda_fragment,
        {
            "bridge_ready_row",
            "bridge_start_row",
            "completed_ns",
            "cuda_memory_rows",
            "cuda_route_rows",
            "evidence_artifacts",
            "execution_groups",
            "gpu_runtime",
            "history_sha256",
            "launch_plan_sha256",
            "mechanism_commands_sha256",
            "memory_certificate",
            "model_id",
            "model_sha256",
            "phase_id",
            "placement_certificate",
            "producer_sha256",
            "protocol_identity",
            "quality_cuda_rows",
            "raw_memory_samples",
            "route_epoch",
            "runtime_model_binding",
            "runtime_process",
            "schema",
            "started_ns",
        },
        "joint.cuda.fragment",
    )
    common.exact_keys(
        phone_fragment,
        {
            "bridge_publication_rows",
            "completed_ns",
            "direct_certificate",
            "direct_frames",
            "evidence_artifacts",
            "execution_groups",
            "history_sha256",
            "launch_plan_sha256",
            "mechanics_rows",
            "mechanism_commands_sha256",
            "model_id",
            "model_sha256",
            "op12_runtime",
            "op15_runtime",
            "phase_id",
            "placement_certificates",
            "placement_op12_rows",
            "placement_op15_rows",
            "producer_sha256",
            "quality_phone_rows",
            "raw_probes",
            "route_epoch",
            "route_transfer_rows",
            "runtime_processes",
            "schema",
            "session_certificates",
            "started_ns",
        },
        "joint.phone.fragment",
    )
    common.exact(cuda_fragment["schema"], "s39-cp0-r1-v24-cuda-route-raw-v1", "joint.cuda.schema")
    common.exact(phone_fragment["schema"], "s39-cp0-r1-v24-phone-route-raw-v1", "joint.phone.schema")
    for name, fragment in (("cuda", cuda_fragment), ("phone", phone_fragment)):
        common.exact(fragment["phase_id"], receipt["phase_id"], f"joint.{name}.phase_id")
        common.exact(fragment["model_id"], MODEL_ID, f"joint.{name}.model_id")
        common.exact(fragment["model_sha256"], receipt["model_sha256"], f"joint.{name}.model_sha256")
        common.exact(fragment["history_sha256"], receipt["history_sha256"], f"joint.{name}.history_sha256")
        common.exact(
            fragment["mechanism_commands_sha256"],
            receipt["mechanism_commands_sha256"],
            f"joint.{name}.mechanism_commands_sha256",
        )
        common.exact(fragment["route_epoch"], receipt["route_epoch"], f"joint.{name}.route_epoch")
        common.require(
            started <= fragment["started_ns"] < fragment["completed_ns"] <= completed,
            f"E_JOINT_FRAGMENT_INTERVAL: {name}",
        )
        _validate_evidence_artifacts(
            bundle_root,
            fragment["evidence_artifacts"],
            f"joint.{name}.evidence_artifacts",
        )
    _validate_fragment_event_intervals(
        cuda_fragment,
        ("bridge_start_row", "bridge_ready_row"),
        ("cuda_memory_rows", "cuda_route_rows", "quality_cuda_rows"),
        "joint.cuda.fragment",
    )
    _validate_fragment_event_intervals(
        phone_fragment,
        (),
        (
            "bridge_publication_rows",
            "mechanics_rows",
            "placement_op12_rows",
            "placement_op15_rows",
            "quality_phone_rows",
            "route_transfer_rows",
        ),
        "joint.phone.fragment",
    )
    model_component = _artifact_root_components(artifact_root)["model.cuda"]
    _validate_joint_cuda_launch(
        cuda_launch,
        model_component,
        history_raw,
        plan,
        runtime,
        receipt,
    )
    _validate_joint_phone_launch(
        phone_launch,
        contract,
        history_raw,
        plan,
        runtime,
        receipt,
    )
    cuda_continuations = _validate_capture_execution_groups(
        cuda_fragment["execution_groups"],
        history,
        receipt["route_epoch"],
        "joint.cuda.execution_groups",
    )
    phone_continuations = _validate_capture_execution_groups(
        phone_fragment["execution_groups"],
        history,
        receipt["route_epoch"],
        "joint.phone.execution_groups",
    )
    for item_index in history["mechanics_b8"]["item_indices"]:
        common.exact(
            phone_continuations[item_index],
            cuda_continuations[item_index],
            f"E_JOINT_MECHANICS_CONTINUATION: {item_index}",
        )
    runtime_processes = receipt["runtime_processes"]
    common.require(
        type(runtime_processes) is list and len(runtime_processes) == 4,
        "E_JOINT_RUNTIME_PROCESSES",
    )
    common.exact(
        [value.get("bundle_id") for value in runtime_processes],
        sorted(
            (
                "cuda_route",
                "op12_stagenet",
                "op15_direct_relay",
                "op15_stagenet",
            )
        ),
        "E_JOINT_RUNTIME_PROCESS_ORDER",
    )
    by_id = {
        value["bundle_id"]: value
        for value in runtime_processes
    }
    common.exact(cuda_evidence["runtime_process"], by_id["cuda_route"], "E_JOINT_CUDA_RUNTIME_PROJECTION")
    common.exact(
        phone_evidence["runtime_processes"],
        [
            by_id["op12_stagenet"],
            by_id["op15_direct_relay"],
            by_id["op15_stagenet"],
        ],
        "E_JOINT_PHONE_RUNTIME_PROJECTION",
    )
    for bundle_id, value in by_id.items():
        _validate_captured_runtime_process(
            value,
            runtime,
            bundle_id,
            started,
            completed,
        )
    identity_processes = _runtime_processes_by_id(runtime)
    common.require(
        by_id["cuda_route"]["pid"]
        != identity_processes["cuda_monolithic"]["pid"],
        "E_JOINT_CUDA_MONOLITHIC_PID_REUSE",
    )
    common.exact(
        cuda_evidence["protocol_identity"],
        cuda_fragment["protocol_identity"],
        "E_JOINT_CUDA_PROTOCOL_PROJECTION",
    )
    _validate_protocol_identity(
        cuda_evidence["protocol_identity"],
        contract,
        model["artifact"]["sha256"],
    )
    common.exact(
        cuda_evidence["placement_certificate"],
        cuda_fragment["placement_certificate"],
        "E_JOINT_CUDA_PLACEMENT_PROJECTION",
    )
    _validate_cuda_placement_certificate(
        cuda_evidence["placement_certificate"],
        by_id["cuda_route"],
    )
    common.exact(
        cuda_evidence["memory_certificate"],
        cuda_fragment["memory_certificate"],
        "E_JOINT_CUDA_MEMORY_CERT_PROJECTION",
    )
    common.exact(
        cuda_evidence["raw_memory_samples"],
        cuda_fragment["raw_memory_samples"],
        "E_JOINT_CUDA_MEMORY_SAMPLE_PROJECTION",
    )
    memory_samples = cuda_evidence["raw_memory_samples"]
    common.require(
        type(memory_samples) is list and len(memory_samples) == 3,
        "E_JOINT_CUDA_MEMORY_SAMPLES",
    )
    for index, (sample, row) in enumerate(
        zip(memory_samples, receipt["cuda_memory_rows"])
    ):
        sample_value, _ = _read_nested_artifact(
            bundle_root,
            {
                key: sample[key]
                for key in ("bytes", "path", "sha256")
            },
            f"joint.cuda.memory_sample[{index}]",
            canonical=True,
        )
        common.exact(sample.get("row"), row, f"E_JOINT_CUDA_MEMORY_SAMPLE_ROW: {index}")
        common.exact(sample_value, row, f"E_JOINT_CUDA_MEMORY_SAMPLE_FILE: {index}")
    _validate_memory_certificate(
        cuda_evidence["memory_certificate"],
        by_id["cuda_route"],
        cuda_evidence["placement_certificate"],
        receipt["cuda_memory_rows"],
    )
    common.exact(
        cuda_evidence["runtime_model_binding"],
        cuda_fragment["runtime_model_binding"],
        "E_JOINT_CUDA_MODEL_BINDING_PROJECTION",
    )
    _validate_cuda_route_model_binding(
        cuda_evidence["runtime_model_binding"],
        by_id["cuda_route"],
        model_component,
        contract,
    )
    for key in (
        "direct_certificate",
        "direct_frames",
        "placement_certificates",
        "raw_probes",
        "runtime_processes",
        "session_certificates",
    ):
        common.exact(
            phone_evidence[key],
            phone_fragment[key],
            f"E_JOINT_PHONE_EVIDENCE_PROJECTION: {key}",
        )
    _validate_phone_direct_evidence(
        phone_evidence,
        receipt["model_sha256"],
        receipt["route_transfer_rows"],
    )
    _validate_phone_raw_probes(
        phone_evidence,
        phone_launch,
        runtime,
        contract,
        phone_fragment["started_ns"],
        phone_fragment["completed_ns"],
    )
    _validate_phone_certificates(phone_evidence, runtime, contract)
    for key in (
        "cuda_memory_rows",
        "cuda_route_rows",
        "mechanics_rows",
        "op12_runtime",
        "op15_runtime",
        "placement_op12_rows",
        "placement_op15_rows",
        "quality_cuda_rows",
        "quality_phone_rows",
        "route_transfer_rows",
    ):
        source_fragment = cuda_fragment if key.startswith(("cuda_", "quality_cuda")) else phone_fragment
        common.exact(
            receipt[key],
            source_fragment[key],
            f"E_JOINT_OUTER_FRAGMENT_PROJECTION: {key}",
        )
    common.exact(
        receipt["gpu_runtime"],
        cuda_fragment["gpu_runtime"],
        "E_JOINT_GPU_RUNTIME_PROJECTION",
    )
    common.exact(
        receipt["bridge_rows"],
        [
            cuda_fragment["bridge_start_row"],
            *phone_fragment["bridge_publication_rows"],
            cuda_fragment["bridge_ready_row"],
        ],
        "E_JOINT_BRIDGE_PROJECTION",
    )
    previous_event = None
    for index, row in enumerate(receipt["bridge_rows"]):
        event = common.integer(row.get("event_ns"), f"joint.bridge[{index}].event_ns", 1)
        if previous_event is not None:
            common.require(previous_event <= event, f"E_JOINT_BRIDGE_ORDER: {index}")
        previous_event = event
    common.require(
        all(
            row["event_ns"] < receipt["bridge_rows"][-1]["event_ns"]
            for row in receipt["bridge_rows"][1:-1]
        ),
        "E_JOINT_PUBLICATION_AFTER_CUDA_READY",
    )
    for role, key in JOINT_ROLE_FIELDS.items():
        _exact_role_projection(
            rows_by_role,
            role,
            receipt[key],
            receipt["phase_id"],
        )


def _project_bridge_rows(
    rows: list[dict[str, Any]],
    raw_phone_rows: list[dict[str, Any]],
    projected_phone_rows: list[dict[str, Any]],
    helpers: dict[str, types.ModuleType],
) -> list[dict[str, Any]]:
    v2 = helpers["v2"]
    v21 = helpers["v21"]

    raw_requests = {
        row["request_id"]: row
        for row in v21._normalize_rows(raw_phone_rows)
        if row["kind"] == "request"
    }
    projected_requests = {
        row["request_id"]: row
        for row in v21._normalize_rows(projected_phone_rows)
        if row["kind"] == "request"
    }
    result = copy.deepcopy(rows)
    publications = [
        row for row in result
        if row["kind"] == "phone_publication_received"
    ]
    common.exact(
        {row["request_id"] for row in publications},
        set(range(1, 9)),
        "E_V2_4_BRIDGE_WIRE_IDS",
    )
    for row in publications:
        wire_id = row["request_id"]
        common.exact(
            row["phone_request_sha256"],
            v2.digest_json(raw_requests[wire_id]),
            f"E_V2_4_BRIDGE_RAW_LINK: {wire_id}",
        )
        row["request_id"] = wire_id - 1
        row["phone_request_sha256"] = v2.digest_json(
            projected_requests[wire_id - 1]
        )
    return result


def evaluate_v24_model_phase(
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate: dict[str, Any],
    candidate_raw: bytes,
    parent: dict[str, Any],
    frozen_corpus: list[dict[str, Any]],
    manifest: dict[str, Any],
    manifest_raw: bytes,
    rows_by_role: dict[str, list[dict[str, Any]]],
    artifact_digests: dict[str, str],
    projected_executions: dict[str, list[dict[str, Any]]],
    helpers: dict[str, types.ModuleType],
) -> dict[str, Any]:
    v2 = helpers["v2"]
    v21 = helpers["v21"]
    v22 = helpers["v22"]

    common.exact(manifest["phase"], PHASE, "E_V2_4_RAW_PHASE")
    model = next(row for row in candidate["models"] if row["slot"] == "A")
    phase_lock = v21.validate_phase_lock(
        rows_by_role["phase.lock"],
        manifest,
        artifact_digests,
        contract,
        contract_raw,
        candidate_raw,
        [],
    )
    prefix = f"model.{model['model_id']}"
    route_role = f"{prefix}.route_lock"
    lock = v21.validate_route_lock(
        rows_by_role[route_role],
        route_role,
        manifest,
        model,
        contract,
        parent,
    )
    v22.validate_incumbent_route(lock, contract, model)
    common.require(
        max(
            max(row["event_ns"] for row in rows_by_role["quality.corpus"]),
            rows_by_role[route_role][0]["event_ns"],
        )
        <= phase_lock["event_ns"],
        "E_LOCK_ORDER: model inputs after phase lock",
    )
    common.require(
        phase_lock["event_ns"]
        < min(row["event_ns"] for row in rows_by_role["phase.preflight"]),
        "E_LOCK_ORDER: phase lock must precede preflight",
    )
    preflight = v21.validate_preflight(
        rows_by_role["phase.preflight"],
        manifest,
        contract,
        [(model, lock)],
    )
    quality, _ = v21._derive_quality(
        rows_by_role,
        artifact_digests,
        model,
        manifest["phase_id"],
        candidate["task_suite"],
        parent,
    )
    v22.validate_exact_corpus(rows_by_role["quality.corpus"], frozen_corpus)
    common.require(
        quality["cuda_correct"]
        >= contract["gates"]["cuda_quality_minimum_correct_items"],
        f"E_CUDA_QUALITY_FLOOR: {model['model_id']}",
    )

    normalized = {
        role: v21._normalize_rows(rows)
        for role, rows in projected_executions.items()
    }
    executions = {
        role: v2.validate_execution(
            normalized[role],
            role,
            manifest["phase_id"],
            model,
            "PHONE_COLLECTIVE" if role.endswith(".mechanics.phone") else "CUDA0",
        )
        for role in normalized
    }
    for role, execution in executions.items():
        for request_id in range(8):
            common.exact(
                len(execution["requests"][request_id]["continuation_tokens"]),
                8,
                f"E_V2_4_CONTINUATION_LENGTH: {role}[{request_id}]",
            )
    oracle = v2.derive_oracle(normalized, model, manifest["phase_id"])
    phone_role = f"{prefix}.mechanics.phone"
    phone = executions[phone_role]
    allocation, cuda_ready = v21._derive_memory(
        rows_by_role[f"{prefix}.cuda_memory"],
        f"{prefix}.cuda_memory",
        manifest["phase_id"],
        model,
        parent,
    )
    projected_bridge = _project_bridge_rows(
        rows_by_role[f"{prefix}.bridge"],
        rows_by_role[phone_role],
        projected_executions[phone_role],
        helpers,
    )
    bridge = v21._derive_bridge(
        projected_bridge,
        f"{prefix}.bridge",
        manifest["phase_id"],
        model,
        phone,
        projected_executions[phone_role],
        cuda_ready,
        manifest["clock_id"],
        parent,
    )
    v22.validate_bridge_causality(rows_by_role, model)
    placement = {}
    for phone_name in ("op15", "op12"):
        role = f"{prefix}.placement.{phone_name}"
        placement[phone_name] = v2.derive_placement(
            v21._normalize_rows(rows_by_role[role]),
            role,
            manifest["phase_id"],
            model,
            phone_name,
            lock,
            parent,
        )
    transfer = v21._derive_transfer(
        rows_by_role[f"{prefix}.route_transfer"],
        f"{prefix}.route_transfer",
        manifest["phase_id"],
        model,
        lock,
        phone,
    )
    return {
        "bundle_manifest_sha256": common.sha256_bytes(manifest_raw),
        "candidate_sha256": common.sha256_bytes(candidate_raw),
        "contract_sha256": common.sha256_bytes(contract_raw),
        "derived": {
            "model": {
                "allocation": allocation,
                "bridge": bridge,
                "cuda_ready": cuda_ready,
                "model_id": model["model_id"],
                "oracle": oracle,
                "placement": placement,
                "quality": quality,
                "route_lock": lock,
                "transfer": transfer,
            },
            "preflight": preflight,
            "v2_4": {
                "canonical_corpus_sha256": contract["quality_corpus"]["sha256"],
                "continuation_semantics": (
                    "PREFILL_FINAL_OUTPUT_PLUS_SEVEN_DECODE_CALLS"
                ),
                "wire_request_ids": list(range(1, 9)),
            },
        },
        "phase": PHASE,
        "phase_closed_ns": manifest["phase_closed_ns"],
        "phase_id": manifest["phase_id"],
        "phase_opened_ns": manifest["phase_opened_ns"],
        "schema": "s39-cp0-r1-raw-predicate-result-v2.4",
        "status": "MODEL_A_QUALIFICATION_PASS",
    }


def authorize_a_only(
    *,
    bundle_root: Path,
    chain_kwargs: dict[str, Path],
    orchestration_plan_path: Path | None = None,
    prospective_root_path: Path | None = None,
    bound_root_path: Path | None = None,
    identity_binding_receipt_path: Path | None = None,
    identity_binding_stage_receipt_path: Path | None = None,
    evaluator: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    common.require(
        orchestration_plan_path is not None,
        "E_ORCHESTRATION_PLAN_REQUIRED",
    )
    common.require(
        prospective_root_path is not None
        and bound_root_path is not None
        and identity_binding_receipt_path is not None
        and identity_binding_stage_receipt_path is not None,
        "E_IDENTITY_BINDING_REQUIRED",
    )
    chain = validate_chain(**chain_kwargs)
    acquisition, _ = common.read_canonical(chain_kwargs["acquisition_path"])
    contract, contract_raw, candidate, candidate_raw = validate_inputs(
        chain_kwargs["contract_path"],
        chain_kwargs["candidate_path"],
    )
    orchestration = validate_orchestration_provenance(
        orchestration_plan_path=orchestration_plan_path,
        bundle_root=bundle_root,
        contract_path=chain_kwargs["contract_path"],
        candidate_path=chain_kwargs["candidate_path"],
        contract=contract,
        contract_raw=contract_raw,
        candidate_raw=candidate_raw,
    )
    common.exact(
        orchestration["phase_id"],
        chain["phase_id"],
        "E_ORCHESTRATION_PHASE_ID",
    )
    identity = validate_identity_binding(
        prospective_root_path=prospective_root_path,
        bound_root_path=bound_root_path,
        identity_binding_receipt_path=identity_binding_receipt_path,
        identity_binding_stage_receipt_path=identity_binding_stage_receipt_path,
        contract_path=chain_kwargs["contract_path"],
        candidate_path=chain_kwargs["candidate_path"],
        runtime_plan_path=chain_kwargs["runtime_plan_path"],
        token_history_path=chain_kwargs["token_history_path"],
        tokenizer_plan_path=chain_kwargs["tokenizer_plan_path"],
        preparation_path=chain_kwargs["preparation_path"],
        phase_lock_path=chain_kwargs["phase_lock_path"],
        fresh_path=chain_kwargs["fresh_path"],
        contract=contract,
        contract_raw=contract_raw,
        candidate_raw=candidate_raw,
        orchestration=orchestration,
    )
    common.exact(
        identity["phase_id"],
        chain["phase_id"],
        "E_IDENTITY_BINDING_PHASE_ID",
    )
    _, raw_contract_raw, _ = raw_predicate_inputs(contract)
    manifest_path = bundle_root / RAW_MANIFEST_NAME
    manifest, manifest_raw = common.read_canonical(manifest_path)
    common.exact(
        common.sha256_bytes(manifest_raw),
        acquisition["raw_manifest_sha256"],
        "E_RAW_MANIFEST_DIGEST",
    )
    common.exact(
        acquisition["raw_predicate_contract_sha256"],
        common.sha256_bytes(raw_contract_raw),
        "E_RAW_PREDICATE_CONTRACT",
    )
    common.exact(manifest.get("phase"), PHASE, "E_RAW_PHASE")
    common.exact(manifest.get("phase_id"), chain["phase_id"], "E_RAW_PHASE_ID")
    common.exact(
        manifest.get("acquisition_started_ns"),
        common.integer(
            common.read_canonical(chain_kwargs["acquisition_path"])[0]["started_ns"],
            "acquisition.started",
            1,
        ),
        "E_RAW_ACQUISITION_START",
    )
    common.exact(
        manifest.get("phase_closed_ns"),
        chain["timing"]["acquisition_completed_ns"],
        "E_RAW_ACQUISITION_END",
    )
    manifest_artifacts = {
        row.get("role"): row.get("sha256")
        for row in manifest.get("artifacts", [])
        if type(row) is dict
    }
    acquisition_digests = {
        row["role"]: row["sha256"] for row in acquisition["artifacts"]
    }
    for role, digest_value in acquisition_digests.items():
        if role in CAPTURE_RECEIPT_ROLES:
            continue
        common.exact(
            manifest_artifacts.get(role),
            digest_value,
            f"E_RAW_ARTIFACT: {role}",
        )
    raw_contract, raw_contract_raw, _ = raw_predicate_inputs(contract)
    helpers = _verified_helpers(contract)
    _, _, rows_by_role, _ = helpers["v21"].load_bundle(
        bundle_root,
        RAW_MANIFEST_NAME,
        raw_contract,
        raw_contract_raw,
        candidate_raw,
    )
    cuda_receipt, _ = _read_bound_artifact(
        bundle_root,
        acquisition,
        "capture.cuda_monolithic",
    )
    joint_receipt, _ = _read_bound_artifact(
        bundle_root,
        acquisition,
        "capture.joint_phone_cuda",
    )
    validate_bound_execution_bindings(
        cuda_receipt=cuda_receipt,
        joint_receipt=joint_receipt,
        identity=identity,
    )
    runtime_plan, _ = common.read_canonical(
        chain_kwargs["runtime_plan_path"]
    )
    runtime_identity, _ = common.read_canonical(
        chain_kwargs["runtime_identity_path"]
    )
    token_history, token_history_raw = common.read_canonical(
        chain_kwargs["token_history_path"]
    )
    artifact_root, _ = common.read_canonical(
        chain_kwargs["artifact_root_path"]
    )
    validate_cuda_monolithic_receipt(
        cuda_receipt,
        bundle_root,
        contract,
        candidate,
        token_history,
        token_history_raw,
        runtime_plan,
        runtime_identity,
        rows_by_role,
    )
    validate_joint_phone_cuda_receipt(
        joint_receipt,
        bundle_root,
        contract,
        candidate,
        token_history,
        token_history_raw,
        runtime_plan,
        runtime_identity,
        artifact_root,
        rows_by_role,
        common.integer(acquisition["started_ns"], "acquisition.started_ns", 1),
        orchestration["plan_sha256"],
    )

    result = (
        evaluate_raw_predicates(
            bundle_root,
            contract,
            candidate,
            candidate_raw,
            token_history,
        )
        if evaluator is None
        else evaluator(bundle_root)
    )
    common.exact(
        result.get("schema"),
        "s39-cp0-r1-raw-predicate-result-v2.4",
        "E_RAW_RESULT_SCHEMA",
    )
    common.exact(
        result.get("status"),
        "MODEL_A_QUALIFICATION_PASS",
        "E_RAW_RESULT_STATUS",
    )
    return {
        **chain,
        "bound_runtime_root_sha256": identity["bound_root_sha256"],
        "identity_binding_receipt_sha256": identity[
            "identity_binding_receipt_sha256"
        ],
        "orchestration_plan_sha256": orchestration["plan_sha256"],
        "schema": "s39-cp0-r1-evidence-result-v2.4",
        "status": "MODEL_A_QUALIFICATION_PASS_V2_4",
        "v2_4_raw_result_sha256": common.sha256_bytes(
            common.canonical_bytes(result)
        ),
    }


def validate_bound_execution_bindings(
    *,
    cuda_receipt: dict[str, Any],
    joint_receipt: dict[str, Any],
    identity: dict[str, Any],
) -> None:
    common.exact(
        cuda_receipt["mechanism_commands_sha256"],
        identity["mechanism_commands_sha256"],
        "E_CUDA_REALIZED_MECHANISM",
    )
    common.exact(
        joint_receipt["capture_plan_sha256"],
        identity["artifact_sha256s"]["joint_capture_plan"],
        "E_JOINT_BOUND_CAPTURE_PLAN",
    )
    common.exact(
        joint_receipt["mechanism_commands_sha256"],
        identity["mechanism_commands_sha256"],
        "E_JOINT_REALIZED_MECHANISM",
    )
    for name, artifact_name in (
        ("cuda", "cuda_route_launch"),
        ("phone", "phone_route_launch"),
    ):
        common.exact(
            joint_receipt["subproducer_bindings"][name][
                "launch_plan_sha256"
            ],
            identity["artifact_sha256s"][artifact_name],
            f"E_JOINT_BOUND_SUBPRODUCER: {name}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--runtime-plan", type=Path)
    parser.add_argument("--tokenizer-plan", type=Path)
    parser.add_argument("--token-history", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--phase-lock", type=Path)
    parser.add_argument("--fresh", type=Path)
    parser.add_argument("--runtime-identity", type=Path)
    parser.add_argument("--acquisition", type=Path)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--orchestration-plan", type=Path)
    parser.add_argument("--prospective-root", type=Path)
    parser.add_argument("--bound-root", type=Path)
    parser.add_argument("--identity-binding-receipt", type=Path)
    parser.add_argument("--identity-binding-stage-receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract, contract_raw, _, candidate_raw = validate_inputs(
            args.contract,
            args.candidate,
        )
        evidence = (
            args.runtime_plan,
            args.tokenizer_plan,
            args.token_history,
            args.artifact_root,
            args.preparation,
            args.phase_lock,
            args.fresh,
            args.runtime_identity,
            args.acquisition,
            args.orchestration_plan,
            args.prospective_root,
            args.bound_root,
            args.identity_binding_receipt,
            args.identity_binding_stage_receipt,
        )
        if all(value is None for value in evidence) and args.bundle_root is None:
            result = {
                "candidate_sha256": common.sha256_bytes(candidate_raw),
                "contract_sha256": common.sha256_bytes(contract_raw),
                "schema": "s39-cp0-r1-evidence-contract-check-v2.4",
                "status": contract["claim_boundary"]["mechanics_status"],
            }
        else:
            common.require(all(value is not None for value in evidence), "E_ARGUMENTS: incomplete V2.4 chain")
            common.require(args.bundle_root is not None, "E_ARGUMENTS: raw V2.2 bundle root required")
            result = authorize_a_only(
                bundle_root=args.bundle_root,
                orchestration_plan_path=args.orchestration_plan,
                prospective_root_path=args.prospective_root,
                bound_root_path=args.bound_root,
                identity_binding_receipt_path=args.identity_binding_receipt,
                identity_binding_stage_receipt_path=(
                    args.identity_binding_stage_receipt
                ),
                chain_kwargs={
                    "contract_path": args.contract,
                    "candidate_path": args.candidate,
                    "runtime_plan_path": args.runtime_plan,
                    "tokenizer_plan_path": args.tokenizer_plan,
                    "token_history_path": args.token_history,
                    "artifact_root_path": args.artifact_root,
                    "preparation_path": args.preparation,
                    "phase_lock_path": args.phase_lock,
                    "fresh_path": args.fresh,
                    "runtime_identity_path": args.runtime_identity,
                    "acquisition_path": args.acquisition,
                },
            )
        print(common.canonical_bytes(result).decode("ascii"), end="")
        return 0
    except (common.EvidenceError, OSError, KeyError, ValueError) as error:
        print(f"CP0_R1_V2_4_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
