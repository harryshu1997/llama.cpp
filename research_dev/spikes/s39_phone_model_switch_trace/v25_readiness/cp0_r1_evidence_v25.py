#!/usr/bin/env python3
"""Validate the CP0-R1 V2.5 post-reboot A_ONLY evidence chain."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import ipaddress
from pathlib import Path
import stat
import sys
import types
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
V24 = S39 / "v24_readiness"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import v25_common as common


PHASE = "A_ONLY"
MODEL_ID = "qwen3-14b-q4_k_m"
MANIFEST_NAME = "EVIDENCE_BUNDLE_V2_5.json"
RAW_MANIFEST_NAME = "EVIDENCE_BUNDLE_V2_5_RAW.json"
MANAGED_PLAN_ROLES = {
    "plan.managed.cuda_monolithic",
    "plan.managed.joint_phone_cuda",
    "plan.managed.remote_fan_in",
}
WRAPPER_PLAN_ROLES = {
    "plan.wrapper.cuda_monolithic",
    "plan.wrapper.joint_phone_cuda",
}
GUARD_PLAN_ROLES = {
    "plan.remote_fan_in",
    "plan.remote_phone_guard",
}
INNER_V24_ROLES = {
    "inner.artifact_root",
    "inner.bound_root",
    "inner.cuda_route_launch",
    "inner.fresh_readiness",
    "inner.identity_binding_attestation",
    "inner.identity_binding_receipt",
    "inner.identity_binding_stage_receipt",
    "inner.joint_capture_plan",
    "inner.orchestration_plan",
    "inner.phase_lock",
    "inner.phone_route_launch",
    "inner.preparation",
    "inner.prospective_root",
    "inner.runtime_plan",
}
PHASE_EVIDENCE_ROLES = {
    "phase.execution_ledger",
    "phase.fresh_identity",
    "phase.inventory",
    "phase.materialization",
}
MATERIALIZED_SUPPORT_ROLES = {
    "plan.remote_phone_guard_policy",
}
REQUIRED_PLAN_ROLES = (
    MANAGED_PLAN_ROLES | WRAPPER_PLAN_ROLES | GUARD_PLAN_ROLES
)
PHASE_LOCK_BOUND_ROLES = (
    REQUIRED_PLAN_ROLES
    | INNER_V24_ROLES
    | MATERIALIZED_SUPPORT_ROLES
    | {
        "history.remote_plan",
        "history.remote_receipt",
        "phase.fresh_identity",
        "phase.inventory",
        "phase.materialization",
    }
)
MATERIALIZED_ROLES = (
    REQUIRED_PLAN_ROLES
    | INNER_V24_ROLES
    | MATERIALIZED_SUPPORT_ROLES
    | {
        "history.remote_plan",
        "phase.discovery",
        "phase.preparation",
    }
)
STAGE_ORDER = (
    "remote_history",
    "phone_guard_before",
    "cuda_monolithic",
    "joint_phone_cuda",
    "remote_fan_in",
    "phone_guard_after",
)
COMPACT_PLAN_ROLES = MANAGED_PLAN_ROLES | {"history.remote_plan"}
REQUIRED_RUNTIME_ROLES = {
    "capture.cuda_monolithic.wrapper",
    "capture.joint_phone_cuda.wrapper",
    "capture.remote_fan_in.wrapper",
    "runtime.remote_phone_guard.after",
    "runtime.remote_phone_guard.before",
}
REQUIRED_ROLES = {
    "history.remote_plan",
    "history.remote_receipt",
    "phase.discovery",
    "phase.lock",
    "phase.preparation",
    *INNER_V24_ROLES,
    *PHASE_EVIDENCE_ROLES,
    *MATERIALIZED_SUPPORT_ROLES,
    *REQUIRED_PLAN_ROLES,
    *REQUIRED_RUNTIME_ROLES,
}
VALIDATOR_WRAPPER = (
    "import runpy,sys;"
    "sys.path.insert(0,sys.argv.pop(1));"
    "runpy.run_path(sys.argv.pop(1),run_name='__main__')"
)


def _load_source(
    name: str,
    path: Path,
    expected: dict[str, Any],
    injected: dict[str, types.ModuleType] | None = None,
) -> types.ModuleType:
    raw = common.read_regular(path)
    common.exact(len(raw), expected["bytes"], f"source.{name}.bytes")
    common.exact(
        common.sha256_bytes(raw),
        expected["sha256"],
        f"source.{name}.sha256",
    )
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    previous = {
        key: sys.modules.get(key)
        for key in (injected or {})
    }
    try:
        if injected:
            sys.modules.update(injected)
        exec(compile(raw, str(path), "exec"), module.__dict__)
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


def _validate_contract(
    contract_path: Path,
    candidate_path: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    contract, contract_raw = common.read_canonical(contract_path)
    builder_raw = common.read_regular(HERE / "build_contract_v25.py")
    builder = types.ModuleType("s39_v25_contract_builder")
    builder.__file__ = str(HERE / "build_contract_v25.py")
    previous_common = sys.modules.get("v25_common")
    try:
        sys.modules["v25_common"] = common
        exec(
            compile(
                builder_raw,
                str(HERE / "build_contract_v25.py"),
                "exec",
            ),
            builder.__dict__,
        )
    finally:
        if previous_common is None:
            sys.modules.pop("v25_common", None)
        else:
            sys.modules["v25_common"] = previous_common
    common.exact(contract, builder.build_contract(), "contract")
    common.exact(
        contract["schema"],
        "s39-cp0-r1-evidence-contract-v2.5",
        "contract.schema",
    )
    common.exact(contract["version"], "2.5", "contract.version")
    common.exact(contract["phase"], PHASE, "contract.phase")
    common.exact(
        contract["claim_boundary"]["acquisition_ready"],
        True,
        "E_REMOTE_CAPTURE_ADAPTER_NOT_FROZEN",
    )
    for name, record in contract["composition"]["v25"].items():
        path = S39 / record["path"]
        raw = common.read_regular(path, 512 * 1024 * 1024)
        common.exact(len(raw), record["bytes"], f"source.v25.{name}.bytes")
        common.exact(
            common.sha256_bytes(raw),
            record["sha256"],
            f"source.v25.{name}.sha256",
        )
    for name, record in contract["composition"]["history"].items():
        source = record["source"]
        source_path = S39 / common.relative_path(
            source["path"],
            f"source.history.{name}.path",
        )
        raw = common.read_regular(source_path, 512 * 1024 * 1024)
        common.exact(len(raw), source["bytes"], f"source.history.{name}.bytes")
        common.exact(
            common.sha256_bytes(raw),
            source["sha256"],
            f"source.history.{name}.sha256",
        )
        common.absolute_path(
            record["remote_path"],
            f"source.history.{name}.remote_path",
        )
    candidate, candidate_raw = common.read_canonical(candidate_path)
    common.exact(
        common.sha256_bytes(candidate_raw),
        contract["candidate"]["sha256"],
        "candidate.sha256",
    )
    common.exact(candidate["schema"], "s39-cp0-r1-candidate-v1", "candidate.schema")
    models = [
        value
        for value in candidate["models"]
        if value.get("slot") == "A"
    ]
    common.require(
        len(models) == 1 and models[0].get("model_id") == MODEL_ID,
        "E_CANDIDATE_A",
    )
    return contract, contract_raw, candidate, candidate_raw


def _artifact_map(
    manifest: dict[str, Any],
    root: Path,
) -> dict[str, tuple[dict[str, Any], bytes]]:
    values = manifest["artifacts"]
    common.require(
        type(values) is list and len(values) == len(REQUIRED_ROLES),
        "E_ARTIFACTS",
    )
    result: dict[str, tuple[dict[str, Any], bytes]] = {}
    paths: set[str] = set()
    previous = None
    for index, value in enumerate(values):
        field = f"manifest.artifacts[{index}]"
        common.exact_keys(
            value,
            {"bytes", "path", "role", "sha256"},
            field,
        )
        role = common.text(value["role"], f"{field}.role", 128)
        common.require(
            role in REQUIRED_ROLES and role not in result,
            f"E_ARTIFACT_ROLE: {role}",
        )
        if previous is not None:
            common.require(previous < role, "E_ARTIFACT_ORDER")
        previous = role
        path_text = common.relative_path(value["path"], f"{field}.path")
        common.require(path_text not in paths, f"E_ARTIFACT_PATH_REUSE: {path_text}")
        paths.add(path_text)
        path = root / path_text
        common.require(path.is_relative_to(root), f"E_ARTIFACT_PATH: {path_text}")
        raw = common.read_regular(path)
        common.exact(len(raw), value["bytes"], f"{field}.bytes")
        common.exact(
            common.sha256_bytes(raw),
            common.digest(value["sha256"], f"{field}.sha256"),
            f"{field}.sha256",
        )
        parsed = common.parse_json(raw, field)
        common.require(type(parsed) is dict, f"E_TYPE: {field}")
        expected_raw = (
            common.canonical_compact(parsed)
            if role in COMPACT_PLAN_ROLES
            else common.canonical_bytes(parsed)
        )
        common.require(expected_raw == raw, f"E_CANONICAL: {field}")
        result[role] = (parsed, raw)
    common.exact(set(result), REQUIRED_ROLES, "manifest.roles")
    return result


def validate_preparation(
    value: Any,
    contract: dict[str, Any],
) -> dict[str, Any]:
    value = common.exact_keys(
        value,
        {
            "completed_ns",
            "controller",
            "local_python",
            "phase",
            "phase_id",
            "phones",
            "schema",
            "started_ns",
        },
        "preparation",
    )
    common.exact(
        value["schema"],
        "s39-cp0-r1-v25-reboot-preparation-v1",
        "preparation.schema",
    )
    common.exact(value["phase"], PHASE, "preparation.phase")
    phase_id = common.text(value["phase_id"], "preparation.phase_id", 128)
    common.require(
        phase_id.startswith("cp0-r1-v25-a-only-"),
        "E_PHASE_ID: preparation",
    )
    started = common.integer(value["started_ns"], "preparation.started_ns", 1)
    completed = common.integer(
        value["completed_ns"],
        "preparation.completed_ns",
        started + 1,
    )
    controller = common.exact_keys(
        value["controller"],
        {"boot_id", "host"},
        "preparation.controller",
    )
    common.exact(
        controller["host"],
        contract["topology"]["controller_host"],
        "preparation.controller.host",
    )
    common.uuid(controller["boot_id"], "preparation.controller.boot_id")
    local_python = common.exact_keys(
        value["local_python"],
        {"bytes", "path", "sha256", "stat"},
        "preparation.local_python",
    )
    python_path = common.absolute_path(
        local_python["path"],
        "preparation.local_python.path",
    )
    python_bytes = common.integer(
        local_python["bytes"],
        "preparation.local_python.bytes",
        1,
    )
    python_sha256 = common.digest(
        local_python["sha256"],
        "preparation.local_python.sha256",
    )
    python_stat = common.exact_keys(
        local_python["stat"],
        {
            "build_id",
            "ctime_ns",
            "device_id",
            "inode",
            "mode",
            "mtime_ns",
            "size",
        },
        "preparation.local_python.stat",
    )
    common.exact(python_stat["build_id"], None, "preparation.local_python.build_id")
    for key in (
        "ctime_ns",
        "device_id",
        "inode",
        "mode",
        "mtime_ns",
        "size",
    ):
        common.integer(
            python_stat[key],
            f"preparation.local_python.stat.{key}",
        )
    common.require(
        python_stat["inode"] > 0
        and python_stat["size"] == python_bytes
        and stat.S_ISREG(python_stat["mode"])
        and python_stat["mode"] & 0o111,
        "E_PREPARATION_PYTHON",
    )
    phones = common.exact_keys(
        value["phones"],
        {"op12", "op15"},
        "preparation.phones",
    )
    adb_identities = {}
    for phone in ("op12", "op15"):
        field = f"preparation.phones.{phone}"
        row = common.exact_keys(
            phones[phone],
            {
                "adb_path",
                "adb_port",
                "adb_sha256",
                "boot_id_before",
                "disconnected_ns",
                "physical_serial",
                "reboot_argv",
                "reboot_returncode",
                "requested_ns",
            },
            field,
        )
        adb_path = common.absolute_path(row["adb_path"], f"{field}.adb_path")
        adb_sha256 = common.digest(row["adb_sha256"], f"{field}.adb_sha256")
        common.exact(row["adb_port"], 5038, f"{field}.adb_port")
        serial = contract["devices"][phone]["serial"]
        common.exact(row["physical_serial"], serial, f"{field}.serial")
        common.uuid(row["boot_id_before"], f"{field}.boot_id_before")
        common.exact(
            row["reboot_argv"],
            [
                adb_path,
                "-P",
                "5038",
                "-s",
                serial,
                "reboot",
            ],
            f"{field}.reboot_argv",
        )
        common.exact(row["reboot_returncode"], 0, f"{field}.returncode")
        requested = common.integer(
            row["requested_ns"],
            f"{field}.requested_ns",
            started,
        )
        disconnected = common.integer(
            row["disconnected_ns"],
            f"{field}.disconnected_ns",
            requested,
        )
        common.require(disconnected <= completed, f"E_PREPARATION_INTERVAL: {phone}")
        adb_identities[phone] = {
            "adb_path": adb_path,
            "adb_port": 5038,
            "adb_sha256": adb_sha256,
        }
    return {
        "completed_ns": completed,
        "controller_boot_id": controller["boot_id"],
        "local_python": {
            "bytes": python_bytes,
            "path": python_path,
            "sha256": python_sha256,
            "stat": python_stat,
        },
        "adb_identities": adb_identities,
        "phase_id": phase_id,
        "phones": phones,
        "started_ns": started,
    }


def validate_discovery(
    value: Any,
    contract: dict[str, Any],
    preparation: dict[str, Any],
    preparation_raw: bytes,
) -> dict[str, Any]:
    value = common.exact_keys(
        value,
        {
            "completed_ns",
            "controller",
            "cuda",
            "phase",
            "phase_id",
            "phones",
            "preparation_completed_ns",
            "preparation_sha256",
            "schema",
            "started_ns",
        },
        "discovery",
    )
    common.exact(
        value["schema"],
        "s39-cp0-r1-v25-post-reboot-discovery-v1",
        "discovery.schema",
    )
    common.exact(value["phase"], PHASE, "discovery.phase")
    common.exact(
        value["phase_id"],
        preparation["phase_id"],
        "discovery.phase_id",
    )
    common.exact(
        value["preparation_sha256"],
        common.sha256_bytes(preparation_raw),
        "discovery.preparation_sha256",
    )
    preparation_completed = common.integer(
        value["preparation_completed_ns"],
        "discovery.preparation_completed_ns",
        1,
    )
    common.exact(
        preparation_completed,
        preparation["completed_ns"],
        "discovery.preparation_completed_ns",
    )
    started = common.integer(value["started_ns"], "discovery.started_ns", 1)
    completed = common.integer(value["completed_ns"], "discovery.completed_ns", 1)
    common.require(preparation_completed <= started < completed, "E_DISCOVERY_ORDER")

    controller = common.exact_keys(
        value["controller"],
        {"boot_id", "host"},
        "discovery.controller",
    )
    common.exact(
        controller["host"],
        contract["topology"]["controller_host"],
        "discovery.controller.host",
    )
    common.uuid(controller["boot_id"], "discovery.controller.boot_id")
    common.exact(
        controller["boot_id"],
        preparation["controller_boot_id"],
        "discovery.controller.boot_id",
    )

    cuda = common.exact_keys(
        value["cuda"],
        {
            "boot_id",
            "gpu_uuid",
            "host",
            "memory_total_bytes",
            "name",
            "ssh_target",
            "system_swap_used_bytes",
        },
        "discovery.cuda",
    )
    for key in ("host", "gpu_uuid", "ssh_target"):
        common.exact(
            cuda[key],
            contract["topology"][f"cuda_{key}"],
            f"discovery.cuda.{key}",
        )
    common.uuid(cuda["boot_id"], "discovery.cuda.boot_id")
    common.exact(
        cuda["name"],
        contract["devices"]["cuda"]["name"],
        "discovery.cuda.name",
    )
    common.exact(
        cuda["memory_total_bytes"],
        contract["devices"]["cuda"]["memory_total_bytes"],
        "discovery.cuda.memory_total_bytes",
    )
    common.integer(
        cuda["system_swap_used_bytes"],
        "discovery.cuda.system_swap_used_bytes",
    )

    phones = common.exact_keys(value["phones"], {"op12", "op15"}, "discovery.phones")
    for phone in ("op12", "op15"):
        field = f"discovery.phones.{phone}"
        record = common.exact_keys(
            phones[phone],
            {
                "adb_port",
                "attested_ns",
                "boot_id",
                "device",
                "interface",
                "model",
                "physical_serial",
                "product",
                "system_swap_used_bytes",
                "usb_observed_ns",
                "wifi_ipv4",
                "wifi_selector",
            },
            field,
        )
        common.exact(
            record["physical_serial"],
            contract["devices"][phone]["serial"],
            f"{field}.serial",
        )
        for key in ("device", "model", "product"):
            common.exact(
                record[key],
                contract["devices"][phone][key],
                f"{field}.{key}",
            )
        common.exact(record["adb_port"], 5038, f"{field}.adb_port")
        common.uuid(record["boot_id"], f"{field}.boot_id")
        common.require(
            record["boot_id"]
            != preparation["phones"][phone]["boot_id_before"],
            f"E_PHONE_NOT_REBOOTED: {phone}",
        )
        interface = common.text(record["interface"], f"{field}.interface", 64)
        common.require(
            all(character.isalnum() or character in "._-" for character in interface),
            f"E_INTERFACE: {phone}",
        )
        try:
            ipv4 = str(ipaddress.IPv4Address(record["wifi_ipv4"]))
        except ipaddress.AddressValueError as error:
            raise common.EvidenceError(f"E_IPV4: {phone}") from error
        common.exact(
            record["wifi_selector"],
            f"{ipv4}:5555",
            f"E_DYNAMIC_SELECTOR: {phone}",
        )
        usb_observed = common.integer(
            record["usb_observed_ns"],
            f"{field}.usb_observed_ns",
            preparation_completed,
        )
        attested = common.integer(
            record["attested_ns"],
            f"{field}.attested_ns",
            usb_observed,
        )
        common.require(attested <= completed, f"E_DISCOVERY_INTERVAL: {phone}")
        common.integer(
            record["system_swap_used_bytes"],
            f"{field}.system_swap_used_bytes",
        )
    common.require(
        phones["op12"]["wifi_selector"] != phones["op15"]["wifi_selector"],
        "E_SELECTOR_ALIAS",
    )
    return {
        "completed_ns": completed,
        "controller_boot_id": controller["boot_id"],
        "cuda": cuda,
        "phase_id": value["phase_id"],
        "phones": phones,
    }


def _parse_inline_plan(
    raw: bytes,
    field: str,
    parser: Callable[[str, str], dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    common.require(not raw.endswith(b"\n"), f"E_INLINE_PLAN_NEWLINE: {field}")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise common.EvidenceError(f"E_INLINE_PLAN_ASCII: {field}") from error
    digest_value = common.sha256_bytes(raw)
    try:
        return parser(text, digest_value), digest_value
    except (RuntimeError, ValueError) as error:
        raise common.EvidenceError(f"E_INLINE_PLAN: {field}: {error}") from error


def _load_v25_program(
    contract: dict[str, Any],
    program: str,
    module_name: str,
) -> types.ModuleType:
    record = contract["composition"]["v25"][program]
    return _load_source(
        module_name,
        S39 / record["path"],
        record,
    )


def _exact_source_content(
    value: dict[str, Any],
    source: dict[str, Any],
    field: str,
) -> None:
    common.exact(value["bytes"], source["bytes"], f"{field}.bytes")
    common.exact(value["sha256"], source["sha256"], f"{field}.sha256")


def validate_wrapper_plans(
    *,
    artifacts: dict[str, tuple[dict[str, Any], bytes]],
    plans: dict[str, dict[str, Any]],
    plan_digests: dict[str, str],
    managed: types.ModuleType,
    adapter: types.ModuleType,
    contract: dict[str, Any],
    preparation: dict[str, Any],
    discovery: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result = {}
    role_map = {
        "cuda_monolithic": (
            "plan.wrapper.cuda_monolithic",
            "plan.managed.cuda_monolithic",
            "cuda_monolithic_producer",
        ),
        "joint_phone_cuda": (
            "plan.wrapper.joint_phone_cuda",
            "plan.managed.joint_phone_cuda",
            "joint_phone_cuda_producer",
        ),
    }
    launcher_source = contract["composition"]["v23"][
        "managed_runtime_launcher"
    ]
    for execution_role, (
        wrapper_role,
        managed_role,
        producer_source_name,
    ) in role_map.items():
        value, raw = artifacts[wrapper_role]
        try:
            wrapper = adapter.validate_plan(value)
        except (RuntimeError, ValueError) as error:
            raise common.EvidenceError(
                f"E_WRAPPER_PLAN: {wrapper_role}: {error}"
            ) from error
        common.exact(wrapper["role"], execution_role, f"{wrapper_role}.role")
        digest_value = common.sha256_bytes(raw)
        plan_digests[wrapper_role] = digest_value
        common.exact(
            wrapper["managed_plan_sha256"],
            plan_digests[managed_role],
            f"{wrapper_role}.managed_plan",
        )
        common.exact(
            wrapper["local_python"],
            preparation["local_python"],
            f"{wrapper_role}.local_python",
        )
        _exact_source_content(
            wrapper["managed_launcher"],
            launcher_source,
            f"{wrapper_role}.managed_launcher",
        )
        _exact_source_content(
            wrapper["frozen_producer"],
            contract["composition"]["v24"][producer_source_name],
            f"{wrapper_role}.frozen_producer",
        )
        for key in ("local_python", "managed_launcher"):
            try:
                adapter.verify_local_artifact(
                    wrapper[key],
                    f"{wrapper_role}.{key}",
                )
            except (OSError, RuntimeError, ValueError) as error:
                raise common.EvidenceError(
                    f"E_WRAPPER_LOCAL_ARTIFACT: {wrapper_role}.{key}: {error}"
                ) from error
        try:
            adapter.validate_managed_plan(
                managed,
                artifacts[managed_role][1],
                plan_digests[managed_role],
                wrapper,
            )
        except (RuntimeError, ValueError) as error:
            raise common.EvidenceError(
                f"E_WRAPPER_MANAGED_PLAN: {wrapper_role}: {error}"
            ) from error
        managed_plan = plans[managed_role]
        common.exact(
            managed_plan["mode"],
            "remote_cuda",
            f"{managed_role}.mode",
        )
        common.exact(
            managed_plan["ssh"]["ssh_target"],
            contract["topology"]["cuda_ssh_target"],
            f"{managed_role}.ssh_target",
        )
        common.exact(
            managed_plan["ssh"]["gpu_uuid"],
            contract["topology"]["cuda_gpu_uuid"],
            f"{managed_role}.gpu_uuid",
        )
        common.exact(
            managed_plan["ssh"]["_expected_boot_id"],
            discovery["cuda"]["boot_id"],
            f"{managed_role}.boot_id",
        )
        result[execution_role] = wrapper

    joint = result["joint_phone_cuda"]["joint_bindings"]
    common.exact(
        joint["adb_server_port"],
        5038,
        "plan.wrapper.joint_phone_cuda.adb_port",
    )
    for phone in ("op12", "op15"):
        common.exact(
            joint[f"{phone}_selector"],
            discovery["phones"][phone]["wifi_selector"],
            f"plan.wrapper.joint_phone_cuda.{phone}_selector",
        )
    common.exact(
        plans["plan.managed.cuda_monolithic"]["ssh"],
        plans["plan.managed.joint_phone_cuda"]["ssh"],
        "E_REMOTE_CUDA_SSH_SPLICE",
    )
    return result


def validate_fan_in_plan_binding(
    *,
    fan_in_plan: dict[str, Any],
    managed_plan: dict[str, Any],
    managed_plan_raw: bytes,
    managed_plan_sha256: str,
    adapter: types.ModuleType,
    contract: dict[str, Any],
    preparation: dict[str, Any],
    discovery: dict[str, Any],
) -> None:
    common.exact(
        fan_in_plan["managed_plan_sha256"],
        managed_plan_sha256,
        "fan_in_plan.managed_plan",
    )
    common.exact(
        fan_in_plan["managed_plan"]["bytes"],
        len(managed_plan_raw),
        "fan_in_plan.managed_plan.bytes",
    )
    common.exact(
        fan_in_plan["managed_plan"]["sha256"],
        common.sha256_bytes(managed_plan_raw),
        "fan_in_plan.managed_plan.sha256",
    )
    try:
        adapter.verify_local_artifact(
            fan_in_plan["managed_plan"],
            "fan_in_plan.managed_plan",
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise common.EvidenceError(
            f"E_FAN_IN_MANAGED_PLAN_ARTIFACT: {error}"
        ) from error
    common.exact(
        fan_in_plan["local_python"],
        preparation["local_python"],
        "fan_in_plan.local_python",
    )
    _exact_source_content(
        fan_in_plan["managed_launcher"],
        contract["composition"]["v23"]["managed_runtime_launcher"],
        "fan_in_plan.managed_launcher",
    )
    source_map = {
        "authority": "authority",
        "fan_in": "remote_fan_in",
        "production_common": "production_common",
        "v24_common": "common",
        "v24_contract_builder": "contract_builder",
    }
    for role, source_name in source_map.items():
        _exact_source_content(
            fan_in_plan["source_artifacts"][role],
            contract["composition"]["v24"][source_name],
            f"fan_in_plan.source.{role}",
        )
    for role, source_name in (
        ("contract_validator", "remote_fan_in_contract"),
        ("executor", "remote_fan_in_execute"),
        ("local_common", "common"),
    ):
        _exact_source_content(
            fan_in_plan[role],
            contract["composition"]["v25"][source_name],
            f"fan_in_plan.{role}",
        )
    remote_python = fan_in_plan["remote_python"]
    common.exact(
        remote_python["path"],
        contract["topology"]["cuda_python_path"],
        "fan_in_plan.remote_python.path",
    )
    common.exact(
        remote_python["bytes"],
        contract["topology"]["cuda_python_bytes"],
        "fan_in_plan.remote_python.bytes",
    )
    common.exact(
        remote_python["sha256"],
        contract["topology"]["cuda_python_sha256"],
        "fan_in_plan.remote_python.sha256",
    )
    common.exact(managed_plan["mode"], "remote_cuda", "fan_in_managed.mode")
    common.exact(managed_plan["endpoint"], "cuda", "fan_in_managed.endpoint")
    common.exact(
        managed_plan["ssh"]["ssh_target"],
        contract["topology"]["cuda_ssh_target"],
        "fan_in_managed.ssh_target",
    )
    common.exact(
        managed_plan["ssh"]["gpu_uuid"],
        contract["topology"]["cuda_gpu_uuid"],
        "fan_in_managed.gpu_uuid",
    )
    common.exact(
        managed_plan["ssh"]["_expected_boot_id"],
        discovery["cuda"]["boot_id"],
        "fan_in_managed.boot_id",
    )
    common.exact(
        managed_plan["_normalized"]["argv"],
        fan_in_plan["producer_argv"],
        "fan_in_managed.argv",
    )
    components = {
        component["path"]: component
        for component in managed_plan["components"]
    }
    required = [
        fan_in_plan["remote_python"],
        *fan_in_plan["source_artifacts"].values(),
        *fan_in_plan["input_artifacts"].values(),
    ]
    for index, artifact in enumerate(required):
        common.require(
            artifact["path"] in components,
            f"E_FAN_IN_COMPONENT_MISSING: {index}",
        )
        common.exact(
            components[artifact["path"]],
            artifact,
            f"fan_in_managed.component[{index}]",
        )


def _materialized_filename(role: str) -> str:
    if role == "phase.preparation":
        return "phase-preparation.json"
    if role == "phase.discovery":
        return "phase-discovery.json"
    if role == "history.remote_plan":
        return "history-remote-plan.json"
    if role.startswith("inner."):
        return f"inner-{role.removeprefix('inner.').replace('_', '-')}.json"
    if role.startswith("plan.managed."):
        return f"managed-{role.removeprefix('plan.managed.')}.json"
    if role.startswith("plan.wrapper."):
        return f"wrapper-{role.removeprefix('plan.wrapper.')}.json"
    if role == "plan.remote_phone_guard":
        return "remote-phone-guard-plan.json"
    if role == "plan.remote_phone_guard_policy":
        return "remote-phone-policy.json"
    if role == "plan.remote_fan_in":
        return "remote-fan-in-plan.json"
    raise common.EvidenceError(f"E_MATERIALIZATION_ROLE: {role}")


def validate_materialized_prelock(
    *,
    artifacts: dict[str, tuple[dict[str, Any], bytes]],
    contract: dict[str, Any],
    discovery_raw: bytes,
    discovery_value: dict[str, Any],
) -> dict[str, Any]:
    materializer = _load_v25_program(
        contract,
        "materializer",
        "s39_v25_materializer_authority",
    )
    try:
        inventory = materializer.validate_inventory(
            copy.deepcopy(artifacts["phase.inventory"][0]),
            contract,
        )
        identity = materializer.validate_identity(
            copy.deepcopy(artifacts["phase.fresh_identity"][0]),
            artifacts["phase.fresh_identity"][1],
            copy.deepcopy(discovery_value),
            discovery_raw,
            contract,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise common.EvidenceError(
            f"E_MATERIALIZATION_INPUT: {error}"
        ) from error

    common.exact(
        inventory["phase_id"],
        discovery_value["phase_id"],
        "materialization.inventory.phase_id",
    )
    common.exact(
        identity["outer_phase_id"],
        inventory["phase_id"],
        "materialization.identity.outer_phase_id",
    )
    common.exact(
        identity["v24_phase_id"],
        inventory["v24_phase_id"],
        "materialization.identity.v24_phase_id",
    )
    common.exact(
        identity["local_artifacts"]["python"],
        artifacts["phase.preparation"][0]["local_python"],
        "materialization.identity.python",
    )

    root = common.exact_keys(
        artifacts["phase.materialization"][0],
        {
            "artifacts",
            "completed_ns",
            "discovery_sha256",
            "fan_in_materialized",
            "fresh_identity_sha256",
            "inventory_sha256",
            "outer_phase_id",
            "phase",
            "schema",
            "stages",
            "started_ns",
            "status",
            "v24_phase_id",
        },
        "materialization",
    )
    common.exact(
        root["schema"],
        "s39-v25-a-only-materialization-v1",
        "materialization.schema",
    )
    common.exact(root["phase"], PHASE, "materialization.phase")
    common.exact(
        root["status"],
        "A_ONLY_PLANS_MATERIALIZED_NO_HARDWARE_RUN",
        "materialization.status",
    )
    common.exact(
        root["fan_in_materialized"],
        True,
        "materialization.fan_in",
    )
    common.exact(
        root["outer_phase_id"],
        inventory["phase_id"],
        "materialization.outer_phase_id",
    )
    common.exact(
        root["v24_phase_id"],
        inventory["v24_phase_id"],
        "materialization.v24_phase_id",
    )
    for key, raw in (
        ("discovery_sha256", discovery_raw),
        ("fresh_identity_sha256", artifacts["phase.fresh_identity"][1]),
        ("inventory_sha256", artifacts["phase.inventory"][1]),
    ):
        common.exact(
            root[key],
            common.sha256_bytes(raw),
            f"materialization.{key}",
        )
    started = common.integer(
        root["started_ns"],
        "materialization.started_ns",
        1,
    )
    completed = common.integer(
        root["completed_ns"],
        "materialization.completed_ns",
        started + 1,
    )
    common.require(
        identity["completed_ns"] <= started,
        "E_MATERIALIZATION_BEFORE_IDENTITY",
    )

    rows = root["artifacts"]
    common.require(
        type(rows) is list and len(rows) == len(MATERIALIZED_ROLES),
        "E_MATERIALIZATION_ARTIFACTS",
    )
    seen: set[str] = set()
    previous = None
    for index, row in enumerate(rows):
        field = f"materialization.artifacts[{index}]"
        row = common.exact_keys(
            row,
            {"bytes", "path", "role", "sha256"},
            field,
        )
        role = common.text(row["role"], f"{field}.role", 128)
        common.require(
            role in MATERIALIZED_ROLES and role not in seen,
            f"E_MATERIALIZATION_ARTIFACT_ROLE: {role}",
        )
        if previous is not None:
            common.require(previous < role, "E_MATERIALIZATION_ARTIFACT_ORDER")
        previous = role
        seen.add(role)
        common.exact(
            row["path"],
            _materialized_filename(role),
            f"{field}.path",
        )
        raw = artifacts[role][1]
        common.exact(row["bytes"], len(raw), f"{field}.bytes")
        common.exact(
            row["sha256"],
            common.sha256_bytes(raw),
            f"{field}.sha256",
        )
    common.exact(seen, MATERIALIZED_ROLES, "materialization.artifact_roles")

    stages = common.exact_keys(
        root["stages"],
        set(STAGE_ORDER),
        "materialization.stages",
    )
    expected_entrypoints = {
        "remote_history": "remote_history",
        "phone_guard_before": "phone_guard",
        "cuda_monolithic": "remote_cuda_capture",
        "joint_phone_cuda": "remote_cuda_capture",
        "remote_fan_in": "remote_fan_in_execute",
        "phone_guard_after": "phone_guard",
    }
    expected_support = {
        "remote_history": {},
        "phone_guard_before": {},
        "cuda_monolithic": {
            "managed_launcher": identity["local_artifacts"]["managed_launcher"],
        },
        "joint_phone_cuda": {
            "managed_launcher": identity["local_artifacts"]["managed_launcher"],
        },
        "remote_fan_in": {
            "contract": identity["local_artifacts"]["remote_fan_in_contract"],
            "managed_launcher": identity["local_artifacts"]["managed_launcher"],
            "v25_common": identity["local_artifacts"]["v25_common"],
        },
        "phone_guard_after": {},
    }
    expected_timeouts = {
        "remote_history": 720,
        "phone_guard_before": 600,
        "cuda_monolithic": 7200,
        "joint_phone_cuda": 7200,
        "remote_fan_in": 7200,
        "phone_guard_after": 600,
    }
    expected_environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    output_root = Path(
        common.absolute_path(
            stages["remote_history"]["expected_output"],
            "materialization.output_root",
        )
    ).parent
    expected_outputs = {
        "remote_history": output_root / "history-remote-receipt.json",
        "phone_guard_before": output_root / "phone-guard-before.json",
        "cuda_monolithic": output_root / "cuda-monolithic-receipt.json",
        "joint_phone_cuda": output_root / "joint-phone-cuda-receipt.json",
        "remote_fan_in": output_root / "remote-fan-in-receipt.json",
        "phone_guard_after": output_root / "phone-guard-after.json",
    }
    python_path = identity["local_artifacts"]["python"]["path"]
    guard_plan_path = output_root / _materialized_filename(
        "plan.remote_phone_guard"
    )
    guard_plan_sha256 = common.sha256_bytes(
        artifacts["plan.remote_phone_guard"][1]
    )
    expected_argv = {
        "remote_history": [
            python_path,
            identity["local_artifacts"]["remote_history"]["path"],
            "--plan-json",
            artifacts["history.remote_plan"][1].decode("ascii"),
            "--plan-sha256",
            common.sha256_bytes(artifacts["history.remote_plan"][1]),
            "--boot-id",
            identity["rtx_boot_id"],
            "--output",
            str(expected_outputs["remote_history"]),
        ],
        "phone_guard_before": [
            python_path,
            identity["local_artifacts"]["phone_guard"]["path"],
            "--plan",
            str(guard_plan_path),
            "--plan-sha256",
            guard_plan_sha256,
            "--moment",
            "before",
            "--receipt",
            str(expected_outputs["phone_guard_before"]),
            "--execute",
            "--confirm",
            "RUN_CP0_R1_V25_PHONE_GUARD",
        ],
        "phone_guard_after": [
            python_path,
            identity["local_artifacts"]["phone_guard"]["path"],
            "--plan",
            str(guard_plan_path),
            "--plan-sha256",
            guard_plan_sha256,
            "--moment",
            "after",
            "--receipt",
            str(expected_outputs["phone_guard_after"]),
            "--execute",
            "--confirm",
            "RUN_CP0_R1_V25_PHONE_GUARD",
        ],
    }
    for stage in ("cuda_monolithic", "joint_phone_cuda"):
        wrapper_role = f"plan.wrapper.{stage}"
        managed_role = f"plan.managed.{stage}"
        expected_argv[stage] = [
            python_path,
            identity["local_artifacts"]["remote_cuda_capture"]["path"],
            "--plan",
            str(output_root / _materialized_filename(wrapper_role)),
            "--plan-sha256",
            common.sha256_bytes(artifacts[wrapper_role][1]),
            "--managed-plan",
            str(output_root / _materialized_filename(managed_role)),
            "--managed-plan-sha256",
            common.sha256_bytes(artifacts[managed_role][1]),
            "--remote-boot-id",
            identity["rtx_boot_id"],
            "--output",
            str(output_root / f"{stage.replace('_', '-')}.json"),
            "--receipt",
            str(expected_outputs[stage]),
            "--execute",
            "--confirm",
            "RUN-S39-V25-REMOTE-CUDA",
        ]
    expected_argv["remote_fan_in"] = [
        python_path,
        identity["local_artifacts"]["remote_fan_in_execute"]["path"],
        "--plan",
        str(output_root / _materialized_filename("plan.remote_fan_in")),
        "--plan-sha256",
        common.sha256_bytes(artifacts["plan.remote_fan_in"][1]),
        "--boot-id",
        identity["rtx_boot_id"],
        "--bundle-root",
        str(output_root / "remote-fan-in-bundle"),
        "--receipt",
        str(expected_outputs["remote_fan_in"]),
        "--execute",
        "--confirm",
        "RUN_CP0_R1_V25_REMOTE_FAN_IN",
    ]
    output_paths: set[str] = set()
    for stage in STAGE_ORDER:
        descriptor = common.exact_keys(
            stages[stage],
            {
                "argv",
                "cwd",
                "entrypoint",
                "environment",
                "expected_output",
                "support",
                "timeout_seconds",
            },
            f"materialization.stages.{stage}",
        )
        entrypoint = identity["local_artifacts"][
            expected_entrypoints[stage]
        ]
        common.exact(
            descriptor["entrypoint"],
            entrypoint,
            f"materialization.stages.{stage}.entrypoint",
        )
        common.exact(
            descriptor["support"],
            expected_support[stage],
            f"materialization.stages.{stage}.support",
        )
        common.exact(
            descriptor["environment"],
            expected_environment,
            f"materialization.stages.{stage}.environment",
        )
        common.exact(
            descriptor["cwd"],
            str(S39.parents[2]),
            f"materialization.stages.{stage}.cwd",
        )
        common.exact(
            descriptor["timeout_seconds"],
            expected_timeouts[stage],
            f"materialization.stages.{stage}.timeout",
        )
        argv = descriptor["argv"]
        common.require(
            type(argv) is list
            and len(argv) >= 2
            and all(type(value) is str and value.isascii() for value in argv),
            f"E_MATERIALIZATION_ARGV: {stage}",
        )
        common.exact(
            argv,
            expected_argv[stage],
            f"materialization.stages.{stage}.argv",
        )
        output = common.absolute_path(
            descriptor["expected_output"],
            f"materialization.stages.{stage}.expected_output",
        )
        common.require(
            output not in output_paths,
            f"E_MATERIALIZATION_OUTPUT_REUSE: {stage}",
        )
        common.exact(
            output,
            str(expected_outputs[stage]),
            f"materialization.stages.{stage}.expected_output",
        )
        output_paths.add(output)
    return {
        "completed_ns": completed,
        "identity": identity,
        "inventory": inventory,
        "stages": stages,
        "started_ns": started,
    }


def validate_phase_lock(
    value: Any,
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate_raw: bytes,
    history_raw: bytes,
    discovery_raw: bytes,
    discovery: dict[str, Any],
    plan_digests: dict[str, str],
) -> dict[str, Any]:
    value = common.exact_keys(
        value,
        {
            "candidate_sha256",
            "contract_sha256",
            "device_boot_ids",
            "discovery_sha256",
            "event_ns",
            "phase",
            "phase_id",
            "plan_sha256s",
            "schema",
            "system_swap_baseline_bytes",
            "token_history_sha256",
            "v24_phase_id",
            "wifi_selectors",
        },
        "phase_lock",
    )
    common.exact(
        value["schema"],
        "s39-cp0-r1-v25-phase-lock-v1",
        "phase_lock.schema",
    )
    common.exact(value["phase"], PHASE, "phase_lock.phase")
    phase_id = common.text(value["phase_id"], "phase_lock.phase_id", 128)
    common.require(
        phase_id.startswith("cp0-r1-v25-a-only-"),
        "E_PHASE_ID",
    )
    common.exact(phase_id, discovery["phase_id"], "phase_lock.phase_id")
    common.exact(
        value["contract_sha256"],
        common.sha256_bytes(contract_raw),
        "phase_lock.contract",
    )
    common.exact(
        value["candidate_sha256"],
        common.sha256_bytes(candidate_raw),
        "phase_lock.candidate",
    )
    common.exact(
        value["token_history_sha256"],
        common.sha256_bytes(history_raw),
        "phase_lock.history",
    )
    common.exact(
        value["discovery_sha256"],
        common.sha256_bytes(discovery_raw),
        "phase_lock.discovery",
    )
    v24_phase_id = common.text(
        value["v24_phase_id"],
        "phase_lock.v24_phase_id",
        128,
    )
    common.require(
        v24_phase_id.startswith("cp0-r1-v24-a-only-")
        and v24_phase_id != phase_id,
        "E_V24_PHASE_ID",
    )
    common.exact(value["plan_sha256s"], plan_digests, "phase_lock.plans")
    event = common.integer(value["event_ns"], "phase_lock.event_ns", 1)
    common.require(discovery["completed_ns"] <= event, "E_LOCK_BEFORE_DISCOVERY")

    expected_boots = {
        "controller": discovery["controller_boot_id"],
        "cuda": discovery["cuda"]["boot_id"],
        "op12": discovery["phones"]["op12"]["boot_id"],
        "op15": discovery["phones"]["op15"]["boot_id"],
    }
    common.exact(value["device_boot_ids"], expected_boots, "phase_lock.boot_ids")
    common.exact(
        value["wifi_selectors"],
        {
            phone: discovery["phones"][phone]["wifi_selector"]
            for phone in ("op12", "op15")
        },
        "phase_lock.wifi_selectors",
    )
    common.exact(
        value["system_swap_baseline_bytes"],
        {
            "cuda": discovery["cuda"]["system_swap_used_bytes"],
            "op12": discovery["phones"]["op12"]["system_swap_used_bytes"],
            "op15": discovery["phones"]["op15"]["system_swap_used_bytes"],
        },
        "phase_lock.swap",
    )
    return {
        "boot_ids": expected_boots,
        "event_ns": event,
        "phase_id": phase_id,
        "swap": value["system_swap_baseline_bytes"],
        "v24_phase_id": v24_phase_id,
    }


def _validate_artifact_identity(
    value: Any,
    expected: dict[str, Any],
    field: str,
) -> None:
    value = common.exact_keys(
        value,
        {"bytes", "path", "sha256", "stat"},
        field,
    )
    for key in ("bytes", "path", "sha256"):
        common.exact(value[key], expected[key], f"{field}.{key}")
    stat_record = common.exact_keys(
        value["stat"],
        {"ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"},
        f"{field}.stat",
    )
    for key in stat_record:
        common.integer(stat_record[key], f"{field}.stat.{key}")
    common.exact(stat_record["size"], value["bytes"], f"{field}.stat.size")
    common.require(
        stat_record["inode"] > 0 and stat.S_ISREG(stat_record["mode"]),
        f"E_STAT: {field}",
    )


def validate_remote_history(
    plan: Any,
    receipt: Any,
    contract: dict[str, Any],
    candidate_raw: bytes,
    history_raw: bytes,
    tokenizer_plan_raw: bytes,
    discovery: dict[str, Any],
    minimum_ns: int,
) -> dict[str, Any]:
    plan = common.exact_keys(
        plan,
        {
            "command_argv",
            "inputs",
            "remote_cwd",
            "schema",
            "ssh",
            "support",
        },
        "history_plan",
    )
    common.exact(
        plan["schema"],
        "s39-v25-remote-history-validation-plan-v1",
        "history_plan.schema",
    )
    ssh = plan["ssh"]
    common.require(type(ssh) is dict, "E_TYPE: history_plan.ssh")
    common.exact(
        ssh.get("ssh_target"),
        contract["topology"]["cuda_ssh_target"],
        "history_plan.ssh_target",
    )
    common.exact(
        ssh.get("boot_id_source"),
        "phase_fresh_snapshot",
        "history_plan.boot_id_source",
    )
    for key in (
        "ssh_sha256",
        "known_hosts_sha256",
        "identity_file_sha256",
        "identity_public_key_sha256",
        "remote_python_sha256",
    ):
        common.digest(ssh.get(key), f"history_plan.ssh.{key}")
    inputs = common.exact_keys(
        plan["inputs"],
        {"candidate", "corpus", "history", "tokenizer_plan"},
        "history_plan.inputs",
    )
    common.exact(
        inputs,
        contract["quality"]["remote_inputs"],
        "history_plan.inputs",
    )
    common.exact(
        inputs["candidate"]["sha256"],
        common.sha256_bytes(candidate_raw),
        "history_plan.candidate",
    )
    common.exact(
        inputs["history"]["sha256"],
        common.sha256_bytes(history_raw),
        "history_plan.history",
    )
    common.exact(
        inputs["tokenizer_plan"]["sha256"],
        common.sha256_bytes(tokenizer_plan_raw),
        "history_plan.tokenizer_plan",
    )
    common.exact(
        inputs["corpus"]["sha256"],
        contract["quality"]["corpus_sha256"],
        "history_plan.corpus",
    )
    support = common.exact_keys(
        plan["support"],
        {"history_common", "python", "validator"},
        "history_plan.support",
    )
    expected_support = contract["composition"]["history"]
    for key in ("history_common", "validator"):
        source = expected_support[key]["source"]
        common.exact(
            support[key],
            {
                "bytes": source["bytes"],
                "path": expected_support[key]["remote_path"],
                "sha256": source["sha256"],
            },
            f"history_plan.support.{key}",
        )
    common.exact(
        support["python"]["path"],
        contract["topology"]["cuda_python_path"],
        "history_plan.support.python.path",
    )
    common.exact(
        ssh.get("remote_python_path"),
        contract["topology"]["cuda_python_path"],
        "history_plan.ssh.remote_python_path",
    )
    common.exact(
        support["python"]["sha256"],
        contract["topology"]["cuda_python_sha256"],
        "history_plan.support.python.sha256",
    )
    common.exact(
        ssh.get("remote_python_sha256"),
        contract["topology"]["cuda_python_sha256"],
        "history_plan.ssh.remote_python_sha256",
    )
    remote_python_stat = ssh.get("remote_python_stat")
    common.require(type(remote_python_stat) is dict, "E_HISTORY_PYTHON_STAT")
    common.exact(
        support["python"]["bytes"],
        contract["topology"]["cuda_python_bytes"],
        "history_plan.support.python.bytes",
    )
    common.exact(
        remote_python_stat.get("size"),
        contract["topology"]["cuda_python_bytes"],
        "history_plan.ssh.remote_python_size",
    )
    common.exact(
        plan["remote_cwd"],
        contract["topology"]["cuda_prephase_root"],
        "history_plan.remote_cwd",
    )
    command = plan["command_argv"]
    common.require(
        type(command) is list
        and len(command) == 14
        and all(type(item) is str and item.isascii() for item in command),
        "E_HISTORY_COMMAND",
    )
    common.exact(
        command[0:6],
        [
            support["python"]["path"],
            "-I",
            "-c",
            VALIDATOR_WRAPPER,
            str(Path(support["validator"]["path"]).parent),
            support["validator"]["path"],
        ],
        "history_plan.command.prefix",
    )
    expected_flags = (
        ("--candidate", inputs["candidate"]["path"]),
        ("--corpus", inputs["corpus"]["path"]),
        ("--history", inputs["history"]["path"]),
        ("--tokenizer-plan", inputs["tokenizer_plan"]["path"]),
    )
    for index, expected in enumerate(expected_flags):
        offset = 6 + index * 2
        common.exact(command[offset:offset + 2], list(expected), f"history_plan.command[{index}]")

    receipt = common.exact_keys(
        receipt,
        {
            "boot_id",
            "completed_ns",
            "executed_argv",
            "host",
            "observed_inputs",
            "observed_support",
            "phase",
            "plan_sha256",
            "returncode",
            "schema",
            "ssh_target",
            "started_ns",
            "stderr",
            "stdout",
        },
        "history_receipt",
    )
    common.exact(
        receipt["schema"],
        "s39-v25-remote-history-validation-receipt-v1",
        "history_receipt.schema",
    )
    common.exact(receipt["phase"], PHASE, "history_receipt.phase")
    common.exact(
        receipt["plan_sha256"],
        common.sha256_bytes(common.canonical_compact(plan)),
        "history_receipt.plan",
    )
    common.exact(
        receipt["boot_id"],
        discovery["cuda"]["boot_id"],
        "history_receipt.boot",
    )
    common.exact(
        receipt["host"],
        contract["topology"]["cuda_host"],
        "history_receipt.host",
    )
    common.exact(
        receipt["ssh_target"],
        contract["topology"]["cuda_ssh_target"],
        "history_receipt.ssh_target",
    )
    common.exact(receipt["executed_argv"], command, "history_receipt.argv")
    common.exact(receipt["returncode"], 0, "history_receipt.returncode")
    common.exact(
        receipt["stdout"],
        "B8_HISTORY_VALIDATE_PASS\n",
        "history_receipt.stdout",
    )
    common.exact(receipt["stderr"], "", "history_receipt.stderr")
    started = common.integer(
        receipt["started_ns"],
        "history_receipt.started",
        minimum_ns,
    )
    completed = common.integer(receipt["completed_ns"], "history_receipt.completed", started + 1)
    observed_inputs = common.exact_keys(
        receipt["observed_inputs"],
        set(inputs),
        "history_receipt.observed_inputs",
    )
    observed_support = common.exact_keys(
        receipt["observed_support"],
        set(support),
        "history_receipt.observed_support",
    )
    for key in inputs:
        _validate_artifact_identity(
            observed_inputs[key],
            inputs[key],
            f"history_receipt.observed_inputs.{key}",
        )
    for key in support:
        _validate_artifact_identity(
            observed_support[key],
            support[key],
            f"history_receipt.observed_support.{key}",
        )
    return {
        "completed_ns": completed,
        "observed_support": observed_support,
        "plan": plan,
    }


def _validate_process_record(
    value: Any,
    plan: dict[str, Any],
    boot_id: str,
    lock_ns: int,
    field: str,
) -> dict[str, Any]:
    value = common.exact_keys(
        value,
        {
            "boot_id",
            "bundle_id",
            "endpoint",
            "launcher_path",
            "loaded_repo_component_ids",
            "observed_ns",
            "pid",
            "schema",
            "start_ticks",
            "system_dependencies",
        },
        field,
    )
    common.exact(value["schema"], "s39-runtime-process-source-v1", f"{field}.schema")
    common.exact(value["boot_id"], boot_id, f"{field}.boot_id")
    for key in ("bundle_id", "endpoint"):
        common.exact(value[key], plan[key], f"{field}.{key}")
    normalized = plan["_normalized"]
    common.exact(
        value["launcher_path"],
        normalized["launcher_path"],
        f"{field}.launcher",
    )
    common.exact(
        value["loaded_repo_component_ids"],
        sorted(normalized["component_map"]),
        f"{field}.components",
    )
    common.integer(value["pid"], f"{field}.pid", 1)
    common.integer(value["start_ticks"], f"{field}.start_ticks", 1)
    observed = common.integer(value["observed_ns"], f"{field}.observed_ns", lock_ns)
    dependencies = value["system_dependencies"]
    common.require(type(dependencies) is list, f"E_TYPE: {field}.dependencies")
    expected_dependencies = sorted(
        [
            {
                **component["stat"],
                "path": component["path"],
            }
            for component in plan["components"]
        ],
        key=lambda item: item["path"],
    )
    common.exact(
        dependencies,
        expected_dependencies,
        f"{field}.dependencies",
    )
    return {
        "argv": normalized["argv"],
        "boot_id": boot_id,
        "bundle_id": value["bundle_id"],
        "endpoint": value["endpoint"],
        "launcher_path": normalized["launcher_path"],
        "observed_ns": observed,
        "pid": value["pid"],
        "start_ticks": value["start_ticks"],
        "record": value,
    }


def _validate_probe_output(
    value: Any,
    plan: dict[str, Any],
    discovery: dict[str, Any],
    process: dict[str, Any],
    network_process: dict[str, Any],
    minimum_available_bytes: int,
    lock_ns: int,
    field: str,
) -> dict[str, Any]:
    raise common.EvidenceError("E_LEGACY_OUTER_PHONE_PROBE_REMOVED")


def _validate_probe_output_unreachable(
    value: Any,
    plan: dict[str, Any],
    process: dict[str, Any],
    phone: str,
    moment: str,
    contract: dict[str, Any],
    discovery: dict[str, Any],
    lock: dict[str, Any],
    field: str,
) -> dict[str, Any]:
    value = common.exact_keys(
        value,
        {
            "adb_path",
            "adb_port",
            "adb_selector",
            "adb_sha256",
            "completed_ns",
            "remote",
            "schema",
            "stage_status_source",
            "started_ns",
        },
        field,
    )
    common.exact(value["schema"], "s39-phone-runtime-probe-v1", f"{field}.schema")
    common.exact(value["stage_status_source"], "relay_owned_status", f"{field}.stage")
    android = plan["android"]
    for key in ("adb_path", "adb_port", "adb_selector", "adb_sha256"):
        common.exact(value[key], android[key], f"{field}.{key}")
    started = common.integer(value["started_ns"], f"{field}.started", lock_ns)
    completed = common.integer(value["completed_ns"], f"{field}.completed", started + 1)
    remote = common.exact_keys(
        value["remote"],
        {
            "available_bytes",
            "boot_id",
            "direct_peer",
            "gpu_max_millic",
            "interface",
            "network_process",
            "physical_serial",
            "process",
            "process_swap_bytes",
            "shard_artifact",
            "system_swap_used_bytes",
            "thermal_status",
            "thermal_zones",
            "worker_artifact",
        },
        f"{field}.remote",
    )
    phone = plan["endpoint"]
    discovered = discovery["phones"][phone]
    common.exact(remote["boot_id"], discovered["boot_id"], f"{field}.boot")
    common.exact(
        remote["physical_serial"],
        discovered["physical_serial"],
        f"{field}.serial",
    )
    common.exact(remote["process_swap_bytes"], 0, f"{field}.process_swap")
    common.exact(remote["thermal_status"], 0, f"{field}.thermal")
    available = common.integer(remote["available_bytes"], f"{field}.available", 1)
    common.require(
        available >= minimum_available_bytes,
        f"E_MEMORY_HEADROOM: {field}",
    )
    gpu_max = common.integer(remote["gpu_max_millic"], f"{field}.gpu_max", 1)
    common.require(
        gpu_max <= plan["telemetry"]["max_gpu_millic"],
        f"E_GPU_THERMAL: {field}",
    )
    zones = remote["thermal_zones"]
    common.require(type(zones) is list and bool(zones), f"E_THERMAL_ZONES: {field}")
    seen_zones: set[str] = set()
    observed_gpu = []
    previous_zone = None
    for index, zone in enumerate(zones):
        zone_field = f"{field}.thermal_zones[{index}]"
        common.exact_keys(zone, {"name", "temp_millic"}, zone_field)
        name = common.text(zone["name"], f"{zone_field}.name", 256)
        common.require(name not in seen_zones, f"E_THERMAL_ZONE_DUPLICATE: {field}")
        seen_zones.add(name)
        if previous_zone is not None:
            common.require(previous_zone < name, f"E_THERMAL_ZONE_ORDER: {field}")
        previous_zone = name
        temperature = common.integer(
            zone["temp_millic"],
            f"{zone_field}.temp_millic",
            1,
        )
        common.require(temperature <= 250_000, f"E_THERMAL_ZONE_RANGE: {field}")
        if "gpu" in name.lower() or "adreno" in name.lower():
            observed_gpu.append(temperature)
    common.require(bool(observed_gpu), f"E_GPU_THERMAL_ZONE: {field}")
    common.exact(max(observed_gpu), gpu_max, f"E_GPU_THERMAL_MAX: {field}")
    system_swap = common.integer(
        remote["system_swap_used_bytes"],
        f"{field}.system_swap",
    )
    worker = common.exact_keys(
        remote["process"],
        {"argv", "executable_path", "pid", "start_ticks"},
        f"{field}.process",
    )
    common.exact(worker["pid"], process["pid"], f"{field}.process.pid")
    common.exact(worker["start_ticks"], process["start_ticks"], f"{field}.process.ticks")
    common.exact(
        worker["executable_path"],
        plan["process"]["executable_path"],
        f"{field}.process.exe",
    )
    common.exact(worker["argv"], plan["process"]["argv"], f"{field}.process.argv")

    network = common.exact_keys(
        remote["network_process"],
        {
            "argv",
            "executable_path",
            "executable_sha256",
            "observed_stat",
            "pid",
            "role",
            "start_ticks",
        },
        f"{field}.network",
    )
    common.exact(network["pid"], network_process["pid"], f"{field}.network.pid")
    common.exact(
        network["start_ticks"],
        network_process["start_ticks"],
        f"{field}.network.ticks",
    )
    common.exact(
        network["role"],
        PHONE_BUNDLES[phone]["network_role"],
        f"{field}.network.role",
    )
    expected_network = plan["network_process"]
    common.exact(
        network["argv"],
        expected_network["argv"],
        f"{field}.network.argv",
    )
    common.exact(
        network["executable_path"],
        expected_network["executable_path"],
        f"{field}.network.executable",
    )
    common.exact(
        network["executable_sha256"],
        expected_network["artifact"]["sha256"],
        f"{field}.network.sha256",
    )
    common.exact(
        network["observed_stat"],
        expected_network["artifact"]["stat"],
        f"{field}.network.stat",
    )
    common.exact(
        remote["worker_artifact"],
        {
            **plan["worker_artifact"],
            "observed_stat": plan["worker_artifact"]["stat"],
        },
        f"{field}.worker_artifact",
    )
    common.exact(
        remote["shard_artifact"],
        {
            **plan["shard_artifact"],
            "observed_stat": plan["shard_artifact"]["stat"],
        },
        f"{field}.shard_artifact",
    )
    peer = common.exact_keys(
        remote["direct_peer"],
        {"local_ipv4", "local_port", "peer_ipv4", "peer_port", "socket_inode"},
        f"{field}.peer",
    )
    common.require(
        common.integer(peer["socket_inode"], f"{field}.peer.socket_inode", 1) > 0,
        f"E_SOCKET_INODE: {field}",
    )
    interface = common.exact_keys(
        remote["interface"],
        {"ipv4", "name", "rx_bytes", "tx_bytes"},
        f"{field}.interface",
    )
    common.exact(interface["ipv4"], discovered["wifi_ipv4"], f"{field}.ipv4")
    common.exact(interface["name"], discovered["interface"], f"{field}.interface")
    common.exact(peer["local_ipv4"], interface["ipv4"], f"{field}.peer.local")
    telemetry = plan["telemetry"]
    common.exact(
        peer["local_ipv4"],
        telemetry["local_ipv4"],
        f"{field}.peer.expected_local",
    )
    common.exact(
        peer["local_port"],
        telemetry["direct_peer_local_port"],
        f"{field}.peer.expected_local_port",
    )
    common.exact(
        peer["peer_ipv4"],
        telemetry["direct_peer_ipv4"],
        f"{field}.peer.expected_peer",
    )
    common.exact(
        peer["peer_port"],
        telemetry["direct_peer_port"],
        f"{field}.peer.expected_peer_port",
    )
    for key in ("rx_bytes", "tx_bytes"):
        common.integer(interface[key], f"{field}.interface.{key}")
    return {
        "completed_ns": completed,
        "interface": interface,
        "network": network,
        "peer": peer,
        "started_ns": started,
        "system_swap_used_bytes": system_swap,
    }


def _validate_phone_pairs(
    artifacts: dict[str, tuple[dict[str, Any], bytes]],
    plans: dict[str, dict[str, Any]],
    processes: dict[str, dict[str, Any]],
    contract: dict[str, Any],
    discovery: dict[str, Any],
    lock: dict[str, Any],
) -> dict[str, Any]:
    result = {}
    for phone in ("op12", "op15"):
        worker_bundle = PHONE_BUNDLES[phone]["worker_bundle"]
        network_bundle = PHONE_BUNDLES[phone]["network_bundle"]
        probe_plan = plans[f"plan.probe.{phone}"]
        before = _validate_probe_output(
            artifacts[f"runtime.phone.{phone}.before"][0],
            probe_plan,
            discovery,
            processes[worker_bundle],
            processes[network_bundle],
            contract["gates"]["phone_minimum_available_bytes"],
            lock["event_ns"],
            f"runtime.phone.{phone}.before",
        )
        after = _validate_probe_output(
            artifacts[f"runtime.phone.{phone}.after"][0],
            probe_plan,
            discovery,
            processes[worker_bundle],
            processes[network_bundle],
            contract["gates"]["phone_minimum_available_bytes"],
            before["completed_ns"],
            f"runtime.phone.{phone}.after",
        )
        common.require(
            before["system_swap_used_bytes"]
            <= lock["swap"][phone],
            f"E_SWAP_GROWTH_BEFORE: {phone}",
        )
        common.require(
            after["system_swap_used_bytes"]
            <= lock["swap"][phone],
            f"E_SWAP_GROWTH: {phone}",
        )
        common.require(
            after["interface"]["rx_bytes"] >= before["interface"]["rx_bytes"]
            and after["interface"]["tx_bytes"] >= before["interface"]["tx_bytes"],
            f"E_INTERFACE_COUNTER_REGRESSION: {phone}",
        )
        common.exact(
            after["network"],
            before["network"],
            f"E_NETWORK_PROCESS_CHANGED: {phone}",
        )
        common.exact(
            after["peer"],
            before["peer"],
            f"E_SOCKET_OWNER_CHANGED: {phone}",
        )
        result[phone] = {"before": before, "after": after}

    op12 = result["op12"]["after"]["peer"]
    op15 = result["op15"]["after"]["peer"]
    common.exact(op12["local_ipv4"], op15["peer_ipv4"], "E_PEER_LINK_OP12_LOCAL")
    common.exact(op12["peer_ipv4"], op15["local_ipv4"], "E_PEER_LINK_OP12_PEER")
    common.exact(op12["local_port"], op15["peer_port"], "E_PEER_LINK_OP12_PORT")
    common.exact(op12["peer_port"], op15["local_port"], "E_PEER_LINK_OP15_PORT")
    return result


def _validate_cuda_probe(
    value: Any,
    contract: dict[str, Any],
    lock: dict[str, Any],
    minimum_ns: int,
    processes: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    value = common.exact_keys(
        value,
        {
            "boot_id",
            "completed_ns",
            "gpu_uuid",
            "host",
            "processes",
            "schema",
            "ssh_target",
            "started_ns",
            "system_swap_used_bytes",
        },
        field,
    )
    common.exact(value["schema"], "s39-v25-remote-cuda-probe-v1", f"{field}.schema")
    for key in ("host", "gpu_uuid", "ssh_target"):
        common.exact(
            value[key],
            contract["topology"][f"cuda_{key}"],
            f"{field}.{key}",
        )
    common.exact(value["boot_id"], lock["boot_ids"]["cuda"], f"{field}.boot")
    started = common.integer(value["started_ns"], f"{field}.started", minimum_ns)
    completed = common.integer(value["completed_ns"], f"{field}.completed", started + 1)
    system_swap = common.integer(
        value["system_swap_used_bytes"],
        f"{field}.system_swap",
    )
    observed_processes = value["processes"]
    common.require(
        type(observed_processes) is list
        and len(observed_processes) == len(processes),
        f"E_CUDA_PROCESSES: {field}",
    )
    for index, process in enumerate(processes):
        observed = common.exact_keys(
            observed_processes[index],
            {
                "argv",
                "bundle_id",
                "executable_path",
                "pid",
                "process_swap_bytes",
                "start_ticks",
            },
            f"{field}.processes[{index}]",
        )
        common.exact(
            observed["bundle_id"],
            process["bundle_id"],
            f"{field}.processes[{index}].bundle_id",
        )
        common.exact(
            observed["pid"],
            process["pid"],
            f"{field}.processes[{index}].pid",
        )
        common.exact(
            observed["start_ticks"],
            process["start_ticks"],
            f"{field}.processes[{index}].ticks",
        )
        common.exact(
            observed["process_swap_bytes"],
            0,
            f"{field}.processes[{index}].swap",
        )
        common.exact(
            observed["argv"],
            process["argv"],
            f"{field}.processes[{index}].argv",
        )
        common.exact(
            observed["executable_path"],
            process["launcher_path"],
            f"{field}.processes[{index}].executable",
        )
    return {
        "completed_ns": completed,
        "started_ns": started,
        "system_swap_used_bytes": system_swap,
    }


def _validate_cleanup(
    value: Any,
    lock: dict[str, Any],
    processes: dict[str, dict[str, Any]],
    minimum_ns: int,
) -> int:
    value = common.exact_keys(
        value,
        {"completed_ns", "phase_id", "processes", "schema", "started_ns"},
        "cleanup",
    )
    common.exact(value["schema"], "s39-v25-runtime-cleanup-v1", "cleanup.schema")
    common.exact(value["phase_id"], lock["phase_id"], "cleanup.phase_id")
    started = common.integer(value["started_ns"], "cleanup.started", minimum_ns)
    completed = common.integer(value["completed_ns"], "cleanup.completed", started + 1)
    rows = value["processes"]
    common.require(type(rows) is list and len(rows) == len(processes), "E_CLEANUP_PROCESSES")
    observed = {}
    previous = None
    for index, row in enumerate(rows):
        field = f"cleanup.processes[{index}]"
        common.exact_keys(
            row,
            {
                "absent",
                "boot_id",
                "bundle_id",
                "checked_ns",
                "pid",
                "start_ticks",
            },
            field,
        )
        bundle = common.text(row["bundle_id"], f"{field}.bundle_id", 128)
        common.require(bundle in processes and bundle not in observed, f"E_CLEANUP_BUNDLE: {bundle}")
        if previous is not None:
            common.require(previous < bundle, "E_CLEANUP_ORDER")
        previous = bundle
        process = processes[bundle]
        for key in ("boot_id", "pid", "start_ticks"):
            common.exact(row[key], process[key], f"{field}.{key}")
        common.exact(row["absent"], True, f"{field}.absent")
        common.integer(row["checked_ns"], f"{field}.checked_ns", started)
        observed[bundle] = row
    return completed


def _validate_legacy_outer_runtime(
    artifacts: dict[str, tuple[dict[str, Any], bytes]],
    plans: dict[str, dict[str, Any]],
    contract: dict[str, Any],
    discovery: dict[str, Any],
    lock: dict[str, Any],
) -> dict[str, Any]:
    raise common.EvidenceError("E_LEGACY_OUTER_RUNTIME_REMOVED")


def _validate_legacy_outer_runtime_unreachable(
    artifacts: dict[str, tuple[dict[str, Any], bytes]],
    plans: dict[str, dict[str, Any]],
    contract: dict[str, Any],
    discovery: dict[str, Any],
    lock: dict[str, Any],
) -> dict[str, Any]:
    process_roles = {
        "cuda_monolithic": "runtime.launch.cuda_monolithic",
        "cuda_route": "runtime.launch.cuda_route",
        "op12_stagenet": "runtime.launch.op12_stagenet",
        "op15_direct_relay": "runtime.launch.op15_direct_relay",
        "op15_stagenet": "runtime.launch.op15_stagenet",
    }
    plan_roles = {
        "cuda_monolithic": "plan.managed.cuda_monolithic",
        "cuda_route": "plan.managed.cuda_route",
        "op12_stagenet": "plan.managed.op12_stagenet",
        "op15_direct_relay": "plan.managed.op15_direct_relay",
        "op15_stagenet": "plan.managed.op15_stagenet",
    }
    processes = {}
    for bundle, role in process_roles.items():
        plan = plans[plan_roles[bundle]]
        endpoint = plan["endpoint"]
        boot_id = lock["boot_ids"][endpoint]
        processes[bundle] = _validate_process_record(
            artifacts[role][0],
            plan,
            boot_id,
            lock["event_ns"],
            role,
        )
    phones = _validate_phone_pairs(
        artifacts,
        plans,
        processes,
        contract,
        discovery,
        lock,
    )
    cuda_before = _validate_cuda_probe(
        artifacts["runtime.cuda.before"][0],
        contract,
        lock,
        lock["event_ns"],
        [],
        "runtime.cuda.before",
    )
    common.require(
        cuda_before["system_swap_used_bytes"] <= lock["swap"]["cuda"],
        "E_SWAP_GROWTH: cuda.before",
    )
    cuda_processes = [
        processes["cuda_monolithic"],
    ]
    common.require(
        processes["cuda_monolithic"]["observed_ns"]
        >= cuda_before["completed_ns"],
        "E_CUDA_LAUNCH_BEFORE_BASELINE",
    )
    cuda_monolithic_during = _validate_cuda_probe(
        artifacts["runtime.cuda.monolithic.during"][0],
        contract,
        lock,
        processes["cuda_monolithic"]["observed_ns"],
        cuda_processes,
        "runtime.cuda.monolithic.during",
    )
    cuda_monolithic_cleanup = _validate_cuda_probe(
        artifacts["runtime.cuda.monolithic.cleanup"][0],
        contract,
        lock,
        cuda_monolithic_during["completed_ns"],
        [],
        "runtime.cuda.monolithic.cleanup",
    )
    later_processes = [
        processes["cuda_route"],
        processes["op12_stagenet"],
        processes["op15_direct_relay"],
        processes["op15_stagenet"],
    ]
    common.require(
        min(value["observed_ns"] for value in later_processes)
        >= cuda_monolithic_cleanup["completed_ns"],
        "E_ROUTE_LAUNCH_BEFORE_MONOLITHIC_CLEANUP",
    )
    cuda_route_during = _validate_cuda_probe(
        artifacts["runtime.cuda.route.during"][0],
        contract,
        lock,
        max(
            processes["cuda_route"]["observed_ns"],
            phones["op12"]["before"]["completed_ns"],
            phones["op15"]["before"]["completed_ns"],
        ),
        [processes["cuda_route"]],
        "runtime.cuda.route.during",
    )
    for phone in ("op12", "op15"):
        common.require(
            phones[phone]["after"]["started_ns"]
            >= cuda_route_during["completed_ns"],
            f"E_PHONE_AFTER_BEFORE_ROUTE: {phone}",
        )
    common.require(
        cuda_monolithic_cleanup["system_swap_used_bytes"]
        <= lock["swap"]["cuda"],
        "E_SWAP_GROWTH: cuda.monolithic",
    )
    common.require(
        cuda_route_during["system_swap_used_bytes"]
        <= lock["swap"]["cuda"],
        "E_SWAP_GROWTH: cuda.route",
    )
    cleanup_completed = _validate_cleanup(
        artifacts["runtime.cleanup"][0],
        lock,
        processes,
        max(
            cuda_route_during["completed_ns"],
            phones["op12"]["after"]["completed_ns"],
            phones["op15"]["after"]["completed_ns"],
        ),
    )
    cuda_cleanup = _validate_cuda_probe(
        artifacts["runtime.cuda.cleanup"][0],
        contract,
        lock,
        cleanup_completed,
        [],
        "runtime.cuda.cleanup",
    )
    common.require(
        cuda_cleanup["system_swap_used_bytes"] <= lock["swap"]["cuda"],
        "E_SWAP_GROWTH: cuda",
    )
    return {
        "completed_ns": cuda_cleanup["completed_ns"],
        "processes": processes,
    }


def _exact_wrapper_receipt(
    *,
    receipt: dict[str, Any],
    wrapper: dict[str, Any],
    wrapper_sha256: str,
    managed_sha256: str,
    discovery: dict[str, Any],
    lock: dict[str, Any],
    role: str,
) -> None:
    common.exact(receipt["role"], role, f"wrapper_receipt.{role}.role")
    common.exact(
        receipt["wrapper_plan_sha256"],
        wrapper_sha256,
        f"wrapper_receipt.{role}.wrapper_plan",
    )
    common.exact(
        receipt["managed_plan_sha256"],
        managed_sha256,
        f"wrapper_receipt.{role}.managed_plan",
    )
    common.exact(
        receipt["phase_id"],
        lock["phase_id"],
        f"wrapper_receipt.{role}.phase_id",
    )
    common.exact(
        receipt["v24_phase_id"],
        lock["v24_phase_id"],
        f"wrapper_receipt.{role}.v24_phase_id",
    )
    common.exact(
        receipt["remote_boot_id"],
        discovery["cuda"]["boot_id"],
        f"wrapper_receipt.{role}.boot_id",
    )
    common.exact(
        receipt["gpu_uuid"],
        discovery["cuda"]["gpu_uuid"],
        f"wrapper_receipt.{role}.gpu_uuid",
    )
    common.exact(
        receipt["local_python"],
        wrapper["local_python"],
        f"wrapper_receipt.{role}.local_python",
    )
    common.exact(
        receipt["frozen_producer"],
        wrapper["frozen_producer"],
        f"wrapper_receipt.{role}.producer",
    )
    common.exact(
        receipt["joint_bindings"],
        wrapper["joint_bindings"],
        f"wrapper_receipt.{role}.joint_bindings",
    )
    common.require(
        receipt["system_swap_used_bytes"] <= lock["swap"]["cuda"],
        f"E_SYSTEM_SWAP_GROWTH: {role}",
    )


def validate_runtime(
    *,
    artifacts: dict[str, tuple[dict[str, Any], bytes]],
    plans: dict[str, dict[str, Any]],
    wrapper_plans: dict[str, dict[str, Any]],
    guard_plan: dict[str, Any],
    fan_in_plan: dict[str, Any],
    adapter: types.ModuleType,
    guard: types.ModuleType,
    fan_in: types.ModuleType,
    contract: dict[str, Any],
    discovery: dict[str, Any],
    lock: dict[str, Any],
    raw_bundle_root: Path,
) -> dict[str, Any]:
    receipts = {}
    receipt_bindings = {}
    for role, artifact_role in (
        ("cuda_monolithic", "capture.cuda_monolithic.wrapper"),
        ("joint_phone_cuda", "capture.joint_phone_cuda.wrapper"),
    ):
        wrapper_role = f"plan.wrapper.{role}"
        managed_role = f"plan.managed.{role}"
        binding = (
            wrapper_plans[role],
            artifacts[wrapper_role][1],
            plans[managed_role],
            artifacts[managed_role][1],
        )
        try:
            receipt = adapter.validate_receipt(
                artifacts[artifact_role][0],
                *binding,
            )
        except (RuntimeError, ValueError) as error:
            raise common.EvidenceError(
                f"E_WRAPPER_RECEIPT: {role}: {error}"
            ) from error
        _exact_wrapper_receipt(
            receipt=receipt,
            wrapper=wrapper_plans[role],
            wrapper_sha256=common.sha256_bytes(
                artifacts[wrapper_role][1]
            ),
            managed_sha256=common.sha256_bytes(
                artifacts[managed_role][1]
            ),
            discovery=discovery,
            lock=lock,
            role=role,
        )
        local_artifacts = [receipt["local_result_artifact"]]
        local_artifacts.extend(
            row["local"] for row in receipt["local_evidence_artifacts"]
        )
        for index, local_artifact in enumerate(local_artifacts):
            try:
                adapter.verify_local_artifact(
                    local_artifact,
                    f"{artifact_role}.local[{index}]",
                )
            except (OSError, RuntimeError, ValueError) as error:
                raise common.EvidenceError(
                    f"E_WRAPPER_LOCAL_ARTIFACT: {artifact_role}[{index}]: "
                    f"{error}"
                ) from error
        receipts[role] = receipt
        receipt_bindings[role] = binding
    try:
        adapter.validate_sequence(
            [
                receipts["cuda_monolithic"],
                receipts["joint_phone_cuda"],
            ],
            [
                receipt_bindings["cuda_monolithic"],
                receipt_bindings["joint_phone_cuda"],
            ],
        )
    except (RuntimeError, ValueError) as error:
        raise common.EvidenceError(f"E_WRAPPER_SEQUENCE: {error}") from error
    common.require(
        lock["event_ns"] <= receipts["cuda_monolithic"]["started_ns"]
        < receipts["cuda_monolithic"]["completed_ns"]
        <= receipts["joint_phone_cuda"]["started_ns"]
        < receipts["joint_phone_cuda"]["completed_ns"],
        "E_CONTROLLER_WRAPPER_ORDER",
    )
    common.require(
        receipts["cuda_monolithic"]["remote_execution_interval"]["completed_ns"]
        <= receipts["joint_phone_cuda"]["remote_execution_interval"]["started_ns"],
        "E_RTX_WRAPPER_ORDER",
    )
    joint = receipts["joint_phone_cuda"]
    server = joint["adb_server_process"]
    server_plan = wrapper_plans["joint_phone_cuda"]["joint_bindings"][
        "adb_server_process"
    ]
    for key in server_plan:
        common.exact(server[key], server_plan[key], f"E_ADB_SERVER: {key}")
    common.exact(
        server["boot_id"],
        discovery["cuda"]["boot_id"],
        "E_ADB_SERVER_BOOT",
    )

    try:
        before = guard.validate_receipt_evidence(
            artifacts["runtime.remote_phone_guard.before"][0],
            guard_plan,
        )
        after = guard.validate_receipt_evidence(
            artifacts["runtime.remote_phone_guard.after"][0],
            guard_plan,
        )
        guard.validate_pair(before, after)
    except (OSError, RuntimeError, ValueError) as error:
        raise common.EvidenceError(f"E_PHONE_GUARD: {error}") from error
    guard_sha256 = common.sha256_bytes(
        artifacts["plan.remote_phone_guard"][1]
    )
    for moment, receipt in (("before", before), ("after", after)):
        common.exact(receipt["moment"], moment, f"guard.{moment}.moment")
        common.exact(
            receipt["plan_sha256"],
            guard_sha256,
            f"guard.{moment}.plan",
        )
        common.exact(
            receipt["outer_phase_id"],
            lock["phase_id"],
            f"guard.{moment}.outer_phase",
        )
        common.exact(
            receipt["inner_phase_id"],
            lock["v24_phase_id"],
            f"guard.{moment}.inner_phase",
        )
        common.exact(
            receipt["rtx_boot_id"],
            discovery["cuda"]["boot_id"],
            f"guard.{moment}.rtx_boot",
        )
        common.exact(
            receipt["ssh_transport_process"]["controller_boot_id"],
            discovery["controller_boot_id"],
            f"guard.{moment}.controller_boot",
        )
        common.exact(
            receipt["ssh_transport_process"]["argv"],
            guard.expected_ssh_argv(guard_plan, moment),
            f"guard.{moment}.ssh_argv",
        )
        for phone in ("op12", "op15"):
            expected = discovery["phones"][phone]
            observed = receipt["phones"][phone]
            for key, expected_value in (
                ("boot_id", expected["boot_id"]),
                ("interface", expected["interface"]),
                ("physical_serial", expected["physical_serial"]),
                ("wifi_ipv4", expected["wifi_ipv4"]),
                ("wifi_selector", expected["wifi_selector"]),
            ):
                common.exact(
                    observed[key],
                    expected_value,
                    f"guard.{moment}.{phone}.{key}",
                )
    common.exact(
        guard_plan["outer_phase_id"],
        lock["phase_id"],
        "guard_plan.outer_phase",
    )
    common.exact(
        guard_plan["inner_phase_id"],
        lock["v24_phase_id"],
        "guard_plan.inner_phase",
    )
    common.exact(
        guard_plan["rtx_boot_id"],
        discovery["cuda"]["boot_id"],
        "guard_plan.rtx_boot",
    )
    common.exact(
        guard_plan["gpu_uuid"],
        discovery["cuda"]["gpu_uuid"],
        "guard_plan.gpu_uuid",
    )
    common.exact(
        guard_plan["ssh_transport"]["ssh_target"],
        contract["topology"]["cuda_ssh_target"],
        "guard_plan.ssh_target",
    )
    remote_python = guard_plan["ssh_transport"]["remote_python"]
    for key, expected in (
        ("path", contract["topology"]["cuda_python_path"]),
        ("bytes", contract["topology"]["cuda_python_bytes"]),
        ("sha256", contract["topology"]["cuda_python_sha256"]),
    ):
        common.exact(
            remote_python[key],
            expected,
            f"guard_plan.remote_python.{key}",
        )
    local_guard_artifacts = [
        *guard_plan["local_artifacts"].values(),
        guard_plan["local_policy_artifact"],
        guard_plan["ssh_transport"]["ssh"],
        guard_plan["ssh_transport"]["identity_file"],
        guard_plan["ssh_transport"]["known_hosts"],
    ]
    for index, local_artifact in enumerate(local_guard_artifacts):
        try:
            guard.verify_local_artifact(
                local_artifact,
                f"guard_plan.local_artifact[{index}]",
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise common.EvidenceError(
                f"E_GUARD_LOCAL_ARTIFACT[{index}]: {error}"
            ) from error
    for phone in ("op12", "op15"):
        expected = discovery["phones"][phone]
        planned = guard_plan["phones"][phone]
        for key, expected_value in (
            ("boot_id", expected["boot_id"]),
            ("interface", expected["interface"]),
            ("physical_serial", expected["physical_serial"]),
            ("wifi_ipv4", expected["wifi_ipv4"]),
            ("wifi_selector", expected["wifi_selector"]),
        ):
            common.exact(
                planned[key],
                expected_value,
                f"guard_plan.{phone}.{key}",
            )
    common.exact(
        guard_plan["remote_artifacts"]["adb"],
        wrapper_plans["joint_phone_cuda"]["joint_bindings"]["adb"],
        "E_GUARD_ADB_SPLICE",
    )
    before_transport = before["ssh_transport_process"]["observed_ns"]
    before_cleanup = before["ssh_transport_cleanup"]["observed_ns"]
    after_transport = after["ssh_transport_process"]["observed_ns"]
    after_cleanup = after["ssh_transport_cleanup"]["observed_ns"]
    common.require(
        lock["event_ns"]
        <= before_transport
        <= before_cleanup
        <= receipts["cuda_monolithic"]["started_ns"],
        "E_CONTROLLER_GUARD_BEFORE_ORDER",
    )
    common.require(
        receipts["joint_phone_cuda"]["completed_ns"]
        <= after_transport
        <= after_cleanup,
        "E_CONTROLLER_GUARD_AFTER_ORDER",
    )
    common.require(
        before["completed_ns"]
        <= receipts["cuda_monolithic"]["remote_execution_interval"]["started_ns"]
        < receipts["cuda_monolithic"]["remote_execution_interval"]["completed_ns"]
        <= receipts["joint_phone_cuda"]["remote_execution_interval"]["started_ns"]
        < receipts["joint_phone_cuda"]["remote_execution_interval"]["completed_ns"],
        "E_RTX_GUARD_BEFORE_ORDER",
    )

    try:
        fan_binding = (
            fan_in_plan,
            artifacts["plan.remote_fan_in"][1],
            plans["plan.managed.remote_fan_in"],
            artifacts["plan.managed.remote_fan_in"][1],
        )
        fan_receipt = fan_in.validate_receipt(
            artifacts["capture.remote_fan_in.wrapper"][0],
            *fan_binding,
        )
        fan_in.validate_materialized_bundle(
            fan_receipt,
            raw_bundle_root,
            *fan_binding,
        )
    except (RuntimeError, ValueError) as error:
        raise common.EvidenceError(f"E_REMOTE_FAN_IN: {error}") from error
    common.exact(
        fan_receipt["wrapper_plan_sha256"],
        common.sha256_bytes(artifacts["plan.remote_fan_in"][1]),
        "fan_in.wrapper_plan",
    )
    common.exact(
        fan_receipt["outer_phase_id"],
        lock["phase_id"],
        "fan_in.outer_phase",
    )
    common.exact(
        fan_receipt["v24_phase_id"],
        lock["v24_phase_id"],
        "fan_in.inner_phase",
    )
    common.exact(
        fan_receipt["remote_boot_id"],
        discovery["cuda"]["boot_id"],
        "fan_in.remote_boot",
    )
    common.exact(
        fan_receipt["gpu_uuid"],
        discovery["cuda"]["gpu_uuid"],
        "fan_in.gpu_uuid",
    )
    common.require(
        fan_receipt["system_swap_used_bytes"] <= lock["swap"]["cuda"],
        "E_SYSTEM_SWAP_GROWTH: fan_in",
    )
    common.exact(
        fan_in_plan["capture_input_paths"]["cuda_monolithic"],
        receipts["cuda_monolithic"]["remote_result_artifact"]["path"],
        "fan_in.cuda_monolithic.path",
    )
    common.exact(
        fan_receipt["capture_input_artifacts"]["cuda_monolithic"],
        receipts["cuda_monolithic"]["remote_result_artifact"],
        "fan_in.cuda_monolithic",
    )
    common.exact(
        fan_in_plan["capture_input_paths"]["joint_phone_cuda"],
        receipts["joint_phone_cuda"]["remote_result_artifact"]["path"],
        "fan_in.joint_phone_cuda.path",
    )
    common.exact(
        fan_receipt["capture_input_artifacts"]["joint_phone_cuda"],
        receipts["joint_phone_cuda"]["remote_result_artifact"],
        "fan_in.joint_phone_cuda",
    )
    common.require(
        receipts["joint_phone_cuda"]["completed_ns"]
        <= fan_receipt["started_ns"]
        < fan_receipt["completed_ns"]
        <= after_transport,
        "E_CONTROLLER_FAN_IN_ORDER",
    )
    common.require(
        receipts["joint_phone_cuda"]["remote_execution_interval"]["completed_ns"]
        <= fan_receipt["remote_execution_interval"]["started_ns"]
        <= fan_receipt["remote_execution_interval"]["phase_closed_ns"]
        <= fan_receipt["remote_execution_interval"]["completed_ns"],
        "E_RTX_FAN_IN_ORDER",
    )
    common.require(
        fan_receipt["remote_execution_interval"]["completed_ns"]
        <= after["started_ns"],
        "E_RTX_GUARD_AFTER_ORDER",
    )
    return {
        "completed_ns": after_cleanup,
        "fan_in": fan_receipt,
        "guard_after": after,
        "guard_before": before,
        "receipts": receipts,
        "started_ns": before_transport,
    }


def _load_v24_authority(
    contract: dict[str, Any],
) -> types.ModuleType:
    support = contract["composition"]["v24"]
    for name in ("common", "contract_builder", "contract"):
        record = support[name]
        raw = common.read_regular(S39 / record["path"], 512 * 1024 * 1024)
        common.exact(len(raw), record["bytes"], f"source.v24.{name}.bytes")
        common.exact(
            common.sha256_bytes(raw),
            record["sha256"],
            f"source.v24.{name}.sha256",
        )
    source = support["authority"]
    return _load_source(
        "s39_v25_bound_v24_authority",
        S39 / source["path"],
        source,
    )


def _validate_v25_phone_swap(
    evidence: dict[str, Any],
    locked_swap: dict[str, int],
    minimum_available_bytes: int,
) -> dict[str, Any]:
    probes = common.exact_keys(
        evidence.get("raw_probes"),
        {"op12", "op15"},
        "v25.phone.raw_probes",
    )
    observed = {}
    for phone in ("op12", "op15"):
        pair = common.exact_keys(
            probes[phone],
            {"after", "after_ns", "before", "before_ns"},
            f"v25.phone.raw_probes.{phone}",
        )
        before = pair["before"]
        after = pair["after"]
        before_swap = common.integer(
            before.get("system_swap_used_bytes"),
            f"v25.phone.{phone}.before.system_swap",
        )
        after_swap = common.integer(
            after.get("system_swap_used_bytes"),
            f"v25.phone.{phone}.after.system_swap",
        )
        common.require(
            after_swap <= before_swap <= locked_swap[phone],
            f"E_PHONE_SYSTEM_SWAP_GROWTH: {phone}",
        )
        for moment, row in (("before", before), ("after", after)):
            common.exact(
                row.get("process_swap_bytes"),
                0,
                f"E_PHONE_PROCESS_SWAP: {phone}.{moment}",
            )
            common.require(
                common.integer(
                    row.get("available_bytes"),
                    f"v25.phone.{phone}.{moment}.available_bytes",
                )
                >= minimum_available_bytes,
                f"E_PHONE_HEADROOM: {phone}.{moment}",
            )
        observed[phone] = {
            "after_bytes": after_swap,
            "before_bytes": before_swap,
            "locked_baseline_bytes": locked_swap[phone],
        }
    return observed


def _v25_phone_probe_validator(
    original: Callable[..., None],
    locked_swap: dict[str, int],
    minimum_available_bytes: int,
) -> Callable[..., None]:
    def validate(
        evidence: dict[str, Any],
        launch: dict[str, Any],
        runtime: dict[str, Any],
        contract: dict[str, Any],
        started_ns: int,
        completed_ns: int,
    ) -> None:
        _validate_v25_phone_swap(
            evidence,
            locked_swap,
            minimum_available_bytes,
        )
        normalized = copy.deepcopy(evidence)
        for phone in ("op12", "op15"):
            for moment in ("before", "after"):
                normalized["raw_probes"][phone][moment][
                    "system_swap_used_bytes"
                ] = 0
        original(
            normalized,
            launch,
            runtime,
            contract,
            started_ns,
            completed_ns,
        )

    return validate


def _validate_v24_identity_projection(
    *,
    artifacts: dict[str, tuple[dict[str, Any], bytes]],
    inner_lock: dict[str, Any],
    v24: types.ModuleType,
    v24_contract: dict[str, Any],
) -> dict[str, Any]:
    receipt, receipt_raw = artifacts["inner.identity_binding_receipt"]
    bound_root, bound_root_raw = artifacts["inner.bound_root"]
    stage = artifacts["inner.identity_binding_stage_receipt"][0]
    attestation = artifacts["inner.identity_binding_attestation"][0]
    prospective_raw = artifacts["inner.prospective_root"][1]
    preparation_raw = artifacts["inner.preparation"][1]
    lock_raw = artifacts["inner.phase_lock"][1]

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
        "v24.identity.receipt",
    )
    common.exact(
        receipt["schema"],
        "s39-cp0-r1-v24-identity-binding-receipt-v1",
        "v24.identity.receipt.schema",
    )
    common.exact(receipt["phase"], PHASE, "v24.identity.receipt.phase")
    common.exact(
        receipt["phase_id"],
        inner_lock["phase_id"],
        "v24.identity.receipt.phase_id",
    )
    common.exact(
        receipt["device_boot_ids"],
        inner_lock["boot_ids"],
        "v24.identity.receipt.boot_ids",
    )
    common.exact(
        receipt["desktop_identity"],
        {
            "cuda_boot_id": inner_lock["boot_ids"]["cuda"],
            "cuda_host": v24_contract["devices"]["cuda"]["host"],
            "cuda_ssh_target": v24.CUDA_SSH_TARGET,
            "cuda_uuid": v24_contract["devices"]["cuda"]["uuid"],
            "phone_adb_port": v24.PHONE_ADB_PORT,
        },
        "v24.identity.receipt.desktop",
    )
    common.exact(
        receipt["phase_lock_sha256"],
        common.sha256_bytes(lock_raw),
        "v24.identity.receipt.phase_lock",
    )
    common.exact(
        receipt["preparation_sha256"],
        common.sha256_bytes(preparation_raw),
        "v24.identity.receipt.preparation",
    )
    common.exact(
        receipt["prospective_root_sha256"],
        common.sha256_bytes(prospective_raw),
        "v24.identity.receipt.prospective",
    )
    started = common.integer(
        receipt["started_ns"],
        "v24.identity.receipt.started",
        1,
    )
    completed = common.integer(
        receipt["completed_ns"],
        "v24.identity.receipt.completed",
        started,
    )

    output_roles = {
        "cuda_route_launch": "inner.cuda_route_launch",
        "joint_capture_plan": "inner.joint_capture_plan",
        "phone_route_launch": "inner.phone_route_launch",
        "runtime_plan": "inner.runtime_plan",
    }
    outputs = common.exact_keys(
        receipt["outputs"],
        set(output_roles),
        "v24.identity.receipt.outputs",
    )
    artifact_sha256s = {}
    for name, role in output_roles.items():
        row = common.exact_keys(
            outputs[name],
            {"bytes", "path", "sha256", "stat"},
            f"v24.identity.receipt.outputs.{name}",
        )
        raw = artifacts[role][1]
        common.exact(
            row["bytes"],
            len(raw),
            f"v24.identity.receipt.outputs.{name}.bytes",
        )
        common.exact(
            row["sha256"],
            common.sha256_bytes(raw),
            f"v24.identity.receipt.outputs.{name}.sha256",
        )
        artifact_sha256s[name] = common.sha256_bytes(raw)

    mechanism = artifacts["inner.phone_route_launch"][0].get(
        "mechanism_commands"
    )
    common.require(type(mechanism) is dict, "E_V24_PHONE_MECHANISM")
    mechanism_sha256 = common.sha256_bytes(common.canonical_bytes(mechanism))
    for role in ("inner.cuda_route_launch", "inner.joint_capture_plan"):
        common.exact(
            artifacts[role][0].get("mechanism_commands"),
            mechanism,
            f"E_V24_MECHANISM: {role}",
        )
    common.exact(
        receipt["mechanism_commands_sha256"],
        mechanism_sha256,
        "v24.identity.receipt.mechanism",
    )
    common.exact(
        artifacts["inner.orchestration_plan"][0].get(
            "mechanism_commands_sha256"
        ),
        mechanism_sha256,
        "v24.orchestration.mechanism",
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
        "v24.bound_root",
    )
    common.exact(
        bound_root["schema"],
        "s39-cp0-r1-v24-bound-runtime-root-v1",
        "v24.bound_root.schema",
    )
    for key in (
        "desktop_identity",
        "device_boot_ids",
        "phase",
        "phase_id",
        "phase_lock_sha256",
        "preparation_sha256",
        "prospective_root_sha256",
    ):
        common.exact(
            bound_root[key],
            receipt[key],
            f"v24.bound_root.{key}",
        )
    common.exact(
        bound_root["mechanism_commands_sha256"],
        mechanism_sha256,
        "v24.bound_root.mechanism",
    )
    common.exact(
        bound_root["artifacts"],
        outputs,
        "v24.bound_root.artifacts",
    )
    projected_receipt = common.exact_keys(
        bound_root["identity_binding_receipt"],
        {"bytes", "path", "sha256", "stat"},
        "v24.bound_root.identity_binding_receipt",
    )
    common.exact(
        projected_receipt["bytes"],
        len(receipt_raw),
        "v24.bound_root.identity_binding_receipt.bytes",
    )
    common.exact(
        projected_receipt["sha256"],
        common.sha256_bytes(receipt_raw),
        "v24.bound_root.identity_binding_receipt.sha256",
    )

    v24._validate_identity_attestation(
        attestation,
        "v25.identity.attestation",
    )
    common.exact(
        attestation["identity_binding_receipt_sha256"],
        common.sha256_bytes(receipt_raw),
        "v24.identity.attestation.receipt",
    )
    common.exact(
        attestation["bound_root_sha256"],
        common.sha256_bytes(bound_root_raw),
        "v24.identity.attestation.root",
    )
    common.exact_keys(
        stage,
        {
            "argv",
            "completed_ns",
            "returncode",
            "schema",
            "stage",
            "started_ns",
        },
        "v24.identity.stage",
    )
    common.exact(
        stage["schema"],
        "s39-cp0-r1-v24-stage-receipt-v1",
        "v24.identity.stage.schema",
    )
    common.exact(stage["stage"], "identity_binding", "v24.identity.stage.name")
    common.exact(stage["returncode"], 0, "v24.identity.stage.returncode")
    stage_started = common.integer(
        stage["started_ns"],
        "v24.identity.stage.started",
        1,
    )
    stage_completed = common.integer(
        stage["completed_ns"],
        "v24.identity.stage.completed",
        stage_started,
    )
    common.require(
        stage_started < stage_completed
        and stage_started <= started <= completed <= stage_completed,
        "E_V24_IDENTITY_STAGE_INTERVAL",
    )
    return {
        "artifact_sha256s": artifact_sha256s,
        "mechanism_commands_sha256": mechanism_sha256,
        "phase_id": receipt["phase_id"],
    }


def _validate_v24_readiness_chain(
    *,
    artifacts: dict[str, tuple[dict[str, Any], bytes]],
    tokenizer_plan_path: Path,
    token_history_path: Path,
    raw_bundle_root: Path,
    acquisition_path: Path,
    runtime_identity_path: Path,
    v24: types.ModuleType,
    v24_contract: dict[str, Any],
    v24_contract_raw: bytes,
    candidate: dict[str, Any],
    candidate_raw: bytes,
    joint_capture: dict[str, Any],
    locked_swap: dict[str, int],
) -> dict[str, Any]:
    plan = artifacts["inner.runtime_plan"][0]
    plan_raw = artifacts["inner.runtime_plan"][1]
    plan_derived = v24.validate_runtime_plan(
        plan,
        v24_contract,
        v24_contract_raw,
        candidate_raw,
    )
    tokenizer_plan, tokenizer_plan_raw = v24.common.read_canonical(
        tokenizer_plan_path
    )
    v24.validate_tokenizer_plan(
        tokenizer_plan,
        tokenizer_plan_raw,
        v24_contract,
        candidate,
        plan_derived,
    )
    history, history_raw = v24.common.read_canonical(token_history_path)
    v24.validate_token_history(
        history,
        history_raw,
        v24_contract,
        candidate,
        plan_derived,
    )
    root = artifacts["inner.artifact_root"][0]
    root_raw = artifacts["inner.artifact_root"][1]
    root_derived = v24.validate_artifact_root(
        root,
        v24_contract,
        v24_contract_raw,
        candidate,
        candidate_raw,
        plan_raw,
        plan_derived,
        history_raw,
        tokenizer_plan_raw,
    )
    preparation = artifacts["inner.preparation"][0]
    preparation_raw = artifacts["inner.preparation"][1]
    preparation_derived = v24.validate_preparation(
        preparation,
        preparation_raw,
        v24_contract,
        root_raw,
        root_derived["completed_ns"],
        plan_raw,
    )
    inner_lock = artifacts["inner.phase_lock"][0]
    inner_lock_raw = artifacts["inner.phase_lock"][1]
    lock_derived = v24.validate_phase_lock(
        inner_lock,
        inner_lock_raw,
        v24_contract,
        v24_contract_raw,
        candidate_raw,
        root_raw,
        root_derived["completed_ns"],
        preparation_raw,
        preparation_derived,
        plan_raw,
    )
    acquisition, acquisition_raw = v24.common.read_canonical(
        acquisition_path
    )
    raw_predicate_contract, _, _ = v24.raw_predicate_inputs(v24_contract)
    required_roles = set(
        raw_predicate_contract["phase_protocol"]["phase_roles"][PHASE]
    )
    dynamic_roles = (
        required_roles - v24.PRE_ACQUISITION_ROLES
    ) | v24.CAPTURE_RECEIPT_ROLES
    acquisition_started = common.integer(
        acquisition.get("started_ns"),
        "v24.acquisition.started",
        1,
    )
    fresh = artifacts["inner.fresh_readiness"][0]
    fresh_raw = artifacts["inner.fresh_readiness"][1]
    fresh_derived = v24.validate_fresh(
        fresh,
        fresh_raw,
        v24_contract,
        inner_lock_raw,
        lock_derived,
        root_raw,
        root_derived,
        preparation_raw,
        plan_raw,
        plan_derived,
        acquisition_started,
    )
    acquisition_derived = v24.validate_acquisition(
        acquisition,
        acquisition_raw,
        v24_contract,
        v24_contract_raw,
        candidate_raw,
        root_raw,
        preparation_raw,
        inner_lock_raw,
        lock_derived,
        fresh_raw,
        plan_raw,
        dynamic_roles,
    )
    runtime, runtime_raw = v24.common.read_canonical(runtime_identity_path)
    probes = joint_capture["phone_evidence"]["raw_probes"]
    normalized_runtime = copy.deepcopy(runtime)
    for phone in ("op12", "op15"):
        before_swap = common.integer(
            probes[phone]["before"]["system_swap_used_bytes"],
            f"v24.splice.{phone}.before_swap",
        )
        after_swap = common.integer(
            probes[phone]["after"]["system_swap_used_bytes"],
            f"v24.splice.{phone}.after_swap",
        )
        fresh_swap = common.integer(
            fresh_derived["devices"][phone]["system_swap_used_bytes"],
            f"v24.splice.{phone}.fresh_swap",
        )
        runtime_swap = common.integer(
            runtime["phone_after"][phone]["system_swap_used_bytes"],
            f"v24.splice.{phone}.runtime_swap",
        )
        common.require(
            after_swap <= before_swap
            and max(fresh_swap, before_swap, after_swap, runtime_swap)
            <= locked_swap[phone],
            f"E_V24_V25_SWAP_SPLICE: {phone}",
        )
        normalized_runtime["phone_after"][phone][
            "system_swap_used_bytes"
        ] = fresh_swap
    runtime_derived = v24.validate_runtime_identity(
        normalized_runtime,
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
        v24_contract,
    )
    identity = _validate_v24_identity_projection(
        artifacts=artifacts,
        inner_lock=lock_derived,
        v24=v24,
        v24_contract=v24_contract,
    )
    common.exact(
        identity["phase_id"],
        lock_derived["phase_id"],
        "v24.identity.phase_id",
    )
    return {
        "acquisition": acquisition,
        "artifact_root": root,
        "history": history,
        "history_raw": history_raw,
        "identity": identity,
        "plan": plan,
        "runtime": runtime,
        "runtime_derived": runtime_derived,
    }


def validate_v25_quality(
    contract: dict[str, Any],
    candidate_path: Path,
    tokenizer_plan_path: Path,
    token_history_path: Path,
    raw_bundle_root: Path,
    lock: dict[str, Any],
    acquisition_path: str,
    runtime_identity_path: str,
    artifacts: dict[str, tuple[dict[str, Any], bytes]],
    wrapper_receipts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    parent_contract_path = S39 / contract["composition"]["v24"]["contract"]["path"]
    v24 = _load_v24_authority(contract)
    parent, parent_raw, candidate, candidate_raw = v24.validate_inputs(
        parent_contract_path,
        candidate_path,
    )
    history, _ = v24.common.read_canonical(token_history_path)
    raw_contract, raw_contract_raw, raw_parent = v24.raw_predicate_inputs(
        parent
    )
    helpers = v24._verified_helpers(parent)
    frozen_corpus = helpers["v22"].load_frozen_corpus(raw_contract)
    (
        loaded_manifest,
        loaded_manifest_raw,
        rows_by_role,
        artifact_digests,
    ) = helpers["v21"].load_bundle(
        raw_bundle_root,
        RAW_MANIFEST_NAME,
        raw_contract,
        raw_contract_raw,
        candidate_raw,
    )
    joint_capture, _ = _read_acquisition_object_role(
        raw_bundle_root,
        acquisition_path,
        "capture.joint_phone_cuda",
        lock["v24_phase_id"],
    )
    swap_observations = _validate_v25_phone_swap(
        joint_capture["phone_evidence"],
        lock["swap"],
        contract["gates"]["phone_minimum_available_bytes"],
    )
    acquisition_file = (
        raw_bundle_root
        / common.relative_path(
            acquisition_path,
            "fan_in.acquisition_artifact.path",
        )
    ).resolve()
    runtime_file = (
        raw_bundle_root
        / common.relative_path(
            runtime_identity_path,
            "fan_in.runtime_identity_artifact.path",
        )
    ).resolve()
    common.require(
        acquisition_file.is_relative_to(raw_bundle_root.resolve())
        and runtime_file.is_relative_to(raw_bundle_root.resolve()),
        "E_V24_BUNDLE_PATH",
    )
    try:
        chain = _validate_v24_readiness_chain(
            artifacts=artifacts,
            tokenizer_plan_path=tokenizer_plan_path,
            token_history_path=token_history_path,
            raw_bundle_root=raw_bundle_root,
            acquisition_path=acquisition_file,
            runtime_identity_path=runtime_file,
            v24=v24,
            v24_contract=parent,
            v24_contract_raw=parent_raw,
            candidate=candidate,
            candidate_raw=candidate_raw,
            joint_capture=joint_capture,
            locked_swap=lock["swap"],
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        v24.common.EvidenceError,
    ) as error:
        raise common.EvidenceError(
            f"E_V24_READINESS_CHAIN: {error}"
        ) from error
    acquisition = chain["acquisition"]
    cuda_capture, cuda_capture_raw = v24._read_bound_artifact(
        raw_bundle_root,
        acquisition,
        "capture.cuda_monolithic",
    )
    validated_joint_capture, joint_capture_raw = v24._read_bound_artifact(
        raw_bundle_root,
        acquisition,
        "capture.joint_phone_cuda",
    )
    common.exact(
        validated_joint_capture,
        joint_capture,
        "E_JOINT_CAPTURE_REOPEN",
    )
    for role, raw in (
        ("cuda_monolithic", cuda_capture_raw),
        ("joint_phone_cuda", joint_capture_raw),
    ):
        remote = wrapper_receipts[role]["remote_result_artifact"]
        common.exact(
            remote["bytes"],
            len(raw),
            f"E_CAPTURE_WRAPPER_BYTES: {role}",
        )
        common.exact(
            remote["sha256"],
            common.sha256_bytes(raw),
            f"E_CAPTURE_WRAPPER_SHA256: {role}",
        )
    v24.validate_bound_execution_bindings(
        cuda_receipt=cuda_capture,
        joint_receipt=joint_capture,
        identity=chain["identity"],
    )
    try:
        v24.validate_cuda_monolithic_receipt(
            cuda_capture,
            raw_bundle_root,
            parent,
            candidate,
            chain["history"],
            chain["history_raw"],
            chain["plan"],
            chain["runtime"],
            rows_by_role,
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        v24.common.EvidenceError,
    ) as error:
        raise common.EvidenceError(
            f"E_V24_CUDA_RECEIPT: {error}"
        ) from error
    projected_executions = v24.validate_path_matched_history(
        rows_by_role,
        history,
    )
    original_phone_probe = v24._validate_phone_raw_probes
    v24._validate_phone_raw_probes = _v25_phone_probe_validator(
        original_phone_probe,
        lock["swap"],
        contract["gates"]["phone_minimum_available_bytes"],
    )
    try:
        v24.validate_joint_phone_cuda_receipt(
            joint_capture,
            raw_bundle_root,
            parent,
            candidate,
            chain["history"],
            chain["history_raw"],
            chain["plan"],
            chain["runtime"],
            chain["artifact_root"],
            rows_by_role,
            common.integer(
                chain["acquisition"]["started_ns"],
                "v24.acquisition.started",
                1,
            ),
            common.sha256_bytes(
                artifacts["inner.orchestration_plan"][1]
            ),
        )
        result = v24.evaluate_v24_model_phase(
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
    finally:
        v24._validate_phone_raw_probes = original_phone_probe
    common.exact(
        result.get("schema"),
        "s39-cp0-r1-raw-predicate-result-v2.4",
        "v24_quality.schema",
    )
    common.exact(
        result.get("status"),
        "MODEL_A_QUALIFICATION_PASS",
        "v24_quality.status",
    )
    derived = result.get("derived", {})
    v24_result = derived.get("v2_4", {})
    common.exact(
        v24_result.get("canonical_corpus_sha256"),
        contract["quality"]["corpus_sha256"],
        "v24_quality.corpus",
    )
    quality = derived.get("model", {}).get("quality", {})
    common.require(
        common.integer(
            quality.get("cuda_correct"),
            "v24_quality.cuda_correct",
        )
        >= contract["quality"]["cuda_minimum_correct_items"],
        "E_CUDA_QUALITY_FLOOR",
    )
    return {
        "derived": {
            "cuda_correct": quality["cuda_correct"],
            "phone_system_swap": swap_observations,
            "preserved_v2_4_predicates": (
                "PATH_HISTORY_ORACLE_QUALITY_PLACEMENT_EXACT"
            ),
            "superseded_v2_4_predicate": (
                "PHONE_SYSTEM_SWAP_ABSOLUTE_ZERO"
            ),
        },
        "phase": result["phase"],
        "phase_closed_ns": result["phase_closed_ns"],
        "phase_id": result["phase_id"],
        "phase_opened_ns": result["phase_opened_ns"],
        "phone_system_swap_policy": "LOCKED_BASELINE_NO_GROWTH",
        "schema": "s39-cp0-r1-raw-predicate-result-v2.5",
        "status": "MODEL_A_QUALIFICATION_PASS_V2_5_SWAP_POLICY",
    }


def _read_raw_object_role(
    root: Path,
    manifest: dict[str, Any],
    role: str,
) -> tuple[dict[str, Any], str]:
    records = [
        value
        for value in manifest["artifacts"]
        if type(value) is dict and value.get("role") == role
    ]
    common.require(len(records) == 1, f"E_RAW_ROLE: {role}")
    record = records[0]
    raw = common.read_regular(root / common.relative_path(record["path"], role))
    common.exact(len(raw), record["bytes"], f"E_RAW_ROLE_BYTES: {role}")
    common.exact(
        common.sha256_bytes(raw),
        record["sha256"],
        f"E_RAW_ROLE_SHA256: {role}",
    )
    value = common.parse_json(raw, role)
    common.require(type(value) is dict, f"E_TYPE: {role}")
    common.require(common.canonical_bytes(value) == raw, f"E_CANONICAL: {role}")
    return value, record["sha256"]


def _read_acquisition_object_role(
    root: Path,
    acquisition_path: str,
    role: str,
    phase_id: str,
) -> tuple[dict[str, Any], str]:
    acquisition_relative = common.relative_path(
        acquisition_path,
        "acquisition.path",
    )
    acquisition, _ = common.read_canonical(root / acquisition_relative)
    common.exact(
        acquisition.get("schema"),
        "s39-cp0-r1-a-only-acquisition-v2.4",
        "acquisition.schema",
    )
    common.exact(acquisition.get("phase"), PHASE, "acquisition.phase")
    common.exact(
        acquisition.get("phase_id"),
        phase_id,
        "acquisition.phase_id",
    )
    common.exact(
        acquisition.get("status"),
        "RAW_CAPTURE_COMPLETE_UNEVALUATED",
        "acquisition.status",
    )
    artifacts = acquisition.get("artifacts")
    common.require(
        type(artifacts) is list and bool(artifacts),
        "E_ACQUISITION_ARTIFACTS",
    )
    matches = []
    previous = None
    paths = set()
    for index, record in enumerate(artifacts):
        field = f"acquisition.artifacts[{index}]"
        common.exact_keys(
            record,
            {"bytes", "path", "role", "sha256"},
            field,
        )
        record_role = common.text(record["role"], f"{field}.role", 128)
        if previous is not None:
            common.require(previous < record_role, "E_ACQUISITION_ORDER")
        previous = record_role
        relative = common.relative_path(record["path"], f"{field}.path")
        common.require(relative not in paths, "E_ACQUISITION_PATH_REUSE")
        paths.add(relative)
        common.integer(record["bytes"], f"{field}.bytes", 1)
        common.digest(record["sha256"], f"{field}.sha256")
        if record_role == role:
            matches.append(record)
    common.require(len(matches) == 1, f"E_ACQUISITION_ROLE: {role}")
    record = matches[0]
    raw = common.read_regular(
        root / common.relative_path(record["path"], f"{role}.path")
    )
    common.exact(len(raw), record["bytes"], f"{role}.bytes")
    common.exact(
        common.sha256_bytes(raw),
        record["sha256"],
        f"{role}.sha256",
    )
    value = common.parse_json(raw, role)
    common.require(type(value) is dict, f"E_TYPE: {role}")
    common.require(common.canonical_bytes(value) == raw, f"E_CANONICAL: {role}")
    return value, record["sha256"]


def _read_raw_role(
    root: Path,
    manifest: dict[str, Any],
    role: str,
) -> tuple[dict[str, Any], str]:
    records = [
        value
        for value in manifest["artifacts"]
        if type(value) is dict and value.get("role") == role
    ]
    common.require(len(records) == 1, f"E_RAW_ROLE: {role}")
    record = records[0]
    raw = common.read_regular(root / common.relative_path(record["path"], role))
    common.exact(len(raw), record["bytes"], f"E_RAW_ROLE_BYTES: {role}")
    common.exact(
        common.sha256_bytes(raw),
        record["sha256"],
        f"E_RAW_ROLE_SHA256: {role}",
    )
    lines = raw.splitlines(keepends=True)
    common.require(len(lines) == 1, f"E_RAW_ROLE_ROWS: {role}")
    row = common.parse_json(lines[0], role)
    common.require(type(row) is dict, f"E_TYPE: {role}")
    common.require(common.canonical_bytes(row) == lines[0], f"E_CANONICAL: {role}")
    for key, expected in (
        ("acquisition_id", manifest["phase_id"]),
        ("phase", manifest["phase"]),
        ("phase_id", manifest["phase_id"]),
        ("role", role),
    ):
        common.exact(row.get(key), expected, f"E_RAW_WRAPPER: {role}.{key}")
    return row, record["sha256"]


def validate_raw_run_linkage(
    *,
    raw_bundle_root: Path,
    expected_manifest_sha256: str,
    raw_result: dict[str, Any],
    lock: dict[str, Any],
    runtime: dict[str, Any],
    evidence_started_ns: int,
    evidence_completed_ns: int,
) -> None:
    manifest, manifest_raw = common.read_canonical(
        raw_bundle_root / RAW_MANIFEST_NAME
    )
    common.exact(
        common.sha256_bytes(manifest_raw),
        expected_manifest_sha256,
        "E_RAW_MANIFEST_LINK",
    )
    common.exact(manifest.get("phase"), PHASE, "E_RAW_PHASE")
    common.exact(
        manifest.get("phase_id"),
        lock["v24_phase_id"],
        "E_RAW_PHASE_ID",
    )
    opened = common.integer(
        manifest.get("phase_opened_ns"),
        "raw_manifest.phase_opened_ns",
        1,
    )
    acquisition = common.integer(
        manifest.get("acquisition_started_ns"),
        "raw_manifest.acquisition_started_ns",
        opened + 1,
    )
    closed = common.integer(
        manifest.get("phase_closed_ns"),
        "raw_manifest.phase_closed_ns",
        acquisition,
    )
    common.require(
        evidence_started_ns
        <= lock["event_ns"]
        <= runtime["started_ns"]
        <= runtime["completed_ns"]
        <= evidence_completed_ns,
        "E_OUTER_INTERVAL_LINK",
    )
    monolithic = runtime["receipts"]["cuda_monolithic"][
        "remote_execution_interval"
    ]
    joint = runtime["receipts"]["joint_phone_cuda"][
        "remote_execution_interval"
    ]
    fan = runtime["fan_in"]["remote_execution_interval"]
    common.require(
        opened
        < acquisition
        <= monolithic["started_ns"]
        < monolithic["completed_ns"]
        <= joint["started_ns"]
        < joint["completed_ns"]
        <= fan["phase_closed_ns"]
        <= fan["completed_ns"]
        <= runtime["guard_after"]["started_ns"],
        "E_RTX_INTERVAL_LINK",
    )
    common.exact(closed, fan["phase_closed_ns"], "E_RTX_PHASE_CLOSED")
    common.exact(
        raw_result.get("phase_id"),
        lock["v24_phase_id"],
        "E_RAW_RESULT_PHASE",
    )
    common.exact(raw_result.get("phase_opened_ns"), opened, "E_RAW_RESULT_OPENED")
    common.exact(raw_result.get("phase_closed_ns"), closed, "E_RAW_RESULT_CLOSED")

    phase_lock, _ = _read_raw_role(
        raw_bundle_root,
        manifest,
        "phase.lock",
    )
    common.exact(
        phase_lock.get("phase_id"),
        lock["v24_phase_id"],
        "E_RAW_PHASE_LOCK_ID",
    )


def authorize_a_only(
    *,
    contract_path: Path,
    candidate_path: Path,
    tokenizer_plan_path: Path,
    token_history_path: Path,
    evidence_root: Path,
    raw_bundle_root: Path,
) -> dict[str, Any]:
    contract, contract_raw, _, candidate_raw = _validate_contract(
        contract_path,
        candidate_path,
    )
    history, history_raw = common.read_canonical(token_history_path)
    _, tokenizer_plan_raw = common.read_canonical(tokenizer_plan_path)
    common.exact(
        common.sha256_bytes(history_raw),
        contract["quality"]["token_history_sha256"],
        "history.sha256",
    )
    common.exact(
        common.sha256_bytes(tokenizer_plan_raw),
        contract["quality"]["tokenizer_plan_sha256"],
        "tokenizer_plan.sha256",
    )
    manifest, manifest_raw = common.read_canonical(evidence_root / MANIFEST_NAME)
    common.exact_keys(
        manifest,
        {
            "artifacts",
            "candidate_sha256",
            "completed_ns",
            "contract_sha256",
            "phase",
            "phase_id",
            "raw_v25_manifest_sha256",
            "schema",
            "started_ns",
            "token_history_sha256",
        },
        "manifest",
    )
    common.exact(
        manifest["schema"],
        "s39-cp0-r1-evidence-bundle-v2.5",
        "manifest.schema",
    )
    common.exact(manifest["phase"], PHASE, "manifest.phase")
    common.exact(
        manifest["contract_sha256"],
        common.sha256_bytes(contract_raw),
        "manifest.contract",
    )
    common.exact(
        manifest["candidate_sha256"],
        common.sha256_bytes(candidate_raw),
        "manifest.candidate",
    )
    common.exact(
        manifest["token_history_sha256"],
        common.sha256_bytes(history_raw),
        "manifest.history",
    )
    started = common.integer(manifest["started_ns"], "manifest.started", 1)
    completed = common.integer(manifest["completed_ns"], "manifest.completed", started + 1)
    artifacts = _artifact_map(manifest, evidence_root)
    preparation = validate_preparation(
        artifacts["phase.preparation"][0],
        contract,
    )
    common.require(
        started <= preparation["started_ns"],
        "E_MANIFEST_PREPARATION_ORDER",
    )
    discovery = validate_discovery(
        artifacts["phase.discovery"][0],
        contract,
        preparation,
        artifacts["phase.preparation"][1],
    )
    materialization = validate_materialized_prelock(
        artifacts=artifacts,
        contract=contract,
        discovery_raw=artifacts["phase.discovery"][1],
        discovery_value=artifacts["phase.discovery"][0],
    )
    common.require(
        started
        <= common.integer(
            materialization["inventory"]["acquisition_started_ns"],
            "materialization.inventory.acquisition_started_ns",
            1,
        )
        <= preparation["started_ns"],
        "E_MANIFEST_INVENTORY_ORDER",
    )

    support = contract["composition"]["v23"]
    managed = _load_source(
        "s39_v25_managed_runtime_launcher",
        S39 / support["managed_runtime_launcher"]["path"],
        support["managed_runtime_launcher"],
    )
    managed_parser = managed.parse_plan_json
    adapter = _load_v25_program(
        contract,
        "remote_cuda_capture",
        "s39_v25_remote_cuda_capture",
    )
    guard = _load_v25_program(
        contract,
        "remote_phone_guard",
        "s39_v25_remote_phone_guard",
    )
    fan_in = _load_v25_program(
        contract,
        "remote_fan_in_contract",
        "s39_v25_remote_fan_in_contract",
    )

    plans: dict[str, dict[str, Any]] = {}
    plan_digests: dict[str, str] = {
        "history.remote_plan": common.sha256_bytes(
            artifacts["history.remote_plan"][1]
        )
    }
    for role in sorted(MANAGED_PLAN_ROLES):
        raw = artifacts[role][1]
        plan, digest_value = _parse_inline_plan(
            raw,
            role,
            managed_parser,
        )
        plans[role] = plan
        plan_digests[role] = digest_value
    try:
        guard_plan = guard.validate_plan(
            artifacts["plan.remote_phone_guard"][0]
        )
        fan_in_plan = fan_in.validate_plan(
            artifacts["plan.remote_fan_in"][0]
        )
    except (RuntimeError, ValueError) as error:
        raise common.EvidenceError(f"E_REMOTE_PLAN: {error}") from error
    policy_raw = artifacts["plan.remote_phone_guard_policy"][1]
    common.exact(
        guard_plan["local_policy_artifact"]["bytes"],
        len(policy_raw),
        "guard_plan.local_policy.bytes",
    )
    common.exact(
        guard_plan["local_policy_artifact"]["sha256"],
        common.sha256_bytes(policy_raw),
        "guard_plan.local_policy.sha256",
    )
    common.exact(
        artifacts["plan.remote_phone_guard_policy"][0],
        guard_plan["remote_policy"],
        "guard_plan.local_policy.value",
    )
    for role in GUARD_PLAN_ROLES:
        plan_digests[role] = common.sha256_bytes(artifacts[role][1])
    wrapper_plans = validate_wrapper_plans(
        artifacts=artifacts,
        plans=plans,
        plan_digests=plan_digests,
        managed=managed,
        adapter=adapter,
        contract=contract,
        preparation=preparation,
        discovery=discovery,
    )
    validate_fan_in_plan_binding(
        fan_in_plan=fan_in_plan,
        managed_plan=plans["plan.managed.remote_fan_in"],
        managed_plan_raw=artifacts["plan.managed.remote_fan_in"][1],
        managed_plan_sha256=plan_digests[
            "plan.managed.remote_fan_in"
        ],
        adapter=adapter,
        contract=contract,
        preparation=preparation,
        discovery=discovery,
    )

    history_validation = validate_remote_history(
        artifacts["history.remote_plan"][0],
        artifacts["history.remote_receipt"][0],
        contract,
        candidate_raw,
        history_raw,
        tokenizer_plan_raw,
        discovery,
        discovery["completed_ns"],
    )

    for role in (
        "plan.managed.cuda_monolithic",
        "plan.managed.joint_phone_cuda",
    ):
        common.exact(plans[role]["mode"], "remote_cuda", f"E_REMOTE_CUDA_MODE: {role}")
        common.exact(
            plans[role]["ssh"]["ssh_target"],
            contract["topology"]["cuda_ssh_target"],
            f"E_REMOTE_CUDA_TARGET: {role}",
        )
        common.exact(
            plans[role]["ssh"]["gpu_uuid"],
            contract["topology"]["cuda_gpu_uuid"],
            f"E_REMOTE_CUDA_GPU: {role}",
        )
    cuda_ssh = plans["plan.managed.cuda_monolithic"]["ssh"]
    common.exact(
        plans["plan.managed.joint_phone_cuda"]["ssh"],
        cuda_ssh,
        "E_REMOTE_CUDA_SSH_SPLICE",
    )
    common.exact(
        history_validation["plan"]["ssh"],
        cuda_ssh,
        "E_HISTORY_SSH_SPLICE",
    )
    python_components = [
        component
        for component in plans["plan.managed.cuda_monolithic"]["components"]
        if component["path"] == cuda_ssh["remote_python_path"]
    ]
    common.require(len(python_components) == 1, "E_HISTORY_PYTHON_COMPONENT")
    python_component = python_components[0]
    common.exact(
        history_validation["plan"]["support"]["python"],
        {
            "bytes": python_component["bytes"],
            "path": python_component["path"],
            "sha256": python_component["sha256"],
        },
        "E_HISTORY_PYTHON_COMPONENT",
    )
    observed_python = history_validation["observed_support"]["python"]
    common.exact(
        observed_python["stat"],
        {
            key: value
            for key, value in python_component["stat"].items()
            if key != "build_id"
        },
        "E_HISTORY_PYTHON_STAT",
    )

    plan_digests = {
        role: common.sha256_bytes(artifacts[role][1])
        for role in sorted(PHASE_LOCK_BOUND_ROLES)
    }
    lock = validate_phase_lock(
        artifacts["phase.lock"][0],
        contract,
        contract_raw,
        candidate_raw,
        history_raw,
        artifacts["phase.discovery"][1],
        discovery,
        plan_digests,
    )
    common.require(
        history_validation["completed_ns"] <= lock["event_ns"],
        "E_LOCK_BEFORE_HISTORY_VALIDATION",
    )
    common.require(
        materialization["completed_ns"] <= lock["event_ns"],
        "E_LOCK_BEFORE_MATERIALIZATION",
    )
    common.exact(
        materialization["inventory"]["v24_phase_id"],
        lock["v24_phase_id"],
        "materialization.inventory.v24_phase_id",
    )
    common.exact(manifest["phase_id"], lock["phase_id"], "manifest.phase_id")
    common.require(started <= lock["event_ns"], "E_MANIFEST_LOCK_ORDER")
    for role, wrapper in wrapper_plans.items():
        common.exact(
            wrapper["phase_id"],
            lock["phase_id"],
            f"wrapper_plan.{role}.outer_phase",
        )
        common.exact(
            wrapper["v24_phase_id"],
            lock["v24_phase_id"],
            f"wrapper_plan.{role}.inner_phase",
        )
    runtime = validate_runtime(
        artifacts=artifacts,
        plans=plans,
        wrapper_plans=wrapper_plans,
        guard_plan=guard_plan,
        fan_in_plan=fan_in_plan,
        adapter=adapter,
        guard=guard,
        fan_in=fan_in,
        contract=contract,
        discovery=discovery,
        lock=lock,
        raw_bundle_root=raw_bundle_root,
    )
    common.require(runtime["completed_ns"] <= completed, "E_MANIFEST_RUNTIME_ORDER")

    raw_manifest = common.read_regular(raw_bundle_root / RAW_MANIFEST_NAME)
    common.exact(
        manifest["raw_v25_manifest_sha256"],
        common.sha256_bytes(raw_manifest),
        "manifest.raw_v25_manifest",
    )
    raw_result = validate_v25_quality(
        contract,
        candidate_path,
        tokenizer_plan_path,
        token_history_path,
        raw_bundle_root,
        lock,
        runtime["fan_in"]["acquisition_artifact"]["path"],
        runtime["fan_in"]["runtime_identity_artifact"]["path"],
        artifacts,
        runtime["receipts"],
    )
    validate_raw_run_linkage(
        raw_bundle_root=raw_bundle_root,
        expected_manifest_sha256=manifest["raw_v25_manifest_sha256"],
        raw_result=raw_result,
        lock=lock,
        runtime=runtime,
        evidence_started_ns=started,
        evidence_completed_ns=completed,
    )
    return {
        "candidate_sha256": common.sha256_bytes(candidate_raw),
        "contract_sha256": common.sha256_bytes(contract_raw),
        "evidence_manifest_sha256": common.sha256_bytes(manifest_raw),
        "phase": PHASE,
        "phase_id": lock["phase_id"],
        "schema": "s39-cp0-r1-evidence-result-v2.5",
        "phone_system_swap_policy": "LOCKED_BASELINE_NO_GROWTH",
        "status": "MODEL_A_QUALIFICATION_PASS_V2_5_SWAP_POLICY",
        "v2_5_predicate_result_sha256": common.sha256_bytes(
            common.canonical_bytes(raw_result)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tokenizer-plan", type=Path, required=True)
    parser.add_argument("--token-history", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--raw-bundle-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = authorize_a_only(
            contract_path=args.contract,
            candidate_path=args.candidate,
            tokenizer_plan_path=args.tokenizer_plan,
            token_history_path=args.token_history,
            evidence_root=args.evidence_root,
            raw_bundle_root=args.raw_bundle_root,
        )
        sys.stdout.buffer.write(common.canonical_bytes(result))
        return 0
    except (common.EvidenceError, OSError, RuntimeError, ValueError) as error:
        print(f"CP0_R1_V2_5_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
