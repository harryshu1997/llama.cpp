#!/usr/bin/env python3
"""Bind post-reboot identities into immutable V2.4 prospective plans."""

from __future__ import annotations

import argparse
import copy
import ipaddress
import json
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any, Callable


HERE = Path(__file__).resolve().parent


def _load_source(name: str, path: Path):
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"E_SOURCE_REGULAR: {path}")
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
    )
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise RuntimeError(f"E_SOURCE_CHANGED: {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(bytes(raw), str(path), "exec"), module.__dict__)
    return module


common = _load_source(
    "s39_v24_production_common",
    HERE / "production_common_v1.py",
)


PROSPECTIVE_SCHEMA = "s39-cp0-r1-v24-prospective-runtime-root-v1"
BOUND_SCHEMA = "s39-cp0-r1-v24-bound-runtime-root-v1"
RECEIPT_SCHEMA = "s39-cp0-r1-v24-identity-binding-receipt-v1"
ATTESTATION_SCHEMA = "s39-cp0-r1-v24-identity-binding-attestation-v1"
ARTIFACT_SCHEMAS = {
    "cuda_route_launch": "s39-cp0-r1-v24-cuda-route-launch-v1",
    "joint_capture_plan": "s39-cp0-r1-v24-joint-capture-plan-v1",
    "phone_route_launch": "s39-cp0-r1-v24-phone-route-launch-v1",
    "runtime_plan": "s39-cp0-r1-runtime-bundle-plan-v2.4",
}


def _record(path: Path, raw: bytes) -> dict[str, Any]:
    return {
        "bytes": len(raw),
        "path": str(path),
        "sha256": common.sha256_bytes(raw),
    }


def _validate_record(
    record: Any,
    path: Path,
    raw: bytes,
    field: str,
) -> None:
    common.exact_keys(record, {"bytes", "path", "sha256"}, field)
    common.exact(record["path"], str(path), f"{field}.path")
    common.exact(record["bytes"], len(raw), f"{field}.bytes")
    common.exact(record["sha256"], common.sha256_bytes(raw), f"{field}.sha256")


def _reopen(path: Path, raw: bytes, field: str) -> None:
    common.exact(
        common.read_regular(path, field),
        raw,
        f"{field}.reopen",
    )


def _validate_external_record(record: Any, field: str) -> bytes:
    common.exact_keys(record, {"bytes", "path", "sha256"}, field)
    path = Path(record["path"])
    raw = common.read_regular(path, field)
    _validate_record(record, path, raw, field)
    return raw


def _reject_sentinels(value: Any, field: str) -> None:
    if type(value) is str:
        sentinels = {
            *common.UNBOUND_BOOT_IDS.values(),
            *(
                item
                for endpoint in common.UNBOUND_PHONE_NETWORK.values()
                for item in endpoint.values()
            ),
        }
        common.require(
            not any(sentinel in value for sentinel in sentinels),
            f"E_SENTINEL_LEAKAGE: {field}",
        )
    elif type(value) is list:
        for index, item in enumerate(value):
            _reject_sentinels(item, f"{field}[{index}]")
    elif type(value) is dict:
        for key, item in value.items():
            _reject_sentinels(item, f"{field}.{key}")


def _replace_exact_strings(
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
        return [_replace_exact_strings(item, replacements) for item in value]
    if type(value) is dict:
        return {
            key: _replace_exact_strings(item, replacements)
            for key, item in value.items()
        }
    return value


def _rebind_structured_argv(
    value: Any,
    replacements: dict[str, str],
    field: str = "phone",
) -> None:
    if type(value) is dict:
        for key, item in value.items():
            _rebind_structured_argv(
                item,
                replacements,
                f"{field}.{key}",
            )
        return
    if type(value) is not list:
        return
    for index, item in enumerate(value):
        _rebind_structured_argv(
            item,
            replacements,
            f"{field}[{index}]",
        )
    if not value or not all(type(item) is str for item in value):
        return
    for option in ("--plan-json", "--expected-argv-json"):
        count = value.count(option)
        common.require(count in (0, 1), f"E_STRUCTURED_ARGV_OPTION: {field}.{option}")
        if count == 0:
            continue
        index = value.index(option) + 1
        common.require(index < len(value), f"E_STRUCTURED_ARGV_OPTION: {field}.{option}")
        try:
            parsed = json.loads(value[index])
        except json.JSONDecodeError as error:
            raise common.ProductionError(
                f"E_STRUCTURED_ARGV_JSON: {field}.{option}"
            ) from error
        rebound = _replace_exact_strings(parsed, replacements)
        _rebind_structured_argv(
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


def _check_inline_plan_digests(
    value: Any,
    *,
    rewrite: bool,
    field: str = "phone",
) -> None:
    if type(value) is dict:
        for key, item in value.items():
            _check_inline_plan_digests(
                item,
                rewrite=rewrite,
                field=f"{field}.{key}",
            )
        return
    if type(value) is not list:
        return
    for index, item in enumerate(value):
        _check_inline_plan_digests(
            item,
            rewrite=rewrite,
            field=f"{field}[{index}]",
        )
    if not value or not all(type(item) is str for item in value):
        return
    plan_count = value.count("--plan-json")
    digest_count = value.count("--plan-sha256")
    common.require(plan_count == digest_count, f"E_INLINE_PLAN_OPTIONS: {field}")
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
        raise common.ProductionError(
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


def _derive_phone_mechanism(plan: dict[str, Any]) -> dict[str, list[list[str]]]:
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


def _replace_option(
    argv: list[str],
    option: str,
    old_value: str,
    new_value: str,
    field: str,
) -> None:
    common.require(
        argv.count(option) == 1 and argv.index(option) + 1 < len(argv),
        f"E_JOINT_OPTION: {field}.{option}",
    )
    index = argv.index(option) + 1
    common.exact(argv[index], old_value, f"{field}.{option}.prospective")
    argv[index] = new_value


def _replace_joint_launch(
    command: dict[str, Any],
    old_path: Path,
    new_path: Path,
    new_raw: bytes,
    field: str,
) -> None:
    argv = command["argv_template"]
    index = common.integer(
        command["launch_plan_argv_index"],
        f"{field}.launch_plan_argv_index",
        1,
    )
    common.require(index < len(argv), f"E_JOINT_ARGV_INDEX: {field}")
    common.exact(argv[index], str(old_path), f"{field}.prospective_path")
    argv[index] = str(new_path)
    records = command["executed_files"]
    matches = [
        value for value in records
        if value.get("argv_index") == index
    ]
    common.require(len(matches) == 1, f"E_JOINT_EXECUTED_FILE: {field}")
    replacement = {
        "argv_index": index,
        **_record(new_path, new_raw),
    }
    records[records.index(matches[0])] = replacement
    records.sort(key=lambda value: value["argv_index"])
    command["launch_plan_sha256"] = common.sha256_bytes(new_raw)


def bind(
    *,
    prospective_root_path: Path,
    contract_path: Path,
    preparation_path: Path,
    phase_lock_path: Path,
    prospective_paths: dict[str, Path],
    bound_paths: dict[str, Path],
    receipt_output: Path,
    bound_root_output: Path,
    clock_ns: Callable[[], int] = common.monotonic_ns,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_paths = {
        **bound_paths,
        "identity_binding_receipt": receipt_output,
        "bound_root": bound_root_output,
    }
    common.exact(set(prospective_paths), set(ARTIFACT_SCHEMAS), "prospective.paths")
    common.exact(set(bound_paths), set(ARTIFACT_SCHEMAS), "bound.paths")
    common.require(
        prospective_root_path.is_absolute()
        and contract_path.is_absolute()
        and preparation_path.is_absolute()
        and phase_lock_path.is_absolute()
        and all(path.is_absolute() for path in prospective_paths.values())
        and all(path.is_absolute() and not path.exists() for path in output_paths.values()),
        "E_BOUND_OUTPUT_EXISTS",
    )
    common.require(
        len(set(output_paths.values())) == len(output_paths),
        "E_BOUND_OUTPUT_REUSE",
    )
    started_ns = clock_ns()
    prospective_root, prospective_root_raw = common.read_canonical(
        prospective_root_path,
        "prospective_root",
    )
    contract, contract_raw = common.read_canonical(contract_path, "contract")
    preparation, preparation_raw = common.read_canonical(
        preparation_path,
        "preparation",
    )
    phase_lock, phase_lock_raw = common.read_canonical(
        phase_lock_path,
        "phase_lock",
    )
    common.exact(
        prospective_root.get("schema"),
        PROSPECTIVE_SCHEMA,
        "prospective_root.schema",
    )
    common.exact_keys(
        prospective_root,
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
    common.exact(prospective_root["acquisition_ready"], False, "prospective_root.ready")
    common.exact(prospective_root["phase"], common.PHASE, "prospective_root.phase")
    common.exact(prospective_root["model_id"], common.MODEL_ID, "prospective_root.model")
    common.exact(
        prospective_root["status"],
        "POST_REBOOT_IDENTITY_BINDING_REQUIRED",
        "prospective_root.status",
    )
    common.exact(
        prospective_root["desktop_control"],
        {
            "cuda_ssh_target": common.CUDA_SSH_TARGET,
            "phone_adb_port": common.PHONE_ADB_PORT,
        },
        "prospective_root.desktop_control",
    )
    common.exact(
        contract.get("schema"),
        "s39-cp0-r1-evidence-contract-v2.4",
        "contract.schema",
    )
    common.exact(
        preparation.get("schema"),
        "s39-cp0-r1-reboot-preparation-v2.4",
        "preparation.schema",
    )
    common.exact(
        phase_lock.get("schema"),
        "s39-cp0-r1-phase-lock-v2.4",
        "phase_lock.schema",
    )
    common.exact(phase_lock["phase"], common.PHASE, "phase_lock.phase")
    common.exact(
        phase_lock["preparation_sha256"],
        common.sha256_bytes(preparation_raw),
        "phase_lock.preparation",
    )
    common.exact(
        phase_lock["contract_sha256"],
        common.sha256_bytes(contract_raw),
        "phase_lock.contract",
    )
    _validate_record(
        prospective_root["contract"],
        contract_path,
        contract_raw,
        "prospective_root.contract",
    )
    candidate_raw = _validate_external_record(
        prospective_root["candidate"],
        "prospective_root.candidate",
    )
    history_raw = _validate_external_record(
        prospective_root["token_history"],
        "prospective_root.token_history",
    )
    tokenizer_raw = _validate_external_record(
        prospective_root["tokenizer_plan"],
        "prospective_root.tokenizer_plan",
    )
    mono_raw = _validate_external_record(
        prospective_root["cuda_monolithic_launch"],
        "prospective_root.cuda_monolithic_launch",
    )
    _validate_external_record(prospective_root["spec"], "prospective_root.spec")
    common.require(
        preparation["completed_ns"] < phase_lock["event_ns"] <= started_ns,
        "E_IDENTITY_BINDING_ORDER",
    )
    values = {}
    raws = {}
    for name, schema in ARTIFACT_SCHEMAS.items():
        value, raw = common.read_canonical(prospective_paths[name], name)
        common.exact(value.get("schema"), schema, f"{name}.schema")
        _validate_record(
            prospective_root["artifacts"][name],
            prospective_paths[name],
            raw,
            f"prospective_root.artifacts.{name}",
        )
        values[name] = value
        raws[name] = raw
    runtime = values["runtime_plan"]
    common.exact(
        runtime["contract_sha256"],
        common.sha256_bytes(contract_raw),
        "prospective.runtime.contract",
    )
    common.exact(
        runtime["candidate_sha256"],
        common.sha256_bytes(candidate_raw),
        "prospective.runtime.candidate",
    )
    common.exact(
        runtime["token_history"]["artifact_path"],
        prospective_root["token_history"]["path"],
        "prospective.runtime.history.path",
    )
    common.exact(
        values["joint_capture_plan"]["history"]["sha256"],
        common.sha256_bytes(history_raw),
        "prospective.joint.history",
    )
    common.exact(
        runtime["token_history"]["tokenizer_plan_sha256"],
        common.sha256_bytes(tokenizer_raw),
        "prospective.runtime.tokenizer",
    )
    common.exact(
        runtime["cuda_monolithic_launch"],
        common.parse_json(mono_raw, "prospective.cuda_monolithic_launch"),
        "prospective.runtime.cuda_monolithic_launch",
    )
    common.exact(
        prospective_root["model_sha256"],
        runtime["token_history"]["model_sha256"],
        "prospective_root.model_sha256",
    )
    common.exact(
        phase_lock["runtime_bundle_plan_sha256"],
        common.sha256_bytes(raws["runtime_plan"]),
        "phase_lock.runtime_plan",
    )
    common.exact(
        prospective_root["identity_placeholders"],
        common.UNBOUND_BOOT_IDS,
        "prospective_root.identity_placeholders",
    )
    common.exact(
        prospective_root["network_placeholders"],
        common.UNBOUND_PHONE_NETWORK,
        "prospective_root.network_placeholders",
    )
    prospective_phones = values["phone_route_launch"]["phones"]
    for phone in ("op12", "op15"):
        common.exact(
            prospective_phones[phone]["boot_id"],
            common.UNBOUND_BOOT_IDS[phone],
            f"E_PROSPECTIVE_IDENTITY_NOT_UNBOUND: {phone}",
        )
        for key, expected in common.UNBOUND_PHONE_NETWORK[phone].items():
            common.exact(
                prospective_phones[phone][key],
                expected,
                f"E_PROSPECTIVE_NETWORK_NOT_UNBOUND: {phone}.{key}",
            )
        peer = "op15" if phone == "op12" else "op12"
        common.exact(
            prospective_phones[phone]["direct_peer_ipv4"],
            common.UNBOUND_PHONE_NETWORK[peer]["local_ipv4"],
            f"E_PROSPECTIVE_NETWORK_NOT_UNBOUND: {phone}.direct_peer_ipv4",
        )
    boot_ids = phase_lock["device_boot_ids"]
    common.exact(set(boot_ids), {"cuda", "op12", "op15"}, "phase_lock.device_boot_ids")
    common.require(len(set(boot_ids.values())) == 3, "E_BOOT_ID_REUSE")
    before_boot_ids = common.exact_keys(
        preparation["before_boot_ids"],
        {"op12", "op15"},
        "preparation.before_boot_ids",
    )
    for endpoint, boot_id in boot_ids.items():
        common.require(
            common.UUID_RE.fullmatch(boot_id) is not None
            and boot_id != common.UNBOUND_BOOT_IDS[endpoint],
            f"E_UNBOUND_BOOT_ID: {endpoint}",
        )
        preparation_key = "host_boot_id" if endpoint == "cuda" else "boot_id"
        common.exact(
            preparation["devices"][endpoint][preparation_key],
            boot_id,
            f"E_PREPARATION_BOOT_ID: {endpoint}",
        )
        if endpoint != "cuda":
            common.require(
                common.UUID_RE.fullmatch(before_boot_ids[endpoint]) is not None
                and before_boot_ids[endpoint] != boot_id,
                f"E_REBOOT_IDENTITY_REUSE: {endpoint}",
            )
    prospective_mechanism = _derive_phone_mechanism(
        values["phone_route_launch"]
    )
    _check_inline_plan_digests(
        values["phone_route_launch"],
        rewrite=False,
    )
    common.exact(
        values["phone_route_launch"]["mechanism_commands"]["op12"],
        prospective_mechanism["op12"],
        "prospective.phone.mechanism.op12",
    )
    common.exact(
        values["phone_route_launch"]["mechanism_commands"]["op15"],
        prospective_mechanism["op15"],
        "prospective.phone.mechanism.op15",
    )
    common.exact(
        values["cuda_route_launch"]["mechanism_commands"],
        values["phone_route_launch"]["mechanism_commands"],
        "prospective.route.mechanism",
    )
    prospective_commands_raw = common.canonical_bytes({
        "probes": values["phone_route_launch"]["probes"],
        "processes": values["phone_route_launch"]["processes"],
    })
    for sentinel in {
        item
        for endpoint in common.UNBOUND_PHONE_NETWORK.values()
        for item in endpoint.values()
    }:
        common.require(
            sentinel.encode("ascii") in prospective_commands_raw,
            f"E_PROSPECTIVE_COMMAND_NETWORK_NOT_UNBOUND: {sentinel}",
        )
    network = {}
    for endpoint in ("op12", "op15"):
        interface = preparation["devices"][endpoint]["interface"]
        local_ipv4 = preparation["devices"][endpoint]["local_ipv4"]
        common.exact(interface, "wlan0", f"E_PHONE_INTERFACE: {endpoint}")
        try:
            address = ipaddress.IPv4Address(local_ipv4)
        except ipaddress.AddressValueError as error:
            raise common.ProductionError(
                f"E_PHONE_IPV4: {endpoint}"
            ) from error
        common.require(
            not (
                address.is_unspecified
                or address.is_loopback
                or address.is_multicast
            ),
            f"E_PHONE_IPV4: {endpoint}",
        )
        network[endpoint] = {
            "interface": interface,
            "local_ipv4": local_ipv4,
        }
    common.require(
        network["op12"]["local_ipv4"] != network["op15"]["local_ipv4"],
        "E_PHONE_IPV4_REUSE",
    )
    replacements = {
        common.UNBOUND_PHONE_NETWORK["op12"]["interface"]:
            network["op12"]["interface"],
        common.UNBOUND_PHONE_NETWORK["op12"]["local_ipv4"]:
            network["op12"]["local_ipv4"],
        common.UNBOUND_PHONE_NETWORK["op15"]["local_ipv4"]:
            network["op15"]["local_ipv4"],
    }
    phone = _replace_exact_strings(
        copy.deepcopy(values["phone_route_launch"]),
        replacements,
    )
    _rebind_structured_argv(phone, replacements)
    _check_inline_plan_digests(phone, rewrite=True)
    for endpoint in ("op12", "op15"):
        phone["phones"][endpoint]["boot_id"] = boot_ids[endpoint]
    realized_mechanism = _derive_phone_mechanism(phone)
    common.exact(
        phone["mechanism_commands"]["op12"],
        realized_mechanism["op12"],
        "bound.phone.mechanism.op12",
    )
    common.exact(
        phone["mechanism_commands"]["op15"],
        realized_mechanism["op15"],
        "bound.phone.mechanism.op15",
    )
    mechanism_sha256 = common.sha256_bytes(
        common.canonical_bytes(realized_mechanism)
    )
    phone_raw = common.canonical_bytes(phone)
    cuda = copy.deepcopy(values["cuda_route_launch"])
    cuda["mechanism_commands"] = copy.deepcopy(realized_mechanism)
    cuda_raw = common.canonical_bytes(cuda)
    runtime_raw = raws["runtime_plan"]
    joint = copy.deepcopy(values["joint_capture_plan"])
    common.exact(
        joint["mechanism_commands"],
        prospective_mechanism,
        "prospective.joint.mechanism",
    )
    prospective_mechanism_sha256 = common.sha256_bytes(
        common.canonical_bytes(prospective_mechanism)
    )
    joint["mechanism_commands"] = copy.deepcopy(realized_mechanism)
    for endpoint in ("cuda", "phone"):
        _replace_option(
            joint["commands"][endpoint]["argv_template"],
            "--mechanism-commands-sha256",
            prospective_mechanism_sha256,
            mechanism_sha256,
            f"joint.{endpoint}",
        )
    _replace_joint_launch(
        joint["commands"]["cuda"],
        prospective_paths["cuda_route_launch"],
        bound_paths["cuda_route_launch"],
        cuda_raw,
        "joint.cuda",
    )
    _replace_joint_launch(
        joint["commands"]["phone"],
        prospective_paths["phone_route_launch"],
        bound_paths["phone_route_launch"],
        phone_raw,
        "joint.phone",
    )
    joint_raw = common.canonical_bytes(joint)
    bound_raws = {
        "cuda_route_launch": cuda_raw,
        "joint_capture_plan": joint_raw,
        "phone_route_launch": phone_raw,
        "runtime_plan": runtime_raw,
    }
    for name, raw in bound_raws.items():
        _reject_sentinels(common.parse_json(raw, f"bound.{name}"), f"bound.{name}")
    completed_ns = clock_ns()
    common.require(started_ns <= completed_ns, "E_IDENTITY_BINDING_INTERVAL")
    desktop_identity = {
        "cuda_boot_id": boot_ids["cuda"],
        "cuda_host": contract["devices"]["cuda"]["host"],
        "cuda_ssh_target": common.CUDA_SSH_TARGET,
        "cuda_uuid": contract["devices"]["cuda"]["uuid"],
        "phone_adb_port": common.PHONE_ADB_PORT,
    }
    receipt = {
        "completed_ns": completed_ns,
        "desktop_identity": desktop_identity,
        "device_boot_ids": boot_ids,
        "mechanism_commands_sha256": mechanism_sha256,
        "outputs": {
            name: _record(bound_paths[name], raw)
            for name, raw in sorted(bound_raws.items())
        },
        "phase": common.PHASE,
        "phase_id": phase_lock["phase_id"],
        "phase_lock_sha256": common.sha256_bytes(phase_lock_raw),
        "preparation_sha256": common.sha256_bytes(preparation_raw),
        "prospective_root_sha256": common.sha256_bytes(prospective_root_raw),
        "schema": RECEIPT_SCHEMA,
        "started_ns": started_ns,
    }
    receipt_raw = common.canonical_bytes(receipt)
    bound_root = {
        "artifacts": receipt["outputs"],
        "desktop_identity": desktop_identity,
        "device_boot_ids": boot_ids,
        "identity_binding_receipt": _record(receipt_output, receipt_raw),
        "mechanism_commands_sha256": mechanism_sha256,
        "phase": common.PHASE,
        "phase_id": phase_lock["phase_id"],
        "phase_lock_sha256": common.sha256_bytes(phase_lock_raw),
        "preparation_sha256": common.sha256_bytes(preparation_raw),
        "prospective_root_sha256": common.sha256_bytes(prospective_root_raw),
        "schema": BOUND_SCHEMA,
    }
    for name, raw in bound_raws.items():
        common.write_raw_new(bound_paths[name], raw)
        _reopen(bound_paths[name], raw, f"published.{name}")
    common.write_raw_new(receipt_output, receipt_raw)
    _reopen(receipt_output, receipt_raw, "published.identity_binding_receipt")
    common.write_new(bound_root_output, bound_root)
    _reopen(
        bound_root_output,
        common.canonical_bytes(bound_root),
        "published.bound_root",
    )
    return receipt, bound_root


def _attestation(
    receipt: dict[str, Any],
    bound_root: dict[str, Any],
) -> dict[str, Any]:
    return {
        "bound_root_sha256": common.sha256_bytes(
            common.canonical_bytes(bound_root)
        ),
        "identity_binding_receipt_sha256": common.sha256_bytes(
            common.canonical_bytes(receipt)
        ),
        "schema": ATTESTATION_SCHEMA,
        "status": "POST_REBOOT_IDENTITY_BINDING_PASS",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prospective-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--phase-lock", type=Path, required=True)
    for name in ARTIFACT_SCHEMAS:
        option = name.replace("_", "-")
        parser.add_argument(f"--prospective-{option}", type=Path, required=True)
        parser.add_argument(f"--bound-{option}", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--bound-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt, bound_root = bind(
            prospective_root_path=args.prospective_root,
            contract_path=args.contract,
            preparation_path=args.preparation,
            phase_lock_path=args.phase_lock,
            prospective_paths={
                name: getattr(args, f"prospective_{name}").resolve()
                for name in ARTIFACT_SCHEMAS
            },
            bound_paths={
                name: getattr(args, f"bound_{name}").resolve()
                for name in ARTIFACT_SCHEMAS
            },
            receipt_output=args.receipt.resolve(),
            bound_root_output=args.bound_root.resolve(),
        )
        sys.stdout.buffer.write(
            common.canonical_bytes(_attestation(receipt, bound_root))
        )
        return 0
    except (OSError, ValueError, common.ProductionError) as error:
        print(f"V24_IDENTITY_BINDING_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
