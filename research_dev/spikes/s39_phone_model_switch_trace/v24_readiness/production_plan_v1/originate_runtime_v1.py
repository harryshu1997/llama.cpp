#!/usr/bin/env python3
"""Originate immutable prospective V2.4 route and runtime plans offline."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any


HERE = Path(__file__).resolve().parent
V24 = HERE.parent
PRODUCERS = V24 / "producers_v1"


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
authority = _load_source(
    "s39_v24_authority",
    V24 / "cp0_r1_evidence_v24.py",
)


SPEC_SCHEMA = "s39-cp0-r1-v24-prospective-runtime-spec-v1"
ROOT_SCHEMA = "s39-cp0-r1-v24-prospective-runtime-root-v1"
REPORT_SCHEMA = "s39-cp0-r1-v24-runtime-origin-dry-run-v1"
ARTIFACT_SCHEMAS = {
    "cuda_route_launch": "s39-cp0-r1-v24-cuda-route-launch-v1",
    "joint_capture_plan": "s39-cp0-r1-v24-joint-capture-plan-v1",
    "phone_route_launch": "s39-cp0-r1-v24-phone-route-launch-v1",
    "runtime_plan": "s39-cp0-r1-runtime-bundle-plan-v2.4",
}
CUDA_DERIVED = {
    "history_path",
    "history_sha256",
    "model_id",
    "model_sha256",
    "quality_corpus_content_sha256",
    "schema",
}
CUDA_KEYS = {
    "codec",
    "expected_capabilities",
    "expected_file_type",
    "expected_max_streams",
    "expected_n_batch",
    "expected_n_ctx_seq",
    "expected_n_embd",
    "expected_n_layer",
    "expected_n_ubatch",
    "history_path",
    "history_sha256",
    "host",
    "io_timeout_ms",
    "mechanism_commands",
    "model_artifact",
    "model_id",
    "model_sha256",
    "nvidia_smi",
    "port",
    "quality_corpus_content_sha256",
    "route_epoch",
    "schema",
    "worker",
}
PHONE_DERIVED = CUDA_DERIVED
PHONE_KEYS = {
    "codec",
    "expected_file_type",
    "expected_max_streams",
    "expected_n_batch",
    "expected_n_ctx_seq",
    "expected_n_embd",
    "expected_n_layer",
    "expected_n_ubatch",
    "history_path",
    "history_sha256",
    "mechanism_commands",
    "model_id",
    "model_sha256",
    "phones",
    "probes",
    "processes",
    "quality_corpus_content_sha256",
    "relay_host",
    "relay_port",
    "route_epoch",
    "schema",
}
PHONE_KEYS_PER_DEVICE = {
    "boot_id",
    "device",
    "direct_peer_ipv4",
    "executed_layers",
    "expected_worker_executable_path",
    "expected_worker_executable_sha256",
    "interface",
    "loaded_shard_path",
    "loaded_shard_sha256",
    "local_ipv4",
    "model",
    "product",
    "serial",
    "stored_layers",
}


def _artifact(path: Path, raw: bytes) -> dict[str, Any]:
    return {
        "bytes": len(raw),
        "path": str(path),
        "sha256": common.sha256_bytes(raw),
    }


def _read_input(path: Path, schema: str, field: str) -> tuple[dict[str, Any], bytes]:
    value, raw = common.read_canonical(path, field)
    common.exact(value.get("schema"), schema, f"{field}.schema")
    return value, raw


def _executed_file(path: Path, argv_index: int) -> dict[str, Any]:
    raw = common.read_regular(path, f"executed_file[{argv_index}]")
    return {
        "argv_index": argv_index,
        **_artifact(path, raw),
    }


def _option(argv: Any, option: str, expected: str, field: str) -> None:
    common.require(
        type(argv) is list
        and argv.count(option) == 1
        and argv.index(option) + 1 < len(argv),
        f"E_OPTION: {field}.{option}",
    )
    common.exact(argv[argv.index(option) + 1], expected, f"{field}.{option}")


def _artifact_pin(value: Any, field: str, *, with_bytes: bool = True) -> None:
    keys = {"path", "sha256", "stat"} | ({"bytes"} if with_bytes else set())
    common.exact_keys(value, keys, field)
    common.require(Path(value["path"]).is_absolute(), f"E_PATH: {field}")
    common.digest(value["sha256"], f"{field}.sha256")
    if with_bytes:
        common.integer(value["bytes"], f"{field}.bytes", 1)
    metadata = common.exact_keys(
        value["stat"],
        {"ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"},
        f"{field}.stat",
    )
    for key, item in metadata.items():
        common.integer(item, f"{field}.stat.{key}")
    common.require(stat.S_ISREG(metadata["mode"]), f"E_STAT_MODE: {field}")
    if with_bytes:
        common.exact(metadata["size"], value["bytes"], f"{field}.stat.size")


def _validate_cuda(
    value: dict[str, Any],
    contract: dict[str, Any],
    model: dict[str, Any],
    history_path: Path,
    history_raw: bytes,
) -> None:
    common.exact_keys(value, CUDA_KEYS, "cuda_route")
    expected = {
        "expected_capabilities": 0x3F,
        "expected_file_type": 15,
        "expected_max_streams": 8,
        "expected_n_batch": 64,
        "expected_n_ctx_seq": 512,
        "expected_n_embd": 5120,
        "expected_n_layer": 40,
        "expected_n_ubatch": 64,
        "history_path": str(history_path),
        "history_sha256": common.sha256_bytes(history_raw),
        "model_id": common.MODEL_ID,
        "model_sha256": model["artifact"]["sha256"],
        "quality_corpus_content_sha256": contract["quality_corpus"]["sha256"],
        "schema": ARTIFACT_SCHEMAS["cuda_route_launch"],
    }
    for key, expected_value in expected.items():
        common.exact(value[key], expected_value, f"cuda_route.{key}")
    common.integer(value["route_epoch"], "cuda_route.route_epoch", 1)
    common.require(
        value["host"] in {"127.0.0.1", "::1"},
        "E_CUDA_ROUTE_HOST",
    )
    port = common.integer(value["port"], "cuda_route.port", 1)
    common.require(port <= 65535, "E_CUDA_ROUTE_PORT")
    geometry = contract["model_geometry"][common.MODEL_ID]
    model_artifact = value["model_artifact"]
    _artifact_pin(model_artifact, "cuda_route.model_artifact")
    common.exact(
        {
            key: model_artifact[key]
            for key in ("bytes", "path", "sha256")
        },
        {
            "bytes": model["artifact"]["bytes"],
            "path": geometry["cuda_model_path"],
            "sha256": model["artifact"]["sha256"],
        },
        "cuda_route.model_artifact.identity",
    )
    worker = common.exact_keys(
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
        "cuda_route.worker",
    )
    _artifact_pin(worker["executable"], "cuda_route.worker.executable")
    _artifact_pin(
        worker["runtime_executable"],
        "cuda_route.worker.runtime_executable",
    )
    common.exact(
        worker["argv"][0],
        worker["executable"]["path"],
        "cuda_route.worker.argv0",
    )
    common.require(
        worker["runtime_executable"]["path"] in worker["argv"],
        "E_CUDA_RUNTIME_ARGV",
    )
    for option, expected_value in (
        ("--mode", "monov3"),
        ("--backend", "CUDA0"),
        ("--layer-start", "0"),
        ("--layer-end", "40"),
        ("--model", geometry["cuda_model_path"]),
    ):
        _option(worker["argv"], option, expected_value, "cuda_route.worker")
    for key, expected_value in (
        ("LAYERSPLIT_MEMORY_CERT", "1"),
        ("LAYERSPLIT_MODEL_SHA256", model["artifact"]["sha256"]),
        ("LAYERSPLIT_PLACEMENT_CERT", "1"),
    ):
        common.exact(
            worker["environment"].get(key),
            expected_value,
            f"cuda_route.worker.environment.{key}",
        )


def _validate_phone(
    value: dict[str, Any],
    contract: dict[str, Any],
    model: dict[str, Any],
    history_path: Path,
    history_raw: bytes,
) -> None:
    common.exact_keys(value, PHONE_KEYS, "phone_route")
    for key, expected_value in (
        ("expected_file_type", 15),
        ("expected_max_streams", 8),
        ("expected_n_batch", 64),
        ("expected_n_ctx_seq", 512),
        ("expected_n_embd", 5120),
        ("expected_n_layer", 40),
        ("expected_n_ubatch", 64),
        ("history_path", str(history_path)),
        ("history_sha256", common.sha256_bytes(history_raw)),
        ("model_id", common.MODEL_ID),
        ("model_sha256", model["artifact"]["sha256"]),
        ("quality_corpus_content_sha256", contract["quality_corpus"]["sha256"]),
        ("schema", ARTIFACT_SCHEMAS["phone_route_launch"]),
    ):
        common.exact(value[key], expected_value, f"phone_route.{key}")
    common.integer(value["route_epoch"], "phone_route.route_epoch", 1)
    geometry = contract["model_geometry"][common.MODEL_ID]
    expected_layers = {
        "op15": ([0, 30], [0, 32]),
        "op12": ([30, 40], [24, 40]),
    }
    for endpoint in ("op12", "op15"):
        phone = common.exact_keys(
            value["phones"][endpoint],
            PHONE_KEYS_PER_DEVICE,
            f"phone_route.phones.{endpoint}",
        )
        for key in ("device", "model", "product", "serial"):
            common.exact(
                phone[key],
                contract["devices"][endpoint][key],
                f"phone_route.phones.{endpoint}.{key}",
            )
        common.exact(
            phone["boot_id"],
            common.UNBOUND_BOOT_IDS[endpoint],
            f"phone_route.phones.{endpoint}.boot_id",
        )
        common.exact(
            phone["executed_layers"],
            expected_layers[endpoint][0],
            f"phone_route.phones.{endpoint}.executed_layers",
        )
        common.exact(
            phone["stored_layers"],
            expected_layers[endpoint][1],
            f"phone_route.phones.{endpoint}.stored_layers",
        )
        shard = geometry["known_shards"][endpoint]
        common.exact(
            phone["loaded_shard_path"],
            shard["path"],
            f"phone_route.phones.{endpoint}.shard_path",
        )
        common.exact(
            phone["loaded_shard_sha256"],
            shard["sha256"],
            f"phone_route.phones.{endpoint}.shard_sha256",
        )
    common.exact(
        value["phones"]["op15"]["direct_peer_ipv4"],
        value["phones"]["op12"]["local_ipv4"],
        "phone_route.op15.direct_peer",
    )
    common.exact(
        value["phones"]["op12"]["direct_peer_ipv4"],
        value["phones"]["op15"]["local_ipv4"],
        "phone_route.op12.direct_peer",
    )
    mechanism = common.exact_keys(
        value["mechanism_commands"],
        {"desktop", "op12", "op15"},
        "phone_route.mechanism_commands",
    )
    common.require(
        type(mechanism["desktop"]) is list
        and len(mechanism["desktop"]) == 9,
        "E_PHONE_DESKTOP_COMMANDS",
    )
    common.exact(
        mechanism["op12"],
        [
            value["processes"]["op12_stagenet"]["argv"],
            value["probes"]["op12"]["before_argv"],
            value["probes"]["op12"]["after_argv"],
        ],
        "phone_route.mechanism.op12",
    )
    common.exact(
        mechanism["op15"],
        [
            value["processes"]["op15_stagenet"]["argv"],
            value["processes"]["op15_direct_relay"]["argv"],
            value["probes"]["op15"]["before_argv"],
            value["probes"]["op15"]["after_argv"],
        ],
        "phone_route.mechanism.op15",
    )


def _validate_joint(
    value: dict[str, Any],
    launch_paths: dict[str, Path],
    launch_raws: dict[str, bytes],
    history_path: Path,
    history_raw: bytes,
    mechanism: dict[str, Any],
    model_sha256: str,
) -> None:
    common.exact_keys(
        value,
        {
            "commands",
            "history",
            "mechanism_commands",
            "model_id",
            "model_sha256",
            "phase",
            "schema",
        },
        "joint",
    )
    common.exact(value["schema"], ARTIFACT_SCHEMAS["joint_capture_plan"], "joint.schema")
    common.exact(value["phase"], common.PHASE, "joint.phase")
    common.exact(value["model_id"], common.MODEL_ID, "joint.model")
    common.exact(value["model_sha256"], model_sha256, "joint.model_sha256")
    common.exact(value["mechanism_commands"], mechanism, "joint.mechanism")
    common.exact(
        value["history"],
        _artifact(history_path, history_raw),
        "joint.history",
    )
    for name in ("cuda", "phone"):
        command = value["commands"][name]
        index = command["launch_plan_argv_index"]
        common.integer(index, f"joint.{name}.launch_index", 1)
        common.require(index < len(command["argv_template"]), f"E_JOINT_INDEX: {name}")
        common.exact(
            command["argv_template"][index],
            str(launch_paths[f"{name}_route_launch"]),
            f"joint.{name}.launch_path",
        )
        common.exact(
            command["launch_plan_sha256"],
            common.sha256_bytes(launch_raws[f"{name}_route_launch"]),
            f"joint.{name}.launch_sha256",
        )
        records = command["executed_files"]
        common.require(
            [record["argv_index"] for record in records]
            == sorted({record["argv_index"] for record in records}),
            f"E_JOINT_EXECUTED_ORDER: {name}",
        )
        matches = [record for record in records if record["argv_index"] == index]
        common.require(len(matches) == 1, f"E_JOINT_LAUNCH_RECORD: {name}")
        common.exact(
            matches[0],
            {
                "argv_index": index,
                **_artifact(
                    launch_paths[f"{name}_route_launch"],
                    launch_raws[f"{name}_route_launch"],
                ),
            },
            f"joint.{name}.launch_record",
        )


def _joint_command(
    *,
    name: str,
    producer: Path,
    launch: Path,
    launch_raw: bytes,
    history: Path,
    mechanism_sha256: str,
    model_sha256: str,
    cwd: Path,
) -> dict[str, Any]:
    argv = [
        str(producer),
        "--output",
        "{output_path}",
        "--phase-id",
        "{phase_id}",
        "--pre-dir",
        "{pre_dir}",
        "--started",
        "{acquisition_started_ns}",
        "--plan",
        "{command_plan_sha256}",
        "--mechanism-commands-sha256",
        mechanism_sha256,
        "--model-sha256",
        model_sha256,
        "--launch-plan",
        str(launch),
    ]
    if name == "cuda":
        argv.extend(["--histories", str(history)])
    argv.extend(
        [
            "--execute",
            "--confirm",
            (
                "RUN_V24_CUDA_ROUTE_A_ONLY"
                if name == "cuda"
                else "RUN_V24_PHONE_ROUTE_A_ONLY"
            ),
        ]
    )
    launch_index = argv.index(str(launch))
    records = [
        _executed_file(producer, 0),
        {
            "argv_index": launch_index,
            **_artifact(launch, launch_raw),
        },
    ]
    if name == "cuda":
        records.append(_executed_file(history, argv.index(str(history))))
    return {
        "argv_template": argv,
        "cwd": str(cwd),
        "environment": {
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "executed_files": sorted(records, key=lambda value: value["argv_index"]),
        "launch_plan_argv_index": launch_index,
        "launch_plan_sha256": next(
            value["sha256"]
            for value in records
            if value["argv_index"] == launch_index
        ),
        "producer_sha256": records[0]["sha256"],
        "result_filename": f"{name}.result.json",
        "timeout_seconds": 7200,
    }


def build(
    *,
    spec_path: Path,
    output_paths: dict[str, Path],
    prospective_root_output: Path,
    report_output: Path,
) -> tuple[dict[str, bytes], dict[str, Any], dict[str, Any]]:
    expected_outputs = set(ARTIFACT_SCHEMAS)
    common.exact(set(output_paths), expected_outputs, "output_paths")
    all_outputs = [*output_paths.values(), prospective_root_output, report_output]
    common.require(
        all(path.is_absolute() and not path.exists() for path in all_outputs),
        "E_OUTPUT_EXISTS",
    )
    common.require(len(set(all_outputs)) == len(all_outputs), "E_OUTPUT_REUSE")
    spec, spec_raw = common.read_canonical(spec_path, "prospective_spec")
    common.exact_keys(
        spec,
        {
            "candidate_path",
            "contract_path",
            "cuda_monolithic_launch_path",
            "cuda_route_static",
            "identity_placeholders",
            "joint",
            "network_placeholders",
            "phone_route_static",
            "runtime_static",
            "schema",
            "token_history_path",
            "tokenizer_plan_path",
        },
        "prospective_spec",
    )
    common.exact(spec["schema"], SPEC_SCHEMA, "prospective_spec.schema")
    common.exact(
        spec["identity_placeholders"],
        common.UNBOUND_BOOT_IDS,
        "prospective_spec.identity_placeholders",
    )
    common.exact(
        spec["network_placeholders"],
        common.UNBOUND_PHONE_NETWORK,
        "prospective_spec.network_placeholders",
    )
    paths = {
        name: Path(spec[f"{name}_path"])
        for name in (
            "candidate",
            "contract",
            "cuda_monolithic_launch",
            "token_history",
            "tokenizer_plan",
        )
    }
    common.require(
        all(path.is_absolute() for path in paths.values()),
        "E_INPUT_ABSOLUTE",
    )
    contract, contract_raw = _read_input(
        paths["contract"],
        "s39-cp0-r1-evidence-contract-v2.4",
        "contract",
    )
    candidate, candidate_raw = _read_input(
        paths["candidate"],
        "s39-cp0-r1-candidate-v1",
        "candidate",
    )
    history, history_raw = _read_input(
        paths["token_history"],
        "s39-cp0-r1-token-history-v2.4",
        "token_history",
    )
    tokenizer, tokenizer_raw = _read_input(
        paths["tokenizer_plan"],
        "s39-cp0-r1-a-only-tokenizer-plan-v2",
        "tokenizer_plan",
    )
    mono, _ = _read_input(
        paths["cuda_monolithic_launch"],
        "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
        "cuda_monolithic_launch",
    )
    model = next(
        (
            value for value in candidate["models"]
            if value.get("slot") == "A"
            and value.get("model_id") == common.MODEL_ID
        ),
        None,
    )
    common.require(type(model) is dict, "E_MODEL_A")
    model_sha256 = model["artifact"]["sha256"]
    common.exact(history["candidate_sha256"], common.sha256_bytes(candidate_raw), "history.candidate")
    common.exact(history["model_sha256"], model_sha256, "history.model")
    common.exact(
        history["corpus_sha256"],
        contract["quality_corpus"]["sha256"],
        "history.corpus",
    )
    common.exact(
        tokenizer["model"]["sha256"],
        model_sha256,
        "tokenizer.model",
    )
    cuda_static = copy.deepcopy(spec["cuda_route_static"])
    common.exact_keys(cuda_static, CUDA_KEYS - CUDA_DERIVED, "cuda_route_static")
    cuda = {
        **cuda_static,
        "history_path": str(paths["token_history"]),
        "history_sha256": common.sha256_bytes(history_raw),
        "model_id": common.MODEL_ID,
        "model_sha256": model_sha256,
        "quality_corpus_content_sha256": history["corpus_sha256"],
        "schema": ARTIFACT_SCHEMAS["cuda_route_launch"],
    }
    phone_static = copy.deepcopy(spec["phone_route_static"])
    common.exact_keys(phone_static, PHONE_KEYS - PHONE_DERIVED, "phone_route_static")
    common.exact(set(phone_static["phones"]), {"op12", "op15"}, "phone_route_static.phones")
    for endpoint in ("op12", "op15"):
        common.exact_keys(
            phone_static["phones"][endpoint],
            PHONE_KEYS_PER_DEVICE
            - {"boot_id", "direct_peer_ipv4", "interface", "local_ipv4"},
            f"phone_route_static.phones.{endpoint}",
        )
        phone_static["phones"][endpoint]["boot_id"] = common.UNBOUND_BOOT_IDS[endpoint]
        phone_static["phones"][endpoint].update(
            common.UNBOUND_PHONE_NETWORK[endpoint]
        )
    phone_static["phones"]["op12"]["direct_peer_ipv4"] = (
        common.UNBOUND_PHONE_NETWORK["op15"]["local_ipv4"]
    )
    phone_static["phones"]["op15"]["direct_peer_ipv4"] = (
        common.UNBOUND_PHONE_NETWORK["op12"]["local_ipv4"]
    )
    phone = {
        **phone_static,
        "history_path": str(paths["token_history"]),
        "history_sha256": common.sha256_bytes(history_raw),
        "model_id": common.MODEL_ID,
        "model_sha256": model_sha256,
        "quality_corpus_content_sha256": history["corpus_sha256"],
        "schema": ARTIFACT_SCHEMAS["phone_route_launch"],
    }
    common.exact(
        cuda["mechanism_commands"],
        phone["mechanism_commands"],
        "E_MECHANISM_MATRIX",
    )
    common.exact(cuda["route_epoch"], phone["route_epoch"], "route_epoch")
    _validate_cuda(cuda, contract, model, paths["token_history"], history_raw)
    _validate_phone(phone, contract, model, paths["token_history"], history_raw)
    runtime_static = copy.deepcopy(spec["runtime_static"])
    common.exact_keys(
        runtime_static,
        {
            "bundle_roots",
            "bundles",
            "capture_entrypoints",
            "components",
            "tokenizer_component_id",
        },
        "runtime_static",
    )
    protocol = contract["token_history_protocol"]
    runtime = {
        "bundle_roots": runtime_static["bundle_roots"],
        "bundles": runtime_static["bundles"],
        "candidate_sha256": common.sha256_bytes(candidate_raw),
        "capture_entrypoints": runtime_static["capture_entrypoints"],
        "components": runtime_static["components"],
        "contract_sha256": common.sha256_bytes(contract_raw),
        "cuda_monolithic_launch": mono,
        "model_id": common.MODEL_ID,
        "phase": common.PHASE,
        "schema": ARTIFACT_SCHEMAS["runtime_plan"],
        "token_history": {
            "artifact_path": str(paths["token_history"]),
            "batch": protocol["batch"],
            "continuation_tokens_per_request": protocol[
                "continuation_tokens_per_request"
            ],
            "corpus_sha256": history["corpus_sha256"],
            "decode_calls_after_prefill": protocol[
                "decode_calls_after_prefill"
            ],
            "mechanics_item_indices": protocol["mechanics_item_indices"],
            "model_sha256": model_sha256,
            "n_batch": protocol["n_batch"],
            "n_ctx_seq": protocol["n_ctx_seq"],
            "n_ubatch": protocol["n_ubatch"],
            "prefill_chunking": protocol["prefill_chunking"],
            "prefill_row_order": protocol["prefill_row_order"],
            "quality_group_count": protocol["quality_group_count"],
            "quality_group_size": protocol["quality_group_size"],
            "quality_items": protocol["quality_items"],
            "tokenizer_component_id": runtime_static["tokenizer_component_id"],
            "tokenizer_plan_bytes": len(tokenizer_raw),
            "tokenizer_plan_path": str(paths["tokenizer_plan"]),
            "tokenizer_plan_sha256": common.sha256_bytes(tokenizer_raw),
        },
    }
    authority.validate_runtime_plan(
        runtime,
        contract,
        contract_raw,
        candidate_raw,
    )
    raw = {
        "cuda_route_launch": common.canonical_bytes(cuda),
        "phone_route_launch": common.canonical_bytes(phone),
        "runtime_plan": common.canonical_bytes(runtime),
    }
    joint_spec = common.exact_keys(spec["joint"], {"cwd"}, "joint")
    cwd = Path(joint_spec["cwd"])
    common.require(cwd.is_absolute() and cwd.is_dir(), "E_JOINT_CWD")
    mechanism_sha256 = common.sha256_bytes(
        common.canonical_bytes(phone["mechanism_commands"])
    )
    joint = {
        "commands": {
            "cuda": _joint_command(
                name="cuda",
                producer=PRODUCERS / "cuda_route_v1.py",
                launch=output_paths["cuda_route_launch"],
                launch_raw=raw["cuda_route_launch"],
                history=paths["token_history"],
                mechanism_sha256=mechanism_sha256,
                model_sha256=model_sha256,
                cwd=cwd,
            ),
            "phone": _joint_command(
                name="phone",
                producer=PRODUCERS / "phone_route_v1.py",
                launch=output_paths["phone_route_launch"],
                launch_raw=raw["phone_route_launch"],
                history=paths["token_history"],
                mechanism_sha256=mechanism_sha256,
                model_sha256=model_sha256,
                cwd=cwd,
            ),
        },
        "history": _artifact(paths["token_history"], history_raw),
        "mechanism_commands": phone["mechanism_commands"],
        "model_id": common.MODEL_ID,
        "model_sha256": model_sha256,
        "phase": common.PHASE,
        "schema": ARTIFACT_SCHEMAS["joint_capture_plan"],
    }
    raw["joint_capture_plan"] = common.canonical_bytes(joint)
    _validate_joint(
        joint,
        output_paths,
        raw,
        paths["token_history"],
        history_raw,
        phone["mechanism_commands"],
        model_sha256,
    )
    root = {
        "acquisition_ready": False,
        "artifacts": {
            name: _artifact(output_paths[name], value)
            for name, value in sorted(raw.items())
        },
        "candidate": _artifact(paths["candidate"], candidate_raw),
        "contract": _artifact(paths["contract"], contract_raw),
        "cuda_monolithic_launch": _artifact(
            paths["cuda_monolithic_launch"],
            common.canonical_bytes(mono),
        ),
        "desktop_control": {
            "cuda_ssh_target": common.CUDA_SSH_TARGET,
            "phone_adb_port": common.PHONE_ADB_PORT,
        },
        "identity_placeholders": common.UNBOUND_BOOT_IDS,
        "model_id": common.MODEL_ID,
        "model_sha256": model_sha256,
        "network_placeholders": common.UNBOUND_PHONE_NETWORK,
        "phase": common.PHASE,
        "schema": ROOT_SCHEMA,
        "spec": _artifact(spec_path, spec_raw),
        "status": "POST_REBOOT_IDENTITY_BINDING_REQUIRED",
        "token_history": _artifact(paths["token_history"], history_raw),
        "tokenizer_plan": _artifact(paths["tokenizer_plan"], tokenizer_raw),
    }
    root_raw = common.canonical_bytes(root)
    report = {
        "acquisition_ready": False,
        "network_execution": False,
        "outputs": root["artifacts"],
        "prospective_root": _artifact(prospective_root_output, root_raw),
        "schema": REPORT_SCHEMA,
        "status": "DRY_RUN_PASS_POST_REBOOT_BINDING_REQUIRED",
        "unbound_identity_fields": [
            "cuda.boot_id",
            "op12.boot_id",
            "op15.boot_id",
        ],
    }
    return raw, root, report


def materialize(
    *,
    spec_path: Path,
    output_paths: dict[str, Path],
    prospective_root_output: Path,
    report_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, root, report = build(
        spec_path=spec_path,
        output_paths=output_paths,
        prospective_root_output=prospective_root_output,
        report_output=report_output,
    )
    for name, value in raw.items():
        common.write_raw_new(output_paths[name], value)
    common.write_new(prospective_root_output, root)
    common.write_new(report_output, report)
    return root, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    for name in ARTIFACT_SCHEMAS:
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=Path,
            required=True,
        )
    parser.add_argument("--prospective-root", type=Path, required=True)
    parser.add_argument("--dry-run-report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        materialize(
            spec_path=args.spec.resolve(),
            output_paths={
                name: getattr(args, name).resolve()
                for name in ARTIFACT_SCHEMAS
            },
            prospective_root_output=args.prospective_root.resolve(),
            report_output=args.dry_run_report.resolve(),
        )
        return 0
    except (OSError, ValueError, common.ProductionError) as error:
        print(f"V24_RUNTIME_ORIGIN_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
