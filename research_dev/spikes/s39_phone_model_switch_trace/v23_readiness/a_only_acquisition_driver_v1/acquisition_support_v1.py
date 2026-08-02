#!/usr/bin/env python3
"""Fail-closed fan-in for one CP0-R1 V2.3 A_ONLY acquisition."""

from __future__ import annotations

import argparse
import ast
import copy
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Protocol


import cp0_r1_evidence_v2 as v2
import cp0_r1_evidence_v21 as v21
import cp0_r1_evidence_v22 as v22
import cp0_r1_evidence_v23 as v23
import v23_common as common


PHASE = "A_ONLY"
MODEL_ID = "qwen3-14b-q4_k_m"
CLOCK_ID = time.CLOCK_MONOTONIC_RAW
PLAN_SCHEMA = "s39-cp0-r1-a-only-runtime-command-plan-v1"
JOINT_SCHEMA = "s39-cp0-r1-a-only-joint-phone-cuda-raw-v1"
MONOLITHIC_SCHEMA = "s39-cp0-r1-a-only-cuda-monolithic-raw-v1"
COMMON_ROW_KEYS = {"acquisition_id", "phase", "phase_id", "role"}
OUTPUT_FILES = {
    f"model.{MODEL_ID}.mechanics.phone": "mechanics-phone.jsonl",
    f"model.{MODEL_ID}.oracle.cuda_route": "oracle-cuda-route.jsonl",
    f"model.{MODEL_ID}.oracle.cuda_monolithic": "oracle-cuda-monolithic.jsonl",
    f"model.{MODEL_ID}.cuda_memory": "cuda-memory.jsonl",
    f"model.{MODEL_ID}.quality.cuda": "quality-cuda.jsonl",
    f"model.{MODEL_ID}.quality.phone": "quality-phone.jsonl",
    f"model.{MODEL_ID}.bridge": "bridge.jsonl",
    f"model.{MODEL_ID}.placement.op15": "placement-op15.jsonl",
    f"model.{MODEL_ID}.placement.op12": "placement-op12.jsonl",
    f"model.{MODEL_ID}.route_transfer": "route-transfer.jsonl",
}
RUNTIME_FILE = "runtime_identity.json"
PLAN_KEYS = {
    "candidate_sha256",
    "contract_sha256",
    "mechanism_commands",
    "model_id",
    "model_sha256",
    "outputs",
    "phase",
    "producers",
    "schema",
}
PRODUCER_KEYS = {
    "argv_template",
    "executed_files",
    "result_filename",
    "timeout_seconds",
}
EXECUTED_FILE_KEYS = {"argv_index", "bytes", "path", "sha256"}
PRODUCER_PLACEHOLDERS = {
    "{acquisition_started_ns}",
    "{command_plan_sha256}",
    "{output_path}",
    "{phase_id}",
    "{pre_dir}",
}
JOINT_KEYS = {
    "completed_ns",
    "cuda_memory_rows",
    "cuda_route_rows",
    "gpu_runtime",
    "mechanism_commands_sha256",
    "mechanics_rows",
    "model_id",
    "model_sha256",
    "op12_runtime",
    "op15_runtime",
    "phase_id",
    "placement_op12_rows",
    "placement_op15_rows",
    "quality_cuda_rows",
    "quality_phone_rows",
    "route_epoch",
    "route_transfer_rows",
    "schema",
    "started_ns",
    "bridge_rows",
}
MONOLITHIC_KEYS = {
    "completed_ns",
    "mechanism_commands_sha256",
    "model_id",
    "model_sha256",
    "oracle_cuda_monolithic_rows",
    "phase_id",
    "schema",
    "started_ns",
}
GPU_RUNTIME_KEYS = {
    "artifact_path",
    "gpu_uuid",
    "host_boot_id",
    "model_id",
    "route_epoch",
}
PHONE_RUNTIME_KEYS = {
    "active_sequences_after_cleanup",
    "available_bytes",
    "boot_id",
    "direct_peer",
    "gpu_max_millic",
    "interface_after",
    "interface_before",
    "loaded_shard_path",
    "model_id",
    "process_swap_bytes",
    "route_epoch",
    "serial",
    "session_protocol_version",
    "worker_boot_nonce",
    "worker_executable_path",
    "worker_model_sha256",
    "worker_pid",
    "worker_start_ticks",
}


class AcquisitionCommandRunner(Protocol):
    def run(
        self,
        argv: list[str],
        *,
        timeout: int,
    ) -> subprocess.CompletedProcess[bytes]:
        ...


class SubprocessCommandRunner:
    def run(
        self,
        argv: list[str],
        *,
        timeout: int,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            argv,
            capture_output=True,
            check=False,
            timeout=timeout,
        )


def clock_ns() -> int:
    return time.clock_gettime_ns(CLOCK_ID)


def durable_write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o644,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _relative(value: Any, field: str) -> str:
    value = common.string(value, field)
    path = Path(value)
    common.require(
        not path.is_absolute()
        and value not in (".", "..")
        and ".." not in path.parts,
        f"E_PATH: {field}",
    )
    return value


def _validate_executed_files(
    records: Any,
    template: list[str],
    field: str,
) -> list[dict[str, Any]]:
    common.require(type(records) is list and bool(records), f"E_TYPE: {field}")
    indexes = set()
    for index, raw in enumerate(records):
        item = f"{field}[{index}]"
        raw = common.exact_keys(raw, EXECUTED_FILE_KEYS, item)
        argv_index = common.integer(raw["argv_index"], f"{item}.argv_index")
        common.require(argv_index < len(template), f"E_RANGE: {item}.argv_index")
        common.require(argv_index not in indexes, f"E_INDEX_REUSE: {item}")
        indexes.add(argv_index)
        path = common.string(raw["path"], f"{item}.path")
        common.require(Path(path).is_absolute(), f"E_PATH: {item}.path")
        common.exact(template[argv_index], path, f"{item}.argv")
        common.integer(raw["bytes"], f"{item}.bytes", 1)
        common.digest(raw["sha256"], f"{item}.sha256")
    common.require(0 in indexes, f"E_ENTRYPOINT: {field}")
    return records


def _verify_self_contained_source(path: Path, field: str) -> None:
    try:
        source = path.read_bytes().decode("ascii")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        raise common.ReadinessError(f"E_PRODUCER_SOURCE: {field}") from error
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            common.require(node.level == 0, f"E_PRODUCER_IMPORT: {field}")
            names = [node.module or ""]
        for name in names:
            root = name.split(".", 1)[0]
            common.require(
                root in sys.stdlib_module_names or root == "__future__",
                f"E_PRODUCER_IMPORT: {field}: {name}",
            )


def _validate_producer(value: Any, name: str) -> dict[str, Any]:
    field = f"plan.producers.{name}"
    value = common.exact_keys(value, PRODUCER_KEYS, field)
    template = value["argv_template"]
    common.require(
        type(template) is list
        and bool(template)
        and all(type(item) is str and bool(item) for item in template),
        f"E_TYPE: {field}.argv_template",
    )
    used = {
        item
        for item in template
        if item.startswith("{") and item.endswith("}")
    }
    common.exact(used, PRODUCER_PLACEHOLDERS, f"{field}.placeholders")
    common.require(
        all(item in PRODUCER_PLACEHOLDERS or "{" not in item for item in template),
        f"E_PLACEHOLDER: {field}.argv_template",
    )
    _validate_executed_files(value["executed_files"], template, f"{field}.executed_files")
    _verify_self_contained_source(Path(template[0]), f"{field}.entrypoint")
    _relative(value["result_filename"], f"{field}.result_filename")
    timeout = common.integer(value["timeout_seconds"], f"{field}.timeout", 1)
    common.require(timeout <= 7200, f"E_RANGE: {field}.timeout")
    return value


def _validate_mechanism_commands(value: Any) -> dict[str, list[list[str]]]:
    value = common.exact_keys(
        value,
        {"desktop", "op12", "op15"},
        "plan.mechanism_commands",
    )
    total = 0
    for endpoint in ("desktop", "op15", "op12"):
        commands = value[endpoint]
        common.require(
            type(commands) is list and bool(commands),
            f"E_TYPE: plan.mechanism_commands.{endpoint}",
        )
        for index, argv in enumerate(commands):
            common.require(
                type(argv) is list
                and bool(argv)
                and all(type(item) is str and bool(item) for item in argv),
                f"E_TYPE: plan.mechanism_commands.{endpoint}[{index}]",
            )
            common.require(
                not any("{" in item or "}" in item for item in argv),
                f"E_PLACEHOLDER: plan.mechanism_commands.{endpoint}[{index}]",
            )
            total += 1
    common.require(total >= 5, "E_MECHANISM_COMMAND_COUNT")
    return value


def load_plan(
    path: Path,
    contract_raw: bytes,
    candidate_raw: bytes,
    model: dict[str, Any],
) -> tuple[dict[str, Any], bytes]:
    plan, raw = common.read_canonical(path)
    common.exact_keys(plan, PLAN_KEYS, "plan")
    common.exact(
        plan["schema"],
        "s39-cp0-r1-a-only-runtime-command-plan-v1",
        "plan.schema",
    )
    common.exact(plan["phase"], PHASE, "plan.phase")
    common.exact(plan["model_id"], MODEL_ID, "plan.model_id")
    common.exact(
        plan["model_sha256"],
        model["artifact"]["sha256"],
        "plan.model_sha256",
    )
    common.exact(
        plan["contract_sha256"],
        common.sha256_bytes(contract_raw),
        "plan.contract_sha256",
    )
    common.exact(
        plan["candidate_sha256"],
        common.sha256_bytes(candidate_raw),
        "plan.candidate_sha256",
    )
    common.exact(plan["outputs"], {**OUTPUT_FILES, "runtime_identity": RUNTIME_FILE}, "plan.outputs")
    _validate_mechanism_commands(plan["mechanism_commands"])
    producers = common.exact_keys(
        plan["producers"],
        {"cuda_monolithic", "joint_phone_cuda"},
        "plan.producers",
    )
    paths = set(OUTPUT_FILES.values()) | {RUNTIME_FILE}
    for name in ("joint_phone_cuda", "cuda_monolithic"):
        producer = _validate_producer(producers[name], name)
        result = producer["result_filename"]
        common.require(result not in paths, f"E_PATH_REUSE: {result}")
        paths.add(result)
    return plan, raw


def _capture_file(path: Path, expected: dict[str, Any], destination: Path) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise common.ReadinessError(f"E_EXECUTED_FILE: {path}: {error}") from error
    try:
        stat = os.fstat(fd)
        chunks = bytearray()
        while chunk := os.read(fd, 1024 * 1024):
            chunks.extend(chunk)
    finally:
        os.close(fd)
    raw = bytes(chunks)
    common.exact(stat.st_size, expected["bytes"], f"E_EXECUTED_BYTES: {path}")
    common.exact(
        common.sha256_bytes(raw),
        expected["sha256"],
        f"E_EXECUTED_SHA256: {path}",
    )
    durable_write_new(destination, raw)
    destination.chmod(stat.st_mode & 0o777)
    return str(destination)


def capture_producers(plan: dict[str, Any], raw_dir: Path) -> dict[tuple[str, int], str]:
    captured = {}
    for name in ("joint_phone_cuda", "cuda_monolithic"):
        producer = plan["producers"][name]
        for record in producer["executed_files"]:
            index = record["argv_index"]
            path = Path(producer["argv_template"][index])
            destination = raw_dir / "executed" / name / f"{index:03d}-{path.name}"
            captured[(name, index)] = _capture_file(path, record, destination)
    return captured


def _producer_argv(
    producer: dict[str, Any],
    name: str,
    raw_dir: Path,
    pre_dir: Path,
    phase_id: str,
    acquisition_started_ns: int,
    plan_sha256: str,
    captured: dict[tuple[str, int], str],
) -> tuple[list[str], Path]:
    output = raw_dir / producer["result_filename"]
    replacements = {
        "{acquisition_started_ns}": str(acquisition_started_ns),
        "{command_plan_sha256}": plan_sha256,
        "{output_path}": str(output),
        "{phase_id}": phase_id,
        "{pre_dir}": str(pre_dir),
    }
    argv = [
        captured.get((name, index), replacements.get(item, item))
        for index, item in enumerate(producer["argv_template"])
    ]
    return argv, output


def run_producer(
    runner: AcquisitionCommandRunner,
    producer: dict[str, Any],
    name: str,
    argv: list[str],
    result_path: Path,
    receipt_dir: Path,
    now_ns: Callable[[], int],
) -> tuple[dict[str, Any], bytes, int, int]:
    source_argv = [sys.executable, "-I", "-B", *argv]
    started_ns = now_ns()
    try:
        completed = runner.run(source_argv, timeout=producer["timeout_seconds"])
    except subprocess.TimeoutExpired as error:
        raise common.ReadinessError(f"E_PRODUCER_TIMEOUT: {name}") from error
    completed_ns = now_ns()
    durable_write_new(receipt_dir / f"{name}.stdout", completed.stdout)
    durable_write_new(receipt_dir / f"{name}.stderr", completed.stderr)
    common.write_exclusive(
        receipt_dir / f"{name}.receipt.json",
        {
            "argv": source_argv,
            "completed_ns": completed_ns,
            "returncode": completed.returncode,
            "schema": "s39-cp0-r1-a-only-producer-receipt-v1",
            "started_ns": started_ns,
        },
    )
    common.exact(completed.returncode, 0, f"E_PRODUCER_EXIT: {name}")
    common.exact(completed.stdout, b"", f"E_PRODUCER_STDOUT: {name}")
    common.exact(completed.stderr, b"", f"E_PRODUCER_STDERR: {name}")
    common.require(
        result_path.is_file() and not result_path.is_symlink(),
        f"E_PRODUCER_OUTPUT: {name}",
    )
    value, raw = common.read_canonical(result_path)
    return value, raw, started_ns, completed_ns


def _source_interval(
    value: dict[str, Any],
    phase_id: str,
    schema: str,
    acquisition_started_ns: int,
    receipt_started_ns: int,
    receipt_completed_ns: int,
    mechanism_sha256: str,
    field: str,
) -> tuple[int, int]:
    common.exact(value["schema"], schema, f"{field}.schema")
    common.exact(value["phase_id"], phase_id, f"{field}.phase_id")
    common.exact(value["model_id"], MODEL_ID, f"{field}.model_id")
    common.digest(value["model_sha256"], f"{field}.model_sha256")
    common.exact(
        value["mechanism_commands_sha256"],
        mechanism_sha256,
        f"{field}.mechanism_commands",
    )
    started = common.integer(value["started_ns"], f"{field}.started_ns", 1)
    completed = common.integer(value["completed_ns"], f"{field}.completed_ns", 1)
    common.require(
        acquisition_started_ns <= receipt_started_ns <= started < completed
        <= receipt_completed_ns,
        f"E_SOURCE_INTERVAL: {field}",
    )
    return started, completed


def _wrap_rows(
    bodies: Any,
    role: str,
    phase_id: str,
    started_ns: int,
    completed_ns: int,
) -> list[dict[str, Any]]:
    common.require(type(bodies) is list and bool(bodies), f"E_ROWS: {role}")
    rows = []
    previous = None
    for index, source in enumerate(bodies):
        field = f"{role}[{index}]"
        common.require(type(source) is dict, f"E_TYPE: {field}")
        common.require(
            not COMMON_ROW_KEYS.intersection(source),
            f"E_WRAPPER_FIELDS: {field}",
        )
        event_ns = common.integer(source.get("event_ns"), f"{field}.event_ns", 1)
        common.require(started_ns <= event_ns <= completed_ns, f"E_EVENT_INTERVAL: {field}")
        if previous is not None:
            common.require(previous <= event_ns, f"E_EVENT_ORDER: {role}")
        previous = event_ns
        rows.append({
            "acquisition_id": phase_id,
            **copy.deepcopy(source),
            "phase": PHASE,
            "phase_id": phase_id,
            "role": role,
        })
    return rows


def _read_pre_rows(path: Path, role: str, phase_id: str) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    rows = v2.parse_jsonl(raw, role, phase_id)
    common.require(bool(rows), f"E_ROWS: {role}")
    return rows, raw


def _validate_roles(
    rows: dict[str, list[dict[str, Any]]],
    model: dict[str, Any],
    route: dict[str, Any],
    phase_id: str,
    corpus_rows: list[dict[str, Any]],
    corpus_raw: bytes,
    v22_contract: dict[str, Any],
    task_suite: dict[str, Any],
    parent: dict[str, Any],
) -> dict[str, Any]:
    prefix = f"model.{MODEL_ID}"
    v22.validate_exact_corpus(corpus_rows, v22.load_frozen_corpus(v22_contract))
    v22.validate_incumbent_route(route, v22_contract, model)
    phone, tested, oracle = v21._validate_execution_geometry(rows, model, phase_id)
    v22.validate_continuations(
        rows,
        model,
        phase_id,
        v22_contract["gates"]["continuation_tokens_per_request"],
    )
    normalized = {
        role: v21._normalize_rows(role_rows)
        for role, role_rows in rows.items()
    }
    oracle_derived = v2.derive_oracle(normalized, model, phase_id)
    allocation, cuda_ready = v21._derive_memory(
        rows[f"{prefix}.cuda_memory"],
        f"{prefix}.cuda_memory",
        phase_id,
        model,
        parent,
    )
    quality, _ = v21._derive_quality(
        {**rows, "quality.corpus": corpus_rows},
        {"quality.corpus": common.sha256_bytes(corpus_raw)},
        model,
        phase_id,
        task_suite,
        parent,
    )
    common.require(
        quality["cuda_correct"]
        >= v22_contract["gates"]["cuda_quality_minimum_correct_items"],
        "E_CUDA_QUALITY_FLOOR",
    )
    bridge = v21._derive_bridge(
        rows[f"{prefix}.bridge"],
        f"{prefix}.bridge",
        phase_id,
        model,
        phone,
        rows[f"{prefix}.mechanics.phone"],
        cuda_ready,
        v22_contract["phase_protocol"]["clock_id"],
        parent,
    )
    v22.validate_bridge_causality(rows, model)
    placement = {}
    for phone_name in ("op15", "op12"):
        role = f"{prefix}.placement.{phone_name}"
        placement[phone_name] = v2.derive_placement(
            normalized[role],
            role,
            phase_id,
            model,
            phone_name,
            route,
            parent,
        )
    transfer = v21._derive_transfer(
        rows[f"{prefix}.route_transfer"],
        f"{prefix}.route_transfer",
        phase_id,
        model,
        route,
        phone,
    )
    del tested, oracle
    return {
        "allocation": allocation,
        "bridge": bridge,
        "cuda_ready": cuda_ready,
        "oracle": oracle_derived,
        "placement": placement,
        "quality": quality,
        "transfer": transfer,
    }


def _artifacts_by_endpoint(
    snapshot: dict[str, Any],
    model: dict[str, Any],
    route: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    artifacts = v23.validate_artifact_snapshot(snapshot, PHASE, model, route)
    result = {}
    for record in artifacts.values():
        endpoint = record["endpoint"]
        common.require(endpoint not in result, f"E_ARTIFACT_ENDPOINT_REUSE: {endpoint}")
        result[endpoint] = record
    common.exact(
        sorted(result),
        ["cuda", "op12", "op12_worker", "op15", "op15_worker"],
        "artifact.endpoints",
    )
    return result


def _phone_executor(
    source: Any,
    phone: str,
    route_epoch: int,
    model: dict[str, Any],
    fresh_phone: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    role_digests: dict[str, str],
) -> dict[str, Any]:
    source = common.exact_keys(source, PHONE_RUNTIME_KEYS, f"runtime.{phone}")
    common.exact(source["model_id"], MODEL_ID, f"runtime.{phone}.model_id")
    common.exact(source["route_epoch"], route_epoch, f"runtime.{phone}.epoch")
    common.exact(source["serial"], fresh_phone["serial"], f"runtime.{phone}.serial")
    common.exact(source["boot_id"], fresh_phone["boot_id"], f"runtime.{phone}.boot")
    common.exact(
        source["worker_executable_path"],
        artifacts[f"{phone}_worker"]["path"],
        f"runtime.{phone}.worker_path",
    )
    common.exact(
        source["loaded_shard_path"],
        artifacts[phone]["path"],
        f"runtime.{phone}.shard_path",
    )
    prefix = f"model.{MODEL_ID}"
    return {
        **copy.deepcopy(source),
        "executor_id": f"PHONE_{phone.upper()}",
        "loaded_shard_sha256": artifacts[phone]["sha256"],
        "loaded_shard_stat": copy.deepcopy(artifacts[phone]["stat"]),
        "mechanics_sha256": role_digests[f"{prefix}.mechanics.phone"],
        "placement_sha256": role_digests[f"{prefix}.placement.{phone}"],
        "route_transfer_sha256": role_digests[f"{prefix}.route_transfer"],
        "worker_executable_sha256": artifacts[f"{phone}_worker"]["sha256"],
        "worker_executable_stat": copy.deepcopy(artifacts[f"{phone}_worker"]["stat"]),
    }


def _build_runtime(
    joint: dict[str, Any],
    fresh: dict[str, Any],
    fresh_raw: bytes,
    artifacts: dict[str, dict[str, Any]],
    role_digests: dict[str, str],
    model: dict[str, Any],
    phase_id: str,
) -> dict[str, Any]:
    route_epoch = common.integer(joint["route_epoch"], "joint.route_epoch", 1)
    gpu = common.exact_keys(joint["gpu_runtime"], GPU_RUNTIME_KEYS, "joint.gpu_runtime")
    common.exact(gpu["model_id"], MODEL_ID, "joint.gpu_runtime.model")
    common.exact(gpu["route_epoch"], route_epoch, "joint.gpu_runtime.epoch")
    common.exact(gpu["gpu_uuid"], fresh["cuda"]["uuid"], "joint.gpu_runtime.uuid")
    common.exact(
        gpu["host_boot_id"],
        fresh["cuda"]["host_boot_id"],
        "joint.gpu_runtime.host_boot",
    )
    common.exact(gpu["artifact_path"], artifacts["cuda"]["path"], "joint.gpu_runtime.path")
    gpu_final = {
        **copy.deepcopy(gpu),
        "artifact_sha256": artifacts["cuda"]["sha256"],
        "artifact_stat": copy.deepcopy(artifacts["cuda"]["stat"]),
        "executor_id": "GPU",
    }
    phone_executors = [
        _phone_executor(
            joint[f"{phone}_runtime"],
            phone,
            route_epoch,
            model,
            fresh["phones"][phone],
            artifacts,
            role_digests,
        )
        for phone in ("op15", "op12")
    ]
    return {
        "completed_ns": joint["completed_ns"],
        "executors": [gpu_final, *phone_executors],
        "fresh_snapshot_sha256": common.sha256_bytes(fresh_raw),
        "phase": PHASE,
        "phase_id": phase_id,
        "route_epoch": route_epoch,
        "schema": "s39-cp0-r1-runtime-identity-v2.3",
        "started_ns": joint["started_ns"],
    }


def acquire(
    command_plan: Path,
    output_dir: Path,
    phase_id: str,
    pre_dir: Path,
    acquisition_started_ns: int,
    *,
    runner: AcquisitionCommandRunner | None = None,
    now_ns: Callable[[], int] = clock_ns,
    contract_path: Path = v23.DEFAULT_CONTRACT,
    candidate_path: Path = v23.DEFAULT_CANDIDATE,
) -> dict[str, Any]:
    common.require(command_plan.is_absolute(), "E_PATH: command_plan")
    common.require(output_dir.is_absolute() and pre_dir.is_absolute(), "E_PATH: directories")
    common.require(output_dir.is_dir() and not output_dir.is_symlink(), "E_OUTPUT_DIR")
    common.require(
        phase_id.startswith("cp0-r1-v23-a-only-")
        and len(phase_id) <= 128
        and all(character.isalnum() or character in ".-_" for character in phase_id),
        "E_PHASE_ID",
    )
    acquisition_started_ns = common.integer(
        acquisition_started_ns,
        "acquisition_started_ns",
        1,
    )
    contract, contract_raw, candidate, candidate_raw = v23.validate_inputs(
        contract_path,
        candidate_path,
    )
    (
        v22_contract,
        _,
        v22_candidate,
        _,
        parent,
        frozen_corpus,
    ) = v22.validate_inputs(v22.DEFAULT_CONTRACT, candidate_path)
    common.exact(candidate, v22_candidate, "E_CANDIDATE_VERSION")
    model = next(
        item for item in candidate["models"]
        if item["slot"] == "A" and item["model_id"] == MODEL_ID
    )
    plan, plan_raw = load_plan(
        command_plan,
        contract_raw,
        candidate_raw,
        model,
    )
    plan_sha256 = common.sha256_bytes(plan_raw)
    mechanism_sha256 = v2.digest_json(plan["mechanism_commands"])

    route_role = f"model.{MODEL_ID}.route_lock"
    route_rows, _ = _read_pre_rows(pre_dir / "route_lock.jsonl", route_role, phase_id)
    common.require(len(route_rows) == 1, "E_ROUTE_ROWS")
    route = route_rows[0]
    v22.validate_incumbent_route(route, v22_contract, model)
    corpus_rows, corpus_raw = _read_pre_rows(
        pre_dir / "quality_corpus.jsonl",
        "quality.corpus",
        phase_id,
    )
    v22.validate_exact_corpus(corpus_rows, frozen_corpus)

    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    captured = capture_producers(plan, raw_dir)
    runner = runner or SubprocessCommandRunner()
    producer_results = {}
    receipts = {}
    for name in ("joint_phone_cuda", "cuda_monolithic"):
        producer = plan["producers"][name]
        argv, result_path = _producer_argv(
            producer,
            name,
            raw_dir,
            pre_dir,
            phase_id,
            acquisition_started_ns,
            plan_sha256,
            captured,
        )
        value, raw, receipt_start, receipt_end = run_producer(
            runner,
            producer,
            name,
            argv,
            result_path,
            raw_dir,
            now_ns,
        )
        producer_results[name] = (value, raw)
        receipts[name] = (receipt_start, receipt_end)

    joint, _ = producer_results["joint_phone_cuda"]
    common.exact_keys(joint, JOINT_KEYS, "joint")
    joint_started, joint_completed = _source_interval(
        joint,
        phase_id,
        "s39-cp0-r1-a-only-joint-phone-cuda-raw-v1",
        acquisition_started_ns,
        *receipts["joint_phone_cuda"],
        mechanism_sha256,
        "joint",
    )
    common.exact(joint["model_sha256"], model["artifact"]["sha256"], "joint.model")
    monolithic, _ = producer_results["cuda_monolithic"]
    common.exact_keys(monolithic, MONOLITHIC_KEYS, "monolithic")
    monolithic_started, monolithic_completed = _source_interval(
        monolithic,
        phase_id,
        "s39-cp0-r1-a-only-cuda-monolithic-raw-v1",
        acquisition_started_ns,
        *receipts["cuda_monolithic"],
        mechanism_sha256,
        "monolithic",
    )
    common.exact(
        monolithic["model_sha256"],
        model["artifact"]["sha256"],
        "monolithic.model",
    )

    prefix = f"model.{MODEL_ID}"
    rows = {
        f"{prefix}.mechanics.phone": _wrap_rows(
            joint["mechanics_rows"],
            f"{prefix}.mechanics.phone",
            phase_id,
            joint_started,
            joint_completed,
        ),
        f"{prefix}.oracle.cuda_route": _wrap_rows(
            joint["cuda_route_rows"],
            f"{prefix}.oracle.cuda_route",
            phase_id,
            joint_started,
            joint_completed,
        ),
        f"{prefix}.oracle.cuda_monolithic": _wrap_rows(
            monolithic["oracle_cuda_monolithic_rows"],
            f"{prefix}.oracle.cuda_monolithic",
            phase_id,
            monolithic_started,
            monolithic_completed,
        ),
        f"{prefix}.cuda_memory": _wrap_rows(
            joint["cuda_memory_rows"],
            f"{prefix}.cuda_memory",
            phase_id,
            joint_started,
            joint_completed,
        ),
        f"{prefix}.quality.cuda": _wrap_rows(
            joint["quality_cuda_rows"],
            f"{prefix}.quality.cuda",
            phase_id,
            joint_started,
            joint_completed,
        ),
        f"{prefix}.quality.phone": _wrap_rows(
            joint["quality_phone_rows"],
            f"{prefix}.quality.phone",
            phase_id,
            joint_started,
            joint_completed,
        ),
        f"{prefix}.bridge": _wrap_rows(
            joint["bridge_rows"],
            f"{prefix}.bridge",
            phase_id,
            joint_started,
            joint_completed,
        ),
        f"{prefix}.placement.op15": _wrap_rows(
            joint["placement_op15_rows"],
            f"{prefix}.placement.op15",
            phase_id,
            joint_started,
            joint_completed,
        ),
        f"{prefix}.placement.op12": _wrap_rows(
            joint["placement_op12_rows"],
            f"{prefix}.placement.op12",
            phase_id,
            joint_started,
            joint_completed,
        ),
        f"{prefix}.route_transfer": _wrap_rows(
            joint["route_transfer_rows"],
            f"{prefix}.route_transfer",
            phase_id,
            joint_started,
            joint_completed,
        ),
    }
    derived = _validate_roles(
        rows,
        model,
        route,
        phase_id,
        corpus_rows,
        corpus_raw,
        v22_contract,
        candidate["task_suite"],
        parent,
    )
    role_raw = {
        role: b"".join(v2.canonical_line(row) for row in role_rows)
        for role, role_rows in rows.items()
    }
    role_digests = {
        role: common.sha256_bytes(raw)
        for role, raw in role_raw.items()
    }

    artifact, _ = common.read_canonical(
        pre_dir.parent / "artifact" / "artifact_snapshot.json"
    )
    artifacts = _artifacts_by_endpoint(artifact, model, route)
    lock, lock_raw = common.read_canonical(
        pre_dir.parent / "fresh" / "readiness_lock.json"
    )
    fresh, fresh_raw = common.read_canonical(
        pre_dir.parent / "fresh" / "fresh_snapshot.json"
    )
    phase_lock_raw = (pre_dir / "phase_lock.jsonl").read_bytes()
    manifest_stub = {
        "acquisition_started_ns": acquisition_started_ns,
        "phase": PHASE,
        "phase_id": phase_id,
    }
    v23.validate_readiness_lock(
        lock,
        manifest_stub,
        common.sha256_bytes(phase_lock_raw),
        common.canonical_bytes(artifact),
        artifact["completed_ns"],
    )
    fresh_derived = v23.validate_fresh_snapshot(
        fresh,
        fresh_raw,
        lock_raw,
        lock,
        manifest_stub,
        contract,
        {
            common.artifact_key(record["endpoint"], record["path"]): record
            for record in artifacts.values()
        },
    )
    runtime = _build_runtime(
        joint,
        fresh,
        fresh_raw,
        artifacts,
        role_digests,
        model,
        phase_id,
    )
    v23.validate_runtime_identity(
        runtime,
        common.canonical_bytes(runtime),
        fresh_raw,
        fresh_derived,
        {
            **manifest_stub,
            "phase_closed_ns": max(joint_completed, monolithic_completed),
        },
        contract,
        {**role_digests, "loaded_artifacts": {
            common.artifact_key(record["endpoint"], record["path"]): record
            for record in artifacts.values()
        }},
        derived["transfer"]["direct_payload_bytes"],
        MODEL_ID,
    )

    for role in sorted(OUTPUT_FILES):
        durable_write_new(output_dir / OUTPUT_FILES[role], role_raw[role])
    common.write_exclusive(output_dir / RUNTIME_FILE, runtime)
    result = {
        "command_plan_sha256": plan_sha256,
        "derived": derived,
        "output_sha256s": {
            **role_digests,
            "runtime_identity": common.sha256_bytes(common.canonical_bytes(runtime)),
        },
        "schema": "s39-cp0-r1-a-only-acquisition-driver-result-v1",
        "status": "RAW_ROLES_EMITTED_PENDING_OUTER_V2_3_VALIDATION",
    }
    common.write_exclusive(output_dir / "ACQUISITION_DRIVER_RESULT.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--pre-dir", type=Path, required=True)
    parser.add_argument("--acquisition-started-ns", type=int, required=True)
    parser.add_argument("--contract", type=Path, default=v23.DEFAULT_CONTRACT)
    parser.add_argument("--candidate", type=Path, default=v23.DEFAULT_CANDIDATE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        acquire(
            args.command_plan.resolve(),
            args.output_dir.resolve(),
            args.phase_id,
            args.pre_dir.resolve(),
            args.acquisition_started_ns,
            contract_path=args.contract.resolve(),
            candidate_path=args.candidate.resolve(),
        )
        return 0
    except (
        common.ReadinessError,
        v2.EvidenceError,
        OSError,
        KeyError,
        ValueError,
    ) as error:
        print(f"A_ONLY_ACQUISITION_REFUSED: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
