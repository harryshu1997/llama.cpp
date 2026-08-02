#!/usr/bin/python3 -I
"""Self-contained fan-in for one CP0-R1 V2.3 A_ONLY acquisition."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import types
from typing import Any, Callable, Protocol


sys.dont_write_bytecode = True

PHASE = "A_ONLY"
MODEL_ID = "qwen3-14b-q4_k_m"
CLOCK_ID = time.CLOCK_MONOTONIC_RAW
MAX_INT = (1 << 63) - 1
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
PLAN_SCHEMA = "s39-cp0-r1-a-only-runtime-command-plan-v1"
JOINT_SCHEMA = "s39-cp0-r1-a-only-joint-phone-cuda-raw-v1"
MONOLITHIC_SCHEMA = "s39-cp0-r1-a-only-cuda-monolithic-raw-v1"
COMMON_ROW_KEYS = {"acquisition_id", "phase", "phase_id", "role"}
OUTPUT_FILES = {
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
RUNTIME_FILE = "runtime_identity.json"
RUNTIME_BUNDLE_FILE = "runtime_bundle_identity.json"
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
CAPTURED_FILE_FLAGS = {
    "--capture-plan",
    "--histories",
    "--launch-plan",
}
JOINT_KEYS = {
    "bridge_rows",
    "completed_ns",
    "cuda_memory_rows",
    "cuda_route_rows",
    "gpu_runtime",
    "mechanics_rows",
    "mechanism_commands_sha256",
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
    "runtime_processes",
    "schema",
    "started_ns",
}
MONOLITHIC_KEYS = {
    "completed_ns",
    "mechanism_commands_sha256",
    "model_id",
    "model_sha256",
    "oracle_cuda_monolithic_rows",
    "phase_id",
    "runtime_process",
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
RAW_RUNTIME_PROCESS_KEYS = {
    "boot_id",
    "bundle_id",
    "endpoint",
    "identity_probe_sha256",
    "launcher_path",
    "loaded_repo_component_ids",
    "observed_ns",
    "pid",
    "start_ticks",
    "system_dependencies",
}
RUNTIME_BUNDLE_PROCESS_KEYS = {
    *RAW_RUNTIME_PROCESS_KEYS,
    "bundle_sha256",
    "evidence_role",
    "evidence_sha256",
    "launcher_component_id",
}
SYSTEM_DEPENDENCY_KEYS = {
    "build_id",
    "ctime_ns",
    "device_id",
    "inode",
    "mode",
    "mtime_ns",
    "path",
    "size",
}
RUNTIME_BUNDLE_EVIDENCE_ROLES = {
    "cuda_monolithic": f"model.{MODEL_ID}.oracle.cuda_monolithic",
    "cuda_route": f"model.{MODEL_ID}.oracle.cuda_route",
    "op12_stagenet": f"model.{MODEL_ID}.placement.op12",
    "op15_direct_relay": f"model.{MODEL_ID}.route_transfer",
    "op15_stagenet": f"model.{MODEL_ID}.placement.op15",
}
RUNTIME_SYSTEM_ROOTS = {
    "cuda": ("/lib/", "/usr/lib/", "/usr/local/cuda/"),
    "op12": ("/apex/", "/system/", "/vendor/"),
    "op15": ("/apex/", "/system/", "/vendor/"),
}
EXECUTION_META_KEYS = {
    "acquisition_id",
    "backend",
    "call_shapes",
    "event_ns",
    "kind",
    "model_id",
    "model_sha256",
    "phase",
    "phase_id",
    "program_sha256",
    "role",
    "state_count_after",
    "state_count_before",
}
EXECUTION_REQUEST_KEYS = {
    "acquisition_id",
    "continuation_tokens",
    "event_ns",
    "input_tokens",
    "kind",
    "model_id",
    "model_sha256",
    "owner_after",
    "owner_before",
    "ownership_epoch_after",
    "ownership_epoch_before",
    "phase",
    "phase_id",
    "positions",
    "request_id",
    "role",
}
MEMORY_KEYS = {
    "acquisition_id",
    "batch",
    "clock_id",
    "completed_requests",
    "config_sha256",
    "device_name",
    "device_uuid",
    "event_ns",
    "free_bytes",
    "host_swap_used_bytes",
    "kind",
    "kv_buffer_bytes",
    "memory_total_bytes",
    "model_buffer_bytes",
    "model_id",
    "model_sha256",
    "phase",
    "phase_id",
    "placement_compute_nodes",
    "process_pid",
    "process_used_bytes",
    "role",
    "sample_id",
    "sampler_sha256",
    "state_count",
    "timestamp_ns",
    "used_bytes",
}
QUALITY_KEYS = {
    "acquisition_id",
    "corpus_item_sha256",
    "corpus_sha256",
    "event_ns",
    "item_index",
    "kind",
    "model_id",
    "model_sha256",
    "phase",
    "phase_id",
    "prompt_sha256",
    "raw_output",
    "role",
}
PLACEMENT_META_KEYS = {
    "acquisition_id",
    "available_after_bytes",
    "available_before_bytes",
    "batch",
    "boot_id",
    "device",
    "event_ns",
    "executed_layers",
    "kind",
    "model",
    "model_id",
    "model_sha256",
    "phase",
    "phase_id",
    "process_swap_bytes",
    "product",
    "role",
    "serial",
    "shard_sha256",
    "stored_layers",
    "system_swap_after_bytes",
    "system_swap_before_bytes",
}
PLACEMENT_NODE_KEYS = {
    "acquisition_id",
    "backend",
    "compute",
    "event_ns",
    "kind",
    "missing_buffer",
    "node_id",
    "op",
    "phase",
    "phase_id",
    "role",
}
CORPUS_DYNAMIC_KEYS = {
    "acquisition_id",
    "event_ns",
    "kind",
    "phase",
    "phase_id",
    "role",
}


class ReadinessError(ValueError):
    pass


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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}: expected {expected!r}, got {value!r}",
    )


def exact_keys(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    actual = set(value)
    require(
        actual == keys,
        f"E_KEYS: {field}: missing={sorted(keys - actual)}, "
        f"unknown={sorted(actual - keys)}",
    )
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(type(value) is int, f"E_TYPE: {field}")
    require(minimum <= value <= MAX_INT, f"E_RANGE: {field}")
    return value


def string(value: Any, field: str) -> str:
    require(type(value) is str and bool(value), f"E_TYPE: {field}")
    return value


def digest(value: Any, field: str) -> str:
    value = string(value, field)
    require(DIGEST_RE.fullmatch(value) is not None, f"E_DIGEST: {field}")
    return value


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ReadinessError(f"E_JSON_NUMBER: {value}")


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
        raise ReadinessError("E_CANONICAL") from error


def canonical_line(value: Any) -> bytes:
    return canonical_bytes(value)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def parse_json(raw: bytes, field: str) -> Any:
    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReadinessError(f"E_JSON: {field}: {error}") from error


def read_regular(path: Path, field: str) -> bytes:
    require(path.is_absolute(), f"E_PATH: {field}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ReadinessError(f"E_OPEN: {field}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"E_REGULAR: {field}")
        chunks = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.extend(chunk)
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
    require(identity(before) == identity(after), f"E_CHANGED: {field}")
    require(len(chunks) == before.st_size, f"E_SIZE: {field}")
    return bytes(chunks)


def read_canonical(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path, field)
    value = parse_json(raw, field)
    require(type(value) is dict, f"E_TYPE: {field}")
    require(canonical_bytes(value) == raw, f"E_CANONICAL: {field}")
    return value, raw


def parse_jsonl(path: Path, role: str, phase_id: str) -> tuple[list[dict[str, Any]], bytes]:
    raw = read_regular(path, role)
    require(bool(raw), f"E_ROWS: {role}")
    rows = []
    for index, line in enumerate(raw.splitlines(keepends=True)):
        row = parse_json(line, f"{role}[{index}]")
        require(type(row) is dict, f"E_TYPE: {role}[{index}]")
        require(canonical_line(row) == line, f"E_CANONICAL: {role}[{index}]")
        exact(row.get("role"), role, f"{role}[{index}].role")
        exact(row.get("phase"), PHASE, f"{role}[{index}].phase")
        exact(row.get("phase_id"), phase_id, f"{role}[{index}].phase_id")
        exact(row.get("acquisition_id"), phase_id, f"{role}[{index}].acquisition")
        rows.append(row)
    return rows, raw


def durable_write_new(path: Path, raw: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        mode,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def clock_ns() -> int:
    return time.clock_gettime_ns(CLOCK_ID)


def _relative(value: Any, field: str) -> str:
    value = string(value, field)
    path = Path(value)
    require(
        not path.is_absolute()
        and value not in (".", "..")
        and ".." not in path.parts,
        f"E_PATH: {field}",
    )
    return value


def _verify_self_contained_source(path: Path, field: str) -> None:
    try:
        source = read_regular(path, field).decode("ascii")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as error:
        raise ReadinessError(f"E_PRODUCER_SOURCE: {field}") from error
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            require(node.level == 0, f"E_PRODUCER_IMPORT: {field}")
            names = [node.module or ""]
        for name in names:
            root = name.split(".", 1)[0]
            require(
                root in sys.stdlib_module_names or root == "__future__",
                f"E_PRODUCER_IMPORT: {field}: {name}",
            )


def _validate_producer(value: Any, name: str) -> dict[str, Any]:
    field = f"plan.producers.{name}"
    value = exact_keys(value, PRODUCER_KEYS, field)
    template = value["argv_template"]
    require(
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
    exact(used, PRODUCER_PLACEHOLDERS, f"{field}.placeholders")
    require(
        all(item in PRODUCER_PLACEHOLDERS or "{" not in item for item in template),
        f"E_PLACEHOLDER: {field}.argv_template",
    )
    required_indexes = {0}
    for index, item in enumerate(template):
        if item in CAPTURED_FILE_FLAGS:
            require(index + 1 < len(template), f"E_EXECUTED_ARGUMENT: {field}[{index}]")
            file_index = index + 1
            require(
                template[file_index] not in PRODUCER_PLACEHOLDERS,
                f"E_EXECUTED_PLACEHOLDER: {field}[{file_index}]",
            )
            required_indexes.add(file_index)
    records = value["executed_files"]
    require(type(records) is list and bool(records), f"E_EXECUTED_FILES: {field}")
    indexes = []
    for index, source in enumerate(records):
        item = f"{field}.executed_files[{index}]"
        record = exact_keys(source, EXECUTED_FILE_KEYS, item)
        argv_index = integer(record["argv_index"], f"{item}.argv_index")
        require(argv_index < len(template), f"E_EXECUTED_INDEX: {item}")
        exact(template[argv_index], record["path"], f"{item}.argv")
        require(Path(record["path"]).is_absolute(), f"E_PATH: {item}")
        integer(record["bytes"], f"{item}.bytes", 1)
        digest(record["sha256"], f"{item}.sha256")
        indexes.append(argv_index)
    exact(indexes, sorted(set(indexes)), f"{field}.executed_file_order")
    exact(set(indexes), required_indexes, f"{field}.executed_file_indexes")
    _verify_self_contained_source(Path(template[0]), f"{field}.entrypoint")
    _relative(value["result_filename"], f"{field}.result_filename")
    timeout = integer(value["timeout_seconds"], f"{field}.timeout", 1)
    require(timeout <= 7200, f"E_RANGE: {field}.timeout")
    return value


def _validate_mechanism_commands(value: Any) -> dict[str, list[list[str]]]:
    value = exact_keys(value, {"desktop", "op12", "op15"}, "mechanism_commands")
    count = 0
    for endpoint in ("desktop", "op15", "op12"):
        commands = value[endpoint]
        require(type(commands) is list and bool(commands), f"E_COMMANDS: {endpoint}")
        for index, argv in enumerate(commands):
            require(
                type(argv) is list
                and bool(argv)
                and all(type(item) is str and bool(item) for item in argv),
                f"E_COMMAND: {endpoint}[{index}]",
            )
            require(
                all("{" not in item and "}" not in item for item in argv),
                f"E_COMMAND_PLACEHOLDER: {endpoint}[{index}]",
            )
            count += 1
    require(count >= 5, "E_MECHANISM_COMMAND_COUNT")
    return value


def _load_inputs(
    contract_path: Path,
    candidate_path: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes, dict[str, Any]]:
    contract, contract_raw = read_canonical(contract_path, "contract")
    candidate, candidate_raw = read_canonical(candidate_path, "candidate")
    exact(contract.get("schema"), "s39-cp0-r1-evidence-contract-v2.3", "contract.schema")
    exact(candidate.get("schema"), "s39-cp0-r1-candidate-v1", "candidate.schema")
    models = candidate.get("models")
    require(type(models) is list, "E_TYPE: candidate.models")
    matches = [
        item
        for item in models
        if type(item) is dict
        and item.get("slot") == "A"
        and item.get("model_id") == MODEL_ID
    ]
    require(len(matches) == 1, "E_MODEL_A")
    model = matches[0]
    exact(model.get("n_layer"), 40, "candidate.model.n_layer")
    exact(
        model.get("route_binding", {}).get("executed_cut_layer"),
        contract["incumbent_route_lock"]["cut_layer"],
        "candidate.model.cut",
    )
    return contract, contract_raw, candidate, candidate_raw, model


def load_plan(
    path: Path,
    contract_raw: bytes,
    candidate_raw: bytes,
    model: dict[str, Any],
) -> tuple[dict[str, Any], bytes]:
    plan, raw = read_canonical(path, "command_plan")
    exact_keys(plan, PLAN_KEYS, "plan")
    exact(plan["schema"], PLAN_SCHEMA, "plan.schema")
    exact(plan["phase"], PHASE, "plan.phase")
    exact(plan["model_id"], MODEL_ID, "plan.model")
    exact(plan["model_sha256"], model["artifact"]["sha256"], "plan.model_sha256")
    exact(plan["contract_sha256"], sha256_bytes(contract_raw), "plan.contract")
    exact(plan["candidate_sha256"], sha256_bytes(candidate_raw), "plan.candidate")
    expected_outputs = {
        **OUTPUT_FILES,
        "runtime_bundle_identity": RUNTIME_BUNDLE_FILE,
        "runtime_identity": RUNTIME_FILE,
    }
    exact(plan["outputs"], expected_outputs, "plan.outputs")
    _validate_mechanism_commands(plan["mechanism_commands"])
    producers = exact_keys(
        plan["producers"],
        {"cuda_monolithic", "joint_phone_cuda"},
        "plan.producers",
    )
    filenames = set(expected_outputs.values())
    for name in ("joint_phone_cuda", "cuda_monolithic"):
        producer = _validate_producer(producers[name], name)
        result = producer["result_filename"]
        require(result not in filenames, f"E_PATH_REUSE: {result}")
        filenames.add(result)
    return plan, raw


def _capture_file(
    path: Path,
    expected: dict[str, Any],
    destination: Path,
    mode: int,
) -> str:
    raw = read_regular(path, f"executed_file:{path}")
    exact(len(raw), expected["bytes"], f"E_EXECUTED_BYTES: {path}")
    exact(sha256_bytes(raw), expected["sha256"], f"E_EXECUTED_SHA256: {path}")
    durable_write_new(destination, raw, mode)
    return str(destination)


def capture_producers(
    plan: dict[str, Any],
    raw_dir: Path,
) -> dict[tuple[str, int], str]:
    captured = {}
    for name in ("joint_phone_cuda", "cuda_monolithic"):
        producer = plan["producers"][name]
        for record in producer["executed_files"]:
            argv_index = record["argv_index"]
            source = Path(record["path"])
            destination = (
                raw_dir
                / "executed"
                / name
                / f"{argv_index:03d}-{source.name}"
            )
            captured[(name, argv_index)] = _capture_file(
                source,
                record,
                destination,
                0o755 if argv_index == 0 else 0o644,
            )
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
        raise ReadinessError(f"E_PRODUCER_TIMEOUT: {name}") from error
    completed_ns = now_ns()
    durable_write_new(receipt_dir / f"{name}.stdout", completed.stdout)
    durable_write_new(receipt_dir / f"{name}.stderr", completed.stderr)
    durable_write_new(
        receipt_dir / f"{name}.receipt.json",
        canonical_bytes({
            "argv": source_argv,
            "completed_ns": completed_ns,
            "returncode": completed.returncode,
            "schema": "s39-cp0-r1-a-only-producer-receipt-v1",
            "started_ns": started_ns,
        }),
    )
    exact(completed.returncode, 0, f"E_PRODUCER_EXIT: {name}")
    exact(completed.stdout, b"", f"E_PRODUCER_STDOUT: {name}")
    exact(completed.stderr, b"", f"E_PRODUCER_STDERR: {name}")
    value, raw = read_canonical(result_path, f"producer.{name}")
    return value, raw, started_ns, completed_ns


def _source_interval(
    value: dict[str, Any],
    phase_id: str,
    schema: str,
    acquisition_started_ns: int,
    receipt_started_ns: int,
    receipt_completed_ns: int,
    mechanism_sha256: str,
    model_sha256: str,
    field: str,
) -> tuple[int, int]:
    exact(value["schema"], schema, f"{field}.schema")
    exact(value["phase_id"], phase_id, f"{field}.phase_id")
    exact(value["model_id"], MODEL_ID, f"{field}.model_id")
    exact(value["model_sha256"], model_sha256, f"{field}.model_sha256")
    exact(
        value["mechanism_commands_sha256"],
        mechanism_sha256,
        f"{field}.mechanism_commands",
    )
    started = integer(value["started_ns"], f"{field}.started_ns", 1)
    completed = integer(value["completed_ns"], f"{field}.completed_ns", 1)
    require(
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
    require(type(bodies) is list and bool(bodies), f"E_ROWS: {role}")
    rows = []
    previous = None
    for index, source in enumerate(bodies):
        field = f"{role}[{index}]"
        require(type(source) is dict, f"E_TYPE: {field}")
        require(not COMMON_ROW_KEYS.intersection(source), f"E_WRAPPER_FIELDS: {field}")
        event_ns = integer(source.get("event_ns"), f"{field}.event_ns", 1)
        require(started_ns <= event_ns <= completed_ns, f"E_EVENT_INTERVAL: {field}")
        if previous is not None:
            require(previous <= event_ns, f"E_EVENT_ORDER: {role}")
        previous = event_ns
        rows.append({
            "acquisition_id": phase_id,
            **copy.deepcopy(source),
            "phase": PHASE,
            "phase_id": phase_id,
            "role": role,
        })
    return rows


def _model_row(row: dict[str, Any], role: str, model_sha256: str, field: str) -> None:
    exact(row["role"], role, f"{field}.role")
    exact(row["model_id"], MODEL_ID, f"{field}.model")
    exact(row["model_sha256"], model_sha256, f"{field}.model_sha256")


def _validate_route(
    rows: list[dict[str, Any]],
    contract: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    require(len(rows) == 1, "E_ROUTE_ROWS")
    route = rows[0]
    expected = contract["incumbent_route_lock"]
    exact(route["kind"], "route_lock", "route.kind")
    exact(route["model_id"], MODEL_ID, "route.model")
    exact(route["model_sha256"], model["artifact"]["sha256"], "route.model_sha256")
    exact(route["n_layer"], model["n_layer"], "route.n_layer")
    geometry = contract["model_geometry"][MODEL_ID]
    for key in (
        "activation_dtype",
        "activation_element_bytes",
        "cuda_model_path",
        "hidden_size",
    ):
        exact(route[key], geometry[key], f"route.{key}")
    for key in (
        "backend",
        "cut_layer",
        "op12_shard_sha256",
        "op12_stored_layers",
        "op15_shard_sha256",
        "op15_stored_layers",
    ):
        exact(route[key], expected[key], f"route.{key}")
    for phone in ("op15", "op12"):
        known = geometry["known_shards"][phone]
        for suffix, key in (
            ("shard_bytes", "bytes"),
            ("shard_path", "path"),
            ("shard_sha256", "sha256"),
        ):
            exact(route[f"{phone}_{suffix}"], known[key], f"route.{phone}_{suffix}")
    return route


def _validate_corpus(
    rows: list[dict[str, Any]],
    contract: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[list[dict[str, Any]], bytes]:
    frozen = contract["quality_corpus"]
    exact(len(rows), frozen["items"], "corpus.items")
    content = []
    for index, row in enumerate(rows):
        exact(row["kind"], "item", f"corpus[{index}].kind")
        exact(row["item_index"], index, f"corpus[{index}].index")
        exact(row["dataset"], frozen["dataset"], f"corpus[{index}].dataset")
        exact(
            row["dataset_revision"],
            frozen["revision"],
            f"corpus[{index}].revision",
        )
        exact_keys(
            row,
            CORPUS_DYNAMIC_KEYS
            | {
                "choices",
                "dataset",
                "dataset_revision",
                "expected_answer",
                "item_index",
                "question",
                "source_row",
                "subject",
            },
            f"corpus[{index}]",
        )
        choices = row["choices"]
        require(
            type(choices) is list
            and len(choices) == 4
            and all(type(item) is str for item in choices),
            f"E_CORPUS_CHOICES: {index}",
        )
        exact(row["expected_answer"] in "ABCD", True, f"corpus[{index}].answer")
        content.append({
            key: copy.deepcopy(value)
            for key, value in row.items()
            if key not in CORPUS_DYNAMIC_KEYS
        })
    raw = b"".join(canonical_line(row) for row in content)
    exact(len(raw), frozen["bytes"], "corpus.bytes")
    exact(sha256_bytes(raw), frozen["sha256"], "corpus.sha256")
    task = candidate["task_suite"]
    exact(task["items"], len(rows), "candidate.task.items")
    exact(task["dataset"], frozen["dataset"], "candidate.task.dataset")
    exact(task["revision"], frozen["revision"], "candidate.task.revision")
    return content, raw


def _validate_call_shapes(value: Any, field: str) -> list[dict[str, Any]]:
    require(type(value) is list and len(value) == 9, f"E_CALL_SHAPES: {field}")
    for index, call in enumerate(value):
        exact_keys(call, {"call_index", "n_seqs", "n_tokens", "phase"}, f"{field}[{index}]")
        exact(call["call_index"], index, f"{field}[{index}].index")
        exact(call["n_seqs"], 8, f"{field}[{index}].n_seqs")
        if index == 0:
            exact(call["phase"], "prefill", f"{field}[0].phase")
            exact(call["n_tokens"], 16, f"{field}[0].n_tokens")
        else:
            exact(call["phase"], "decode", f"{field}[{index}].phase")
            exact(call["n_tokens"], 8, f"{field}[{index}].n_tokens")
    return value


def _int_list(value: Any, field: str, length: int | None = None) -> list[int]:
    require(
        type(value) is list and all(type(item) is int and 0 <= item <= MAX_INT for item in value),
        f"E_TYPE: {field}",
    )
    if length is not None:
        exact(len(value), length, f"{field}.length")
    return value


def _validate_execution(
    rows: list[dict[str, Any]],
    role: str,
    model_sha256: str,
    backend: str,
    owner: str,
) -> dict[str, Any]:
    require(len(rows) == 9, f"E_EXECUTION_ROWS: {role}")
    meta = exact_keys(rows[0], EXECUTION_META_KEYS, f"{role}.meta")
    exact(meta["kind"], "meta", f"{role}.meta.kind")
    _model_row(meta, role, model_sha256, f"{role}.meta")
    exact(meta["backend"], backend, f"{role}.meta.backend")
    digest(meta["program_sha256"], f"{role}.meta.program")
    exact(meta["state_count_before"], 0, f"{role}.state_before")
    exact(meta["state_count_after"], 0, f"{role}.state_after")
    calls = _validate_call_shapes(meta["call_shapes"], f"{role}.call_shapes")
    requests = {}
    for index, row in enumerate(rows[1:]):
        field = f"{role}.request[{index}]"
        exact_keys(row, EXECUTION_REQUEST_KEYS, field)
        exact(row["kind"], "request", f"{field}.kind")
        _model_row(row, role, model_sha256, field)
        exact(row["request_id"], index, f"{field}.request_id")
        inputs = _int_list(row["input_tokens"], f"{field}.input_tokens")
        positions = _int_list(row["positions"], f"{field}.positions", len(inputs))
        require(bool(inputs), f"E_INPUT_EMPTY: {field}")
        exact(
            positions,
            list(range(positions[0], positions[0] + len(positions))),
            f"{field}.positions",
        )
        continuations = _int_list(
            row["continuation_tokens"],
            f"{field}.continuations",
            8,
        )
        exact(row["owner_before"], owner, f"{field}.owner_before")
        exact(row["owner_after"], "RELEASED", f"{field}.owner_after")
        exact(row["ownership_epoch_before"], 1, f"{field}.epoch_before")
        exact(row["ownership_epoch_after"], 2, f"{field}.epoch_after")
        requests[index] = {
            "continuation_tokens": continuations,
            "input_tokens": inputs,
            "positions": positions,
            "row": row,
        }
    exact(sum(len(item["input_tokens"]) for item in requests.values()), 16, f"{role}.prefill")
    return {"calls": calls, "meta": meta, "requests": requests}


def _normalized(row: dict[str, Any], extras: set[str] | None = None) -> dict[str, Any]:
    result = copy.deepcopy(row)
    for key in ("event_ns", "phase", "phase_id"):
        result.pop(key, None)
    for key in extras or set():
        result.pop(key)
    return result


def _validate_memory(
    rows: list[dict[str, Any]],
    role: str,
    model_sha256: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    require(len(rows) == 3, f"E_MEMORY_ROWS: {role}")
    exact([row["kind"] for row in rows], ["before", "ready", "after"], f"{role}.kinds")
    sampler = None
    for index, row in enumerate(rows):
        field = f"{role}[{index}]"
        exact_keys(row, MEMORY_KEYS, field)
        _model_row(row, role, model_sha256, field)
        exact(row["clock_id"], "HOST_MONOTONIC_RAW", f"{field}.clock")
        exact(row["event_ns"], row["timestamp_ns"], f"{field}.time")
        exact(row["device_name"], contract["devices"]["cuda"]["name"], f"{field}.device")
        exact(row["device_uuid"], contract["devices"]["cuda"]["uuid"], f"{field}.uuid")
        for key in (
            "batch",
            "completed_requests",
            "free_bytes",
            "host_swap_used_bytes",
            "kv_buffer_bytes",
            "memory_total_bytes",
            "model_buffer_bytes",
            "placement_compute_nodes",
            "process_pid",
            "process_used_bytes",
            "state_count",
            "used_bytes",
        ):
            integer(row[key], f"{field}.{key}")
        exact(
            row["memory_total_bytes"],
            row["used_bytes"] + row["free_bytes"],
            f"{field}.accounting",
        )
        exact(
            row["memory_total_bytes"],
            contract["devices"]["cuda"]["memory_total_bytes"],
            f"{field}.memory_total",
        )
        digest(row["sampler_sha256"], f"{field}.sampler")
        if sampler is None:
            sampler = row["sampler_sha256"]
        exact(row["sampler_sha256"], sampler, f"{field}.sampler")
    before, ready, after = rows
    exact(ready["batch"], 8, f"{role}.ready.batch")
    exact(ready["completed_requests"], 8, f"{role}.ready.completed")
    exact(ready["state_count"], 8, f"{role}.ready.state_count")
    require(ready["process_pid"] > 0, f"E_MEMORY_PID: {role}")
    require(ready["placement_compute_nodes"] > 0, f"E_MEMORY_PLACEMENT: {role}")
    exact(
        ready["process_used_bytes"],
        ready["model_buffer_bytes"] + ready["kv_buffer_bytes"],
        f"{role}.ready.process_accounting",
    )
    require(
        ready["free_bytes"] >= contract["gates"]["cuda_minimum_free_bytes"],
        f"E_CUDA_HEADROOM: {role}",
    )
    exact(
        after["host_swap_used_bytes"] - before["host_swap_used_bytes"],
        0,
        f"{role}.swap_growth",
    )
    return {"ready": ready, "before": before, "after": after}


def _prompt_sha(item: dict[str, Any], candidate: dict[str, Any]) -> str:
    prompt = candidate["task_suite"]["prompt_format"].format(
        question=item["question"],
        choice0=item["choices"][0],
        choice1=item["choices"][1],
        choice2=item["choices"][2],
        choice3=item["choices"][3],
    )
    return sha256_bytes(prompt.encode("utf-8"))


def _answer(raw: str) -> str | None:
    match = re.match(r"^([A-D])(?:\b|$)", raw.lstrip())
    return None if match is None else match.group(1)


def _validate_quality(
    rows: list[dict[str, Any]],
    role: str,
    model_sha256: str,
    corpus_rows: list[dict[str, Any]],
    corpus_wrapped: list[dict[str, Any]],
    corpus_role_raw: bytes,
    candidate: dict[str, Any],
) -> int:
    exact(len(rows), 64, f"{role}.count")
    correct = 0
    for index, row in enumerate(rows):
        field = f"{role}[{index}]"
        exact_keys(row, QUALITY_KEYS, field)
        _model_row(row, role, model_sha256, field)
        exact(row["kind"], "output", f"{field}.kind")
        exact(row["item_index"], index, f"{field}.index")
        exact(row["corpus_sha256"], sha256_bytes(corpus_role_raw), f"{field}.corpus")
        exact(
            row["corpus_item_sha256"],
            digest_json(_normalized(corpus_wrapped[index])),
            f"{field}.corpus_item",
        )
        exact(row["prompt_sha256"], _prompt_sha(corpus_rows[index], candidate), f"{field}.prompt")
        raw_output = string(row["raw_output"], f"{field}.raw_output")
        if _answer(raw_output) == corpus_rows[index]["expected_answer"]:
            correct += 1
    return correct


def _validate_bridge(
    rows: list[dict[str, Any]],
    role: str,
    model_sha256: str,
    phone: dict[str, Any],
    memory: dict[str, Any],
) -> dict[str, Any]:
    require(len(rows) == 10, f"E_BRIDGE_ROWS: {role}")
    exact(rows[0]["kind"], "cuda_load_start", f"{role}[0].kind")
    exact(rows[-1]["kind"], "cuda_ready", f"{role}[-1].kind")
    publications = rows[1:-1]
    exact(
        [row["kind"] for row in publications],
        ["phone_publication_received"] * 8,
        f"{role}.publications",
    )
    for index, row in enumerate(rows):
        field = f"{role}[{index}]"
        _model_row(row, role, model_sha256, field)
        exact(row["clock_id"], "HOST_MONOTONIC_RAW", f"{field}.clock")
        exact(row["event_ns"], row["timestamp_ns"], f"{field}.time")
    ready = rows[-1]
    ready_normalized = _normalized(
        memory["ready"],
        {"process_pid", "process_used_bytes", "sample_id", "sampler_sha256"},
    )
    exact(
        ready["cuda_memory_ready_sha256"],
        digest_json(ready_normalized),
        f"{role}.ready_link",
    )
    for request_id, row in enumerate(publications):
        field = f"{role}.publication[{request_id}]"
        exact(row["request_id"], request_id, f"{field}.request_id")
        request = phone["requests"][request_id]
        exact(row["token_ids"], request["continuation_tokens"], f"{field}.tokens")
        exact(
            row["phone_request_sha256"],
            digest_json(_normalized(request["row"])),
            f"E_BRIDGE_LINK: {request_id}",
        )
        require(
            request["row"]["event_ns"] <= row["timestamp_ns"] < ready["timestamp_ns"],
            f"E_BRIDGE_CAUSAL_ORDER: {request_id}",
        )
    return {"cuda_ready_ns": ready["timestamp_ns"], "publications": 8}


def _range(value: Any, field: str) -> list[int]:
    require(type(value) is list and len(value) == 2, f"E_TYPE: {field}")
    start = integer(value[0], f"{field}[0]")
    end = integer(value[1], f"{field}[1]", 1)
    require(start < end, f"E_RANGE: {field}")
    return value


def _validate_placement(
    rows: list[dict[str, Any]],
    role: str,
    phone: str,
    model_sha256: str,
    route: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    require(len(rows) >= 2, f"E_PLACEMENT_ROWS: {role}")
    meta = exact_keys(rows[0], PLACEMENT_META_KEYS, f"{role}.meta")
    exact(meta["kind"], "meta", f"{role}.meta.kind")
    _model_row(meta, role, model_sha256, f"{role}.meta")
    identity = contract["devices"][phone]
    for key in ("serial", "model", "product", "device"):
        exact(meta[key], identity[key], f"{role}.meta.{key}")
    exact(meta["batch"], 8, f"{role}.meta.batch")
    exact(meta["stored_layers"], route[f"{phone}_stored_layers"], f"{role}.stored")
    exact(meta["shard_sha256"], route[f"{phone}_shard_sha256"], f"{role}.shard")
    expected_executed = (
        [0, route["cut_layer"]]
        if phone == "op15"
        else [route["cut_layer"], route["n_layer"]]
    )
    exact(meta["executed_layers"], expected_executed, f"{role}.executed")
    _range(meta["stored_layers"], f"{role}.stored")
    _range(meta["executed_layers"], f"{role}.executed")
    exact(meta["process_swap_bytes"], 0, f"{role}.process_swap")
    require(
        meta["system_swap_after_bytes"] - meta["system_swap_before_bytes"] <= 0,
        f"E_PHONE_SWAP_GROWTH: {role}",
    )
    require(
        meta["available_after_bytes"] >= contract["gates"]["phone_minimum_available_bytes"],
        f"E_PHONE_HEADROOM: {role}",
    )
    gpu_nodes = 0
    cpu_ops = []
    seen_nodes = set()
    for index, row in enumerate(rows[1:]):
        field = f"{role}.node[{index}]"
        exact_keys(row, PLACEMENT_NODE_KEYS, field)
        exact(row["kind"], "node", f"{field}.kind")
        exact(row["compute"], True, f"{field}.compute")
        exact(row["missing_buffer"], False, f"{field}.missing_buffer")
        node_id = integer(row["node_id"], f"{field}.node_id")
        require(node_id not in seen_nodes, f"E_NODE_REUSE: {field}")
        seen_nodes.add(node_id)
        backend = string(row["backend"], f"{field}.backend")
        op = string(row["op"], f"{field}.op")
        if backend == route["backend"]:
            gpu_nodes += 1
        elif backend == "CPU":
            cpu_ops.append(op)
        else:
            raise ReadinessError(f"E_PLACEMENT_BACKEND: {field}: {backend}")
    require(gpu_nodes > 0, f"E_PLACEMENT_COMPUTE: {role}")
    require(all(op == "GET_ROWS" for op in cpu_ops), f"E_CPU_FALLBACK: {role}")
    return {"boot_id": meta["boot_id"], "gpu_nodes": gpu_nodes}


def _validate_transfer(
    rows: list[dict[str, Any]],
    role: str,
    model_sha256: str,
    route: dict[str, Any],
    calls: list[dict[str, Any]],
) -> int:
    require(len(rows) == 10, f"E_TRANSFER_ROWS: {role}")
    meta = rows[0]
    exact(
        set(meta),
        COMMON_ROW_KEYS
        | {
            "batch",
            "cut_layer",
            "event_ns",
            "kind",
            "model_id",
            "model_sha256",
            "request_ids",
        },
        f"{role}.meta.keys",
    )
    _model_row(meta, role, model_sha256, f"{role}.meta")
    exact(meta["kind"], "meta", f"{role}.meta.kind")
    exact(meta["batch"], 8, f"{role}.meta.batch")
    exact(meta["cut_layer"], route["cut_layer"], f"{role}.meta.cut")
    exact(meta["request_ids"], list(range(8)), f"{role}.meta.requests")
    total = 0
    transfer_keys = COMMON_ROW_KEYS | {
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
    }
    for index, (row, call) in enumerate(zip(rows[1:], calls)):
        field = f"{role}.transfer[{index}]"
        exact_keys(row, transfer_keys, field)
        exact(row["kind"], "transfer", f"{field}.kind")
        exact(row["call_index"], index, f"{field}.call_index")
        exact(row["sender"], "op15", f"{field}.sender")
        exact(row["receiver"], "op12", f"{field}.receiver")
        exact(row["path"], "WIFI_TCP_DIRECT", f"{field}.path")
        exact(row["host_payload_bytes"], 0, f"{field}.host_bytes")
        exact(row["row_count"], call["n_tokens"], f"{field}.rows")
        expected = (
            call["n_tokens"]
            * route["hidden_size"]
            * route["activation_element_bytes"]
        )
        exact(row["payload_bytes"], expected, f"{field}.payload_bytes")
        digest(row["payload_sha256"], f"{field}.payload_sha256")
        total += expected
    return total


def _validate_roles(
    rows: dict[str, list[dict[str, Any]]],
    model: dict[str, Any],
    route: dict[str, Any],
    contract: dict[str, Any],
    candidate: dict[str, Any],
    corpus_rows: list[dict[str, Any]],
    corpus_wrapped: list[dict[str, Any]],
    corpus_role_raw: bytes,
) -> dict[str, Any]:
    prefix = f"model.{MODEL_ID}"
    model_sha = model["artifact"]["sha256"]
    phone = _validate_execution(
        rows[f"{prefix}.mechanics.phone"],
        f"{prefix}.mechanics.phone",
        model_sha,
        "PHONE_COLLECTIVE",
        "PHONE",
    )
    cuda_route = _validate_execution(
        rows[f"{prefix}.oracle.cuda_route"],
        f"{prefix}.oracle.cuda_route",
        model_sha,
        "CUDA0",
        "CUDA",
    )
    cuda_monolithic = _validate_execution(
        rows[f"{prefix}.oracle.cuda_monolithic"],
        f"{prefix}.oracle.cuda_monolithic",
        model_sha,
        "CUDA0",
        "CUDA",
    )
    exact(cuda_route["calls"], cuda_monolithic["calls"], "E_ORACLE_CALL_GEOMETRY")
    require(
        cuda_route["meta"]["program_sha256"]
        != cuda_monolithic["meta"]["program_sha256"],
        "E_ORACLE_PROGRAM_INDEPENDENCE",
    )
    for request_id in range(8):
        route_request = cuda_route["requests"][request_id]
        mono_request = cuda_monolithic["requests"][request_id]
        phone_request = phone["requests"][request_id]
        for key in ("input_tokens", "positions"):
            exact(route_request[key], mono_request[key], f"E_ORACLE_{key}: {request_id}")
            exact(route_request[key], phone_request[key], f"E_PHONE_{key}: {request_id}")
        exact(
            route_request["continuation_tokens"],
            mono_request["continuation_tokens"],
            f"E_ORACLE_CONTINUATION: {request_id}",
        )
    memory = _validate_memory(
        rows[f"{prefix}.cuda_memory"],
        f"{prefix}.cuda_memory",
        model_sha,
        contract,
    )
    cuda_correct = _validate_quality(
        rows[f"{prefix}.quality.cuda"],
        f"{prefix}.quality.cuda",
        model_sha,
        corpus_rows,
        corpus_wrapped,
        corpus_role_raw,
        candidate,
    )
    phone_correct = _validate_quality(
        rows[f"{prefix}.quality.phone"],
        f"{prefix}.quality.phone",
        model_sha,
        corpus_rows,
        corpus_wrapped,
        corpus_role_raw,
        candidate,
    )
    require(
        cuda_correct >= contract["gates"]["cuda_quality_minimum_correct_items"],
        "E_CUDA_QUALITY_FLOOR",
    )
    require(
        cuda_correct - phone_correct
        <= contract["gates"]["quality_maximum_score_regression_items"],
        "E_PHONE_QUALITY_NONINFERIORITY",
    )
    bridge = _validate_bridge(
        rows[f"{prefix}.bridge"],
        f"{prefix}.bridge",
        model_sha,
        phone,
        memory,
    )
    placement = {
        phone_name: _validate_placement(
            rows[f"{prefix}.placement.{phone_name}"],
            f"{prefix}.placement.{phone_name}",
            phone_name,
            model_sha,
            route,
            contract,
        )
        for phone_name in ("op15", "op12")
    }
    transfer_bytes = _validate_transfer(
        rows[f"{prefix}.route_transfer"],
        f"{prefix}.route_transfer",
        model_sha,
        route,
        phone["calls"],
    )
    return {
        "bridge": bridge,
        "cuda_correct": cuda_correct,
        "phone_correct": phone_correct,
        "placement": placement,
        "transfer_bytes": transfer_bytes,
    }


def _stat_record(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(
        value,
        {"ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"},
        field,
    )
    for key in value:
        integer(value[key], f"{field}.{key}")
    require(value["inode"] > 0 and value["size"] > 0, f"E_STAT: {field}")
    return value


def _load_legacy_readiness(
    pre_dir: Path,
    phase_id: str,
    acquisition_started_ns: int,
    contract: dict[str, Any],
    model: dict[str, Any],
    route: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], bytes]:
    artifact, artifact_raw = read_canonical(
        pre_dir.parent / "artifact" / "artifact_snapshot.json",
        "artifact_snapshot",
    )
    exact(artifact["schema"], "s39-cp0-r1-artifact-snapshot-v2.3", "artifact.schema")
    exact(artifact["phase"], PHASE, "artifact.phase")
    exact(artifact["slot"], "A", "artifact.slot")
    exact(artifact["model_id"], MODEL_ID, "artifact.model")
    require(artifact["completed_ns"] < acquisition_started_ns, "E_ARTIFACT_PRELOCK")
    records = artifact["artifacts"]
    require(type(records) is list and len(records) == 5, "E_ARTIFACT_COUNT")
    artifacts = {}
    for index, record in enumerate(records):
        field = f"artifact[{index}]"
        exact_keys(record, {"bytes", "endpoint", "path", "sha256", "stat"}, field)
        endpoint = string(record["endpoint"], f"{field}.endpoint")
        require(endpoint not in artifacts, f"E_ARTIFACT_ENDPOINT_REUSE: {endpoint}")
        digest(record["sha256"], f"{field}.sha256")
        integer(record["bytes"], f"{field}.bytes", 1)
        _stat_record(record["stat"], f"{field}.stat")
        exact(record["bytes"], record["stat"]["size"], f"{field}.size")
        artifacts[endpoint] = record
    exact(
        set(artifacts),
        {"cuda", "op12", "op12_worker", "op15", "op15_worker"},
        "artifact.endpoints",
    )
    expected = {
        "cuda": (route["cuda_model_path"], model["artifact"]["sha256"]),
        "op12": (route["op12_shard_path"], route["op12_shard_sha256"]),
        "op15": (route["op15_shard_path"], route["op15_shard_sha256"]),
    }
    for endpoint, (path, sha) in expected.items():
        exact(artifacts[endpoint]["path"], path, f"artifact.{endpoint}.path")
        exact(artifacts[endpoint]["sha256"], sha, f"artifact.{endpoint}.sha256")

    lock, lock_raw = read_canonical(
        pre_dir.parent / "fresh" / "readiness_lock.json",
        "readiness_lock",
    )
    exact(lock["schema"], "s39-cp0-r1-readiness-lock-v2.3", "readiness_lock.schema")
    exact(lock["phase"], PHASE, "readiness_lock.phase")
    exact(lock["phase_id"], phase_id, "readiness_lock.phase_id")
    exact(lock["artifact_snapshot_sha256"], sha256_bytes(artifact_raw), "readiness_lock.artifact")
    require(lock["event_ns"] < acquisition_started_ns, "E_READINESS_LOCK_ORDER")

    fresh, fresh_raw = read_canonical(
        pre_dir.parent / "fresh" / "fresh_snapshot.json",
        "fresh_snapshot",
    )
    exact(fresh["schema"], "s39-cp0-r1-fresh-identity-v2.3", "fresh.schema")
    exact(fresh["phase"], PHASE, "fresh.phase")
    exact(fresh["phase_id"], phase_id, "fresh.phase_id")
    exact(fresh["readiness_lock_sha256"], sha256_bytes(lock_raw), "fresh.lock")
    require(
        lock["event_ns"] <= fresh["started_ns"] < fresh["completed_ns"]
        < acquisition_started_ns,
        "E_FRESH_INTERVAL",
    )
    cuda = fresh["cuda"]
    for key in ("host", "memory_total_bytes", "name", "uuid"):
        exact(cuda[key], contract["readiness_v2_3"]["cuda_identity"][key], f"fresh.cuda.{key}")
    for phone in ("op15", "op12"):
        value = fresh["phones"][phone]
        for key in ("serial", "model", "product", "device"):
            exact(
                value[key],
                contract["readiness_v2_3"]["phone_identity"][phone][key],
                f"fresh.{phone}.{key}",
            )
        exact(value["swap_used_bytes"], 0, f"E_PHONE_SWAP: {phone}")
        require(
            value["available_bytes"]
            >= contract["readiness_v2_3"]["phone_minimum_available_bytes"],
            f"E_PHONE_HEADROOM: fresh.{phone}",
        )
    stats = {}
    for record in fresh["artifact_stats"]:
        endpoint = record["endpoint"]
        require(endpoint not in stats, f"E_FRESH_ARTIFACT_REUSE: {endpoint}")
        _stat_record(record["stat"], f"fresh.artifact.{endpoint}")
        stats[endpoint] = record
    exact(set(stats), set(artifacts), "fresh.artifact.endpoints")
    for endpoint in artifacts:
        exact(stats[endpoint]["path"], artifacts[endpoint]["path"], f"fresh.{endpoint}.path")
        exact(stats[endpoint]["stat"], artifacts[endpoint]["stat"], f"fresh.{endpoint}.stat")
    return artifacts, fresh, fresh_raw


def _phone_executor(
    source: Any,
    phone: str,
    route_epoch: int,
    model: dict[str, Any],
    fresh_phone: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    role_digests: dict[str, str],
    transfer_bytes: int,
) -> dict[str, Any]:
    source = exact_keys(source, PHONE_RUNTIME_KEYS, f"runtime.{phone}")
    exact(source["model_id"], MODEL_ID, f"runtime.{phone}.model")
    exact(source["route_epoch"], route_epoch, f"runtime.{phone}.epoch")
    exact(source["serial"], fresh_phone["serial"], f"runtime.{phone}.serial")
    exact(source["boot_id"], fresh_phone["boot_id"], f"runtime.{phone}.boot")
    exact(
        source["worker_executable_path"],
        artifacts[f"{phone}_worker"]["path"],
        f"runtime.{phone}.worker_path",
    )
    exact(
        source["loaded_shard_path"],
        artifacts[phone]["path"],
        f"runtime.{phone}.shard_path",
    )
    exact(source["worker_model_sha256"], model["artifact"]["sha256"], f"runtime.{phone}.model_sha")
    exact(source["active_sequences_after_cleanup"], 0, f"runtime.{phone}.cleanup")
    exact(source["process_swap_bytes"], 0, f"runtime.{phone}.swap")
    require(source["available_bytes"] >= 536870912, f"E_RUNTIME_HEADROOM: {phone}")
    integer(source["worker_pid"], f"runtime.{phone}.pid", 1)
    integer(source["worker_start_ticks"], f"runtime.{phone}.start_ticks", 1)
    exact(source["session_protocol_version"], 2, f"runtime.{phone}.protocol")
    nonce = string(source["worker_boot_nonce"], f"runtime.{phone}.nonce")
    require(len(nonce) == 16 and all(ch in "0123456789abcdef" for ch in nonce), f"E_RUNTIME_NONCE: {phone}")
    before = exact_keys(
        source["interface_before"],
        {"interface", "rx_bytes", "tx_bytes"},
        f"runtime.{phone}.interface_before",
    )
    after = exact_keys(
        source["interface_after"],
        {"interface", "rx_bytes", "tx_bytes"},
        f"runtime.{phone}.interface_after",
    )
    exact(after["interface"], before["interface"], f"runtime.{phone}.interface")
    for key in ("rx_bytes", "tx_bytes"):
        integer(before[key], f"runtime.{phone}.before.{key}")
        integer(after[key], f"runtime.{phone}.after.{key}")
        require(after[key] >= before[key], f"E_INTERFACE_COUNTER: {phone}.{key}")
    peer = exact_keys(
        source["direct_peer"],
        {"interface", "local_ipv4", "peer_ipv4", "socket_peer_observed"},
        f"runtime.{phone}.peer",
    )
    exact(peer["interface"], before["interface"], f"runtime.{phone}.peer.interface")
    exact(peer["socket_peer_observed"], True, f"runtime.{phone}.peer.observed")
    if phone == "op15":
        require(after["tx_bytes"] - before["tx_bytes"] >= transfer_bytes, "E_INTERFACE_TRANSFER_BYTES: op15.tx")
    else:
        require(after["rx_bytes"] - before["rx_bytes"] >= transfer_bytes, "E_INTERFACE_TRANSFER_BYTES: op12.rx")
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
    transfer_bytes: int,
) -> dict[str, Any]:
    route_epoch = integer(joint["route_epoch"], "joint.route_epoch", 1)
    gpu = exact_keys(joint["gpu_runtime"], GPU_RUNTIME_KEYS, "joint.gpu_runtime")
    exact(gpu["model_id"], MODEL_ID, "joint.gpu_runtime.model")
    exact(gpu["route_epoch"], route_epoch, "joint.gpu_runtime.epoch")
    exact(gpu["gpu_uuid"], fresh["cuda"]["uuid"], "joint.gpu_runtime.uuid")
    exact(gpu["host_boot_id"], fresh["cuda"]["host_boot_id"], "joint.gpu_runtime.boot")
    exact(gpu["artifact_path"], artifacts["cuda"]["path"], "joint.gpu_runtime.path")
    gpu_final = {
        **copy.deepcopy(gpu),
        "artifact_sha256": artifacts["cuda"]["sha256"],
        "artifact_stat": copy.deepcopy(artifacts["cuda"]["stat"]),
        "executor_id": "GPU",
    }
    phones = [
        _phone_executor(
            joint[f"{phone}_runtime"],
            phone,
            route_epoch,
            model,
            fresh["phones"][phone],
            artifacts,
            role_digests,
            transfer_bytes,
        )
        for phone in ("op15", "op12")
    ]
    exact(
        phones[0]["direct_peer"]["peer_ipv4"],
        phones[1]["direct_peer"]["local_ipv4"],
        "E_DIRECT_PEER: op15->op12",
    )
    exact(
        phones[1]["direct_peer"]["peer_ipv4"],
        phones[0]["direct_peer"]["local_ipv4"],
        "E_DIRECT_PEER: op12->op15",
    )
    return {
        "completed_ns": joint["completed_ns"],
        "executors": [gpu_final, *phones],
        "fresh_snapshot_sha256": sha256_bytes(fresh_raw),
        "phase": PHASE,
        "phase_id": phase_id,
        "route_epoch": route_epoch,
        "schema": "s39-cp0-r1-runtime-identity-v2.3",
        "started_ns": joint["started_ns"],
    }


def _load_runtime_bundle_overlay(
    pre_dir: Path,
    phase_id: str,
    acquisition_started_ns: int,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes, dict[str, Any], bytes]:
    snapshot, snapshot_raw = read_canonical(
        pre_dir.parent / "artifact" / "runtime_bundle_snapshot.json",
        "runtime_bundle_snapshot",
    )
    lock, lock_raw = read_canonical(
        pre_dir.parent / "fresh" / "runtime_bundle_readiness_lock.json",
        "runtime_bundle_readiness_lock",
    )
    fresh, fresh_raw = read_canonical(
        pre_dir.parent / "fresh" / "runtime_bundle_fresh.json",
        "runtime_bundle_fresh",
    )
    exact(snapshot["schema"], "s39-cp0-r1-runtime-bundle-snapshot-v1", "bundle_snapshot.schema")
    exact(snapshot["phase"], PHASE, "bundle_snapshot.phase")
    exact(snapshot["model_id"], MODEL_ID, "bundle_snapshot.model")
    exact(snapshot["slot"], "A", "bundle_snapshot.slot")
    require(snapshot["completed_ns"] < acquisition_started_ns, "E_BUNDLE_SNAPSHOT_ORDER")
    exact(lock["schema"], "s39-cp0-r1-runtime-bundle-readiness-lock-v1", "bundle_lock.schema")
    exact(lock["phase"], PHASE, "bundle_lock.phase")
    exact(lock["phase_id"], phase_id, "bundle_lock.phase_id")
    exact(lock["runtime_bundle_snapshot_sha256"], sha256_bytes(snapshot_raw), "bundle_lock.snapshot")
    exact(fresh["schema"], "s39-cp0-r1-runtime-bundle-fresh-v1", "bundle_fresh.schema")
    exact(fresh["phase"], PHASE, "bundle_fresh.phase")
    exact(fresh["phase_id"], phase_id, "bundle_fresh.phase_id")
    exact(fresh["readiness_lock_sha256"], sha256_bytes(lock_raw), "bundle_fresh.lock")
    exact(fresh["runtime_bundle_snapshot_sha256"], sha256_bytes(snapshot_raw), "bundle_fresh.snapshot")
    exact(
        fresh["runtime_bundle_plan_sha256"],
        snapshot["runtime_bundle_plan_sha256"],
        "bundle_fresh.plan",
    )
    require(
        lock["event_ns"] <= fresh["started_ns"] < fresh["completed_ns"]
        < acquisition_started_ns,
        "E_BUNDLE_FRESH_INTERVAL",
    )
    bundle_ids = [row["bundle_id"] for row in snapshot["runtime_bundles"]]
    exact(
        bundle_ids,
        [
            "cuda_monolithic",
            "cuda_route",
            "op12_stagenet",
            "op15_direct_relay",
            "op15_stagenet",
        ],
        "runtime_bundles",
    )
    component_ids = [row["component_id"] for row in snapshot["runtime_components"]]
    require(component_ids == sorted(set(component_ids)), "E_RUNTIME_COMPONENT_ORDER")
    fresh_ids = [row["component_id"] for row in fresh["runtime_component_stats"]]
    exact(fresh_ids, component_ids, "runtime_component_stats")
    artifact_components = {row["component_id"]: row for row in snapshot["runtime_components"]}
    for record in fresh["runtime_component_stats"]:
        expected = artifact_components[record["component_id"]]
        for key in ("bundle_id", "endpoint", "path"):
            exact(record[key], expected[key], f"runtime_component.{record['component_id']}.{key}")
        exact(record["stat"], expected["stat"], f"runtime_component.{record['component_id']}.stat")
    return snapshot, snapshot_raw, lock, lock_raw, fresh, fresh_raw


def _validate_system_dependencies(
    value: Any,
    endpoint: str,
    field: str,
) -> list[dict[str, Any]]:
    require(type(value) is list and bool(value), f"E_SYSTEM_DEPS: {field}")
    previous_path = None
    for index, dependency in enumerate(value):
        item = f"{field}[{index}]"
        exact_keys(dependency, SYSTEM_DEPENDENCY_KEYS, item)
        path = string(dependency["path"], f"{item}.path")
        require(
            any(path.startswith(root) for root in RUNTIME_SYSTEM_ROOTS[endpoint]),
            f"E_SYSTEM_DEP_PATH: {item}",
        )
        if previous_path is not None:
            require(previous_path < path, f"E_SYSTEM_DEP_ORDER: {field}")
        previous_path = path
        for key in ("ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"):
            integer(dependency[key], f"{item}.{key}")
        build_id = dependency["build_id"]
        require(
            build_id is None or (type(build_id) is str and bool(build_id)),
            f"E_SYSTEM_BUILD_ID: {item}",
        )
    return value


def _build_runtime_bundle_identity(
    joint: dict[str, Any],
    monolithic: dict[str, Any],
    runtime: dict[str, Any],
    runtime_snapshot: dict[str, Any],
    runtime_snapshot_raw: bytes,
    runtime_fresh_raw: bytes,
    role_digests: dict[str, str],
    phase_id: str,
    identity_started: int,
    identity_completed: int,
) -> dict[str, Any]:
    require(identity_started < identity_completed, "E_RUNTIME_BUNDLE_INTERVAL")
    bundles = {}
    for index, value in enumerate(runtime_snapshot["runtime_bundles"]):
        field = f"runtime_bundles[{index}]"
        exact_keys(
            value,
            {
                "bundle_id",
                "bundle_sha256",
                "endpoint",
                "launcher_component_id",
                "process_role",
                "required_component_ids",
            },
            field,
        )
        bundle_id = string(value["bundle_id"], f"{field}.bundle_id")
        require(bundle_id not in bundles, f"E_RUNTIME_BUNDLE_REUSE: {bundle_id}")
        bundles[bundle_id] = value
    exact(
        sorted(bundles),
        sorted(RUNTIME_BUNDLE_EVIDENCE_ROLES),
        "runtime_bundle_ids",
    )
    components = {}
    for index, value in enumerate(runtime_snapshot["runtime_components"]):
        field = f"runtime_components[{index}]"
        exact_keys(
            value,
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
        component_id = string(value["component_id"], f"{field}.component_id")
        require(
            component_id not in components,
            f"E_RUNTIME_COMPONENT_REUSE: {component_id}",
        )
        components[component_id] = value

    base_executors = {
        value["executor_id"]: value
        for value in runtime["executors"]
    }
    exact(
        set(base_executors),
        {"GPU", "PHONE_OP12", "PHONE_OP15"},
        "runtime.executors",
    )
    boot_ids = {
        "cuda": base_executors["GPU"]["host_boot_id"],
        "op12": base_executors["PHONE_OP12"]["boot_id"],
        "op15": base_executors["PHONE_OP15"]["boot_id"],
    }
    require(
        type(joint["runtime_processes"]) is list,
        "E_RUNTIME_PROCESSES: joint",
    )
    raw_processes = [*joint["runtime_processes"], monolithic["runtime_process"]]
    require(
        all(type(value) is dict for value in raw_processes),
        "E_RUNTIME_PROCESS_TYPE",
    )
    raw_processes.sort(key=lambda value: value.get("bundle_id", ""))
    process_ids = [
        value.get("bundle_id")
        for value in raw_processes
    ]
    exact(
        process_ids,
        sorted(RUNTIME_BUNDLE_EVIDENCE_ROLES),
        "runtime_process_order",
    )

    processes = []
    for index, source in enumerate(raw_processes):
        field = f"runtime_processes[{index}]"
        exact_keys(source, RAW_RUNTIME_PROCESS_KEYS, field)
        bundle_id = string(source["bundle_id"], f"{field}.bundle_id")
        bundle = bundles[bundle_id]
        endpoint = string(bundle["endpoint"], f"bundle.{bundle_id}.endpoint")
        require(endpoint in RUNTIME_SYSTEM_ROOTS, f"E_RUNTIME_ENDPOINT: {bundle_id}")
        exact(source["endpoint"], endpoint, f"{field}.endpoint")
        exact(source["boot_id"], boot_ids[endpoint], f"{field}.boot_id")
        integer(source["pid"], f"{field}.pid", 1)
        integer(source["start_ticks"], f"{field}.start_ticks", 1)
        observed_ns = integer(source["observed_ns"], f"{field}.observed_ns", 1)
        require(
            identity_started <= observed_ns <= identity_completed,
            f"E_RUNTIME_OBSERVED: {bundle_id}",
        )
        launcher_component_id = string(
            bundle["launcher_component_id"],
            f"bundle.{bundle_id}.launcher_component_id",
        )
        require(
            launcher_component_id in components,
            f"E_RUNTIME_LAUNCHER_COMPONENT: {bundle_id}",
        )
        launcher = components[launcher_component_id]
        exact(launcher["bundle_id"], bundle_id, f"{field}.launcher.bundle")
        exact(launcher["endpoint"], endpoint, f"{field}.launcher.endpoint")
        exact(source["launcher_path"], launcher["path"], f"{field}.launcher_path")
        loaded = bundle["required_component_ids"]
        require(
            type(loaded) is list
            and bool(loaded)
            and loaded == sorted(set(loaded)),
            f"E_RUNTIME_REQUIRED_COMPONENTS: {bundle_id}",
        )
        for component_id in loaded:
            require(
                component_id in components
                and components[component_id]["bundle_id"] == bundle_id
                and components[component_id]["endpoint"] == endpoint,
                f"E_RUNTIME_COMPONENT_BINDING: {component_id}",
            )
        exact(
            source["loaded_repo_component_ids"],
            loaded,
            f"{field}.loaded_repo_component_ids",
        )
        _validate_system_dependencies(
            source["system_dependencies"],
            endpoint,
            f"{field}.system_dependencies",
        )
        if bundle_id in ("op12_stagenet", "op15_stagenet"):
            executor = base_executors[
                "PHONE_OP12" if endpoint == "op12" else "PHONE_OP15"
            ]
            exact(source["pid"], executor["worker_pid"], f"{field}.worker_pid")
            exact(
                source["start_ticks"],
                executor["worker_start_ticks"],
                f"{field}.worker_start_ticks",
            )
            exact(
                source["launcher_path"],
                executor["worker_executable_path"],
                f"{field}.worker_path",
            )
        role = RUNTIME_BUNDLE_EVIDENCE_ROLES[bundle_id]
        require(role in role_digests, f"E_RUNTIME_EVIDENCE_ROLE: {bundle_id}")
        process = {
            **copy.deepcopy(source),
            "bundle_sha256": digest(
                bundle["bundle_sha256"],
                f"bundle.{bundle_id}.sha256",
            ),
            "evidence_role": role,
            "evidence_sha256": digest(
                role_digests[role],
                f"{field}.evidence_sha256",
            ),
            "launcher_component_id": launcher_component_id,
        }
        exact_keys(process, RUNTIME_BUNDLE_PROCESS_KEYS, f"{field}.final")
        processes.append(process)

    return {
        "base_runtime_identity_sha256": sha256_bytes(canonical_bytes(runtime)),
        "completed_ns": identity_completed,
        "phase": PHASE,
        "phase_id": phase_id,
        "processes": processes,
        "runtime_bundle_fresh_sha256": sha256_bytes(runtime_fresh_raw),
        "runtime_bundle_plan_sha256": digest(
            runtime_snapshot["runtime_bundle_plan_sha256"],
            "runtime_bundle_plan_sha256",
        ),
        "runtime_bundle_snapshot_sha256": sha256_bytes(runtime_snapshot_raw),
        "schema": "s39-cp0-r1-runtime-bundle-runtime-identity-v1",
        "started_ns": identity_started,
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
    contract_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    for path, field in (
        (command_plan, "command_plan"),
        (output_dir, "output_dir"),
        (pre_dir, "pre_dir"),
        (contract_path, "contract"),
        (candidate_path, "candidate"),
    ):
        require(path.is_absolute(), f"E_PATH: {field}")
    require(output_dir.is_dir() and not output_dir.is_symlink(), "E_OUTPUT_DIR")
    require(pre_dir.is_dir() and not pre_dir.is_symlink(), "E_PRE_DIR")
    require(
        phase_id.startswith("cp0-r1-v23-a-only-")
        and len(phase_id) <= 128
        and all(ch.isalnum() or ch in ".-_" for ch in phase_id),
        "E_PHASE_ID",
    )
    acquisition_started_ns = integer(acquisition_started_ns, "acquisition_started_ns", 1)
    contract, contract_raw, candidate, candidate_raw, model = _load_inputs(
        contract_path,
        candidate_path,
    )
    plan, plan_raw = load_plan(command_plan, contract_raw, candidate_raw, model)
    plan_sha256 = sha256_bytes(plan_raw)
    mechanism_sha256 = digest_json(plan["mechanism_commands"])

    route_role = f"model.{MODEL_ID}.route_lock"
    route_rows, _ = parse_jsonl(pre_dir / "route_lock.jsonl", route_role, phase_id)
    route = _validate_route(route_rows, contract, model)
    corpus_wrapped, corpus_role_raw = parse_jsonl(
        pre_dir / "quality_corpus.jsonl",
        "quality.corpus",
        phase_id,
    )
    corpus_rows, _ = _validate_corpus(corpus_wrapped, contract, candidate)

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
    exact_keys(joint, JOINT_KEYS, "joint")
    joint_started, joint_completed = _source_interval(
        joint,
        phase_id,
        JOINT_SCHEMA,
        acquisition_started_ns,
        *receipts["joint_phone_cuda"],
        mechanism_sha256,
        model["artifact"]["sha256"],
        "joint",
    )
    monolithic, _ = producer_results["cuda_monolithic"]
    exact_keys(monolithic, MONOLITHIC_KEYS, "monolithic")
    monolithic_started, monolithic_completed = _source_interval(
        monolithic,
        phase_id,
        MONOLITHIC_SCHEMA,
        acquisition_started_ns,
        *receipts["cuda_monolithic"],
        mechanism_sha256,
        model["artifact"]["sha256"],
        "monolithic",
    )

    prefix = f"model.{MODEL_ID}"
    rows = {
        f"{prefix}.mechanics.phone": _wrap_rows(
            joint["mechanics_rows"], f"{prefix}.mechanics.phone",
            phase_id, joint_started, joint_completed,
        ),
        f"{prefix}.oracle.cuda_route": _wrap_rows(
            joint["cuda_route_rows"], f"{prefix}.oracle.cuda_route",
            phase_id, joint_started, joint_completed,
        ),
        f"{prefix}.oracle.cuda_monolithic": _wrap_rows(
            monolithic["oracle_cuda_monolithic_rows"],
            f"{prefix}.oracle.cuda_monolithic",
            phase_id, monolithic_started, monolithic_completed,
        ),
        f"{prefix}.cuda_memory": _wrap_rows(
            joint["cuda_memory_rows"], f"{prefix}.cuda_memory",
            phase_id, joint_started, joint_completed,
        ),
        f"{prefix}.quality.cuda": _wrap_rows(
            joint["quality_cuda_rows"], f"{prefix}.quality.cuda",
            phase_id, joint_started, joint_completed,
        ),
        f"{prefix}.quality.phone": _wrap_rows(
            joint["quality_phone_rows"], f"{prefix}.quality.phone",
            phase_id, joint_started, joint_completed,
        ),
        f"{prefix}.bridge": _wrap_rows(
            joint["bridge_rows"], f"{prefix}.bridge",
            phase_id, joint_started, joint_completed,
        ),
        f"{prefix}.placement.op15": _wrap_rows(
            joint["placement_op15_rows"], f"{prefix}.placement.op15",
            phase_id, joint_started, joint_completed,
        ),
        f"{prefix}.placement.op12": _wrap_rows(
            joint["placement_op12_rows"], f"{prefix}.placement.op12",
            phase_id, joint_started, joint_completed,
        ),
        f"{prefix}.route_transfer": _wrap_rows(
            joint["route_transfer_rows"], f"{prefix}.route_transfer",
            phase_id, joint_started, joint_completed,
        ),
    }
    derived = _validate_roles(
        rows,
        model,
        route,
        contract,
        candidate,
        corpus_rows,
        corpus_wrapped,
        corpus_role_raw,
    )
    role_raw = {
        role: b"".join(canonical_line(row) for row in role_rows)
        for role, role_rows in rows.items()
    }
    role_digests = {
        role: sha256_bytes(raw)
        for role, raw in role_raw.items()
    }
    artifacts, fresh, fresh_raw = _load_legacy_readiness(
        pre_dir,
        phase_id,
        acquisition_started_ns,
        contract,
        model,
        route,
    )
    runtime = _build_runtime(
        joint,
        fresh,
        fresh_raw,
        artifacts,
        role_digests,
        model,
        phase_id,
        derived["transfer_bytes"],
    )
    (
        bundle_snapshot,
        bundle_snapshot_raw,
        bundle_lock,
        _bundle_lock_raw,
        bundle_fresh,
        bundle_fresh_raw,
    ) = _load_runtime_bundle_overlay(pre_dir, phase_id, acquisition_started_ns)
    runtime_bundle = _build_runtime_bundle_identity(
        joint,
        monolithic,
        runtime,
        bundle_snapshot,
        bundle_snapshot_raw,
        bundle_fresh_raw,
        role_digests,
        phase_id,
        min(joint_started, monolithic_started),
        max(joint_completed, monolithic_completed),
    )
    del bundle_lock, bundle_fresh

    for role in sorted(OUTPUT_FILES):
        durable_write_new(output_dir / OUTPUT_FILES[role], role_raw[role])
    durable_write_new(output_dir / RUNTIME_FILE, canonical_bytes(runtime))
    durable_write_new(output_dir / RUNTIME_BUNDLE_FILE, canonical_bytes(runtime_bundle))
    output_sha256s = {
        **role_digests,
        "runtime_bundle_identity": sha256_bytes(canonical_bytes(runtime_bundle)),
        "runtime_identity": sha256_bytes(canonical_bytes(runtime)),
    }
    result = {
        "command_plan_sha256": plan_sha256,
        "derived": derived,
        "output_sha256s": output_sha256s,
        "schema": "s39-cp0-r1-a-only-acquisition-driver-result-v1",
        "status": "RAW_ROLES_EMITTED_PENDING_OUTER_V2_3_VALIDATION",
    }
    durable_write_new(
        output_dir / "ACQUISITION_DRIVER_RESULT.json",
        canonical_bytes(result),
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--pre-dir", type=Path, required=True)
    parser.add_argument("--acquisition-started-ns", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
        ReadinessError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"A_ONLY_ACQUISITION_REFUSED: {error}")
        return 2


# Tests may call the helpers through the historical common namespace.
common = types.SimpleNamespace(**globals())


if __name__ == "__main__":
    raise SystemExit(main())
