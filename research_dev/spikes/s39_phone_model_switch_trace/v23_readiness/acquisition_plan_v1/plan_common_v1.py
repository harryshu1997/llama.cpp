#!/usr/bin/env python3
"""Canonical CP0-R1 V2.3 A_ONLY acquisition-plan construction."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
V23 = HERE.parent
S39 = V23.parent
REPO_ROOT = HERE.parents[4]

LEGACY_PLAN_SCHEMA = "s39-cp0-r1-a-only-acquisition-plan-v2.3"
PLAN_SCHEMA = "s39-cp0-r1-a-only-acquisition-plan-v2.3-production-v2"
MODEL_ID = "qwen3-14b-q4_k_m"
PHASE = "A_ONLY"
WORKER_PATH = (
    "/data/local/tmp/s39-active-warm/v1/runtime-qwen-partial-v1/"
    "llama-layersplit"
)

CANDIDATE = S39 / "CP0_R1_CANDIDATE.json"
CONTRACT = V23 / "CP0_R1_EVIDENCE_CONTRACT_V2_3.json"
OUTER_DRIVER = V23 / "acquire_a_only_v23.py"
HISTORICAL_ARTIFACT_DRIVER = (
    V23 / "production_v1" / "artifact_snapshot_driver_v1.py"
)
HISTORICAL_FRESH_DRIVER = V23 / "production_v1" / "fresh_readiness_driver_v1.py"
HISTORICAL_READINESS_SUPPORT = V23 / "production_v1" / "driver_common_v1.py"
RUNTIME_BUNDLE_OVERLAY = V23 / "production_v2" / "runtime_bundle_overlay_v1.py"
ACQUISITION_DRIVER = (
    V23
    / "a_only_acquisition_driver_v1"
    / "run_a_only_acquisition_v1.py"
)
COMMAND_PLAN = (
    V23
    / "a_only_acquisition_driver_v1"
    / "A_ONLY_COMMAND_PLAN_V1.json"
)
JOINT_PRODUCER = (
    V23
    / "a_only_acquisition_driver_v1"
    / "producers_v1"
    / "joint_phone_cuda_v1.py"
)
MONOLITHIC_PRODUCER = (
    V23
    / "a_only_acquisition_driver_v1"
    / "producers_v1"
    / "cuda_monolithic_v1.py"
)
PLAN_PATH = HERE / "A_ONLY_ACQUISITION_PLAN_V2_3.json"
MANIFEST_PATH = HERE / "SOURCE_SHA256SUMS.txt"
BUILDER = HERE / "build_a_only_plan_v1.py"
VALIDATOR = HERE / "validate_a_only_plan_v1.py"

CANDIDATE_SHA256 = (
    "ee3196ca660fa7eb7ea293260dc98dd6fdbf14571a4d4c5aece5343fb29b28d8"
)
CONTRACT_SHA256 = (
    "7c5ce73fc858abf0be061e830f474753f47d9bf66fc56e5df4123c3964e6a5bc"
)
PINNED_HISTORICAL_FILES = {
    HISTORICAL_ARTIFACT_DRIVER: (
        "afcb9882a176afc0ee6c63580ed2bb1350e39b487477cc22bde32fe05ef78564"
    ),
    HISTORICAL_FRESH_DRIVER: (
        "f8e61fd2ff319c012ba6c8473283fe6ebe634374150e81047fd6c228f25c9fdc"
    ),
    HISTORICAL_READINESS_SUPPORT: (
        "f3903363038cfa97db5e60c042963d87a59eee77c09cc48c238125512fb4e735"
    ),
}
PINNED_ACQUISITION_FILES: dict[Path, str] = {}
PINNED_READINESS_V2_FILES: dict[Path, str] = {}
READINESS_V2_SPECS: dict[str, dict[str, Any]] = {}
ACQUISITION_SPEC: dict[str, Any] = {}

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
OUTPUT_FILES = {
    "artifact_snapshot": "artifact_snapshot.json",
    "fresh_snapshot": "fresh_snapshot.json",
    "readiness_lock": "readiness_lock.json",
    "runtime_bundle_identity": "runtime_bundle_identity.json",
    "runtime_identity": "runtime_identity.json",
}

DIGEST_RE = re.compile(r"[0-9a-f]{64}")
MANIFEST_RE = re.compile(r"([0-9a-f]{64})  ([^\n]+)\n")
DRIVER_FILE_FLAGS = {
    "artifact": {
        "--base-support",
        "--candidate",
        "--contract",
        "--entry-support",
        "--runtime-bundle-plan",
        "--support",
    },
    "fresh": {
        "--base-support",
        "--candidate",
        "--contract",
        "--entry-support",
        "--runtime-bundle-plan",
        "--support",
    },
    "acquisition": {
        "--candidate",
        "--command-plan",
        "--contract",
    },
}
SOURCE_BINDING_KEYS = {
    "source_manifest_path",
    "source_manifest_sha256",
    "source_root",
}
COMMAND_PLAN_SCHEMA = "s39-cp0-r1-a-only-runtime-command-plan-v1"
JOINT_PLAN_SCHEMA = "s39-cp0-r1-a-only-joint-capture-plan-v1"
PHONE_LAUNCH_SCHEMA = "s39-cp0-r1-a-only-phone-route-launch-v1"
CUDA_LAUNCH_SCHEMA = "s39-cp0-r1-a-only-cuda-route-launch-v1"
MONOLITHIC_LAUNCH_SCHEMA = "s39-cp0-r1-a-only-cuda-monolithic-launch-v1"
HISTORY_SCHEMA = "s39-cp0-r1-a-only-b8-histories-v1"
RUNTIME_BUNDLE_PLAN_SCHEMA = "s39-cp0-r1-runtime-bundle-plan-v1"
COMMAND_PLAN_KEYS = {
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
JOINT_PLAN_KEYS = {
    "commands",
    "mechanism_commands",
    "model_id",
    "model_sha256",
    "phase",
    "schema",
}
HISTORY_KEYS = {
    "histories",
    "history_width",
    "model_id",
    "model_sha256",
    "request_ids",
    "route_epoch",
    "schema",
}
PHONE_LAUNCH_KEYS = {
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
    "relay_process_probe",
    "relay_host",
    "relay_port",
    "route_epoch",
    "schema",
}
RELAY_PROCESS_PROBE_KEYS = {
    "argv",
    "cwd",
    "environment",
    "expected_argv",
    "expected_executable_path",
    "expected_port",
    "launcher_bytes",
    "launcher_sha256",
    "timeout_ms",
}
CUDA_LAUNCH_KEYS = {
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
MONOLITHIC_LAUNCH_KEYS = {
    "command",
    "cwd",
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
    "mechanism_commands",
    "mechanism_commands_sha256",
    "model_id",
    "model_sha256",
    "port",
    "schema",
    "shutdown_timeout_ms",
    "startup_timeout_ms",
}
RUNTIME_PLAN_KEYS = {
    "bundle_roots",
    "bundles",
    "candidate_sha256",
    "components",
    "contract_sha256",
    "model_id",
    "phase",
    "schema",
}
NESTED_COMMAND_FILE_FLAGS = {
    "joint_phone_cuda": {"--capture-plan"},
    "cuda_monolithic": {"--histories", "--launch-plan"},
}
JOINT_COMMAND_FILE_FLAGS = {
    "phone": {"--histories", "--launch-plan"},
    "cuda": {"--histories", "--launch-plan"},
}
EXPLICIT_SUPPORT_MODULES = {
    "build_contract_v23",
    "build_cp0_r1_mmlu64_v22",
    "build_cp0_r1_v21",
    "build_cp0_r1_v22",
    "cp0_r1_evidence_v2",
    "cp0_r1_evidence_v21",
    "cp0_r1_evidence_v22",
    "cp0_r1_evidence_v23",
    "v23_common",
}


class PlanError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}: expected {expected!r}, got {value!r}",
    )


def reject_constant(value: str) -> None:
    raise PlanError(f"E_JSON_NUMBER: {value}")


def canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return (text + "\n").encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise PlanError("E_CANONICAL") from error


def read_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path)
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlanError(f"E_JSON: {path}") from error
    require(type(value) is dict, f"E_TYPE: {path}")
    require(canonical_bytes(value) == raw, f"E_CANONICAL: {path}")
    return value, raw


def read_regular(path: Path) -> bytes:
    require(path.is_absolute(), f"E_ABSOLUTE_PATH: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PlanError(f"E_SOURCE_OPEN: {path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode), f"E_SOURCE_TYPE: {path}")
        raw = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            raw.extend(chunk)
    finally:
        os.close(descriptor)
    return bytes(raw)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify_digest(path: Path, expected: str) -> bytes:
    require(DIGEST_RE.fullmatch(expected) is not None, f"E_DIGEST: {path}")
    raw = read_regular(path)
    require(sha256_bytes(raw) == expected, f"E_SOURCE_SHA256: {path}")
    return raw


def verify_mode(path: Path, expected: int) -> None:
    metadata = path.stat(follow_symlinks=False)
    require(stat.S_IMODE(metadata.st_mode) == expected, f"E_SOURCE_MODE: {path}")


def exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    exact(set(value), expected, f"{field}.keys")
    return value


def digest(value: Any, field: str) -> str:
    require(
        type(value) is str and DIGEST_RE.fullmatch(value) is not None,
        f"E_DIGEST: {field}",
    )
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"E_INTEGER: {field}")
    return value


def nonempty_string(value: Any, field: str) -> str:
    require(type(value) is str and bool(value), f"E_STRING: {field}")
    return value


def flag_value_indexes(
    argv: list[str],
    flags: set[str],
    field: str,
) -> dict[str, int]:
    result = {}
    for flag in sorted(flags):
        exact(argv.count(flag), 1, f"{field}.{flag}.count")
        flag_index = argv.index(flag)
        require(flag_index + 1 < len(argv), f"E_EXECUTED_ARGUMENT: {field}.{flag}")
        value_index = flag_index + 1
        value = argv[value_index]
        require(
            type(value) is str
            and bool(value)
            and not value.startswith("{"),
            f"E_EXECUTED_ARGUMENT: {field}.{flag}",
        )
        require(
            value not in flags,
            f"E_EXECUTED_ARGUMENT: {field}.{flag}",
        )
        result[flag] = value_index
    return result


def verify_no_symlink_components(root: Path, path: Path) -> None:
    require(root.is_absolute() and path.is_absolute(), f"E_ABSOLUTE_PATH: {path}")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise PlanError(f"E_SOURCE_ROOT_ESCAPE: {path}") from error
    require(relative.parts and ".." not in relative.parts, f"E_SOURCE_ROOT_ESCAPE: {path}")
    current = root
    root_metadata = os.lstat(root)
    require(not stat.S_ISLNK(root_metadata.st_mode), f"E_SOURCE_SYMLINK: {root}")
    for part in relative.parts:
        current /= part
        metadata = os.lstat(current)
        require(not stat.S_ISLNK(metadata.st_mode), f"E_SOURCE_SYMLINK: {current}")


def verify_no_unbound_python_imports(
    paths: list[Path],
    bound_modules: set[str] | None = None,
) -> None:
    bound_modules = bound_modules or set()
    for path in paths:
        source = read_regular(path)
        try:
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as error:
            raise PlanError(f"E_PYTHON_SOURCE: {path}") from error
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                require(node.level == 0, f"E_UNBOUND_IMPORT: {path}")
                names = [node.module or ""]
            for name in names:
                root = name.split(".", 1)[0]
                require(
                    root in sys.stdlib_module_names or root in bound_modules,
                    f"E_UNBOUND_IMPORT: {path}: {name}",
                )


def repo_relative(path: Path) -> str:
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise PlanError(f"E_REPO_PATH: {path}") from error
    require(
        relative.parts
        and ".." not in relative.parts
        and not relative.is_absolute(),
        f"E_REPO_PATH: {path}",
    )
    return relative.as_posix()


def executed_file(path: Path, argv_index: int) -> dict[str, Any]:
    raw = read_regular(path)
    return {
        "argv_index": argv_index,
        "bytes": len(raw),
        "path": str(path),
        "sha256": sha256_bytes(raw),
    }


def verify_historical_readiness_sources() -> None:
    verify_digest(CANDIDATE, CANDIDATE_SHA256)
    verify_digest(CONTRACT, CONTRACT_SHA256)
    for path, digest in PINNED_HISTORICAL_FILES.items():
        verify_digest(path, digest)
    verify_mode(HISTORICAL_ARTIFACT_DRIVER, 0o755)
    verify_mode(HISTORICAL_FRESH_DRIVER, 0o755)
    verify_mode(HISTORICAL_READINESS_SUPPORT, 0o644)


def verify_frozen_acquisition_sources() -> None:
    required = (
        ACQUISITION_DRIVER,
        COMMAND_PLAN,
        JOINT_PRODUCER,
        MONOLITHIC_PRODUCER,
    )
    for path in required:
        require(path.is_file(), f"E_PRODUCTION_INPUT_ABSENT: {path}")
        require(
            path in PINNED_ACQUISITION_FILES,
            f"E_PRODUCTION_INPUT_NOT_FROZEN: {path}",
        )
        verify_digest(path, PINNED_ACQUISITION_FILES[path])
    verify_mode(ACQUISITION_DRIVER, 0o755)
    verify_mode(COMMAND_PLAN, 0o644)


def validate_frozen_contract() -> tuple[dict[str, Any], bytes, bytes]:
    candidate_raw = verify_digest(CANDIDATE, CANDIDATE_SHA256)
    contract_raw = verify_digest(CONTRACT, CONTRACT_SHA256)
    candidate, _ = read_canonical(CANDIDATE)
    contract, _ = read_canonical(CONTRACT)
    exact(contract.get("schema"), "s39-cp0-r1-evidence-contract-v2.3", "contract.schema")
    models = candidate.get("models")
    require(type(models) is list, "E_TYPE: candidate.models")
    model = next(
        (
            value
            for value in models
            if type(value) is dict and value.get("model_id") == MODEL_ID
        ),
        None,
    )
    require(type(model) is dict and model.get("slot") == "A", "E_MODEL_BINDING")
    expected_roles = set(PAYLOAD_ROLES) | {
        "phase.lock",
        "phase.preflight",
        "quality.corpus",
        f"model.{MODEL_ID}.route_lock",
    }
    phase_roles = contract.get("phase_protocol", {}).get("phase_roles", {})
    roles = phase_roles.get(PHASE) if type(phase_roles) is dict else None
    require(type(roles) is list, "E_TYPE: contract.phase_roles.A_ONLY")
    exact(set(roles), expected_roles, "contract.phase_roles.A_ONLY")
    return contract, contract_raw, candidate_raw


def production_blockers() -> list[str]:
    blockers = []
    if set(READINESS_V2_SPECS) != {"artifact", "fresh"}:
        blockers.append("runtime_bundle_readiness_not_frozen")
    if not PINNED_READINESS_V2_FILES:
        blockers.append("runtime_bundle_sources_not_frozen")
    for path in (
        ACQUISITION_DRIVER,
        COMMAND_PLAN,
        JOINT_PRODUCER,
        MONOLITHIC_PRODUCER,
    ):
        if not path.is_file():
            blockers.append(f"absent:{repo_relative(path)}")
        elif path not in PINNED_ACQUISITION_FILES:
            blockers.append(f"not_frozen:{repo_relative(path)}")
    if not ACQUISITION_SPEC:
        blockers.append("acquisition_argv_not_frozen")
    return blockers


def require_production_ready() -> None:
    blockers = production_blockers()
    require(not blockers, "E_PRODUCTION_NOT_READY: " + ",".join(blockers))
    for path, digest in PINNED_READINESS_V2_FILES.items():
        verify_digest(path, digest)
    verify_frozen_acquisition_sources()


def validate_driver_spec(
    value: Any,
    name: str,
    file_flags: set[str] | None = None,
) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {name}")
    exact(set(value), {"argv_template", "executed_files", "timeout_seconds"}, name)
    argv = value["argv_template"]
    require(
        type(argv) is list
        and bool(argv)
        and all(type(item) is str and bool(item) for item in argv),
        f"E_TYPE: {name}.argv_template",
    )
    required_indexes = {0}
    for flag, value_index in flag_value_indexes(
        argv,
        DRIVER_FILE_FLAGS.get(name, set()) if file_flags is None else file_flags,
        f"{name}.argv_template",
    ).items():
        path = Path(argv[value_index])
        require(path.is_absolute(), f"E_PATH: {name}.{flag}")
        required_indexes.add(value_index)
    for index, item in enumerate(argv):
        path = Path(item)
        if not path.is_absolute():
            continue
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise PlanError(f"E_SOURCE_STAT: {path}: {error}") from error
        require(not stat.S_ISLNK(metadata.st_mode), f"E_SOURCE_SYMLINK: {path}")
        if stat.S_ISREG(metadata.st_mode):
            required_indexes.add(index)
    records = value["executed_files"]
    require(type(records) is list and bool(records), f"E_TYPE: {name}.executed_files")
    indexes = []
    for index, record in enumerate(records):
        field = f"{name}.executed_files[{index}]"
        require(type(record) is dict, f"E_TYPE: {field}")
        exact(set(record), {"argv_index", "bytes", "path", "sha256"}, field)
        argv_index = record["argv_index"]
        require(type(argv_index) is int and 0 <= argv_index < len(argv), f"E_INDEX: {field}")
        require(argv_index not in indexes, f"E_INDEX_REUSE: {field}")
        indexes.append(argv_index)
        exact(argv[argv_index], record["path"], f"{field}.argv")
        path = Path(record["path"])
        require(path.is_absolute(), f"E_PATH: {field}")
        require(type(record["bytes"]) is int and record["bytes"] > 0, f"E_BYTES: {field}")
        require(DIGEST_RE.fullmatch(record["sha256"]) is not None, f"E_DIGEST: {field}")
        raw = verify_digest(path, record["sha256"])
        exact(len(raw), record["bytes"], f"{field}.bytes")
    exact(indexes, sorted(indexes), f"{name}.executed_file_order")
    exact(set(indexes), required_indexes, f"{name}.executed_file_indexes")
    timeout = value["timeout_seconds"]
    require(type(timeout) is int and 1 <= timeout <= 7200, f"E_TIMEOUT: {name}")
    return value


def bound_flag_paths(
    value: dict[str, Any],
    flags: set[str],
    field: str,
) -> dict[str, Path]:
    indexes = flag_value_indexes(value["argv_template"], flags, field)
    records = {
        record["argv_index"]: record
        for record in value["executed_files"]
    }
    require(
        {0, *indexes.values()}.issubset(records),
        f"E_EXECUTED_FILE_MISSING: {field}",
    )
    result = {}
    for flag, index in indexes.items():
        require(index in records, f"E_EXECUTED_FILE_MISSING: {field}.{flag}")
        result[flag] = Path(records[index]["path"])
    return result


def parse_inline_plan(argv: list[str], field: str) -> dict[str, Any] | None:
    if "--plan-json" not in argv:
        require(
            all("{" not in argument and "}" not in argument for argument in argv),
            f"E_INLINE_PLAN: {field}",
        )
        return None
    exact(argv.count("--plan-json"), 1, f"{field}.--plan-json.count")
    index = argv.index("--plan-json")
    require(index + 1 < len(argv), f"E_INLINE_PLAN: {field}")
    raw = argv[index + 1]
    try:
        value = json.loads(
            raw,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise PlanError(f"E_INLINE_PLAN_JSON: {field}") from error
    require(type(value) is dict, f"E_INLINE_PLAN_TYPE: {field}")
    exact(
        canonical_bytes(value)[:-1].decode("ascii"),
        raw,
        f"{field}.--plan-json.canonical",
    )
    for argument_index, argument in enumerate(argv):
        if argument_index == index + 1:
            continue
        require(
            "{" not in argument and "}" not in argument,
            f"E_INLINE_PLAN_ARGUMENT: {field}[{argument_index}]",
        )
    exact(argv.count("--plan-sha256"), 1, f"{field}.--plan-sha256.count")
    digest_value = exact_argv_value(argv, "--plan-sha256", field)
    exact(
        digest_value,
        sha256_bytes(raw.encode("ascii")),
        f"{field}.--plan-sha256",
    )
    return value


def validate_command_matrix(value: Any, field: str) -> dict[str, list[list[str]]]:
    value = exact_keys(value, {"desktop", "op12", "op15"}, field)
    for endpoint in ("desktop", "op12", "op15"):
        commands = value[endpoint]
        require(type(commands) is list and bool(commands), f"E_COMMANDS: {field}.{endpoint}")
        for index, argv in enumerate(commands):
            item = f"{field}.{endpoint}[{index}]"
            require(
                type(argv) is list
                and bool(argv)
                and all(
                    type(argument) is str
                    and bool(argument)
                    and "\x00" not in argument
                    and "\n" not in argument
                    for argument in argv
                ),
                f"E_COMMAND: {item}",
            )
            parse_inline_plan(argv, item)
            require(Path(argv[0]).is_absolute(), f"E_COMMAND_PATH: {item}")
            require(
                Path(argv[0]).name not in {"bash", "dash", "sh", "zsh"}
                and "-c" not in argv,
                f"E_COMMAND_SHELL: {item}",
            )
    exact(len(value["desktop"]), 9, f"{field}.desktop_count")
    return value


def validate_nested_command_spec(
    value: Any,
    name: str,
    file_flags: set[str],
) -> tuple[dict[str, Any], dict[str, Path]]:
    value = exact_keys(
        value,
        {"argv_template", "executed_files", "result_filename", "timeout_seconds"},
        name,
    )
    template = value["argv_template"]
    require(
        type(template) is list
        and bool(template)
        and all(type(item) is str and bool(item) for item in template),
        f"E_TYPE: {name}.argv_template",
    )
    placeholders = {
        item
        for item in template
        if item.startswith("{") and item.endswith("}")
    }
    exact(
        placeholders,
        {
            "{acquisition_started_ns}",
            "{command_plan_sha256}",
            "{output_path}",
            "{phase_id}",
            "{pre_dir}",
        },
        f"{name}.placeholders",
    )
    require(
        all(item in placeholders or "{" not in item and "}" not in item for item in template),
        f"E_PLACEHOLDER: {name}",
    )
    driver_value = {
        "argv_template": template,
        "executed_files": value["executed_files"],
        "timeout_seconds": value["timeout_seconds"],
    }
    validate_driver_spec(driver_value, name, file_flags)
    result = nonempty_string(value["result_filename"], f"{name}.result_filename")
    result_path = Path(result)
    require(
        not result_path.is_absolute()
        and result not in (".", "..")
        and ".." not in result_path.parts,
        f"E_PATH: {name}.result_filename",
    )
    return value, bound_flag_paths(driver_value, file_flags, name)


def validate_histories(
    path: Path,
    model_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    value, raw = read_canonical(path)
    exact_keys(value, HISTORY_KEYS, "histories")
    exact(value["schema"], HISTORY_SCHEMA, "histories.schema")
    exact(value["model_id"], MODEL_ID, "histories.model_id")
    exact(value["model_sha256"], model_sha256, "histories.model_sha256")
    exact(value["request_ids"], list(range(8)), "histories.request_ids")
    exact(value["history_width"], 2, "histories.history_width")
    histories = value["histories"]
    require(type(histories) is list and len(histories) == 8, "E_HISTORY_BATCH")
    for index, history in enumerate(histories):
        require(
            type(history) is list
            and len(history) == value["history_width"]
            and all(type(token) is int and token >= 0 for token in history),
            f"E_HISTORY_ROW: {index}",
        )
    integer(value["route_epoch"], "histories.route_epoch", 1)
    return value, raw


def validate_launch_identity(
    value: dict[str, Any],
    schema: str,
    model_sha256: str,
    field: str,
) -> None:
    exact(value["schema"], schema, f"{field}.schema")
    exact(value["model_id"], MODEL_ID, f"{field}.model_id")
    exact(value["model_sha256"], model_sha256, f"{field}.model_sha256")
    exact(value["expected_n_layer"], 40, f"{field}.expected_n_layer")
    exact(value["expected_n_embd"], 5120, f"{field}.expected_n_embd")
    exact(value["expected_max_streams"], 8, f"{field}.expected_max_streams")
    exact(value["expected_n_ctx_seq"], 256, f"{field}.expected_n_ctx_seq")
    exact(value["expected_n_batch"], 64, f"{field}.expected_n_batch")
    exact(value["expected_n_ubatch"], 64, f"{field}.expected_n_ubatch")


def exact_argv_value(
    argv: list[str],
    flag: str,
    field: str,
) -> str:
    exact(argv.count(flag), 1, f"{field}.{flag}.count")
    index = argv.index(flag)
    require(index + 1 < len(argv), f"E_EXECUTED_ARGUMENT: {field}.{flag}")
    return nonempty_string(argv[index + 1], f"{field}.{flag}")


def validate_relay_probe_dependencies(
    phone: dict[str, Any],
) -> list[Path]:
    spec = exact_keys(
        phone["relay_process_probe"],
        RELAY_PROCESS_PROBE_KEYS,
        "phone_launch.relay_process_probe",
    )
    argv = spec["argv"]
    require(
        type(argv) is list
        and bool(argv)
        and all(type(item) is str and bool(item) for item in argv),
        "E_TYPE: phone_launch.relay_process_probe.argv",
    )
    launcher = Path(argv[0])
    require(launcher.is_absolute(), "E_PATH: relay_process_probe.launcher")
    launcher_raw = verify_digest(
        launcher,
        digest(
            spec["launcher_sha256"],
            "relay_process_probe.launcher_sha256",
        ),
    )
    exact(
        len(launcher_raw),
        integer(
            spec["launcher_bytes"],
            "relay_process_probe.launcher_bytes",
            1,
        ),
        "relay_process_probe.launcher_bytes",
    )
    require(
        os.stat(launcher, follow_symlinks=False).st_mode & 0o111 != 0,
        "E_SOURCE_MODE: relay_process_probe.launcher",
    )
    adb = Path(
        exact_argv_value(
            argv,
            "--adb",
            "phone_launch.relay_process_probe.argv",
        )
    )
    require(adb.is_absolute(), "E_PATH: relay_process_probe.adb")
    adb_raw = verify_digest(
        adb,
        digest(
            exact_argv_value(
                argv,
                "--adb-sha256",
                "phone_launch.relay_process_probe.argv",
            ),
            "relay_process_probe.adb_sha256",
        ),
    )
    require(bool(adb_raw), "E_BYTES: relay_process_probe.adb")
    require(
        os.stat(adb, follow_symlinks=False).st_mode & 0o111 != 0,
        "E_SOURCE_MODE: relay_process_probe.adb",
    )
    exact(
        exact_argv_value(
            argv,
            "--adb-port",
            "phone_launch.relay_process_probe.argv",
        ),
        "5038",
        "relay_process_probe.adb_port",
    )
    return [launcher, adb]


def validate_runtime_bundle_plan(
    path: Path,
    contract_sha256: str,
    candidate_sha256: str,
) -> dict[str, Any]:
    value, _ = read_canonical(path)
    exact_keys(value, RUNTIME_PLAN_KEYS, "runtime_bundle_plan")
    exact(value["schema"], RUNTIME_BUNDLE_PLAN_SCHEMA, "runtime_bundle_plan.schema")
    exact(value["phase"], PHASE, "runtime_bundle_plan.phase")
    exact(value["model_id"], MODEL_ID, "runtime_bundle_plan.model_id")
    exact(value["contract_sha256"], contract_sha256, "runtime_bundle_plan.contract")
    exact(value["candidate_sha256"], candidate_sha256, "runtime_bundle_plan.candidate")
    roots = exact_keys(
        value["bundle_roots"],
        {
            "cuda_monolithic",
            "cuda_route",
            "op12_stagenet",
            "op15_direct_relay",
            "op15_stagenet",
        },
        "runtime_bundle_plan.bundle_roots",
    )
    for name, root in roots.items():
        root_path = Path(
            nonempty_string(
                root,
                f"runtime_bundle_plan.bundle_roots.{name}",
            )
        )
        require(root_path.is_absolute(), f"E_RUNTIME_ROOT: {name}")
    components = value["components"]
    require(type(components) is list and bool(components), "E_RUNTIME_COMPONENTS")
    component_ids = []
    for index, component in enumerate(components):
        field = f"runtime_bundle_plan.components[{index}]"
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
            },
            field,
        )
        component_ids.append(nonempty_string(component["component_id"], f"{field}.component_id"))
        component_path = Path(nonempty_string(component["path"], f"{field}.path"))
        require(component_path.is_absolute(), f"E_PATH: {field}")
        integer(component["bytes"], f"{field}.bytes", 1)
        digest(component["sha256"], f"{field}.sha256")
    exact(component_ids, sorted(set(component_ids)), "runtime_bundle_plan.component_order")
    bundles = value["bundles"]
    require(type(bundles) is list and len(bundles) == 5, "E_RUNTIME_BUNDLES")
    bundle_ids = []
    referenced = set()
    for index, bundle in enumerate(bundles):
        field = f"runtime_bundle_plan.bundles[{index}]"
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
        bundle_ids.append(nonempty_string(bundle["bundle_id"], f"{field}.bundle_id"))
        required = bundle["required_component_ids"]
        require(
            type(required) is list
            and required == sorted(set(required))
            and all(item in component_ids for item in required),
            f"E_RUNTIME_COMPONENTS: {field}",
        )
        require(bundle["launcher_component_id"] in required, f"E_RUNTIME_LAUNCHER: {field}")
        referenced.update(required)
    exact(bundle_ids, sorted(roots), "runtime_bundle_plan.bundle_order")
    exact(referenced, set(component_ids), "runtime_bundle_plan.component_coverage")
    return value


def validate_nested_execution_graph(
    command_plan_path: Path,
    contract_sha256: str,
    candidate_sha256: str,
) -> dict[str, Any]:
    command_plan, _ = read_canonical(command_plan_path)
    exact_keys(command_plan, COMMAND_PLAN_KEYS, "command_plan")
    exact(command_plan["schema"], COMMAND_PLAN_SCHEMA, "command_plan.schema")
    exact(command_plan["phase"], PHASE, "command_plan.phase")
    exact(command_plan["model_id"], MODEL_ID, "command_plan.model_id")
    exact(command_plan["contract_sha256"], contract_sha256, "command_plan.contract")
    exact(command_plan["candidate_sha256"], candidate_sha256, "command_plan.candidate")
    model_sha256 = digest(command_plan["model_sha256"], "command_plan.model_sha256")
    matrix = validate_command_matrix(
        command_plan["mechanism_commands"],
        "command_plan.mechanism_commands",
    )
    exact(
        command_plan["outputs"],
        {
            **PAYLOAD_ROLES,
            "runtime_bundle_identity": OUTPUT_FILES["runtime_bundle_identity"],
            "runtime_identity": OUTPUT_FILES["runtime_identity"],
        },
        "command_plan.outputs",
    )
    producers = exact_keys(
        command_plan["producers"],
        {"cuda_monolithic", "joint_phone_cuda"},
        "command_plan.producers",
    )
    _, joint_flags = validate_nested_command_spec(
        producers["joint_phone_cuda"],
        "command_plan.producers.joint_phone_cuda",
        NESTED_COMMAND_FILE_FLAGS["joint_phone_cuda"],
    )
    _, monolithic_flags = validate_nested_command_spec(
        producers["cuda_monolithic"],
        "command_plan.producers.cuda_monolithic",
        NESTED_COMMAND_FILE_FLAGS["cuda_monolithic"],
    )

    joint_path = joint_flags["--capture-plan"]
    joint_plan, _ = read_canonical(joint_path)
    exact_keys(joint_plan, JOINT_PLAN_KEYS, "joint_capture_plan")
    exact(joint_plan["schema"], JOINT_PLAN_SCHEMA, "joint_capture_plan.schema")
    exact(joint_plan["phase"], PHASE, "joint_capture_plan.phase")
    exact(joint_plan["model_id"], MODEL_ID, "joint_capture_plan.model_id")
    exact(joint_plan["model_sha256"], model_sha256, "joint_capture_plan.model_sha256")
    exact(joint_plan["mechanism_commands"], matrix, "joint_capture_plan.mechanism_commands")
    commands = exact_keys(
        joint_plan["commands"],
        {"cuda", "phone"},
        "joint_capture_plan.commands",
    )
    _, phone_flags = validate_nested_command_spec(
        commands["phone"],
        "joint_capture_plan.commands.phone",
        JOINT_COMMAND_FILE_FLAGS["phone"],
    )
    _, cuda_flags = validate_nested_command_spec(
        commands["cuda"],
        "joint_capture_plan.commands.cuda",
        JOINT_COMMAND_FILE_FLAGS["cuda"],
    )

    history_paths = {
        phone_flags["--histories"],
        cuda_flags["--histories"],
        monolithic_flags["--histories"],
    }
    require(len(history_paths) == 1, "E_SHARED_HISTORIES_PATH")
    histories_path = next(iter(history_paths))
    histories, histories_raw = validate_histories(histories_path, model_sha256)

    phone_path = phone_flags["--launch-plan"]
    phone, _ = read_canonical(phone_path)
    exact_keys(phone, PHONE_LAUNCH_KEYS, "phone_launch")
    validate_launch_identity(phone, PHONE_LAUNCH_SCHEMA, model_sha256, "phone_launch")
    exact(phone["history_path"], str(histories_path), "phone_launch.history_path")
    exact(phone["history_sha256"], sha256_bytes(histories_raw), "phone_launch.history_sha256")
    exact(phone["route_epoch"], histories["route_epoch"], "phone_launch.route_epoch")
    exact(phone["mechanism_commands"], matrix, "phone_launch.mechanism_commands")
    relay_dependency_paths = validate_relay_probe_dependencies(phone)

    cuda_path = cuda_flags["--launch-plan"]
    cuda, _ = read_canonical(cuda_path)
    exact_keys(cuda, CUDA_LAUNCH_KEYS, "cuda_launch")
    validate_launch_identity(cuda, CUDA_LAUNCH_SCHEMA, model_sha256, "cuda_launch")
    exact(cuda["history_path"], str(histories_path), "cuda_launch.history_path")
    exact(cuda["history_sha256"], sha256_bytes(histories_raw), "cuda_launch.history_sha256")
    exact(cuda["route_epoch"], histories["route_epoch"], "cuda_launch.route_epoch")
    exact(cuda["mechanism_commands"], matrix, "cuda_launch.mechanism_commands")

    monolithic_path = monolithic_flags["--launch-plan"]
    monolithic, _ = read_canonical(monolithic_path)
    exact_keys(monolithic, MONOLITHIC_LAUNCH_KEYS, "monolithic_launch")
    validate_launch_identity(
        monolithic,
        MONOLITHIC_LAUNCH_SCHEMA,
        model_sha256,
        "monolithic_launch",
    )
    exact(monolithic["mechanism_commands"], matrix, "monolithic_launch.mechanism_commands")
    exact(
        monolithic["mechanism_commands_sha256"],
        sha256_bytes(canonical_bytes(matrix)),
        "monolithic_launch.mechanism_commands_sha256",
    )
    command = monolithic["command"]
    require(type(command) is list and bool(command), "E_MONOLITHIC_COMMAND")
    exact(command, matrix["desktop"][8], "monolithic_launch.command")
    exact(
        sum(value == command for value in matrix["desktop"]),
        1,
        "monolithic_launch.command_count",
    )

    executed_paths = []
    for command in (
        producers["joint_phone_cuda"],
        producers["cuda_monolithic"],
        commands["phone"],
        commands["cuda"],
    ):
        executed_paths.extend(Path(record["path"]) for record in command["executed_files"])
    plan_paths = [
        command_plan_path,
        joint_path,
        histories_path,
        phone_path,
        cuda_path,
        monolithic_path,
    ]
    repo_relay_sources = [
        path
        for path in relay_dependency_paths
        if path.is_relative_to(REPO_ROOT)
    ]
    return {
        "command_matrix": matrix,
        "executed_paths": executed_paths,
        "model_sha256": model_sha256,
        "plan_paths": plan_paths,
        "runtime_dependency_paths": relay_dependency_paths,
        "source_paths": [
            *executed_paths,
            *plan_paths,
            *repo_relay_sources,
        ],
    }


def build_plan() -> dict[str, Any]:
    _, contract_raw, candidate_raw = validate_frozen_contract()
    require_production_ready()
    artifact = validate_driver_spec(READINESS_V2_SPECS["artifact"], "artifact")
    fresh = validate_driver_spec(READINESS_V2_SPECS["fresh"], "fresh")
    acquisition = validate_driver_spec(ACQUISITION_SPEC, "acquisition")
    plan = {
        "candidate_sha256": sha256_bytes(candidate_raw),
        "contract_sha256": sha256_bytes(contract_raw),
        "drivers": {
            "acquisition": acquisition,
            "artifact": artifact,
            "fresh": fresh,
        },
        "model_id": MODEL_ID,
        "output_files": OUTPUT_FILES,
        "payload_roles": PAYLOAD_ROLES,
        "phase": PHASE,
        "schema": PLAN_SCHEMA,
    }
    manifest_raw = manifest_bytes(source_paths(plan))
    plan.update({
        "source_manifest_path": str(MANIFEST_PATH),
        "source_manifest_sha256": sha256_bytes(manifest_raw),
        "source_root": str(REPO_ROOT),
    })
    return plan


def validate_plan(path: Path) -> dict[str, Any]:
    expected = build_plan()
    actual, raw = read_canonical(path)
    exact(actual, expected, "plan")
    exact(raw, canonical_bytes(expected), "plan.bytes")
    return actual


def driver_source_paths(plan: dict[str, Any]) -> list[Path]:
    paths = []
    for name in ("artifact", "fresh", "acquisition"):
        for record in plan["drivers"][name]["executed_files"]:
            path = Path(record["path"])
            require(path.is_absolute(), f"E_PATH: drivers.{name}")
            paths.append(path)
    return paths


def source_paths(plan: dict[str, Any]) -> list[Path]:
    artifact_flags = bound_flag_paths(
        plan["drivers"]["artifact"],
        DRIVER_FILE_FLAGS["artifact"],
        "drivers.artifact",
    )
    fresh_flags = bound_flag_paths(
        plan["drivers"]["fresh"],
        DRIVER_FILE_FLAGS["fresh"],
        "drivers.fresh",
    )
    for flag in DRIVER_FILE_FLAGS["artifact"]:
        exact(
            artifact_flags[flag],
            fresh_flags[flag],
            f"readiness_driver_agreement.{flag}",
        )
    exact(artifact_flags["--contract"], CONTRACT, "readiness.contract")
    exact(artifact_flags["--candidate"], CANDIDATE, "readiness.candidate")
    acquisition_flags = bound_flag_paths(
        plan["drivers"]["acquisition"],
        DRIVER_FILE_FLAGS["acquisition"],
        "drivers.acquisition",
    )
    exact(acquisition_flags["--contract"], CONTRACT, "acquisition.contract")
    exact(acquisition_flags["--candidate"], CANDIDATE, "acquisition.candidate")
    graph = validate_nested_execution_graph(
        acquisition_flags["--command-plan"],
        plan["contract_sha256"],
        plan["candidate_sha256"],
    )
    validate_runtime_bundle_plan(
        artifact_flags["--runtime-bundle-plan"],
        plan["contract_sha256"],
        plan["candidate_sha256"],
    )
    paths = [
        CANDIDATE,
        CONTRACT,
        OUTER_DRIVER,
        V23 / "build_contract_v23.py",
        V23 / "cp0_r1_evidence_v23.py",
        V23 / "v23_common.py",
        RUNTIME_BUNDLE_OVERLAY,
        S39 / "CP0_R1_EVIDENCE_CONTRACT_V2.json",
        S39 / "CP0_R1_EVIDENCE_CONTRACT_V2_1.json",
        S39 / "CP0_R1_EVIDENCE_CONTRACT_V2_2.json",
        S39 / "CP0_R1_MMLU64_CORPUS_V2_2.jsonl",
        S39 / "CP0_R1_MMLU64_SOURCES_V2_2.json",
        S39 / "CP0_R1_V2_SHA256SUMS.txt",
        S39 / "CP0_R1_V2_1_SHA256SUMS.txt",
        S39 / "CP0_R1_V2_2_SHA256SUMS.txt",
        S39 / "SHARD_MANIFEST.json",
        S39 / "build_cp0_r1_mmlu64_v22.py",
        S39 / "build_cp0_r1_v2.py",
        S39 / "build_cp0_r1_v21.py",
        S39 / "build_cp0_r1_v22.py",
        S39 / "cp0_r1_evidence_v2.py",
        S39 / "cp0_r1_evidence_v21.py",
        S39 / "cp0_r1_evidence_v22.py",
        S39 / "cp0_r1_phase_preflight_v21.py",
        ACQUISITION_DRIVER,
        COMMAND_PLAN,
        JOINT_PRODUCER,
        MONOLITHIC_PRODUCER,
        HERE / "plan_common_v1.py",
        BUILDER,
        VALIDATOR,
        *driver_source_paths(plan),
        *graph["source_paths"],
    ]
    unique = {repo_relative(path): path for path in paths}
    require(len(unique) == len(set(paths)), "E_SOURCE_ALIAS")
    return [unique[name] for name in sorted(unique)]


def verify_production_python_sources(plan: dict[str, Any]) -> None:
    acquisition_flags = bound_flag_paths(
        plan["drivers"]["acquisition"],
        DRIVER_FILE_FLAGS["acquisition"],
        "drivers.acquisition",
    )
    graph = validate_nested_execution_graph(
        acquisition_flags["--command-plan"],
        plan["contract_sha256"],
        plan["candidate_sha256"],
    )
    python_sources = [
        path
        for path in [
            *driver_source_paths(plan),
            *graph["executed_paths"],
            *graph["runtime_dependency_paths"],
            RUNTIME_BUNDLE_OVERLAY,
        ]
        if path.suffix == ".py"
    ]
    verify_no_unbound_python_imports(
        python_sources,
        EXPLICIT_SUPPORT_MODULES,
    )


def build_bundle() -> tuple[dict[str, Any], bytes, bytes]:
    plan = build_plan()
    verify_production_python_sources(plan)
    plan_raw = canonical_bytes(plan)
    sources = source_paths(plan)
    manifest_raw = manifest_bytes(sources)
    exact(
        sha256_bytes(manifest_raw),
        plan["source_manifest_sha256"],
        "source_manifest_sha256",
    )
    return plan, plan_raw, manifest_raw


def write_exclusive(path: Path, raw: bytes, mode: int = 0o644) -> None:
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


def manifest_bytes(paths: list[Path]) -> bytes:
    relatives = [repo_relative(path) for path in paths]
    require(len(relatives) == len(set(relatives)), "E_MANIFEST_DUPLICATE")
    records = [
        f"{sha256_bytes(read_regular(path))}  {relative}\n"
        for relative, path in sorted(zip(relatives, paths))
    ]
    return "".join(records).encode("ascii")


def parse_manifest(raw: bytes) -> list[tuple[str, str]]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise PlanError("E_MANIFEST_ASCII") from error
    matches = MANIFEST_RE.findall(text)
    require(
        "".join(f"{digest}  {path}\n" for digest, path in matches) == text,
        "E_MANIFEST_FORMAT",
    )
    require(bool(matches), "E_MANIFEST_EMPTY")
    paths = [path for _, path in matches]
    require(paths == sorted(paths), "E_MANIFEST_ORDER")
    require(len(paths) == len(set(paths)), "E_MANIFEST_DUPLICATE")
    for digest, value in matches:
        require(DIGEST_RE.fullmatch(digest) is not None, "E_MANIFEST_DIGEST")
        path = Path(value)
        require(
            not path.is_absolute()
            and value not in (".", "..")
            and ".." not in path.parts,
            f"E_MANIFEST_PATH: {value}",
        )
        require(
            "__pycache__" not in path.parts and path.suffix != ".pyc",
            f"E_PYCACHE_SOURCE: {value}",
        )
    return matches


def validate_manifest_at_root(
    path: Path,
    source_root: Path,
    expected_relatives: list[str] | None = None,
) -> None:
    require(path.is_absolute(), f"E_ABSOLUTE_PATH: {path}")
    require(source_root.is_absolute(), f"E_ABSOLUTE_PATH: {source_root}")
    verify_no_symlink_components(source_root, path)
    raw = read_regular(path)
    records = parse_manifest(raw)
    if expected_relatives is not None:
        require(
            [value for _, value in records] == sorted(expected_relatives),
            "E_MANIFEST_SOURCE_SET",
        )
    for digest, relative in records:
        source = source_root / relative
        verify_no_symlink_components(source_root, source)
        verify_digest(source, digest)


def validate_manifest(path: Path, expected_paths: list[Path]) -> None:
    records = parse_manifest(read_regular(path))
    expected_relatives = sorted(repo_relative(value) for value in expected_paths)
    require(
        [value for _, value in records] == expected_relatives,
        "E_MANIFEST_SOURCE_SET",
    )
    for digest_value, relative in records:
        source = REPO_ROOT / relative
        verify_no_symlink_components(REPO_ROOT, source)
        verify_digest(source, digest_value)


def validate_bound_source_manifest(
    plan: dict[str, Any],
    expected_paths: list[Path] | None = None,
) -> None:
    source_root = Path(nonempty_string(plan.get("source_root"), "source_root"))
    manifest_path = Path(
        nonempty_string(plan.get("source_manifest_path"), "source_manifest_path")
    )
    exact(source_root, REPO_ROOT, "source_root")
    exact(manifest_path, MANIFEST_PATH, "source_manifest_path")
    expected_sha256 = digest(
        plan.get("source_manifest_sha256"),
        "source_manifest_sha256",
    )
    raw = read_regular(manifest_path)
    exact(sha256_bytes(raw), expected_sha256, "source_manifest_sha256")
    relatives = []
    for value in source_paths(plan) if expected_paths is None else expected_paths:
        try:
            relative = value.relative_to(source_root)
        except ValueError as error:
            raise PlanError(f"E_SOURCE_ROOT_ESCAPE: {value}") from error
        relatives.append(relative.as_posix())
    validate_manifest_at_root(
        manifest_path,
        source_root,
        relatives,
    )
