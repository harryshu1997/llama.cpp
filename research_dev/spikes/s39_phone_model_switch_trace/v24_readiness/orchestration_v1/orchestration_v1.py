#!/usr/bin/env python3
"""Build and run the bounded V2.4 A_ONLY orchestration plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Protocol


PLAN_SCHEMA = "s39-cp0-r1-v24-a-only-orchestration-plan-v2"
CONFIG_SCHEMA = "s39-cp0-r1-v24-a-only-orchestration-config-v2"
PREFLIGHT_SCHEMA = "s39-cp0-r1-v24-a-only-orchestration-preflight-v2"
RUN_SCHEMA = "s39-cp0-r1-v24-a-only-orchestration-result-v2"
RECEIPT_SCHEMA = "s39-cp0-r1-v24-stage-receipt-v1"
PHASE = "A_ONLY"
MODEL_ID = "qwen3-14b-q4_k_m"
CONFIRMATION = "RUN_CP0_R1_V24_A_ONLY"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TIMEOUT_SECONDS = 4 * 60 * 60

INPUT_NAMES = (
    "candidate",
    "contract",
    "cuda_monolithic_launch",
    "cuda_route_launch",
    "joint_capture_plan",
    "phone_route_launch",
    "prospective_root",
    "quality_corpus",
    "runtime_plan",
    "token_history",
    "tokenizer_plan",
)
STAGE_ORDER = (
    "artifact_root",
    "preparation",
    "phase_lock",
    "identity_binding",
    "fresh_readiness",
    "readiness_projection",
    "cuda_monolithic",
    "joint_phone_cuda",
    "fan_in",
    "authority",
)
PRODUCER_STAGES = {"cuda_monolithic", "joint_phone_cuda"}
ORCHESTRATION_SOURCE_STAGES = (
    "preparation",
    "phase_lock",
    "identity_binding",
    "readiness_projection",
    "fan_in",
)
OUTPUTS = {
    "artifact_root": (
        "artifact/artifact_root.json",
        "s39-cp0-r1-artifact-root-v2.4",
    ),
    "preparation": (
        "pre/reboot-preparation.json",
        "s39-cp0-r1-reboot-preparation-v2.4",
    ),
    "phase_lock": ("pre/phase_lock.jsonl", "s39-cp0-r1-phase-lock-v2.4"),
    "bound_cuda_route_launch": (
        "bound/cuda-route-launch.json",
        "s39-cp0-r1-v24-cuda-route-launch-v1",
    ),
    "bound_joint_capture_plan": (
        "bound/joint-capture-plan.json",
        "s39-cp0-r1-v24-joint-capture-plan-v1",
    ),
    "bound_phone_route_launch": (
        "bound/phone-route-launch.json",
        "s39-cp0-r1-v24-phone-route-launch-v1",
    ),
    "bound_root": (
        "bound/bound-runtime-root.json",
        "s39-cp0-r1-v24-bound-runtime-root-v1",
    ),
    "bound_runtime_plan": (
        "bound/runtime-bundle-plan.json",
        "s39-cp0-r1-runtime-bundle-plan-v2.4",
    ),
    "identity_binding_receipt": (
        "bound/identity-binding-receipt.json",
        "s39-cp0-r1-v24-identity-binding-receipt-v1",
    ),
    "fresh": (
        "fresh/fresh_snapshot.json",
        "s39-cp0-r1-fast-fresh-readiness-v2.4",
    ),
    "cuda_monolithic": (
        "capture/cuda-monolithic.json",
        "s39-cp0-r1-v24-cuda-monolithic-raw-v1",
    ),
    "joint_phone_cuda": (
        "capture/joint-phone-cuda.json",
        "s39-cp0-r1-v24-joint-phone-cuda-raw-v1",
    ),
    "runtime_identity": (
        "runtime-identity.json",
        "s39-cp0-r1-runtime-identity-v2.4",
    ),
    "acquisition": (
        "acquisition.json",
        "s39-cp0-r1-a-only-acquisition-v2.4",
    ),
}
STAGE_OUTPUTS = {
    "artifact_root": ("artifact_root",),
    "preparation": ("preparation",),
    "phase_lock": ("phase_lock",),
    "identity_binding": (
        "bound_cuda_route_launch",
        "bound_joint_capture_plan",
        "bound_phone_route_launch",
        "bound_root",
        "bound_runtime_plan",
        "identity_binding_receipt",
    ),
    "fresh_readiness": ("fresh",),
    "readiness_projection": (),
    "cuda_monolithic": ("cuda_monolithic",),
    "joint_phone_cuda": ("joint_phone_cuda",),
    "fan_in": ("runtime_identity", "acquisition"),
    "authority": (),
}
STDOUT_EXPECTATIONS = {
    "identity_binding": (
        "s39-cp0-r1-v24-identity-binding-attestation-v1",
        "POST_REBOOT_IDENTITY_BINDING_PASS",
    ),
    "readiness_projection": (
        "s39-cp0-r1-v24-pre-acquisition-projection-v1",
        "V2_4_PRE_ACQUISITION_READINESS_PASS",
    ),
    "authority": (
        "s39-cp0-r1-evidence-result-v2.4",
        "MODEL_A_QUALIFICATION_PASS_V2_4",
    ),
}
REQUIRED_PLACEHOLDERS = {
    "artifact_root": {
        "{artifact_root}",
        "{candidate}",
        "{contract}",
        "{runtime_plan}",
        "{token_history}",
        "{tokenizer_plan}",
    },
    "preparation": {
        "{artifact_root}",
        "{contract}",
        "{preparation}",
        "{runtime_plan}",
    },
    "phase_lock": {
        "{artifact_root}",
        "{candidate}",
        "{contract}",
        "{phase_id}",
        "{phase_lock}",
        "{preparation}",
        "{quality_corpus}",
        "{runtime_plan}",
    },
    "identity_binding": {
        "{bound_cuda_route_launch}",
        "{bound_joint_capture_plan}",
        "{bound_phone_route_launch}",
        "{bound_root}",
        "{bound_runtime_plan}",
        "{contract}",
        "{cuda_route_launch}",
        "{identity_binding_receipt}",
        "{joint_capture_plan}",
        "{phase_lock}",
        "{phone_route_launch}",
        "{preparation}",
        "{prospective_root}",
        "{runtime_plan}",
    },
    "fresh_readiness": {
        "{artifact_root}",
        "{contract}",
        "{fresh}",
        "{phase_id}",
        "{phase_lock}",
        "{preparation}",
        "{bound_runtime_plan}",
    },
    "readiness_projection": {
        "{acquisition_started_ns}",
        "{artifact_root}",
        "{candidate}",
        "{contract}",
        "{fresh}",
        "{phase_lock}",
        "{preparation}",
        "{bound_runtime_plan}",
        "{token_history}",
        "{tokenizer_plan}",
    },
    "cuda_monolithic": {
        "{acquisition_started_ns}",
        "{cuda_monolithic}",
        "{cuda_monolithic_launch}",
        "{cuda_monolithic_launch_sha256}",
        "{mechanism_commands_sha256}",
        "{model_sha256}",
        "{orchestration_plan_sha256}",
        "{phase_id}",
        "{pre_dir}",
        "{token_history}",
        "{token_history_sha256}",
    },
    "joint_phone_cuda": {
        "{acquisition_started_ns}",
        "{bound_joint_capture_plan}",
        "{bound_joint_capture_plan_sha256}",
        "{joint_phone_cuda}",
        "{orchestration_plan_sha256}",
        "{phase_id}",
        "{pre_dir}",
    },
    "fan_in": {
        "{acquisition}",
        "{acquisition_started_ns}",
        "{artifact_root}",
        "{bound_runtime_plan}",
        "{bundle_root}",
        "{candidate}",
        "{contract}",
        "{cuda_monolithic}",
        "{fresh}",
        "{joint_phone_cuda}",
        "{phase_lock}",
        "{pre_dir}",
        "{preparation}",
        "{runtime_identity}",
    },
    "authority": {
        "{acquisition}",
        "{artifact_root}",
        "{bundle_root}",
        "{candidate}",
        "{contract}",
        "{fresh}",
        "{bound_root}",
        "{identity_binding_receipt}",
        "{identity_binding_stage_receipt}",
        "{orchestration_plan}",
        "{phase_lock}",
        "{preparation}",
        "{prospective_root}",
        "{runtime_identity}",
        "{bound_runtime_plan}",
        "{token_history}",
        "{tokenizer_plan}",
    },
}
DIGEST_RE = re.compile(r"[0-9a-f]{64}")


class OrchestrationError(ValueError):
    pass


class CommandRunner(Protocol):
    def run(
        self,
        argv: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        timeout: int,
    ) -> subprocess.CompletedProcess[bytes]:
        ...


class SubprocessRunner:
    def run(
        self,
        argv: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        timeout: int,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            capture_output=True,
            check=False,
            timeout=timeout,
        )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OrchestrationError(message)


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


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(type(value) is int and minimum <= value < (1 << 63), f"E_INT: {field}")
    return value


def text(value: Any, field: str) -> str:
    require(type(value) is str and bool(value) and value.isascii(), f"E_TEXT: {field}")
    return value


def digest(value: Any, field: str) -> str:
    value = text(value, field)
    require(DIGEST_RE.fullmatch(value) is not None, f"E_DIGEST: {field}")
    return value


def absolute_path(value: Any, field: str) -> Path:
    value = text(value, field)
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, f"E_PATH: {field}")
    return path


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise OrchestrationError(f"E_JSON_NUMBER: {value}")


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
        raise OrchestrationError("E_CANONICAL") from error


def parse_json(raw: bytes, field: str) -> Any:
    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OrchestrationError(f"E_JSON: {field}: {error}") from error


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def secure_read(path: Path, field: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OrchestrationError(f"E_READ: {field}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"E_REGULAR: {field}")
        require(0 < before.st_size <= MAX_FILE_BYTES, f"E_SIZE: {field}")
        data = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            data.extend(block)
            require(len(data) <= MAX_FILE_BYTES, f"E_SIZE: {field}")
        after = os.fstat(descriptor)
        require(
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ),
            f"E_CHANGED_DURING_READ: {field}",
        )
        require(len(data) == before.st_size, f"E_SHORT_READ: {field}")
        return bytes(data), before
    finally:
        os.close(descriptor)


def stat_record(value: os.stat_result) -> dict[str, int]:
    return {
        "ctime_ns": value.st_ctime_ns,
        "device_id": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "mtime_ns": value.st_mtime_ns,
        "size": value.st_size,
    }


def file_record(path: Path, field: str) -> dict[str, Any]:
    require(path.is_absolute(), f"E_PATH: {field}")
    raw, value = secure_read(path, field)
    return {
        "bytes": len(raw),
        "path": str(path),
        "sha256": sha256_bytes(raw),
        "stat": stat_record(value),
    }


def validate_file_record(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, {"bytes", "path", "sha256", "stat"}, field)
    path = absolute_path(value["path"], f"{field}.path")
    integer(value["bytes"], f"{field}.bytes", 1)
    digest(value["sha256"], f"{field}.sha256")
    stat_value = exact_keys(
        value["stat"],
        {"ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"},
        f"{field}.stat",
    )
    for key in stat_value:
        integer(stat_value[key], f"{field}.stat.{key}")
    require(stat_value["size"] > 0 and stat_value["inode"] > 0, f"E_STAT: {field}")
    raw, current = secure_read(path, field)
    exact(len(raw), value["bytes"], f"{field}.bytes")
    exact(sha256_bytes(raw), value["sha256"], f"{field}.sha256")
    exact(stat_record(current), stat_value, f"{field}.stat")
    return value


def read_canonical(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    raw, _ = secure_read(path, field)
    value = parse_json(raw, field)
    require(type(value) is dict, f"E_TYPE: {field}")
    exact(canonical_bytes(value), raw, f"{field}.canonical")
    return value, raw


def write_exclusive(path: Path, raw: bytes) -> None:
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
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _json_schema(path: Path, expected: str, field: str) -> tuple[dict[str, Any], bytes]:
    value, raw = read_canonical(path, field)
    exact(value.get("schema"), expected, f"{field}.schema")
    return value, raw


def _command_source_pin(contract: dict[str, Any], stage: str) -> dict[str, Any] | None:
    if stage == "authority":
        return contract["exit_authority"]["entrypoint"]
    if stage in ORCHESTRATION_SOURCE_STAGES:
        return contract["orchestration_requirements"]["source_programs"][stage]
    if stage in PRODUCER_STAGES:
        name = "cuda_monolithic" if stage == "cuda_monolithic" else "joint_phone_cuda"
        return contract["producer_requirements"]["source_programs"][name]
    return None


def _validate_environment(value: Any, field: str) -> dict[str, str]:
    require(type(value) is dict, f"E_TYPE: {field}")
    result = {}
    for key in sorted(value):
        result[text(key, f"{field}.key")] = text(value[key], f"{field}.{key}")
    exact(value, result, f"{field}.order")
    return value


def _placeholders(argv: list[str], field: str) -> set[str]:
    result = set()
    for index, item in enumerate(argv):
        item = text(item, f"{field}[{index}]")
        if "{" in item or "}" in item:
            require(
                item.startswith("{")
                and item.endswith("}")
                and item.count("{") == 1
                and item.count("}") == 1,
                f"E_PLACEHOLDER: {field}[{index}]",
            )
            result.add(item)
    return result


def _option(argv: list[str], name: str, expected: str, field: str) -> None:
    require(argv.count(name) == 1, f"E_OPTION_COUNT: {field}: {name}")
    index = argv.index(name)
    require(index + 1 < len(argv), f"E_OPTION_VALUE: {field}: {name}")
    exact(argv[index + 1], expected, f"{field}.{name}")


def _validate_stage(
    value: Any,
    stage: str,
    contract: dict[str, Any],
    contract_root: Path,
) -> dict[str, Any]:
    field = f"plan.stages.{stage}"
    value = exact_keys(
        value,
        {
            "argv_template",
            "cwd",
            "entrypoint",
            "environment",
            "support_files",
            "timeout_seconds",
        },
        field,
    )
    validate_file_record(value["entrypoint"], f"{field}.entrypoint")
    support_files = value["support_files"]
    require(type(support_files) is list, f"E_TYPE: {field}.support_files")
    support_paths = []
    for index, record in enumerate(support_files):
        validate_file_record(record, f"{field}.support_files[{index}]")
        support_paths.append(record["path"])
    require(
        support_paths == sorted(set(support_paths)),
        f"E_SUPPORT_FILES: {stage}",
    )
    cwd = absolute_path(value["cwd"], f"{field}.cwd")
    require(cwd.is_dir(), f"E_CWD: {field}")
    _validate_environment(value["environment"], f"{field}.environment")
    timeout = integer(value["timeout_seconds"], f"{field}.timeout", 1)
    require(timeout <= MAX_TIMEOUT_SECONDS, f"E_TIMEOUT: {field}")
    argv = value["argv_template"]
    require(type(argv) is list and bool(argv), f"E_TYPE: {field}.argv")
    used = _placeholders(argv, f"{field}.argv")
    require(
        REQUIRED_PLACEHOLDERS[stage] <= used,
        f"E_REQUIRED_PLACEHOLDER: {stage}: "
        f"{sorted(REQUIRED_PLACEHOLDERS[stage] - used)}",
    )
    allowed = {
        "{acquisition}",
        "{acquisition_started_ns}",
        "{artifact_root}",
        "{bound_cuda_route_launch}",
        "{bound_joint_capture_plan}",
        "{bound_joint_capture_plan_sha256}",
        "{bound_phone_route_launch}",
        "{bound_root}",
        "{bound_runtime_plan}",
        "{bundle_root}",
        "{candidate}",
        "{contract}",
        "{cuda_monolithic}",
        "{cuda_monolithic_launch}",
        "{cuda_monolithic_launch_sha256}",
        "{cuda_route_launch}",
        "{fresh}",
        "{identity_binding_receipt}",
        "{identity_binding_stage_receipt}",
        "{joint_capture_plan}",
        "{joint_phone_cuda}",
        "{mechanism_commands_sha256}",
        "{model_sha256}",
        "{orchestration_plan_sha256}",
        "{orchestration_plan}",
        "{phase_id}",
        "{phase_lock}",
        "{phone_route_launch}",
        "{pre_dir}",
        "{preparation}",
        "{prospective_root}",
        "{quality_corpus}",
        "{run_root}",
        "{runtime_identity}",
        "{runtime_plan}",
        "{token_history}",
        "{token_history_sha256}",
        "{tokenizer_plan}",
    }
    require(used <= allowed, f"E_UNKNOWN_PLACEHOLDER: {stage}: {sorted(used - allowed)}")
    source_pin = _command_source_pin(contract, stage)
    if source_pin is not None:
        exact(
            value["entrypoint"]["bytes"],
            source_pin["bytes"],
            f"E_SOURCE_PIN_BYTES: {stage}",
        )
        exact(
            value["entrypoint"]["sha256"],
            source_pin["sha256"],
            f"E_SOURCE_PIN_SHA256: {stage}",
        )
        exact(
            Path(value["entrypoint"]["path"]),
            (contract_root / source_pin["path"]).resolve(),
            f"E_SOURCE_PIN_PATH: {stage}",
        )
    if stage == "authority":
        expected_support = sorted(
            contract["exit_authority"]["support"].values(),
            key=lambda record: record["path"],
        )
        exact(
            sorted(
                (record["bytes"], record["path"], record["sha256"])
                for record in support_files
            ),
            sorted(
                (
                    record["bytes"],
                    str((contract_root / record["path"]).resolve()),
                    record["sha256"],
                )
                for record in expected_support
            ),
            "E_AUTHORITY_SUPPORT_PINS",
        )
        _option(argv, "--orchestration-plan", "{orchestration_plan}", field)
    if stage in ORCHESTRATION_SOURCE_STAGES:
        requirements = contract["orchestration_requirements"]
        support = requirements["support"]
        expected_support = {
            name: support[name]
            for name in requirements["stage_support"][stage]
        }
        exact(
            sorted(
                (record["bytes"], record["path"], record["sha256"])
                for record in support_files
            ),
            sorted(
                (
                    record["bytes"],
                    str((contract_root / record["path"]).resolve()),
                    record["sha256"],
                )
                for record in expected_support.values()
            ),
            f"E_ORCHESTRATION_SUPPORT_PINS: {stage}",
        )
    if stage in PRODUCER_STAGES:
        exact(support_files, [], f"E_PRODUCER_NOT_SELF_CONTAINED: {stage}")
    if stage == "cuda_monolithic":
        _option(argv, "--output", "{cuda_monolithic}", field)
        _option(argv, "--phase-id", "{phase_id}", field)
        _option(argv, "--pre-dir", "{pre_dir}", field)
        _option(argv, "--started", "{acquisition_started_ns}", field)
        _option(argv, "--plan", "{orchestration_plan_sha256}", field)
        _option(argv, "--mechanism-commands-sha256", "{mechanism_commands_sha256}", field)
        _option(argv, "--model-sha256", "{model_sha256}", field)
        _option(argv, "--histories", "{token_history}", field)
        _option(argv, "--histories-sha256", "{token_history_sha256}", field)
        _option(argv, "--launch-plan", "{cuda_monolithic_launch}", field)
        _option(
            argv,
            "--launch-plan-sha256",
            "{cuda_monolithic_launch_sha256}",
            field,
        )
        _option(argv, "--confirm", "RUN_V24_CUDA_MONOLITHIC_A_ONLY", field)
        require(argv.count("--execute") == 1, f"E_EXECUTE_FLAG: {stage}")
    if stage == "joint_phone_cuda":
        _option(argv, "--capture-plan", "{bound_joint_capture_plan}", field)
        _option(
            argv,
            "--capture-plan-sha256",
            "{bound_joint_capture_plan_sha256}",
            field,
        )
        _option(argv, "--output", "{joint_phone_cuda}", field)
        _option(argv, "--phase-id", "{phase_id}", field)
        _option(argv, "--pre-dir", "{pre_dir}", field)
        _option(argv, "--acquisition-started-ns", "{acquisition_started_ns}", field)
        _option(argv, "--command-plan-sha256", "{orchestration_plan_sha256}", field)
        _option(argv, "--confirm", "RUN_V24_JOINT_PHONE_CUDA_A_ONLY", field)
        require(argv.count("--execute") == 1, f"E_EXECUTE_FLAG: {stage}")
    return value


def _validate_input_schemas(inputs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    expected = {
        "candidate": "s39-cp0-r1-candidate-v1",
        "contract": "s39-cp0-r1-evidence-contract-v2.4",
        "cuda_monolithic_launch": "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
        "cuda_route_launch": "s39-cp0-r1-v24-cuda-route-launch-v1",
        "joint_capture_plan": "s39-cp0-r1-v24-joint-capture-plan-v1",
        "phone_route_launch": "s39-cp0-r1-v24-phone-route-launch-v1",
        "prospective_root": "s39-cp0-r1-v24-prospective-runtime-root-v1",
        "runtime_plan": "s39-cp0-r1-runtime-bundle-plan-v2.4",
        "token_history": "s39-cp0-r1-token-history-v2.4",
        "tokenizer_plan": "s39-cp0-r1-a-only-tokenizer-plan-v2",
    }
    result = {
        name: _json_schema(Path(inputs[name]["path"]), schema, f"input.{name}")[0]
        for name, schema in expected.items()
    }
    return result


def _validate_cross_bindings(
    inputs: dict[str, dict[str, Any]],
    values: dict[str, dict[str, Any]],
    stages: dict[str, dict[str, Any]],
) -> str:
    contract = values["contract"]
    contract_root = Path(inputs["contract"]["path"]).parent.parent
    candidate = values["candidate"]
    runtime = values["runtime_plan"]
    exact(runtime["contract_sha256"], inputs["contract"]["sha256"], "runtime.contract")
    exact(runtime["candidate_sha256"], inputs["candidate"]["sha256"], "runtime.candidate")
    exact(runtime["cuda_monolithic_launch"], values["cuda_monolithic_launch"], "runtime.cuda_launch")
    exact(
        runtime["token_history"]["artifact_path"],
        inputs["token_history"]["path"],
        "runtime.token_history.path",
    )
    exact(
        runtime["token_history"]["tokenizer_plan_path"],
        inputs["tokenizer_plan"]["path"],
        "runtime.tokenizer_plan.path",
    )
    exact(
        runtime["token_history"]["tokenizer_plan_sha256"],
        inputs["tokenizer_plan"]["sha256"],
        "runtime.tokenizer_plan.sha256",
    )
    model = next(
        (
            value
            for value in candidate["models"]
            if value.get("slot") == "A" and value.get("model_id") == MODEL_ID
        ),
        None,
    )
    require(type(model) is dict, "E_MODEL_A")
    model_sha256 = digest(model["artifact"]["sha256"], "candidate.model.sha256")
    quality = contract["quality_corpus"]
    exact(
        inputs["quality_corpus"]["bytes"],
        quality["bytes"],
        "quality_corpus.bytes",
    )
    exact(
        inputs["quality_corpus"]["sha256"],
        quality["sha256"],
        "quality_corpus.sha256",
    )
    expected_corpus_path = (
        Path(inputs["contract"]["path"]).parent.parent / quality["path"]
    ).resolve()
    exact(
        Path(inputs["quality_corpus"]["path"]),
        expected_corpus_path,
        "quality_corpus.path",
    )
    exact(runtime["token_history"]["model_sha256"], model_sha256, "runtime.history.model")
    history = values["token_history"]
    exact(history["model_sha256"], model_sha256, "history.model")
    joint = values["joint_capture_plan"]
    exact(joint["history"]["sha256"], inputs["token_history"]["sha256"], "joint.history")
    prospective = values["prospective_root"]
    exact(prospective["acquisition_ready"], False, "prospective.ready")
    exact(prospective["phase"], PHASE, "prospective.phase")
    exact(prospective["model_id"], MODEL_ID, "prospective.model")
    for name in (
        "cuda_route_launch",
        "joint_capture_plan",
        "phone_route_launch",
        "runtime_plan",
    ):
        exact(
            prospective["artifacts"][name]["sha256"],
            inputs[name]["sha256"],
            f"prospective.artifacts.{name}.sha256",
        )
        exact(
            prospective["artifacts"][name]["path"],
            inputs[name]["path"],
            f"prospective.artifacts.{name}.path",
        )
    exact(
        joint["commands"]["cuda"]["launch_plan_sha256"],
        inputs["cuda_route_launch"]["sha256"],
        "joint.cuda_launch",
    )
    exact(
        joint["commands"]["phone"]["launch_plan_sha256"],
        inputs["phone_route_launch"]["sha256"],
        "joint.phone_launch",
    )
    mechanism_sha256 = sha256_bytes(canonical_bytes(joint["mechanism_commands"]))
    captures = {
        value["kind"]: value for value in runtime["capture_entrypoints"]
    }
    components = {
        value["component_id"]: value for value in runtime["components"]
    }
    for stage, kind in (
        ("artifact_root", "artifact_root"),
        ("fresh_readiness", "fast_fresh_readiness"),
        ("cuda_monolithic", "cuda_monolithic"),
        ("joint_phone_cuda", "joint_phone_cuda"),
    ):
        component = components[captures[kind]["component_id"]]
        exact(
            stages[stage]["entrypoint"]["bytes"],
            component["bytes"],
            f"E_RUNTIME_CAPTURE_BYTES: {stage}",
        )
        exact(
            stages[stage]["entrypoint"]["sha256"],
            component["sha256"],
            f"E_RUNTIME_CAPTURE_SHA256: {stage}",
        )
    exact(contract["schema"], "s39-cp0-r1-evidence-contract-v2.4", "contract.schema")
    return mechanism_sha256


def _build_plan(config: dict[str, Any]) -> dict[str, Any]:
    exact_keys(
        config,
        {
            "commands",
            "inputs",
            "model_id",
            "phase",
            "phase_id",
            "run_root",
            "schema",
        },
        "config",
    )
    exact(config["schema"], CONFIG_SCHEMA, "config.schema")
    exact(config["phase"], PHASE, "config.phase")
    exact(config["model_id"], MODEL_ID, "config.model")
    phase_id = text(config["phase_id"], "config.phase_id")
    require(
        phase_id.startswith("cp0-r1-v24-a-only-") and len(phase_id) <= 128,
        "E_PHASE_ID",
    )
    run_root = absolute_path(config["run_root"], "config.run_root")
    inputs_config = exact_keys(config["inputs"], set(INPUT_NAMES), "config.inputs")
    inputs = {
        name: file_record(absolute_path(inputs_config[name], f"config.inputs.{name}"), name)
        for name in INPUT_NAMES
    }
    values = _validate_input_schemas(inputs)
    contract = values["contract"]
    contract_root = Path(inputs["contract"]["path"]).parent.parent
    commands = exact_keys(config["commands"], set(STAGE_ORDER), "config.commands")
    stages = {}
    for stage in STAGE_ORDER:
        command = exact_keys(
            commands[stage],
            {
                "argv_template",
                "cwd",
                "entrypoint",
                "environment",
                "support_files",
                "timeout_seconds",
            },
            f"config.commands.{stage}",
        )
        command["entrypoint"] = file_record(
            absolute_path(command["entrypoint"], f"config.commands.{stage}.entrypoint"),
            f"command.{stage}.entrypoint",
        )
        support = [
            file_record(
                absolute_path(path, f"config.commands.{stage}.support_files[{index}]"),
                f"command.{stage}.support_files[{index}]",
            )
            for index, path in enumerate(command["support_files"])
        ]
        command["support_files"] = sorted(support, key=lambda record: record["path"])
        stages[stage] = _validate_stage(
            command,
            stage,
            contract,
            contract_root,
        )
    mechanism_sha256 = _validate_cross_bindings(inputs, values, stages)
    python = file_record(Path(sys.executable).resolve(), "python")
    return {
        "inputs": inputs,
        "mechanism_commands_sha256": mechanism_sha256,
        "model_id": MODEL_ID,
        "model_sha256": values["token_history"]["model_sha256"],
        "phase": PHASE,
        "phase_id": phase_id,
        "python": python,
        "run_root": str(run_root),
        "schema": PLAN_SCHEMA,
        "stages": stages,
    }


def build_plan(config_path: Path, output_path: Path) -> dict[str, Any]:
    config, _ = read_canonical(config_path, "config")
    require(not output_path.exists(), "E_OUTPUT_EXISTS")
    plan = _build_plan(config)
    write_exclusive(output_path, canonical_bytes(plan))
    return plan


def load_plan(path: Path) -> tuple[dict[str, Any], bytes]:
    plan, raw = read_canonical(path, "plan")
    exact_keys(
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
        "plan",
    )
    exact(plan["schema"], PLAN_SCHEMA, "plan.schema")
    exact(plan["phase"], PHASE, "plan.phase")
    exact(plan["model_id"], MODEL_ID, "plan.model")
    digest(plan["model_sha256"], "plan.model_sha256")
    digest(plan["mechanism_commands_sha256"], "plan.mechanism")
    phase_id = text(plan["phase_id"], "plan.phase_id")
    require(
        phase_id.startswith("cp0-r1-v24-a-only-") and len(phase_id) <= 128,
        "E_PHASE_ID",
    )
    absolute_path(plan["run_root"], "plan.run_root")
    validate_file_record(plan["python"], "plan.python")
    inputs = exact_keys(plan["inputs"], set(INPUT_NAMES), "plan.inputs")
    for name in INPUT_NAMES:
        validate_file_record(inputs[name], f"plan.inputs.{name}")
    values = _validate_input_schemas(inputs)
    contract_root = Path(inputs["contract"]["path"]).parent.parent
    stages = exact_keys(plan["stages"], set(STAGE_ORDER), "plan.stages")
    for stage in STAGE_ORDER:
        _validate_stage(
            stages[stage],
            stage,
            values["contract"],
            contract_root,
        )
    exact(
        _validate_cross_bindings(inputs, values, stages),
        plan["mechanism_commands_sha256"],
        "plan.mechanism",
    )
    exact(values["token_history"]["model_sha256"], plan["model_sha256"], "plan.model")
    return plan, raw


def _output_paths(run_root: Path) -> dict[str, Path]:
    values = {
        name: run_root / relative
        for name, (relative, _) in OUTPUTS.items()
    }
    values["bundle_root"] = run_root / "raw-bundle"
    values["identity_binding_stage_receipt"] = (
        run_root / "receipts" / "identity_binding" / "receipt.json"
    )
    values["pre_dir"] = run_root / "pre"
    values["run_root"] = run_root
    return values


def _render_context(
    plan: dict[str, Any],
    plan_raw: bytes,
    plan_path: Path,
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    acquisition_started_ns: int | None,
    mechanism_commands_sha256: str | None = None,
    bound_joint_capture_plan_sha256: str | None = None,
) -> dict[str, str]:
    context = {
        **{name: str(path) for name, path in input_paths.items()},
        **{name: str(path) for name, path in output_paths.items()},
        "mechanism_commands_sha256": (
            mechanism_commands_sha256
            if mechanism_commands_sha256 is not None
            else plan["mechanism_commands_sha256"]
        ),
        "bound_joint_capture_plan_sha256": (
            bound_joint_capture_plan_sha256
            if bound_joint_capture_plan_sha256 is not None
            else "<bound-joint-capture-plan-sha256>"
        ),
        "model_sha256": plan["model_sha256"],
        "cuda_monolithic_launch_sha256": plan["inputs"][
            "cuda_monolithic_launch"
        ]["sha256"],
        "token_history_sha256": plan["inputs"]["token_history"]["sha256"],
        "orchestration_plan": str(plan_path),
        "orchestration_plan_sha256": sha256_bytes(plan_raw),
        "phase_id": plan["phase_id"],
    }
    context["acquisition_started_ns"] = (
        str(acquisition_started_ns)
        if acquisition_started_ns is not None
        else "<acquisition-started-ns>"
    )
    return {"{" + key + "}": value for key, value in context.items()}


def _render(argv: list[str], context: dict[str, str], field: str) -> list[str]:
    result = []
    for index, value in enumerate(argv):
        if value.startswith("{") and value.endswith("}"):
            require(value in context, f"E_RENDER: {field}[{index}]: {value}")
            result.append(context[value])
        else:
            require("{" not in value and "}" not in value, f"E_RENDER: {field}[{index}]")
            result.append(value)
    return result


def _preview(
    plan: dict[str, Any],
    plan_raw: bytes,
    plan_path: Path,
) -> list[dict[str, Any]]:
    outputs = _output_paths(Path(plan["run_root"]))
    inputs = {
        name: Path(record["path"]) for name, record in plan["inputs"].items()
    }
    context = _render_context(plan, plan_raw, plan_path, inputs, outputs, None)
    return [
        {
            "argv": _render(
                plan["stages"][stage]["argv_template"],
                context,
                f"preview.{stage}",
            ),
            "cwd": plan["stages"][stage]["cwd"],
            "stage": stage,
        }
        for stage in STAGE_ORDER
    ]


def preflight(plan_path: Path) -> dict[str, Any]:
    plan, raw = load_plan(plan_path)
    return {
        "phase": PHASE,
        "phase_id": plan["phase_id"],
        "plan_sha256": sha256_bytes(raw),
        "schema": PREFLIGHT_SCHEMA,
        "stages": _preview(plan, raw, plan_path),
        "status": "NO_MODEL_PREFLIGHT_PASS_STAGED_IDENTITY_BINDING_REQUIRED",
        "unresolved_provenance": [
            "cuda.boot_id",
            "op12.boot_id",
            "op15.boot_id",
        ],
    }


def _copy_pinned(record: dict[str, Any], destination: Path, field: str) -> None:
    validate_file_record(record, field)
    raw, _ = secure_read(Path(record["path"]), field)
    write_exclusive(destination, raw)


def _capture_inputs(
    plan: dict[str, Any],
    run_root: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    input_paths = {}
    for name, record in plan["inputs"].items():
        destination = run_root / "inputs" / f"{name}.json"
        if name == "quality_corpus":
            destination = run_root / "pre" / "quality_corpus.jsonl"
        _copy_pinned(record, destination, f"capture.input.{name}")
        input_paths[name] = Path(record["path"])
    entrypoints = {}
    for stage in STAGE_ORDER:
        record = plan["stages"][stage]["entrypoint"]
        destination = run_root / "executed" / stage / "entrypoint.py"
        _copy_pinned(record, destination, f"capture.entrypoint.{stage}")
        for index, support in enumerate(plan["stages"][stage]["support_files"]):
            _copy_pinned(
                support,
                run_root / "executed" / stage / "support" / f"{index:03d}-{Path(support['path']).name}",
                f"capture.support.{stage}[{index}]",
            )
        entrypoints[stage] = Path(record["path"])
    return input_paths, entrypoints


def _validate_stage_outputs(
    stage: str,
    plan: dict[str, Any],
    output_paths: dict[str, Path],
) -> None:
    for name in STAGE_OUTPUTS[stage]:
        path = output_paths[name]
        value, _ = _json_schema(path, OUTPUTS[name][1], f"output.{stage}.{name}")
        if "phase" in value:
            exact(value["phase"], PHASE, f"output.{stage}.{name}.phase")
        if "phase_id" in value:
            exact(
                value["phase_id"],
                plan["phase_id"],
                f"output.{stage}.{name}.phase_id",
            )
    if stage == "fan_in":
        bundle_root = output_paths["bundle_root"]
        require(bundle_root.is_dir() and not bundle_root.is_symlink(), "E_BUNDLE_ROOT")
        manifest = bundle_root / "EVIDENCE_BUNDLE_V2_4.json"
        require(manifest.is_file() and not manifest.is_symlink(), "E_RAW_MANIFEST")
        read_canonical(manifest, "raw_manifest")


def _bound_record(
    value: Any,
    path: Path,
    raw: bytes,
    field: str,
) -> None:
    record = exact_keys(value, {"bytes", "path", "sha256"}, field)
    exact(absolute_path(record["path"], f"{field}.path"), path, f"{field}.path")
    exact(integer(record["bytes"], f"{field}.bytes", 1), len(raw), f"{field}.bytes")
    exact(digest(record["sha256"], f"{field}.sha256"), sha256_bytes(raw), f"{field}.sha256")


def _pin_bound_outputs(
    output_paths: dict[str, Path],
    attestation: Any,
) -> dict[str, bytes]:
    attestation = exact_keys(
        attestation,
        {
            "bound_root_sha256",
            "identity_binding_receipt_sha256",
            "schema",
            "status",
        },
        "bound_pin.attestation",
    )
    exact(
        attestation["schema"],
        "s39-cp0-r1-v24-identity-binding-attestation-v1",
        "bound_pin.attestation.schema",
    )
    exact(
        attestation["status"],
        "POST_REBOOT_IDENTITY_BINDING_PASS",
        "bound_pin.attestation.status",
    )
    root_path = output_paths["bound_root"]
    receipt_path = output_paths["identity_binding_receipt"]
    root, root_raw = read_canonical(root_path, "bound_pin.root")
    receipt, receipt_raw = read_canonical(receipt_path, "bound_pin.receipt")
    exact(
        sha256_bytes(root_raw),
        digest(
            attestation["bound_root_sha256"],
            "bound_pin.attestation.bound_root_sha256",
        ),
        "E_BOUND_ATTESTATION: root",
    )
    exact(
        sha256_bytes(receipt_raw),
        digest(
            attestation["identity_binding_receipt_sha256"],
            "bound_pin.attestation.identity_binding_receipt_sha256",
        ),
        "E_BOUND_ATTESTATION: receipt",
    )
    exact(
        root.get("schema"),
        "s39-cp0-r1-v24-bound-runtime-root-v1",
        "bound_pin.root.schema",
    )
    exact(
        receipt.get("schema"),
        "s39-cp0-r1-v24-identity-binding-receipt-v1",
        "bound_pin.receipt.schema",
    )
    _bound_record(
        root["identity_binding_receipt"],
        receipt_path,
        receipt_raw,
        "bound_pin.root.receipt",
    )
    artifact_paths = {
        "cuda_route_launch": output_paths["bound_cuda_route_launch"],
        "joint_capture_plan": output_paths["bound_joint_capture_plan"],
        "phone_route_launch": output_paths["bound_phone_route_launch"],
        "runtime_plan": output_paths["bound_runtime_plan"],
    }
    exact(set(root["artifacts"]), set(artifact_paths), "bound_pin.root.artifacts")
    exact(set(receipt["outputs"]), set(artifact_paths), "bound_pin.receipt.outputs")
    pins = {
        "bound_root": root_raw,
        "identity_binding_receipt": receipt_raw,
    }
    for name, path in artifact_paths.items():
        raw, _ = secure_read(path, f"bound_pin.artifact.{name}")
        _bound_record(
            root["artifacts"][name],
            path,
            raw,
            f"bound_pin.root.artifacts.{name}",
        )
        _bound_record(
            receipt["outputs"][name],
            path,
            raw,
            f"bound_pin.receipt.outputs.{name}",
        )
        exact(
            root["artifacts"][name],
            receipt["outputs"][name],
            f"bound_pin.cross.{name}",
        )
        pins[name] = raw
    exact(
        root["mechanism_commands_sha256"],
        receipt["mechanism_commands_sha256"],
        "bound_pin.mechanism_commands_sha256",
    )
    return pins


def _revalidate_bound_outputs(
    pins: dict[str, bytes],
    output_paths: dict[str, Path],
    stage: str,
) -> None:
    paths = {
        "bound_root": output_paths["bound_root"],
        "cuda_route_launch": output_paths["bound_cuda_route_launch"],
        "identity_binding_receipt": output_paths["identity_binding_receipt"],
        "joint_capture_plan": output_paths["bound_joint_capture_plan"],
        "phone_route_launch": output_paths["bound_phone_route_launch"],
        "runtime_plan": output_paths["bound_runtime_plan"],
    }
    exact(set(pins), set(paths), f"pre_stage.{stage}.bound_pins")
    for name, path in paths.items():
        raw, _ = secure_read(path, f"pre_stage.{stage}.bound.{name}")
        exact(raw, pins[name], f"E_BOUND_MUTATION: {stage}.{name}")


def _validate_stdout(stage: str, raw: bytes) -> dict[str, Any] | None:
    if stage not in STDOUT_EXPECTATIONS:
        exact(raw, b"", f"E_STDOUT: {stage}")
        return None
    value = parse_json(raw, f"stdout.{stage}")
    require(type(value) is dict, f"E_STDOUT_TYPE: {stage}")
    exact(canonical_bytes(value), raw, f"E_STDOUT_CANONICAL: {stage}")
    schema, status = STDOUT_EXPECTATIONS[stage]
    exact(value.get("schema"), schema, f"stdout.{stage}.schema")
    exact(value.get("status"), status, f"stdout.{stage}.status")
    return value


def _run_stage(
    stage: str,
    plan: dict[str, Any],
    entrypoint: Path,
    context: dict[str, str],
    receipt_root: Path,
    runner: CommandRunner,
    now_ns: Callable[[], int],
) -> dict[str, Any] | None:
    command = plan["stages"][stage]
    validate_file_record(plan["python"], f"pre_stage.{stage}.python")
    validate_file_record(command["entrypoint"], f"pre_stage.{stage}.entrypoint")
    for index, support in enumerate(command["support_files"]):
        validate_file_record(support, f"pre_stage.{stage}.support[{index}]")
    argv = [
        plan["python"]["path"],
        "-B",
        str(entrypoint),
        *_render(command["argv_template"], context, f"run.{stage}"),
    ]
    started_ns = now_ns()
    try:
        completed = runner.run(
            argv,
            cwd=command["cwd"],
            env=command["environment"],
            timeout=command["timeout_seconds"],
        )
    except subprocess.TimeoutExpired as error:
        raise OrchestrationError(f"E_TIMEOUT: {stage}") from error
    completed_ns = now_ns()
    receipt_dir = receipt_root / stage
    write_exclusive(receipt_dir / "stdout", completed.stdout)
    write_exclusive(receipt_dir / "stderr", completed.stderr)
    receipt = {
        "argv": argv,
        "completed_ns": completed_ns,
        "returncode": completed.returncode,
        "schema": RECEIPT_SCHEMA,
        "stage": stage,
        "started_ns": started_ns,
    }
    write_exclusive(receipt_dir / "receipt.json", canonical_bytes(receipt))
    validate_file_record(plan["python"], f"post_stage.{stage}.python")
    validate_file_record(command["entrypoint"], f"post_stage.{stage}.entrypoint")
    for index, support in enumerate(command["support_files"]):
        validate_file_record(support, f"post_stage.{stage}.support[{index}]")
    exact(completed.returncode, 0, f"E_STAGE_EXIT: {stage}")
    return _validate_stdout(stage, completed.stdout)


def run_acquisition(
    plan_path: Path,
    *,
    runner: CommandRunner | None = None,
    now_ns: Callable[[], int] | None = None,
    test_only: bool = False,
) -> dict[str, Any]:
    require(runner is None or test_only, "E_TEST_RUNNER_REQUIRES_TEST_ONLY")
    plan, plan_raw = load_plan(plan_path)
    run_root = Path(plan["run_root"])
    require(not run_root.exists(), "E_RUN_ROOT_EXISTS")
    run_root.mkdir(parents=True)
    directory = os.open(run_root.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    input_paths, entrypoints = _capture_inputs(plan, run_root)
    output_paths = _output_paths(run_root)
    output_paths["pre_dir"].mkdir(parents=True, exist_ok=True)
    receipts = run_root / "receipts"
    runner = runner or SubprocessRunner()
    now_ns = now_ns or (lambda: time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
    acquisition_started_ns = None
    projection_result = None
    authority_result = None
    realized_mechanism_sha256 = None
    bound_pins = None
    bound_joint_capture_plan_sha256 = None
    completed_stages = []
    for stage in STAGE_ORDER:
        if bound_pins is not None:
            _revalidate_bound_outputs(bound_pins, output_paths, stage)
        if stage == "readiness_projection":
            acquisition_started_ns = now_ns()
        if stage == "fresh_readiness":
            require(
                "identity_binding" in completed_stages
                and output_paths["bound_root"].is_file()
                and output_paths["identity_binding_receipt"].is_file(),
                "E_FRESH_BEFORE_IDENTITY_BINDING",
            )
        if stage in PRODUCER_STAGES:
            require(
                projection_result is not None
                and projection_result["status"]
                == "V2_4_PRE_ACQUISITION_READINESS_PASS",
                "E_PRODUCER_BEFORE_FRESH_PROJECTION",
            )
        if stage == "fan_in":
            for producer in ("cuda_monolithic", "joint_phone_cuda"):
                require(
                    producer in completed_stages
                    and output_paths[producer].is_file(),
                    f"E_PRODUCER_RECEIPT_MISSING: {producer}",
                )
        if stage == "authority":
            require(
                all(
                    output_paths[name].is_file()
                    for name in (
                        "cuda_monolithic",
                        "joint_phone_cuda",
                        "runtime_identity",
                        "acquisition",
                    )
                ),
                "E_AUTHORITY_BEFORE_RECEIPTS",
            )
            require(
                (output_paths["bundle_root"] / "EVIDENCE_BUNDLE_V2_4.json").is_file(),
                "E_AUTHORITY_BEFORE_MANIFEST",
            )
        context = _render_context(
            plan,
            plan_raw,
            plan_path,
            input_paths,
            output_paths,
            acquisition_started_ns,
            realized_mechanism_sha256,
            bound_joint_capture_plan_sha256,
        )
        result = _run_stage(
            stage,
            plan,
            entrypoints[stage],
            context,
            receipts,
            runner,
            now_ns,
        )
        _validate_stage_outputs(stage, plan, output_paths)
        completed_stages.append(stage)
        if stage == "identity_binding":
            bound_pins = _pin_bound_outputs(output_paths, result)
            bound_joint_capture_plan_sha256 = sha256_bytes(
                bound_pins["joint_capture_plan"]
            )
            bound_root, _ = read_canonical(
                output_paths["bound_root"],
                "bound_root",
            )
            exact(
                bound_root["schema"],
                "s39-cp0-r1-v24-bound-runtime-root-v1",
                "bound_root.schema",
            )
            realized_mechanism_sha256 = digest(
                bound_root["mechanism_commands_sha256"],
                "bound_root.mechanism_commands_sha256",
            )
        elif stage == "readiness_projection":
            projection_result = result
        elif stage == "authority":
            authority_result = result
    require(authority_result is not None, "E_AUTHORITY_RESULT")
    write_exclusive(
        run_root / "AUTHORITY_RESULT.json",
        canonical_bytes(authority_result),
    )
    result = {
        "authority_result_sha256": sha256_bytes(canonical_bytes(authority_result)),
        "completed_stages": completed_stages,
        "phase": PHASE,
        "phase_id": plan["phase_id"],
        "plan_sha256": sha256_bytes(plan_raw),
        "schema": RUN_SCHEMA,
        "status": (
            "TEST_ONLY_SEQUENCE_PASS_NOT_ACQUISITION_EVIDENCE"
            if test_only
            else "A_ONLY_ACQUISITION_SEQUENCE_PASS"
        ),
    }
    write_exclusive(run_root / "ORCHESTRATION_RESULT.json", canonical_bytes(result))
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    check = subparsers.add_parser("preflight")
    check.add_argument("--plan", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--confirm")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.action == "build":
            result = build_plan(args.config.resolve(), args.output.resolve())
            print(canonical_bytes(result).decode("ascii"), end="")
        elif args.action == "preflight":
            print(canonical_bytes(preflight(args.plan.resolve())).decode("ascii"), end="")
        else:
            require(args.execute, "E_EXECUTE_REQUIRED")
            exact(args.confirm, CONFIRMATION, "confirm")
            print(
                canonical_bytes(run_acquisition(args.plan.resolve())).decode("ascii"),
                end="",
            )
        return 0
    except (OSError, OrchestrationError, subprocess.SubprocessError) as error:
        print(f"V24_A_ONLY_ORCHESTRATION_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
