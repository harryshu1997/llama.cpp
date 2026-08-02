#!/usr/bin/env python3
"""Shared implementation for the CP0-R1 V2.3 readiness producers."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import time
from typing import Any, Callable, Protocol


CLOCK_ID = time.CLOCK_MONOTONIC_RAW
PHASE = "A_ONLY"
MODEL_ID = "qwen3-14b-q4_k_m"
ENDPOINTS = ("cuda", "op15", "op12", "op15_worker", "op12_worker")
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
ANDROID_TIME_RE = re.compile(
    r"(?P<base>[0-9]{4}-[0-9]{2}-[0-9]{2} "
    r"[0-9]{2}:[0-9]{2}:[0-9]{2})"
    r"\.(?P<fraction>[0-9]{1,9}) (?P<zone>[+-][0-9]{4})"
)
ROUTE_KEYS = {
    "acquisition_id",
    "activation_dtype",
    "activation_element_bytes",
    "backend",
    "batch_config_sha256",
    "clock_id",
    "cuda_model_path",
    "cut_layer",
    "event_ns",
    "frozen_ns",
    "hidden_size",
    "kind",
    "model_id",
    "model_sha256",
    "n_layer",
    "op12_shard_bytes",
    "op12_shard_path",
    "op12_shard_sha256",
    "op12_stored_layers",
    "op15_shard_bytes",
    "op15_shard_path",
    "op15_shard_sha256",
    "op15_stored_layers",
    "phase",
    "phase_id",
    "role",
}
PHASE_LOCK_KEYS = {
    "acquisition_id",
    "candidate_sha256",
    "clock_id",
    "contract_sha256",
    "event_ns",
    "kind",
    "model_slot",
    "phase",
    "phase_id",
    "prior_phase_result_sha256s",
    "quality_corpus_sha256",
    "role",
    "route_lock_sha256",
}


class DriverError(ValueError):
    pass


class ProbeRunner(Protocol):
    def run(
        self,
        argv: list[str],
        *,
        timeout: int,
    ) -> subprocess.CompletedProcess[bytes]:
        ...


class SubprocessProbeRunner:
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
        raise DriverError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}: expected {expected!r}, got {value!r}",
    )


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"E_INTEGER: {field}")
    return value


def text(value: Any, field: str) -> str:
    require(type(value) is str and bool(value), f"E_TEXT: {field}")
    return value


def digest(value: Any, field: str) -> str:
    value = text(value, field)
    require(DIGEST_RE.fullmatch(value) is not None, f"E_DIGEST: {field}")
    return value


def exact_keys(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_OBJECT: {field}")
    actual = set(value)
    require(
        actual == keys,
        f"E_KEYS: {field}: missing={sorted(keys - actual)}, "
        f"unknown={sorted(actual - keys)}",
    )
    return value


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise DriverError(f"E_JSON_NUMBER: {value}")


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
        raise DriverError("E_CANONICAL") from error


def parse_json(raw: bytes, field: str) -> Any:
    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DriverError(f"E_JSON: {field}: {error}") from error


def secure_read(path: Path, field: str) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DriverError(f"E_READ: {field}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"E_NOT_REGULAR: {field}")
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
    exact(identity(after), identity(before), f"E_CHANGED_DURING_READ: {field}")
    exact(len(raw), before.st_size, f"E_READ_SIZE: {field}")
    return bytes(raw)


def read_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = secure_read(path, str(path))
    value = parse_json(raw, str(path))
    require(type(value) is dict, f"E_OBJECT: {path}")
    require(canonical_bytes(value) == raw, f"E_CANONICAL: {path}")
    return value, raw


def read_one_jsonl(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    raw = secure_read(path, field)
    lines = raw.splitlines(keepends=True)
    require(len(lines) == 1 and lines[0].endswith(b"\n"), f"E_ROWS: {field}")
    value = parse_json(lines[0], field)
    require(type(value) is dict, f"E_OBJECT: {field}")
    require(canonical_bytes(value) == raw, f"E_CANONICAL: {field}")
    return value, raw


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def clock_ns() -> int:
    return time.clock_gettime_ns(CLOCK_ID)


def durable_write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    fd = os.open(path, flags, 0o644)
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


def durable_json(path: Path, value: Any) -> bytes:
    raw = canonical_bytes(value)
    durable_write_new(path, raw)
    return raw


def validate_phase_id(value: str) -> str:
    require(
        value.startswith("cp0-r1-v23-a-only-")
        and len(value) <= 128
        and all(character.isalnum() or character in ".-_" for character in value),
        "E_PHASE_ID",
    )
    return value


def validate_absolute_path(value: str, field: str) -> str:
    value = text(value, field)
    require(Path(value).is_absolute(), f"E_PATH: {field}")
    require("\n" not in value and "\r" not in value, f"E_PATH: {field}")
    return value


def load_inputs(
    contract_path: Path,
    candidate_path: Path,
    pre_dir: Path,
    phase_id: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    bytes,
]:
    contract, _ = read_canonical(contract_path)
    candidate, _ = read_canonical(candidate_path)
    exact(contract.get("schema"), "s39-cp0-r1-evidence-contract-v2.3", "contract")
    exact(contract.get("status"), "FROZEN_BEFORE_QWEN3_14B_QUALIFICATION", "status")
    models = candidate.get("models")
    require(type(models) is list, "E_MODELS")
    matching = [
        model
        for model in models
        if type(model) is dict
        and model.get("slot") == "A"
        and model.get("model_id") == MODEL_ID
    ]
    require(len(matching) == 1, "E_MODEL_A")
    model = matching[0]
    locked_models = contract.get("candidate_lock", {}).get("models")
    require(type(locked_models) is list, "E_CONTRACT_MODELS")
    locked = [
        value
        for value in locked_models
        if type(value) is dict and value.get("slot") == "A"
    ]
    require(len(locked) == 1, "E_CONTRACT_MODEL_A")
    exact(locked[0]["model_id"], model["model_id"], "candidate.model_id")
    exact(locked[0]["architecture"], model["architecture"], "candidate.architecture")
    exact(locked[0]["n_layer"], model["n_layer"], "candidate.n_layer")
    exact(locked[0]["quantization"], model["quantization"], "candidate.quantization")
    exact(
        locked[0]["artifact_bytes"],
        model["artifact"]["bytes"],
        "candidate.artifact.bytes",
    )
    exact(
        locked[0]["artifact_sha256"],
        model["artifact"]["sha256"],
        "candidate.artifact.sha256",
    )
    route, route_raw = read_one_jsonl(pre_dir / "route_lock.jsonl", "route_lock")
    exact_keys(route, ROUTE_KEYS, "route_lock")
    exact(route["phase"], PHASE, "route_lock.phase")
    exact(route["phase_id"], phase_id, "route_lock.phase_id")
    exact(route["acquisition_id"], phase_id, "route_lock.acquisition_id")
    exact(route["role"], f"model.{MODEL_ID}.route_lock", "route_lock.role")
    exact(route["kind"], "route_lock", "route_lock.kind")
    exact(route["model_id"], MODEL_ID, "route_lock.model_id")
    exact(route["model_sha256"], model["artifact"]["sha256"], "route_lock.model")
    exact(route["n_layer"], model["n_layer"], "route_lock.n_layer")
    incumbent = contract["incumbent_route_lock"]
    for key in (
        "backend",
        "cut_layer",
        "op12_shard_sha256",
        "op12_stored_layers",
        "op15_shard_sha256",
        "op15_stored_layers",
    ):
        exact(route[key], incumbent[key], f"route_lock.{key}")
    geometry = contract["model_geometry"][MODEL_ID]
    exact(route["cuda_model_path"], geometry["cuda_model_path"], "route_lock.cuda")
    exact(route["hidden_size"], geometry["hidden_size"], "route_lock.hidden")
    exact(
        route["activation_dtype"],
        geometry["activation_dtype"],
        "route_lock.activation_dtype",
    )
    exact(
        route["activation_element_bytes"],
        geometry["activation_element_bytes"],
        "route_lock.activation_element_bytes",
    )
    for phone in ("op15", "op12"):
        known = geometry["known_shards"][phone]
        exact(route[f"{phone}_shard_path"], known["path"], f"route_lock.{phone}.path")
        exact(route[f"{phone}_shard_bytes"], known["bytes"], f"route_lock.{phone}.bytes")
        exact(
            route[f"{phone}_shard_sha256"],
            known["sha256"],
            f"route_lock.{phone}.sha256",
        )
    return contract, candidate, model, route, route_raw


def stat_record(value: os.stat_result) -> dict[str, int]:
    return {
        "ctime_ns": value.st_ctime_ns,
        "device_id": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "mtime_ns": value.st_mtime_ns,
        "size": value.st_size,
    }


def validate_stat(value: Any, field: str) -> dict[str, int]:
    value = exact_keys(
        value,
        {"ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"},
        field,
    )
    for key in value:
        integer(value[key], f"{field}.{key}")
    require(value["inode"] > 0 and value["size"] > 0, f"E_STAT: {field}")
    require(stat.S_ISREG(value["mode"]), f"E_NOT_REGULAR: {field}")
    return value


def run_probe(
    runner: ProbeRunner,
    argv: list[str],
    timeout: int,
    label: str,
) -> bytes:
    try:
        completed = runner.run(argv, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise DriverError(f"E_TIMEOUT: {label}") from error
    exact(completed.returncode, 0, f"E_PROBE_EXIT: {label}")
    exact(completed.stderr, b"", f"E_PROBE_STDERR: {label}")
    require(type(completed.stdout) is bytes and bool(completed.stdout), f"E_PROBE_EMPTY: {label}")
    try:
        completed.stdout.decode("ascii")
    except UnicodeDecodeError as error:
        raise DriverError(f"E_PROBE_ASCII: {label}") from error
    return completed.stdout


REMOTE_PYTHON = """\
import hashlib,json,os,stat,sys
p=sys.argv[1]
fd=os.open(p,os.O_RDONLY|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0))
try:
 s0=os.fstat(fd)
 if not stat.S_ISREG(s0.st_mode): raise SystemExit(30)
 h=hashlib.sha256()
 while True:
  b=os.read(fd,1048576)
  if not b: break
  h.update(b)
 s1=os.fstat(fd)
 if (s0.st_dev,s0.st_ino,s0.st_size,s0.st_mtime_ns,s0.st_ctime_ns,s0.st_mode)!=(s1.st_dev,s1.st_ino,s1.st_size,s1.st_mtime_ns,s1.st_ctime_ns,s1.st_mode): raise SystemExit(31)
 print(json.dumps({"sha256":h.hexdigest(),"stat":{"ctime_ns":s1.st_ctime_ns,"device_id":s1.st_dev,"inode":s1.st_ino,"mode":s1.st_mode,"mtime_ns":s1.st_mtime_ns,"size":s1.st_size}},sort_keys=True,separators=(",",":")))
finally:
 os.close(fd)
"""

REMOTE_STAT_PYTHON = """\
import json,os,stat,sys
p=sys.argv[1]
fd=os.open(p,os.O_RDONLY|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0))
try:
 s=os.fstat(fd)
 if not stat.S_ISREG(s.st_mode): raise SystemExit(30)
 print(json.dumps({"stat":{"ctime_ns":s.st_ctime_ns,"device_id":s.st_dev,"inode":s.st_ino,"mode":s.st_mode,"mtime_ns":s.st_mtime_ns,"size":s.st_size}},sort_keys=True,separators=(",",":")))
finally:
 os.close(fd)
"""


def ssh_python_argv(target: str, source: str, path: str) -> list[str]:
    command = "python3 -c {} {}".format(shlex.quote(source), shlex.quote(path))
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        target,
        command,
    ]


def android_stat_command(path: str, include_digest: bool) -> str:
    quoted = shlex.quote(path)
    stat_format = (
        "DEV=%d|INO=%i|SIZE=%s|MODE=%f|"
        "MTIME_S=%Y|MTIME=%y|CTIME_S=%Z|CTIME=%z"
    )
    emit = f"stat -c {shlex.quote(stat_format)} -- {quoted}"
    parts = [
        "set -eu",
        f"test -f {quoted}",
        f"test ! -L {quoted}",
        "printf 'BEFORE\\n'",
        emit,
    ]
    if include_digest:
        parts.extend(
            [
                f"printf 'DIGEST='; sha256sum {quoted} | cut -d' ' -f1",
                "printf 'AFTER\\n'",
                emit,
            ]
        )
    return "; ".join(parts)


def adb_argv(port: int, selector: str, script: str) -> list[str]:
    return ["adb", "-P", str(port), "-s", selector, "shell", script]


def parse_remote_json(raw: bytes, field: str, require_digest: bool) -> dict[str, Any]:
    value = parse_json(raw, field)
    keys = {"stat", "sha256"} if require_digest else {"stat"}
    exact_keys(value, keys, field)
    value["stat"] = validate_stat(value["stat"], f"{field}.stat")
    if require_digest:
        digest(value["sha256"], f"{field}.sha256")
    return value


def parse_android_time(value: str, seconds: int, field: str) -> int:
    match = ANDROID_TIME_RE.fullmatch(value)
    require(match is not None, f"E_ANDROID_TIME: {field}")
    base = datetime.datetime.strptime(
        f"{match.group('base')} {match.group('zone')}",
        "%Y-%m-%d %H:%M:%S %z",
    )
    fraction = match.group("fraction").ljust(9, "0")
    result = int(base.timestamp()) * 1_000_000_000 + int(fraction)
    exact(result // 1_000_000_000, seconds, f"{field}.seconds")
    return result


def parse_android_stat_block(line: str, field: str) -> dict[str, int]:
    expected = ("DEV", "INO", "SIZE", "MODE", "MTIME_S", "MTIME", "CTIME_S", "CTIME")
    parts = line.split("|")
    require(len(parts) == len(expected), f"E_ANDROID_STAT_FIELDS: {field}")
    values = {}
    for item, key in zip(parts, expected):
        prefix = f"{key}="
        require(item.startswith(prefix), f"E_ANDROID_STAT_FIELD: {field}.{key}")
        values[key] = item[len(prefix):]
    try:
        device = int(values["DEV"])
        inode = int(values["INO"])
        size = int(values["SIZE"])
        mode = int(values["MODE"], 16)
        mtime_s = int(values["MTIME_S"])
        ctime_s = int(values["CTIME_S"])
    except ValueError as error:
        raise DriverError(f"E_ANDROID_STAT_INTEGER: {field}") from error
    result = {
        "ctime_ns": parse_android_time(values["CTIME"], ctime_s, f"{field}.ctime"),
        "device_id": device,
        "inode": inode,
        "mode": mode,
        "mtime_ns": parse_android_time(values["MTIME"], mtime_s, f"{field}.mtime"),
        "size": size,
    }
    return validate_stat(result, field)


def parse_android_stat(raw: bytes, field: str, require_digest: bool) -> dict[str, Any]:
    lines = raw.decode("ascii").splitlines()
    require(lines and lines[0] == "BEFORE", f"E_ANDROID_STAT_HEADER: {field}")
    if not require_digest:
        require(len(lines) == 2, f"E_ANDROID_STAT_LINES: {field}")
        return {"stat": parse_android_stat_block(lines[1], f"{field}.before")}
    require(len(lines) == 5, f"E_ANDROID_STAT_LINES: {field}")
    before = parse_android_stat_block(lines[1], f"{field}.before")
    require(lines[2].startswith("DIGEST="), f"E_ANDROID_DIGEST: {field}")
    checksum = lines[2].split("=", 1)[1]
    digest(checksum, f"{field}.sha256")
    exact(lines[3], "AFTER", f"{field}.after_header")
    after = parse_android_stat_block(lines[4], f"{field}.after")
    exact(after, before, f"E_ARTIFACT_CHANGED_DURING_HASH: {field}")
    return {"sha256": checksum, "stat": after}


def artifact_specs(
    contract: dict[str, Any],
    model: dict[str, Any],
    route: dict[str, Any],
    op15_worker: str,
    op12_worker: str,
) -> dict[str, dict[str, Any]]:
    return {
        "cuda": {
            "bytes": model["artifact"]["bytes"],
            "digest": model["artifact"]["sha256"],
            "path": route["cuda_model_path"],
        },
        "op15": {
            "bytes": route["op15_shard_bytes"],
            "digest": route["op15_shard_sha256"],
            "path": route["op15_shard_path"],
            "adb_selector": contract["readiness_v2_3"]["phone_identity"]["op15"][
                "adb_selector"
            ],
        },
        "op12": {
            "bytes": route["op12_shard_bytes"],
            "digest": route["op12_shard_sha256"],
            "path": route["op12_shard_path"],
            "adb_selector": contract["readiness_v2_3"]["phone_identity"]["op12"][
                "adb_selector"
            ],
        },
        "op15_worker": {
            "path": op15_worker,
            "adb_selector": contract["readiness_v2_3"]["phone_identity"]["op15"][
                "adb_selector"
            ],
        },
        "op12_worker": {
            "path": op12_worker,
            "adb_selector": contract["readiness_v2_3"]["phone_identity"]["op12"][
                "adb_selector"
            ],
        },
    }


def collect_artifact(
    runner: ProbeRunner,
    target: str,
    adb_port: int,
    endpoint: str,
    spec: dict[str, Any],
    timeout: int,
    include_digest: bool,
) -> dict[str, Any]:
    if endpoint == "cuda":
        source = REMOTE_PYTHON if include_digest else REMOTE_STAT_PYTHON
        raw = run_probe(
            runner,
            ssh_python_argv(target, source, spec["path"]),
            timeout,
            endpoint,
        )
        return parse_remote_json(raw, endpoint, include_digest)
    raw = run_probe(
        runner,
        adb_argv(
            adb_port,
            spec["adb_selector"],
            android_stat_command(spec["path"], include_digest),
        ),
        timeout,
        endpoint,
    )
    return parse_android_stat(raw, endpoint, include_digest)


def validate_collected_artifact(
    endpoint: str,
    spec: dict[str, Any],
    collected: dict[str, Any],
) -> None:
    if "bytes" in spec:
        exact(collected["stat"]["size"], spec["bytes"], f"E_BYTES: {endpoint}")
    if "digest" in spec:
        exact(collected["sha256"], spec["digest"], f"E_SHA256: {endpoint}")


def artifact_driver(
    *,
    contract_path: Path,
    candidate_path: Path,
    pre_dir: Path,
    output_dir: Path,
    phase_id: str,
    op15_worker: str,
    op12_worker: str,
    timeout: int,
    runner: ProbeRunner | None = None,
    now_ns: Callable[[], int] = clock_ns,
) -> dict[str, Any]:
    validate_phase_id(phase_id)
    validate_absolute_path(op15_worker, "op15_worker")
    validate_absolute_path(op12_worker, "op12_worker")
    require(pre_dir.is_absolute() and output_dir.is_absolute(), "E_PATH: directories")
    require(output_dir.is_dir() and not output_dir.is_symlink(), "E_OUTPUT_DIR")
    contract, _, model, route, route_raw = load_inputs(
        contract_path,
        candidate_path,
        pre_dir,
        phase_id,
    )
    runner = runner or SubprocessProbeRunner()
    timeout = integer(timeout, "timeout", 1)
    require(timeout <= 7200, "E_TIMEOUT_RANGE")
    started_ns = now_ns()
    specs = artifact_specs(contract, model, route, op15_worker, op12_worker)
    records = []
    target = contract["preflight"]["ssh_target"]
    adb_port = contract["preflight"]["phone_adb_port"]
    for endpoint in ENDPOINTS:
        collected = collect_artifact(
            runner,
            target,
            adb_port,
            endpoint,
            specs[endpoint],
            timeout,
            True,
        )
        validate_collected_artifact(endpoint, specs[endpoint], collected)
        records.append(
            {
                "bytes": collected["stat"]["size"],
                "endpoint": endpoint,
                "path": specs[endpoint]["path"],
                "sha256": collected["sha256"],
                "stat": collected["stat"],
            }
        )
    completed_ns = now_ns()
    require(started_ns < completed_ns, "E_ARTIFACT_INTERVAL")
    result = {
        "artifacts": records,
        "completed_ns": completed_ns,
        "model_id": MODEL_ID,
        "phase": PHASE,
        "route_lock_sha256": sha256_bytes(route_raw),
        "schema": "s39-cp0-r1-artifact-snapshot-v2.3",
        "slot": "A",
        "started_ns": started_ns,
    }
    durable_json(output_dir / "artifact_snapshot.json", result)
    return result


def parse_cuda_identity(raw: bytes, expected_uuid: str) -> dict[str, Any]:
    lines = raw.decode("ascii").splitlines()
    require(len(lines) == 3, "E_CUDA_IDENTITY_LINES")
    host = lines[0]
    host_boot_id = lines[1]
    require(UUID_RE.fullmatch(host_boot_id) is not None, "E_CUDA_BOOT_ID")
    fields = [field.strip() for field in lines[2].split(",")]
    require(len(fields) == 4, "E_CUDA_IDENTITY_FIELDS")
    try:
        memory_total_bytes = int(fields[2]) * 1024 * 1024
    except ValueError as error:
        raise DriverError("E_CUDA_MEMORY_TOTAL") from error
    exact(fields[1], expected_uuid, "E_CUDA_UUID")
    return {
        "host": host,
        "host_boot_id": host_boot_id,
        "memory_total_bytes": memory_total_bytes,
        "name": fields[0],
        "pci_bus_id": fields[3],
        "uuid": fields[1],
    }


def cuda_identity_argv(target: str, uuid: str) -> list[str]:
    command = (
        "set -eu; hostname; cat /proc/sys/kernel/random/boot_id; "
        f"nvidia-smi --id={shlex.quote(uuid)} "
        "--query-gpu=name,uuid,memory.total,pci.bus_id "
        "--format=csv,noheader,nounits"
    )
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        target,
        command,
    ]


def phone_identity_command() -> str:
    return """\
set -eu
printf 'SERIAL='; getprop ro.serialno
printf 'MODEL='; getprop ro.product.model
printf 'PRODUCT='; getprop ro.product.name
printf 'DEVICE='; getprop ro.product.device
printf 'BOOT_ID='; cat /proc/sys/kernel/random/boot_id
awk '/^MemAvailable:/{print "MEM_AVAILABLE_KB="$2}' /proc/meminfo
awk '/^SwapTotal:/{print "SWAP_TOTAL_KB="$2}' /proc/meminfo
awk '/^SwapFree:/{print "SWAP_FREE_KB="$2}' /proc/meminfo
dumpsys thermalservice | awk -F: '/Thermal Status:/{gsub(/[[:space:]]/,"",$2); print "THERMAL_STATUS="$2; found=1; exit} END{if(!found) exit 42}'
ip -o -4 addr show up scope global | awk '{split($4,a,"/"); print $2 "|" a[1]}' | while IFS='|' read -r iface ipv4; do
  test -r "/sys/class/net/$iface/statistics/rx_bytes"
  test -r "/sys/class/net/$iface/statistics/tx_bytes"
  rx=$(cat "/sys/class/net/$iface/statistics/rx_bytes")
  tx=$(cat "/sys/class/net/$iface/statistics/tx_bytes")
  printf 'IF=%s|%s|%s|%s\\n' "$iface" "$ipv4" "$rx" "$tx"
done
"""


def parse_phone_identity(
    raw: bytes,
    physical_serial: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    exact_keys(
        expected,
        {"adb_selector", "device", "model", "product", "serial"},
        f"phone_identity.{physical_serial}",
    )
    exact(expected["serial"], physical_serial, f"E_PHONE_SERIAL_ARGUMENT: {physical_serial}")
    lines = raw.decode("ascii").splitlines()
    values: dict[str, str] = {}
    interfaces: dict[str, dict[str, Any]] = {}
    for line in lines:
        if line.startswith("IF="):
            fields = line[3:].split("|")
            require(len(fields) == 4, f"E_PHONE_INTERFACE: {physical_serial}")
            interface, ipv4, rx, tx = fields
            require(
                interface and interface not in interfaces,
                f"E_PHONE_INTERFACE: {physical_serial}",
            )
            try:
                rx_bytes = int(rx)
                tx_bytes = int(tx)
            except ValueError as error:
                raise DriverError(f"E_PHONE_COUNTER: {physical_serial}") from error
            require(
                rx_bytes >= 0 and tx_bytes >= 0,
                f"E_PHONE_COUNTER: {physical_serial}",
            )
            interfaces[interface] = {
                "ipv4": text(ipv4, f"{physical_serial}.ipv4"),
                "rx_bytes": rx_bytes,
                "tx_bytes": tx_bytes,
            }
            continue
        key, separator, value = line.partition("=")
        require(
            separator == "=" and key and key not in values,
            f"E_PHONE_FIELD: {physical_serial}",
        )
        values[key] = value
    exact(
        set(values),
        {
            "BOOT_ID",
            "DEVICE",
            "MEM_AVAILABLE_KB",
            "MODEL",
            "PRODUCT",
            "SERIAL",
            "SWAP_FREE_KB",
            "SWAP_TOTAL_KB",
            "THERMAL_STATUS",
        },
        f"E_PHONE_FIELDS: {physical_serial}",
    )
    require(bool(interfaces), f"E_PHONE_INTERFACES: {physical_serial}")
    require(
        UUID_RE.fullmatch(values["BOOT_ID"]) is not None,
        f"E_PHONE_BOOT: {physical_serial}",
    )
    exact(
        values["SERIAL"],
        expected["serial"],
        f"E_PHONE_IDENTITY: {physical_serial}.serial",
    )
    for key, source in (("model", "MODEL"), ("product", "PRODUCT"), ("device", "DEVICE")):
        exact(
            values[source],
            expected[key],
            f"E_PHONE_IDENTITY: {physical_serial}.{key}",
        )
    try:
        available = int(values["MEM_AVAILABLE_KB"]) * 1024
        swap_total = int(values["SWAP_TOTAL_KB"]) * 1024
        swap_free = int(values["SWAP_FREE_KB"]) * 1024
        thermal = int(values["THERMAL_STATUS"])
    except ValueError as error:
        raise DriverError(f"E_PHONE_INTEGER: {physical_serial}") from error
    require(
        0 <= swap_free <= swap_total,
        f"E_PHONE_SWAP_RANGE: {physical_serial}",
    )
    return {
        "available_bytes": available,
        "boot_id": values["BOOT_ID"],
        "device": values["DEVICE"],
        "interfaces": interfaces,
        "model": values["MODEL"],
        "product": values["PRODUCT"],
        "serial": values["SERIAL"],
        "swap_total_bytes": swap_total,
        "swap_used_bytes": swap_total - swap_free,
        "thermal_status": thermal,
    }


def load_artifact_snapshot(
    path: Path,
    route_raw: bytes,
) -> tuple[dict[str, Any], bytes, dict[str, dict[str, Any]]]:
    snapshot, raw = read_canonical(path)
    exact_keys(
        snapshot,
        {
            "artifacts",
            "completed_ns",
            "model_id",
            "phase",
            "route_lock_sha256",
            "schema",
            "slot",
            "started_ns",
        },
        "artifact_snapshot",
    )
    exact(snapshot.get("schema"), "s39-cp0-r1-artifact-snapshot-v2.3", "artifact.schema")
    exact(snapshot.get("phase"), PHASE, "artifact.phase")
    exact(snapshot.get("model_id"), MODEL_ID, "artifact.model")
    exact(snapshot.get("slot"), "A", "artifact.slot")
    exact(snapshot.get("route_lock_sha256"), sha256_bytes(route_raw), "artifact.route")
    started_ns = integer(snapshot["started_ns"], "artifact.started_ns", 1)
    completed_ns = integer(snapshot["completed_ns"], "artifact.completed_ns", 1)
    require(started_ns < completed_ns, "E_ARTIFACT_INTERVAL")
    records = snapshot.get("artifacts")
    require(type(records) is list and len(records) == len(ENDPOINTS), "E_ARTIFACT_COUNT")
    by_endpoint = {}
    for index, record in enumerate(records):
        record = exact_keys(
            record,
            {"bytes", "endpoint", "path", "sha256", "stat"},
            f"artifact[{index}]",
        )
        endpoint = text(record["endpoint"], f"artifact[{index}].endpoint")
        require(endpoint in ENDPOINTS and endpoint not in by_endpoint, "E_ARTIFACT_ENDPOINT")
        exact(record["bytes"], record["stat"]["size"], f"artifact[{index}].bytes")
        digest(record["sha256"], f"artifact[{index}].sha256")
        validate_stat(record["stat"], f"artifact[{index}].stat")
        by_endpoint[endpoint] = record
    exact(tuple(by_endpoint), ENDPOINTS, "artifact.order")
    return snapshot, raw, by_endpoint


def load_phase_lock(
    path: Path,
    phase_id: str,
    route_raw: bytes,
) -> tuple[dict[str, Any], bytes]:
    value, raw = read_one_jsonl(path, "phase_lock")
    exact_keys(value, PHASE_LOCK_KEYS, "phase_lock")
    exact(value["phase"], PHASE, "phase_lock.phase")
    exact(value["phase_id"], phase_id, "phase_lock.phase_id")
    exact(value["acquisition_id"], phase_id, "phase_lock.acquisition_id")
    exact(value["kind"], "phase_lock", "phase_lock.kind")
    exact(value["role"], "phase.lock", "phase_lock.role")
    exact(value["model_slot"], "A", "phase_lock.model_slot")
    exact(value["clock_id"], "HOST_MONOTONIC_RAW", "phase_lock.clock")
    exact(value["route_lock_sha256"], sha256_bytes(route_raw), "phase_lock.route")
    return value, raw


def fresh_driver(
    *,
    contract_path: Path,
    candidate_path: Path,
    pre_dir: Path,
    output_dir: Path,
    phase_id: str,
    op15_worker: str,
    op12_worker: str,
    timeout: int,
    runner: ProbeRunner | None = None,
    now_ns: Callable[[], int] = clock_ns,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_phase_id(phase_id)
    validate_absolute_path(op15_worker, "op15_worker")
    validate_absolute_path(op12_worker, "op12_worker")
    require(pre_dir.is_absolute() and output_dir.is_absolute(), "E_PATH: directories")
    require(output_dir.is_dir() and not output_dir.is_symlink(), "E_OUTPUT_DIR")
    contract, _, model, route, route_raw = load_inputs(
        contract_path,
        candidate_path,
        pre_dir,
        phase_id,
    )
    phase_lock, phase_lock_raw = load_phase_lock(
        pre_dir / "phase_lock.jsonl",
        phase_id,
        route_raw,
    )
    artifact_path = pre_dir.parent / "artifact" / "artifact_snapshot.json"
    artifact, artifact_raw, artifact_by_endpoint = load_artifact_snapshot(
        artifact_path,
        route_raw,
    )
    runner = runner or SubprocessProbeRunner()
    timeout = integer(timeout, "timeout", 1)
    require(timeout <= 7200, "E_TIMEOUT_RANGE")

    lock_event_ns = now_ns()
    require(
        artifact["completed_ns"] <= lock_event_ns,
        "E_READINESS_LOCK_BEFORE_ARTIFACT",
    )
    lock = {
        "artifact_snapshot_sha256": sha256_bytes(artifact_raw),
        "event_ns": lock_event_ns,
        "phase": PHASE,
        "phase_id": phase_id,
        "schema": "s39-cp0-r1-readiness-lock-v2.3",
        "v2_2_phase_lock_sha256": sha256_bytes(phase_lock_raw),
    }
    lock_raw = durable_json(output_dir / "readiness_lock.json", lock)
    started_ns = now_ns()
    require(lock_event_ns <= started_ns, "E_FRESH_PRELOCK")

    specs = artifact_specs(contract, model, route, op15_worker, op12_worker)
    for endpoint in ENDPOINTS:
        record = artifact_by_endpoint[endpoint]
        exact(record["path"], specs[endpoint]["path"], f"E_ARTIFACT_PATH: {endpoint}")
        if "bytes" in specs[endpoint]:
            exact(record["bytes"], specs[endpoint]["bytes"], f"E_ARTIFACT_BYTES: {endpoint}")
        if "digest" in specs[endpoint]:
            exact(
                record["sha256"],
                specs[endpoint]["digest"],
                f"E_ARTIFACT_SHA256: {endpoint}",
            )
    stats = []
    target = contract["preflight"]["ssh_target"]
    adb_port = contract["preflight"]["phone_adb_port"]
    for endpoint in ENDPOINTS:
        collected = collect_artifact(
            runner,
            target,
            adb_port,
            endpoint,
            specs[endpoint],
            timeout,
            False,
        )
        exact(
            collected["stat"],
            artifact_by_endpoint[endpoint]["stat"],
            f"E_ARTIFACT_CHANGED: {endpoint}",
        )
        stats.append(
            {
                "endpoint": endpoint,
                "path": specs[endpoint]["path"],
                "stat": collected["stat"],
            }
        )

    cuda_expected = contract["readiness_v2_3"]["cuda_identity"]
    cuda_raw = run_probe(
        runner,
        cuda_identity_argv(target, cuda_expected["uuid"]),
        timeout,
        "cuda_identity",
    )
    cuda = parse_cuda_identity(cuda_raw, cuda_expected["uuid"])
    for key in ("host", "memory_total_bytes", "name", "uuid"):
        exact(cuda[key], cuda_expected[key], f"E_CUDA_IDENTITY: {key}")

    phones = {}
    for phone in ("op15", "op12"):
        expected = contract["readiness_v2_3"]["phone_identity"][phone]
        raw = run_probe(
            runner,
            adb_argv(
                adb_port,
                expected["adb_selector"],
                phone_identity_command(),
            ),
            timeout,
            f"{phone}_identity",
        )
        value = parse_phone_identity(raw, expected["serial"], expected)
        require(
            value["available_bytes"]
            >= contract["readiness_v2_3"]["phone_minimum_available_bytes"],
            f"E_PHONE_HEADROOM: {phone}",
        )
        exact(value["swap_used_bytes"], 0, f"E_PHONE_SWAP: {phone}")
        exact(value["thermal_status"], 0, f"E_PHONE_THERMAL: {phone}")
        phones[phone] = value

    completed_ns = now_ns()
    require(started_ns < completed_ns, "E_FRESH_INTERVAL")
    fresh = {
        "artifact_stats": stats,
        "completed_ns": completed_ns,
        "cuda": cuda,
        "phase": PHASE,
        "phase_id": phase_id,
        "phones": phones,
        "readiness_lock_sha256": sha256_bytes(lock_raw),
        "schema": "s39-cp0-r1-fresh-identity-v2.3",
        "started_ns": started_ns,
    }
    durable_json(output_dir / "fresh_snapshot.json", fresh)
    del phase_lock
    return lock, fresh


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--op15-worker", required=True)
    parser.add_argument("--op12-worker", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)


def artifact_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture V2.3 immutable artifact identities")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    try:
        value = artifact_driver(
            contract_path=args.contract,
            candidate_path=args.candidate,
            pre_dir=args.pre,
            output_dir=args.output,
            phase_id=args.phase_id,
            op15_worker=args.op15_worker,
            op12_worker=args.op12_worker,
            timeout=args.timeout_seconds,
        )
        print(
            f"V23_ARTIFACT_SNAPSHOT_PASS "
            f"{sha256_bytes(canonical_bytes(value))}"
        )
        return 0
    except Exception as error:
        print(f"V23_ARTIFACT_SNAPSHOT_REFUSED: {type(error).__name__}: {error}")
        return 2


def fresh_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture V2.3 fresh runtime readiness")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    try:
        lock, fresh = fresh_driver(
            contract_path=args.contract,
            candidate_path=args.candidate,
            pre_dir=args.pre,
            output_dir=args.output,
            phase_id=args.phase_id,
            op15_worker=args.op15_worker,
            op12_worker=args.op12_worker,
            timeout=args.timeout_seconds,
        )
        print(
            f"V23_FRESH_READINESS_PASS "
            f"{sha256_bytes(canonical_bytes(lock))} "
            f"{sha256_bytes(canonical_bytes(fresh))}"
        )
        return 0
    except Exception as error:
        print(f"V23_FRESH_READINESS_REFUSED: {type(error).__name__}: {error}")
        return 2
