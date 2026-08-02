#!/usr/bin/env python3
"""Assemble and validate one fail-closed CP0-R1 A_ONLY acquisition."""

from __future__ import annotations

import argparse
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


BOOTSTRAP_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
BOOTSTRAP_MANIFEST_RE = re.compile(r"([0-9a-f]{64})  ([^\n]+)\n")
BOOTSTRAP_PLAN_KEYS = {
    "source_manifest_path",
    "source_manifest_sha256",
    "source_root",
}
BOOTSTRAP_MODULES = (
    (
        "build_cp0_r1_v21",
        "research_dev/spikes/s39_phone_model_switch_trace/"
        "build_cp0_r1_v21.py",
    ),
    (
        "build_cp0_r1_mmlu64_v22",
        "research_dev/spikes/s39_phone_model_switch_trace/"
        "build_cp0_r1_mmlu64_v22.py",
    ),
    (
        "build_cp0_r1_v22",
        "research_dev/spikes/s39_phone_model_switch_trace/"
        "build_cp0_r1_v22.py",
    ),
    (
        "cp0_r1_evidence_v2",
        "research_dev/spikes/s39_phone_model_switch_trace/"
        "cp0_r1_evidence_v2.py",
    ),
    (
        "cp0_r1_evidence_v21",
        "research_dev/spikes/s39_phone_model_switch_trace/"
        "cp0_r1_evidence_v21.py",
    ),
    (
        "cp0_r1_evidence_v22",
        "research_dev/spikes/s39_phone_model_switch_trace/"
        "cp0_r1_evidence_v22.py",
    ),
    (
        "v23_common",
        "research_dev/spikes/s39_phone_model_switch_trace/"
        "v23_readiness/v23_common.py",
    ),
    (
        "build_contract_v23",
        "research_dev/spikes/s39_phone_model_switch_trace/"
        "v23_readiness/build_contract_v23.py",
    ),
    (
        "cp0_r1_evidence_v23",
        "research_dev/spikes/s39_phone_model_switch_trace/"
        "v23_readiness/cp0_r1_evidence_v23.py",
    ),
    (
        "cp0_r1_phase_preflight_v21",
        "research_dev/spikes/s39_phone_model_switch_trace/"
        "cp0_r1_phase_preflight_v21.py",
    ),
    (
        "runtime_bundle_overlay_v1",
        "research_dev/spikes/s39_phone_model_switch_trace/"
        "v23_readiness/production_v2/runtime_bundle_overlay_v1.py",
    ),
    (
        "acquisition_plan_common_v1",
        "research_dev/spikes/s39_phone_model_switch_trace/"
        "v23_readiness/acquisition_plan_v1/plan_common_v1.py",
    ),
)


class BootstrapError(ValueError):
    pass


def _bootstrap_require(condition: bool, message: str) -> None:
    if not condition:
        raise BootstrapError(message)


def _bootstrap_strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        _bootstrap_require(key not in result, f"E_BOOTSTRAP_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def _bootstrap_reject_constant(value: str) -> None:
    raise BootstrapError(f"E_BOOTSTRAP_JSON_NUMBER: {value}")


def _bootstrap_canonical_bytes(value: Any) -> bytes:
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
        raise BootstrapError("E_BOOTSTRAP_CANONICAL") from error


def _bootstrap_read_regular(path: Path, field: str) -> bytes:
    _bootstrap_require(path.is_absolute(), f"E_BOOTSTRAP_PATH: {field}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BootstrapError(f"E_BOOTSTRAP_OPEN: {field}: {error}") from error
    try:
        before = os.fstat(descriptor)
        _bootstrap_require(
            stat.S_ISREG(before.st_mode),
            f"E_BOOTSTRAP_FILE_TYPE: {field}",
        )
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
    _bootstrap_require(
        identity(before) == identity(after) and len(raw) == before.st_size,
        f"E_BOOTSTRAP_CHANGED: {field}",
    )
    return bytes(raw)


def _bootstrap_read_plan(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _bootstrap_read_regular(path, "plan")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_bootstrap_strict_object,
            parse_constant=_bootstrap_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BootstrapError("E_BOOTSTRAP_PLAN_JSON") from error
    _bootstrap_require(type(value) is dict, "E_BOOTSTRAP_PLAN_TYPE")
    _bootstrap_require(
        _bootstrap_canonical_bytes(value) == raw,
        "E_BOOTSTRAP_PLAN_CANONICAL",
    )
    return value, raw


def _bootstrap_plan_path(argv: list[str]) -> Path:
    indexes = [index for index, value in enumerate(argv) if value == "--plan"]
    _bootstrap_require(len(indexes) == 1, "E_BOOTSTRAP_PLAN_ARGUMENT")
    index = indexes[0] + 1
    _bootstrap_require(index < len(argv), "E_BOOTSTRAP_PLAN_ARGUMENT")
    path = Path(argv[index])
    _bootstrap_require(path.is_absolute(), "E_BOOTSTRAP_PLAN_PATH")
    return path


def _bootstrap_lstat_tree(root: Path, relative: Path, field: str) -> Path:
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise BootstrapError(
                f"E_BOOTSTRAP_SOURCE_STAT: {field}: {error}"
            ) from error
        _bootstrap_require(
            not stat.S_ISLNK(metadata.st_mode),
            f"E_BOOTSTRAP_SOURCE_SYMLINK: {field}",
        )
        if index + 1 < len(relative.parts):
            _bootstrap_require(
                stat.S_ISDIR(metadata.st_mode),
                f"E_BOOTSTRAP_SOURCE_PARENT: {field}",
            )
        else:
            _bootstrap_require(
                stat.S_ISREG(metadata.st_mode),
                f"E_BOOTSTRAP_SOURCE_TYPE: {field}",
            )
    return current


def _bootstrap_verify_sources(
    argv: list[str],
) -> tuple[
    Path,
    dict[str, Any],
    bytes,
    dict[str, tuple[Path, bytes]],
]:
    plan_path = _bootstrap_plan_path(argv)
    plan, plan_raw = _bootstrap_read_plan(plan_path)
    _bootstrap_require(
        BOOTSTRAP_PLAN_KEYS.issubset(plan),
        "E_BOOTSTRAP_PLAN_SOURCE_KEYS",
    )
    source_root_value = plan["source_root"]
    manifest_value = plan["source_manifest_path"]
    expected_manifest_sha256 = plan["source_manifest_sha256"]
    _bootstrap_require(
        type(source_root_value) is str
        and type(manifest_value) is str
        and type(expected_manifest_sha256) is str,
        "E_BOOTSTRAP_PLAN_SOURCE_TYPES",
    )
    _bootstrap_require(
        BOOTSTRAP_DIGEST_RE.fullmatch(expected_manifest_sha256) is not None,
        "E_BOOTSTRAP_MANIFEST_DIGEST",
    )
    source_root = Path(source_root_value)
    manifest_path = Path(manifest_value)
    _bootstrap_require(
        source_root.is_absolute() and manifest_path.is_absolute(),
        "E_BOOTSTRAP_SOURCE_PATH",
    )
    try:
        root_metadata = os.lstat(source_root)
    except OSError as error:
        raise BootstrapError(f"E_BOOTSTRAP_SOURCE_ROOT: {error}") from error
    _bootstrap_require(
        stat.S_ISDIR(root_metadata.st_mode)
        and not stat.S_ISLNK(root_metadata.st_mode),
        "E_BOOTSTRAP_SOURCE_ROOT",
    )
    try:
        manifest_relative = manifest_path.relative_to(source_root)
        plan_relative = plan_path.relative_to(source_root)
    except ValueError as error:
        raise BootstrapError("E_BOOTSTRAP_SOURCE_ESCAPE") from error
    _bootstrap_lstat_tree(source_root, manifest_relative, "source_manifest")
    _bootstrap_lstat_tree(source_root, plan_relative, "plan")
    manifest_raw = _bootstrap_read_regular(
        manifest_path,
        "source_manifest",
    )
    _bootstrap_require(
        hashlib.sha256(manifest_raw).hexdigest() == expected_manifest_sha256,
        "E_BOOTSTRAP_MANIFEST_SHA256",
    )
    try:
        manifest_text = manifest_raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise BootstrapError("E_BOOTSTRAP_MANIFEST_ASCII") from error
    records = BOOTSTRAP_MANIFEST_RE.findall(manifest_text)
    _bootstrap_require(
        "".join(
            f"{digest_value}  {relative}\n"
            for digest_value, relative in records
        )
        == manifest_text,
        "E_BOOTSTRAP_MANIFEST_FORMAT",
    )
    _bootstrap_require(bool(records), "E_BOOTSTRAP_MANIFEST_EMPTY")
    relative_values = [relative for _, relative in records]
    _bootstrap_require(
        relative_values == sorted(relative_values)
        and len(relative_values) == len(set(relative_values)),
        "E_BOOTSTRAP_MANIFEST_ORDER",
    )
    sources = {}
    for expected_sha256, relative_value in records:
        _bootstrap_require(
            BOOTSTRAP_DIGEST_RE.fullmatch(expected_sha256) is not None,
            f"E_BOOTSTRAP_SOURCE_DIGEST: {relative_value}",
        )
        relative = Path(relative_value)
        _bootstrap_require(
            relative.parts
            and not relative.is_absolute()
            and relative_value not in (".", "..")
            and ".." not in relative.parts
            and "__pycache__" not in relative.parts
            and relative.suffix != ".pyc",
            f"E_BOOTSTRAP_SOURCE_PATH: {relative_value}",
        )
        source = _bootstrap_lstat_tree(
            source_root,
            relative,
            relative_value,
        )
        raw = _bootstrap_read_regular(source, relative_value)
        _bootstrap_require(
            hashlib.sha256(raw).hexdigest() == expected_sha256,
            f"E_BOOTSTRAP_SOURCE_SHA256: {relative_value}",
        )
        sources[relative.as_posix()] = (source, raw)
    current_source = Path(__file__).absolute()
    _bootstrap_require(
        any(
            source == current_source
            for source, _ in sources.values()
        ),
        "E_BOOTSTRAP_ENTRYPOINT_UNBOUND",
    )
    for _, relative in BOOTSTRAP_MODULES:
        _bootstrap_require(
            relative in sources,
            f"E_BOOTSTRAP_MODULE_UNBOUND: {relative}",
        )
    return plan_path.absolute(), plan, plan_raw, sources


def _bootstrap_load_module(
    name: str,
    relative: str,
    sources: dict[str, tuple[Path, bytes]],
) -> types.ModuleType:
    _bootstrap_require(relative in sources, f"E_BOOTSTRAP_MODULE_UNBOUND: {name}")
    _bootstrap_require(name not in sys.modules, f"E_BOOTSTRAP_MODULE_PRELOADED: {name}")
    path, raw = sources[relative]
    try:
        source = raw.decode("ascii")
        code = compile(source, str(path), "exec", dont_inherit=True, optimize=0)
    except (SyntaxError, UnicodeDecodeError) as error:
        raise BootstrapError(f"E_BOOTSTRAP_MODULE_SOURCE: {name}") from error
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


_BOOTSTRAP_PLAN = None
_BOOTSTRAP_PLAN_PATH = None
_BOOTSTRAP_PLAN_RAW = None
_BOOTSTRAP_SOURCES = None
if __name__ == "__main__":
    try:
        (
            _BOOTSTRAP_PLAN_PATH,
            _BOOTSTRAP_PLAN,
            _BOOTSTRAP_PLAN_RAW,
            _BOOTSTRAP_SOURCES,
        ) = _bootstrap_verify_sources(sys.argv[1:])
    except (BootstrapError, OSError, ValueError) as error:
        print(f"CP0_R1_A_ONLY_V2_3_BOOTSTRAP_REFUSED: {error}")
        raise SystemExit(2)


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
if _BOOTSTRAP_SOURCES is None:
    if str(S39) not in sys.path:
        sys.path.insert(0, str(S39))
    import cp0_r1_evidence_v2 as v2
    import cp0_r1_evidence_v21 as v21
    import cp0_r1_evidence_v22 as v22
    import cp0_r1_phase_preflight_v21 as preflight
    import cp0_r1_evidence_v23 as v23
    import v23_common as common
else:
    _bound_modules = {
        name: _bootstrap_load_module(name, relative, _BOOTSTRAP_SOURCES)
        for name, relative in BOOTSTRAP_MODULES
    }
    v2 = _bound_modules["cp0_r1_evidence_v2"]
    v21 = _bound_modules["cp0_r1_evidence_v21"]
    v22 = _bound_modules["cp0_r1_evidence_v22"]
    preflight = _bound_modules["cp0_r1_phase_preflight_v21"]
    v23 = _bound_modules["cp0_r1_evidence_v23"]
    common = _bound_modules["v23_common"]


CLOCK_ID = time.CLOCK_MONOTONIC_RAW
PHASE = "A_ONLY"
PLAN_SCHEMA = "s39-cp0-r1-a-only-acquisition-plan-v2.3-production-v2"
PRE_ROLES = {
    "phase.lock",
    "phase.preflight",
    "quality.corpus",
    "model.qwen3-14b-q4_k_m.route_lock",
}
OUTPUT_KEYS = {
    "artifact_snapshot",
    "fresh_snapshot",
    "readiness_lock",
    "runtime_bundle_identity",
    "runtime_identity",
}
PLAN_KEYS = {
    "candidate_sha256",
    "contract_sha256",
    "drivers",
    "model_id",
    "output_files",
    "payload_roles",
    "phase",
    "schema",
    *BOOTSTRAP_PLAN_KEYS,
}
DRIVER_KEYS = {
    "argv_template",
    "executed_files",
    "timeout_seconds",
}
EXECUTED_FILE_KEYS = {
    "argv_index",
    "bytes",
    "path",
    "sha256",
}
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
READINESS_PLACEHOLDERS = {
    "{output_dir}",
    "{phase_id}",
    "{pre_dir}",
}
ACQUISITION_PLACEHOLDERS = {
    "{acquisition_started_ns}",
    "{output_dir}",
    "{phase_id}",
    "{pre_dir}",
}
RUNTIME_BUNDLE_RESULT_KEYS = {
    "runtime_bundle_fresh_sha256",
    "runtime_bundle_identity_sha256",
    "runtime_bundle_plan_sha256",
    "runtime_bundle_snapshot_sha256",
    "schema",
    "status",
}


class AcquisitionRunner(Protocol):
    def run(
        self,
        argv: list[str],
        *,
        timeout: int,
    ) -> subprocess.CompletedProcess[bytes]:
        ...


class SubprocessRunner:
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


def _read_source(path: Path, field: str) -> str:
    common.require(path.is_absolute(), f"E_PATH: {field}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise common.ReadinessError(f"E_SOURCE_OPEN: {field}: {error}") from error
    try:
        before = os.fstat(descriptor)
        common.require(stat.S_ISREG(before.st_mode), f"E_SOURCE_TYPE: {field}")
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
    common.require(
        identity(before) == identity(after) and len(raw) == before.st_size,
        f"E_SOURCE_CHANGED: {field}",
    )
    try:
        return bytes(raw).decode("ascii")
    except UnicodeDecodeError as error:
        raise common.ReadinessError(f"E_SOURCE_ASCII: {field}") from error


def _load_source(path: Path, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    source = _read_source(path, name)
    exec(
        compile(source, str(path), "exec", dont_inherit=True, optimize=0),
        module.__dict__,
    )
    return module


def _default_runtime_bundle_validator(**paths: Path) -> dict[str, Any]:
    if _BOOTSTRAP_SOURCES is None:
        overlay = _load_source(
            (HERE / "production_v2" / "runtime_bundle_overlay_v1.py").resolve(),
            "s39_v23_runtime_bundle_overlay_exit",
        )
    else:
        overlay = _bound_modules["runtime_bundle_overlay_v1"]
    return overlay.validate(argparse.Namespace(**paths))


def _default_plan_validator(
    plan_path: Path,
    plan: dict[str, Any],
) -> None:
    if _BOOTSTRAP_SOURCES is None:
        plan_common = _load_source(
            (
                HERE
                / "acquisition_plan_v1"
                / "plan_common_v1.py"
            ).resolve(),
            "s39_v23_acquisition_plan_common_exit",
        )
    else:
        plan_common = _bound_modules["acquisition_plan_common_v1"]
    try:
        expected = plan_common.build_plan()
        plan_common.exact(plan, expected, "paid.plan")
        expected_raw = plan_common.canonical_bytes(expected)
        if _BOOTSTRAP_PLAN_RAW is None:
            actual_raw = plan_common.read_regular(plan_path)
        else:
            actual_raw = _BOOTSTRAP_PLAN_RAW
        plan_common.exact(actual_raw, expected_raw, "paid.plan.bytes")
        expected_sources = plan_common.source_paths(plan)
        plan_common.validate_bound_source_manifest(
            plan,
            expected_sources,
        )
    except (OSError, ValueError) as error:
        raise common.ReadinessError(
            f"E_CANONICAL_ACQUISITION_PLAN: {error}"
        ) from error


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


def durable_json(path: Path, value: Any) -> bytes:
    raw = v2.canonical_bytes(value)
    durable_write_new(path, raw)
    return raw


def _relative(value: Any, field: str) -> str:
    text = common.string(value, field)
    path = Path(text)
    common.require(
        not path.is_absolute()
        and text not in (".", "..")
        and ".." not in path.parts,
        f"E_PATH: {field}",
    )
    return text


def _read_jsonl(path: Path, field: str) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    common.require(raw.endswith(b"\n"), f"E_CANONICAL: {field}")
    rows = []
    for index, line in enumerate(raw.splitlines(keepends=True)):
        common.require(line != b"\n", f"E_EMPTY: {field}[{index}]")
        row = v2.parse_json(line, f"{field}[{index}]")
        common.require(type(row) is dict, f"E_TYPE: {field}[{index}]")
        common.require(
            v2.canonical_line(row) == line,
            f"E_CANONICAL: {field}[{index}]",
        )
        rows.append(row)
    common.require(bool(rows), f"E_EMPTY: {field}")
    return rows, raw


def _validate_driver(
    driver: Any,
    field: str,
    placeholders: set[str],
) -> dict[str, Any]:
    driver = common.exact_keys(driver, DRIVER_KEYS, field)
    timeout = common.integer(
        driver["timeout_seconds"],
        f"{field}.timeout_seconds",
        1,
    )
    common.require(timeout <= 7200, f"E_RANGE: {field}.timeout_seconds")
    template = driver["argv_template"]
    common.require(
        type(template) is list
        and template
        and all(type(value) is str and bool(value) for value in template),
        f"E_TYPE: {field}.argv_template",
    )
    used = {
        value
        for value in template
        if value.startswith("{") and value.endswith("}")
    }
    common.exact(used, placeholders, f"{field}.placeholders")
    common.require(
        all(value in placeholders or "{" not in value for value in template),
        f"E_PLACEHOLDER: {field}.argv_template",
    )

    executed = driver["executed_files"]
    common.require(
        type(executed) is list and bool(executed),
        f"E_TYPE: {field}.executed_files",
    )
    indexes = set()
    for index, record in enumerate(executed):
        item = f"{field}.executed_files[{index}]"
        common.exact_keys(record, EXECUTED_FILE_KEYS, item)
        argv_index = common.integer(record["argv_index"], f"{item}.argv_index")
        common.require(argv_index < len(template), f"E_RANGE: {item}.argv_index")
        common.require(argv_index not in indexes, f"E_INDEX_REUSE: {argv_index}")
        indexes.add(argv_index)
        path = common.string(record["path"], f"{item}.path")
        common.exact(template[argv_index], path, f"{item}.argv")
        common.require(Path(path).is_absolute(), f"E_PATH: {item}.path")
        common.integer(record["bytes"], f"{item}.bytes", 1)
        common.digest(record["sha256"], f"{item}.sha256")
    required_indexes = {0}
    for index, value in enumerate(template):
        if value in placeholders:
            continue
        path = Path(value)
        if not path.is_absolute():
            continue
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise common.ReadinessError(
                f"E_DRIVER_FILE_STAT: {field}[{index}]: {error}"
            ) from error
        common.require(
            not stat.S_ISLNK(metadata.st_mode),
            f"E_DRIVER_FILE_SYMLINK: {field}[{index}]",
        )
        if stat.S_ISREG(metadata.st_mode):
            required_indexes.add(index)
    common.exact(indexes, required_indexes, f"{field}.executed_file_indexes")
    return driver


def _driver_file_flag(
    driver: dict[str, Any],
    flag: str,
    field: str,
) -> tuple[int, str]:
    template = driver["argv_template"]
    indexes = [index for index, value in enumerate(template) if value == flag]
    common.require(len(indexes) == 1, f"E_DRIVER_FLAG: {field}.{flag}")
    value_index = indexes[0] + 1
    common.require(value_index < len(template), f"E_DRIVER_FLAG: {field}.{flag}")
    value = template[value_index]
    common.require(
        type(value) is str
        and bool(value)
        and not value.startswith("{")
        and Path(value).is_absolute(),
        f"E_DRIVER_FLAG_VALUE: {field}.{flag}",
    )
    records = [
        record
        for record in driver["executed_files"]
        if record["argv_index"] == value_index
    ]
    common.require(
        len(records) == 1 and records[0]["path"] == value,
        f"E_DRIVER_FLAG_BINDING: {field}.{flag}",
    )
    return value_index, value


def _load_plan(
    plan_path: Path,
    contract_raw: bytes,
    candidate_raw: bytes,
    expected_roles: list[str],
) -> dict[str, Any]:
    if _BOOTSTRAP_PLAN is None:
        plan, _ = common.read_canonical(plan_path)
    else:
        common.exact(
            plan_path.absolute(),
            _BOOTSTRAP_PLAN_PATH,
            "plan.path",
        )
        plan = copy.deepcopy(_BOOTSTRAP_PLAN)
        common.exact(
            common.canonical_bytes(plan),
            _BOOTSTRAP_PLAN_RAW,
            "plan.bootstrapped_bytes",
        )
    common.exact_keys(plan, PLAN_KEYS, "plan")
    common.exact(
        plan["schema"],
        PLAN_SCHEMA,
        "plan.schema",
    )
    common.exact(plan["phase"], PHASE, "plan.phase")
    common.exact(plan["model_id"], "qwen3-14b-q4_k_m", "plan.model_id")
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
    source_root = Path(common.string(plan["source_root"], "plan.source_root"))
    source_manifest = Path(
        common.string(
            plan["source_manifest_path"],
            "plan.source_manifest_path",
        )
    )
    common.require(
        source_root.is_absolute() and source_manifest.is_absolute(),
        "E_PATH: plan.source_binding",
    )
    try:
        source_manifest.relative_to(source_root)
    except ValueError as error:
        raise common.ReadinessError(
            "E_PATH: plan.source_manifest_path"
        ) from error
    common.digest(
        plan["source_manifest_sha256"],
        "plan.source_manifest_sha256",
    )

    payload = plan["payload_roles"]
    common.require(type(payload) is dict, "E_TYPE: plan.payload_roles")
    expected_payload = sorted(set(expected_roles) - PRE_ROLES)
    common.exact(sorted(payload), expected_payload, "plan.payload_roles")
    paths = set()
    for role in expected_payload:
        path = _relative(payload[role], f"plan.payload_roles.{role}")
        common.require(path not in paths, f"E_PATH_REUSE: {path}")
        paths.add(path)

    outputs = common.exact_keys(
        plan["output_files"],
        OUTPUT_KEYS,
        "plan.output_files",
    )
    for name in sorted(outputs):
        path = _relative(outputs[name], f"plan.output_files.{name}")
        common.require(path not in paths, f"E_PATH_REUSE: {path}")
        paths.add(path)

    drivers = common.exact_keys(
        plan["drivers"],
        {"acquisition", "artifact", "fresh"},
        "plan.drivers",
    )
    artifact = _validate_driver(
        drivers["artifact"],
        "plan.drivers.artifact",
        READINESS_PLACEHOLDERS,
    )
    fresh = _validate_driver(
        drivers["fresh"],
        "plan.drivers.fresh",
        READINESS_PLACEHOLDERS,
    )
    acquisition = _validate_driver(
        drivers["acquisition"],
        "plan.drivers.acquisition",
        ACQUISITION_PLACEHOLDERS,
    )
    for name, driver in (
        ("artifact", artifact),
        ("fresh", fresh),
        ("acquisition", acquisition),
    ):
        for flag in sorted(DRIVER_FILE_FLAGS[name]):
            _driver_file_flag(
                driver,
                flag,
                f"plan.drivers.{name}",
            )
    for flag in sorted(
        DRIVER_FILE_FLAGS["artifact"] & DRIVER_FILE_FLAGS["fresh"]
    ):
        _, artifact_path = _driver_file_flag(
            artifact,
            flag,
            "plan.drivers.artifact",
        )
        _, fresh_path = _driver_file_flag(
            fresh,
            flag,
            "plan.drivers.fresh",
        )
        common.exact(
            artifact_path,
            fresh_path,
            f"E_DRIVER_FLAG_AGREEMENT: {flag}",
        )
    return plan


def _capture_executed_files(
    plan: dict[str, Any],
    output_root: Path,
) -> dict[tuple[str, int], str]:
    captured = {}
    for name in ("artifact", "fresh", "acquisition"):
        driver = plan["drivers"][name]
        template = driver["argv_template"]
        for index, record in enumerate(driver["executed_files"]):
            field = f"plan.drivers.{name}.executed_files[{index}]"
            path = Path(template[record["argv_index"]])
            flags = os.O_RDONLY | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(path, flags)
            except OSError as error:
                raise common.ReadinessError(
                    f"E_DRIVER_FILE: {path}: {error}"
                ) from error
            try:
                before = os.fstat(descriptor)
                common.require(
                    stat.S_ISREG(before.st_mode),
                    f"E_DRIVER_FILE_TYPE: {path}",
                )
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
            common.require(
                identity(before) == identity(after)
                and len(raw) == before.st_size,
                f"E_DRIVER_FILE_CHANGED: {path}",
            )
            common.exact(before.st_size, record["bytes"], f"{field}.bytes")
            common.exact(
                v2.sha256_bytes(bytes(raw)),
                record["sha256"],
                f"{field}.sha256",
            )
            destination = (
                output_root
                / "executed"
                / name
                / f"{record['argv_index']:03d}-{path.name}"
            )
            durable_write_new(destination, bytes(raw))
            destination.chmod(before.st_mode & 0o777)
            captured[(name, record["argv_index"])] = str(destination)
    return captured


def _captured_driver_file(
    plan: dict[str, Any],
    captured: dict[tuple[str, int], str],
    driver_name: str,
    flag: str,
) -> Path:
    value_index, _ = _driver_file_flag(
        plan["drivers"][driver_name],
        flag,
        f"plan.drivers.{driver_name}",
    )
    key = (driver_name, value_index)
    common.require(key in captured, f"E_CAPTURED_DRIVER_FILE: {driver_name}.{flag}")
    path = Path(captured[key])
    common.require(path.is_file() and not path.is_symlink(), f"E_CAPTURE: {flag}")
    return path


def _runtime_bundle_sources(
    plan: dict[str, Any],
    captured: dict[tuple[str, int], str],
) -> dict[str, Path]:
    result = {}
    for name, flag in (
        ("base_support", "--base-support"),
        ("driver_support", "--support"),
        ("runtime_bundle_plan", "--runtime-bundle-plan"),
    ):
        artifact = _captured_driver_file(plan, captured, "artifact", flag)
        fresh = _captured_driver_file(plan, captured, "fresh", flag)
        common.exact(
            _read_source(artifact, f"captured.artifact.{name}"),
            _read_source(fresh, f"captured.fresh.{name}"),
            f"E_CAPTURED_DRIVER_AGREEMENT: {name}",
        )
        result[name] = artifact
    return result


def route_lock_rows(
    contract: dict[str, Any],
    candidate: dict[str, Any],
    phase_id: str,
    event_ns: int,
) -> list[dict[str, Any]]:
    model = next(model for model in candidate["models"] if model["slot"] == "A")
    route = contract["incumbent_route_lock"]
    geometry = contract["model_geometry"][model["model_id"]]
    envelope_sha = v2.digest_json(contract["serving_envelope"])
    return [{
        "acquisition_id": phase_id,
        "activation_dtype": geometry["activation_dtype"],
        "activation_element_bytes": geometry["activation_element_bytes"],
        "backend": route["backend"],
        "batch_config_sha256": envelope_sha,
        "clock_id": contract["phase_protocol"]["clock_id"],
        "cuda_model_path": geometry["cuda_model_path"],
        "cut_layer": route["cut_layer"],
        "event_ns": event_ns,
        "frozen_ns": event_ns,
        "hidden_size": geometry["hidden_size"],
        "kind": "route_lock",
        "model_id": model["model_id"],
        "model_sha256": model["artifact"]["sha256"],
        "n_layer": model["n_layer"],
        "op12_shard_bytes": geometry["known_shards"]["op12"]["bytes"],
        "op12_shard_path": geometry["known_shards"]["op12"]["path"],
        "op12_shard_sha256": route["op12_shard_sha256"],
        "op12_stored_layers": route["op12_stored_layers"],
        "op15_shard_bytes": geometry["known_shards"]["op15"]["bytes"],
        "op15_shard_path": geometry["known_shards"]["op15"]["path"],
        "op15_shard_sha256": route["op15_shard_sha256"],
        "op15_stored_layers": route["op15_stored_layers"],
        "phase": PHASE,
        "phase_id": phase_id,
        "role": f"model.{model['model_id']}.route_lock",
    }]


def corpus_rows(
    frozen: list[dict[str, Any]],
    phase_id: str,
    event_ns: int,
) -> list[dict[str, Any]]:
    return [
        {
            "acquisition_id": phase_id,
            **copy.deepcopy(row),
            "event_ns": event_ns,
            "kind": "corpus_item",
            "phase": PHASE,
            "phase_id": phase_id,
            "role": "quality.corpus",
        }
        for row in frozen
    ]


def phase_lock_rows(
    contract_raw: bytes,
    candidate_raw: bytes,
    phase_id: str,
    event_ns: int,
    route_raw: bytes,
    corpus_raw: bytes,
) -> list[dict[str, Any]]:
    return [{
        "acquisition_id": phase_id,
        "candidate_sha256": v2.sha256_bytes(candidate_raw),
        "clock_id": "HOST_MONOTONIC_RAW",
        "contract_sha256": v2.sha256_bytes(contract_raw),
        "event_ns": event_ns,
        "kind": "phase_lock",
        "model_slot": "A",
        "phase": PHASE,
        "phase_id": phase_id,
        "prior_phase_result_sha256s": [],
        "quality_corpus_sha256": v2.sha256_bytes(corpus_raw),
        "role": "phase.lock",
        "route_lock_sha256": v2.sha256_bytes(route_raw),
    }]


def _rows_raw(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(v2.canonical_line(row) for row in rows)


def _substitute_argv(
    template: list[str],
    phase_id: str,
    output_dir: Path,
    pre_dir: Path,
    acquisition_started_ns: int | None,
    captured: dict[int, str],
) -> list[str]:
    replacements = {
        "{output_dir}": str(output_dir),
        "{phase_id}": phase_id,
        "{pre_dir}": str(pre_dir),
    }
    if acquisition_started_ns is not None:
        replacements["{acquisition_started_ns}"] = str(acquisition_started_ns)
    result = []
    for index, value in enumerate(template):
        if index in captured:
            result.append(captured[index])
            continue
        if value in replacements:
            result.append(replacements[value])
            continue
        path = Path(value)
        if path.is_absolute():
            try:
                metadata = os.lstat(path)
            except FileNotFoundError:
                pass
            except OSError as error:
                raise common.ReadinessError(
                    f"E_DRIVER_FILE_STAT: argv[{index}]: {error}"
                ) from error
            else:
                common.require(
                    not stat.S_ISLNK(metadata.st_mode),
                    f"E_DRIVER_FILE_SYMLINK: argv[{index}]",
                )
                common.require(
                    not stat.S_ISREG(metadata.st_mode),
                    f"E_UNCAPTURED_DRIVER_FILE: argv[{index}]",
                )
        result.append(value)
    return result


def _validate_payload_role(
    path: Path,
    role: str,
    phase_id: str,
    acquisition_started_ns: int,
    phase_closed_ns: int,
) -> bytes:
    rows, raw = _read_jsonl(path, role)
    previous = None
    for index, row in enumerate(rows):
        field = f"{role}[{index}]"
        common.exact(row.get("role"), role, f"{field}.role")
        common.exact(row.get("phase"), PHASE, f"{field}.phase")
        common.exact(row.get("phase_id"), phase_id, f"{field}.phase_id")
        common.exact(
            row.get("acquisition_id"),
            phase_id,
            f"{field}.acquisition_id",
        )
        event_ns = common.integer(row.get("event_ns"), f"{field}.event_ns", 1)
        common.require(
            acquisition_started_ns <= event_ns <= phase_closed_ns,
            f"E_ACQUISITION_INTERVAL: {field}",
        )
        if previous is not None:
            common.require(previous <= event_ns, f"E_EVENT_ORDER: {role}")
        previous = event_ns
    return raw


def _artifact_record(role: str, path: str, raw: bytes) -> dict[str, Any]:
    return {
        "bytes": len(raw),
        "format": "CANONICAL_ASCII_JSONL",
        "path": path,
        "role": role,
        "sha256": v2.sha256_bytes(raw),
    }


def acquire(
    plan_path: Path,
    output_root: Path,
    phase_id: str,
    *,
    runner: AcquisitionRunner | None = None,
    now_ns: Callable[[], int] = clock_ns,
    preflight_collector: Callable[..., bytes] = preflight.collect,
    v22_evaluator: Callable[..., dict[str, Any]] = v22.evaluate_root,
    v23_validator: Callable[..., dict[str, Any]] = v23.validate_readiness,
    runtime_bundle_validator: Callable[..., dict[str, Any]] =
        _default_runtime_bundle_validator,
    plan_validator: Callable[[Path, dict[str, Any]], None] =
        _default_plan_validator,
    contract_path: Path = v23.DEFAULT_CONTRACT,
    candidate_path: Path = v23.DEFAULT_CANDIDATE,
) -> dict[str, Any]:
    common.require(output_root.is_absolute(), "E_PATH: output_root")
    common.require(not output_root.exists(), "E_EXISTS: output_root")
    common.require(
        phase_id.startswith("cp0-r1-v23-a-only-")
        and len(phase_id) <= 128
        and all(value.isalnum() or value in ".-_" for value in phase_id),
        "E_PHASE_ID",
    )
    contract, contract_raw, candidate, candidate_raw = v23.validate_inputs(
        contract_path,
        candidate_path,
    )
    (
        v22_contract,
        v22_contract_raw,
        v22_candidate,
        v22_candidate_raw,
        v22_parent,
        frozen_corpus,
    ) = v22.validate_inputs(v22.DEFAULT_CONTRACT, candidate_path)
    del v22_parent
    common.exact(candidate, v22_candidate, "E_CANDIDATE_VERSION")
    common.exact(candidate_raw, v22_candidate_raw, "E_CANDIDATE_BYTES")
    expected_roles = v22_contract["phase_protocol"]["phase_roles"][PHASE]
    plan = _load_plan(plan_path, contract_raw, candidate_raw, expected_roles)
    plan_validator(plan_path, plan)
    runner = runner or SubprocessRunner()

    output_root.mkdir(parents=True, exist_ok=False)
    captured = _capture_executed_files(plan, output_root)
    runtime_bundle_sources = _runtime_bundle_sources(plan, captured)
    raw_dir = output_root / "raw"
    raw_dir.mkdir()
    pre_dir = output_root / "pre"
    pre_dir.mkdir()
    artifact_dir = output_root / "artifact"
    artifact_dir.mkdir()
    fresh_dir = output_root / "fresh"
    fresh_dir.mkdir()
    acquisition_dir = output_root / "acquisition"
    acquisition_dir.mkdir()
    phase_opened_ns = now_ns()

    route_role = "model.qwen3-14b-q4_k_m.route_lock"
    route_raw = _rows_raw(
        route_lock_rows(
            v22_contract,
            candidate,
            phase_id,
            phase_opened_ns,
        )
    )
    corpus_raw = _rows_raw(
        corpus_rows(frozen_corpus, phase_id, phase_opened_ns)
    )
    lock_event_ns = now_ns()
    phase_lock_raw = _rows_raw(
        phase_lock_rows(
            v22_contract_raw,
            candidate_raw,
            phase_id,
            lock_event_ns,
            route_raw,
            corpus_raw,
        )
    )
    role_raw = {
        route_role: route_raw,
        "quality.corpus": corpus_raw,
        "phase.lock": phase_lock_raw,
    }
    for name, raw in (
        ("route_lock.jsonl", route_raw),
        ("quality_corpus.jsonl", corpus_raw),
        ("phase_lock.jsonl", phase_lock_raw),
    ):
        durable_write_new(pre_dir / name, raw)

    artifact_driver = plan["drivers"]["artifact"]
    artifact_argv = _substitute_argv(
        artifact_driver["argv_template"],
        phase_id,
        artifact_dir,
        pre_dir,
        None,
        {
            index: path
            for (name, index), path in captured.items()
            if name == "artifact"
        },
    )
    artifact_started_ns = now_ns()
    artifact_completed = runner.run(
        artifact_argv,
        timeout=artifact_driver["timeout_seconds"],
    )
    artifact_completed_ns = now_ns()
    durable_write_new(
        output_root / "artifact-driver.stdout",
        artifact_completed.stdout,
    )
    durable_write_new(
        output_root / "artifact-driver.stderr",
        artifact_completed.stderr,
    )
    durable_json(
        output_root / "ARTIFACT_DRIVER_RECEIPT.json",
        {
            "argv": artifact_argv,
            "completed_ns": artifact_completed_ns,
            "returncode": artifact_completed.returncode,
            "schema": "s39-cp0-r1-artifact-driver-receipt-v2.3",
            "started_ns": artifact_started_ns,
        },
    )
    common.exact(artifact_completed.returncode, 0, "E_ARTIFACT_DRIVER_EXIT")
    common.exact(artifact_completed.stderr, b"", "E_ARTIFACT_DRIVER_STDERR")

    preflight_raw = preflight_collector(
        v22_contract,
        candidate,
        PHASE,
        phase_id,
        {"A": v2.parse_json(route_raw.rstrip(b"\n"), route_role)},
        artifact_driver["timeout_seconds"],
    )
    role_raw["phase.preflight"] = preflight_raw
    durable_write_new(pre_dir / "phase_preflight.jsonl", preflight_raw)

    fresh_driver = plan["drivers"]["fresh"]
    fresh_argv = _substitute_argv(
        fresh_driver["argv_template"],
        phase_id,
        fresh_dir,
        pre_dir,
        None,
        {
            index: path
            for (name, index), path in captured.items()
            if name == "fresh"
        },
    )
    fresh_started_ns = now_ns()
    fresh_completed = runner.run(
        fresh_argv,
        timeout=fresh_driver["timeout_seconds"],
    )
    fresh_completed_ns = now_ns()
    durable_write_new(
        output_root / "fresh-driver.stdout",
        fresh_completed.stdout,
    )
    durable_write_new(
        output_root / "fresh-driver.stderr",
        fresh_completed.stderr,
    )
    durable_json(
        output_root / "FRESH_DRIVER_RECEIPT.json",
        {
            "argv": fresh_argv,
            "completed_ns": fresh_completed_ns,
            "returncode": fresh_completed.returncode,
            "schema": "s39-cp0-r1-fresh-driver-receipt-v2.3",
            "started_ns": fresh_started_ns,
        },
    )
    common.exact(fresh_completed.returncode, 0, "E_FRESH_DRIVER_EXIT")
    common.exact(fresh_completed.stderr, b"", "E_FRESH_DRIVER_STDERR")
    acquisition_started_ns = now_ns()

    acquisition_driver = plan["drivers"]["acquisition"]
    argv = _substitute_argv(
        acquisition_driver["argv_template"],
        phase_id,
        acquisition_dir,
        pre_dir,
        acquisition_started_ns,
        {
            index: path
            for (name, index), path in captured.items()
            if name == "acquisition"
        },
    )
    completed = runner.run(
        argv,
        timeout=acquisition_driver["timeout_seconds"],
    )
    driver_completed_ns = now_ns()
    durable_write_new(output_root / "driver.stdout", completed.stdout)
    durable_write_new(output_root / "driver.stderr", completed.stderr)
    durable_json(
        output_root / "DRIVER_RECEIPT.json",
        {
            "argv": argv,
            "completed_ns": driver_completed_ns,
            "returncode": completed.returncode,
            "schema": "s39-cp0-r1-a-only-driver-receipt-v2.3",
            "started_ns": acquisition_started_ns,
        },
    )
    common.exact(completed.returncode, 0, "E_DRIVER_EXIT")
    common.exact(completed.stderr, b"", "E_DRIVER_STDERR")

    phase_closed_ns = driver_completed_ns
    for role in sorted(set(expected_roles) - PRE_ROLES):
        path = acquisition_dir / plan["payload_roles"][role]
        common.require(path.is_file() and not path.is_symlink(), f"E_OUTPUT: {role}")
        role_raw[role] = _validate_payload_role(
            path,
            role,
            phase_id,
            acquisition_started_ns,
            phase_closed_ns,
        )

    artifacts = []
    for index, role in enumerate(expected_roles):
        raw = role_raw[role]
        relative = f"raw/{index:02d}.jsonl"
        durable_write_new(output_root / relative, raw)
        artifacts.append(_artifact_record(role, relative, raw))
    manifest = {
        "acquisition_started_ns": acquisition_started_ns,
        "artifacts": artifacts,
        "candidate_sha256": v2.sha256_bytes(candidate_raw),
        "clock_id": "HOST_MONOTONIC_RAW",
        "contract_sha256": v2.sha256_bytes(v22_contract_raw),
        "phase": PHASE,
        "phase_closed_ns": phase_closed_ns,
        "phase_id": phase_id,
        "phase_opened_ns": phase_opened_ns,
        "schema": "s39-cp0-r1-evidence-bundle-v2.1",
    }
    durable_json(output_root / v22.MANIFEST_NAME, manifest)

    v22_result = v22_evaluator(
        output_root,
        v22.MANIFEST_NAME,
        v22_contract,
        v22_contract_raw,
        candidate,
        candidate_raw,
        v22.validate_inputs(v22.DEFAULT_CONTRACT, candidate_path)[4],
        frozen_corpus,
        [],
    )
    common.exact(
        v22_result["status"],
        "MODEL_A_QUALIFICATION_PASS",
        "E_V22_RESULT",
    )

    output_files = {
        "artifact_snapshot":
            artifact_dir / plan["output_files"]["artifact_snapshot"],
        "fresh_snapshot":
            fresh_dir / plan["output_files"]["fresh_snapshot"],
        "readiness_lock":
            fresh_dir / plan["output_files"]["readiness_lock"],
        "runtime_bundle_identity":
            acquisition_dir / plan["output_files"]["runtime_bundle_identity"],
        "runtime_identity":
            acquisition_dir / plan["output_files"]["runtime_identity"],
    }
    for name, path in output_files.items():
        common.require(path.is_file() and not path.is_symlink(), f"E_OUTPUT: {name}")
    v23_result = v23_validator(
        contract,
        candidate,
        output_root,
        output_files["artifact_snapshot"],
        output_files["readiness_lock"],
        output_files["fresh_snapshot"],
        output_files["runtime_identity"],
        candidate_path=candidate_path,
    )
    runtime_bundle_result = runtime_bundle_validator(
        base_support=runtime_bundle_sources["base_support"],
        driver_support=runtime_bundle_sources["driver_support"],
        contract=contract_path,
        candidate=candidate_path,
        runtime_bundle_plan=runtime_bundle_sources["runtime_bundle_plan"],
        manifest=output_root / v22.MANIFEST_NAME,
        base_artifact_snapshot=output_files["artifact_snapshot"],
        base_readiness_lock=output_files["readiness_lock"],
        base_fresh_snapshot=output_files["fresh_snapshot"],
        base_runtime_identity=output_files["runtime_identity"],
        runtime_bundle_snapshot=artifact_dir / "runtime_bundle_snapshot.json",
        runtime_bundle_readiness_lock=(
            fresh_dir / "runtime_bundle_readiness_lock.json"
        ),
        runtime_bundle_fresh=fresh_dir / "runtime_bundle_fresh.json",
        runtime_bundle_identity=output_files["runtime_bundle_identity"],
    )
    common.exact_keys(
        runtime_bundle_result,
        RUNTIME_BUNDLE_RESULT_KEYS,
        "runtime_bundle_result",
    )
    common.exact(
        runtime_bundle_result["schema"],
        "s39-cp0-r1-runtime-bundle-validation-v1",
        "runtime_bundle_result.schema",
    )
    common.exact(
        runtime_bundle_result["status"],
        "RUNTIME_BUNDLE_PROVENANCE_PASS",
        "runtime_bundle_result.status",
    )
    for name in sorted(
        RUNTIME_BUNDLE_RESULT_KEYS - {"schema", "status"}
    ):
        common.digest(
            runtime_bundle_result[name],
            f"runtime_bundle_result.{name}",
        )
    result = {
        "bundle_manifest_sha256": v2.sha256_bytes(
            (output_root / v22.MANIFEST_NAME).read_bytes()
        ),
        "phase": PHASE,
        "phase_id": phase_id,
        "readiness": v23_result,
        "runtime_bundle": runtime_bundle_result,
        "schema": "s39-cp0-r1-a-only-acquisition-result-v2.3",
        "status": "MODEL_A_QUALIFICATION_PASS",
        "v2_2_result_sha256": v2.sha256_bytes(v2.canonical_bytes(v22_result)),
    }
    durable_json(output_root / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one digest-pinned CP0-R1 V2.3 A_ONLY acquisition"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--contract", type=Path, default=v23.DEFAULT_CONTRACT)
    parser.add_argument("--candidate", type=Path, default=v23.DEFAULT_CANDIDATE)
    args = parser.parse_args()
    try:
        result = acquire(
            args.plan,
            args.output_root,
            args.phase_id,
            contract_path=args.contract,
            candidate_path=args.candidate,
        )
        print(v2.canonical_bytes(result).decode("ascii"), end="")
        return 0
    except (
        common.ReadinessError,
        v2.EvidenceError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        if args.output_root.is_dir():
            failure_path = args.output_root / "FAILURE.json"
            if not failure_path.exists():
                try:
                    durable_json(
                        failure_path,
                        {
                            "error": str(error),
                            "phase": PHASE,
                            "phase_id": args.phase_id,
                            "schema":
                                "s39-cp0-r1-a-only-acquisition-failure-v2.3",
                            "status": "REFUSED",
                        },
                    )
                except OSError:
                    pass
        print(f"CP0_R1_A_ONLY_V2_3_REFUSED: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
