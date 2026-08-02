#!/usr/bin/env python3
"""Validate the V2.5 RTX-local V2.4 fan-in handoff."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import stat
from typing import Any

import v25_common as common


PLAN_SCHEMA = "s39-v25-remote-fan-in-plan-v1"
RECEIPT_SCHEMA = "s39-v25-remote-fan-in-receipt-v1"
BUNDLE_SCHEMA = "s39-v25-remote-fan-in-bundle-manifest-v1"
CONTROLLER_TRANSPORT_SCHEMA = (
    "s39-v25-remote-fan-in-controller-transport-v1"
)
REMOTE_CLEANUP_SCHEMA = "s39-v25-remote-fan-in-post-cleanup-v1"
MANAGED_TRANSPORT_SCHEMA = "s39-managed-remote-transport-process-v1"
RUNTIME_PROCESS_SCHEMA = "s39-runtime-process-source-v1"
MANAGED_CLEANUP_SCHEMA = "s39-managed-remote-cleanup-v1"
PHASE = "A_ONLY"
ROLE = "remote_fan_in"
RAW_MANIFEST_NAME = "EVIDENCE_BUNDLE_V2_5_RAW.json"
V24_RAW_MANIFEST_NAME = "EVIDENCE_BUNDLE_V2_4.json"
RUNTIME_IDENTITY_NAME = "runtime.json"
ACQUISITION_NAME = "acquisition.json"
INPUT_ROLES = {
    "artifact_root",
    "candidate",
    "contract",
    "fresh_readiness",
    "orchestration_plan",
    "phase_lock",
    "pre.phase_lock",
    "pre.phase_preflight",
    "pre.quality_corpus",
    "pre.route_lock",
    "preparation",
    "runtime_plan",
}
PRE_INPUT_NAMES = {
    "pre.phase_lock": "phase-lock.jsonl",
    "pre.phase_preflight": "phase-preflight.jsonl",
    "pre.quality_corpus": "quality-corpus.jsonl",
    "pre.route_lock": "route-lock.jsonl",
}
SOURCE_ROLES = {
    "authority",
    "fan_in",
    "production_common",
    "v24_common",
    "v24_contract_builder",
}
BOOTSTRAP_SOURCE_ORDER = (
    "fan_in",
    "production_common",
    "authority",
    "v24_common",
    "v24_contract_builder",
)
BOOTSTRAP_PRE_ORDER = tuple(sorted(PRE_INPUT_NAMES))
ARTIFACT_KEYS = {"bytes", "path", "sha256", "stat"}
STAT_KEYS = {
    "build_id",
    "ctime_ns",
    "device_id",
    "inode",
    "mode",
    "mtime_ns",
    "size",
}
DEPENDENCY_KEYS = STAT_KEYS | {"path", "sha256"}
REMOTE_PROCESS_KEYS = {
    "boot_id",
    "bundle_id",
    "controller_clock",
    "controller_observed_ns",
    "endpoint",
    "launch_token",
    "launcher_path",
    "loaded_repo_component_ids",
    "pgid",
    "pid",
    "remote_clock",
    "remote_observed_ns",
    "schema",
    "start_ticks",
    "system_dependencies",
}
MANAGED_TRANSPORT_KEYS = {
    "argv",
    "bundle_id",
    "endpoint",
    "host_boot_id",
    "managed_launcher_pid",
    "managed_launcher_start_ticks",
    "observed_ns",
    "pid",
    "plan_sha256",
    "remote_boot_id",
    "schema",
    "start_ticks",
}
MANAGED_CLEANUP_KEYS = {
    "absent",
    "boot_id",
    "clock",
    "gpu_uuid",
    "launch_token",
    "matching_nvml_pids",
    "matching_process_groups",
    "matching_processes",
    "observed_ns",
    "pgid",
    "pid",
    "schema",
    "start_ticks",
}

_REMOTE_BOOTSTRAP_SOURCE = r"""
import os,pathlib,runpy,stat,sys
def die(message):
    raise SystemExit(message)
def one(argv,name):
    if argv.count(name)!=1:
        die("E_BOOTSTRAP_FLAG")
    index=argv.index(name)
    if index+1>=len(argv):
        die("E_BOOTSTRAP_FLAG")
    return index
def stable_copy(source,destination):
    flags=os.O_RDONLY|os.O_CLOEXEC
    if hasattr(os,"O_NOFOLLOW"):
        flags|=os.O_NOFOLLOW
    source_fd=os.open(source,flags)
    try:
        before=os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=67108864:
            die("E_BOOTSTRAP_CAPTURE")
        output=os.open(
            destination,
            os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,
            0o400,
        )
        try:
            consumed=0
            while True:
                block=os.read(source_fd,1048576)
                if not block:
                    break
                offset=0
                while offset<len(block):
                    written=os.write(output,block[offset:])
                    if written<1:
                        die("E_BOOTSTRAP_WRITE")
                    offset+=written
                consumed+=len(block)
            os.fsync(output)
        finally:
            os.close(output)
        after=os.fstat(source_fd)
    finally:
        os.close(source_fd)
    identity=lambda value:(
        value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,
        value.st_ctime_ns,value.st_mode
    )
    if identity(before)!=identity(after) or consumed!=before.st_size:
        die("E_BOOTSTRAP_CAPTURE_CHANGED")
arguments=sys.argv[1:]
if len(arguments)<10 or arguments[9]!="--":
    die("E_BOOTSTRAP_ARGV")
fan,production,authority,v24_common,builder=arguments[:5]
pre_values=arguments[5:9]
argv=arguments[10:]
private=[fan,production,authority,v24_common,builder,*pre_values]
root=pathlib.Path(fan).parent
if (
    not root.is_absolute()
    or len(set(private))!=len(private)
    or any(pathlib.Path(value).parent!=root for value in private)
):
    die("E_BOOTSTRAP_PRIVATE")
pre_index=one(argv,"--pre-dir")
pre_dir=pathlib.Path(argv[pre_index+1])
pre_names=(
    "phase-lock.jsonl","phase-preflight.jsonl",
    "quality-corpus.jsonl","route-lock.jsonl"
)
mapping={
    str(root/"production_common_v1.py"):production,
    str(root.parent/"cp0_r1_evidence_v24.py"):authority,
    str(root.parent/"v24_common.py"):v24_common,
    str(root.parent/"build_contract_v24.py"):builder,
}
for name,value in zip(pre_names,pre_values):
    mapping[str((pre_dir/"raw"/name).resolve())]=value
mono_index=one(argv,"--mono")
joint_index=one(argv,"--joint")
os.chmod(root,0o700)
mono_private=str(root/"capture-mono")
joint_private=str(root/"capture-joint")
stable_copy(argv[mono_index+1],mono_private)
stable_copy(argv[joint_index+1],joint_private)
os.chmod(root,0o500)
argv[mono_index+1]=mono_private
argv[joint_index+1]=joint_private
original_open=os.open
original_read_bytes=pathlib.Path.read_bytes
def redirected_open(path,flags,mode=0o777,*,dir_fd=None):
    if dir_fd is None:
        try:
            path=mapping.get(os.fspath(path),path)
        except TypeError:
            pass
    if dir_fd is None:
        return original_open(path,flags,mode)
    return original_open(path,flags,mode,dir_fd=dir_fd)
def redirected_read_bytes(path):
    target=pathlib.Path(mapping.get(str(path),str(path)))
    return original_read_bytes(target)
os.open=redirected_open
pathlib.Path.read_bytes=redirected_read_bytes
sys.argv=[fan,*argv]
try:
    runpy.run_path(fan,run_name="__main__")
finally:
    os.open=original_open
    pathlib.Path.read_bytes=original_read_bytes
""".strip()
REMOTE_BOOTSTRAP = (
    "import base64;"
    "exec(compile(base64.b64decode("
    + repr(base64.b64encode(_REMOTE_BOOTSTRAP_SOURCE.encode("ascii")))
    + "),'<s39-v25-fan-in-bootstrap>','exec'))"
)


def _canonical_absolute(value: Any, field: str) -> str:
    value = common.absolute_path(value, field)
    common.require(str(Path(value)) == value, f"E_PATH_CANONICAL: {field}")
    return value


def _artifact(value: Any, field: str) -> dict[str, Any]:
    value = common.exact_keys(value, ARTIFACT_KEYS, field)
    size = common.integer(value["bytes"], f"{field}.bytes", 1)
    _canonical_absolute(value["path"], f"{field}.path")
    common.digest(value["sha256"], f"{field}.sha256")
    metadata = common.exact_keys(value["stat"], STAT_KEYS, f"{field}.stat")
    common.exact(metadata["build_id"], None, f"{field}.stat.build_id")
    for key in STAT_KEYS - {"build_id"}:
        common.integer(metadata[key], f"{field}.stat.{key}")
    common.require(
        metadata["inode"] > 0
        and metadata["size"] == size
        and stat.S_ISREG(metadata["mode"]),
        f"E_STAT: {field}",
    )
    return value


def _argv(value: Any, field: str) -> list[str]:
    common.require(
        type(value) is list and 1 <= len(value) <= 256,
        f"E_ARGV: {field}",
    )
    for index, item in enumerate(value):
        common.text(item, f"{field}[{index}]")
    return value


def _phase_ids(outer: Any, inner: Any, field: str) -> tuple[str, str]:
    outer = common.text(outer, f"{field}.outer", 128)
    inner = common.text(inner, f"{field}.inner", 128)
    outer_prefix = "cp0-r1-v25-a-only-"
    inner_prefix = "cp0-r1-v24-a-only-"
    common.require(
        outer.startswith(outer_prefix)
        and inner.startswith(inner_prefix)
        and outer[len(outer_prefix):] == inner[len(inner_prefix):],
        f"E_PHASE_LINK: {field}",
    )
    return outer, inner


def expected_producer_argv(value: dict[str, Any]) -> list[str]:
    inputs = value["input_artifacts"]
    sources = value["source_artifacts"]
    pre_dir = Path(inputs["pre.phase_lock"]["path"]).parents[1]
    prefix = [
        value["remote_python"]["path"],
        "-I",
        "-c",
        REMOTE_BOOTSTRAP,
        *(sources[role]["path"] for role in BOOTSTRAP_SOURCE_ORDER),
        *(inputs[role]["path"] for role in BOOTSTRAP_PRE_ORDER),
        "--",
    ]
    return prefix + [
        "--runtime",
        value["remote_runtime_output"],
        "--acquisition",
        value["remote_acquisition_output"],
        "--bundle-root",
        value["remote_bundle_root"],
        "--pre-dir",
        str(pre_dir),
        "--started",
        str(value["acquisition_started_ns"]),
        "--contract",
        inputs["contract"]["path"],
        "--candidate",
        inputs["candidate"]["path"],
        "--runtime-plan",
        inputs["runtime_plan"]["path"],
        "--root",
        inputs["artifact_root"]["path"],
        "--preparation",
        inputs["preparation"]["path"],
        "--phase-lock",
        inputs["phase_lock"]["path"],
        "--fresh",
        inputs["fresh_readiness"]["path"],
        "--mono",
        value["capture_input_paths"]["cuda_monolithic"],
        "--joint",
        value["capture_input_paths"]["joint_phone_cuda"],
    ]


def _within(path: str, root: str) -> bool:
    return Path(path).is_relative_to(Path(root))


def validate_plan(value: Any) -> dict[str, Any]:
    value = common.exact_keys(
        value,
        {
            "acquisition_started_ns",
            "capture_input_paths",
            "contract_validator",
            "executor",
            "input_artifacts",
            "local_common",
            "local_python",
            "managed_launcher",
            "managed_plan",
            "managed_plan_sha256",
            "outer_phase_id",
            "phase",
            "producer_argv",
            "remote_acquisition_output",
            "remote_bundle_root",
            "remote_python",
            "remote_runtime_output",
            "role",
            "schema",
            "source_artifacts",
            "timeout_seconds",
            "v24_phase_id",
        },
        "fan_in_plan",
    )
    common.exact(value["schema"], PLAN_SCHEMA, "fan_in_plan.schema")
    common.exact(value["phase"], PHASE, "fan_in_plan.phase")
    common.exact(value["role"], ROLE, "fan_in_plan.role")
    _phase_ids(
        value["outer_phase_id"],
        value["v24_phase_id"],
        "fan_in_plan.phase_id",
    )
    common.digest(
        value["managed_plan_sha256"],
        "fan_in_plan.managed_plan_sha256",
    )
    for key in (
        "contract_validator",
        "executor",
        "local_common",
        "local_python",
        "managed_launcher",
        "remote_python",
    ):
        _artifact(value[key], f"fan_in_plan.{key}")
    managed_plan = _artifact(
        value["managed_plan"],
        "fan_in_plan.managed_plan",
    )
    common.exact(
        managed_plan["sha256"],
        value["managed_plan_sha256"],
        "fan_in_plan.managed_plan.sha256",
    )
    local_paths = [
        value[key]["path"]
        for key in (
            "contract_validator",
            "executor",
            "local_common",
            "managed_launcher",
            "managed_plan",
        )
    ]
    common.require(
        len(local_paths) == len(set(local_paths)),
        "E_LOCAL_PATH_ALIAS",
    )
    sources = common.exact_keys(
        value["source_artifacts"],
        SOURCE_ROLES,
        "fan_in_plan.sources",
    )
    for role in sorted(SOURCE_ROLES):
        _artifact(sources[role], f"fan_in_plan.sources.{role}")
    inputs = common.exact_keys(
        value["input_artifacts"],
        INPUT_ROLES,
        "fan_in_plan.inputs",
    )
    for role in sorted(INPUT_ROLES):
        _artifact(inputs[role], f"fan_in_plan.inputs.{role}")
    capture_paths = common.exact_keys(
        value["capture_input_paths"],
        {"cuda_monolithic", "joint_phone_cuda"},
        "fan_in_plan.capture_input_paths",
    )
    for role in sorted(capture_paths):
        _canonical_absolute(
            capture_paths[role],
            f"fan_in_plan.capture_input_paths.{role}",
        )
    common.require(
        len(set(capture_paths.values())) == 2,
        "E_CAPTURE_INPUT_ALIAS",
    )
    bundle_root = _canonical_absolute(
        value["remote_bundle_root"],
        "fan_in_plan.remote_bundle_root",
    )
    runtime_output = _canonical_absolute(
        value["remote_runtime_output"],
        "fan_in_plan.remote_runtime_output",
    )
    acquisition_output = _canonical_absolute(
        value["remote_acquisition_output"],
        "fan_in_plan.remote_acquisition_output",
    )
    remote_paths = [
        value["remote_python"]["path"],
        *(row["path"] for row in sources.values()),
        *(row["path"] for row in inputs.values()),
        *capture_paths.values(),
        runtime_output,
        acquisition_output,
    ]
    common.require(
        len(remote_paths) == len(set(remote_paths)),
        "E_REMOTE_PATH_ALIAS",
    )
    common.require(
        all(not _within(path, bundle_root) for path in remote_paths),
        "E_REMOTE_BUNDLE_CONTAINMENT",
    )
    pre_dir = Path(inputs["pre.phase_lock"]["path"]).parents[1]
    for role, name in PRE_INPUT_NAMES.items():
        common.exact(
            inputs[role]["path"],
            str(pre_dir / "raw" / name),
            f"E_PRE_RAW_PATH: {role}",
        )
    common.integer(
        value["acquisition_started_ns"],
        "fan_in_plan.acquisition_started_ns",
        1,
    )
    timeout = common.integer(
        value["timeout_seconds"],
        "fan_in_plan.timeout_seconds",
        1,
    )
    common.require(timeout <= 7200, "E_TIMEOUT")
    argv = _argv(value["producer_argv"], "fan_in_plan.producer_argv")
    common.exact(
        argv,
        expected_producer_argv(value),
        "fan_in_plan.producer_argv",
    )
    return value


def parse_plan(path: Path, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    common.digest(expected_sha256, "fan_in_plan.sha256")
    value, raw = common.read_canonical(path)
    common.exact(
        common.sha256_bytes(raw),
        expected_sha256,
        "fan_in_plan.sha256",
    )
    return validate_plan(value), raw


def _file_row(value: Any, field: str) -> dict[str, Any]:
    value = common.exact_keys(
        value,
        {"bytes", "path", "sha256"},
        field,
    )
    common.integer(value["bytes"], f"{field}.bytes", 1)
    relative = common.relative_path(value["path"], f"{field}.path")
    common.require(str(Path(relative)) == relative, f"E_PATH_CANONICAL: {field}")
    common.digest(value["sha256"], f"{field}.sha256")
    return value


def bundle_manifest(
    root_path: str,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "file_count": len(files),
        "files": files,
        "root_path": root_path,
        "root_sha256": common.sha256_bytes(
            b"s39:v25:remote-fan-in-bundle:v1\0"
            + common.canonical_bytes(files)
        ),
        "schema": BUNDLE_SCHEMA,
        "total_bytes": sum(row["bytes"] for row in files),
    }


def _bundle_manifest(value: Any, field: str) -> dict[str, Any]:
    value = common.exact_keys(
        value,
        {
            "file_count",
            "files",
            "root_path",
            "root_sha256",
            "schema",
            "total_bytes",
        },
        field,
    )
    common.exact(value["schema"], BUNDLE_SCHEMA, f"{field}.schema")
    _canonical_absolute(value["root_path"], f"{field}.root_path")
    files = value["files"]
    common.require(type(files) is list and bool(files), f"E_FILES: {field}")
    paths = []
    total = 0
    for index, row in enumerate(files):
        row = _file_row(row, f"{field}.files[{index}]")
        paths.append(row["path"])
        total += row["bytes"]
    common.exact(paths, sorted(set(paths)), f"{field}.file_order")
    common.exact(value["file_count"], len(files), f"{field}.file_count")
    common.exact(value["total_bytes"], total, f"{field}.total_bytes")
    common.exact(
        value,
        bundle_manifest(value["root_path"], files),
        f"{field}.root",
    )
    return value


def _controller_transport(value: Any, field: str) -> dict[str, Any]:
    value = common.exact_keys(
        value,
        {
            "argv",
            "clock",
            "completed_ns",
            "controller_boot_id",
            "exit_code",
            "pid",
            "schema",
            "start_ticks",
            "started_ns",
        },
        field,
    )
    common.exact(
        value["schema"],
        CONTROLLER_TRANSPORT_SCHEMA,
        f"{field}.schema",
    )
    common.exact(
        value["clock"],
        "CONTROLLER_MONOTONIC",
        f"{field}.clock",
    )
    _argv(value["argv"], f"{field}.argv")
    common.uuid(value["controller_boot_id"], f"{field}.controller_boot_id")
    started = common.integer(value["started_ns"], f"{field}.started_ns", 1)
    common.integer(value["completed_ns"], f"{field}.completed_ns", started + 1)
    common.integer(value["pid"], f"{field}.pid", 1)
    common.integer(value["start_ticks"], f"{field}.start_ticks", 1)
    common.exact(value["exit_code"], 0, f"{field}.exit_code")
    return value


def _managed_transport(value: Any, field: str) -> dict[str, Any]:
    value = common.exact_keys(value, MANAGED_TRANSPORT_KEYS, field)
    common.exact(value["schema"], MANAGED_TRANSPORT_SCHEMA, f"{field}.schema")
    common.exact(value["endpoint"], "cuda", f"{field}.endpoint")
    _argv(value["argv"], f"{field}.argv")
    common.uuid(value["host_boot_id"], f"{field}.host_boot_id")
    common.uuid(value["remote_boot_id"], f"{field}.remote_boot_id")
    common.text(value["bundle_id"], f"{field}.bundle_id", 128)
    common.digest(value["plan_sha256"], f"{field}.plan_sha256")
    for key in (
        "managed_launcher_pid",
        "managed_launcher_start_ticks",
        "observed_ns",
        "pid",
        "start_ticks",
    ):
        common.integer(value[key], f"{field}.{key}", 1)
    return value


def _remote_process(value: Any, field: str) -> dict[str, Any]:
    value = common.exact_keys(value, REMOTE_PROCESS_KEYS, field)
    common.exact(value["schema"], RUNTIME_PROCESS_SCHEMA, f"{field}.schema")
    common.exact(
        value["controller_clock"],
        "CONTROLLER_MONOTONIC",
        f"{field}.controller_clock",
    )
    common.exact(
        value["remote_clock"],
        "RTX_CLOCK_MONOTONIC_RAW",
        f"{field}.remote_clock",
    )
    common.uuid(value["boot_id"], f"{field}.boot_id")
    common.text(value["bundle_id"], f"{field}.bundle_id", 128)
    common.exact(value["endpoint"], "cuda", f"{field}.endpoint")
    _canonical_absolute(value["launcher_path"], f"{field}.launcher_path")
    ids = value["loaded_repo_component_ids"]
    common.require(
        type(ids) is list and ids == sorted(set(ids)) and bool(ids),
        f"E_COMPONENT_IDS: {field}",
    )
    for index, component_id in enumerate(ids):
        common.text(component_id, f"{field}.components[{index}]", 128)
    for key in (
        "controller_observed_ns",
        "pgid",
        "pid",
        "remote_observed_ns",
        "start_ticks",
    ):
        common.integer(value[key], f"{field}.{key}", 1)
    common.exact(value["pgid"], value["pid"], f"{field}.pgid")
    token = common.text(value["launch_token"], f"{field}.launch_token", 32)
    common.require(
        len(token) == 32
        and all(character in "0123456789abcdef" for character in token),
        f"E_LAUNCH_TOKEN: {field}",
    )
    dependencies = value["system_dependencies"]
    common.require(
        type(dependencies) is list and bool(dependencies),
        f"E_DEPENDENCIES: {field}",
    )
    paths = []
    for index, dependency in enumerate(dependencies):
        item = f"{field}.system_dependencies[{index}]"
        dependency = common.exact_keys(dependency, DEPENDENCY_KEYS, item)
        _canonical_absolute(dependency["path"], f"{item}.path")
        common.digest(dependency["sha256"], f"{item}.sha256")
        paths.append(dependency["path"])
        build_id = dependency["build_id"]
        common.require(
            build_id is None
            or (type(build_id) is str and bool(build_id) and build_id.isascii()),
            f"E_BUILD_ID: {item}",
        )
        for key in STAT_KEYS - {"build_id"}:
            common.integer(dependency[key], f"{item}.{key}")
        common.require(
            dependency["inode"] > 0
            and dependency["size"] > 0
            and stat.S_ISREG(dependency["mode"]),
            f"E_STAT: {item}",
        )
    common.exact(paths, sorted(set(paths)), f"{field}.dependency_order")
    return value


def _managed_cleanup(value: Any, field: str) -> dict[str, Any]:
    value = common.exact_keys(value, MANAGED_CLEANUP_KEYS, field)
    common.exact(value["schema"], MANAGED_CLEANUP_SCHEMA, f"{field}.schema")
    common.uuid(value["boot_id"], f"{field}.boot_id")
    common.exact(value["clock"], "RTX_CLOCK_MONOTONIC_RAW", f"{field}.clock")
    common.text(value["gpu_uuid"], f"{field}.gpu_uuid", 128)
    common.text(value["launch_token"], f"{field}.launch_token", 32)
    for key in ("observed_ns", "pgid", "pid", "start_ticks"):
        common.integer(value[key], f"{field}.{key}", 1)
    for key in (
        "matching_nvml_pids",
        "matching_process_groups",
        "matching_processes",
    ):
        common.exact(value[key], [], f"{field}.{key}")
    absent = value["absent"]
    common.require(type(absent) is list and bool(absent), f"E_ABSENT: {field}")
    pairs = []
    for index, row in enumerate(absent):
        row = common.exact_keys(
            row,
            {"pid", "start_ticks"},
            f"{field}.absent[{index}]",
        )
        pairs.append((
            common.integer(row["pid"], f"{field}.absent[{index}].pid", 1),
            common.integer(
                row["start_ticks"],
                f"{field}.absent[{index}].start_ticks",
                1,
            ),
        ))
    common.exact(pairs, sorted(set(pairs)), f"{field}.absent.order")
    return value


def _managed_public_plan(value: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: item
        for key, item in value.items()
        if key != "_normalized"
    }
    if type(result.get("ssh")) is dict and "_expected_boot_id" in result["ssh"]:
        result["ssh"] = {
            key: item
            for key, item in result["ssh"].items()
            if key != "_expected_boot_id"
        }
    return result


def _bind_plan_contents(
    wrapper_plan: dict[str, Any],
    wrapper_plan_raw: bytes,
    managed_plan: dict[str, Any],
    managed_plan_raw: bytes,
) -> tuple[str, str]:
    common.require(
        type(wrapper_plan_raw) is bytes
        and 0 < len(wrapper_plan_raw) <= common.MAX_JSON_BYTES,
        "E_WRAPPER_PLAN_CONTENT",
    )
    parsed_wrapper = common.parse_json(wrapper_plan_raw, "wrapper_plan")
    common.exact(
        common.canonical_bytes(parsed_wrapper),
        wrapper_plan_raw,
        "wrapper_plan.canonical",
    )
    common.exact(
        validate_plan(parsed_wrapper),
        wrapper_plan,
        "wrapper_plan.content",
    )
    common.require(
        type(managed_plan_raw) is bytes
        and 0 < len(managed_plan_raw) <= common.MAX_JSON_BYTES
        and not managed_plan_raw.endswith(b"\n"),
        "E_MANAGED_PLAN_CONTENT",
    )
    parsed_managed = common.parse_json(managed_plan_raw, "managed_plan")
    common.exact(
        common.canonical_compact(parsed_managed),
        managed_plan_raw,
        "managed_plan.canonical",
    )
    common.exact(
        parsed_managed,
        _managed_public_plan(managed_plan),
        "managed_plan.content",
    )
    wrapper_sha256 = hashlib.sha256(wrapper_plan_raw).hexdigest()
    managed_sha256 = hashlib.sha256(managed_plan_raw).hexdigest()
    common.exact(
        wrapper_plan["managed_plan_sha256"],
        managed_sha256,
        "wrapper_plan.managed_plan_sha256",
    )
    common.exact(
        wrapper_plan["managed_plan"]["bytes"],
        len(managed_plan_raw),
        "wrapper_plan.managed_plan.bytes",
    )
    return wrapper_sha256, managed_sha256


def _snapshot_rows(
    value: Any,
    remote_bundle: dict[str, Any],
    field: str,
) -> list[dict[str, Any]]:
    common.require(type(value) is list and bool(value), f"E_SNAPSHOTS: {field}")
    rows = []
    materialized = []
    sources = []
    file_map = {
        row["path"]: row
        for row in remote_bundle["files"]
    }
    for index, row in enumerate(value):
        row = common.exact_keys(
            row,
            {"artifact", "materialized_path"},
            f"{field}[{index}]",
        )
        artifact = _artifact(row["artifact"], f"{field}[{index}].artifact")
        path = common.relative_path(
            row["materialized_path"],
            f"{field}[{index}].materialized_path",
        )
        common.require(str(Path(path)) == path, f"E_PATH_CANONICAL: {field}")
        common.require(path in file_map, f"E_SNAPSHOT_FILE: {path}")
        common.exact(
            {
                "bytes": artifact["bytes"],
                "path": path,
                "sha256": artifact["sha256"],
            },
            file_map[path],
            f"{field}[{index}].file",
        )
        rows.append(row)
        materialized.append(path)
        sources.append(artifact["path"])
    common.exact(materialized, sorted(set(materialized)), f"{field}.order")
    common.require(len(sources) == len(set(sources)), f"E_SOURCE_PATH_REUSE: {field}")
    common.exact(
        materialized,
        sorted(file_map),
        f"{field}.coverage",
    )
    return rows


def validate_receipt(
    value: Any,
    wrapper_plan: dict[str, Any],
    wrapper_plan_raw: bytes,
    managed_plan: dict[str, Any],
    managed_plan_raw: bytes,
) -> dict[str, Any]:
    wrapper_sha256, managed_sha256 = _bind_plan_contents(
        wrapper_plan,
        wrapper_plan_raw,
        managed_plan,
        managed_plan_raw,
    )
    value = common.exact_keys(
        value,
        {
            "acquisition_artifact",
            "capture_input_artifacts",
            "completed_ns",
            "contract_validator",
            "controller_cleanup",
            "controller_clock",
            "execution_transport",
            "executor",
            "fetch_transport",
            "gpu_uuid",
            "local_bundle",
            "local_common",
            "managed_plan_artifact",
            "managed_plan_sha256",
            "managed_remote_cleanup",
            "managed_transport_process",
            "outer_phase_id",
            "phase",
            "remote_boot_id",
            "remote_bundle",
            "remote_cleanup",
            "remote_execution_interval",
            "remote_producer_process",
            "remote_snapshot_artifacts",
            "role",
            "runtime_identity_artifact",
            "schema",
            "started_ns",
            "system_swap_used_bytes",
            "v24_phase_id",
            "wrapper_plan_sha256",
        },
        "fan_in_receipt",
    )
    common.exact(value["schema"], RECEIPT_SCHEMA, "fan_in_receipt.schema")
    common.exact(value["phase"], PHASE, "fan_in_receipt.phase")
    common.exact(value["role"], ROLE, "fan_in_receipt.role")
    common.exact(
        value["controller_clock"],
        "CONTROLLER_MONOTONIC",
        "fan_in_receipt.controller_clock",
    )
    _phase_ids(
        value["outer_phase_id"],
        value["v24_phase_id"],
        "fan_in_receipt.phase_id",
    )
    common.exact(
        common.digest(
            value["wrapper_plan_sha256"],
            "fan_in_receipt.wrapper_plan_sha256",
        ),
        wrapper_sha256,
        "fan_in_receipt.wrapper_plan_sha256",
    )
    common.exact(
        common.digest(
            value["managed_plan_sha256"],
            "fan_in_receipt.managed_plan_sha256",
        ),
        managed_sha256,
        "fan_in_receipt.managed_plan_sha256",
    )
    for key in (
        "contract_validator",
        "executor",
        "local_common",
        "managed_plan_artifact",
    ):
        _artifact(value[key], f"fan_in_receipt.{key}")
        common.exact(
            value[key],
            wrapper_plan[
                "managed_plan" if key == "managed_plan_artifact" else key
            ],
            f"fan_in_receipt.plan.{key}",
        )
    common.uuid(value["remote_boot_id"], "fan_in_receipt.remote_boot_id")
    common.text(value["gpu_uuid"], "fan_in_receipt.gpu_uuid", 128)
    common.integer(
        value["system_swap_used_bytes"],
        "fan_in_receipt.system_swap_used_bytes",
    )
    started = common.integer(
        value["started_ns"],
        "fan_in_receipt.started_ns",
        1,
    )
    completed = common.integer(
        value["completed_ns"],
        "fan_in_receipt.completed_ns",
        started + 1,
    )
    common.exact(
        value["outer_phase_id"],
        wrapper_plan["outer_phase_id"],
        "fan_in_receipt.plan.outer_phase_id",
    )
    common.exact(
        value["v24_phase_id"],
        wrapper_plan["v24_phase_id"],
        "fan_in_receipt.plan.v24_phase_id",
    )
    capture_inputs = common.exact_keys(
        value["capture_input_artifacts"],
        {"cuda_monolithic", "joint_phone_cuda"},
        "fan_in_receipt.capture_input_artifacts",
    )
    for role in sorted(capture_inputs):
        artifact = _artifact(
            capture_inputs[role],
            f"fan_in_receipt.capture_input_artifacts.{role}",
        )
        common.exact(
            artifact["path"],
            wrapper_plan["capture_input_paths"][role],
            f"fan_in_receipt.capture_input_artifacts.{role}.path",
        )
    execution = _controller_transport(
        value["execution_transport"],
        "fan_in_receipt.execution_transport",
    )
    fetch = _controller_transport(
        value["fetch_transport"],
        "fan_in_receipt.fetch_transport",
    )
    common.exact(
        execution["controller_boot_id"],
        fetch["controller_boot_id"],
        "fan_in_receipt.controller_boot",
    )
    common.require(
        started <= execution["started_ns"]
        < execution["completed_ns"]
        <= fetch["started_ns"]
        < fetch["completed_ns"]
        <= completed,
        "E_TRANSPORT_ORDER",
    )
    transport = _managed_transport(
        value["managed_transport_process"],
        "fan_in_receipt.managed_transport_process",
    )
    producer = _remote_process(
        value["remote_producer_process"],
        "fan_in_receipt.remote_producer_process",
    )
    managed_cleanup = _managed_cleanup(
        value["managed_remote_cleanup"],
        "fan_in_receipt.managed_remote_cleanup",
    )
    common.exact(transport["host_boot_id"], execution["controller_boot_id"],
                 "fan_in_receipt.managed_transport.host_boot")
    common.exact(transport["remote_boot_id"], value["remote_boot_id"],
                 "fan_in_receipt.managed_transport.remote_boot")
    common.exact(transport["plan_sha256"], managed_sha256,
                 "fan_in_receipt.managed_transport.plan")
    common.exact(transport["bundle_id"], managed_plan["bundle_id"],
                 "fan_in_receipt.managed_transport.bundle")
    common.exact(
        transport["managed_launcher_pid"],
        execution["pid"],
        "fan_in_receipt.managed_transport.launcher_pid",
    )
    common.exact(
        transport["managed_launcher_start_ticks"],
        execution["start_ticks"],
        "fan_in_receipt.managed_transport.launcher_ticks",
    )
    common.require(
        execution["started_ns"]
        <= transport["observed_ns"]
        <= execution["completed_ns"],
        "E_MANAGED_TRANSPORT_INTERVAL",
    )
    common.exact(producer["boot_id"], value["remote_boot_id"],
                 "fan_in_receipt.remote_producer.boot")
    common.exact(producer["bundle_id"], managed_plan["bundle_id"],
                 "fan_in_receipt.remote_producer.bundle")
    common.exact(
        producer["launcher_path"],
        managed_plan["_normalized"]["launcher_path"],
        "fan_in_receipt.remote_producer.launcher",
    )
    common.exact(
        producer["loaded_repo_component_ids"],
        sorted(managed_plan["_normalized"]["component_map"]),
        "fan_in_receipt.remote_producer.components",
    )
    expected_dependencies = sorted(
        [
            {
                **component["stat"],
                "path": component["path"],
                "sha256": component["sha256"],
            }
            for component in managed_plan["components"]
        ],
        key=lambda item: item["path"],
    )
    common.exact(
        producer["system_dependencies"],
        expected_dependencies,
        "fan_in_receipt.remote_producer.dependencies",
    )
    common.require(
        execution["started_ns"]
        <= producer["controller_observed_ns"]
        <= execution["completed_ns"],
        "E_REMOTE_PROCESS_CONTROLLER_INTERVAL",
    )
    for key, expected in (
        ("boot_id", value["remote_boot_id"]),
        ("gpu_uuid", value["gpu_uuid"]),
        ("launch_token", producer["launch_token"]),
        ("pid", producer["pid"]),
        ("pgid", producer["pgid"]),
        ("start_ticks", producer["start_ticks"]),
    ):
        common.exact(
            managed_cleanup[key],
            expected,
            f"fan_in_receipt.managed_cleanup.{key}",
        )
    common.require(
        {
            "pid": producer["pid"],
            "start_ticks": producer["start_ticks"],
        }
        in managed_cleanup["absent"],
        "E_MANAGED_CLEANUP_PRODUCER",
    )
    interval = common.exact_keys(
        value["remote_execution_interval"],
        {"clock", "completed_ns", "phase_closed_ns", "started_ns"},
        "fan_in_receipt.remote_interval",
    )
    common.exact(
        interval["clock"],
        "RTX_CLOCK_MONOTONIC_RAW",
        "fan_in_receipt.remote_interval.clock",
    )
    common.exact(
        interval["started_ns"],
        producer["remote_observed_ns"],
        "fan_in_receipt.remote_interval.started_ns",
    )
    phase_closed = common.integer(
        interval["phase_closed_ns"],
        "fan_in_receipt.remote_interval.phase_closed_ns",
        interval["started_ns"],
    )
    common.exact(
        interval["completed_ns"],
        managed_cleanup["observed_ns"],
        "fan_in_receipt.remote_interval.completed_ns",
    )
    common.require(
        interval["started_ns"]
        <= phase_closed
        <= interval["completed_ns"],
        "E_REMOTE_INTERVAL",
    )
    cleanup = common.exact_keys(
        value["remote_cleanup"],
        {
            "boot_id",
            "clock",
            "gpu_uuid",
            "nvml_compute_pids",
            "observed_ns",
            "producer_absent",
            "schema",
        },
        "fan_in_receipt.remote_cleanup",
    )
    common.exact(cleanup["schema"], REMOTE_CLEANUP_SCHEMA,
                 "fan_in_receipt.remote_cleanup.schema")
    common.exact(cleanup["boot_id"], value["remote_boot_id"],
                 "fan_in_receipt.remote_cleanup.boot")
    common.exact(cleanup["gpu_uuid"], value["gpu_uuid"],
                 "fan_in_receipt.remote_cleanup.gpu")
    common.exact(cleanup["clock"], "RTX_CLOCK_MONOTONIC_RAW",
                 "fan_in_receipt.remote_cleanup.clock")
    common.integer(
        cleanup["observed_ns"],
        "fan_in_receipt.remote_cleanup.observed_ns",
        interval["completed_ns"],
    )
    common.exact(cleanup["nvml_compute_pids"], [],
                 "fan_in_receipt.remote_cleanup.nvml")
    common.exact(
        cleanup["producer_absent"],
        {"pid": producer["pid"], "start_ticks": producer["start_ticks"]},
        "fan_in_receipt.remote_cleanup.producer_absent",
    )
    controller_cleanup = common.exact_keys(
        value["controller_cleanup"],
        {
            "clock",
            "execution_transport_absent",
            "fetch_transport_absent",
            "local_forward_listener_absent",
            "managed_transport_absent",
            "observed_ns",
        },
        "fan_in_receipt.controller_cleanup",
    )
    common.exact(controller_cleanup["clock"], "CONTROLLER_MONOTONIC",
                 "fan_in_receipt.controller_cleanup.clock")
    common.exact(controller_cleanup["execution_transport_absent"], True,
                 "fan_in_receipt.controller_cleanup.execution")
    common.exact(controller_cleanup["fetch_transport_absent"], True,
                 "fan_in_receipt.controller_cleanup.fetch")
    common.exact(controller_cleanup["managed_transport_absent"], True,
                 "fan_in_receipt.controller_cleanup.managed")
    common.exact(controller_cleanup["local_forward_listener_absent"], True,
                 "fan_in_receipt.controller_cleanup.forward")
    common.integer(
        controller_cleanup["observed_ns"],
        "fan_in_receipt.controller_cleanup.observed_ns",
        fetch["completed_ns"],
    )
    common.require(
        controller_cleanup["observed_ns"] <= completed,
        "E_CONTROLLER_CLEANUP_INTERVAL",
    )
    remote = _bundle_manifest(value["remote_bundle"], "fan_in_receipt.remote")
    local = _bundle_manifest(value["local_bundle"], "fan_in_receipt.local")
    common.exact(
        remote["root_path"],
        wrapper_plan["remote_bundle_root"],
        "fan_in_receipt.remote.root_path",
    )
    common.exact(local["files"], remote["files"], "fan_in_receipt.bundle_files")
    common.exact(
        local["root_sha256"],
        remote["root_sha256"],
        "fan_in_receipt.bundle_root",
    )
    snapshots = _snapshot_rows(
        value["remote_snapshot_artifacts"],
        remote,
        "fan_in_receipt.remote_snapshot_artifacts",
    )
    source_map = {
        row["materialized_path"]: row["artifact"]["path"]
        for row in snapshots
    }
    for relative, source in source_map.items():
        if relative == RAW_MANIFEST_NAME:
            expected = str(
                Path(wrapper_plan["remote_bundle_root"])
                / V24_RAW_MANIFEST_NAME
            )
        elif relative == RUNTIME_IDENTITY_NAME:
            expected = wrapper_plan["remote_runtime_output"]
        elif relative == ACQUISITION_NAME:
            expected = wrapper_plan["remote_acquisition_output"]
        else:
            expected = str(Path(wrapper_plan["remote_bundle_root"]) / relative)
        common.exact(source, expected, f"fan_in_receipt.snapshot.{relative}")
    for key, expected_path in (
        ("runtime_identity_artifact", RUNTIME_IDENTITY_NAME),
        ("acquisition_artifact", ACQUISITION_NAME),
    ):
        row = _file_row(value[key], f"fan_in_receipt.{key}")
        common.exact(row["path"], expected_path, f"fan_in_receipt.{key}.path")
        common.require(row in local["files"], f"E_BUNDLE_ARTIFACT: {key}")
    raw_rows = [
        row for row in local["files"] if row["path"] == RAW_MANIFEST_NAME
    ]
    common.require(len(raw_rows) == 1, "E_RAW_MANIFEST_FILE")
    return value


def _read_local_file(path: Path, maximum: int = 512 * 1024 * 1024) -> bytes:
    return common.read_regular(path, maximum)


def validate_materialized_bundle(
    receipt: dict[str, Any],
    bundle_root: Path,
    wrapper_plan: dict[str, Any],
    wrapper_plan_raw: bytes,
    managed_plan: dict[str, Any],
    managed_plan_raw: bytes,
) -> dict[str, Any]:
    receipt = validate_receipt(
        receipt,
        wrapper_plan,
        wrapper_plan_raw,
        managed_plan,
        managed_plan_raw,
    )
    common.require(
        bundle_root.is_absolute()
        and str(bundle_root) == str(Path(str(bundle_root)))
        and bundle_root.is_dir()
        and not bundle_root.is_symlink(),
        "E_LOCAL_BUNDLE_ROOT",
    )
    common.exact(
        str(bundle_root),
        receipt["local_bundle"]["root_path"],
        "local_bundle.root_path",
    )
    observed = []
    for path in sorted(bundle_root.rglob("*")):
        common.require(not path.is_symlink(), f"E_BUNDLE_SYMLINK: {path}")
        if path.is_dir():
            continue
        raw = _read_local_file(path)
        observed.append({
            "bytes": len(raw),
            "path": str(path.relative_to(bundle_root)),
            "sha256": common.sha256_bytes(raw),
        })
    common.exact(
        observed,
        receipt["local_bundle"]["files"],
        "local_bundle.files",
    )
    file_map = {row["path"]: row for row in observed}
    for role, relative in (
        ("cuda_monolithic", "raw/cuda-monolithic.json"),
        ("joint_phone_cuda", "raw/joint-phone-cuda.json"),
    ):
        capture_raw = _read_local_file(bundle_root / relative)
        common.exact(
            len(capture_raw),
            receipt["capture_input_artifacts"][role]["bytes"],
            f"bundle.capture.{role}.bytes",
        )
        common.exact(
            common.sha256_bytes(capture_raw),
            receipt["capture_input_artifacts"][role]["sha256"],
            f"bundle.capture.{role}.sha256",
        )
    raw_manifest, raw = common.read_canonical(
        bundle_root / RAW_MANIFEST_NAME
    )
    common.exact(
        file_map[RAW_MANIFEST_NAME]["sha256"],
        common.sha256_bytes(raw),
        "local_bundle.raw_manifest",
    )
    common.exact(
        raw_manifest.get("schema"),
        "s39-cp0-r1-evidence-bundle-v2.1",
        "raw_manifest.schema",
    )
    common.exact(raw_manifest.get("phase"), PHASE, "raw_manifest.phase")
    common.exact(
        raw_manifest.get("phase_id"),
        receipt["v24_phase_id"],
        "raw_manifest.phase_id",
    )
    common.exact(
        raw_manifest.get("phase_closed_ns"),
        receipt["remote_execution_interval"]["phase_closed_ns"],
        "raw_manifest.phase_closed_ns",
    )
    artifacts = raw_manifest.get("artifacts")
    common.require(type(artifacts) is list and bool(artifacts), "E_RAW_ARTIFACTS")
    for index, row in enumerate(artifacts):
        field = f"raw_manifest.artifacts[{index}]"
        common.require(type(row) is dict, f"E_TYPE: {field}")
        path = common.relative_path(row.get("path"), f"{field}.path")
        common.require(path in file_map, f"E_RAW_ARTIFACT_MISSING: {path}")
        common.exact(row.get("bytes"), file_map[path]["bytes"], f"{field}.bytes")
        common.exact(
            row.get("sha256"),
            file_map[path]["sha256"],
            f"{field}.sha256",
        )
    runtime_path = bundle_root / RUNTIME_IDENTITY_NAME
    runtime, runtime_raw = common.read_canonical(runtime_path)
    common.exact(runtime.get("schema"), "s39-cp0-r1-runtime-identity-v2.4",
                 "runtime_identity.schema")
    common.exact(runtime.get("phase"), PHASE, "runtime_identity.phase")
    common.exact(runtime.get("phase_id"), receipt["v24_phase_id"],
                 "runtime_identity.phase_id")
    acquisition_path = bundle_root / ACQUISITION_NAME
    acquisition, acquisition_raw = common.read_canonical(acquisition_path)
    common.exact(
        acquisition.get("schema"),
        "s39-cp0-r1-a-only-acquisition-v2.4",
        "acquisition.schema",
    )
    common.exact(acquisition.get("phase"), PHASE, "acquisition.phase")
    common.exact(acquisition.get("phase_id"), receipt["v24_phase_id"],
                 "acquisition.phase_id")
    common.exact(
        acquisition.get("status"),
        "RAW_CAPTURE_COMPLETE_UNEVALUATED",
        "acquisition.status",
    )
    common.exact(
        acquisition.get("raw_manifest_name"),
        V24_RAW_MANIFEST_NAME,
        "acquisition.raw_manifest_name",
    )
    common.exact(
        acquisition.get("raw_manifest_sha256"),
        common.sha256_bytes(raw),
        "acquisition.raw_manifest_sha256",
    )
    common.exact(
        acquisition.get("runtime_identity_sha256"),
        common.sha256_bytes(runtime_raw),
        "acquisition.runtime_identity_sha256",
    )
    common.exact(
        acquisition.get("completed_ns"),
        raw_manifest["phase_closed_ns"],
        "acquisition.completed_ns",
    )
    acquisition_artifacts = acquisition.get("artifacts")
    common.require(
        type(acquisition_artifacts) is list and bool(acquisition_artifacts),
        "E_ACQUISITION_ARTIFACTS",
    )
    for index, row in enumerate(acquisition_artifacts):
        field = f"acquisition.artifacts[{index}]"
        common.require(type(row) is dict, f"E_TYPE: {field}")
        path = common.relative_path(row.get("path"), f"{field}.path")
        common.require(path in file_map, f"E_ACQUISITION_FILE: {path}")
        common.exact(row.get("bytes"), file_map[path]["bytes"], f"{field}.bytes")
        common.exact(row.get("sha256"), file_map[path]["sha256"],
                     f"{field}.sha256")
    by_role = {
        row.get("role"): row
        for row in acquisition_artifacts
        if type(row) is dict
    }
    for role, relative in (
        ("cuda_monolithic", "raw/cuda-monolithic.json"),
        ("joint_phone_cuda", "raw/joint-phone-cuda.json"),
    ):
        acquisition_role = f"capture.{role}"
        common.require(
            acquisition_role in by_role,
            f"E_ACQUISITION_CAPTURE: {acquisition_role}",
        )
        common.exact(
            {
                "bytes": by_role[acquisition_role].get("bytes"),
                "sha256": by_role[acquisition_role].get("sha256"),
            },
            {
                "bytes": receipt["capture_input_artifacts"][role]["bytes"],
                "sha256": receipt["capture_input_artifacts"][role]["sha256"],
            },
            f"acquisition.capture.{role}",
        )
        common.exact(
            by_role[acquisition_role].get("path"),
            relative,
            f"acquisition.capture.{role}.path",
        )
    common.exact(
        receipt["runtime_identity_artifact"],
        file_map[RUNTIME_IDENTITY_NAME],
        "runtime_identity.artifact",
    )
    common.exact(
        receipt["acquisition_artifact"],
        file_map[ACQUISITION_NAME],
        "acquisition.artifact",
    )
    return raw_manifest
