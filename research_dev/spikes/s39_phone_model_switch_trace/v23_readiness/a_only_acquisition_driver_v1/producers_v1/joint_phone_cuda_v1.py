#!/usr/bin/python3 -I
"""Capture one concurrent phone-route and CUDA-route A_ONLY execution."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time


PHASE = "A_ONLY"
MODEL_ID = "qwen3-14b-q4_k_m"
PLAN_SCHEMA = "s39-cp0-r1-a-only-joint-capture-plan-v1"
OUTPUT_SCHEMA = "s39-cp0-r1-a-only-joint-phone-cuda-raw-v1"
PHONE_SCHEMA = "s39-cp0-r1-a-only-phone-route-capture-v1"
CUDA_SCHEMA = "s39-cp0-r1-a-only-cuda-route-capture-v1"
CLOCK_ID = time.CLOCK_MONOTONIC_RAW
DIGEST_LENGTH = 64
PLACEHOLDERS = {
    "{acquisition_started_ns}",
    "{command_plan_sha256}",
    "{output_path}",
    "{phase_id}",
    "{pre_dir}",
}
COMMAND_KEYS = {
    "argv_template",
    "executed_files",
    "result_filename",
    "timeout_seconds",
}
EXECUTED_FILE_KEYS = {"argv_index", "bytes", "path", "sha256"}
CAPTURED_FILE_FLAGS = {
    "--histories",
    "--launch-plan",
}
PLAN_KEYS = {
    "commands",
    "mechanism_commands",
    "model_id",
    "model_sha256",
    "phase",
    "schema",
}
PHONE_KEYS = {
    "bridge_publication_rows",
    "completed_ns",
    "mechanics_rows",
    "mechanism_commands_sha256",
    "model_id",
    "model_sha256",
    "op12_runtime",
    "op15_runtime",
    "phase_id",
    "placement_op12_rows",
    "placement_op15_rows",
    "quality_phone_rows",
    "route_epoch",
    "route_transfer_rows",
    "runtime_processes",
    "schema",
    "started_ns",
}
CUDA_KEYS = {
    "bridge_ready_row",
    "bridge_start_row",
    "completed_ns",
    "cuda_memory_rows",
    "cuda_route_rows",
    "gpu_runtime",
    "mechanism_commands_sha256",
    "model_id",
    "model_sha256",
    "phase_id",
    "quality_cuda_rows",
    "route_epoch",
    "runtime_process",
    "schema",
    "started_ns",
}
RUNTIME_PROCESS_KEYS = {
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
PHONE_RUNTIME_BUNDLES = {
    "op12_stagenet": "op12",
    "op15_direct_relay": "op15",
    "op15_stagenet": "op15",
}
CUDA_RUNTIME_BUNDLES = {"cuda_route": "cuda"}


class CaptureError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CaptureError(message)


def exact(value, expected, field):
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}",
    )


def exact_keys(value, expected, field):
    require(type(value) is dict, f"E_TYPE: {field}")
    exact(set(value), expected, f"{field}.keys")
    return value


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise CaptureError(f"E_JSON_NUMBER: {value}")


def canonical_bytes(value):
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
        raise CaptureError("E_CANONICAL") from error


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value, field):
    require(
        type(value) is str
        and len(value) == DIGEST_LENGTH
        and all(character in "0123456789abcdef" for character in value),
        f"E_DIGEST: {field}",
    )
    return value


def integer(value, field, minimum=0):
    require(type(value) is int and value >= minimum, f"E_INTEGER: {field}")
    return value


def string(value, field):
    require(type(value) is str and bool(value), f"E_STRING: {field}")
    return value


def read_regular(path, field):
    require(path.is_absolute(), f"E_PATH: {field}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CaptureError(f"E_OPEN: {field}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"E_FILE_TYPE: {field}")
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
    require(identity(before) == identity(after), f"E_FILE_CHANGED: {field}")
    require(len(raw) == before.st_size, f"E_FILE_SIZE: {field}")
    return bytes(raw), before


def read_canonical(path, field):
    raw, _ = read_regular(path, field)
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"E_JSON: {field}") from error
    require(type(value) is dict, f"E_TYPE: {field}")
    exact(canonical_bytes(value), raw, f"{field}.canonical")
    return value, raw


def write_new(path, raw, mode=0o644):
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


def write_json(path, value):
    write_new(path, canonical_bytes(value))


def validate_executed_files(value, template, field):
    require(type(value) is list and bool(value), f"E_TYPE: {field}")
    indexes = set()
    records = {}
    for offset, record in enumerate(value):
        item = f"{field}[{offset}]"
        exact_keys(record, EXECUTED_FILE_KEYS, item)
        index = integer(record["argv_index"], f"{item}.argv_index")
        require(index < len(template), f"E_RANGE: {item}.argv_index")
        require(index not in indexes, f"E_INDEX_REUSE: {item}")
        indexes.add(index)
        path = string(record["path"], f"{item}.path")
        require(Path(path).is_absolute(), f"E_PATH: {item}.path")
        exact(template[index], path, f"{item}.argv")
        integer(record["bytes"], f"{item}.bytes", 1)
        digest(record["sha256"], f"{item}.sha256")
        records[index] = record
    require(0 in indexes, f"E_ENTRYPOINT: {field}")
    return records


def verify_self_contained_source(path, field):
    try:
        source = read_regular(path, field)[0].decode("ascii")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as error:
        raise CaptureError(f"E_PRODUCER_SOURCE: {field}") from error
    for node in ast.walk(tree):
        names = []
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


def validate_command(value, field):
    exact_keys(value, COMMAND_KEYS, field)
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
    exact(used, PLACEHOLDERS, f"{field}.placeholders")
    require(
        all(item in PLACEHOLDERS or "{" not in item for item in template),
        f"E_PLACEHOLDER: {field}",
    )
    records = validate_executed_files(
        value["executed_files"],
        template,
        f"{field}.executed_files",
    )
    required_indexes = {0}
    for index, item in enumerate(template):
        if item not in CAPTURED_FILE_FLAGS:
            path = Path(item)
            if not path.is_absolute():
                continue
            try:
                metadata = os.lstat(path)
            except FileNotFoundError:
                continue
            except OSError as error:
                raise CaptureError(f"E_EXECUTED_STAT: {field}[{index}]") from error
            require(
                not stat.S_ISLNK(metadata.st_mode),
                f"E_EXECUTED_SYMLINK: {field}[{index}]",
            )
            if stat.S_ISREG(metadata.st_mode):
                required_indexes.add(index)
            continue
        require(index + 1 < len(template), f"E_EXECUTED_ARGUMENT: {field}[{index}]")
        file_index = index + 1
        require(
            template[file_index] not in PLACEHOLDERS,
            f"E_EXECUTED_PLACEHOLDER: {field}[{file_index}]",
        )
        required_indexes.add(file_index)
    require(
        required_indexes.issubset(records),
        f"E_EXECUTED_FILE_MISSING: {field}",
    )
    verify_self_contained_source(Path(template[0]), f"{field}.entrypoint")
    result = Path(string(value["result_filename"], f"{field}.result_filename"))
    require(
        not result.is_absolute()
        and result.parts
        and ".." not in result.parts,
        f"E_PATH: {field}.result_filename",
    )
    timeout = integer(value["timeout_seconds"], f"{field}.timeout", 1)
    require(timeout <= 7200, f"E_RANGE: {field}.timeout")
    return records


def validate_mechanism_commands(value):
    exact_keys(value, {"desktop", "op12", "op15"}, "mechanism_commands")
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
            inline = [offset for offset, item in enumerate(argv) if item == "--plan-json"]
            require(
                len(inline) <= 1,
                f"E_COMMAND_INLINE_PLAN_COUNT: {endpoint}[{index}]",
            )
            inline_value = None
            if inline:
                inline_value = inline[0] + 1
                require(
                    inline_value < len(argv),
                    f"E_COMMAND_INLINE_PLAN: {endpoint}[{index}]",
                )
                raw = argv[inline_value]
                try:
                    plan = json.loads(
                        raw,
                        object_pairs_hook=strict_object,
                        parse_constant=reject_constant,
                    )
                except json.JSONDecodeError as error:
                    raise CaptureError(
                        f"E_COMMAND_INLINE_PLAN_JSON: {endpoint}[{index}]"
                    ) from error
                require(
                    type(plan) is dict,
                    f"E_COMMAND_INLINE_PLAN_TYPE: {endpoint}[{index}]",
                )
                exact(
                    canonical_bytes(plan)[:-1].decode("ascii"),
                    raw,
                    f"mechanism_commands.{endpoint}[{index}].inline_plan",
                )
                exact(
                    argv.count("--plan-sha256"),
                    1,
                    f"mechanism_commands.{endpoint}[{index}].plan_sha.count",
                )
                digest_index = argv.index("--plan-sha256") + 1
                require(
                    digest_index < len(argv),
                    f"E_COMMAND_INLINE_PLAN_DIGEST: {endpoint}[{index}]",
                )
                exact(
                    argv[digest_index],
                    sha256_bytes(raw.encode("ascii")),
                    f"mechanism_commands.{endpoint}[{index}].plan_sha",
                )
            else:
                exact(
                    argv.count("--plan-sha256"),
                    0,
                    f"mechanism_commands.{endpoint}[{index}].plan_sha.count",
                )
            for item_index, item in enumerate(argv):
                if item_index == inline_value:
                    continue
                require(
                    "{" not in item and "}" not in item,
                    f"E_COMMAND_PLACEHOLDER: {endpoint}[{index}]",
                )


def load_plan(path):
    value, raw = read_canonical(path, "capture_plan")
    exact_keys(value, PLAN_KEYS, "capture_plan")
    exact(value["schema"], PLAN_SCHEMA, "capture_plan.schema")
    exact(value["phase"], PHASE, "capture_plan.phase")
    exact(value["model_id"], MODEL_ID, "capture_plan.model_id")
    digest(value["model_sha256"], "capture_plan.model_sha256")
    validate_mechanism_commands(value["mechanism_commands"])
    commands = exact_keys(value["commands"], {"cuda", "phone"}, "commands")
    for name in ("phone", "cuda"):
        validate_command(commands[name], f"commands.{name}")
    require(
        commands["phone"]["result_filename"]
        != commands["cuda"]["result_filename"],
        "E_RESULT_PATH_REUSE",
    )
    return value, raw


def capture_commands(plan, evidence_dir):
    captured = {}
    for name in ("phone", "cuda"):
        command = plan["commands"][name]
        records = validate_executed_files(
            command["executed_files"],
            command["argv_template"],
            f"commands.{name}.executed_files",
        )
        for index in sorted(records):
            record = records[index]
            source = Path(record["path"])
            raw, metadata = read_regular(
                source,
                f"commands.{name}.executed_files[{index}]",
            )
            exact(len(raw), record["bytes"], f"E_EXECUTED_BYTES: {name}[{index}]")
            exact(
                sha256_bytes(raw),
                record["sha256"],
                f"E_EXECUTED_SHA256: {name}[{index}]",
            )
            destination = evidence_dir / name / f"{index:03d}-{source.name}"
            write_new(destination, raw, stat.S_IMODE(metadata.st_mode))
            captured[(name, index)] = str(destination)
    return captured


def command_argv(
    plan,
    name,
    output_path,
    phase_id,
    pre_dir,
    acquisition_started_ns,
    command_plan_sha256,
    captured,
):
    replacements = {
        "{acquisition_started_ns}": str(acquisition_started_ns),
        "{command_plan_sha256}": command_plan_sha256,
        "{output_path}": str(output_path),
        "{phase_id}": phase_id,
        "{pre_dir}": str(pre_dir),
    }
    return [
        captured.get((name, index), replacements.get(item, item))
        for index, item in enumerate(plan["commands"][name]["argv_template"])
    ]


def kill_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_pair(plan, output_path, phase_id, pre_dir, acquisition_started_ns,
             command_plan_sha256):
    root = output_path.parent / f".{output_path.name}.evidence"
    root.mkdir(parents=True, exist_ok=False)
    captured = capture_commands(plan, root / "executed")
    processes = {}
    intervals = {}
    result_paths = {}
    try:
        for name in ("phone", "cuda"):
            command = plan["commands"][name]
            result_path = root / command["result_filename"]
            argv = command_argv(
                plan,
                name,
                result_path,
                phase_id,
                pre_dir,
                acquisition_started_ns,
                command_plan_sha256,
                captured,
            )
            started_ns = time.clock_gettime_ns(CLOCK_ID)
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            processes[name] = process
            intervals[name] = [started_ns, 0]
            result_paths[name] = result_path
        results = {}
        for name in ("phone", "cuda"):
            process = processes[name]
            command = plan["commands"][name]
            try:
                stdout, stderr = process.communicate(
                    timeout=command["timeout_seconds"]
                )
            except subprocess.TimeoutExpired as error:
                kill_process(process)
                raise CaptureError(f"E_COMMAND_TIMEOUT: {name}") from error
            intervals[name][1] = time.clock_gettime_ns(CLOCK_ID)
            write_new(root / f"{name}.stdout", stdout)
            write_new(root / f"{name}.stderr", stderr)
            write_json(
                root / f"{name}.receipt.json",
                {
                    "argv": command_argv(
                        plan,
                        name,
                        result_paths[name],
                        phase_id,
                        pre_dir,
                        acquisition_started_ns,
                        command_plan_sha256,
                        captured,
                    ),
                    "completed_ns": intervals[name][1],
                    "returncode": process.returncode,
                    "schema": "s39-cp0-r1-a-only-subproducer-receipt-v1",
                    "started_ns": intervals[name][0],
                },
            )
            exact(process.returncode, 0, f"E_COMMAND_EXIT: {name}")
            exact(stdout, b"", f"E_COMMAND_STDOUT: {name}")
            exact(stderr, b"", f"E_COMMAND_STDERR: {name}")
            value, _ = read_canonical(result_paths[name], f"{name}.result")
            results[name] = value
        return results, intervals
    except BaseException:
        for process in processes.values():
            kill_process(process)
        raise


def validate_event_rows(rows, field, count=None):
    require(type(rows) is list and bool(rows), f"E_ROWS: {field}")
    if count is not None:
        exact(len(rows), count, f"{field}.count")
    previous = None
    for index, row in enumerate(rows):
        require(type(row) is dict, f"E_TYPE: {field}[{index}]")
        require(
            not {"acquisition_id", "phase", "phase_id", "role"}.intersection(row),
            f"E_WRAPPER_FIELDS: {field}[{index}]",
        )
        event_ns = integer(row.get("event_ns"), f"{field}[{index}].event_ns", 1)
        if previous is not None:
            require(previous <= event_ns, f"E_EVENT_ORDER: {field}")
        previous = event_ns


def validate_runtime_process(value, expected_bundles, started, completed, field):
    exact_keys(value, RUNTIME_PROCESS_KEYS, field)
    bundle_id = string(value["bundle_id"], f"{field}.bundle_id")
    require(bundle_id in expected_bundles, f"E_RUNTIME_BUNDLE: {field}")
    exact(value["endpoint"], expected_bundles[bundle_id], f"{field}.endpoint")
    string(value["boot_id"], f"{field}.boot_id")
    integer(value["pid"], f"{field}.pid", 1)
    integer(value["start_ticks"], f"{field}.start_ticks", 1)
    observed_ns = integer(value["observed_ns"], f"{field}.observed_ns", 1)
    require(
        started <= observed_ns <= completed,
        f"E_RUNTIME_PROCESS_INTERVAL: {field}",
    )
    launcher_path = string(value["launcher_path"], f"{field}.launcher_path")
    require(Path(launcher_path).is_absolute(), f"E_PATH: {field}.launcher_path")
    component_ids = value["loaded_repo_component_ids"]
    require(
        type(component_ids) is list
        and bool(component_ids)
        and all(type(item) is str and bool(item) for item in component_ids),
        f"E_RUNTIME_COMPONENTS: {field}",
    )
    exact(
        component_ids,
        sorted(set(component_ids)),
        f"{field}.loaded_repo_component_ids",
    )
    dependencies = value["system_dependencies"]
    require(
        type(dependencies) is list and bool(dependencies),
        f"E_SYSTEM_DEPENDENCIES: {field}",
    )
    previous_path = None
    for index, dependency in enumerate(dependencies):
        dependency_field = f"{field}.system_dependencies[{index}]"
        exact_keys(dependency, SYSTEM_DEPENDENCY_KEYS, dependency_field)
        path = string(dependency["path"], f"{dependency_field}.path")
        require(Path(path).is_absolute(), f"E_PATH: {dependency_field}.path")
        if previous_path is not None:
            require(previous_path < path, f"E_SYSTEM_DEPENDENCY_ORDER: {field}")
        previous_path = path
        for key in ("ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"):
            integer(dependency[key], f"{dependency_field}.{key}")
        build_id = dependency["build_id"]
        require(
            build_id is None or (type(build_id) is str and bool(build_id)),
            f"E_SYSTEM_BUILD_ID: {dependency_field}",
        )
    return bundle_id


def validate_runtime_processes(
    value,
    expected_bundles,
    started,
    completed,
    field,
):
    require(
        type(value) is list and len(value) == len(expected_bundles),
        f"E_RUNTIME_PROCESSES: {field}",
    )
    bundle_ids = []
    for index, process in enumerate(value):
        bundle_ids.append(
            validate_runtime_process(
                process,
                expected_bundles,
                started,
                completed,
                f"{field}[{index}]",
            )
        )
    exact(bundle_ids, sorted(expected_bundles), f"{field}.order")
    return value


def validate_fragment_common(
    value,
    expected_keys,
    schema,
    phase_id,
    model_sha256,
    mechanism_sha256,
    receipt_interval,
    field,
):
    exact_keys(value, expected_keys, field)
    exact(value["schema"], schema, f"{field}.schema")
    exact(value["phase_id"], phase_id, f"{field}.phase_id")
    exact(value["model_id"], MODEL_ID, f"{field}.model_id")
    exact(value["model_sha256"], model_sha256, f"{field}.model_sha256")
    exact(
        value["mechanism_commands_sha256"],
        mechanism_sha256,
        f"{field}.mechanism_commands_sha256",
    )
    started = integer(value["started_ns"], f"{field}.started_ns", 1)
    completed = integer(value["completed_ns"], f"{field}.completed_ns", 1)
    require(
        receipt_interval[0] <= started < completed <= receipt_interval[1],
        f"E_INTERVAL: {field}",
    )
    return started, completed


def validate_phone(
    value,
    phase_id,
    model_sha256,
    mechanism_sha256,
    receipt_interval,
):
    started, completed = validate_fragment_common(
        value,
        PHONE_KEYS,
        PHONE_SCHEMA,
        phase_id,
        model_sha256,
        mechanism_sha256,
        receipt_interval,
        "phone",
    )
    row_fields = (
        ("mechanics_rows", 9),
        ("quality_phone_rows", 64),
        ("placement_op15_rows", None),
        ("placement_op12_rows", None),
        ("route_transfer_rows", None),
        ("bridge_publication_rows", 8),
    )
    for field, count in row_fields:
        validate_event_rows(value[field], f"phone.{field}", count)
        for row in value[field]:
            require(
                started <= row["event_ns"] <= completed,
                f"E_EVENT_INTERVAL: phone.{field}",
            )
    exact(value["mechanics_rows"][0].get("kind"), "meta", "phone.mechanics.meta")
    exact(
        value["placement_op15_rows"][0].get("kind"),
        "meta",
        "phone.placement_op15.meta",
    )
    exact(
        value["placement_op12_rows"][0].get("kind"),
        "meta",
        "phone.placement_op12.meta",
    )
    exact(
        value["route_transfer_rows"][0].get("kind"),
        "meta",
        "phone.route_transfer.meta",
    )
    require(
        all(
            row.get("kind") == "phone_publication_received"
            for row in value["bridge_publication_rows"]
        ),
        "E_BRIDGE_PUBLICATION_KIND",
    )
    integer(value["route_epoch"], "phone.route_epoch", 1)
    require(type(value["op15_runtime"]) is dict, "E_TYPE: phone.op15_runtime")
    require(type(value["op12_runtime"]) is dict, "E_TYPE: phone.op12_runtime")
    validate_runtime_processes(
        value["runtime_processes"],
        PHONE_RUNTIME_BUNDLES,
        started,
        completed,
        "phone.runtime_processes",
    )
    return started, completed


def validate_cuda(
    value,
    phase_id,
    model_sha256,
    mechanism_sha256,
    receipt_interval,
):
    started, completed = validate_fragment_common(
        value,
        CUDA_KEYS,
        CUDA_SCHEMA,
        phase_id,
        model_sha256,
        mechanism_sha256,
        receipt_interval,
        "cuda",
    )
    for field, count in (
        ("cuda_route_rows", 9),
        ("cuda_memory_rows", 3),
        ("quality_cuda_rows", 64),
    ):
        validate_event_rows(value[field], f"cuda.{field}", count)
        for row in value[field]:
            require(
                started <= row["event_ns"] <= completed,
                f"E_EVENT_INTERVAL: cuda.{field}",
            )
    for field, kind in (
        ("bridge_start_row", "cuda_load_start"),
        ("bridge_ready_row", "cuda_ready"),
    ):
        validate_event_rows([value[field]], f"cuda.{field}", 1)
        exact(value[field].get("kind"), kind, f"cuda.{field}.kind")
        require(
            started <= value[field]["event_ns"] <= completed,
            f"E_EVENT_INTERVAL: cuda.{field}",
        )
    require(
        value["bridge_start_row"]["event_ns"]
        < value["bridge_ready_row"]["event_ns"],
        "E_CUDA_LOAD_INTERVAL",
    )
    exact(value["cuda_route_rows"][0].get("kind"), "meta", "cuda.route.meta")
    exact(value["cuda_memory_rows"][0].get("kind"), "before", "cuda.memory.before")
    exact(value["cuda_memory_rows"][1].get("kind"), "ready", "cuda.memory.ready")
    exact(value["cuda_memory_rows"][2].get("kind"), "after", "cuda.memory.after")
    integer(value["route_epoch"], "cuda.route_epoch", 1)
    require(type(value["gpu_runtime"]) is dict, "E_TYPE: cuda.gpu_runtime")
    validate_runtime_process(
        value["runtime_process"],
        CUDA_RUNTIME_BUNDLES,
        started,
        completed,
        "cuda.runtime_process",
    )
    return started, completed


def build_result(
    plan,
    results,
    intervals,
    phase_id,
    acquisition_started_ns,
):
    mechanism_sha256 = sha256_bytes(
        canonical_bytes(plan["mechanism_commands"])
    )
    phone = results["phone"]
    cuda = results["cuda"]
    phone_interval = validate_phone(
        phone,
        phase_id,
        plan["model_sha256"],
        mechanism_sha256,
        intervals["phone"],
    )
    cuda_interval = validate_cuda(
        cuda,
        phase_id,
        plan["model_sha256"],
        mechanism_sha256,
        intervals["cuda"],
    )
    exact(phone["route_epoch"], cuda["route_epoch"], "route_epoch")
    start = cuda["bridge_start_row"]
    ready = cuda["bridge_ready_row"]
    publications = phone["bridge_publication_rows"]
    bridge_rows = [start, *publications, ready]
    previous = None
    for index, row in enumerate(bridge_rows):
        event_ns = row["event_ns"]
        if previous is not None:
            require(previous <= event_ns, f"E_BRIDGE_ORDER: {index}")
        previous = event_ns
    require(
        all(row["event_ns"] < ready["event_ns"] for row in publications),
        "E_PUBLICATION_AFTER_CUDA_READY",
    )
    started_ns = min(phone_interval[0], cuda_interval[0])
    completed_ns = max(phone_interval[1], cuda_interval[1])
    require(acquisition_started_ns <= started_ns, "E_ACQUISITION_INTERVAL")
    return {
        "bridge_rows": bridge_rows,
        "completed_ns": completed_ns,
        "cuda_memory_rows": cuda["cuda_memory_rows"],
        "cuda_route_rows": cuda["cuda_route_rows"],
        "gpu_runtime": cuda["gpu_runtime"],
        "mechanics_rows": phone["mechanics_rows"],
        "mechanism_commands_sha256": mechanism_sha256,
        "model_id": MODEL_ID,
        "model_sha256": plan["model_sha256"],
        "op12_runtime": phone["op12_runtime"],
        "op15_runtime": phone["op15_runtime"],
        "phase_id": phase_id,
        "placement_op12_rows": phone["placement_op12_rows"],
        "placement_op15_rows": phone["placement_op15_rows"],
        "quality_cuda_rows": cuda["quality_cuda_rows"],
        "quality_phone_rows": phone["quality_phone_rows"],
        "route_epoch": phone["route_epoch"],
        "route_transfer_rows": phone["route_transfer_rows"],
        "runtime_processes": sorted(
            [*phone["runtime_processes"], cuda["runtime_process"]],
            key=lambda value: value["bundle_id"],
        ),
        "schema": OUTPUT_SCHEMA,
        "started_ns": started_ns,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--pre-dir", type=Path, required=True)
    parser.add_argument("--acquisition-started-ns", type=int, required=True)
    parser.add_argument("--command-plan-sha256", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        require(args.capture_plan.is_absolute(), "E_PATH: capture_plan")
        require(args.output.is_absolute(), "E_PATH: output")
        require(args.pre_dir.is_absolute(), "E_PATH: pre_dir")
        require(not args.output.exists(), "E_EXISTS: output")
        string(args.phase_id, "phase_id")
        integer(args.acquisition_started_ns, "acquisition_started_ns", 1)
        digest(args.command_plan_sha256, "command_plan_sha256")
        plan, _ = load_plan(args.capture_plan)
        results, intervals = run_pair(
            plan,
            args.output,
            args.phase_id,
            args.pre_dir,
            args.acquisition_started_ns,
            args.command_plan_sha256,
        )
        value = build_result(
            plan,
            results,
            intervals,
            args.phase_id,
            args.acquisition_started_ns,
        )
        write_json(args.output, value)
        return 0
    except (
        CaptureError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"A_ONLY_JOINT_CAPTURE_REFUSED: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
