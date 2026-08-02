#!/usr/bin/python3 -I
"""Capture one concurrent phone-route and CUDA-route A_ONLY execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time


PHASE = "A_ONLY"
MODEL_ID = "qwen3-14b-q4_k_m"
PLAN_SCHEMA = "s39-cp0-r1-v24-joint-capture-plan-v1"
OUTPUT_SCHEMA = "s39-cp0-r1-v24-joint-phone-cuda-raw-v1"
PHONE_SCHEMA = "s39-cp0-r1-v24-phone-route-raw-v1"
CUDA_SCHEMA = "s39-cp0-r1-v24-cuda-route-raw-v1"
HISTORY_SCHEMA = "s39-cp0-r1-token-history-v2.4"
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
    "cwd",
    "environment",
    "executed_files",
    "launch_plan_argv_index",
    "launch_plan_sha256",
    "producer_sha256",
    "result_filename",
    "timeout_seconds",
}
EXECUTED_FILE_KEYS = {"argv_index", "bytes", "path", "sha256"}
PLAN_KEYS = {
    "commands",
    "history",
    "mechanism_commands",
    "model_id",
    "model_sha256",
    "phase",
    "schema",
}
HISTORY_KEYS = {"bytes", "path", "sha256"}
PHONE_KEYS = {
    "bridge_publication_rows",
    "completed_ns",
    "direct_certificate",
    "direct_frames",
    "evidence_artifacts",
    "execution_groups",
    "history_sha256",
    "launch_plan_sha256",
    "mechanics_rows",
    "mechanism_commands_sha256",
    "model_id",
    "model_sha256",
    "op12_runtime",
    "op15_runtime",
    "phase_id",
    "placement_certificates",
    "placement_op12_rows",
    "placement_op15_rows",
    "producer_sha256",
    "quality_phone_rows",
    "raw_probes",
    "route_epoch",
    "route_transfer_rows",
    "runtime_processes",
    "schema",
    "session_certificates",
    "started_ns",
}
CUDA_KEYS = {
    "bridge_ready_row",
    "bridge_start_row",
    "completed_ns",
    "cuda_memory_rows",
    "cuda_route_rows",
    "evidence_artifacts",
    "execution_groups",
    "gpu_runtime",
    "history_sha256",
    "launch_plan_sha256",
    "mechanism_commands_sha256",
    "model_id",
    "model_sha256",
    "memory_certificate",
    "phase_id",
    "placement_certificate",
    "producer_sha256",
    "protocol_identity",
    "quality_cuda_rows",
    "raw_memory_samples",
    "route_epoch",
    "runtime_process",
    "runtime_model_binding",
    "schema",
    "started_ns",
}
CUDA_PROTOCOL_IDENTITY_KEYS = {
    "capabilities",
    "file_type",
    "layer_end",
    "layer_start",
    "max_streams",
    "model_sha256",
    "n_batch",
    "n_ctx_seq",
    "n_embd",
    "n_layer",
    "n_ubatch",
    "schema",
    "stage_identity_version",
    "stage_protocol_version",
}
CUDA_MEMORY_CERTIFICATE_KEYS = {
    "compute_buffer_bytes",
    "host_compute_buffer_bytes",
    "host_context_buffer_bytes",
    "host_model_buffer_bytes",
    "kv_buffer_bytes",
    "model_buffer_bytes",
    "pid",
    "role",
    "schema",
}
RUNTIME_PROCESS_KEYS = {
    "boot_id",
    "bundle_id",
    "endpoint",
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


def parse_canonical(raw, field):
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
    return value


def read_canonical(path, field):
    raw, _ = read_regular(path, field)
    value = parse_canonical(raw, field)
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
    cwd = Path(string(value["cwd"], f"{field}.cwd"))
    require(cwd.is_absolute() and cwd.is_dir(), f"E_CWD: {field}")
    environment = value["environment"]
    require(type(environment) is dict and bool(environment), f"E_ENV: {field}")
    for key, item in environment.items():
        require(
            type(key) is str
            and type(item) is str
            and bool(key)
            and key.isascii()
            and item.isascii()
            and "=" not in key
            and "\x00" not in item,
            f"E_ENV: {field}.{key}",
        )
    producer_sha256 = digest(value["producer_sha256"], f"{field}.producer_sha256")
    exact(
        records[0]["sha256"],
        producer_sha256,
        f"{field}.producer_binding",
    )
    launch_index = integer(
        value["launch_plan_argv_index"],
        f"{field}.launch_plan_argv_index",
    )
    require(launch_index in records, f"E_LAUNCH_PLAN_INDEX: {field}")
    exact(
        records[launch_index]["sha256"],
        digest(value["launch_plan_sha256"], f"{field}.launch_plan_sha256"),
        f"{field}.launch_plan_binding",
    )
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
            require(
                not any("{" in item or "}" in item for item in argv),
                f"E_COMMAND_PLACEHOLDER: {endpoint}[{index}]",
            )


def load_plan(path, expected_sha256=None):
    raw, _ = read_regular(path, "capture_plan")
    if expected_sha256 is not None:
        expected_sha256 = digest(
            expected_sha256,
            "capture_plan_sha256",
        )
        exact(
            sha256_bytes(raw),
            expected_sha256,
            "E_CAPTURE_PLAN_SHA256",
        )
    value = parse_canonical(raw, "capture_plan")
    exact_keys(value, PLAN_KEYS, "capture_plan")
    exact(value["schema"], PLAN_SCHEMA, "capture_plan.schema")
    exact(value["phase"], PHASE, "capture_plan.phase")
    exact(value["model_id"], MODEL_ID, "capture_plan.model_id")
    digest(value["model_sha256"], "capture_plan.model_sha256")
    validate_mechanism_commands(value["mechanism_commands"])
    history = exact_keys(value["history"], HISTORY_KEYS, "history")
    history_path = Path(string(history["path"], "history.path"))
    require(history_path.is_absolute(), "E_HISTORY_PATH")
    history_raw, _ = read_regular(history_path, "history")
    exact(len(history_raw), integer(history["bytes"], "history.bytes", 1), "history.bytes")
    exact(
        sha256_bytes(history_raw),
        digest(history["sha256"], "history.sha256"),
        "history.sha256",
    )
    commands = exact_keys(value["commands"], {"cuda", "phone"}, "commands")
    for name in ("phone", "cuda"):
        validate_command(commands[name], f"commands.{name}")
    require(
        commands["phone"]["result_filename"]
        != commands["cuda"]["result_filename"],
        "E_RESULT_PATH_REUSE",
    )
    return value, raw


def load_history(plan):
    path = Path(plan["history"]["path"])
    value, raw = read_canonical(path, "history")
    exact(sha256_bytes(raw), plan["history"]["sha256"], "history.sha256")
    exact(value.get("schema"), HISTORY_SCHEMA, "history.schema")
    exact(value.get("model_id"), MODEL_ID, "history.model_id")
    exact(value.get("model_sha256"), plan["model_sha256"], "history.model_sha256")
    exact(value.get("batch"), 8, "history.batch")
    exact(value.get("continuation_tokens_per_request"), 8, "history.continuation")
    exact(value.get("n_ctx_seq"), 512, "history.n_ctx_seq")
    exact(value.get("n_batch"), 64, "history.n_batch")
    exact(value.get("n_ubatch"), 64, "history.n_ubatch")
    requests = value.get("requests")
    require(type(requests) is list and len(requests) == 64, "E_HISTORY_REQUESTS")
    for item_index, request in enumerate(requests):
        require(type(request) is dict, f"E_HISTORY_REQUEST: {item_index}")
        exact(request.get("item_index"), item_index, f"history.request[{item_index}].item")
        exact(request.get("request_id"), item_index % 8 + 1,
              f"history.request[{item_index}].request_id")
        exact(request.get("seq_id"), item_index % 8,
              f"history.request[{item_index}].seq_id")
        tokens = request.get("token_ids")
        require(
            type(tokens) is list
            and 0 < len(tokens) <= 504
            and all(type(token) is int and token >= 0 for token in tokens),
            f"E_HISTORY_TOKENS: {item_index}",
        )
    groups = value.get("quality_groups")
    require(type(groups) is list and len(groups) == 8, "E_HISTORY_GROUPS")
    exact(value.get("mechanics_b8"), groups[0], "history.mechanics_b8")
    for group_index, group in enumerate(groups):
        require(type(group) is dict, f"E_HISTORY_GROUP: {group_index}")
        exact(group.get("group_index"), group_index,
              f"history.group[{group_index}].index")
        exact(group.get("item_indices"),
              list(range(group_index * 8, group_index * 8 + 8)),
              f"history.group[{group_index}].items")
        prefill = group.get("prefill_partitions")
        decode = group.get("decode_calls")
        require(type(prefill) is list and bool(prefill), "E_HISTORY_PREFILL")
        require(type(decode) is list and len(decode) == 7, "E_HISTORY_DECODE")
    return value, raw


def expected_call_shapes(group):
    return [
        {
            "call_index": partition["call_index"],
            "n_seqs": len({row["seq_id"] for row in partition["rows"]}),
            "n_tokens": len(partition["rows"]),
            "phase": "prefill",
            "positions": [row["position"] for row in partition["rows"]],
            "request_ids": [row["request_id"] for row in partition["rows"]],
            "seq_ids": [row["seq_id"] for row in partition["rows"]],
        }
        for partition in group["prefill_partitions"]
    ] + [
        {
            "call_index": call["call_index"],
            "n_seqs": len({row["seq_id"] for row in call["rows"]}),
            "n_tokens": len(call["rows"]),
            "phase": "decode",
            "positions": [row["position"] for row in call["rows"]],
            "request_ids": [row["request_id"] for row in call["rows"]],
            "seq_ids": [row["seq_id"] for row in call["rows"]],
        }
        for call in group["decode_calls"]
    ]


def validate_execution_groups(values, history, field):
    require(type(values) is list and len(values) == 8, f"E_GROUPS: {field}")
    requests = {row["item_index"]: row for row in history["requests"]}
    all_continuations = {}
    next_frame = 0
    for group_index, (value, plan) in enumerate(
        zip(values, history["quality_groups"])
    ):
        item = f"{field}[{group_index}]"
        exact_keys(
            value,
            {
                "call_receipts",
                "continuations",
                "group_index",
                "item_indices",
                "wire_request_ids",
            },
            item,
        )
        exact(value["group_index"], group_index, f"{item}.group_index")
        exact(value["item_indices"], plan["item_indices"], f"{item}.item_indices")
        wires = value["wire_request_ids"]
        require(
            type(wires) is list
            and len(wires) == 8
            and len(set(wires)) == 8
            and all(type(wire) is int and wire > 0 for wire in wires),
            f"E_WIRE_IDS: {item}",
        )
        expected_calls = [
            ("prefill", call)
            for call in plan["prefill_partitions"]
        ] + [
            ("decode", call)
            for call in plan["decode_calls"]
        ]
        receipts = value["call_receipts"]
        require(
            type(receipts) is list and len(receipts) == len(expected_calls),
            f"E_CALL_RECEIPTS: {item}",
        )
        current = {}
        continuations = [[] for _ in range(8)]
        for call_offset, (receipt, expected) in enumerate(
            zip(receipts, expected_calls)
        ):
            phase, call = expected
            call_field = f"{item}.calls[{call_offset}]"
            exact_keys(
                receipt,
                {"call_index", "frame_call_index", "phase", "rows"},
                call_field,
            )
            exact(receipt["call_index"], call["call_index"],
                  f"{call_field}.call_index")
            exact(receipt["frame_call_index"], next_frame,
                  f"{call_field}.frame_call_index")
            next_frame += 1
            exact(receipt["phase"], phase, f"{call_field}.phase")
            rows = receipt["rows"]
            require(type(rows) is list and len(rows) == len(call["rows"]),
                    f"E_CALL_ROWS: {call_field}")
            next_tokens = {}
            for row_index, (row, expected_row) in enumerate(
                zip(rows, call["rows"])
            ):
                row_field = f"{call_field}.rows[{row_index}]"
                exact_keys(
                    row,
                    {
                        "input_token",
                        "item_index",
                        "output_token",
                        "position",
                        "request_id",
                        "route_epoch",
                        "seq_id",
                        "wire_request_id",
                    },
                    row_field,
                )
                for key in ("item_index", "position", "request_id", "seq_id"):
                    exact(row[key], expected_row[key], f"{row_field}.{key}")
                seq_id = row["seq_id"]
                exact(row["wire_request_id"], wires[seq_id],
                      f"{row_field}.wire_request_id")
                integer(row["route_epoch"], f"{row_field}.route_epoch", 1)
                output = integer(row["output_token"], f"{row_field}.output")
                if phase == "prefill":
                    exact(row["input_token"], expected_row["token_id"],
                          f"{row_field}.input")
                    request = requests[row["item_index"]]
                    if row["position"] == len(request["token_ids"]) - 1:
                        require(seq_id not in current,
                                f"E_FINAL_PREFILL_REUSE: {row_field}")
                        current[seq_id] = output
                else:
                    exact(row["input_token"], current[seq_id],
                          f"{row_field}.decode_input")
                    next_tokens[seq_id] = output
            if phase == "decode":
                exact(sorted(next_tokens), list(range(8)),
                      f"{call_field}.decode_outputs")
                current = next_tokens
                for seq_id in range(8):
                    continuations[seq_id].append(current[seq_id])
        exact(sorted(current), list(range(8)), f"{item}.final_state")
        first_outputs = {}
        for receipt in receipts:
            if receipt["phase"] != "prefill":
                continue
            for row in receipt["rows"]:
                request = requests[row["item_index"]]
                if row["position"] == len(request["token_ids"]) - 1:
                    first_outputs[row["seq_id"]] = row["output_token"]
        exact(sorted(first_outputs), list(range(8)), f"{item}.prefill_outputs")
        for seq_id in range(8):
            continuations[seq_id].insert(0, first_outputs[seq_id])
        exact(value["continuations"], continuations, f"{item}.continuations")
        require(all(len(tokens) == 8 for tokens in continuations),
                f"E_CONTINUATION_LENGTH: {item}")
        for seq_id, item_index in enumerate(plan["item_indices"]):
            all_continuations[item_index] = continuations[seq_id]
    exact(sorted(all_continuations), list(range(64)),
          f"{field}.continuation_items")
    return all_continuations


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
                cwd=command["cwd"],
                env=command["environment"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            processes[name] = process
            intervals[name] = [started_ns, 0]
            result_paths[name] = result_path
        results = {}
        result_digests = {}
        result_artifacts = {}
        receipts = {}
        receipt_artifacts = {}
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
            receipt_path = root / f"{name}.receipt.json"
            receipt = {
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
                "cwd": command["cwd"],
                "environment": command["environment"],
                "launch_plan_sha256": command["launch_plan_sha256"],
                "producer_sha256": command["producer_sha256"],
                "returncode": process.returncode,
                "schema": "s39-cp0-r1-v24-subproducer-receipt-v1",
                "started_ns": intervals[name][0],
            }
            write_json(receipt_path, receipt)
            exact(process.returncode, 0, f"E_COMMAND_EXIT: {name}")
            exact(stdout, b"", f"E_COMMAND_STDOUT: {name}")
            exact(stderr, b"", f"E_COMMAND_STDERR: {name}")
            value, raw = read_canonical(result_paths[name], f"{name}.result")
            receipt_value, receipt_raw = read_canonical(
                receipt_path,
                f"{name}.receipt",
            )
            exact(receipt_value, receipt, f"E_RECEIPT_REOPEN: {name}")
            results[name] = value
            result_digests[name] = sha256_bytes(raw)
            result_artifacts[name] = {
                "bytes": len(raw),
                "path": str(result_paths[name]),
                "sha256": result_digests[name],
            }
            receipts[name] = receipt_value
            receipt_artifacts[name] = {
                "bytes": len(receipt_raw),
                "path": str(receipt_path),
                "sha256": sha256_bytes(receipt_raw),
            }
        return (
            results,
            intervals,
            result_digests,
            result_artifacts,
            receipts,
            receipt_artifacts,
        )
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


def validate_mechanics_rows(rows, history, continuations, field, backend):
    validate_event_rows(rows, field, 9)
    meta = rows[0]
    exact(meta.get("kind"), "meta", f"{field}.meta.kind")
    exact(meta.get("backend"), backend, f"{field}.meta.backend")
    exact(
        meta.get("call_shapes"),
        expected_call_shapes(history["mechanics_b8"]),
        f"{field}.meta.call_shapes",
    )
    request_rows = {}
    for index, row in enumerate(rows[1:]):
        exact(row.get("kind"), "request", f"{field}[{index + 1}].kind")
        request_id = integer(row.get("request_id"), f"{field}.request_id", 1)
        require(request_id not in request_rows, f"E_REQUEST_REUSE: {field}")
        request_rows[request_id] = row
    exact(set(request_rows), set(range(1, 9)), f"{field}.request_ids")
    for seq_id, item_index in enumerate(history["mechanics_b8"]["item_indices"]):
        request_id = seq_id + 1
        row = request_rows[request_id]
        source = history["requests"][item_index]
        exact(row.get("input_tokens"), source["token_ids"],
              f"{field}[{request_id}].input_tokens")
        exact(row.get("positions"), list(range(len(source["token_ids"]))),
              f"{field}[{request_id}].positions")
        exact(row.get("continuation_tokens"), continuations[item_index],
              f"{field}[{request_id}].continuation")
    return request_rows


def validate_quality_rows(rows, history, field):
    validate_event_rows(rows, field, 64)
    seen = set()
    for index, row in enumerate(rows):
        item = f"{field}[{index}]"
        exact(row.get("kind"), "output", f"{item}.kind")
        item_index = integer(row.get("item_index"), f"{item}.item_index")
        require(item_index not in seen and item_index < 64,
                f"E_QUALITY_ITEM: {item}")
        seen.add(item_index)
        exact(row.get("prompt_sha256"),
              history["requests"][item_index]["prompt_sha256"],
              f"{item}.prompt_sha256")
        require(type(row.get("raw_output")) is str, f"E_QUALITY_OUTPUT: {item}")
    exact(seen, set(range(64)), f"{field}.items")


def validate_evidence_artifacts(values, field):
    require(type(values) is list and bool(values), f"E_ARTIFACTS: {field}")
    seen = set()
    artifacts = {}
    for index, value in enumerate(values):
        item = f"{field}[{index}]"
        exact_keys(value, {"bytes", "path", "sha256"}, item)
        path = Path(string(value["path"], f"{item}.path"))
        require(path.is_absolute() and str(path) not in seen, f"E_ARTIFACT_PATH: {item}")
        seen.add(str(path))
        raw, _ = read_regular(path, item)
        exact(len(raw), integer(value["bytes"], f"{item}.bytes", 1),
              f"{item}.bytes")
        exact(sha256_bytes(raw), digest(value["sha256"], f"{item}.sha256"),
              f"{item}.sha256")
        artifacts[str(path)] = raw
    return artifacts


def validate_cuda_memory_sources(artifacts, rows, field):
    by_name = {}
    for path, raw in artifacts.items():
        name = Path(path).name
        require(name not in by_name, f"E_CUDA_ARTIFACT_NAME_REUSE: {name}")
        by_name[name] = raw
    for index, kind in enumerate(("before", "ready", "after")):
        item = f"{field}.{kind}"
        names = {
            "device": f"{kind}.device.stdout",
            "process": f"{kind}.process.stdout",
            "swap": f"{kind}.meminfo",
            "sample": f"{kind}.sample.json",
        }
        require(
            all(name in by_name for name in names.values()),
            f"E_CUDA_RAW_SOURCE: {item}",
        )
        device_lines = by_name[names["device"]].decode("ascii").splitlines()
        exact(len(device_lines), 1, f"{item}.device.lines")
        device = [part.strip() for part in device_lines[0].split(",")]
        exact(len(device), 4, f"{item}.device.fields")
        exact(device[0], "NVIDIA GeForce RTX 4060 Ti", f"{item}.device.name")
        exact(
            device[1],
            "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
            f"{item}.device.uuid",
        )
        require(device[2].isdigit() and device[3].isdigit(),
                f"E_CUDA_RAW_DEVICE_INTEGER: {item}")
        total = int(device[2]) * 1024 * 1024
        used = int(device[3]) * 1024 * 1024
        row = rows[index]
        exact(row["memory_total_bytes"], total, f"{item}.total")
        exact(row["used_bytes"], used, f"{item}.used")
        exact(row["free_bytes"], total - used, f"{item}.free")

        process_lines = [
            line
            for line in by_name[names["process"]].decode("ascii").splitlines()
            if line.strip()
        ]
        if kind == "ready":
            exact(len(process_lines), 1, f"{item}.process.lines")
            process = [part.strip() for part in process_lines[0].split(",")]
            exact(len(process), 2, f"{item}.process.fields")
            require(process[0].isdigit() and process[1].isdigit(),
                    f"E_CUDA_RAW_PROCESS_INTEGER: {item}")
            exact(row["process_pid"], int(process[0]), f"{item}.process.pid")
            exact(
                row["process_used_bytes"],
                int(process[1]) * 1024 * 1024,
                f"{item}.process.used",
            )
        else:
            exact(process_lines, [], f"{item}.process.lines")

        swap = {}
        for line in by_name[names["swap"]].decode("ascii").splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[0] in {"SwapTotal:", "SwapFree:"}:
                require(parts[1].isdigit() and parts[2] == "kB",
                        f"E_CUDA_RAW_SWAP: {item}")
                swap[parts[0]] = int(parts[1]) * 1024
        exact(set(swap), {"SwapFree:", "SwapTotal:"}, f"{item}.swap.fields")
        exact(
            row["host_swap_used_bytes"],
            swap["SwapTotal:"] - swap["SwapFree:"],
            f"{item}.swap.used",
        )
        sample = json.loads(
            by_name[names["sample"]].decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
        exact(
            canonical_bytes(sample),
            by_name[names["sample"]],
            f"{item}.sample.canonical",
        )
        exact(sample["kind"], kind, f"{item}.sample.kind")
        exact(
            sample["device_stdout_sha256"],
            sha256_bytes(by_name[names["device"]]),
            f"{item}.sample.device",
        )
        exact(
            sample["process_stdout_sha256"],
            sha256_bytes(by_name[names["process"]]),
            f"{item}.sample.process",
        )
        exact(
            sample["swap_raw_sha256"],
            sha256_bytes(by_name[names["swap"]]),
            f"{item}.sample.swap",
        )
        exact(row["sample_id"], sha256_bytes(canonical_bytes(sample)),
              f"{item}.sample.id")
        exact(row["timestamp_ns"], sample["sample_completed_ns"],
              f"{item}.sample.timestamp")


def validate_phone_rich(value, history, field):
    continuations = validate_execution_groups(
        value["execution_groups"],
        history,
        f"{field}.execution_groups",
    )
    request_rows = validate_mechanics_rows(
        value["mechanics_rows"],
        history,
        continuations,
        f"{field}.mechanics_rows",
        "PHONE_COLLECTIVE",
    )
    validate_quality_rows(value["quality_phone_rows"], history,
                          f"{field}.quality_phone_rows")
    publications = value["bridge_publication_rows"]
    validate_event_rows(publications, f"{field}.bridge_publication_rows", 8)
    publication_ids = []
    for index, row in enumerate(publications):
        item = f"{field}.bridge_publication_rows[{index}]"
        exact(row.get("kind"), "phone_publication_received", f"{item}.kind")
        request_id = integer(row.get("request_id"), f"{item}.request_id", 1)
        publication_ids.append(request_id)
        exact(row.get("token_ids"), request_rows[request_id]["continuation_tokens"],
              f"{item}.tokens")
        normalized_request = {
            "acquisition_id": value["phase_id"],
            **{
                key: item_value
                for key, item_value in request_rows[request_id].items()
                if key != "event_ns"
            },
            "role": f"model.{MODEL_ID}.mechanics.phone",
        }
        exact(
            row.get("phone_request_sha256"),
            sha256_bytes(canonical_bytes(normalized_request)),
            f"{item}.mechanics_link",
        )
        require(
            request_rows[request_id]["event_ns"] <= row["event_ns"],
            f"E_PUBLICATION_BEFORE_MECHANICS: {item}",
        )
    exact(publication_ids, list(range(1, 9)), f"{field}.publication_ids")

    frames = value["direct_frames"]
    receipts = [
        call
        for group in value["execution_groups"]
        for call in group["call_receipts"]
    ]
    require(type(frames) is list and len(frames) == len(receipts),
            f"E_DIRECT_FRAMES: {field}")
    payload_total = 0
    for index, (frame, call) in enumerate(zip(frames, receipts)):
        item = f"{field}.direct_frames[{index}]"
        exact(frame.get("call_index"), index, f"{item}.call_index")
        exact(frame.get("rows"), len(call["rows"]), f"{item}.rows")
        exact(frame.get("positions"), [row["position"] for row in call["rows"]],
              f"{item}.positions")
        exact(frame.get("request_ids"),
              [row["wire_request_id"] for row in call["rows"]],
              f"{item}.request_ids")
        exact(frame.get("seq_ids"), [row["seq_id"] for row in call["rows"]],
              f"{item}.seq_ids")
        exact(frame.get("route_epochs"),
              [row["route_epoch"] for row in call["rows"]],
              f"{item}.route_epochs")
        payload = len(call["rows"]) * 5120 * 4
        exact(frame.get("activation_payload_bytes"), payload, f"{item}.payload")
        digest(frame.get("payload_sha256"), f"{item}.payload_sha256")
        payload_total += payload
    certificate = value["direct_certificate"]
    exact(certificate.get("activation_payload_bytes"), payload_total,
          f"{field}.direct_certificate.payload")
    exact(certificate.get("host_activation_payload_bytes"), 0,
          f"{field}.direct_certificate.host_payload")
    exact(certificate.get("cut_layer"), 30, f"{field}.direct_certificate.cut")
    exact(certificate.get("status"), "DIRECT_RELAY_OK",
          f"{field}.direct_certificate.status")

    probes = value["raw_probes"]
    exact_keys(probes, {"op12", "op15"}, f"{field}.raw_probes")
    for phone in ("op12", "op15"):
        probe = exact_keys(
            probes[phone],
            {"after", "after_ns", "before", "before_ns"},
            f"{field}.raw_probes.{phone}",
        )
        before = probe["before"]
        after = probe["after"]
        exact(before.get("process_swap_bytes"), 0,
              f"{field}.{phone}.before.process_swap")
        exact(after.get("process_swap_bytes"), 0,
              f"{field}.{phone}.after.process_swap")
        exact(after.get("system_swap_used_bytes"),
              before.get("system_swap_used_bytes"),
              f"{field}.{phone}.system_swap_growth")
        exact(before.get("thermal_status"), 0,
              f"{field}.{phone}.before.thermal")
        exact(after.get("thermal_status"), 0,
              f"{field}.{phone}.after.thermal")
        require(
            min(before.get("available_bytes", 0), after.get("available_bytes", 0))
            >= 512 * 1024 * 1024,
            f"E_PHONE_HEADROOM: {field}.{phone}",
        )
    op15_delta = (
        probes["op15"]["after"]["interface"]["tx_bytes"]
        - probes["op15"]["before"]["interface"]["tx_bytes"]
    )
    op12_delta = (
        probes["op12"]["after"]["interface"]["rx_bytes"]
        - probes["op12"]["before"]["interface"]["rx_bytes"]
    )
    require(op15_delta >= payload_total, f"E_OP15_LINK_COUNTER: {field}")
    require(op12_delta >= payload_total, f"E_OP12_LINK_COUNTER: {field}")

    expected_placement = {
        "op15": ([0, 32], [0, 30],
                 "ba56b9c5e19b3a4512777e6a47803cc"
                 "03261c2d3c2734965cd5ec96b7c6c59fb"),
        "op12": ([24, 40], [30, 40],
                 "72e312af745160dc33a0ba39ba94fbbc"
                 "e6112950d0409d39c42ddc3b25e756ab"),
    }
    for phone, (stored, executed, shard_sha256) in expected_placement.items():
        rows = value[f"placement_{phone}_rows"]
        validate_event_rows(rows, f"{field}.placement.{phone}")
        meta = rows[0]
        exact(meta.get("kind"), "meta", f"{field}.placement.{phone}.kind")
        exact(meta.get("stored_layers"), stored,
              f"{field}.placement.{phone}.stored")
        exact(meta.get("executed_layers"), executed,
              f"{field}.placement.{phone}.executed")
        exact(meta.get("shard_sha256"), shard_sha256,
              f"{field}.placement.{phone}.shard")
        exact(meta.get("process_swap_bytes"), 0,
              f"{field}.placement.{phone}.process_swap")
        exact(meta.get("system_swap_after_bytes"),
              meta.get("system_swap_before_bytes"),
              f"{field}.placement.{phone}.system_swap")
        require(
            min(meta.get("available_before_bytes", 0),
                meta.get("available_after_bytes", 0))
            >= 512 * 1024 * 1024,
            f"E_PHONE_HEADROOM: {field}.placement.{phone}",
        )
        for row_index, row in enumerate(rows[1:]):
            backend = row.get("backend")
            op = row.get("op")
            require(
                backend == "GPUOpenCL"
                or (phone == "op15" and backend == "CPU" and op == "GET_ROWS"),
                f"E_PHONE_FALLBACK: {field}.{phone}[{row_index + 1}]",
            )
            exact(row.get("missing_buffer"), False,
                  f"{field}.{phone}[{row_index + 1}].missing")
    for phone in ("op12", "op15"):
        runtime = value[f"{phone}_runtime"]
        exact(runtime.get("active_sequences_after_cleanup"), 0,
              f"{field}.{phone}.cleanup")
        exact(runtime.get("process_swap_bytes"), 0,
              f"{field}.{phone}.runtime_swap")
    validate_evidence_artifacts(
        value["evidence_artifacts"],
        f"{field}.evidence_artifacts",
    )
    return continuations


def validate_cuda_rich(value, history, field):
    continuations = validate_execution_groups(
        value["execution_groups"],
        history,
        f"{field}.execution_groups",
    )
    validate_mechanics_rows(
        value["cuda_route_rows"],
        history,
        continuations,
        f"{field}.cuda_route_rows",
        "CUDA0",
    )
    validate_quality_rows(value["quality_cuda_rows"], history,
                          f"{field}.quality_cuda_rows")
    samples = value["raw_memory_samples"]
    rows = value["cuda_memory_rows"]
    require(type(samples) is list and len(samples) == 3,
            f"E_MEMORY_SAMPLES: {field}")
    require(type(rows) is list and len(rows) == 3, f"E_MEMORY_ROWS: {field}")
    protocol = exact_keys(
        value.get("protocol_identity"),
        CUDA_PROTOCOL_IDENTITY_KEYS,
        f"{field}.protocol_identity",
    )
    exact(
        protocol["schema"],
        "layersplit-stage-v3-identity-v1",
        f"{field}.protocol_identity.schema",
    )
    exact(protocol["stage_protocol_version"], 3,
          f"{field}.protocol_identity.stage_protocol_version")
    exact(protocol["stage_identity_version"], 1,
          f"{field}.protocol_identity.stage_identity_version")
    exact(protocol["capabilities"], 0x3F,
          f"{field}.protocol_identity.capabilities")
    exact(protocol["layer_start"], 0,
          f"{field}.protocol_identity.layer_start")
    exact(protocol["layer_end"], 40,
          f"{field}.protocol_identity.layer_end")
    exact(protocol["n_layer"], 40, f"{field}.protocol_identity.n_layer")
    exact(protocol["n_embd"], 5120, f"{field}.protocol_identity.n_embd")
    exact(protocol["max_streams"], 8,
          f"{field}.protocol_identity.max_streams")
    exact(protocol["n_ctx_seq"], 512,
          f"{field}.protocol_identity.n_ctx_seq")
    exact(protocol["n_batch"], 64, f"{field}.protocol_identity.n_batch")
    exact(protocol["n_ubatch"], 64, f"{field}.protocol_identity.n_ubatch")
    exact(protocol["model_sha256"], value["model_sha256"],
          f"{field}.protocol_identity.model_sha256")
    exact(protocol["file_type"], 15, f"{field}.protocol_identity.file_type")

    memory_certificate = exact_keys(
        value.get("memory_certificate"),
        CUDA_MEMORY_CERTIFICATE_KEYS,
        f"{field}.memory_certificate",
    )
    exact(
        memory_certificate["schema"],
        "layersplit-memory-breakdown-v1",
        f"{field}.memory_certificate.schema",
    )
    exact(memory_certificate["role"], "monov3",
          f"{field}.memory_certificate.role")
    for key in CUDA_MEMORY_CERTIFICATE_KEYS - {"pid", "role", "schema"}:
        integer(
            memory_certificate[key],
            f"{field}.memory_certificate.{key}",
        )
    require(
        memory_certificate["model_buffer_bytes"] > 0
        and memory_certificate["kv_buffer_bytes"] > 0,
        f"E_MEMORY_CERTIFICATE_DEVICE_BYTES: {field}",
    )
    for index, (sample, row) in enumerate(zip(samples, rows)):
        item = f"{field}.raw_memory_samples[{index}]"
        exact_keys(sample, {"bytes", "path", "row", "sha256"}, item)
        path = Path(string(sample["path"], f"{item}.path"))
        require(path.is_absolute(), f"E_MEMORY_SAMPLE_PATH: {item}")
        raw, _ = read_regular(path, item)
        exact(len(raw), integer(sample["bytes"], f"{item}.bytes", 1),
              f"{item}.bytes")
        exact(sha256_bytes(raw), digest(sample["sha256"], f"{item}.sha256"),
              f"{item}.sha256")
        parsed, parsed_raw = read_canonical(path, item)
        del parsed_raw
        exact(parsed, sample["row"], f"{item}.parsed")
        exact(sample["row"], row, f"{item}.projection")
        exact(row.get("kind"), ("before", "ready", "after")[index],
              f"{item}.kind")
        exact(row.get("clock_id"), "HOST_MONOTONIC_RAW", f"{item}.clock")
        exact(row.get("device_name"), "NVIDIA GeForce RTX 4060 Ti",
              f"{item}.device_name")
        exact(row.get("device_uuid"),
              "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
              f"{item}.device_uuid")
        exact(row.get("memory_total_bytes"), 17175674880,
              f"{item}.memory_total")
        used = integer(row.get("used_bytes"), f"{item}.used")
        free = integer(row.get("free_bytes"), f"{item}.free")
        exact(used + free, 17175674880, f"{item}.accounting")
        integer(row.get("host_swap_used_bytes"), f"{item}.host_swap")
        if index == 1:
            integer(row.get("model_buffer_bytes"), f"{item}.model_buffer", 1)
            integer(row.get("kv_buffer_bytes"), f"{item}.kv_buffer", 1)
            integer(row.get("placement_compute_nodes"), f"{item}.placement", 1)
            integer(row.get("process_pid"), f"{item}.process_pid", 1)
            process_used = integer(
                row.get("process_used_bytes"),
                f"{item}.process_used",
                1,
            )
            require(
                row["model_buffer_bytes"] + row["kv_buffer_bytes"]
                <= process_used <= used,
                f"E_MEMORY_PROCESS_ACCOUNTING: {item}",
            )
            exact(
                row["model_buffer_bytes"],
                memory_certificate["model_buffer_bytes"],
                f"{item}.memory_certificate.model_buffer_bytes",
            )
            exact(
                row["kv_buffer_bytes"],
                memory_certificate["kv_buffer_bytes"],
                f"{item}.memory_certificate.kv_buffer_bytes",
            )
            require(free >= 512 * 1024 * 1024, f"E_CUDA_HEADROOM: {item}")
        else:
            for key in (
                "kv_buffer_bytes",
                "model_buffer_bytes",
                "placement_compute_nodes",
                "process_pid",
                "process_used_bytes",
            ):
                exact(row.get(key), 0, f"{item}.{key}")
    exact(
        rows[2]["host_swap_used_bytes"],
        rows[0]["host_swap_used_bytes"],
        f"{field}.host_swap_growth",
    )
    placement = value["placement_certificate"]
    exact(placement.get("status"), "SCHEDULED_PLACEMENT_OK",
          f"{field}.placement.status")
    exact(placement.get("layer_start"), 0, f"{field}.placement.layer_start")
    exact(placement.get("layer_end"), 40, f"{field}.placement.layer_end")
    exact(placement.get("missing_buffer_compute_nodes"), 0,
          f"{field}.placement.missing")
    buffers = placement.get("compute_by_buffer_type")
    require(
        type(buffers) is dict
        and type(buffers.get("CUDA0")) is int
        and buffers["CUDA0"] > 0
        and set(buffers) <= {"CUDA0", "CUDA_Host"},
        f"E_CUDA_PLACEMENT: {field}",
    )
    runtime = value["runtime_process"]
    exact(runtime.get("bundle_id"), "cuda_route", f"{field}.runtime.bundle")
    exact(runtime.get("endpoint"), "cuda", f"{field}.runtime.endpoint")
    integer(runtime.get("pid"), f"{field}.runtime.pid", 1)
    integer(runtime.get("start_ticks"), f"{field}.runtime.start_ticks", 1)
    exact(memory_certificate["pid"], runtime["pid"],
          f"{field}.memory_certificate.pid")
    bridge_start = value["bridge_start_row"]
    bridge_ready = value["bridge_ready_row"]
    exact(bridge_start.get("kind"), "cuda_load_start",
          f"{field}.bridge_start.kind")
    exact(bridge_ready.get("kind"), "cuda_ready",
          f"{field}.bridge_ready.kind")
    require(
        bridge_start.get("event_ns") < bridge_ready.get("event_ns"),
        f"E_CUDA_READY_ORDER: {field}",
    )
    artifacts = validate_evidence_artifacts(
        value["evidence_artifacts"],
        f"{field}.evidence_artifacts",
    )
    validate_cuda_memory_sources(artifacts, rows, f"{field}.raw_memory")
    return continuations


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
    plan,
    history,
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
    exact(value["history_sha256"], plan["history"]["sha256"],
          "phone.history_sha256")
    exact(value["producer_sha256"],
          plan["commands"]["phone"]["producer_sha256"],
          "phone.producer_sha256")
    exact(value["launch_plan_sha256"],
          plan["commands"]["phone"]["launch_plan_sha256"],
          "phone.launch_plan_sha256")
    continuations = validate_phone_rich(value, history, "phone")
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
    return started, completed, continuations


def validate_cuda(
    value,
    plan,
    history,
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
    exact(value["history_sha256"], plan["history"]["sha256"],
          "cuda.history_sha256")
    exact(value["producer_sha256"],
          plan["commands"]["cuda"]["producer_sha256"],
          "cuda.producer_sha256")
    exact(value["launch_plan_sha256"],
          plan["commands"]["cuda"]["launch_plan_sha256"],
          "cuda.launch_plan_sha256")
    continuations = validate_cuda_rich(value, history, "cuda")
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
    return started, completed, continuations


def build_result(
    plan,
    history,
    results,
    intervals,
    result_digests,
    result_artifacts,
    receipts,
    receipt_artifacts,
    phase_id,
    acquisition_started_ns,
):
    mechanism_sha256 = sha256_bytes(
        canonical_bytes(plan["mechanism_commands"])
    )
    phone = results["phone"]
    cuda = results["cuda"]
    phone_start, phone_completed, phone_continuations = validate_phone(
        phone,
        plan,
        history,
        phase_id,
        plan["model_sha256"],
        mechanism_sha256,
        intervals["phone"],
    )
    cuda_start, cuda_completed, cuda_continuations = validate_cuda(
        cuda,
        plan,
        history,
        phase_id,
        plan["model_sha256"],
        mechanism_sha256,
        intervals["cuda"],
    )
    phone_interval = (phone_start, phone_completed)
    cuda_interval = (cuda_start, cuda_completed)
    exact(phone["route_epoch"], cuda["route_epoch"], "route_epoch")
    for item_index in history["mechanics_b8"]["item_indices"]:
        exact(
            phone_continuations[item_index],
            cuda_continuations[item_index],
            f"E_MECHANICS_CROSS_BACKEND: {item_index}",
        )
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
    ready_base = {
        key: value
        for key, value in cuda["cuda_memory_rows"][1].items()
        if key not in {
            "process_pid",
            "process_used_bytes",
            "sample_id",
            "sampler_sha256",
        }
    }
    ready_normalized = {
        "acquisition_id": phase_id,
        **ready_base,
        "phase": PHASE,
        "phase_id": phase_id,
        "role": f"model.{MODEL_ID}.cuda_memory",
    }
    exact(
        ready.get("cuda_memory_ready_sha256"),
        sha256_bytes(canonical_bytes(ready_normalized)),
        "E_BRIDGE_CUDA_MEMORY_LINK",
    )
    started_ns = min(phone_interval[0], cuda_interval[0])
    completed_ns = max(phone_interval[1], cuda_interval[1])
    require(acquisition_started_ns <= started_ns, "E_ACQUISITION_INTERVAL")
    return {
        "bridge_rows": bridge_rows,
        "completed_ns": completed_ns,
        "cuda_memory_rows": cuda["cuda_memory_rows"],
        "cuda_route_rows": cuda["cuda_route_rows"],
        "cuda_evidence": {
            "fragment": result_artifacts["cuda"],
            "launch_plan_sha256": cuda["launch_plan_sha256"],
            "memory_certificate": cuda["memory_certificate"],
            "placement_certificate": cuda["placement_certificate"],
            "protocol_identity": cuda["protocol_identity"],
            "raw_memory_samples": cuda["raw_memory_samples"],
            "receipt": receipts["cuda"],
            "receipt_artifact": receipt_artifacts["cuda"],
            "runtime_model_binding": cuda["runtime_model_binding"],
            "runtime_process": cuda["runtime_process"],
        },
        "fragment_sha256": {
            "cuda": result_digests["cuda"],
            "phone": result_digests["phone"],
        },
        "gpu_runtime": cuda["gpu_runtime"],
        "mechanics_rows": phone["mechanics_rows"],
        "mechanism_commands_sha256": mechanism_sha256,
        "history_sha256": plan["history"]["sha256"],
        "model_id": MODEL_ID,
        "model_sha256": plan["model_sha256"],
        "op12_runtime": phone["op12_runtime"],
        "op15_runtime": phone["op15_runtime"],
        "phase_id": phase_id,
        "placement_op12_rows": phone["placement_op12_rows"],
        "placement_op15_rows": phone["placement_op15_rows"],
        "phone_evidence": {
            "direct_certificate": phone["direct_certificate"],
            "direct_frames": phone["direct_frames"],
            "fragment": result_artifacts["phone"],
            "launch_plan_sha256": phone["launch_plan_sha256"],
            "placement_certificates": phone["placement_certificates"],
            "raw_probes": phone["raw_probes"],
            "receipt": receipts["phone"],
            "receipt_artifact": receipt_artifacts["phone"],
            "runtime_processes": phone["runtime_processes"],
            "session_certificates": phone["session_certificates"],
        },
        "quality_cuda_rows": cuda["quality_cuda_rows"],
        "quality_phone_rows": phone["quality_phone_rows"],
        "route_epoch": phone["route_epoch"],
        "route_transfer_rows": phone["route_transfer_rows"],
        "runtime_processes": sorted(
            [*phone["runtime_processes"], cuda["runtime_process"]],
            key=lambda value: value["bundle_id"],
        ),
        "schema": OUTPUT_SCHEMA,
        "subproducer_bindings": {
            name: {
                "launch_plan_sha256": plan["commands"][name][
                    "launch_plan_sha256"
                ],
                "producer_sha256": plan["commands"][name]["producer_sha256"],
            }
            for name in ("cuda", "phone")
        },
        "started_ns": started_ns,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-plan", type=Path, required=True)
    parser.add_argument("--capture-plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--pre-dir", type=Path, required=True)
    parser.add_argument("--acquisition-started-ns", type=int, required=True)
    parser.add_argument("--command-plan-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        require(args.capture_plan.is_absolute(), "E_PATH: capture_plan")
        require(args.output.is_absolute(), "E_PATH: output")
        require(args.pre_dir.is_absolute(), "E_PATH: pre_dir")
        require(not args.output.exists(), "E_EXISTS: output")
        string(args.phase_id, "phase_id")
        require(
            args.phase_id.startswith("cp0-r1-v24-a-only-"),
            "E_PHASE_ID",
        )
        require(
            args.execute and args.confirm == "RUN_V24_JOINT_PHONE_CUDA_A_ONLY",
            "E_EXECUTION_NOT_CONFIRMED",
        )
        integer(args.acquisition_started_ns, "acquisition_started_ns", 1)
        digest(args.command_plan_sha256, "command_plan_sha256")
        plan, plan_raw = load_plan(
            args.capture_plan,
            args.capture_plan_sha256,
        )
        history, _ = load_history(plan)
        (
            results,
            intervals,
            result_digests,
            result_artifacts,
            receipts,
            receipt_artifacts,
        ) = run_pair(
            plan,
            args.output,
            args.phase_id,
            args.pre_dir,
            args.acquisition_started_ns,
            args.command_plan_sha256,
        )
        value = build_result(
            plan,
            history,
            results,
            intervals,
            result_digests,
            result_artifacts,
            receipts,
            receipt_artifacts,
            args.phase_id,
            args.acquisition_started_ns,
        )
        evidence_root = (
            args.output.parent / f".{args.output.name}.evidence"
        )
        source_raw, _ = read_regular(Path(__file__).resolve(), "joint.source")
        captured_plan_path = evidence_root / "capture-plan.json"
        captured_source_path = evidence_root / "joint-phone-cuda-v1.py"
        write_new(captured_plan_path, plan_raw)
        write_new(captured_source_path, source_raw, mode=0o755)
        value["capture_plan_sha256"] = sha256_bytes(plan_raw)
        value["capture_plan_artifact"] = {
            "bytes": len(plan_raw),
            "path": str(captured_plan_path),
            "sha256": sha256_bytes(plan_raw),
        }
        value["command_plan_sha256"] = args.command_plan_sha256
        value["executed_file_artifacts"] = {
            name: plan["commands"][name]["executed_files"]
            for name in ("cuda", "phone")
        }
        value["joint_producer_sha256"] = sha256_bytes(source_raw)
        value["joint_producer_artifact"] = {
            "bytes": len(source_raw),
            "path": str(captured_source_path),
            "sha256": sha256_bytes(source_raw),
        }
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
