#!/usr/bin/env python3
"""Run one non-qualification B8 phone/CUDA prototype on live hardware."""

from __future__ import annotations

import argparse
import calendar
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import traceback
from typing import Any


SCHEMA = "s39-cp0-r1-joint-b8-prototype-v1"
MODEL_SHA256 = "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"
PHONE_SERIALS = {
    "op12": "5ae7a43d",
    "op15": "3C15AU002CL00000",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    require(type(value) is dict, f"E_JSON_TYPE: {path}")
    return value, raw


def write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    descriptor = os.open(path, flags, 0o644)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"E_MODULE: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_checked(argv: list[str], timeout: int = 60) -> str:
    completed = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    require(
        completed.returncode == 0 and completed.stderr == b"",
        f"E_COMMAND: {argv[0]} rc={completed.returncode} "
        f"stderr={completed.stderr.decode('utf-8', 'replace')}",
    )
    return completed.stdout.decode("ascii").strip()


def adb(adb_path: str, serial: str, *args: str) -> list[str]:
    return [adb_path, "-P", "5038", "-s", serial, *args]


def phone_live_state(adb_path: str, serial: str) -> dict[str, str]:
    script = (
        "set -eu; "
        "printf 'BOOT='; cat /proc/sys/kernel/random/boot_id; "
        "printf 'IP='; "
        "ip -4 -o addr show dev wlan0 scope global | "
        "awk 'NR==1{split($4,a,\"/\");print a[1]}END{if(NR!=1)exit 43}'"
    )
    raw = run_checked(adb(adb_path, serial, "shell", script))
    values = {}
    for line in raw.splitlines():
        key, separator, value = line.partition("=")
        require(bool(separator) and key not in values, "E_PHONE_STATE")
        values[key] = value
    require(set(values) == {"BOOT", "IP"}, "E_PHONE_STATE_KEYS")
    return {"boot_id": values["BOOT"], "local_ipv4": values["IP"]}


def timestamp_ns(value: str) -> int:
    date, time_value, offset = value.split()
    whole, separator, fraction = time_value.partition(".")
    require(bool(separator) and len(fraction) == 9, "E_STAT_TIMESTAMP")
    parsed = datetime.datetime.strptime(
        f"{date} {whole} {offset}",
        "%Y-%m-%d %H:%M:%S %z",
    )
    return calendar.timegm(parsed.utctimetuple()) * 1_000_000_000 + int(fraction)


def live_component_stats(
    adb_path: str,
    serial: str,
    paths: list[str],
) -> dict[str, dict[str, int]]:
    require(paths and len(paths) == len(set(paths)), "E_COMPONENT_PATHS")
    quoted = " ".join(f"'{path}'" for path in paths)
    script = (
        f"for path in {quoted}; do "
        "printf 'PATH|%s|' \"$path\"; "
        "stat -c '%d|%i|%s|%f|%y|%z' -- \"$path\"; "
        "done"
    )
    raw = run_checked(adb(adb_path, serial, "shell", script))
    result = {}
    for line in raw.splitlines():
        fields = line.split("|")
        require(len(fields) == 8 and fields[0] == "PATH", "E_COMPONENT_STAT")
        path = fields[1]
        require(path in paths and path not in result, "E_COMPONENT_STAT_PATH")
        result[path] = {
            "ctime_ns": timestamp_ns(fields[7]),
            "device_id": int(fields[2]),
            "inode": int(fields[3]),
            "mode": int(fields[5], 16),
            "mtime_ns": timestamp_ns(fields[6]),
            "size": int(fields[4]),
        }
    require(set(result) == set(paths), "E_COMPONENT_STAT_COUNT")
    return result


def patch_phone_plan(
    source: dict[str, Any],
    op12: dict[str, str],
    op15: dict[str, str],
    component_stats: dict[str, dict[str, dict[str, int]]] | None = None,
) -> dict[str, Any]:
    plan = json.loads(json.dumps(source))
    plan["phones"]["op12"]["boot_id"] = op12["boot_id"]
    plan["phones"]["op12"]["local_ipv4"] = op12["local_ipv4"]
    plan["phones"]["op12"]["direct_peer_ipv4"] = op15["local_ipv4"]
    plan["phones"]["op15"]["boot_id"] = op15["boot_id"]
    plan["phones"]["op15"]["local_ipv4"] = op15["local_ipv4"]
    plan["phones"]["op15"]["direct_peer_ipv4"] = op12["local_ipv4"]
    plan["relay_host"] = op15["local_ipv4"]

    for name, endpoint, mechanism_index in (
        ("op12_stagenet", "op12", 0),
        ("op15_stagenet", "op15", 0),
        ("op15_direct_relay", "op15", 1),
    ):
        argv = plan["processes"][name]["argv"]
        require(
            argv.count("--plan-json") == 1
            and argv.count("--plan-sha256") == 1,
            f"E_INLINE_PLAN: {name}",
        )
        plan_index = argv.index("--plan-json") + 1
        digest_index = argv.index("--plan-sha256") + 1
        inline = json.loads(argv[plan_index])
        require(inline["endpoint"] == endpoint, f"E_INLINE_ENDPOINT: {name}")
        if component_stats is not None:
            live = component_stats[endpoint]
            for component in inline["components"]:
                require(
                    component["path"] in live,
                    f"E_COMPONENT_STAT_MISSING: {component['path']}",
                )
                require(
                    component["bytes"] == live[component["path"]]["size"],
                    f"E_COMPONENT_SIZE: {component['path']}",
                )
                component["stat"] = live[component["path"]]
        if name == "op15_direct_relay":
            require(
                inline["route"]["kind"] == "direct_relay",
                "E_RELAY_PLAN",
            )
            inline["route"]["head_host"] = "127.0.0.1"
            inline["route"]["tail_host"] = op12["local_ipv4"]
        else:
            require(
                inline["route"]["kind"] == "stagenet_worker",
                f"E_STAGE_ROUTE: {name}",
            )
            inline["route"]["driver_batch"] = 8
            inline["route"]["driver_max_prefill"] = 8
        inline_raw = json.dumps(
            inline,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        argv[plan_index] = inline_raw
        argv[digest_index] = sha256(inline_raw.encode("ascii"))
        plan["mechanism_commands"][endpoint][mechanism_index] = list(argv)

    for phone, state in (("op12", op12), ("op15", op15)):
        probe = plan["probes"][phone]
        peer = op15 if phone == "op12" else op12
        stage_argv = plan["processes"][f"{phone}_stagenet"]["argv"]
        for when in ("before_argv", "after_argv"):
            probe_argv = probe[when]
            for option in ("--plan-json", "--plan-sha256"):
                require(
                    probe_argv.count(option) == 1
                    and stage_argv.count(option) == 1,
                    f"E_PROBE_PLAN_OPTION: {option}",
                )
                probe_argv[probe_argv.index(option) + 1] = stage_argv[
                    stage_argv.index(option) + 1
                ]
            for option, value in (
                ("--local-ipv4", state["local_ipv4"]),
                ("--peer-ipv4", peer["local_ipv4"]),
            ):
                require(probe_argv.count(option) == 1, f"E_PROBE_OPTION: {option}")
                probe_argv[probe_argv.index(option) + 1] = value
        mechanism_index = 1 if phone == "op12" else 2
        plan["mechanism_commands"][phone][mechanism_index] = list(
            probe["before_argv"]
        )
        plan["mechanism_commands"][phone][mechanism_index + 1] = list(
            probe["after_argv"]
        )
    return plan


def direct_android_processes(
    plan: dict[str, Any],
    usb_launcher_path: Path,
    adb_path: str,
    desktop_cwd: Path,
) -> None:
    usb = load_module("s39_usb_launcher_prototype", usb_launcher_path)
    frozen = usb.load_frozen_launcher()
    for name, process in plan["processes"].items():
        inline_argv = process["argv"]
        plan_index = inline_argv.index("--plan-json") + 1
        digest_index = inline_argv.index("--plan-sha256") + 1
        inline_raw = inline_argv[plan_index]
        inline = usb.validate_plan(
            inline_raw,
            inline_argv[digest_index],
            frozen,
        )
        launcher = next(
            component
            for component in inline["components"]
            if component["component_id"] == inline["launcher_component_id"]
        )
        runtime_argv, runtime_env, runtime_cwd = frozen.build_runtime_command(
            launcher["path"],
            inline["route"],
        )
        if inline["route"]["kind"] == "direct_relay":
            if "--tail-source-port" in runtime_argv:
                index = runtime_argv.index("--tail-source-port")
                del runtime_argv[index:index + 2]
            if "--emit-direct-frames" in runtime_argv:
                runtime_argv.remove("--emit-direct-frames")
        command = " ".join(
            [
                "cd",
                shlex.quote(runtime_cwd),
                "&&",
                "exec",
                "env",
                *(
                    f"{key}={shlex.quote(value)}"
                    for key, value in sorted(runtime_env.items())
                ),
                *(shlex.quote(value) for value in runtime_argv),
            ]
        )
        process["argv"] = adb(
            adb_path,
            inline["android"]["physical_serial"],
            "shell",
            "sh -c " + shlex.quote(command),
        )
        process["cwd"] = str(desktop_cwd)
        process["environment"] = dict(os.environ)


def wait_port(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError as error:
            last_error = error
            time.sleep(0.1)
    raise RuntimeError(f"E_PORT: {host}:{port}: {last_error}")


def patch_cuda_plan(source: dict[str, Any]) -> dict[str, Any]:
    plan = json.loads(json.dumps(source))
    argv = plan["worker"]["argv"]
    require(argv.count("--model") == 1, "E_CUDA_OPTION: --model")
    argv[argv.index("--model")] = "-m"
    require(argv.count("--backend") == 1, "E_CUDA_OPTION: --backend")
    argv[argv.index("--backend")] = "--devices"
    for option in ("--layer-start", "--layer-end"):
        require(argv.count(option) == 1, f"E_CUDA_OPTION: {option}")
        index = argv.index(option)
        del argv[index:index + 2]
    require("-ngl" not in argv, "E_CUDA_OPTION: -ngl")
    argv.extend(["-ngl", "999"])
    for option, value in (
        ("--driver-batch", "8"),
        ("--driver-max-prefill", "8"),
    ):
        require(argv.count(option) == 1, f"E_CUDA_OPTION: {option}")
        argv[argv.index(option) + 1] = value
    return plan


def prefixed_json(path: Path, prefixes: tuple[bytes, ...]) -> list[dict[str, Any]]:
    values = []
    raw = path.read_bytes()
    for line in raw.splitlines():
        for prefix in prefixes:
            if line.startswith(prefix):
                values.append(
                    {
                        "prefix": prefix.decode("ascii").strip(),
                        "value": json.loads(line[len(prefix):]),
                    }
                )
    return values


def run_group(module, client, history, route_epoch, request_base):
    request_ids = list(range(request_base, request_base + 8))
    started_ns = time.monotonic_ns()
    value = module.run_history_group(
        client,
        history,
        history["mechanics_b8"],
        request_ids,
        route_epoch,
        0,
    )
    completed_ns = time.monotonic_ns()
    return request_ids, value, started_ns, completed_ns


def observed_hello(module, client, expected_capabilities: int) -> dict[str, Any]:
    client.connection.sendall(module.pack_i32([module.STAGE_V3_HELLO]))
    words = client.recv_i32(11)
    require(words[0] == module.STAGE_V3_MAGIC, "E_HELLO_MAGIC")
    require(words[1] == module.STAGE_V3_VERSION, "E_HELLO_VERSION")
    require(list(words[2:5]) == [0, 40, 40], "E_HELLO_LAYERS")
    require(words[5] == 5120, "E_HELLO_EMBEDDING")
    require(words[6] == 8, f"E_HELLO_STREAMS: {words[6]}")
    require(words[7] >= 512, f"E_HELLO_CONTEXT: {words[7]}")
    require(words[8] == 64, f"E_HELLO_BATCH: {words[8]}")
    require(words[9] == 64, f"E_HELLO_UBATCH: {words[9]}")
    require(words[10] == expected_capabilities, "E_HELLO_CAPABILITIES")
    client.n_batch = words[8]
    client.n_ubatch = words[9]
    client.connection.sendall(module.pack_i32([module.STAGE_V3_IDENTITY]))
    identity = client.recv_i32(3)
    require(
        identity
        == (
            module.STAGE_IDENTITY_MAGIC,
            module.STAGE_IDENTITY_VERSION,
            module.FILE_TYPE,
        ),
        "E_IDENTITY_HEADER",
    )
    model_sha256 = client.recv_exact(32).hex()
    require(model_sha256 == MODEL_SHA256, "E_IDENTITY_MODEL")
    return {
        "capabilities": words[10],
        "file_type": identity[2],
        "layer_end": words[3],
        "layer_start": words[2],
        "max_streams": words[6],
        "model_sha256": model_sha256,
        "n_batch": words[8],
        "n_ctx_reported": words[7],
        "n_embd": words[5],
        "n_layer": words[4],
        "n_ubatch": words[9],
        "stage_identity_version": identity[1],
        "stage_protocol_version": words[1],
    }


def cleanup_remote(adb_path: str) -> None:
    roots = (
        "/data/local/tmp/s39-v24-a-only/runtime-v1/op12-stagenet/",
        "/data/local/tmp/s39-v24-a-only/runtime-v1/op15-stagenet/",
        "/data/local/tmp/s39-v24-a-only/runtime-v1/op15-direct-relay/",
    )
    root_tests = " || ".join(
        f'[ \"$exe\" = \"{root}llama-layersplit\" ]'
        if "stagenet" in root
        else f'[ \"$exe\" = \"{root}llama-stage-direct-relay\" ]'
        for root in roots
    )
    script = (
        "for pid in $(pidof llama-layersplit llama-stage-direct-relay "
        "2>/dev/null); do "
        "exe=$(readlink /proc/$pid/exe 2>/dev/null || true); "
        f"if {root_tests}; then kill -9 $pid; fi; "
        "done"
    )
    for serial in PHONE_SERIALS.values():
        subprocess.run(
            adb(adb_path, serial, "shell", script),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phone-plan", type=Path, required=True)
    parser.add_argument("--cuda-plan", type=Path, required=True)
    parser.add_argument("--phone-producer", type=Path, required=True)
    parser.add_argument("--cuda-producer", type=Path, required=True)
    parser.add_argument("--usb-launcher", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--adb-path",
        default="/usr/lib/android-sdk/platform-tools/adb",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    require(
        args.execute and args.confirm == "RUN_S39_JOINT_B8_PROTOTYPE",
        "E_CONFIRMATION",
    )
    require(args.output_root.is_absolute(), "E_OUTPUT_ROOT")
    args.output_root.mkdir(parents=True, exist_ok=False)

    phone = load_module("s39_phone_route_prototype", args.phone_producer)
    cuda = load_module("s39_cuda_route_prototype", args.cuda_producer)
    phone_source, phone_source_raw = read_json(args.phone_plan)
    cuda_source, cuda_source_raw = read_json(args.cuda_plan)
    require(
        phone_source["model_sha256"] == MODEL_SHA256
        and cuda_source["model_sha256"] == MODEL_SHA256,
        "E_MODEL",
    )

    states = {
        name: phone_live_state(args.adb_path, serial)
        for name, serial in PHONE_SERIALS.items()
    }
    require(
        states["op12"]["local_ipv4"].startswith("172.20.")
        and states["op15"]["local_ipv4"].startswith("172.20."),
        "E_WIFI_NETWORK",
    )
    component_paths = {"op12": set(), "op15": set()}
    for process in phone_source["processes"].values():
        argv = process["argv"]
        inline = json.loads(argv[argv.index("--plan-json") + 1])
        component_paths[inline["endpoint"]].update(
            component["path"] for component in inline["components"]
        )
    component_stats = {
        endpoint: live_component_stats(
            args.adb_path,
            PHONE_SERIALS[endpoint],
            sorted(paths),
        )
        for endpoint, paths in component_paths.items()
    }
    phone_plan_value = patch_phone_plan(
        phone_source,
        states["op12"],
        states["op15"],
        component_stats,
    )
    patched_phone_path = args.output_root / "phone-route-prototype.json"
    phone_plan_raw = canonical_bytes(phone_plan_value)
    write_new(patched_phone_path, phone_plan_raw)
    phone_plan = phone_plan_value
    require(
        phone_plan["model_sha256"] == MODEL_SHA256
        and phone_plan["expected_n_layer"] == 40
        and phone_plan["expected_n_embd"] == 5120
        and phone_plan["expected_max_streams"] == 8
        and phone_plan["expected_n_batch"] == 64
        and phone_plan["expected_n_ubatch"] == 64,
        "E_PHONE_PLAN_GEOMETRY",
    )
    direct_android_processes(
        phone_plan,
        args.usb_launcher,
        args.adb_path,
        args.output_root,
    )

    histories_path = Path(phone_plan["history_path"])
    history, history_raw = phone.load_histories(
        histories_path,
        phone_plan["history_sha256"],
        MODEL_SHA256,
    )
    cuda_validated, _ = cuda.load_plan(
        args.cuda_plan,
        histories_path,
        history_raw,
        cuda_source["model_artifact"],
    )
    cuda_prototype_value = patch_cuda_plan(cuda_validated)
    cuda_prototype_path = args.output_root / "cuda-route-prototype.json"
    cuda_prototype_raw = canonical_bytes(cuda_prototype_value)
    write_new(cuda_prototype_path, cuda_prototype_raw)
    cuda_plan = cuda_prototype_value

    result = {
        "cuda_plan_parent_sha256": sha256(cuda_source_raw),
        "cuda_plan_prototype_sha256": sha256(cuda_prototype_raw),
        "model_sha256": MODEL_SHA256,
        "phone_plan_parent_sha256": sha256(phone_source_raw),
        "phone_plan_prototype_sha256": sha256(phone_plan_raw),
        "prototype_only": True,
        "prototype_bypasses": [
            "phone_route_v1.load_plan argv item limit rejects its own "
            "embedded managed-plan JSON",
            "materialized Android component stats lose subsecond ctime",
            "managed Android launcher rejects the live process executable "
            "identity; prototype uses direct ADB with its derived argv",
            "deployed relay lacks tail-source-port and direct-frame options",
            "materialized driver batch creates 64 streams instead of 8",
            "runtime hello reports total rather than per-stream context",
            "materialized CUDA argv uses unsupported model/backend and "
            "layer-bound spellings",
            "V2.4 reboot and fresh-readiness evidence",
        ],
        "schema": SCHEMA,
        "states": states,
        "status": "PROTOTYPE_STARTED",
    }
    processes = {}
    phone_client = None
    cuda_client = None
    cuda_process = None
    cuda_log = None
    cuda_log_path = args.output_root / "cuda.log"
    try:
        cleanup_remote(args.adb_path)
        processes = phone.start_processes(phone_plan, args.output_root)
        phone_client = phone.connect_route(
            phone_plan["relay_host"],
            phone_plan["relay_port"],
            phone_plan["processes"]["op15_direct_relay"]["startup_timeout_ms"],
            processes["op15_direct_relay"].process,
        )
        phone_identity = observed_hello(
            phone,
            phone_client,
            phone.STAGE_V3_REQUIRED_CAPABILITIES,
        )
        require(phone_client.status()[0] == 0, "E_PHONE_STATE_BEFORE")

        cuda_process, cuda_log, cuda_started_ns = cuda.start_worker(
            cuda_plan,
            cuda_log_path,
        )
        cuda_client = cuda.connect(cuda_plan, cuda_process)
        cuda_identity = observed_hello(
            cuda,
            cuda_client,
            cuda.CAPABILITIES,
        )
        require(cuda_client.status()[0] == 0, "E_CUDA_STATE_BEFORE")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            phone_future = executor.submit(
                run_group,
                phone,
                phone_client,
                history,
                phone_plan["route_epoch"],
                1001,
            )
            cuda_future = executor.submit(
                run_group,
                cuda,
                cuda_client,
                history,
                cuda_plan["route_epoch"],
                2001,
            )
            phone_run = phone_future.result()
            cuda_run = cuda_future.result()

        phone_ids, phone_value, phone_started_ns, phone_completed_ns = phone_run
        cuda_ids, cuda_value, cuda_run_started_ns, cuda_completed_ns = cuda_run
        phone_tokens, phone_calls, phone_frames, _phone_receipt = phone_value
        cuda_tokens, cuda_calls, _cuda_receipt = cuda_value
        require(phone_client.status()[0] == 8, "E_PHONE_STATE_READY")
        require(cuda_client.status()[0] == 8, "E_CUDA_STATE_READY")
        phone.remove_group(
            phone_client,
            phone_ids,
            phone_plan["route_epoch"],
        )
        cuda.remove_group(
            cuda_client,
            cuda_ids,
            cuda_plan["route_epoch"],
        )
        require(phone_client.status()[0] == 0, "E_PHONE_STATE_AFTER")
        require(cuda_client.status()[0] == 0, "E_CUDA_STATE_AFTER")

        phone_client.stop()
        phone_client.connection.close()
        phone_client = None
        cuda_client.stop()
        cuda_client.connection.close()
        cuda_client = None
        phone.stop_processes(processes)
        processes = {}
        cuda_completed_process_ns = cuda.stop_process(
            cuda_process,
            cuda_log,
            cuda_plan["worker"]["shutdown_timeout_ms"],
        )
        cuda_process = None
        cuda_log = None

        agreement = sum(
            phone_token == cuda_token
            for phone_row, cuda_row in zip(phone_tokens, cuda_tokens)
            for phone_token, cuda_token in zip(phone_row, cuda_row)
        )
        total = sum(len(row) for row in phone_tokens)
        result.update(
            {
                "agreement": {
                    "matching_tokens": agreement,
                    "total_tokens": total,
                },
                "calls": {
                    "cuda": cuda_calls,
                    "phone": phone_calls,
                },
                "certificates": {
                    "cuda": prefixed_json(
                        cuda_log_path,
                        (b"PLACEMENTCERT ", b"MEMORYCERT ", b"RUNTIMECERT "),
                    ),
                    "op12": prefixed_json(
                        args.output_root / "op12_stagenet.log",
                        (b"PLACEMENTCERT ", b"SESSIONCERT "),
                    ),
                    "op15": prefixed_json(
                        args.output_root / "op15_stagenet.log",
                        (b"PLACEMENTCERT ", b"SESSIONCERT "),
                    ),
                    "relay": prefixed_json(
                        args.output_root / "op15_direct_relay.log",
                        (b"DIRECTCERT ",),
                    ),
                },
                "cuda_identity": cuda_identity,
                "phone_identity": phone_identity,
                "cuda_tokens": cuda_tokens,
                "phone_tokens": phone_tokens,
                "timing_ns": {
                    "cuda_process_completed": cuda_completed_process_ns,
                    "cuda_process_started": cuda_started_ns,
                    "cuda_run_completed": cuda_completed_ns,
                    "cuda_run_started": cuda_run_started_ns,
                    "phone_run_completed": phone_completed_ns,
                    "phone_run_started": phone_started_ns,
                },
                "transfer": {
                    "frames": len(phone_frames),
                    "path": "WIFI_TCP_DIRECT",
                },
                "status": (
                    "JOINT_B8_PROTOTYPE_PASS"
                    if agreement == total
                    else "JOINT_B8_PROTOTYPE_EXECUTION_PASS_TOKEN_DIVERGENCE"
                ),
            }
        )
        write_new(args.output_root / "RESULT.json", canonical_bytes(result))
        return 0
    except BaseException as error:
        result.update(
            {
                "error": f"{type(error).__name__}: {error}",
                "status": "JOINT_B8_PROTOTYPE_FAILED",
                "traceback": traceback.format_exc(),
            }
        )
        try:
            write_new(args.output_root / "FAILURE.json", canonical_bytes(result))
        except FileExistsError:
            pass
        return 2
    finally:
        if phone_client is not None:
            try:
                phone_client.connection.close()
            except OSError:
                pass
        if cuda_client is not None:
            try:
                cuda_client.connection.close()
            except OSError:
                pass
        for process in processes.values():
            process.kill()
        if cuda_process is not None:
            cuda.kill_process(cuda_process, cuda_log)
        cleanup_remote(args.adb_path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        print(f"JOINT_B8_PROTOTYPE_REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
