#!/usr/bin/env python3

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import types


SOURCE = Path(__file__).resolve().parents[1] / "remote_cuda_capture_v1.py"
MODULE = types.ModuleType("remote_cuda_capture_v1")
MODULE.__file__ = str(SOURCE)
SOURCE_RAW = SOURCE.read_bytes()
exec(compile(SOURCE_RAW, str(SOURCE), "exec"), MODULE.__dict__)

BOOT_A = "11111111-2222-3333-4444-555555555555"
BOOT_B = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def expect_error(name, function, needle):
    try:
        function()
    except MODULE.CaptureError as error:
        assert needle in str(error), (name, error)
        return
    raise AssertionError(f"{name}: accepted")


def stat_row(size=100):
    return {
        "build_id": None,
        "ctime_ns": 10,
        "device_id": 11,
        "inode": 12,
        "mode": 0o100755,
        "mtime_ns": 13,
        "size": size,
    }


def artifact(path, digest_value=DIGEST_A, size=100):
    return {
        "bytes": size,
        "path": path,
        "sha256": digest_value,
        "stat": stat_row(size),
    }


def wrapper_plan(role="cuda_monolithic"):
    config = MODULE.ROLE_CONFIG[role]
    producer = artifact(
        "/home/zhihao/llama.cpp-s40/research_dev/spikes/"
        "s39_phone_model_switch_trace/v24_readiness/producers_v1/"
        + config["producer_name"],
        DIGEST_B,
        200,
    )
    remote_output = f"/home/zhihao/run/{role}.json"
    v24_phase_id = "cp0-r1-v24-a-only-test"
    producer_argv = [
        "/usr/bin/python3.14",
        "-I",
        producer["path"],
    ]
    joint_bindings = None
    if role == "joint_phone_cuda":
        joint_bindings = {
            "adb": artifact("/usr/bin/adb", "2" * 64, 401),
            "adb_server_port": 5038,
            "adb_server_process": {
                "argv": ["/usr/bin/adb", "-P", "5038", "server", "nodaemon"],
                "boot_id": BOOT_B,
                "executable_path": "/usr/bin/adb",
                "listen_host": "127.0.0.1",
                "listen_port": 5038,
                "pid": 505,
                "start_ticks": 5005,
            },
            "capture_plan": artifact(
                "/home/zhihao/run/joint-capture-plan.json",
                "3" * 64,
                402,
            ),
            "cuda_launch_plan": artifact(
                "/home/zhihao/run/cuda-launch.json",
                "4" * 64,
                403,
            ),
            "op12_selector": "172.20.74.12:5555",
            "op15_selector": "172.20.74.15:5555",
            "phone_launch_plan": artifact(
                "/home/zhihao/run/phone-launch.json",
                "5" * 64,
                404,
            ),
        }
        producer_argv.extend([
            "--capture-plan",
            joint_bindings["capture_plan"]["path"],
            "--capture-plan-sha256",
            joint_bindings["capture_plan"]["sha256"],
        ])
    producer_argv.extend([
        "--output",
        remote_output,
        "--phase-id",
        v24_phase_id,
    ])
    if role == "joint_phone_cuda":
        producer_argv.extend([
            "--pre-dir",
            "/home/zhihao/run/pre",
            "--acquisition-started-ns",
            "1",
            "--command-plan-sha256",
            "6" * 64,
        ])
    producer_argv.extend([
        "--execute",
        "--confirm",
        (
            "RUN_V24_JOINT_PHONE_CUDA_A_ONLY"
            if role == "joint_phone_cuda"
            else "RUN-S39-CP0-R1-V24"
        ),
    ])
    return {
        "frozen_producer": producer,
        "joint_bindings": joint_bindings,
        "local_python": artifact("/opt/python3.13", "9" * 64, 250),
        "managed_launcher": artifact(
            "/repo/managed_runtime_launcher_v1.py",
            DIGEST_A,
            100,
        ),
        "managed_plan_sha256": "c" * 64,
        "phase": "A_ONLY",
        "phase_id": "cp0-r1-v25-a-only-test",
        "producer_argv": producer_argv,
        "remote_output_path": remote_output,
        "role": role,
        "schema": MODULE.PLAN_SCHEMA,
        "sequence_index": config["sequence_index"],
        "timeout_seconds": 60,
        "v24_phase_id": v24_phase_id,
    }


def managed_plan(role="cuda_monolithic"):
    plan = wrapper_plan(role)
    producer = plan["frozen_producer"]
    argv = plan["producer_argv"]
    component_map = {
        "python": artifact("/usr/bin/python3.14", "d" * 64, 300),
        "producer": producer,
    }
    if plan["joint_bindings"] is not None:
        component_map.update({
            key: value
            for key, value in plan["joint_bindings"].items()
            if key in {
                "adb",
                "capture_plan",
                "cuda_launch_plan",
                "phone_launch_plan",
            }
        })
    return {
        "bundle_id": f"v24_{role}_producer",
        "components": list(component_map.values()),
        "endpoint": "cuda",
        "mode": "remote_cuda",
        "route": {
            "argv": argv,
            "cwd": "/home/zhihao/llama.cpp-s40",
            "environment": {},
            "kind": "remote_exec",
            "local_forward": {
                "local_host": "127.0.0.1",
                "local_port": 49111,
                "remote_host": "127.0.0.1",
                "remote_port": 49112,
            },
        },
        "ssh": {
            "gpu_uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
            "remote_python_path": "/usr/bin/python3.14",
            "remote_python_sha256": "d" * 64,
            "remote_python_stat": stat_row(300),
        },
        "_normalized": {
            "argv": argv,
            "component_map": component_map,
            "launcher_path": "/usr/bin/python3.14",
        },
    }


class FakeLauncher:
    def __init__(self, plan):
        self.plan = plan

    def parse_plan_json(self, raw, digest):
        del raw, digest
        return copy.deepcopy(self.plan)


def transport_row(plan):
    return {
        "argv": ["/usr/bin/ssh", "-L", "127.0.0.1:49111:127.0.0.1:49112"],
        "bundle_id": plan["bundle_id"],
        "endpoint": "cuda",
        "host_boot_id": BOOT_A,
        "managed_launcher_pid": 100,
        "managed_launcher_start_ticks": 1000,
        "observed_ns": 10000,
        "pid": 101,
        "plan_sha256": "c" * 64,
        "remote_boot_id": BOOT_B,
        "schema": MODULE.TRANSPORT_SCHEMA,
        "start_ticks": 1001,
    }


def runtime_row(plan):
    return {
        "boot_id": BOOT_B,
        "bundle_id": plan["bundle_id"],
        "controller_clock": "CONTROLLER_MONOTONIC",
        "controller_observed_ns": 10001,
        "endpoint": "cuda",
        "launch_token": "1" * 32,
        "launcher_path": "/usr/bin/python3.14",
        "loaded_repo_component_ids": sorted(plan["_normalized"]["component_map"]),
        "pgid": 202,
        "pid": 202,
        "remote_observed_ns": 90,
        "remote_clock": "RTX_CLOCK_MONOTONIC_RAW",
        "schema": MODULE.RUNTIME_SCHEMA,
        "start_ticks": 2002,
        "system_dependencies": sorted(
            [
                {
                    **component["stat"],
                    "path": component["path"],
                    "sha256": component["sha256"],
                }
                for component in plan["components"]
            ],
            key=lambda item: item["path"],
        ),
    }


def managed_cleanup_row(runtime):
    return {
        "absent": [{
            "pid": runtime["pid"],
            "start_ticks": runtime["start_ticks"],
        }],
        "boot_id": BOOT_B,
        "clock": "RTX_CLOCK_MONOTONIC_RAW",
        "gpu_uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
        "launch_token": runtime["launch_token"],
        "matching_nvml_pids": [],
        "matching_process_groups": [],
        "matching_processes": [],
        "observed_ns": 210,
        "pgid": runtime["pgid"],
        "pid": runtime["pid"],
        "schema": MODULE.MANAGED_CLEANUP_SCHEMA,
        "start_ticks": runtime["start_ticks"],
    }


def launcher_output(plan, transport=None, runtime=None, cleanup=None):
    transport = transport or transport_row(plan)
    runtime = runtime or runtime_row(plan)
    cleanup = cleanup or managed_cleanup_row(runtime)
    return (
        b"TRANSPORTPROCESS " + MODULE.canonical_bytes(transport)
        + b"RUNTIMEPROCESS " + MODULE.canonical_bytes(runtime)
        + MODULE.MANAGED_CLEANUP_PREFIX + MODULE.canonical_bytes(cleanup)
    )


def legacy_runtime_row(plan):
    row = runtime_row(plan)
    for key in (
        "controller_observed_ns",
        "controller_clock",
        "launch_token",
        "pgid",
        "remote_observed_ns",
        "remote_clock",
    ):
        row.pop(key)
    row["observed_ns"] = 101
    row["system_dependencies"] = []
    return row


def remote_result(plan):
    role = plan["role"]
    result = {
        "completed_ns": 200,
        "phase_id": plan["v24_phase_id"],
        "schema": MODULE.ROLE_CONFIG[role]["result_schema"],
        "started_ns": 100,
    }
    cuda_runtime = legacy_runtime_row(managed_plan(role))
    cuda_runtime["bundle_id"] = role
    cuda_runtime["pid"] = 404
    cuda_runtime["start_ticks"] = 4004
    if role == "cuda_monolithic":
        result["producer_sha256"] = plan["frozen_producer"]["sha256"]
        result["runtime_process"] = cuda_runtime
        result["producer_process_receipt"] = {
            "boot_id": BOOT_B,
            "pid": 202,
            "start_ticks": 2002,
        }
    else:
        joint = plan["joint_bindings"]
        result.update({
            "capture_plan_sha256": joint["capture_plan"]["sha256"],
            "cuda_evidence": {"runtime_process": cuda_runtime},
            "joint_producer_sha256": plan["frozen_producer"]["sha256"],
            "runtime_processes": [
                {
                    "endpoint": endpoint,
                    "pid": pid,
                    "start_ticks": ticks,
                }
                for endpoint, pid, ticks in (
                    ("op12", 701, 7001),
                    ("op15", 702, 7002),
                    ("op15", 703, 7003),
                )
            ],
            "subproducer_bindings": {
                "cuda": {
                    "launch_plan_sha256": joint["cuda_launch_plan"]["sha256"],
                },
                "phone": {
                    "launch_plan_sha256": joint["phone_launch_plan"]["sha256"],
                },
            },
        })
    return result


def fetched(plan, result=None):
    result = result or remote_result(plan)
    raw = MODULE.canonical_bytes(result)
    primary = artifact(
        plan["remote_output_path"],
        MODULE.hashlib.sha256(raw).hexdigest(),
        len(raw),
    )
    primary["content_base64"] = base64.b64encode(raw).decode("ascii")
    joint = plan["role"] == "joint_phone_cuda"
    return {
        "adb_server_process": (
            {
                **plan["joint_bindings"]["adb_server_process"],
                "listener_inode": 606,
                "observed_ns": 250,
            }
            if joint
            else None
        ),
        "artifacts": [],
        "boot_id": BOOT_B,
        "gpu_uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
        "nvml_compute_pids": [],
        "primary": primary,
        "phone_processes_absent": (
            [
                {
                    "endpoint": endpoint,
                    "pid": pid,
                    "start_ticks": ticks,
                    "state": "absent",
                }
                for endpoint, pid, ticks in (
                    ("op12", 701, 7001),
                    ("op15", 702, 7002),
                    ("op15", 703, 7003),
                )
            ]
            if joint
            else []
        ),
        "processes_absent": [
            {"pid": 202, "start_ticks": 2002},
            {"pid": 404, "start_ticks": 4004},
        ],
        "remote_cleanup_observed_ns": 300,
        "schema": MODULE.FETCH_SCHEMA,
        "system_swap_used_bytes": 0,
    }


def receipt(role, started, completed):
    managed = managed_plan(role)
    transport = transport_row(managed)
    runtime = runtime_row(managed)
    middle = started + 1
    transport["observed_ns"] = middle
    runtime["controller_observed_ns"] = middle
    result_artifact = artifact(
        f"/remote/{role}.json",
        "f" * 64,
        123,
    )
    return {
        "cleanup": {
            "execution_transport_absent": True,
            "fetch_transport_absent": True,
            "local_forward_listener_absent": True,
            "observed_ns": completed,
            "remote_cuda_processes_absent": [{
                "pid": runtime["pid"],
                "start_ticks": runtime["start_ticks"],
            }, {
                "pid": 404,
                "start_ticks": 4004,
            }],
            "remote_cleanup_observed_ns": 300,
        },
        "completed_ns": completed,
        "adb_server_process": (
            {
                **wrapper_plan(role)["joint_bindings"]["adb_server_process"],
                "listener_inode": 606,
                "observed_ns": 250,
            }
            if role == "joint_phone_cuda"
            else None
        ),
        "execution_transport_process": transport,
        "fetch_transport_process": {
            "argv": ["/usr/bin/ssh"],
            "completed_ns": completed - 1,
            "pid": 303,
            "schema": MODULE.FETCH_TRANSPORT_SCHEMA,
            "start_ticks": 3003,
            "started_ns": started + 2,
        },
        "frozen_producer": wrapper_plan(role)["frozen_producer"],
        "gpu_uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
        "joint_bindings": wrapper_plan(role)["joint_bindings"],
        "local_python": wrapper_plan(role)["local_python"],
        "local_evidence_artifacts": [],
        "local_result_artifact": artifact(
            f"/local/{role}.json",
            "f" * 64,
            123,
        ),
        "managed_remote_cleanup": managed_cleanup_row(runtime),
        "managed_plan_sha256": "c" * 64,
        "phase": "A_ONLY",
        "phase_id": "cp0-r1-v25-a-only-test",
        "phone_processes_absent": (
            [
                {
                    "endpoint": endpoint,
                    "pid": pid,
                    "start_ticks": ticks,
                    "state": "absent",
                }
                for endpoint, pid, ticks in (
                    ("op12", 701, 7001),
                    ("op15", 702, 7002),
                    ("op15", 703, 7003),
                )
            ]
            if role == "joint_phone_cuda"
            else []
        ),
        "remote_boot_id": BOOT_B,
        "remote_evidence_artifacts": [],
        "remote_execution_interval": {
            "clock": "RTX_CLOCK_MONOTONIC_RAW",
            "completed_ns": 200,
            "started_ns": 100,
        },
        "remote_producer_process": runtime,
        "remote_result_artifact": result_artifact,
        "role": role,
        "schema": MODULE.RECEIPT_SCHEMA,
        "sequence_index": MODULE.ROLE_CONFIG[role]["sequence_index"],
        "started_ns": started,
        "system_swap_used_bytes": 0,
        "v24_phase_id": "cp0-r1-v24-a-only-test",
        "wrapper_plan_sha256": "e" * 64,
    }


def bound_receipt(role, started, completed, root, remote_started):
    managed = managed_plan(role)
    managed_public = {
        key: value
        for key, value in managed.items()
        if key != "_normalized"
    }
    managed_raw = MODULE.canonical_compact(managed_public)
    wrapper = wrapper_plan(role)
    wrapper["managed_plan_sha256"] = MODULE.hashlib.sha256(
        managed_raw
    ).hexdigest()
    wrapper_raw = MODULE.canonical_bytes(wrapper)
    value = receipt(role, started, completed)
    value["managed_plan_sha256"] = wrapper["managed_plan_sha256"]
    value["execution_transport_process"]["plan_sha256"] = (
        wrapper["managed_plan_sha256"]
    )
    value["wrapper_plan_sha256"] = MODULE.hashlib.sha256(
        wrapper_raw
    ).hexdigest()
    value["frozen_producer"] = wrapper["frozen_producer"]
    value["joint_bindings"] = wrapper["joint_bindings"]
    value["local_python"] = wrapper["local_python"]
    value["remote_producer_process"] = runtime_row(managed)
    value["remote_producer_process"]["controller_observed_ns"] = started + 1
    value["remote_producer_process"]["remote_observed_ns"] = remote_started - 10
    value["managed_remote_cleanup"] = managed_cleanup_row(
        value["remote_producer_process"]
    )
    remote_completed = remote_started + 100
    value["managed_remote_cleanup"]["observed_ns"] = remote_completed + 10
    value["remote_execution_interval"] = {
        "clock": "RTX_CLOCK_MONOTONIC_RAW",
        "completed_ns": remote_completed,
        "started_ns": remote_started,
    }
    value["cleanup"]["remote_cleanup_observed_ns"] = remote_completed + 20
    result = remote_result(wrapper)
    result["started_ns"] = remote_started
    result["completed_ns"] = remote_completed
    raw = MODULE.canonical_bytes(result)
    local_path = (root / f"{role}.json").resolve()
    local_path.write_bytes(raw)
    local_artifact = MODULE.artifact_from_raw(local_path, raw)
    value["local_result_artifact"] = local_artifact
    value["remote_result_artifact"] = {
        **local_artifact,
        "path": wrapper["remote_output_path"],
    }
    if role == "joint_phone_cuda":
        value["adb_server_process"]["observed_ns"] = remote_completed + 15
    binding = (wrapper, wrapper_raw, managed, managed_raw)
    return value, binding


def test_plan():
    MODULE.validate_plan(wrapper_plan())
    bad = wrapper_plan()
    bad["sequence_index"] = 2
    expect_error(
        "wrong sequence",
        lambda: MODULE.validate_plan(bad),
        "plan.sequence_index",
    )
    bad = wrapper_plan()
    bad["frozen_producer"]["path"] = "/tmp/cuda_route_v1.py"
    expect_error(
        "wrong producer",
        lambda: MODULE.validate_plan(bad),
        "frozen_producer.name",
    )
    bad = wrapper_plan()
    bad["v24_phase_id"] = "cp0-r1-v24-a-only-other"
    expect_error(
        "phase link",
        lambda: MODULE.validate_plan(bad),
        "E_V24_PHASE_ID",
    )


def test_managed_binding():
    wrapper = wrapper_plan()
    managed = managed_plan()
    launcher = FakeLauncher(managed)
    observed = MODULE.validate_managed_plan(
        launcher,
        b"{}",
        "c" * 64,
        wrapper,
    )
    assert observed["_normalized"]["argv"][2] == wrapper["frozen_producer"]["path"]
    bad = managed_plan()
    bad["_normalized"]["argv"][2] = "/tmp/other.py"
    bad["route"]["argv"][2] = "/tmp/other.py"
    expect_error(
        "producer argv",
        lambda: MODULE.validate_managed_plan(
            FakeLauncher(bad),
            b"{}",
            "c" * 64,
            wrapper,
        ),
        "managed.argv",
    )
    bad = managed_plan()
    bad["_normalized"]["argv"][0] = "/usr/bin/python3"
    bad["route"]["argv"][0] = "/usr/bin/python3"
    expect_error(
        "python symlink",
        lambda: MODULE.validate_managed_plan(
            FakeLauncher(bad),
            b"{}",
            "c" * 64,
            wrapper,
        ),
        "managed.argv",
    )


def test_real_launcher_component_shape():
    wrapper = wrapper_plan("joint_phone_cuda")
    managed = managed_plan("joint_phone_cuda")
    for index, component in enumerate(managed["components"]):
        component["component_id"] = f"component_{index}"
    MODULE.validate_managed_plan(
        FakeLauncher(managed),
        b"{}",
        "c" * 64,
        wrapper,
    )

    bad = copy.deepcopy(managed)
    bad["components"][0]["unbound"] = True
    expect_error(
        "unknown component field",
        lambda: MODULE.validate_managed_plan(
            FakeLauncher(bad),
            b"{}",
            "c" * 64,
            wrapper,
        ),
        "E_KEYS: managed.components[0]",
    )


def test_launcher_rows():
    plan = managed_plan()
    transport, runtime, cleanup = MODULE.parse_launcher_output(
        launcher_output(plan),
        plan,
        "c" * 64,
        BOOT_B,
    )
    assert transport["pid"] == 101
    assert runtime["pid"] == 202
    assert cleanup["pid"] == 202

    stale = runtime_row(plan)
    stale["boot_id"] = BOOT_A
    expect_error(
        "stale remote boot",
        lambda: MODULE.parse_launcher_output(
            launcher_output(plan, runtime=stale),
            plan,
            "c" * 64,
            BOOT_B,
        ),
        "runtime.boot",
    )
    substituted = runtime_row(plan)
    substituted["pid"] = 101
    substituted["pgid"] = 101
    expect_error(
        "ssh pid substituted",
        lambda: MODULE.parse_launcher_output(
            launcher_output(plan, runtime=substituted),
            plan,
            "c" * 64,
            BOOT_B,
        ),
        "E_PID_SUBSTITUTION",
    )
    expect_error(
        "forged stdout",
        lambda: MODULE.parse_launcher_output(
            launcher_output(plan) + b"junk\n",
            plan,
            "c" * 64,
            BOOT_B,
        ),
        "E_LAUNCHER_STDOUT",
    )

    orphan = managed_cleanup_row(runtime_row(plan))
    orphan["matching_processes"] = [{
        "pgid": 303,
        "pid": 303,
        "start_ticks": 3003,
    }]
    expect_error(
        "orphan process",
        lambda: MODULE.parse_launcher_output(
            launcher_output(plan, cleanup=orphan),
            plan,
            "c" * 64,
            BOOT_B,
        ),
        "managed_cleanup.matching_processes",
    )

    orphan = managed_cleanup_row(runtime_row(plan))
    orphan["matching_nvml_pids"] = [303]
    expect_error(
        "orphan nvml pid",
        lambda: MODULE.parse_launcher_output(
            launcher_output(plan, cleanup=orphan),
            plan,
            "c" * 64,
            BOOT_B,
        ),
        "managed_cleanup.matching_nvml_pids",
    )


def test_fetch_binding():
    wrapper = wrapper_plan()
    managed = managed_plan()
    runtime = runtime_row(managed)
    raw, attachments, primary = MODULE.validate_fetch(
        fetched(wrapper),
        wrapper,
        runtime,
        managed,
    )
    assert MODULE.parse_json(raw, "test")["schema"] == (
        MODULE.ROLE_CONFIG["cuda_monolithic"]["result_schema"]
    )
    assert attachments == []
    assert primary["path"] == wrapper["remote_output_path"]

    forged = fetched(wrapper)
    decoded = MODULE.parse_json(
        base64.b64decode(forged["primary"]["content_base64"]),
        "test",
    )
    decoded["producer_sha256"] = "e" * 64
    changed = MODULE.canonical_bytes(decoded)
    forged["primary"]["content_base64"] = base64.b64encode(changed).decode("ascii")
    forged["primary"]["bytes"] = len(changed)
    forged["primary"]["stat"]["size"] = len(changed)
    forged["primary"]["sha256"] = MODULE.hashlib.sha256(changed).hexdigest()
    expect_error(
        "forged producer",
        lambda: MODULE.validate_fetch(forged, wrapper, runtime, managed),
        "remote_result.producer",
    )

    live = fetched(wrapper)
    live["nvml_compute_pids"] = [202]
    expect_error(
        "nvml process live",
        lambda: MODULE.validate_fetch(live, wrapper, runtime, managed),
        "E_FETCH_NVML_NOT_EMPTY",
    )

    missing = fetched(wrapper)
    missing["processes_absent"] = [{"pid": 999, "start_ticks": 1}]
    expect_error(
        "producer not proven absent",
        lambda: MODULE.validate_fetch(missing, wrapper, runtime, managed),
        "E_FETCH_PRODUCER_ABSENT",
    )


def test_joint_binding():
    wrapper = wrapper_plan("joint_phone_cuda")
    MODULE.validate_plan(wrapper)
    managed = managed_plan("joint_phone_cuda")
    MODULE.validate_managed_plan(
        FakeLauncher(managed),
        b"{}",
        "c" * 64,
        wrapper,
    )
    runtime = runtime_row(managed)
    raw, attachments, unused_primary = MODULE.validate_fetch(
        fetched(wrapper),
        wrapper,
        runtime,
        managed,
    )
    del unused_primary
    assert attachments == []
    result = MODULE.parse_json(raw, "joint")
    assert result["schema"] == (
        "s39-cp0-r1-v24-joint-phone-cuda-raw-v1"
    )
    assert "cuda_route" not in MODULE.ROLE_CONFIG

    missing = managed_plan("joint_phone_cuda")
    capture_path = wrapper["joint_bindings"]["capture_plan"]["path"]
    missing["components"] = [
        item
        for item in missing["components"]
        if item["path"] != capture_path
    ]
    expect_error(
        "capture plan component",
        lambda: MODULE.validate_managed_plan(
            FakeLauncher(missing),
            b"{}",
            "c" * 64,
            wrapper,
        ),
        "E_JOINT_COMPONENT: capture_plan",
    )

    forged = remote_result(wrapper)
    forged["subproducer_bindings"]["phone"]["launch_plan_sha256"] = "0" * 64
    expect_error(
        "phone launch plan",
        lambda: MODULE.validate_fetch(
            fetched(wrapper, forged),
            wrapper,
            runtime,
            managed,
        ),
        "remote_result.phone_launch_plan",
    )


def test_sequence():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first, first_binding = bound_receipt(
            "cuda_monolithic",
            10,
            20,
            root,
            100,
        )
        second, second_binding = bound_receipt(
            "joint_phone_cuda",
            20,
            30,
            root,
            300,
        )
        MODULE.validate_sequence(
            [first, second],
            [first_binding, second_binding],
        )

        evidence_path = (root / "evidence.bin").resolve()
        evidence_path.write_bytes(b"evidence")
        evidence_artifact = MODULE.artifact_from_raw(
            evidence_path,
            b"evidence",
        )
        with_evidence = copy.deepcopy(first)
        with_evidence["remote_evidence_artifacts"] = [{
            **evidence_artifact,
            "path": "/remote/evidence.bin",
        }]
        with_evidence["local_evidence_artifacts"] = [{
            "local": evidence_artifact,
            "remote_path": "/remote/evidence.bin",
        }]
        MODULE.validate_receipt(with_evidence, *first_binding)
        evidence_path.write_bytes(b"EVIDENCE")
        expect_error(
            "changed local evidence",
            lambda: MODULE.validate_receipt(with_evidence, *first_binding),
            "receipt.local_evidence[0].disk",
        )

        overlap = copy.deepcopy(first)
        overlap["completed_ns"] = 25
        overlap["cleanup"]["observed_ns"] = 25
        overlap["fetch_transport_process"]["completed_ns"] = 24
        expect_error(
            "overlap",
            lambda: MODULE.validate_sequence(
                [overlap, second],
                [first_binding, second_binding],
            ),
            "E_SEQUENCE_OVERLAP",
        )

        bad = copy.deepcopy(first)
        bad["wrapper_plan_sha256"] = "0" * 64
        expect_error(
            "wrapper splice",
            lambda: MODULE.validate_receipt(bad, *first_binding),
            "receipt.wrapper_plan_sha256",
        )

        bad = copy.deepcopy(first)
        bad["managed_plan_sha256"] = "0" * 64
        bad["execution_transport_process"]["plan_sha256"] = "0" * 64
        expect_error(
            "managed splice",
            lambda: MODULE.validate_receipt(bad, *first_binding),
            "receipt.managed_plan_sha256",
        )

        bad = copy.deepcopy(first)
        bad["local_result_artifact"]["path"] = str(
            (root / "missing.json").resolve()
        )
        expect_error(
            "missing local result",
            lambda: MODULE.validate_receipt(bad, *first_binding),
            "E_LOCAL_ARTIFACT",
        )

        bad = copy.deepcopy(first)
        bad["remote_producer_process"]["remote_observed_ns"] = 101
        expect_error(
            "clock substitution",
            lambda: MODULE.validate_receipt(bad, *first_binding),
            "E_RECEIPT_REMOTE_CLOCK_ORDER",
        )


def test_listener():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert not MODULE.listener_absent("127.0.0.1", port)
    assert MODULE.listener_absent("127.0.0.1", port)


def test_timeout_cleanup():
    with tempfile.TemporaryDirectory() as directory:
        helper = Path(directory) / "sleep.py"
        helper.write_text(
            "import time\ntime.sleep(60)\n",
            encoding="ascii",
        )
        expect_error(
            "managed timeout",
            lambda: MODULE.run_managed(
                [sys.executable, "-I", str(helper)],
                1,
            ),
            "E_MANAGED_TIMEOUT",
        )


def test_remote_helper_compiles():
    compile(MODULE.REMOTE_FETCH_SOURCE, "<remote-fetch>", "exec")


def main():
    tests = [
        test_plan,
        test_managed_binding,
        test_real_launcher_component_shape,
        test_launcher_rows,
        test_fetch_binding,
        test_joint_binding,
        test_sequence,
        test_listener,
        test_timeout_cleanup,
        test_remote_helper_compiles,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"RESULT {len(tests)}/{len(tests)} PASS")


if __name__ == "__main__":
    main()
