#!/usr/bin/env python3

from __future__ import annotations

import copy
import base64
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
EXECUTORS = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(EXECUTORS) not in sys.path:
    sys.path.insert(0, str(EXECUTORS))

from executor_bundle import ISOLATED_LAUNCHER, build_executor_bundle
from desktop_gateway import SERVING_ENVELOPE, parse_desktop_config
from phone_gateway import canonical_bytes
from readiness_v23 import local_stat
from test_phone_observer import (
    OP12_BOOT,
    OP15_BOOT,
    identity as phone_identity,
    snapshot,
)
from validate_executor_evidence import validate_executor_bundle


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_bytes(value))


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_bytes(b"".join(canonical_bytes(value) for value in values))


def phone_placement(
    phone: str,
    layer_start: int,
    layer_end: int,
) -> dict:
    pid = 1001 if phone == "op15" else 2001
    boot_id = OP15_BOOT if phone == "op15" else OP12_BOOT
    compute = {"MUL_MAT": {"HTP0": 1}}
    session = {
        "compute_by_op_and_buffer": compute,
        "device_boot_id": boot_id,
        "expected_backend": "HTP0",
        "layer_end": layer_end,
        "layer_start": layer_start,
        "missing_buffer_compute_nodes": 0,
        "n_layer": 40,
        "placement_status": "SCHEDULED_PLACEMENT_OK",
        "proto_version": 2,
        "reset_applied": False,
        "schema": "ls-stagenet-session-v2",
        "session_end": "STOP",
        "session_id": 1,
        "steps_session": 8,
        "steps_total": 8,
        "worker_boot_nonce": "worker-nonce",
        "worker_pid": pid,
    }
    placement = {
        "compute_by_buffer_type": {"HTP0": 1},
        "compute_by_op": {"MUL_MAT": 1},
        "compute_by_op_and_buffer": compute,
        "compute_nodes": 1,
        "copy_by_buffer_type": {},
        "copy_nodes": 0,
        "layer_end": layer_end,
        "layer_start": layer_start,
        "metadata_nodes": 0,
        "missing_buffer_compute_nodes": 0,
        "mode": "stagenet" if phone == "op15" else "tailv3",
        "n_layer": 40,
        "pid": pid,
        "role": "phone_stage",
        "run_rc": 0,
        "schema": "layersplit-scheduled-placement-v2",
        "status": "SCHEDULED_PLACEMENT_OK",
    }
    lines = [
        "SESSIONCERT " + canonical_bytes(session).decode("ascii").strip(),
        "PLACEMENTCERT "
        + canonical_bytes(placement).decode("ascii").strip(),
    ]
    return {
        "certificate_lines": lines,
        "certificate_lines_sha256": hashlib.sha256(
            "".join(line + "\n" for line in lines).encode("ascii")
        ).hexdigest(),
        "placement": placement,
        "session": session,
    }


def htp_snapshot(name: str, started_ns: int, completed_ns: int) -> dict:
    value = snapshot(name, started_ns, completed_ns, 1)
    stage = next(
        process
        for process in value["processes"]
        if process["kind"] in ("STAGE_HEAD", "STAGE_TAIL")
    )
    stage["backend"] = "HTP0"
    device_index = stage["argv"].index("--devices") + 1
    stage["argv"][device_index] = "HTP0"
    from validate_phone_observer import argv_sha256

    stage["cmdline_sha256"] = argv_sha256(stage["argv"])
    return value


def execute_command(
    executor_id: str,
    prompt_tokens: list[int],
) -> dict:
    return {
        "command_id": 1,
        "controller_epoch": 1,
        "executor_id": executor_id,
        "executor_instance_id": f"instance-{executor_id.lower()}",
        "kind": 0,
        "max_output_tokens": 1,
        "model_id": "model-a",
        "request": {
            "committed_output_tokens": [],
            "model_id": "model-a",
            "owner_id": executor_id,
            "ownership_epoch": 1,
            "position": len(prompt_tokens),
            "prompt_tokens": prompt_tokens,
            "publication_index": 0,
            "request_id": "request-a",
            "state": 1,
        },
        "request_id": "request-a",
        "schema": "llama-server-warm-tier-command-v3",
        "total_output_tokens": 8,
    }


def model_command(
    executor_id: str,
    command_id: int,
    model_id: str,
    kind: int,
) -> dict:
    return {
        "command_id": command_id,
        "controller_epoch": 1,
        "executor_id": executor_id,
        "executor_instance_id": f"instance-{executor_id.lower()}",
        "kind": kind,
        "max_output_tokens": 0,
        "model_id": model_id,
        "request": {
            "committed_output_tokens": [],
            "model_id": "",
            "owner_id": "",
            "ownership_epoch": 0,
            "position": 0,
            "prompt_tokens": [],
            "publication_index": 0,
            "request_id": "",
            "state": 0,
        },
        "request_id": "",
        "schema": "llama-server-warm-tier-command-v3",
        "total_output_tokens": 0,
    }


class EvidenceFixture:
    def __init__(self, test: unittest.TestCase):
        self.temporary = tempfile.TemporaryDirectory()
        test.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stdout = self.root / "gateway.stdout"
        self.stderr = self.root / "gateway.stderr"
        self.stdout.write_bytes(b"")
        self.stderr.write_bytes(b"")
        self.socket = self.root / "gateway.sock"
        bundle = build_executor_bundle(self.root / "executor-bundle")
        self.bundle_manifest = Path(bundle["manifest_path"])
        self.python = self.root / "python-captured"
        self.python.write_bytes(b"fixture python executable\n")
        self.python.chmod(0o755)
        evidence_root = self.root / "evidence-bundle"
        evidence_root.mkdir()
        self.evidence_manifest = evidence_root / "MANIFEST.json"
        write_json(self.evidence_manifest, {
            "files": [],
            "python_flags": ["-I", "-S", "-B"],
            "schema": "s40-evidence-bundle-v2",
        })
        for directory in (
            "captured",
            "captured-runtime/bin",
            "captured-runtime/lib",
            "cuda-cache",
            "run-home",
            "run-tmp",
        ):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        self.gateway_argv = self.root / "gateway-argv.json"
        self.transport_descriptor = self.root / "transport-descriptor.json"
        self.gateway_source = None

    def configure_transport(
        self,
        kind: str,
        executor_id: str,
        config_path: Path,
        command_path: Path,
        wire_path: Path | None = None,
        route_path: Path | None = None,
    ) -> None:
        source_name = (
            "desktop_gateway.py" if kind == "desktop" else "phone_gateway.py"
        )
        executor_instance_id = f"instance-{executor_id.lower()}"
        gateway_pid = 2001 if kind == "desktop" else 2002
        gateway_start_time_ticks = 3001 if kind == "desktop" else 3002
        runtime_config = self.root / "runtime-config.json"
        write_json(runtime_config, {
            "c3_profile_lock_sha256": None,
            "configuration": "C1_GPU_ONLY_OPTIMIZED",
            "evidence_root_sha256": "8" * 64,
            "event_log_path": str(self.root / "events.jsonl"),
            "executors": [{
                "credits": 1,
                "execute_concurrency": 1,
                "executor_id": executor_id,
                "executor_instance_id": executor_instance_id,
                "expected_peer_pid": gateway_pid,
                "expected_peer_start_time_ticks":
                    gateway_start_time_ticks,
                "order": 0,
                "output_limit_bytes": 4096,
                "queue_capacity": 1,
                "role": "CPU" if kind == "desktop" else "PHONE",
                "socket_path": str(self.socket),
                "timeout_ms": 1000,
                "transport": "UNIX_SOCKET",
            }],
            "initial_models": [],
            "promotion_enabled": True,
            "run_id": "run-a",
            "runtime_plan_sha256": "7" * 64,
            "schema": "llama-server-warm-tier-runtime-v4",
        })
        runtime_config_sha256 = hashlib.sha256(
            runtime_config.read_bytes()).hexdigest()
        runtime_stat = runtime_config.stat(follow_symlinks=False)
        controller_executable = self.root / "llama-server-controller"
        controller_executable.write_bytes(b"fixture controller\n")
        controller_executable.chmod(0o755)
        controller_identity = self.root / "controller-identity-lock.json"
        controller_pid = 3001
        controller_start_time_ticks = 4001
        controller_uid = 1000
        controller_gid = 1001
        controller_record = {
            "controller_executable_path": str(controller_executable),
            "controller_executable_sha256": hashlib.sha256(
                controller_executable.read_bytes()
            ).hexdigest(),
            "controller_gid": controller_gid,
            "controller_pid": controller_pid,
            "controller_start_time_ticks": controller_start_time_ticks,
            "controller_uid": controller_uid,
            "host_boot_id": "fixture-host-boot",
            "run_id": "run-a",
            "runtime_config_device": runtime_stat.st_dev,
            "runtime_config_inode": runtime_stat.st_ino,
            "runtime_config_path": str(runtime_config),
            "runtime_config_sha256": runtime_config_sha256,
            "schema": "s40-controller-identity-lock-v1",
        }
        write_json(controller_identity, controller_record)
        controller_identity_stat = controller_identity.stat(
            follow_symlinks=False)
        controller_identity_sha256 = hashlib.sha256(
            controller_identity.read_bytes()).hexdigest()
        controller_binding = self.root / "controller-binding.json"
        binding_record = {
            "authenticated_ns": 3,
            "controller_executable_path": str(controller_executable),
            "controller_executable_sha256":
                controller_record["controller_executable_sha256"],
            "controller_gid": controller_gid,
            "controller_identity_device": controller_identity_stat.st_dev,
            "controller_identity_inode": controller_identity_stat.st_ino,
            "controller_identity_path": str(controller_identity),
            "controller_identity_sha256": controller_identity_sha256,
            "controller_pid": controller_pid,
            "controller_start_time_ticks": controller_start_time_ticks,
            "controller_uid": controller_uid,
            "executor_id": executor_id,
            "executor_instance_id": executor_instance_id,
            "gateway_pid": gateway_pid,
            "gateway_start_time_ticks": gateway_start_time_ticks,
            "host_boot_id": "fixture-host-boot",
            "peer_gid": controller_gid,
            "peer_pid": controller_pid,
            "peer_uid": controller_uid,
            "run_id": "run-a",
            "runtime_config_device": runtime_stat.st_dev,
            "runtime_config_inode": runtime_stat.st_ino,
            "runtime_config_path": str(runtime_config),
            "runtime_config_sha256": runtime_config_sha256,
            "schema": "s40-executor-controller-binding-v1",
        }
        write_json(controller_binding, binding_record)
        controller_binding_stat = controller_binding.stat(
            follow_symlinks=False)
        command_rows = [
            json.loads(line)
            for line in command_path.read_text(encoding="ascii").splitlines()
        ]
        command_rows[0]["executor_instance_id"] = executor_instance_id
        command_rows[0]["gateway_pid"] = gateway_pid
        command_rows[0][
            "gateway_start_time_ticks"] = gateway_start_time_ticks
        command_rows[0]["runtime_config_device"] = runtime_stat.st_dev
        command_rows[0]["runtime_config_inode"] = runtime_stat.st_ino
        command_rows[0]["runtime_config_path"] = str(runtime_config)
        command_rows[0]["runtime_config_sha256"] = runtime_config_sha256
        for row in command_rows[1:]:
            row["executor_instance_id"] = executor_instance_id
            row["runtime_config_sha256"] = runtime_config_sha256
            if isinstance(row.get("command"), dict):
                row["command"]["executor_instance_id"] = executor_instance_id
            if isinstance(row.get("result"), dict):
                row["result"]["executor_instance_id"] = executor_instance_id
        write_jsonl(command_path, command_rows)
        self.gateway_source = self.bundle_manifest.parent / source_name
        argv = [
            str(self.python),
            "-I",
            "-S",
            "-B",
            "-c",
            ISOLATED_LAUNCHER,
            str(self.bundle_manifest.parent),
            str(self.gateway_source),
            "--socket",
            str(self.socket),
            "--evidence",
            str(command_path),
            "--config" if kind == "desktop" else "--route-config",
            str(config_path),
            "--run-id",
            "run-a",
            "--runtime-config",
            str(runtime_config),
            "--controller-identity",
            str(controller_identity),
            "--controller-binding-evidence",
            str(controller_binding),
            "--executor-instance-id",
            executor_instance_id,
        ]
        if kind == "phone":
            assert wire_path is not None and route_path is not None
            argv.extend([
                "--wire-evidence",
                str(wire_path),
                "--route-evidence",
                str(route_path),
            ])
        write_json(
            self.gateway_argv,
            {"argv": argv, "schema": "s40-gateway-argv-v4"},
        )
        nvidia_smi = self.root / "nvidia-smi"
        if not nvidia_smi.exists():
            nvidia_smi.write_bytes(b"fake nvidia-smi\n")
            nvidia_smi.chmod(0o755)
        launch_environment = {
            "CUDA_CACHE_PATH": str(self.root / "cuda-cache"),
            "CUDA_VISIBLE_DEVICES": "GPU-fixture",
            "HOME": str(self.root / "run-home"),
            "LANG": "C",
            "LC_ALL": "C",
            "LD_LIBRARY_PATH": str(self.root / "captured-runtime" / "lib"),
            "LLAMA_SERVER_WARM_TIER_CONFIG": str(runtime_config),
            "NVIDIA_VISIBLE_DEVICES": "GPU-fixture",
            "PATH":
                f"{self.root / 'captured-runtime' / 'bin'}:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "S40_EVIDENCE_BUNDLE": "1",
            "S40_EVIDENCE_BUNDLE_MANIFEST":
                str(self.evidence_manifest),
            "S40_EVIDENCE_BUNDLE_SHA256":
                hashlib.sha256(
                    self.evidence_manifest.read_bytes()).hexdigest(),
            "S40_EXECUTOR_BUNDLE": "1",
            "S40_EXECUTOR_BUNDLE_MANIFEST": str(self.bundle_manifest),
            "S40_EXECUTOR_BUNDLE_SHA256":
                hashlib.sha256(
                    self.bundle_manifest.read_bytes()).hexdigest(),
            "S40_NVIDIA_SMI_PATH": str(nvidia_smi),
            "S40_NVIDIA_SMI_SHA256":
                hashlib.sha256(nvidia_smi.read_bytes()).hexdigest(),
            "TMPDIR": str(self.root / "run-tmp"),
            "TZ": "UTC",
        }
        if kind == "desktop":
            launch_environment[
                "LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE"
            ] = str(self.root / "warm-tier-internal.token")

        def executed_file(argv_index: int, source: Path) -> dict:
            metadata = source.stat(follow_symlinks=False)
            target = (
                self.root / "captured"
                / f"gateway-{argv_index}-{source.name}"
            )
            target.write_bytes(source.read_bytes())
            return {
                "argv_index": argv_index,
                "bytes": target.stat().st_size,
                "captured_path": str(target.relative_to(self.root)),
                "executed_path": str(source),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "source_ctime_ns": metadata.st_ctime_ns,
                "source_device": metadata.st_dev,
                "source_inode": metadata.st_ino,
                "source_mtime_ns": metadata.st_mtime_ns,
                "source_size": metadata.st_size,
            }

        gateway_executed_files = [
            executed_file(0, self.python),
            executed_file(7, self.gateway_source),
            executed_file(argv.index(str(config_path)), config_path),
        ]
        cmdline = b"".join(
            argument.encode("ascii") + b"\x00" for argument in argv)
        executable_stat = self.python.stat(follow_symlinks=False)

        def gateway_identity(observed_ns: int) -> dict:
            return {
                "cmdline_base64": base64.b64encode(cmdline).decode("ascii"),
                "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
                "executable_ctime_ns": executable_stat.st_ctime_ns,
                "executable_device": executable_stat.st_dev,
                "executable_inode": executable_stat.st_ino,
                "executable_mtime_ns": executable_stat.st_mtime_ns,
                "executable_path": str(self.python),
                "executable_sha256":
                    hashlib.sha256(self.python.read_bytes()).hexdigest(),
                "executable_size": executable_stat.st_size,
                "gateway_pid": gateway_pid,
                "gateway_start_time_ticks": gateway_start_time_ticks,
                "observed_ns": observed_ns,
            }

        write_json(
            self.transport_descriptor,
            {
                "controller_binding_device":
                    controller_binding_stat.st_dev,
                "controller_binding_inode":
                    controller_binding_stat.st_ino,
                "controller_binding_path": str(controller_binding),
                "controller_binding_sha256": hashlib.sha256(
                    controller_binding.read_bytes()
                ).hexdigest(),
                "controller_executable_path": str(controller_executable),
                "controller_executable_sha256":
                    controller_record["controller_executable_sha256"],
                "controller_gid": controller_gid,
                "controller_identity_device":
                    controller_identity_stat.st_dev,
                "controller_identity_inode":
                    controller_identity_stat.st_ino,
                "controller_identity_path": str(controller_identity),
                "controller_identity_published_ns": 3,
                "controller_identity_sha256": controller_identity_sha256,
                "controller_pid": controller_pid,
                "controller_start_time_ticks": controller_start_time_ticks,
                "controller_uid": controller_uid,
                "executor_bundle_manifest_sha256":
                    hashlib.sha256(
                        self.bundle_manifest.read_bytes()
                    ).hexdigest(),
                "executor_id": executor_id,
                "executor_instance_id": executor_instance_id,
                "gateway_argv_sha256":
                    hashlib.sha256(
                        self.gateway_argv.read_bytes()
                    ).hexdigest(),
                "gateway_config_sha256":
                    hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "gateway_environment": launch_environment,
                "gateway_executed_files": gateway_executed_files,
                "gateway_pid": gateway_pid,
                "gateway_post_auth_identity": gateway_identity(4),
                "gateway_prepublication_identity": gateway_identity(1),
                "gateway_source_sha256":
                    hashlib.sha256(
                        self.gateway_source.read_bytes()
                    ).hexdigest(),
                "gateway_start_time_ticks": gateway_start_time_ticks,
                "host_boot_id": "fixture-host-boot",
                "identity_captured_ns": 1,
                "runtime_config_device": runtime_stat.st_dev,
                "runtime_config_inode": runtime_stat.st_ino,
                "runtime_config_path": str(runtime_config),
                "runtime_config_published_ns": 2,
                "runtime_config_sha256": runtime_config_sha256,
                "schema": "s40-executor-transport-descriptor-v4",
                "socket_path": str(self.socket),
                "transport": "UNIX_SOCKET",
            },
        )
        self.controller_binding = controller_binding
        self.controller_executable = controller_executable
        self.controller_identity = controller_identity
        self.runtime_config = runtime_config

    def readiness(
        self,
        model_id: str,
        model_path: Path,
        model_sha256: str,
    ) -> tuple[Path, str, Path, str]:
        route_lock = "b" * 64
        certificate = {
            "artifacts": [{
                "bytes": model_path.stat().st_size,
                "endpoint": "cuda",
                "path": str(model_path),
                "sha256": model_sha256,
                "stat": local_stat(model_path),
            }],
            "completed_ns": 2,
            "model_id": model_id,
            "phase": "A_ONLY",
            "route_lock_sha256": route_lock,
            "schema": "s39-cp0-r1-artifact-snapshot-v2.3",
            "slot": "A",
            "started_ns": 1,
        }
        certificate_path = self.root / "artifact-certificate.json"
        write_json(certificate_path, certificate)
        certificate_sha256 = hashlib.sha256(
            certificate_path.read_bytes()
        ).hexdigest()
        lock = {
            "artifact_snapshot_sha256": certificate_sha256,
            "event_ns": 3,
            "phase": "A_ONLY",
            "phase_id": "phase-a",
            "schema": "s39-cp0-r1-readiness-lock-v2.3",
            "v2_2_phase_lock_sha256": "c" * 64,
        }
        lock_path = self.root / "readiness-lock.json"
        write_json(lock_path, lock)
        lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        return certificate_path, certificate_sha256, lock_path, lock_sha256


class ValidateExecutorEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.authority = mock.patch(
            "desktop_gateway.derive_route_qualification",
            side_effect=lambda _value, **kwargs: {
                "a_chain_phase_id": None,
                "bundle_manifest_sha256": "d" * 64,
                "model_id": kwargs["model_id"],
                "phase": kwargs["phase"],
                "phase_id": "phase-a",
                "phase_lock_sha256": "c" * 64,
                "schema": "s40-route-qualification-derived-v1",
                "scope": "QUALIFIED_ROUTE",
                "status": "MODEL_A_QUALIFICATION_PASS",
                "v2_2_result_sha256": "e" * 64,
            },
        )
        self.authority.start()
        self.addCleanup(self.authority.stop)
        self.phone_authority = mock.patch(
            "phone_gateway.derive_route_qualification",
            side_effect=lambda _value, **kwargs: {
                "a_chain_phase_id": None,
                "bundle_manifest_sha256": "d" * 64,
                "model_id": kwargs["model_id"],
                "phase": kwargs["phase"],
                "phase_id": "phase-a",
                "phase_lock_sha256": "c" * 64,
                "schema": "s40-route-qualification-derived-v1",
                "scope": "QUALIFIED_ROUTE",
                "status": "MODEL_A_QUALIFICATION_PASS",
                "v2_2_result_sha256": "e" * 64,
            },
        )
        self.phone_authority.start()
        self.addCleanup(self.phone_authority.stop)

    def desktop_fixture(self) -> tuple[EvidenceFixture, dict, list[dict]]:
        fixture = EvidenceFixture(self)
        model = fixture.root / "model.gguf"
        model.write_bytes(b"model")
        model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
        nvidia_smi = fixture.root / "nvidia-smi"
        nvidia_smi.write_bytes(b"fake nvidia-smi\n")
        nvidia_smi.chmod(0o755)
        nvidia_smi_identity = {
            "bytes": nvidia_smi.stat().st_size,
            "path": str(nvidia_smi),
            "sha256": hashlib.sha256(
                nvidia_smi.read_bytes()
            ).hexdigest(),
        }
        certificate, certificate_sha, lock, lock_sha = fixture.readiness(
            "model-a",
            model,
            model_sha256,
        )
        slot_path = fixture.root / "slots-a"
        slot_path.mkdir()
        child = fixture.root / "llama-server"
        child.write_bytes(b"fake llama-server\n")
        child.chmod(0o755)
        child_identity = {
            "bytes": child.stat().st_size,
            "path": str(child),
            "sha256": hashlib.sha256(child.read_bytes()).hexdigest(),
        }
        child_argv_template = [
            str(child),
            "-m",
            str(model),
            "-ngl",
            "0",
            "--slot-save-path",
            str(slot_path),
            "--port",
            "{PORT}",
            "--ctx-size",
            "4096",
            "--batch-size",
            "2048",
            "--ubatch-size",
            "512",
            "--parallel",
            "8",
            "--split-mode",
            "none",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--flash-attn",
            "on",
            "--cont-batching",
        ]
        config = {
            "base_url": "http://127.0.0.1:8080",
            "cache_regime": "WARM_CACHE",
            "executor_id": "CPU",
            "mode": "SINGLE_ACTIVE",
            "nvidia_smi": nvidia_smi_identity,
            "profile_lock_sha256": None,
            "role": "CPU",
            "routes": [{
                "artifact_certificate_path": str(certificate),
                "artifact_certificate_sha256": certificate_sha,
                "backend": "CPU",
                "child_argv_template": child_argv_template,
                "child_executable": child_identity,
                "device_memory_total_mib": 0,
                "device_name": "NONE",
                "device_uuid": "NONE",
                "host_boot_id": "desktop-boot",
                "minimum_free_device_memory_mib": 0,
                "model_id": "model-a",
                "model_path": str(model),
                "model_sha256": model_sha256,
                "native_model_id": "model-a-cpu",
                "n_gpu_layers": "0",
                "phase": "A_ONLY",
                "phase_lock_sha256": "c" * 64,
                "qualification": {"fixture": "authority"},
                "readiness_lock_path": str(lock),
                "readiness_lock_sha256": lock_sha,
                "route_lock_sha256": "b" * 64,
                "slot": "A",
                "slot_save_path": str(slot_path),
                "slots": list(range(8)),
            }],
            "schema": "s40-desktop-executor-config-v4",
        }
        config_path = fixture.root / "desktop-config.json"
        write_json(config_path, config)
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        probe = {
            "argv": [
                "8081" if item == "{PORT}" else item
                for item in child_argv_template
            ],
            "artifact_certificate_sha256": certificate_sha,
            "backend": "CPU",
            "child_argv_sha256": hashlib.sha256(
                b"\0".join(
                    (
                        "8081" if item == "{PORT}" else item
                    ).encode("utf-8")
                    for item in child_argv_template
                )
                + b"\0"
            ).hexdigest(),
            "child_executable": child_identity,
            "device_memory_free_mib": None,
            "device_memory_total_mib": None,
            "device_name": None,
            "device_uuid": None,
            "host_boot_id": "desktop-boot",
            "minimum_free_device_memory_mib": 0,
            "model_path": str(model),
            "model_sha256": model_sha256,
            "native_model_id": "model-a-cpu",
            "nvidia_smi": nvidia_smi_identity,
            "n_gpu_layers": "0",
            "process_id": 101,
            "process_start_ticks": 10001,
            "port": 8081,
            "qualification": {
                "a_chain_phase_id": None,
                "bundle_manifest_sha256": "d" * 64,
                "model_id": "model-a",
                "phase": "A_ONLY",
                "phase_id": "phase-a",
                "phase_lock_sha256": "c" * 64,
                "schema": "s40-route-qualification-derived-v1",
                "scope": "QUALIFIED_ROUTE",
                "status": "MODEL_A_QUALIFICATION_PASS",
                "v2_2_result_sha256": "e" * 64,
            },
            "readiness_lock_sha256": lock_sha,
            "readiness_phase_id": "phase-a",
            "schema": "s40-desktop-runtime-probe-v3",
            "serving_envelope": SERVING_ENVELOPE,
            "slot_save_path": str(slot_path),
        }
        rows = [{
            "cache_regime": "WARM_CACHE",
            "executor_config_sha256": config_sha256,
            "executor_id": "CPU",
            "executor_instance_id": "instance-cpu",
            "gateway_pid": 2001,
            "gateway_start_time_ticks": 3001,
            "mode": "SINGLE_ACTIVE",
            "profile_lock_sha256": None,
            "role": "CPU",
            "routes": [],
            "run_id": "run-a",
            "runtime_config_device": 11,
            "runtime_config_inode": 12,
            "runtime_config_path": str(fixture.root / "runtime-config.json"),
            "runtime_config_sha256": "9" * 64,
            "schema": "s40-desktop-startup-evidence-v2",
        }, {
            "command": model_command("CPU", 1, "model-a", 3),
            "command_id": 1,
            "completed_ns": 9,
            "controller_epoch": 1,
            "durability": "fsync_each_record",
            "execute": None,
            "executor_id": "CPU",
            "executor_instance_id": "instance-cpu",
            "kind": 3,
            "lifecycle": {
                "cache_control": {
                    "argv": [
                        sys.executable,
                        "-B",
                        str(EXECUTORS / "cache_control_runner.py"),
                        "--regime",
                        "WARM_CACHE",
                        "--model",
                        str(model),
                    ],
                    "completed_ns": 8,
                    "exit_code": 0,
                    "output": {
                        "after": {
                            "bytes": model.stat().st_size,
                            "elapsed_ns": 1,
                            "method": "COMPLETE_SEQUENTIAL_READ",
                            "page_size": 4096,
                            "pages": 1,
                            "resident_pages": 1,
                            "resident_ppm": 1_000_000,
                        },
                        "before": {
                            "bytes": model.stat().st_size,
                            "page_size": 4096,
                            "pages": 1,
                            "resident_pages": 0,
                            "resident_ppm": 0,
                        },
                        "completed_ns": 7,
                        "comparison": "at_least",
                        "limit_ppm": 950_000,
                        "model_path": str(model),
                        "model_stat": local_stat(model),
                        "regime": "WARM_CACHE",
                        "schema": "s40-cache-control-result-v1",
                        "started_ns": 6,
                        "success": True,
                    },
                    "schema": "s40-cache-control-evidence-v1",
                    "started_ns": 5,
                    "stderr": "",
                    "success": True,
                },
                "instance_id": "101:8081:1",
                "operation": "LOAD",
                "preflight": {
                    "models": [{
                        "logical_model_id": "model-a",
                        "native_model_id": "model-a-cpu",
                        "status": "unloaded",
                    }],
                    "schema": "s40-desktop-empty-router-preflight-v1",
                },
                "runtime_probe": probe,
            },
            "model_id": "model-a",
            "request_id": None,
            "result": {
                "command_id": 1,
                "controller_epoch": 1,
                "detail": "desktop route loaded",
                "executor_id": "CPU",
                "executor_instance_id": "instance-cpu",
                "has_replay_snapshot": False,
                "kind": 3,
                "model_id": "model-a",
                "publications": [],
                "replay_snapshot": None,
                "request_complete": False,
                "request_id": "",
                "schema": "llama-server-warm-tier-result-v2",
                "success": True,
            },
            "role": "CPU",
            "run_id": "run-a",
            "runtime_config_sha256": "9" * 64,
            "schema": "s40-desktop-command-evidence-v4",
            "started_ns": 4,
            "success": True,
        }, {
            "command": {
                **execute_command("CPU", [1, 2, 3]),
                "command_id": 2,
            },
            "command_id": 2,
            "completed_ns": 20,
            "controller_epoch": 1,
            "durability": "fsync_each_record",
            "execute": {
                "execute_quantum_tokens": 1,
                "full_history_per_token_reprefill": False,
                "initial_history_replay": True,
                "instance_id": "101:8081:1",
                "publication_count": 1,
                "resident_session_reused": False,
                "runtime_probe": probe,
                "sampler": {
                    "seed": 0,
                    "temperature": 0.0,
                    "type": "greedy",
                },
                "slot_id": 0,
                "tokens_cached": 0,
                "tokens_evaluated": 4,
            },
            "executor_id": "CPU",
            "executor_instance_id": "instance-cpu",
            "kind": 0,
            "lifecycle": None,
            "model_id": "model-a",
            "request_id": "request-a",
            "result": {
                "command_id": 2,
                "controller_epoch": 1,
                "detail": "desktop route executed",
                "executor_id": "CPU",
                "executor_instance_id": "instance-cpu",
                "has_replay_snapshot": False,
                "kind": 0,
                "model_id": "model-a",
                "publications": [{
                    "owner_id": "CPU",
                    "ownership_epoch": 1,
                    "position": 3,
                    "publication_index": 0,
                    "token": 11,
                }],
                "replay_snapshot": None,
                "request_complete": False,
                "request_id": "request-a",
                "schema": "llama-server-warm-tier-result-v2",
                "success": True,
            },
            "role": "CPU",
            "run_id": "run-a",
            "runtime_config_sha256": "9" * 64,
            "schema": "s40-desktop-command-evidence-v4",
            "started_ns": 10,
            "success": True,
        }, {
            "completed_ns": 40,
            "executor_id": "CPU",
            "executor_instance_id": "instance-cpu",
            "initial_active_models": ["model-a"],
            "initial_busy_requests": [],
            "initial_request_sessions": [],
            "problems": [],
            "remaining_active_models": [],
            "remaining_request_sessions": [],
            "run_id": "run-a",
            "runtime_config_sha256": "9" * 64,
            "schema": "s40-desktop-cleanup-evidence-v2",
            "started_ns": 31,
            "success": True,
            "unloaded": [{
                "instance_id": "101:8081:1",
                "logical_model_id": "model-a",
                "native_model_id": "model-a-cpu",
                "process_exited": True,
                "process_id": 101,
                "process_start_ticks": 10001,
            }],
        }]
        fixture.config = config_path
        fixture.command = fixture.root / "desktop-command.jsonl"
        write_jsonl(fixture.command, rows)
        fixture.configure_transport(
            "desktop",
            "CPU",
            fixture.config,
            fixture.command,
        )
        rows = [
            json.loads(line)
            for line in fixture.command.read_text(encoding="ascii").splitlines()
        ]
        return fixture, config, rows

    def phone_fixture(self) -> tuple[EvidenceFixture, dict, list[dict], list[dict]]:
        fixture = EvidenceFixture(self)
        control_paths = {}
        for role in (
            "a6000_ssh_control",
            "executor_bundle_manifest",
            "python",
            "ssh_config",
        ):
            path = fixture.root / role
            path.write_text(role + "\n", encoding="ascii")
            if role in ("a6000_ssh_control", "python"):
                path.chmod(0o755)
            control_paths[role] = path
        manifest_sha256 = hashlib.sha256(
            control_paths["executor_bundle_manifest"].read_bytes()
        ).hexdigest()
        control = {
            "environment": {
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(fixture.root),
                "S40_EXECUTOR_BUNDLE": "1",
                "S40_EXECUTOR_BUNDLE_MANIFEST":
                    str(control_paths["executor_bundle_manifest"]),
                "S40_EXECUTOR_BUNDLE_SHA256": manifest_sha256,
            },
            "files": [
                {
                    "bytes": path.stat().st_size,
                    "path": str(path),
                    "role": role,
                    "sha256": hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest(),
                }
                for role, path in sorted(control_paths.items())
            ],
        }
        spec = {
            "a6000_identity": "6" * 64,
            "artifact_certificate_sha256": "4" * 64,
            "batch_knee": 8,
            "file_type": 7,
            "gather_us": 1000,
            "layer_end": 40,
            "layer_start": 0,
            "max_streams": 8,
            "model_id": "model-a",
            "model_sha256": "a" * 64,
            "n_batch": 64,
            "n_embd": 5120,
            "n_layer": 40,
            "n_ubatch": 64,
            "op12_boot_id": OP12_BOOT,
            "op12_shard_sha256": "2" * 64,
            "op15_boot_id": OP15_BOOT,
            "op15_shard_sha256": "1" * 64,
            "prefill_chunk": 64,
            "phase": "A_ONLY",
            "phase_lock_sha256": "c" * 64,
            "qualification": {"fixture": "authority"},
            "queue_depth": 64,
            "readiness_lock_sha256": "5" * 64,
            "readiness_phase_id": "phase-a",
            "relay_host": "10.0.0.15",
            "relay_port": 19090,
            "slot": "A",
            "worker_sha256": "3" * 64,
        }
        config = {
            "control": control,
            "executor_id": "PHONE",
            "routes": [spec],
            "schema": "s40-phone-route-config-v3",
        }
        config_path = fixture.root / "phone-config.json"
        write_json(config_path, config)
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        route_rows = [{
            "executor_id": "PHONE",
            "route_config_sha256": config_sha256,
            "routes": [{
                "artifact_certificate_sha256": "4" * 64,
                "batch_knee": 8,
                "model_id": "model-a",
                "n_batch": 64,
                "n_ubatch": 64,
                "phase_lock_sha256": "c" * 64,
                "qualification": {
                    "a_chain_phase_id": None,
                    "bundle_manifest_sha256": "d" * 64,
                    "model_id": "model-a",
                    "phase": "A_ONLY",
                    "phase_id": "phase-a",
                    "phase_lock_sha256": "c" * 64,
                    "schema": "s40-route-qualification-derived-v1",
                    "scope": "QUALIFIED_ROUTE",
                    "status": "MODEL_A_QUALIFICATION_PASS",
                    "v2_2_result_sha256": "e" * 64,
                },
                "readiness_lock_sha256": "5" * 64,
                "readiness_phase_id": "phase-a",
            }],
            "schema": "s40-phone-route-config-evidence-v1",
        }, {
            "a6000_identity": "6" * 64,
            "artifact_certificate_sha256": "4" * 64,
            "model_id": "model-a",
            "model_sha256": "a" * 64,
            "op12_boot_id": OP12_BOOT,
            "op12_shard_sha256": "2" * 64,
            "op15_boot_id": OP15_BOOT,
            "op15_shard_sha256": "1" * 64,
            "qualification_sha256": hashlib.sha256(
                canonical_bytes({"fixture": "authority"})
            ).hexdigest(),
            "readiness_lock_sha256": "5" * 64,
            "readiness_phase_id": "phase-a",
            "route_observation": {
                "direct_peer": phone_identity(1, 2, 1)["direct_peer"],
                "model_id": "model-a",
                "phones": {
                    "op12": htp_snapshot("op12", 1, 2),
                    "op15": htp_snapshot("op15", 3, 4),
                },
                "route_instance_id": "route-a",
                "schema": "s40-phone-route-observation-v1",
            },
            "route_instance_id": "route-a",
            "schema": "s40-phone-route-load-v3",
            "success": True,
            "worker_sha256": "3" * 64,
        }, {
            "model_id": "model-a",
            "placements": {
                "op12": phone_placement("op12", 30, 40),
                "op15": phone_placement("op15", 0, 30),
            },
            "route_instance_id": "route-a",
            "schema": "s40-phone-route-unload-v2",
            "success": True,
        }]
        wire_rows = [{
            "batch_index": 1,
            "batch_size": 1,
            "completed_ns": 20,
            "executor_id": "PHONE",
            "input_rows": [{
                "position": 3,
                "request_id": 1,
                "route_epoch": 1,
                "seq_id": 0,
                "token": 10,
            }],
            "model_id": "model-a",
            "output_rows": [{
                "position": 3,
                "request_id": 1,
                "route_epoch": 1,
                "seq_id": 0,
                "token": 11,
            }],
            "route_epoch": 1,
            "route_instance_id": "route-a",
            "schema": "s40-phone-wire-batch-v1",
            "started_ns": 10,
        }]
        command_rows = [{
            "configured_models": ["model-a"],
            "executor_id": "PHONE",
            "executor_instance_id": "instance-phone",
            "gateway_pid": 2002,
            "gateway_start_time_ticks": 3002,
            "initial_active_models": [],
            "route_config_sha256": config_sha256,
            "run_id": "run-a",
            "runtime_config_device": 11,
            "runtime_config_inode": 12,
            "runtime_config_path": str(fixture.root / "runtime-config.json"),
            "runtime_config_sha256": "9" * 64,
            "schema": "s40-phone-startup-evidence-v2",
        }, {
            "command": execute_command("PHONE", [1, 2, 3, 10]),
            "command_id": 1,
            "completed_ns": 30,
            "controller_epoch": 1,
            "durability": "fsync_each_record",
            "execute_quantum_tokens": 1,
            "executor_id": "PHONE",
            "executor_instance_id": "instance-phone",
            "full_history_per_token_reprefill": False,
            "initial_history_replay": True,
            "internal_request_id": 1,
            "kind": 0,
            "model_id": "model-a",
            "publication_count": 1,
            "request_id": "request-a",
            "resident_session_reused": False,
            "result": {
                "command_id": 1,
                "controller_epoch": 1,
                "detail": "phone route executed",
                "executor_id": "PHONE",
                "executor_instance_id": "instance-phone",
                "has_replay_snapshot": False,
                "kind": 0,
                "model_id": "model-a",
                "publications": [{
                    "owner_id": "PHONE",
                    "ownership_epoch": 1,
                    "position": 4,
                    "publication_index": 0,
                    "token": 11,
                }],
                "replay_snapshot": None,
                "request_complete": False,
                "request_id": "request-a",
                "schema": "llama-server-warm-tier-result-v2",
                "success": True,
            },
            "role": "PHONE",
            "route_epoch": 1,
            "route_instance_id": "route-a",
            "run_id": "run-a",
            "runtime_config_sha256": "9" * 64,
            "sampler": {
                "temperature": 0.0,
                "type": "greedy_argmax",
            },
            "schema": "s40-phone-command-evidence-v4",
            "seq_id": 0,
            "started_ns": 21,
            "success": True,
        }]
        fixture.config = config_path
        fixture.route = fixture.root / "phone-route.jsonl"
        fixture.wire = fixture.root / "phone-wire.jsonl"
        fixture.command = fixture.root / "phone-command.jsonl"
        write_jsonl(fixture.route, route_rows)
        write_jsonl(fixture.wire, wire_rows)
        write_jsonl(fixture.command, command_rows)
        fixture.configure_transport(
            "phone",
            "PHONE",
            fixture.config,
            fixture.command,
            fixture.wire,
            fixture.route,
        )
        command_rows = [
            json.loads(line)
            for line in fixture.command.read_text(encoding="ascii").splitlines()
        ]
        return fixture, config, route_rows, command_rows

    def validate(
        self,
        fixture: EvidenceFixture,
        kind: str,
        gateway_source_path: Path | None = None,
    ) -> dict:
        return validate_executor_bundle(
            kind=kind,
            config_path=fixture.config,
            command_path=fixture.command,
            executor_id="CPU" if kind == "desktop" else "PHONE",
            transport_descriptor_path=fixture.transport_descriptor,
            gateway_argv_path=fixture.gateway_argv,
            gateway_source_path=(
                fixture.gateway_source
                if gateway_source_path is None
                else gateway_source_path
            ),
            executor_bundle_manifest_path=fixture.bundle_manifest,
            socket_path=fixture.socket,
            gateway_stdout_path=fixture.stdout,
            gateway_stderr_path=fixture.stderr,
            wire_path=None if kind == "desktop" else fixture.wire,
            route_path=None if kind == "desktop" else fixture.route,
        )

    def test_desktop_evidence_passes(self):
        fixture, _, rows = self.desktop_fixture()
        result = self.validate(fixture, "desktop")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["configured_models"], ["model-a"])
        self.assertEqual(result["initial_active_models"], [])
        self.assertEqual(
            result["command_lineage"][1]["publications"][0]["token"],
            11,
        )
        self.assertEqual(
            result["runtime_processes"],
            [{
                "argv": rows[1]["lifecycle"]["runtime_probe"]["argv"],
                "pid": 101,
                "start_ticks": 10001,
            }],
        )

    def test_desktop_failed_command_is_rejected(self):
        fixture, _, rows = self.desktop_fixture()
        rows[1]["success"] = False
        write_jsonl(fixture.command, rows)
        with self.assertRaisesRegex(RuntimeError, "command failed"):
            self.validate(fixture, "desktop")

    def test_desktop_slot_path_is_bound(self):
        fixture, config, _ = self.desktop_fixture()
        missing = str(fixture.root / "missing")
        config["routes"][0]["slot_save_path"] = missing
        template = config["routes"][0]["child_argv_template"]
        template[template.index("--slot-save-path") + 1] = missing
        write_json(fixture.config, config)
        fixture.configure_transport(
            "desktop",
            "CPU",
            fixture.config,
            fixture.command,
        )
        with self.assertRaisesRegex(RuntimeError, "directory is missing"):
            self.validate(fixture, "desktop")

    def test_desktop_phase_lock_is_distinct_from_route_lock(self):
        fixture, config, _ = self.desktop_fixture()
        config["routes"][0]["phase_lock_sha256"] = (
            config["routes"][0]["route_lock_sha256"]
        )
        write_json(fixture.config, config)
        fixture.configure_transport(
            "desktop",
            "CPU",
            fixture.config,
            fixture.command,
        )
        with self.assertRaisesRegex(RuntimeError, "phase root"):
            self.validate(fixture, "desktop")

    def test_desktop_runtime_argv_is_exact(self):
        fixture, _, rows = self.desktop_fixture()
        probe = rows[1]["lifecycle"]["runtime_probe"]
        index = probe["argv"].index("--batch-size") + 1
        probe["argv"][index] = "1024"
        probe["child_argv_sha256"] = hashlib.sha256(
            b"\0".join(
                argument.encode("utf-8")
                for argument in probe["argv"]
            )
            + b"\0"
        ).hexdigest()
        write_jsonl(fixture.command, rows)
        with self.assertRaisesRegex(RuntimeError, "runtime child argv"):
            self.validate(fixture, "desktop")

    def test_phone_evidence_passes(self):
        fixture, _, _, _ = self.phone_fixture()
        result = self.validate(fixture, "phone")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["configured_models"], ["model-a"])
        self.assertEqual(result["initial_active_models"], [])
        self.assertEqual(result["wire_batches"], 1)
        self.assertEqual(result["runtime_processes"], [])
        self.assertEqual(
            result["route_instances"][0]["route_instance_id"],
            "route-a",
        )

    def test_phone_phase_lock_is_distinct_from_route_lock(self):
        fixture, config, _, _ = self.phone_fixture()
        config["routes"][0]["phase_lock_sha256"] = "b" * 64
        write_json(fixture.config, config)
        fixture.configure_transport(
            "phone",
            "PHONE",
            fixture.config,
            fixture.command,
            fixture.wire,
            fixture.route,
        )
        with self.assertRaisesRegex(RuntimeError, "readiness roots"):
            self.validate(fixture, "phone")

    def test_phone_terminal_placement_is_recomputed(self):
        fixture, _, route_rows, _ = self.phone_fixture()
        placement = route_rows[2]["placements"]["op15"]
        placement["session"]["session_end"] = "EOF"
        placement["certificate_lines"][0] = (
            "SESSIONCERT "
            + canonical_bytes(placement["session"]).decode("ascii").strip()
        )
        placement["certificate_lines_sha256"] = hashlib.sha256(
            "".join(
                line + "\n"
                for line in placement["certificate_lines"]
            ).encode("ascii")
        ).hexdigest()
        write_jsonl(fixture.route, route_rows)
        with self.assertRaisesRegex(RuntimeError, "terminal placement"):
            self.validate(fixture, "phone")

    def test_phone_placement_line_digest_is_recomputed(self):
        fixture, _, route_rows, _ = self.phone_fixture()
        route_rows[2]["placements"]["op12"][
            "certificate_lines_sha256"
        ] = "f" * 64
        write_jsonl(fixture.route, route_rows)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.validate(fixture, "phone")

    def test_phone_failed_command_is_rejected(self):
        fixture, _, _, rows = self.phone_fixture()
        rows[1]["success"] = False
        rows[1]["result"]["success"] = False
        write_jsonl(fixture.command, rows)
        with self.assertRaisesRegex(RuntimeError, "command failed"):
            self.validate(fixture, "phone")

    def test_phone_unclosed_route_is_rejected(self):
        fixture, _, route_rows, _ = self.phone_fixture()
        write_jsonl(fixture.route, route_rows[:-1])
        with self.assertRaisesRegex(RuntimeError, "was not unloaded"):
            self.validate(fixture, "phone")

    def test_phone_route_extra_field_is_rejected(self):
        fixture, _, route_rows, _ = self.phone_fixture()
        route_rows[1]["verdict"] = "PASS"
        write_jsonl(fixture.route, route_rows)
        with self.assertRaisesRegex(RuntimeError, "fields do not match"):
            self.validate(fixture, "phone")

    def test_phone_load_identity_fields_are_exact(self):
        fields = {
            "a6000_identity": "7" * 64,
            "model_sha256": "b" * 64,
            "op12_boot_id": "wrong-op12",
            "op12_shard_sha256": "8" * 64,
            "op15_boot_id": "wrong-op15",
            "op15_shard_sha256": "9" * 64,
            "worker_sha256": "0" * 64,
        }
        for field, value in fields.items():
            with self.subTest(field=field):
                fixture, _, route_rows, _ = self.phone_fixture()
                route_rows[1][field] = value
                write_jsonl(fixture.route, route_rows)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "readiness binding",
                ):
                    self.validate(fixture, "phone")

    def test_phone_configured_model_omission_is_rejected(self):
        fixture, _, _, rows = self.phone_fixture()
        rows[0]["configured_models"] = []
        write_jsonl(fixture.command, rows)
        with self.assertRaisesRegex(RuntimeError, "startup identity"):
            self.validate(fixture, "phone")

    def test_phone_result_token_must_match_wire_output(self):
        fixture, _, _, rows = self.phone_fixture()
        rows[1]["result"]["publications"][0]["token"] = 12
        write_jsonl(fixture.command, rows)
        with self.assertRaisesRegex(RuntimeError, "differs from wire"):
            self.validate(fixture, "phone")

    def test_desktop_result_identity_is_exact(self):
        fixture, _, rows = self.desktop_fixture()
        rows[1]["result"]["command_id"] = 2
        write_jsonl(fixture.command, rows)
        with self.assertRaisesRegex(RuntimeError, "command binding"):
            self.validate(fixture, "desktop")

    def test_desktop_nvidia_smi_mutation_is_rejected(self):
        fixture, config, _ = self.desktop_fixture()
        Path(config["nvidia_smi"]["path"]).write_bytes(b"changed\n")
        with self.assertRaisesRegex(RuntimeError, "nvidia-smi executable changed"):
            parse_desktop_config(fixture.config)

    def test_desktop_nvidia_smi_path_shadow_is_rejected(self):
        fixture, config, _ = self.desktop_fixture()
        config["nvidia_smi"]["path"] = "nvidia-smi"
        write_json(fixture.config, config)
        with self.assertRaisesRegex(RuntimeError, "nvidia-smi path"):
            parse_desktop_config(fixture.config)

    def test_desktop_probe_nvidia_smi_identity_is_exact(self):
        fixture, _, rows = self.desktop_fixture()
        rows[1]["lifecycle"]["runtime_probe"]["nvidia_smi"]["path"] = (
            "/tmp/shadow-nvidia-smi"
        )
        write_jsonl(fixture.command, rows)
        with self.assertRaisesRegex(
            RuntimeError,
            "nvidia-smi identity",
        ):
            self.validate(fixture, "desktop")

    def test_desktop_cache_gate_is_recomputed(self):
        fixture, _, rows = self.desktop_fixture()
        rows[1]["lifecycle"]["cache_control"]["output"]["after"][
            "resident_ppm"
        ] = 949_999
        write_jsonl(fixture.command, rows)
        with self.assertRaisesRegex(RuntimeError, "warm cache gate"):
            self.validate(fixture, "desktop")

    def test_desktop_publication_frontier_is_exact(self):
        fixture, _, rows = self.desktop_fixture()
        rows[2]["result"]["publications"][0]["position"] = 4
        write_jsonl(fixture.command, rows)
        with self.assertRaisesRegex(RuntimeError, "publication frontier"):
            self.validate(fixture, "desktop")

    def test_desktop_cleanup_is_mandatory(self):
        fixture, _, rows = self.desktop_fixture()
        write_jsonl(fixture.command, rows[:-1])
        with self.assertRaisesRegex(
            RuntimeError,
            "cleanup|fields do not match",
        ):
            self.validate(fixture, "desktop")

    def test_desktop_unload_requires_exact_child_exit(self):
        fixture, _, rows = self.desktop_fixture()
        probe = copy.deepcopy(rows[1]["lifecycle"]["runtime_probe"])

        def lifecycle_row(command_id, kind, started_ns, lifecycle):
            command = model_command("CPU", command_id, "model-a", kind)
            return {
                "command": command,
                "command_id": command_id,
                "completed_ns": started_ns + 1,
                "controller_epoch": 1,
                "durability": "fsync_each_record",
                "execute": None,
                "executor_id": "CPU",
                "executor_instance_id": "instance-cpu",
                "kind": kind,
                "lifecycle": lifecycle,
                "model_id": "model-a",
                "request_id": None,
                "result": {
                    "command_id": command_id,
                    "controller_epoch": 1,
                    "detail": "desktop lifecycle",
                    "executor_id": "CPU",
                    "executor_instance_id": "instance-cpu",
                    "has_replay_snapshot": False,
                    "kind": kind,
                    "model_id": "model-a",
                    "publications": [],
                    "replay_snapshot": None,
                    "request_complete": False,
                    "request_id": "",
                    "schema": "llama-server-warm-tier-result-v2",
                    "success": True,
                },
                "role": "CPU",
                "run_id": "run-a",
                "runtime_config_sha256": rows[0]["runtime_config_sha256"],
                "schema": "s40-desktop-command-evidence-v4",
                "started_ns": started_ns,
                "success": True,
            }

        rows[-1]["initial_active_models"] = []
        rows[-1]["started_ns"] = 50
        rows[-1]["completed_ns"] = 51
        rows[-1]["unloaded"] = []
        rows[-1:-1] = [
            lifecycle_row(3, 1, 30, None),
            lifecycle_row(
                4,
                2,
                40,
                {
                    "instance_id": "101:8081:1",
                    "operation": "UNLOAD",
                    "process_exited": True,
                    "router_status": "unloaded",
                    "runtime_probe": probe,
                },
            ),
        ]
        write_jsonl(fixture.command, rows)
        self.assertEqual(self.validate(fixture, "desktop")["status"], "PASS")
        rows[-2]["lifecycle"]["process_exited"] = False
        write_jsonl(fixture.command, rows)
        with self.assertRaisesRegex(RuntimeError, "lifecycle operation"):
            self.validate(fixture, "desktop")

    def test_phone_wire_input_matches_command_frontier(self):
        fixture, _, _, _ = self.phone_fixture()
        wire = [
            {
                **row,
                "input_rows": [
                    {**item, "token": 99}
                    for item in row["input_rows"]
                ],
            }
            for row in [
                json.loads(line)
                for line in fixture.wire.read_text(encoding="ascii").splitlines()
            ]
        ]
        write_jsonl(fixture.wire, wire)
        with self.assertRaisesRegex(
            RuntimeError,
            "wire input differs from command frontier",
        ):
            self.validate(fixture, "phone")

    def test_legacy_bridge_descriptor_is_rejected(self):
        fixture, _, _ = self.desktop_fixture()
        write_json(
            fixture.transport_descriptor,
            {
                "argv": [
                    sys.executable,
                    str(fixture.bundle_manifest.parent / "gateway_bridge.py"),
                    "--socket",
                    str(fixture.socket),
                ],
                "schema": "s40-bridge-argv-v1",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "fields do not match"):
            self.validate(fixture, "desktop")

    def test_controller_authentication_argv_is_mandatory(self):
        for flag in (
            "--controller-identity",
            "--controller-binding-evidence",
        ):
            with self.subTest(flag=flag):
                fixture, _, _ = self.desktop_fixture()
                argv = json.loads(
                    fixture.gateway_argv.read_text(encoding="ascii"))
                index = argv["argv"].index(flag)
                del argv["argv"][index:index + 2]
                write_json(fixture.gateway_argv, argv)
                descriptor = json.loads(
                    fixture.transport_descriptor.read_text(encoding="ascii"))
                descriptor["gateway_argv_sha256"] = hashlib.sha256(
                    fixture.gateway_argv.read_bytes()).hexdigest()
                cmdline = b"".join(
                    argument.encode("ascii") + b"\x00"
                    for argument in argv["argv"]
                )
                for identity_key in (
                        "gateway_prepublication_identity",
                        "gateway_post_auth_identity"):
                    descriptor[identity_key]["cmdline_base64"] = (
                        base64.b64encode(cmdline).decode("ascii"))
                    descriptor[identity_key]["cmdline_sha256"] = (
                        hashlib.sha256(cmdline).hexdigest())
                write_json(fixture.transport_descriptor, descriptor)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "gateway controller .* flag",
                ):
                    self.validate(fixture, "desktop")

    def test_v4_transport_controller_fields_are_mandatory(self):
        for field in (
            "controller_binding_sha256",
            "controller_identity_sha256",
            "controller_pid",
        ):
            with self.subTest(field=field):
                fixture, _, _ = self.desktop_fixture()
                descriptor = json.loads(
                    fixture.transport_descriptor.read_text(encoding="ascii"))
                del descriptor[field]
                write_json(fixture.transport_descriptor, descriptor)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "fields do not match",
                ):
                    self.validate(fixture, "desktop")

    def test_legacy_v3_transport_is_rejected(self):
        fixture, _, _ = self.desktop_fixture()
        descriptor = json.loads(
            fixture.transport_descriptor.read_text(encoding="ascii"))
        descriptor["schema"] = "s40-executor-transport-descriptor-v3"
        write_json(fixture.transport_descriptor, descriptor)
        with self.assertRaisesRegex(
                RuntimeError, "executor transport descriptor identity"):
            self.validate(fixture, "desktop")

    def test_controller_peer_tuple_is_recomputed(self):
        fixture, _, _ = self.desktop_fixture()
        binding = json.loads(
            fixture.controller_binding.read_text(encoding="ascii"))
        binding["peer_pid"] += 1
        write_json(fixture.controller_binding, binding)
        descriptor = json.loads(
            fixture.transport_descriptor.read_text(encoding="ascii"))
        descriptor["controller_binding_sha256"] = hashlib.sha256(
            fixture.controller_binding.read_bytes()).hexdigest()
        write_json(fixture.transport_descriptor, descriptor)
        with self.assertRaisesRegex(
            RuntimeError,
            "controller binding evidence peer mismatch",
        ):
            self.validate(fixture, "desktop")

    def test_controller_authentication_cannot_predate_identity(self):
        fixture, _, _ = self.desktop_fixture()
        binding = json.loads(
            fixture.controller_binding.read_text(encoding="ascii"))
        binding["authenticated_ns"] = 2
        write_json(fixture.controller_binding, binding)
        descriptor = json.loads(
            fixture.transport_descriptor.read_text(encoding="ascii"))
        descriptor["controller_binding_sha256"] = hashlib.sha256(
            fixture.controller_binding.read_bytes()).hexdigest()
        write_json(fixture.transport_descriptor, descriptor)
        with self.assertRaisesRegex(
            RuntimeError,
            "authentication time",
        ):
            self.validate(fixture, "desktop")

    def test_command_cannot_predate_controller_authentication(self):
        fixture, _, _ = self.desktop_fixture()
        binding = json.loads(
            fixture.controller_binding.read_text(encoding="ascii"))
        binding["authenticated_ns"] = 5
        write_json(fixture.controller_binding, binding)
        descriptor = json.loads(
            fixture.transport_descriptor.read_text(encoding="ascii"))
        descriptor["controller_binding_sha256"] = hashlib.sha256(
            fixture.controller_binding.read_bytes()).hexdigest()
        write_json(fixture.transport_descriptor, descriptor)
        with self.assertRaisesRegex(
            RuntimeError,
            "command predates controller authentication",
        ):
            self.validate(fixture, "desktop")

    def test_controller_executable_digest_is_recomputed(self):
        fixture, _, _ = self.desktop_fixture()
        fixture.controller_executable.write_bytes(b"replaced controller\n")
        with self.assertRaisesRegex(
            RuntimeError,
            "controller executable identity",
        ):
            self.validate(fixture, "desktop")

    def test_controller_binding_gateway_identity_is_exact(self):
        fixture, _, _ = self.desktop_fixture()
        binding = json.loads(
            fixture.controller_binding.read_text(encoding="ascii"))
        binding["gateway_pid"] += 1
        write_json(fixture.controller_binding, binding)
        descriptor = json.loads(
            fixture.transport_descriptor.read_text(encoding="ascii"))
        descriptor["controller_binding_sha256"] = hashlib.sha256(
            fixture.controller_binding.read_bytes()).hexdigest()
        write_json(fixture.transport_descriptor, descriptor)
        with self.assertRaisesRegex(
            RuntimeError,
            "controller binding evidence identity",
        ):
            self.validate(fixture, "desktop")

    def test_controller_binding_inode_is_load_bearing(self):
        fixture, _, _ = self.desktop_fixture()
        raw = fixture.controller_binding.read_bytes()
        original_inode = fixture.controller_binding.stat().st_ino
        replacement = fixture.controller_binding.with_name(
            "replacement-controller-binding.json")
        replacement.write_bytes(raw)
        self.assertNotEqual(replacement.stat().st_ino, original_inode)
        os.replace(replacement, fixture.controller_binding)
        with self.assertRaisesRegex(
            RuntimeError,
            "controller binding evidence identity",
        ):
            self.validate(fixture, "desktop")

    def test_transport_descriptor_and_executed_files_are_load_bearing(self):
        cases = (
            "socket",
            "argv_config",
            "runtime",
            "gateway_bridge",
            "bundle_source_mutation",
            "hidden_socket_override",
            "injected_environment",
            "forged_cmdline",
            "process_replacement",
            "forged_file_metadata",
        )
        for case in cases:
            with self.subTest(case=case):
                fixture, _, _ = self.desktop_fixture()
                descriptor = json.loads(
                    fixture.transport_descriptor.read_text(encoding="ascii")
                )
                argv_record = json.loads(
                    fixture.gateway_argv.read_text(encoding="ascii")
                )
                source = fixture.gateway_source
                if case == "socket":
                    descriptor["socket_path"] = str(
                        fixture.root / "other.sock"
                    )
                elif case == "argv_config":
                    index = argv_record["argv"].index("--config") + 1
                    argv_record["argv"][index] = str(
                        fixture.root / "other-config.json"
                    )
                    write_json(fixture.gateway_argv, argv_record)
                    descriptor["gateway_argv_sha256"] = hashlib.sha256(
                        fixture.gateway_argv.read_bytes()
                    ).hexdigest()
                elif case == "runtime":
                    runtime = Path(descriptor["runtime_config_path"])
                    runtime.write_bytes(runtime.read_bytes() + b" ")
                elif case == "gateway_bridge":
                    source = (
                        fixture.bundle_manifest.parent / "gateway_bridge.py"
                    )
                    argv_record["argv"][7] = str(source)
                    write_json(fixture.gateway_argv, argv_record)
                    descriptor["gateway_argv_sha256"] = hashlib.sha256(
                        fixture.gateway_argv.read_bytes()
                    ).hexdigest()
                    descriptor["gateway_source_sha256"] = hashlib.sha256(
                        source.read_bytes()
                    ).hexdigest()
                elif case == "hidden_socket_override":
                    argv_record["argv"].append(
                        "--socket=" + str(fixture.root / "other.sock")
                    )
                    write_json(fixture.gateway_argv, argv_record)
                    descriptor["gateway_argv_sha256"] = hashlib.sha256(
                        fixture.gateway_argv.read_bytes()
                    ).hexdigest()
                elif case == "injected_environment":
                    descriptor["gateway_environment"]["PYTHONPATH"] = (
                        "/tmp/attacker")
                elif case == "forged_cmdline":
                    raw = b"/forged/python\x00"
                    identity = descriptor["gateway_post_auth_identity"]
                    identity["cmdline_base64"] = (
                        base64.b64encode(raw).decode("ascii"))
                    identity["cmdline_sha256"] = hashlib.sha256(
                        raw).hexdigest()
                elif case == "process_replacement":
                    descriptor["gateway_post_auth_identity"][
                        "executable_inode"] += 1
                elif case == "forged_file_metadata":
                    descriptor["gateway_executed_files"][0][
                        "source_inode"] += 1
                else:
                    source.write_bytes(source.read_bytes() + b"\n")
                    descriptor["gateway_source_sha256"] = hashlib.sha256(
                        source.read_bytes()
                    ).hexdigest()
                write_json(fixture.transport_descriptor, descriptor)
                with self.assertRaises(RuntimeError):
                    self.validate(
                        fixture,
                        "desktop",
                        gateway_source_path=source,
                    )


if __name__ == "__main__":
    unittest.main()
