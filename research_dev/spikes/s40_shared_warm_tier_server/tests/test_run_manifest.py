#!/usr/bin/env python3

import base64
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import quote
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
S40 = HERE.parent
sys.path.insert(0, str(S40))
sys.path.insert(0, str(HERE))

from build_runtime_config import build_runtime_config  # noqa: E402
from campaign_plan import CAMPAIGN_TOOL_PATHS, build_campaign  # noqa: E402
from evidence_common import (  # noqa: E402
    EvidenceError,
    canonical_bytes,
    digest_bytes,
    digest_file,
    read_jsonl,
)
from fake_ledger import Ledger, snapshot  # noqa: E402
from event_evidence import extract_command_ledger  # noqa: E402
from gpu_isolation import canonical_lock_path  # noqa: E402
from run_manifest import (  # noqa: E402
    _isolated_python_argv,
    _validate_command_bijection,
    _validate_phone_route_coverage,
    _validate_route_qualification_summaries,
    qualification_source_records,
    validate_qualification_route_identity,
    validate_run_manifest,
)
from validate_inputs import DEFAULT_CONTRACT  # noqa: E402
from executor_bundle import build_executor_bundle  # noqa: E402
from evidence_bundle import build_bundle as build_evidence_bundle  # noqa: E402
from bridge_overhead import summarize as summarize_bridge_overhead  # noqa: E402


GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
SERVING_ARGV = [
    "--ctx-size", "4096",
    "--parallel", "8",
    "--batch-size", "2048",
    "--ubatch-size", "512",
    "--flash-attn", "on",
    "--cont-batching",
    "--cache-type-k", "f16",
    "--cache-type-v", "f16",
    "--split-mode", "none",
]


def write_jsonl(path, rows):
    path.write_bytes(b"".join(canonical_bytes(row) for row in rows))


class RunManifestTests(unittest.TestCase):
    def setUp(self):
        self.executor_validation = patch(
            "run_manifest.validate_executor_bundle",
            side_effect=self.fake_executor_summary,
        )
        self.executor_validation.start()
        self.addCleanup(self.executor_validation.stop)

    @staticmethod
    def fake_executor_summary(**kwargs):
        runtime_path = kwargs["command_path"].parent / "runtime.json"
        runtime_stat = runtime_path.stat(follow_symlinks=False)
        events = read_jsonl(
            kwargs["command_path"].parent / "events.jsonl",
            "fixture_events",
        )
        lineage = [{
            "command": {"executor_instance_id": "instance-gpu0"},
            "command_id": row["command_id"],
            "controller_epoch": row["controller_epoch"],
            "executor_id": row["executor_id"],
            "kind": row["command_kind"],
            "model_id": row["model_id"],
            "publications": row["publications"],
            "request_complete": row["request_complete"],
            "request_id": row["request_id"],
            "success": row["success"],
        } for row in extract_command_ledger(events)]
        return {
            "command_lineage": lineage,
            "configured_models": [
                "qwen3-14b-q4_k_m",
                "qwen3-8b-q8_0",
            ],
            "initial_active_models": [],
            "fastest_execute_duration_ns": 100_000_000,
            "executor_instance_id": "instance-gpu0",
            "gateway_pid": 2345,
            "gateway_start_time_ticks": 3456,
            "host_boot_id": "boot-host",
            "identity_captured_ns": 1,
            "runtime_config_device": runtime_stat.st_dev,
            "runtime_config_inode": runtime_stat.st_ino,
            "runtime_config_path": str(runtime_path),
            "runtime_config_published_ns": 2,
            "role": "GPU",
            "route_qualifications": [{
                "a_chain_phase_id": None,
                "bundle_manifest_sha256": "1" * 64,
                "model_id": "qwen3-14b-q4_k_m",
                "phase": "A_ONLY",
                "phase_id": "phase-a",
                "phase_lock_sha256": "2" * 64,
                "schema": "s40-route-qualification-derived-v1",
                "scope": "QUALIFIED_ROUTE",
                "status": "MODEL_A_QUALIFICATION_PASS",
                "v2_2_result_sha256": "3" * 64,
            }, {
                "a_chain_phase_id": "phase-a",
                "bundle_manifest_sha256": "4" * 64,
                "model_id": "qwen3-8b-q8_0",
                "phase": "B_ONLY",
                "phase_id": "phase-b",
                "phase_lock_sha256": "5" * 64,
                "schema": "s40-route-qualification-derived-v1",
                "scope": "QUALIFIED_ROUTE",
                "status": "MODEL_B_QUALIFICATION_PASS",
                "v2_2_result_sha256": "6" * 64,
            }],
            "run_id": events[0]["run_id"],
            "runtime_processes": [{
                "argv": ["/fixture/gpu-server", "--model", "fixture"],
                "pid": 4321,
                "start_ticks": 777,
            }],
            "runtime_config_sha256":
                events[0]["runtime_config_sha256"],
            "schema": "s40-executor-evidence-validation-v1",
            "status": "PASS",
        }

    def artifact(self, root, role, name, file_format, raw):
        path = root / name
        if path.exists():
            self.assertEqual(path.read_bytes(), raw)
        else:
            path.write_bytes(raw)
        return {
            "bytes": path.stat().st_size,
            "format": file_format,
            "path": name,
            "record_count": (
                len(raw.splitlines()) if file_format == "JSONL" else None
            ),
            "role": role,
            "sha256": digest_file(path),
        }

    def make_events(self, runtime_sha, run_id, origin_ns):
        requests = read_jsonl(
            S40.parent / "s39_desktop_swap_baseline"
            / "DESKTOP_REQUESTS.jsonl",
            "requests",
        )
        ledger = Ledger(run_id)
        ledger.add("run_start")
        for expected in requests:
            ledger.t_ns = max(
                ledger.t_ns,
                origin_ns + expected["arrival_us"] * 1000,
            )
            prompt = expected["prompt_tokens"]
            committed = []
            queued = snapshot(
                expected["event_id"],
                expected["model_id"],
                prompt,
                committed,
                None,
                0,
                "QUEUED",
            )
            ledger.add(
                "request_arrived",
                model_id=expected["model_id"],
                request_id=expected["event_id"],
                request=queued,
            )
            active = snapshot(
                expected["event_id"],
                expected["model_id"],
                prompt,
                committed,
                "gpu0",
                1,
                "ACTIVE",
            )
            ledger.add(
                "request_dispatched",
                model_id=expected["model_id"],
                request_id=expected["event_id"],
                executor_id="gpu0",
                request=active,
            )
            for publication in range(8):
                committed.append(100 + publication)
                active = snapshot(
                    expected["event_id"],
                    expected["model_id"],
                    prompt,
                    committed,
                    "gpu0",
                    1,
                    "ACTIVE",
                )
                ledger.add(
                    "token_committed",
                    model_id=expected["model_id"],
                    request_id=expected["event_id"],
                    executor_id="gpu0",
                    request=active,
                    publication_index=publication,
                )
            complete = snapshot(
                expected["event_id"],
                expected["model_id"],
                prompt,
                committed,
                "gpu0",
                1,
                "COMPLETED",
            )
            ledger.add(
                "request_completed",
                model_id=expected["model_id"],
                request_id=expected["event_id"],
                executor_id="gpu0",
                request=complete,
            )
        ledger.add("run_end", detail="TRACE_COMPLETE")
        for row in ledger.rows:
            row["runtime_config_sha256"] = runtime_sha
        return ledger.rows

    def make_resources(self, run_id, end_ns):
        rows = []
        points = list(range(0, end_ns + 1, 500_000_000))
        if points[-1] < end_ns:
            points.append(end_ns)
        for sequence, t_ns in enumerate(points):
            rows.append({
                "controller_pid": 1234,
                "controller_process_cpu_ticks": 100 + sequence,
                "controller_process_cpu_utilization_milli_pct": 5_000,
                "controller_process_rss_bytes": 8_000_000_000,
                "controller_process_start_ticks": 99,
                "controller_process_swap_bytes": 0,
                "cpu_utilization_milli_pct": 25_000,
                "gpu_memory_free_bytes": 4_000_000_000,
                "gpu_memory_used_bytes": 12_000_000_000,
                "gpu_power_mw": 100_000 + sequence % 2,
                "gpu_uuid": GPU_UUID,
                "host_boot_id": "boot-host",
                "process_metric_scope": "CONTROLLER_PROCESS_ONLY",
                "run_id": run_id,
                "schema": "s40-selected-gpu-resource-v3",
                "sequence": sequence,
                "system_mem_available_bytes": 16_000_000_000,
                "system_swap_free_bytes": 1_000_000_000,
                "system_swap_total_bytes": 1_000_000_000,
                "t_ns": t_ns,
            })
        return rows

    def make_http(self, run_id, trace_start, requests):
        rows = []

        def add(
                method, path, operation, request_id, attempt,
                started, request, response):
            request_raw = (
                canonical_bytes(request) if request is not None else b"")
            response_raw = canonical_bytes(response)
            rows.append({
                "attempt": attempt,
                "http_status": 200,
                "method": method,
                "operation": operation,
                "path": path,
                "request_body_base64":
                    base64.b64encode(request_raw).decode("ascii"),
                "request_body_sha256": digest_bytes(request_raw),
                "request_id": request_id,
                "response_body_base64":
                    base64.b64encode(response_raw).decode("ascii"),
                "response_body_sha256": digest_bytes(response_raw),
                "response_headers": [],
                "run_id": run_id,
                "schema": "s40-http-evidence-v1",
                "sequence": len(rows),
                "t_end_ns": started + 1,
                "t_start_ns": started,
                "trace_start_sha256":
                    digest_bytes(canonical_bytes(trace_start)),
            })

        for arrival_order, expected in enumerate(requests):
            request = {
                "arrival_order": arrival_order,
                "max_output_tokens": 8,
                "model": expected["model_id"],
                "prompt_tokens": expected["prompt_tokens"],
                "request_id": expected["event_id"],
                "schema": "llama-server-warm-tier-request-v1",
            }
            add(
                "POST",
                "/experimental/warm-tier/requests",
                "ADMIT",
                expected["event_id"],
                0,
                trace_start["trace_origin_ns"]
                + expected["arrival_us"] * 1000,
                request,
                {
                    "arrival_order": arrival_order,
                    "controller_epoch": 0,
                    "request_id": expected["event_id"],
                    "schema": "llama-server-warm-tier-admission-v1",
                    "state": "QUEUED",
                },
            )
        started = max(row["t_end_ns"] for row in rows) + 1
        for expected in requests:
            tokens = list(range(100, 108))
            add(
                "GET",
                "/experimental/warm-tier/requests/"
                + quote(expected["event_id"], safe=""),
                "STATUS",
                expected["event_id"],
                0,
                started,
                None,
                {
                    "committed_output_tokens": tokens,
                    "controller_epoch": 0,
                    "model": expected["model_id"],
                    "owner_id": "gpu0",
                    "ownership_epoch": 1,
                    "position": len(expected["prompt_tokens"]) + len(tokens),
                    "publication_index": len(tokens),
                    "request_id": expected["event_id"],
                    "schema":
                        "llama-server-warm-tier-request-status-v1",
                    "state": "COMPLETED",
                },
            )
            started += 2
        add(
            "POST",
            "/experimental/warm-tier/finalize",
            "FINALIZE",
            "",
            0,
            started,
            {
                "reason": "TRACE_COMPLETE",
                "schema": "llama-server-warm-tier-finalize-v1",
            },
            {
                "controller_epoch": 0,
                "reason": "TRACE_COMPLETE",
                "schema": "llama-server-warm-tier-finalize-result-v1",
                "state": "FINALIZED",
            },
        )
        result = {
            "admission_count": len(requests),
            "campaign_horizon_ns": trace_start["campaign_horizon_ns"],
            "completed_count": len(requests),
            "drain_bound_ns": trace_start["drain_bound_us"] * 1000,
            "finalization_reason": "TRACE_COMPLETE",
            "finalization_state": "FINALIZED",
            "maximum_admission_lateness_ns": 0,
            "schema": "s40-trace-acquisition-result-v2",
            "stranded_count": 0,
            "terminal_count": len(requests),
            "trace_start_sha256":
                digest_bytes(canonical_bytes(trace_start)),
        }
        return rows, result

    def make_fixture(self, root):
        run_id = "fixture-000"
        bundle = build_executor_bundle(root / "executor-bundle")
        evidence_bundle = build_evidence_bundle(root / "evidence-bundle")
        binary = root / "llama-server"
        binary.write_bytes(b"fixture-binary\n")
        native_binary = root / "test-warm-tier-executors"
        native_binary.write_bytes(b"fixture-native-bench\n")
        nvidia_smi = root / "nvidia-smi"
        nvidia_smi.write_bytes(b"fixture-nvidia-smi\n")
        ldd = root / "ldd"
        ldd.write_bytes(b"fixture-ldd\n")
        python_binary = root / "python-captured"
        python_binary.write_bytes(b"fixture-python\n")
        bridge = root / "gateway_bridge.py"
        bridge.write_bytes(b"#!/usr/bin/env python3\n")
        gateway_config = root / "gateway-config.json"
        gateway_config.write_bytes(canonical_bytes({
            "executor_id": "gpu0",
            "nvidia_smi": {
                "bytes": nvidia_smi.stat().st_size,
                "path": str(nvidia_smi),
                "sha256": digest_file(nvidia_smi),
            },
            "routes": [],
            "schema": "s40-desktop-executor-config-v4",
        }))
        gateway_argv = _isolated_python_argv(
            python_binary,
            root / "executor-bundle",
            root / "executor-bundle" / "desktop_gateway.py",
            [
                "--config", str(gateway_config),
                "--runtime-config", str(root / "runtime.json"),
                "--executor-instance-id", "instance-gpu0",
                "--controller-identity",
                str(root / "controller-identity-lock.json"),
                "--controller-binding-evidence",
                str(root / "executor-controller-binding-gpu0.json"),
            ],
        )
        evidence_root = root / "evidence-root.json"
        evidence_root.write_bytes(canonical_bytes({
            "c3_placement_artifacts": {},
            "configuration": "C1_GPU_ONLY_OPTIMIZED",
            "executor_configs": [{
                "executor_id": "gpu0",
                "gateway_config_path": str(gateway_config),
                "gateway_config_sha256": digest_file(gateway_config),
            }],
            "experiment_contract_sha256": digest_file(DEFAULT_CONTRACT),
            "schema": "s40-runtime-evidence-root-v1",
        }))
        event_path = root / "events.jsonl"
        plan = {
            "c3_profile_lock_path": None,
            "c3_profile_lock_sha256": None,
            "evidence_root_path": str(evidence_root),
            "evidence_root_sha256": digest_file(evidence_root),
            "event_log_path": str(event_path),
            "executors": [{
                "credits": 1,
                "execute_concurrency": 1,
                "executor_id": "gpu0",
                "executor_instance_id": "instance-gpu0",
                "expected_peer_pid": 2345,
                "expected_peer_start_time_ticks": 3456,
                "kind": "GPU_PRIMARY",
                "output_limit_bytes": 4096,
                "queue_capacity": 128,
                "socket_path": str(root / "gpu.sock"),
                "timeout_ms": 1000,
                "transport": "UNIX_SOCKET",
            }],
            "hot_model_id": "qwen3-8b-q8_0",
            "mode": "C1_GPU_ONLY_OPTIMIZED",
            "run_id": run_id,
            "schema": "s40-runtime-config-plan-v3",
        }
        plan_path = root / "runtime-plan.json"
        plan_path.write_bytes(canonical_bytes(plan))
        source_plan_path = root / "source-runtime-plan.json"
        source_plan = copy.deepcopy(plan)
        source_plan["schema"] = "s40-runtime-config-plan-v2"
        for executor in source_plan["executors"]:
            del executor["executor_instance_id"]
            del executor["expected_peer_pid"]
            del executor["expected_peer_start_time_ticks"]
        source_plan_path.write_bytes(canonical_bytes(source_plan))
        runtime = build_runtime_config(plan_path)["runtime"]
        runtime_path = root / "runtime.json"
        runtime_path.write_bytes(canonical_bytes(runtime))
        runtime_sha = digest_file(runtime_path)
        runtime_stat = runtime_path.stat(follow_symlinks=False)
        controller_identity_path = root / "controller-identity-lock.json"
        controller_identity_path.write_bytes(canonical_bytes({
            "controller_executable_path": str(binary),
            "controller_executable_sha256": digest_file(binary),
            "controller_gid": 1000,
            "controller_pid": 1234,
            "controller_start_time_ticks": 99,
            "controller_uid": 1000,
            "host_boot_id": "boot-host",
            "run_id": run_id,
            "runtime_config_device": runtime_stat.st_dev,
            "runtime_config_inode": runtime_stat.st_ino,
            "runtime_config_path": str(runtime_path),
            "runtime_config_sha256": runtime_sha,
            "schema": "s40-controller-identity-lock-v1",
        }))
        controller_identity_stat = controller_identity_path.stat(
            follow_symlinks=False)
        controller_binding_path = (
            root / "executor-controller-binding-gpu0.json")
        controller_binding_path.write_bytes(canonical_bytes({
            "authenticated_ns": 41,
            "controller_executable_path": str(binary),
            "controller_executable_sha256": digest_file(binary),
            "controller_gid": 1000,
            "controller_identity_device": controller_identity_stat.st_dev,
            "controller_identity_inode": controller_identity_stat.st_ino,
            "controller_identity_path": str(controller_identity_path),
            "controller_identity_sha256":
                digest_file(controller_identity_path),
            "controller_pid": 1234,
            "controller_start_time_ticks": 99,
            "controller_uid": 1000,
            "executor_id": "gpu0",
            "executor_instance_id": "instance-gpu0",
            "gateway_pid": 2345,
            "gateway_start_time_ticks": 3456,
            "host_boot_id": "boot-host",
            "peer_gid": 1000,
            "peer_pid": 1234,
            "peer_uid": 1000,
            "run_id": run_id,
            "runtime_config_device": runtime_stat.st_dev,
            "runtime_config_inode": runtime_stat.st_ino,
            "runtime_config_path": str(runtime_path),
            "runtime_config_sha256": runtime_sha,
            "schema": "s40-executor-controller-binding-v1",
        }))
        controller_binding_stat = controller_binding_path.stat(
            follow_symlinks=False)
        server_argv = [
            str(binary),
            *SERVING_ARGV,
            "--host",
            "127.0.0.1",
            "--port",
            "48991",
            "--threads-http",
            "128",
        ]
        source_physical = root / "source-physical-plan.json"
        source_physical.write_bytes(canonical_bytes({
            "cache_regime": "WARM_HOST_CACHE",
            "campaign_binding": None,
            "contract_path": str(DEFAULT_CONTRACT.resolve()),
            "development": True,
            "devices": [{
                "boot_id": "boot-host",
                "device_role": "GPU",
                "stable_id": GPU_UUID,
            }],
            "gateway_processes": [{
                "argv": [
                    str(bridge),
                    "--config",
                    str(gateway_config),
                ],
                "executor_id": "gpu0",
                "socket_name": "gpu.sock",
            }],
            "gpu_index": 0,
            "gpu_lock_path": str(canonical_lock_path(GPU_UUID)),
            "gpu_pci_bus_id": "00000000:01:00.0",
            "ldd_path": str(ldd),
            "mode": "C1_GPU_ONLY_OPTIMIZED",
            "nvidia_smi_path": str(nvidia_smi),
            "output_dir": str(root),
            "phone_identity_argv": None,
            "phone_telemetry_argv": None,
            "repeat_index": 0,
            "run_id": run_id,
            "runtime_plan_template": str(source_plan_path),
            "schema": "s40-physical-run-plan-v2",
            "server_argv": server_argv,
            "server_base_url": "http://127.0.0.1:48991",
        }))
        source_lock_dir = root / "source-locks"
        source_lock_dir.mkdir()
        source_rows = []
        for index, (role, source) in enumerate((
                ("physical_plan", source_physical),
                ("experiment_contract", DEFAULT_CONTRACT.resolve()),
                ("runtime_plan_template", source_plan_path),
                ("evidence_root", evidence_root),
                ("gateway_config::gpu0", gateway_config),
                ("runtime_binary::controller", binary),
                ("runtime_binary::native_bench", native_binary),
                ("runtime_binary::nvidia_smi", nvidia_smi),
                ("runtime_binary::python", python_binary),
                ("runtime_tool::ldd", ldd))):
            captured = source_lock_dir / f"{index:03d}-{role.replace(':', '-')}.artifact"
            captured.write_bytes(source.read_bytes())
            source_rows.append({
                "bytes": captured.stat().st_size,
                "captured_path": str(captured.relative_to(root)),
                "role": role,
                "sha256": digest_file(captured),
                "source_path": str(source),
            })
        source_preflight = root / "source-preflight.json"
        source_preflight.write_bytes(canonical_bytes({
            "captured_ns": 0,
            "rows": source_rows,
            "schema": "s40-source-preflight-v1",
        }))
        origin_ns = 1_000_000_100
        events = self.make_events(runtime_sha, run_id, origin_ns)
        write_jsonl(event_path, events)
        end_ns = events[-1]["t_ns"]
        resources = self.make_resources(run_id, end_ns)
        gpu_identity_raw = (
            f"{GPU_UUID},NVIDIA GeForce RTX 4060 Ti,"
            "00000000:01:00.0,0\n"
        ).encode("ascii")
        gpu_process_argv = ["/fixture/gpu-server", "--model", "fixture"]
        gpu_process_raw = (
            f"4321,{GPU_UUID},/fixture/gpu-server,100\n"
        ).encode("ascii")
        stat_fields = [
            "S", *[str(value) for value in range(1, 19)], "777",
        ]
        gpu_stat_raw = (
            f"4321 (gpu server) {' '.join(stat_fields)}\n"
        ).encode("ascii")

        def gpu_sample(sequence, started, busy):
            process_raw = gpu_process_raw if busy else b""
            observations = (
                [{
                    "cmdline_base64": base64.b64encode(
                        b"\0".join(
                            item.encode("utf-8")
                            for item in gpu_process_argv
                        ) + b"\0"
                    ).decode("ascii"),
                    "pid": 4321,
                    "stat_base64":
                        base64.b64encode(gpu_stat_raw).decode("ascii"),
                }]
                if busy else []
            )
            return {
                "completed_ns": started + 10,
                "identity_stderr_base64": "",
                "identity_stdout_base64":
                    base64.b64encode(gpu_identity_raw).decode("ascii"),
                "process_observations": observations,
                "process_stderr_base64": "",
                "process_stdout_base64":
                    base64.b64encode(process_raw).decode("ascii"),
                "sequence": sequence,
                "started_ns": started,
                "type": "SAMPLE",
            }

        gpu_rows = [{
            "gpu_name": "NVIDIA GeForce RTX 4060 Ti",
            "gpu_uuid": GPU_UUID,
            "host_boot_id": "boot-host",
            "identity_argv": [
                str(nvidia_smi),
                "--id",
                GPU_UUID,
                "--query-gpu=uuid,name,pci.bus_id,index",
                "--format=csv,noheader,nounits",
            ],
            "nvidia_smi_bytes": nvidia_smi.stat().st_size,
            "nvidia_smi_path": str(nvidia_smi),
            "nvidia_smi_sha256": digest_file(nvidia_smi),
            "process_argv": [
                str(nvidia_smi),
                "--id",
                GPU_UUID,
                "--query-compute-apps="
                "pid,gpu_uuid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            "run_id": run_id,
            "schema": "s40-selected-gpu-observer-v1",
            "started_ns": 1,
            "type": "START",
        }]
        gpu_rows.append(gpu_sample(0, 10, False))
        sequence = 1
        point = 100
        while point <= end_ns:
            gpu_rows.append(gpu_sample(sequence, point, True))
            sequence += 1
            point += 500_000_000
        final_gpu_started = max(end_ns + 100, point)
        gpu_rows.append(gpu_sample(sequence, final_gpu_started, False))
        gpu_rows.append({
            "completed_ns": final_gpu_started + 20,
            "sample_count": sequence + 1,
            "type": "STOP",
        })
        gpu_observer_raw = b"".join(
            canonical_bytes(row) for row in gpu_rows)
        gpu_lock_record = {
            "acquired_ns": 1,
            "device": 1,
            "gpu_uuid": GPU_UUID,
            "host_boot_id": "boot-host",
            "inode": 1,
            "lock_path": str(canonical_lock_path(GPU_UUID)),
            "owner_pid": 999,
            "owner_start_ticks": 999,
            "released_ns": final_gpu_started + 30,
            "run_id": run_id,
            "schema": "s40-selected-gpu-lock-v1",
        }

        trace_start = {
            "active_drain_deadline_ns": 1_021_000_000_100,
            "campaign_horizon_ns": 901_000_000_100,
            "campaign_horizon_us": 900_000_000,
            "created_ns": 100,
            "drain_bound_us": 120_000_000,
            "event_log_run_start_ns": 0,
            "experiment_contract_sha256": digest_file(DEFAULT_CONTRACT),
            "host_boot_id": "boot-host",
            "requests_sha256":
                "c6b99c54bf55ffc885057d074780eea99000bceace9eedb64c4ce7eb6c537145",
            "run_id": run_id,
            "runtime_config_sha256": runtime_sha,
            "schema": "s40-trace-start-v1",
            "trace_origin_ns": origin_ns,
            "trace_start_lead_us": 1_000_000,
        }
        requests = read_jsonl(
            S40.parent / "s39_desktop_swap_baseline"
            / "DESKTOP_REQUESTS.jsonl",
            "requests",
        )
        http_rows, acquisition_result = self.make_http(
            run_id, trace_start, requests)
        activation_rows = []

        def activation(method, started, request, response):
            request_raw = (
                canonical_bytes(request) if request is not None else b"")
            response_raw = canonical_bytes(response)
            activation_rows.append({
                "http_status": 200,
                "method": method,
                "path": "/experimental/warm-tier/activate",
                "request_body_base64":
                    base64.b64encode(request_raw).decode("ascii"),
                "request_body_sha256": digest_bytes(request_raw),
                "response_body_base64":
                    base64.b64encode(response_raw).decode("ascii"),
                "response_body_sha256": digest_bytes(response_raw),
                "run_id": run_id,
                "schema": "s40-activation-http-evidence-v1",
                "sequence": len(activation_rows),
                "t_end_ns": started + 5,
                "t_start_ns": started,
            })

        activation("GET", 30, None, {
            "controller_epoch": 0,
            "schema": "llama-server-warm-tier-activate-status-v1",
            "state": "WAITING",
        })
        activation("POST", 40, {
            "schema": "llama-server-warm-tier-activate-v1",
        }, {
            "controller_epoch": 0,
            "schema": "llama-server-warm-tier-activate-result-v1",
            "state": "READY",
        })
        activation("GET", 50, None, {
            "controller_epoch": 0,
            "schema": "llama-server-warm-tier-activate-status-v1",
            "state": "READY",
        })
        activation_raw = b"".join(
            canonical_bytes(row) for row in activation_rows)
        bridge_rows = []
        for bridge_mode, base in (
                ("DIRECT_SOCKET", 100_000),
                ("FRESH_PYTHON_BRIDGE", 200_000)):
            for fanout in (1, 8):
                for sample in range(50):
                    bridge_rows.append({
                        "batch_index": sample,
                        "batch_makespan_ns": base + sample,
                        "command_bytes": 100,
                        "fanout": fanout,
                        "item_index": 0,
                        "latency_ns": base + sample,
                        "mode": bridge_mode,
                        "result_bytes": 100,
                    })
        native_invocations = []
        for fanout in (1, 8):
            sample_count = 50 if fanout == 1 else 56
            native_rows = []
            for item_index in range(sample_count):
                batch_index = item_index // fanout
                started_ns = 10_000_000 + item_index * 1_000
                latency_ns = 120_000
                native_rows.append({
                    "batch_index": batch_index,
                    "batch_makespan_ns": latency_ns,
                    "command_bytes": 100,
                    "command_id": item_index + 1,
                    "completed_ns": started_ns + latency_ns,
                    "item_index": item_index % fanout,
                    "latency_ns": latency_ns,
                    "result_bytes": 100,
                    "started_ns": started_ns,
                })
            native_value = {
                "fanout": fanout,
                "rows": native_rows,
                "sample_count": sample_count,
                "schema": "s40-native-unix-executor-bench-v1",
                "socket_path": str(root / "noop.sock"),
                "transport": "UNIX_SOCKET",
            }
            native_stdout = canonical_bytes(native_value)
            native_invocations.append({
                "argv": [
                    str(native_binary),
                    "--unix-bench",
                    str(root / "noop.sock"),
                    str(sample_count),
                    str(fanout),
                    "2345",
                    "3456",
                ],
                "completed_ns": 2_000_000 + fanout,
                "exit_code": 0,
                "fanout": fanout,
                "peer_pid": 2345,
                "peer_start_time_ticks": 3456,
                "started_ns": 1_000_000 + fanout,
                "stderr_base64": "",
                "stdout_base64":
                    base64.b64encode(native_stdout).decode("ascii"),
            })
        bridge_measurement = {
            "executor_bundle_manifest_sha256":
                digest_file(root / "executor-bundle" / "MANIFEST.json"),
            "host_boot_id": "boot-host",
            "native_bench_binary": {
                "bytes": native_binary.stat().st_size,
                "path": str(native_binary),
                "sha256": digest_file(native_binary),
            },
            "native_invocations": native_invocations,
            "python_executable_sha256":
                digest_file(Path(sys.executable)),
            "rows": bridge_rows,
            "schema": "s40-transport-overhead-v2",
            "summary": summarize_bridge_overhead(bridge_rows),
        }
        runtime_library_path = root / "captured-runtime" / "lib"
        runtime_library_path.mkdir(parents=True)
        for directory in ("cuda-cache", "run-home", "run-tmp"):
            (root / directory).mkdir()
        launch_environment = {
            "CUDA_CACHE_PATH": str(root / "cuda-cache"),
            "CUDA_VISIBLE_DEVICES": GPU_UUID,
            "HOME": str(root / "run-home"),
            "LANG": "C",
            "LC_ALL": "C",
            "LD_LIBRARY_PATH": str(runtime_library_path),
            "LLAMA_SERVER_WARM_TIER_CONFIG": str(runtime_path),
            "LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE":
                str(root / "warm-tier-internal.token"),
            "NVIDIA_VISIBLE_DEVICES": GPU_UUID,
            "PATH": f"{root / 'captured-runtime' / 'bin'}:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "S40_EVIDENCE_BUNDLE": "1",
            "S40_EVIDENCE_BUNDLE_MANIFEST":
                str(root / "evidence-bundle" / "MANIFEST.json"),
            "S40_EVIDENCE_BUNDLE_SHA256":
                digest_file(root / "evidence-bundle" / "MANIFEST.json"),
            "S40_EXECUTOR_BUNDLE": "1",
            "S40_EXECUTOR_BUNDLE_MANIFEST":
                str(root / "executor-bundle" / "MANIFEST.json"),
            "S40_EXECUTOR_BUNDLE_SHA256":
                digest_file(root / "executor-bundle" / "MANIFEST.json"),
            "S40_NVIDIA_SMI_PATH": str(nvidia_smi),
            "S40_NVIDIA_SMI_SHA256": digest_file(nvidia_smi),
            "TMPDIR": str(root / "run-tmp"),
            "TZ": "UTC",
        }
        cmdline = b"".join(
            argument.encode("ascii") + b"\x00"
            for argument in gateway_argv
        )
        executable_stat = python_binary.stat(follow_symlinks=False)

        def gateway_identity(observed_ns):
            return {
                "cmdline_base64": base64.b64encode(cmdline).decode("ascii"),
                "cmdline_sha256": digest_bytes(cmdline),
                "executable_ctime_ns": executable_stat.st_ctime_ns,
                "executable_device": executable_stat.st_dev,
                "executable_inode": executable_stat.st_ino,
                "executable_mtime_ns": executable_stat.st_mtime_ns,
                "executable_path": str(python_binary),
                "executable_sha256": digest_file(python_binary),
                "executable_size": executable_stat.st_size,
                "gateway_pid": 2345,
                "gateway_start_time_ticks": 3456,
                "observed_ns": observed_ns,
            }

        prepublication_identity = gateway_identity(1)
        post_auth_identity = gateway_identity(42)
        final_identity = gateway_identity(end_ns)

        def executed_file(argv_index, source):
            metadata = source.stat(follow_symlinks=False)
            return {
                "argv_index": argv_index,
                "bytes": metadata.st_size,
                "captured_path": str(source.relative_to(root)),
                "executed_path": str(source),
                "sha256": digest_file(source),
                "source_ctime_ns": metadata.st_ctime_ns,
                "source_device": metadata.st_dev,
                "source_inode": metadata.st_ino,
                "source_mtime_ns": metadata.st_mtime_ns,
                "source_size": metadata.st_size,
            }

        gateway_executed_files = [
            executed_file(0, python_binary),
            executed_file(
                7, root / "executor-bundle" / "desktop_gateway.py"),
            executed_file(gateway_argv.index(str(gateway_config)),
                          gateway_config),
        ]
        runtime_dependencies = root / "runtime-dependencies.json"
        runtime_dependencies.write_bytes(canonical_bytes({
            "binaries": [{
                "bytes": source.stat().st_size,
                "captured_path": source.name,
                "label": label,
                "sha256": digest_file(source),
                "source_path": str(source),
            } for label, source in (
                ("controller", binary),
                ("native_bench", native_binary),
                ("nvidia_smi", nvidia_smi),
                ("python", python_binary),
            )],
            "dependencies": [],
            "library_path": str(runtime_library_path),
            "schema": "s40-captured-runtime-v1",
        }))
        artifacts = [
            self.artifact(
                root, "activation_evidence", "activation.jsonl", "JSONL",
                activation_raw),
            self.artifact(
                root, "native_bench_binary",
                "test-warm-tier-executors", "BINARY",
                native_binary.read_bytes()),
            self.artifact(
                root, "runtime_dependency_manifest",
                "runtime-dependencies.json", "JSON",
                runtime_dependencies.read_bytes()),
            self.artifact(
                root, "transport_overhead", "transport-overhead.json", "JSON",
                canonical_bytes(bridge_measurement)),
            self.artifact(
                root, "transport_overhead_stderr",
                "transport-overhead.stderr", "TEXT", b""),
            self.artifact(
                root, "transport_overhead_stdout",
                "transport-overhead.stdout", "TEXT", b""),
            self.artifact(
                root, "controller_identity",
                "controller-identity-lock.json", "JSON",
                controller_identity_path.read_bytes()),
            self.artifact(
                root, "controller_launch", "controller-launch.json", "JSON",
                canonical_bytes({
                    "binary_sha256": digest_file(binary),
                    "command_argv": server_argv,
                    "controller_identity_device":
                        controller_identity_stat.st_dev,
                    "controller_identity_inode":
                        controller_identity_stat.st_ino,
                    "controller_identity_path":
                        str(controller_identity_path),
                    "controller_identity_sha256":
                        digest_file(controller_identity_path),
                    "controller_pid": 1234,
                    "controller_start_ticks": 99,
                    "experiment_contract_sha256":
                        digest_file(DEFAULT_CONTRACT),
                    "exit_code": 0,
                    "host_boot_id": "boot-host",
                    "launch_environment": launch_environment,
                    "library_path": str(runtime_library_path),
                    "physical_plan_sha256": digest_file(source_physical),
                    "run_id": run_id,
                    "runtime_config_sha256": runtime_sha,
                    "schema": "s40-controller-launch-v4",
                    "selected_gpu_environment": {
                        "CUDA_VISIBLE_DEVICES": GPU_UUID,
                        "NVIDIA_VISIBLE_DEVICES": GPU_UUID,
                    },
                    "source_preflight_sha256": digest_file(source_preflight),
                    "started_ns": 20,
                    "stopped_ns": end_ns,
                })),
            self.artifact(
                root, "controller_events", "events.jsonl", "JSONL",
                event_path.read_bytes()),
            self.artifact(
                root, "evidence_root", "evidence-root.json", "JSON",
                evidence_root.read_bytes()),
            self.artifact(
                root, "evidence_bundle_manifest",
                "evidence-bundle/MANIFEST.json", "JSON",
                (root / "evidence-bundle" / "MANIFEST.json").read_bytes()),
            self.artifact(
                root, "executor_bundle_manifest",
                "executor-bundle/MANIFEST.json", "JSON",
                (root / "executor-bundle" / "MANIFEST.json").read_bytes()),
            self.artifact(
                root, "http_evidence", "http.jsonl", "JSONL",
                b"".join(canonical_bytes(row) for row in http_rows)),
            self.artifact(
                root, "gpu_lock_record", "gpu-lock-record.json", "JSON",
                canonical_bytes(gpu_lock_record)),
            self.artifact(
                root, "gpu_observer", "gpu-observer.jsonl", "JSONL",
                gpu_observer_raw),
            self.artifact(
                root, "gpu_observer_argv", "gpu-observer-argv.json", "JSON",
                canonical_bytes({
                    "argv": _isolated_python_argv(
                        python_binary,
                        root / "evidence-bundle",
                        root / "evidence-bundle" / "gpu_isolation.py",
                        [
                            "--output", str(root / "gpu-observer.jsonl"),
                            "--stop-file", str(root / "gpu-observer.stop"),
                            "--lock-path",
                            str(canonical_lock_path(GPU_UUID)),
                            "--lock-output",
                            str(root / "gpu-lock-record.json"),
                            "--run-id", run_id,
                            "--gpu-uuid", GPU_UUID,
                            "--gpu-name", "NVIDIA GeForce RTX 4060 Ti",
                            "--nvidia-smi", str(nvidia_smi),
                            "--nvidia-smi-sha256",
                            digest_file(nvidia_smi),
                            "--interval-ms", "200",
                        ],
                    ),
                    "schema": "s40-gpu-observer-argv-v1",
                })),
            self.artifact(
                root, "gpu_observer_stderr",
                "gpu-observer.stderr", "TEXT", b""),
            self.artifact(
                root, "gpu_observer_stdout",
                "gpu-observer.stdout", "TEXT", b""),
            self.artifact(
                root, "orchestrator_evidence", "orchestrator.json", "JSON",
                canonical_bytes({
                    "acquisition_returncode": 0,
                    "cleanup": [{
                        "exit_code": 0,
                        "name": "gateway::gpu0",
                    }, {
                        "exit_code": 0,
                        "name": "controller",
                    }, {
                        "exit_code": 0,
                        "name": "gpu_observer",
                    }],
                    "completed_ns": final_gpu_started + 40,
                    "run_id": run_id,
                    "schema": "s40-orchestrator-evidence-v2",
                })),
            self.artifact(
                root, "resource_samples", "resources.jsonl", "JSONL",
                b"".join(canonical_bytes(row) for row in resources)),
            self.artifact(
                root, "resource_sampler_stderr", "sampler.stderr", "TEXT",
                b""),
            self.artifact(
                root, "resource_sampler_stdout", "sampler.stdout", "TEXT",
                b""),
            self.artifact(
                root, "resource_sampler_argv", "sampler-argv.json", "JSON",
                canonical_bytes({
                    "argv": _isolated_python_argv(
                        python_binary,
                        root / "evidence-bundle",
                        root / "evidence-bundle" / "resource_sampler.py",
                        [
                            "--run-id", run_id,
                            "--gpu-uuid", GPU_UUID,
                            "--controller-pid", "1234",
                            "--output", str(root / "resources.jsonl"),
                            "--stop-file",
                            str(root / "resource-sampler.stop"),
                            "--interval-ms", "200",
                            "--nvidia-smi-path", str(nvidia_smi),
                            "--nvidia-smi-sha256",
                            digest_file(nvidia_smi),
                        ],
                    ),
                    "schema": "s40-resource-sampler-argv-v1",
                })),
            self.artifact(
                root, "runtime_config", "runtime.json", "JSON",
                runtime_path.read_bytes()),
            self.artifact(
                root, "runtime_plan", "runtime-plan.json", "JSON",
                plan_path.read_bytes()),
            self.artifact(
                root, "server_stderr", "server.stderr", "TEXT", b""),
            self.artifact(
                root, "server_readiness", "readiness.json", "JSON",
                canonical_bytes({
                    "activation_evidence_sha256":
                        digest_bytes(activation_raw),
                    "activation_ready": {
                        "controller_epoch": 0,
                        "schema":
                            "llama-server-warm-tier-activate-status-v1",
                        "state": "READY",
                    },
                    "base_readiness": {
                        "completed_ns": 25,
                        "http_status": 200,
                        "response": {
                            "controller_epoch": 0,
                            "schema":
                                "llama-server-warm-tier-activate-status-v1",
                            "state": "WAITING",
                        },
                        "response_sha256": "0" * 64,
                        "schema": "s40-server-readiness-v1",
                        "started_ns": 22,
                    },
                    "controller_bindings": [{
                        "device": controller_binding_stat.st_dev,
                        "executor_id": "gpu0",
                        "inode": controller_binding_stat.st_ino,
                        "path": str(controller_binding_path),
                        "sha256": digest_file(controller_binding_path),
                    }],
                    "controller_identity": {
                        "captured_ns": 20,
                        "device": controller_identity_stat.st_dev,
                        "inode": controller_identity_stat.st_ino,
                        "path": str(controller_identity_path),
                        "published_ns": 21,
                        "sha256": digest_file(controller_identity_path),
                    },
                    "controller_started_ns": 20,
                    "gateway_final": [{
                        "executor_id": "gpu0",
                        "identity": final_identity,
                    }],
                    "gateway_post_auth": [{
                        "executor_id": "gpu0",
                        "identity": post_auth_identity,
                    }],
                    "gateway_ready": [{
                        "executor_id": "gpu0",
                        "executor_instance_id": "instance-gpu0",
                        "identity_captured_ns": 1,
                        "pid": 2345,
                        "process_identity": prepublication_identity,
                        "process_start_time_ticks": 3456,
                        "socket_path": str(root / "gpu.sock"),
                        "t_ns": 3,
                    }],
                    "host_boot_id": "boot-host",
                    "launch_environment": launch_environment,
                    "phone_observer": None,
                    "run_id": run_id,
                    "runtime_config_device":
                        runtime_path.stat(follow_symlinks=False).st_dev,
                    "runtime_config_inode":
                        runtime_path.stat(follow_symlinks=False).st_ino,
                    "runtime_config_path": str(runtime_path),
                    "runtime_config_published_ns": 2,
                    "runtime_config_sha256": runtime_sha,
                    "schema": "s40-server-readiness-v6",
                })),
            self.artifact(
                root, "server_stdout", "server.stdout", "TEXT", b""),
            self.artifact(
                root, "source_preflight", "source-preflight.json", "JSON",
                source_preflight.read_bytes()),
            self.artifact(
                root, "trace_driver_stderr", "driver.stderr", "TEXT", b""),
            self.artifact(
                root, "trace_driver_stdout", "driver.stdout", "TEXT", b""),
            self.artifact(
                root, "trace_acquisition_result", "acquire.json", "JSON",
                canonical_bytes(acquisition_result)),
            self.artifact(
                root, "trace_start", "trace-start.json", "JSON",
                canonical_bytes(trace_start)),
            self.artifact(
                root, "executor_gateway_argv::gpu0",
                "gateway-argv.json", "JSON",
                canonical_bytes({
                    "argv": gateway_argv,
                    "schema": "s40-gateway-argv-v4",
                })),
            self.artifact(
                root, "executor_command::gpu0", "gpu-command.jsonl",
                "JSONL", b'{"started_ns":42}\n'),
            self.artifact(
                root, "executor_controller_binding::gpu0",
                "executor-controller-binding-gpu0.json",
                "JSON", controller_binding_path.read_bytes()),
            self.artifact(
                root, "executor_transport::gpu0",
                "transport-descriptor.json", "JSON",
                canonical_bytes({
                    "controller_executable_path": str(binary),
                    "controller_executable_sha256": digest_file(binary),
                    "controller_gid": 1000,
                    "controller_binding_device":
                        controller_binding_stat.st_dev,
                    "controller_binding_inode":
                        controller_binding_stat.st_ino,
                    "controller_binding_path":
                        str(controller_binding_path),
                    "controller_binding_sha256":
                        digest_file(controller_binding_path),
                    "controller_identity_device":
                        controller_identity_stat.st_dev,
                    "controller_identity_inode":
                        controller_identity_stat.st_ino,
                    "controller_identity_path":
                        str(controller_identity_path),
                    "controller_identity_published_ns": 21,
                    "controller_identity_sha256":
                        digest_file(controller_identity_path),
                    "controller_pid": 1234,
                    "controller_start_time_ticks": 99,
                    "controller_uid": 1000,
                    "executor_bundle_manifest_sha256":
                        digest_file(root / "executor-bundle"
                                    / "MANIFEST.json"),
                    "executor_id": "gpu0",
                    "executor_instance_id": "instance-gpu0",
                    "gateway_argv_sha256":
                        digest_file(root / "gateway-argv.json"),
                    "gateway_config_sha256": digest_file(gateway_config),
                    "gateway_environment": launch_environment,
                    "gateway_executed_files": gateway_executed_files,
                    "gateway_pid": 2345,
                    "gateway_post_auth_identity": post_auth_identity,
                    "gateway_prepublication_identity":
                        prepublication_identity,
                    "gateway_source_sha256":
                        digest_file(root / "executor-bundle"
                                    / "desktop_gateway.py"),
                    "gateway_start_time_ticks": 3456,
                    "host_boot_id": "boot-host",
                    "identity_captured_ns": 1,
                    "runtime_config_device": runtime_stat.st_dev,
                    "runtime_config_inode": runtime_stat.st_ino,
                    "runtime_config_path": str(runtime_path),
                    "runtime_config_published_ns": 2,
                    "runtime_config_sha256": runtime_sha,
                    "schema": "s40-executor-transport-descriptor-v4",
                    "socket_path": str(root / "gpu.sock"),
                    "transport": "UNIX_SOCKET",
                })),
            self.artifact(
                root, "gateway_stderr::gpu0", "gateway.stderr", "TEXT", b""),
            self.artifact(
                root, "gateway_stdout::gpu0", "gateway.stdout", "TEXT", b""),
            self.artifact(
                root, "gateway_config::gpu0", "gateway-config.json", "JSON",
                gateway_config.read_bytes()),
        ]
        for row in bundle["files"]:
            name = row["name"]
            artifacts.append(self.artifact(
                root,
                f"executor_bundle::{name}",
                f"executor-bundle/{name}",
                "TEXT",
                (root / "executor-bundle" / name).read_bytes(),
            ))
        for row in evidence_bundle["files"]:
            name = row["name"]
            artifacts.append(self.artifact(
                root,
                f"evidence_bundle::{name}",
                f"evidence-bundle/{name}",
                "TEXT",
                (root / "evidence-bundle" / name).read_bytes(),
            ))
        manifest = {
            "artifacts": artifacts,
            "cache_regime": "WARM_HOST_CACHE",
            "campaign_binding": None,
            "command_argv": server_argv,
            "controller_binary": {
                "bytes": binary.stat().st_size,
                "captured_path": binary.name,
                "executed_path": str(binary),
                "sha256": digest_file(binary),
            },
            "development": True,
            "devices": [{
                "boot_id": "boot-host",
                "device_role": "GPU",
                "stable_id": GPU_UUID,
            }],
            "executor_bindings": [{
                "command_role": "executor_command::gpu0",
                "controller_binding_role":
                    "executor_controller_binding::gpu0",
                "executor_id": "gpu0",
                "gateway_argv": gateway_argv,
                "gateway_argv_role": "executor_gateway_argv::gpu0",
                "gateway_config_role": "gateway_config::gpu0",
                "gateway_executed_files": gateway_executed_files,
                "gateway_final_identity": final_identity,
                "gateway_launch_environment": launch_environment,
                "gateway_post_auth_identity": post_auth_identity,
                "gateway_prepublication_identity":
                    prepublication_identity,
                "gateway_source_role":
                    "executor_bundle::desktop_gateway.py",
                "gateway_stderr_role": "gateway_stderr::gpu0",
                "gateway_stdout_role": "gateway_stdout::gpu0",
                "transport_descriptor_role": "executor_transport::gpu0",
            }],
            "experiment_contract_sha256": digest_file(DEFAULT_CONTRACT),
            "mode": "C1_GPU_ONLY_OPTIMIZED",
            "policy_id": "s40-oldest-demand-work-conserving-v1",
            "repeat_index": 0,
            "run_id": run_id,
            "schema": "s40-physical-run-manifest-v6",
            "schema_version": 6,
        }
        path = root / "manifest.json"
        path.write_bytes(canonical_bytes(manifest))
        return path, manifest

    def write(self, path, manifest):
        path.write_bytes(canonical_bytes(manifest))

    def refresh_artifact(self, root, manifest, role):
        item = next(
            row for row in manifest["artifacts"] if row["role"] == role)
        artifact_path = root / item["path"]
        item["bytes"] = artifact_path.stat().st_size
        item["sha256"] = digest_file(artifact_path)
        if item["format"] == "JSONL":
            item["record_count"] = len(artifact_path.read_bytes().splitlines())

    def mutate_json_artifact(self, root, manifest, role, mutate):
        item = next(
            row for row in manifest["artifacts"] if row["role"] == role)
        artifact_path = root / item["path"]
        value = __import__("json").loads(
            artifact_path.read_text(encoding="ascii"))
        mutate(value)
        artifact_path.write_bytes(canonical_bytes(value))
        self.refresh_artifact(root, manifest, role)

    def rewrite_orchestrator(self, root, manifest, value):
        item = next(
            row for row in manifest["artifacts"]
            if row["role"] == "orchestrator_evidence"
        )
        (root / item["path"]).write_bytes(canonical_bytes(value))
        self.refresh_artifact(root, manifest, "orchestrator_evidence")

    def rewrite_server_argv(
            self,
            root: Path,
            manifest: dict,
            source_argv: list[str]) -> None:
        source_physical_path = root / "source-physical-plan.json"
        source_physical = __import__("json").loads(
            source_physical_path.read_text(encoding="ascii"))
        source_physical["server_argv"] = source_argv
        source_physical_path.write_bytes(canonical_bytes(source_physical))

        source_preflight_path = root / "source-preflight.json"
        source_preflight = __import__("json").loads(
            source_preflight_path.read_text(encoding="ascii"))
        physical_row = next(
            row for row in source_preflight["rows"]
            if row["role"] == "physical_plan")
        physical_capture = root / physical_row["captured_path"]
        physical_capture.write_bytes(source_physical_path.read_bytes())
        physical_row["bytes"] = physical_capture.stat().st_size
        physical_row["sha256"] = digest_file(physical_capture)
        source_preflight_path.write_bytes(canonical_bytes(source_preflight))
        self.refresh_artifact(root, manifest, "source_preflight")

        manifest["command_argv"] = [
            manifest["command_argv"][0],
            *source_argv[1:],
        ]
        launch_path = root / "controller-launch.json"
        launch = __import__("json").loads(
            launch_path.read_text(encoding="ascii"))
        launch["command_argv"] = manifest["command_argv"]
        launch["physical_plan_sha256"] = digest_file(source_physical_path)
        launch["source_preflight_sha256"] = digest_file(
            source_preflight_path)
        launch_path.write_bytes(canonical_bytes(launch))
        self.refresh_artifact(root, manifest, "controller_launch")

    def bind_primary_campaign(self, root, manifest):
        campaign = build_campaign(
            1,
            campaign_id="fixture",
            experiment_contract_sha256=digest_file(DEFAULT_CONTRACT),
            software_lock={
                **{
                    f"{name}_sha256": digest_file(tool)
                    for name, tool in CAMPAIGN_TOOL_PATHS.items()
                },
                "controller_binary_sha256":
                    manifest["controller_binary"]["sha256"],
                "evidence_bundle_manifest_sha256":
                    next(
                        row["sha256"] for row in manifest["artifacts"]
                        if row["role"] == "evidence_bundle_manifest"),
                "executor_bundle_manifest_sha256":
                    next(
                        row["sha256"] for row in manifest["artifacts"]
                        if row["role"] == "executor_bundle_manifest"),
                "ldd_sha256": digest_file(root / "ldd"),
                "native_bench_binary_sha256":
                    next(
                        row["sha256"] for row in manifest["artifacts"]
                        if row["role"] == "native_bench_binary"),
                "nvidia_smi_sha256": digest_file(root / "nvidia-smi"),
                "python_sha256": digest_file(root / "python-captured"),
                "schema": "s40-primary-software-lock-v2",
            },
        )
        campaign_path = root / "campaign.json"
        campaign_path.write_bytes(canonical_bytes(campaign))
        binding = {
            "campaign_id": "fixture",
            "campaign_sha256": digest_file(campaign_path),
            "order": 0,
            "phase": "PRIMARY",
        }
        source_physical_path = root / "source-physical-plan.json"
        source_physical = __import__("json").loads(
            source_physical_path.read_text(encoding="ascii"))
        source_physical["campaign_binding"] = {
            "campaign_path": str(campaign_path),
            "campaign_sha256": binding["campaign_sha256"],
            "order": 0,
            "phase": "PRIMARY",
        }
        source_physical["development"] = False
        source_physical_path.write_bytes(canonical_bytes(source_physical))

        source_preflight_path = root / "source-preflight.json"
        source_preflight = __import__("json").loads(
            source_preflight_path.read_text(encoding="ascii"))
        physical_row = next(
            row for row in source_preflight["rows"]
            if row["role"] == "physical_plan")
        physical_capture = root / physical_row["captured_path"]
        physical_capture.write_bytes(source_physical_path.read_bytes())
        physical_row["bytes"] = physical_capture.stat().st_size
        physical_row["sha256"] = digest_file(physical_capture)
        campaign_capture = (
            root / "source-locks" / "campaign-plan.artifact")
        campaign_capture.write_bytes(campaign_path.read_bytes())
        source_preflight["rows"].append({
            "bytes": campaign_capture.stat().st_size,
            "captured_path": str(campaign_capture.relative_to(root)),
            "role": "campaign_plan",
            "sha256": digest_file(campaign_capture),
            "source_path": str(campaign_path),
        })
        source_preflight_path.write_bytes(canonical_bytes(source_preflight))
        self.refresh_artifact(root, manifest, "source_preflight")

        launch_path = root / "controller-launch.json"
        launch = __import__("json").loads(
            launch_path.read_text(encoding="ascii"))
        launch["physical_plan_sha256"] = digest_file(source_physical_path)
        launch["source_preflight_sha256"] = digest_file(
            source_preflight_path)
        launch_path.write_bytes(canonical_bytes(launch))
        self.refresh_artifact(root, manifest, "controller_launch")
        manifest["campaign_binding"] = binding
        manifest["development"] = False
        return campaign_path

    def test_valid_manifest(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            path, _ = self.make_fixture(Path(directory))
            result = validate_run_manifest(path)
            self.assertEqual(result["status"], "S40_RUN_MANIFEST_V6_VALID")
            self.assertEqual(result["executor_ids"], ["gpu0"])

    def test_manifest_rejects_persisted_internal_token(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, _ = self.make_fixture(root)
            token = root / "warm-tier-internal.token"
            token.write_text("a" * 64 + "\n", encoding="ascii")
            token.chmod(0o600)
            with self.assertRaisesRegex(
                    EvidenceError, "internal token was not removed"):
                validate_run_manifest(path)

    def test_v4_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            path, manifest = self.make_fixture(Path(directory))
            manifest["schema"] = "s40-physical-run-manifest-v4"
            manifest["schema_version"] = 4
            self.write(path, manifest)
            with self.assertRaisesRegex(EvidenceError, "unsupported identity"):
                validate_run_manifest(path)

    def test_controller_authentication_roles_are_mandatory(self):
        for role in (
                "controller_identity",
                "executor_controller_binding::gpu0"):
            with self.subTest(role=role), tempfile.TemporaryDirectory(
                    prefix="s40_manifest_") as directory:
                path, manifest = self.make_fixture(Path(directory))
                manifest["artifacts"] = [
                    row for row in manifest["artifacts"]
                    if row["role"] != role
                ]
                self.write(path, manifest)
                with self.assertRaisesRegex(
                        EvidenceError,
                        "artifact role set mismatch|missing artifact"):
                    validate_run_manifest(path)

    def test_controller_peer_and_runtime_binding_are_load_bearing(self):
        mutations = (
            (
                "peer PID",
                lambda value: value.update(peer_pid=4321),
                "bound identity mismatch",
            ),
            (
                "runtime digest",
                lambda value: value.update(runtime_config_sha256="f" * 64),
                "bound identity mismatch",
            ),
            (
                "executor instance",
                lambda value: value.update(executor_instance_id="foreign"),
                "bound identity mismatch",
            ),
            (
                "late authentication",
                lambda value: value.update(authenticated_ns=56),
                "outside activation",
            ),
            (
                "pre-activation authentication",
                lambda value: value.update(authenticated_ns=22),
                "authenticated_ns",
            ),
        )
        for label, mutate, message in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                    prefix="s40_manifest_") as directory:
                root = Path(directory)
                path, manifest = self.make_fixture(root)
                self.mutate_json_artifact(
                    root,
                    manifest,
                    "executor_controller_binding::gpu0",
                    mutate,
                )
                self.write(path, manifest)
                with self.assertRaisesRegex(EvidenceError, message):
                    validate_run_manifest(path)

    def test_command_must_follow_controller_authentication(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            command_path = root / "gpu-command.jsonl"
            command_path.write_bytes(b'{"started_ns":40}\n')
            self.refresh_artifact(
                root, manifest, "executor_command::gpu0")
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "command predates"):
                validate_run_manifest(path)

    def test_controller_publication_and_launch_binding_are_load_bearing(self):
        mutations = (
            (
                "publication digest",
                "server_readiness",
                lambda value: value["controller_identity"].update(
                    sha256="f" * 64),
                "publication mismatch",
            ),
            (
                "binding readiness digest",
                "server_readiness",
                lambda value: value["controller_bindings"][0].update(
                    sha256="f" * 64),
                "readiness binding mismatch",
            ),
            (
                "launch digest",
                "controller_launch",
                lambda value: value.update(
                    controller_identity_sha256="f" * 64),
                "controller identity mismatch",
            ),
            (
                "descriptor binding digest",
                "executor_transport::gpu0",
                lambda value: value.update(
                    controller_binding_sha256="f" * 64),
                "controller binding mismatch",
            ),
        )
        for label, role, mutate, message in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                    prefix="s40_manifest_") as directory:
                root = Path(directory)
                path, manifest = self.make_fixture(root)
                self.mutate_json_artifact(root, manifest, role, mutate)
                self.write(path, manifest)
                with self.assertRaisesRegex(EvidenceError, message):
                    validate_run_manifest(path)

    def test_controller_authentication_wrong_types_fail_closed(self):
        mutations = (
            (
                "identity null path",
                "controller_identity",
                lambda value: value.update(runtime_config_path=None),
            ),
            (
                "readiness list path",
                "server_readiness",
                lambda value: value["controller_identity"].update(path=[]),
            ),
            (
                "binding float PID",
                "executor_controller_binding::gpu0",
                lambda value: value.update(peer_pid=1234.0),
            ),
            (
                "binding numeric digest",
                "executor_controller_binding::gpu0",
                lambda value: value.update(
                    controller_identity_sha256=1.0),
            ),
            (
                "descriptor list path",
                "executor_transport::gpu0",
                lambda value: value.update(controller_binding_path=[]),
            ),
            (
                "descriptor float device",
                "executor_transport::gpu0",
                lambda value: value.update(
                    controller_identity_device=float(
                        value["controller_identity_device"])),
            ),
        )
        for label, role, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                    prefix="s40_manifest_") as directory:
                root = Path(directory)
                path, manifest = self.make_fixture(root)
                self.mutate_json_artifact(root, manifest, role, mutate)
                self.write(path, manifest)
                with self.assertRaises(EvidenceError):
                    validate_run_manifest(path)

    def test_controller_launch_is_after_gateway_readiness(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            readiness_path = root / "readiness.json"
            readiness = __import__("json").loads(
                readiness_path.read_text(encoding="ascii"))
            readiness["controller_started_ns"] = 2
            readiness_path.write_bytes(canonical_bytes(readiness))
            self.refresh_artifact(root, manifest, "server_readiness")
            self.write(path, manifest)
            with self.assertRaisesRegex(EvidenceError, "startup ordering"):
                validate_run_manifest(path)

    def test_runtime_file_and_gateway_identity_are_load_bearing(self):
        mutations = {
            "device": lambda value: value.__setitem__(
                "runtime_config_device", value["runtime_config_device"] + 1),
            "inode": lambda value: value.__setitem__(
                "runtime_config_inode", value["runtime_config_inode"] + 1),
            "sha": lambda value: value.__setitem__(
                "runtime_config_sha256", "f" * 64),
            "gateway_pid": lambda value: value["gateway_ready"][0].__setitem__(
                "pid", value["gateway_ready"][0]["pid"] + 1),
            "gateway_start": lambda value:
                value["gateway_ready"][0].__setitem__(
                    "process_start_time_ticks",
                    value["gateway_ready"][0]["process_start_time_ticks"] + 1,
                ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                    prefix="s40_manifest_") as directory:
                root = Path(directory)
                path, manifest = self.make_fixture(root)
                readiness_path = root / "readiness.json"
                readiness = __import__("json").loads(
                    readiness_path.read_text(encoding="ascii"))
                mutate(readiness)
                readiness_path.write_bytes(canonical_bytes(readiness))
                self.refresh_artifact(root, manifest, "server_readiness")
                self.write(path, manifest)
                with self.assertRaises(EvidenceError):
                    validate_run_manifest(path)

    def test_coordinated_gateway_cmdline_forgery_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            forged = b"/forged/python\x00/forged/gateway.py\x00"

            def forge(identity):
                identity["cmdline_base64"] = (
                    base64.b64encode(forged).decode("ascii"))
                identity["cmdline_sha256"] = digest_bytes(forged)

            binding = manifest["executor_bindings"][0]
            for key in (
                    "gateway_prepublication_identity",
                    "gateway_post_auth_identity",
                    "gateway_final_identity"):
                forge(binding[key])
            self.mutate_json_artifact(
                root,
                manifest,
                "executor_transport::gpu0",
                lambda value: [
                    forge(value[key])
                    for key in (
                        "gateway_prepublication_identity",
                        "gateway_post_auth_identity",
                    )
                ],
            )
            self.mutate_json_artifact(
                root,
                manifest,
                "server_readiness",
                lambda value: [
                    forge(value["gateway_ready"][0]["process_identity"]),
                    forge(value["gateway_post_auth"][0]["identity"]),
                    forge(value["gateway_final"][0]["identity"]),
                ],
            )
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "cmdline mismatch"):
                validate_run_manifest(path)

    def test_coordinated_gateway_executable_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)

            def forge(identity):
                identity["executable_inode"] += 1

            binding = manifest["executor_bindings"][0]
            for key in (
                    "gateway_prepublication_identity",
                    "gateway_post_auth_identity",
                    "gateway_final_identity"):
                forge(binding[key])
            self.mutate_json_artifact(
                root,
                manifest,
                "executor_transport::gpu0",
                lambda value: [
                    forge(value[key])
                    for key in (
                        "gateway_prepublication_identity",
                        "gateway_post_auth_identity",
                    )
                ],
            )
            self.mutate_json_artifact(
                root,
                manifest,
                "server_readiness",
                lambda value: [
                    forge(value["gateway_ready"][0]["process_identity"]),
                    forge(value["gateway_post_auth"][0]["identity"]),
                    forge(value["gateway_final"][0]["identity"]),
                ],
            )
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError,
                    "live executable is not the prelaunch file"):
                validate_run_manifest(path)

    def test_coordinated_gateway_environment_injection_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)

            def inject(environment):
                environment["PYTHONPATH"] = "/tmp/attacker"

            inject(manifest["executor_bindings"][0][
                "gateway_launch_environment"])
            self.mutate_json_artifact(
                root,
                manifest,
                "executor_transport::gpu0",
                lambda value: inject(value["gateway_environment"]),
            )
            self.mutate_json_artifact(
                root,
                manifest,
                "server_readiness",
                lambda value: inject(value["launch_environment"]),
            )
            self.mutate_json_artifact(
                root,
                manifest,
                "controller_launch",
                lambda value: inject(value["launch_environment"]),
            )
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "invalid deterministic environment"):
                validate_run_manifest(path)

    def test_coordinated_prelaunch_file_metadata_forgery_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)

            manifest["executor_bindings"][0][
                "gateway_executed_files"][0]["source_inode"] += 1
            self.mutate_json_artifact(
                root,
                manifest,
                "executor_transport::gpu0",
                lambda value: value["gateway_executed_files"][0].__setitem__(
                    "source_inode",
                    value["gateway_executed_files"][0]["source_inode"] + 1,
                ),
            )
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError,
                    "executed file differs from captured bytes"):
                validate_run_manifest(path)

    def test_coordinated_serving_envelope_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            source_physical = __import__("json").loads(
                (root / "source-physical-plan.json").read_text(
                    encoding="ascii"))
            argv = source_physical["server_argv"]
            argv[argv.index("--ctx-size") + 1] = "2048"
            self.rewrite_server_argv(root, manifest, argv)
            self.write(path, manifest)
            with self.assertRaisesRegex(EvidenceError, "serving contract"):
                validate_run_manifest(path)

    def test_coordinated_forbidden_server_feature_is_rejected(self):
        mutations = (
            ["--ui-mcp-proxy"],
            ["--ui_mcp_proxy"],
            ["--webui-mcp-proxy"],
            ["--webui_mcp_proxy"],
            ["-ag"],
            ["--agent"],
            ["--tools", "exec_shell_command"],
            ["--ui-mcp-proxy=true"],
            ["--tools=exec_shell_command"],
        )
        for extra_argv in mutations:
            with self.subTest(extra_argv=extra_argv), tempfile.TemporaryDirectory(
                    prefix="s40_manifest_") as directory:
                root = Path(directory)
                path, manifest = self.make_fixture(root)
                source_physical = __import__("json").loads(
                    (root / "source-physical-plan.json").read_text(
                        encoding="ascii"))
                argv = [*source_physical["server_argv"], *extra_argv]
                self.rewrite_server_argv(root, manifest, argv)
                self.write(path, manifest)
                with self.assertRaisesRegex(
                        EvidenceError, "forbidden server feature"):
                    validate_run_manifest(path)

    def test_primary_manifest_binds_campaign_and_software(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            campaign_path = self.bind_primary_campaign(root, manifest)
            self.write(path, manifest)
            result = validate_run_manifest(path)
            self.assertEqual(
                result["campaign_binding"]["campaign_sha256"],
                digest_file(campaign_path),
            )
            campaign = __import__("json").loads(
                campaign_path.read_text(encoding="ascii"))
            campaign["software_lock"]["controller_binary_sha256"] = "0" * 64
            campaign_path.write_bytes(canonical_bytes(campaign))
            with self.assertRaisesRegex(
                    EvidenceError,
                    "source changed after preflight|campaign identity"):
                validate_run_manifest(path)

    def test_source_runtime_mutation_after_preflight_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, _ = self.make_fixture(root)
            source_plan = root / "source-runtime-plan.json"
            value = __import__("json").loads(
                source_plan.read_text(encoding="ascii"))
            value["executors"][0]["credits"] = 2
            source_plan.write_bytes(canonical_bytes(value))
            with self.assertRaisesRegex(
                    EvidenceError, "source changed after preflight"):
                validate_run_manifest(path)

    def test_runtime_binary_mutation_after_preflight_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            native = root / "test-warm-tier-executors"
            native.write_bytes(b"substituted-native-bench\n")
            self.refresh_artifact(root, manifest, "native_bench_binary")
            transport = __import__("json").loads(
                (root / "transport-overhead.json").read_text(
                    encoding="ascii"))
            transport["native_bench_binary"]["bytes"] = native.stat().st_size
            transport["native_bench_binary"]["sha256"] = digest_file(native)
            (root / "transport-overhead.json").write_bytes(
                canonical_bytes(transport))
            self.refresh_artifact(root, manifest, "transport_overhead")
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "source changed after preflight"):
                validate_run_manifest(path)

    def test_dependency_resolver_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, _ = self.make_fixture(root)
            (root / "ldd").write_bytes(b"changed dependency resolver\n")
            with self.assertRaisesRegex(
                    EvidenceError, "source changed after preflight"):
                validate_run_manifest(path)

    def test_runtime_dependency_manifest_is_load_bearing(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            dependency_path = root / "runtime-dependencies.json"
            value = __import__("json").loads(
                dependency_path.read_text(encoding="ascii"))
            value["binaries"].pop()
            dependency_path.write_bytes(canonical_bytes(value))
            self.refresh_artifact(
                root, manifest, "runtime_dependency_manifest")
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "binary set mismatch"):
                validate_run_manifest(path)

    def test_resource_sampler_cannot_substitute_nvidia_smi(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            argv_path = root / "sampler-argv.json"
            value = __import__("json").loads(
                argv_path.read_text(encoding="ascii"))
            index = value["argv"].index("--nvidia-smi-path") + 1
            value["argv"][index] = "/tmp/path-shadow/nvidia-smi"
            argv_path.write_bytes(canonical_bytes(value))
            self.refresh_artifact(
                root, manifest, "resource_sampler_argv")
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "command mismatch"):
                validate_run_manifest(path)

    def test_orchestrator_cleanup_failure_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            orchestrator = __import__("json").loads(
                (root / "orchestrator.json").read_text(encoding="ascii")
            )
            orchestrator["cleanup"][0]["exit_code"] = 137
            self.rewrite_orchestrator(root, manifest, orchestrator)
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "process did not exit cleanly"):
                validate_run_manifest(path)

    def test_orchestrator_identity_and_process_set_are_exact(self):
        mutations = (
            ("run_id", lambda value: value.update(run_id="other"), "identity"),
            (
                "acquisition",
                lambda value: value.update(acquisition_returncode=2),
                "acquisition failed",
            ),
            (
                "missing process",
                lambda value: value["cleanup"].pop(),
                "cleanup count mismatch",
            ),
            (
                "reordered process",
                lambda value: value["cleanup"].reverse(),
                "cleanup order or process set mismatch",
            ),
        )
        for label, mutate, message in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                    prefix="s40_manifest_") as directory:
                root = Path(directory)
                path, manifest = self.make_fixture(root)
                orchestrator = __import__("json").loads(
                    (root / "orchestrator.json").read_text(encoding="ascii")
                )
                mutate(orchestrator)
                self.rewrite_orchestrator(root, manifest, orchestrator)
                self.write(path, manifest)
                with self.assertRaisesRegex(EvidenceError, message):
                    validate_run_manifest(path)

    def test_orchestrator_cleanup_must_follow_launch_stop(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            orchestrator = __import__("json").loads(
                (root / "orchestrator.json").read_text(encoding="ascii")
            )
            launch = __import__("json").loads(
                (root / "controller-launch.json").read_text(encoding="ascii")
            )
            orchestrator["completed_ns"] = launch["stopped_ns"] - 1
            self.rewrite_orchestrator(root, manifest, orchestrator)
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "cleanup completed before launch stop"):
                validate_run_manifest(path)

    def test_controller_launch_exit_must_be_clean(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            launch = __import__("json").loads(
                (root / "controller-launch.json").read_text(encoding="ascii")
            )
            launch["exit_code"] = 143
            (root / "controller-launch.json").write_bytes(
                canonical_bytes(launch)
            )
            self.refresh_artifact(root, manifest, "controller_launch")
            self.write(path, manifest)
            with self.assertRaisesRegex(EvidenceError, "unclean exit"):
                validate_run_manifest(path)

    def command_pair(self):
        controller = [{
            "command_id": 1,
            "command_kind": 0,
            "controller_epoch": 2,
            "disposition": "RECEIVED",
            "executor_id": "gpu0",
            "model_id": "qwen3-8b-q8_0",
            "publications": [{
                "owner_id": "gpu0",
                "ownership_epoch": 1,
                "position": 2,
                "publication_index": 0,
                "token": 42,
            }],
            "request_complete": False,
            "request_id": "r0",
            "success": True,
        }]
        executor = [{
            "command": {"executor_instance_id": "instance-gpu0"},
            "command_id": 1,
            "controller_epoch": 2,
            "executor_id": "gpu0",
            "kind": 0,
            "model_id": "qwen3-8b-q8_0",
            "publications": copy.deepcopy(
                controller[0]["publications"]),
            "request_complete": False,
            "request_id": "r0",
            "success": True,
        }]
        summaries = {
            "gpu0": {
                "command_lineage": executor,
                "executor_instance_id": "instance-gpu0",
                "run_id": "run",
                "runtime_config_sha256": "a" * 64,
            },
        }
        return summaries, controller

    def test_command_bijection_accepts_exact_pair(self):
        summaries, controller = self.command_pair()
        _validate_command_bijection(
            summaries, controller, "run", "a" * 64)

    def test_command_bijection_rejects_missing_command(self):
        summaries, controller = self.command_pair()
        summaries["gpu0"]["command_lineage"] = []
        with self.assertRaisesRegex(EvidenceError, "lineage mismatch"):
            _validate_command_bijection(
                summaries, controller, "run", "a" * 64)

    def test_command_bijection_rejects_duplicate_command(self):
        summaries, controller = self.command_pair()
        summaries["gpu0"]["command_lineage"].append(copy.deepcopy(
            summaries["gpu0"]["command_lineage"][0]))
        with self.assertRaisesRegex(EvidenceError, "duplicate command"):
            _validate_command_bijection(
                summaries, controller, "run", "a" * 64)

    def test_command_bijection_rejects_publication_mutation(self):
        summaries, controller = self.command_pair()
        summaries["gpu0"]["command_lineage"][0][
            "publications"][0]["token"] += 1
        with self.assertRaisesRegex(EvidenceError, "lineage mismatch"):
            _validate_command_bijection(
                summaries, controller, "run", "a" * 64)

    def test_command_bijection_rejects_epoch_mutation(self):
        summaries, controller = self.command_pair()
        summaries["gpu0"]["command_lineage"][0]["controller_epoch"] += 1
        with self.assertRaisesRegex(EvidenceError, "lineage mismatch"):
            _validate_command_bijection(
                summaries, controller, "run", "a" * 64)

    def test_command_bijection_rejects_run_or_runtime_substitution(self):
        summaries, controller = self.command_pair()
        summaries["gpu0"]["run_id"] = "other"
        with self.assertRaisesRegex(EvidenceError, "run identity mismatch"):
            _validate_command_bijection(
                summaries, controller, "run", "a" * 64)
        summaries["gpu0"]["run_id"] = "run"
        summaries["gpu0"]["runtime_config_sha256"] = "b" * 64
        with self.assertRaisesRegex(EvidenceError, "run identity mismatch"):
            _validate_command_bijection(
                summaries, controller, "run", "a" * 64)

    def test_command_bijection_rejects_quarantined_result(self):
        summaries, controller = self.command_pair()
        controller[0]["disposition"] = "QUARANTINED"
        with self.assertRaisesRegex(
                EvidenceError, "not received on the live frontier"):
            _validate_command_bijection(
                summaries, controller, "run", "a" * 64)

    def test_activation_missing_post_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            activation_path = root / "activation.jsonl"
            rows = read_jsonl(activation_path, "activation")
            write_jsonl(
                activation_path,
                [row for row in rows if row["method"] != "POST"],
            )
            readiness_path = root / "readiness.json"
            readiness = __import__("json").loads(
                readiness_path.read_text(encoding="ascii"))
            readiness["activation_evidence_sha256"] = digest_file(
                activation_path)
            readiness_path.write_bytes(canonical_bytes(readiness))
            self.refresh_artifact(
                root, manifest, "activation_evidence")
            self.refresh_artifact(root, manifest, "server_readiness")
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "incomplete exchange|POST count"):
                validate_run_manifest(path)

    def test_activation_must_precede_trace(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            readiness_path = root / "readiness.json"
            readiness = __import__("json").loads(
                readiness_path.read_text(encoding="ascii"))
            readiness["gateway_ready"][0]["t_ns"] = 60
            readiness_path.write_bytes(canonical_bytes(readiness))
            self.refresh_artifact(root, manifest, "server_readiness")
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "startup ordering mismatch"):
                validate_run_manifest(path)
    def test_legacy_v1_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            path, manifest = self.make_fixture(Path(directory))
            manifest["schema"] = "s40-physical-run-manifest-v1"
            manifest["schema_version"] = 1
            self.write(path, manifest)
            with self.assertRaisesRegex(EvidenceError, "unsupported identity"):
                validate_run_manifest(path)

    def test_missing_executor_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            path, manifest = self.make_fixture(Path(directory))
            manifest["artifacts"] = [
                item for item in manifest["artifacts"]
                if item["role"] != "executor_command::gpu0"
            ]
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "artifact role set mismatch"):
                validate_run_manifest(path)

    def test_executor_semantic_rejection_is_load_bearing(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            path, _ = self.make_fixture(Path(directory))
            with patch(
                    "run_manifest.validate_executor_bundle",
                    side_effect=EvidenceError("failed command")):
                with self.assertRaisesRegex(EvidenceError, "failed command"):
                    validate_run_manifest(path)

    def test_runtime_plan_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            plan = next(
                item for item in manifest["artifacts"]
                if item["role"] == "runtime_plan")
            plan_path = root / plan["path"]
            value = copy.deepcopy(
                __import__("json").loads(plan_path.read_text(encoding="ascii")))
            value["hot_model_id"] = "qwen3-14b-q4_k_m"
            plan_path.write_bytes(canonical_bytes(value))
            plan["bytes"] = plan_path.stat().st_size
            plan["sha256"] = digest_file(plan_path)
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "does not match bound runtime plan"):
                validate_run_manifest(path)

    def test_executed_gateway_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, _ = self.make_fixture(root)
            (root / "executor-bundle" / "desktop_gateway.py").write_bytes(
                b"changed\n")
            with self.assertRaisesRegex(
                    EvidenceError, "byte count mismatch|digest mismatch|differs"):
                validate_run_manifest(path)

    def test_malformed_manifest_cli_fails_without_traceback(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            malformed = Path(directory) / "manifest.json"
            malformed.write_bytes(b"0\n")
            process = subprocess.run(
                [
                    sys.executable,
                    str(S40 / "run_manifest.py"),
                    "--manifest",
                    str(malformed),
                    "--contract",
                    str(DEFAULT_CONTRACT),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )
            combined = process.stdout + process.stderr
            self.assertEqual(process.returncode, 2)
            self.assertEqual(
                combined,
                b"ERROR: EvidenceError: run_manifest: expected object\n",
            )
            self.assertNotIn(b"Traceback", combined)

    def test_trace_runtime_digest_is_load_bearing(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            item = next(
                row for row in manifest["artifacts"]
                if row["role"] == "trace_start")
            trace_path = root / item["path"]
            trace = __import__("json").loads(
                trace_path.read_text(encoding="ascii"))
            trace["runtime_config_sha256"] = "f" * 64
            trace_path.write_bytes(canonical_bytes(trace))
            item["bytes"] = trace_path.stat().st_size
            item["sha256"] = digest_file(trace_path)
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "runtime config digest mismatch"):
                validate_run_manifest(path)

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            path, manifest = self.make_fixture(Path(directory))
            manifest["artifacts"][0]["path"] = "../foreign.jsonl"
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "contained relative path"):
                validate_run_manifest(path)

    def test_host_boot_identity_is_required(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            path, manifest = self.make_fixture(Path(directory))
            manifest["devices"][0]["boot_id"] = "foreign"
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError,
                    "physical plan identity mismatch|host boot mismatch"
                    "|trace host boot"):
                validate_run_manifest(path)

    def test_launch_pid_must_match_resource_rows(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            item = next(
                row for row in manifest["artifacts"]
                if row["role"] == "controller_launch")
            launch_path = root / item["path"]
            launch = __import__("json").loads(
                launch_path.read_text(encoding="ascii"))
            launch["controller_pid"] = 4321
            launch_path.write_bytes(canonical_bytes(launch))
            item["bytes"] = launch_path.stat().st_size
            item["sha256"] = digest_file(launch_path)
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError,
                    "resource identity mismatch|controller PID mismatch"):
                validate_run_manifest(path)

    def test_resource_gpu_must_match_selected_gpu(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            item = next(
                row for row in manifest["artifacts"]
                if row["role"] == "resource_samples")
            resource_path = root / item["path"]
            rows = read_jsonl(resource_path, "resources")
            for row in rows:
                row["gpu_uuid"] = "GPU-foreign"
            write_jsonl(resource_path, rows)
            item["bytes"] = resource_path.stat().st_size
            item["sha256"] = digest_file(resource_path)
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "resource identity mismatch|selected"):
                validate_run_manifest(path)

    def test_gpu_lock_must_bracket_controller_launch(self):
        for label, field, offset in (
                ("late-acquire", "acquired_ns", 1),
                ("early-release", "released_ns", -1)):
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                    prefix="s40_manifest_") as directory:
                root = Path(directory)
                path, manifest = self.make_fixture(root)
                launch = __import__("json").loads(
                    (root / "controller-launch.json").read_text(
                        encoding="ascii"))
                lock_path = root / "gpu-lock-record.json"
                lock = __import__("json").loads(
                    lock_path.read_text(encoding="ascii"))
                lock[field] = (
                    launch["started_ns"] + offset
                    if field == "acquired_ns"
                    else launch["stopped_ns"] + offset
                )
                lock_path.write_bytes(canonical_bytes(lock))
                self.refresh_artifact(root, manifest, "gpu_lock_record")
                self.write(path, manifest)
                with self.assertRaisesRegex(
                        EvidenceError, "interval does not cover"):
                    validate_run_manifest(path)

    def test_gpu_observer_prelaunch_busy_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            observer_path = root / "gpu-observer.jsonl"
            rows = read_jsonl(observer_path, "gpu observer")
            rows[1]["process_stdout_base64"] = base64.b64encode(
                (
                    f"4321,{GPU_UUID},/fixture/gpu-server,100\n"
                ).encode("ascii")
            ).decode("ascii")
            rows[1]["process_observations"] = copy.deepcopy(
                rows[2]["process_observations"])
            write_jsonl(observer_path, rows)
            self.refresh_artifact(root, manifest, "gpu_observer")
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "busy before launch"):
                validate_run_manifest(path)

    def test_gpu_observer_argv_substitution_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            argv_path = root / "gpu-observer-argv.json"
            value = __import__("json").loads(
                argv_path.read_text(encoding="ascii"))
            value["argv"][0] = "/usr/bin/python3"
            argv_path.write_bytes(canonical_bytes(value))
            self.refresh_artifact(root, manifest, "gpu_observer_argv")
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "executed argv mismatch"):
                validate_run_manifest(path)

    def test_http_admission_payload_is_recomputed(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            http_path = root / next(
                row["path"] for row in manifest["artifacts"]
                if row["role"] == "http_evidence")
            rows = read_jsonl(http_path, "http")
            raw = base64.b64decode(
                rows[0]["request_body_base64"], validate=True)
            body = __import__("json").loads(raw)
            body["model"] = "qwen3-8b-q8_0"
            raw = canonical_bytes(body)
            rows[0]["request_body_base64"] = base64.b64encode(
                raw).decode("ascii")
            rows[0]["request_body_sha256"] = digest_bytes(raw)
            write_jsonl(http_path, rows)
            self.refresh_artifact(
                root, manifest, "http_evidence")
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "admission request mismatch"):
                validate_run_manifest(path)

    def test_http_terminal_set_is_recomputed(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            http_path = root / next(
                row["path"] for row in manifest["artifacts"]
                if row["role"] == "http_evidence")
            rows = read_jsonl(http_path, "http")
            for row in rows:
                if row["operation"] != "STATUS":
                    continue
                raw = base64.b64decode(
                    row["response_body_base64"], validate=True)
                response = __import__("json").loads(raw)
                response["committed_output_tokens"] = []
                response["owner_id"] = None
                response["ownership_epoch"] = 0
                response["position"] -= 8
                response["publication_index"] = 0
                response["state"] = "QUEUED"
                raw = canonical_bytes(response)
                row["response_body_base64"] = base64.b64encode(
                    raw).decode("ascii")
                row["response_body_sha256"] = digest_bytes(raw)
                break
            write_jsonl(http_path, rows)
            self.refresh_artifact(
                root, manifest, "http_evidence")
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "terminal status set is incomplete"):
                validate_run_manifest(path)

    def test_trace_result_counts_are_recomputed(self):
        with tempfile.TemporaryDirectory(prefix="s40_manifest_") as directory:
            root = Path(directory)
            path, manifest = self.make_fixture(root)
            result_path = root / next(
                row["path"] for row in manifest["artifacts"]
                if row["role"] == "trace_acquisition_result")
            result = __import__("json").loads(
                result_path.read_text(encoding="ascii"))
            result["completed_count"] = 0
            result_path.write_bytes(canonical_bytes(result))
            self.refresh_artifact(
                root, manifest, "trace_acquisition_result")
            self.write(path, manifest)
            with self.assertRaisesRegex(
                    EvidenceError, "count or time mismatch"):
                validate_run_manifest(path)

    def test_phone_route_coverage_accepts_synchronous_superset(self):
        runtime = {
            "executors": [{
                "executor_id": "phone0",
                "role": "PHONE",
            }],
        }
        executor_summaries = {
            "phone0": {
                "role": "PHONE",
                "route_instances": [{
                    "model_id": "model-a",
                    "placements": {"op12": {}, "op15": {}},
                    "route_instance_id": "route-a",
                }, {
                    "model_id": "model-b",
                    "placements": {"op12": {}, "op15": {}},
                    "route_instance_id": "route-b",
                }],
                "synchronous_route_observations": [{
                    "model_id": "model-a",
                    "route_instance_id": "route-a",
                    "schema": "s40-phone-route-observation-v1",
                }, {
                    "model_id": "model-b",
                    "route_instance_id": "route-b",
                    "schema": "s40-phone-route-observation-v1",
                }],
            },
        }
        phone_summary = {
            "observed_routes": [{
                "model_id": "model-b",
                "route_instance_id": "route-b",
                "sample_count": 2,
            }],
            "schema": "s40-phone-observer-summary-v1",
        }
        self.assertEqual(
            _validate_phone_route_coverage(
                executor_summaries, runtime, phone_summary),
            [{
                "model_id": "model-a",
                "route_instance_id": "route-a",
            }, {
                "model_id": "model-b",
                "route_instance_id": "route-b",
            }],
        )

    def test_phone_route_coverage_rejects_incoherent_evidence(self):
        runtime = {
            "executors": [{
                "executor_id": "phone0",
                "role": "PHONE",
            }],
        }
        executor_summaries = {
            "phone0": {
                "role": "PHONE",
                "route_instances": [{
                    "model_id": "model-a",
                    "placements": {"op12": {}, "op15": {}},
                    "route_instance_id": "route-a",
                }],
                "synchronous_route_observations": [{
                    "model_id": "model-a",
                    "route_instance_id": "route-a",
                    "schema": "s40-phone-route-observation-v1",
                }],
            },
        }
        phone_summary = {
            "observed_routes": [{
                "model_id": "model-a",
                "route_instance_id": "route-a",
                "sample_count": 1,
            }],
            "schema": "s40-phone-observer-summary-v1",
        }
        mutations = []

        missing = copy.deepcopy(executor_summaries)
        missing["phone0"]["synchronous_route_observations"] = []
        mutations.append(("missing synchronous", missing, phone_summary))

        mismatch = copy.deepcopy(executor_summaries)
        mismatch["phone0"]["synchronous_route_observations"][0][
            "model_id"] = "model-b"
        mutations.append(("synchronous mismatch", mismatch, phone_summary))

        duplicate = copy.deepcopy(executor_summaries)
        duplicate["phone0"]["route_instances"].append(
            copy.deepcopy(duplicate["phone0"]["route_instances"][0]))
        mutations.append(("duplicate route", duplicate, phone_summary))

        unknown_periodic = copy.deepcopy(phone_summary)
        unknown_periodic["observed_routes"][0][
            "route_instance_id"] = "route-foreign"
        mutations.append((
            "unknown periodic",
            executor_summaries,
            unknown_periodic,
        ))

        wrong_model = copy.deepcopy(phone_summary)
        wrong_model["observed_routes"][0]["model_id"] = "model-b"
        mutations.append((
            "periodic model mismatch",
            executor_summaries,
            wrong_model,
        ))

        zero_samples = copy.deepcopy(phone_summary)
        zero_samples["observed_routes"][0]["sample_count"] = 0
        mutations.append((
            "periodic zero samples",
            executor_summaries,
            zero_samples,
        ))

        for label, summaries, periodic in mutations:
            with self.subTest(label=label), self.assertRaises(EvidenceError):
                _validate_phone_route_coverage(summaries, runtime, periodic)

        gpu_runtime = {
            "executors": [{
                "executor_id": "gpu0",
                "role": "GPU",
            }],
        }
        gpu_summaries = {"gpu0": {"role": "GPU"}}
        self.assertEqual(
            _validate_phone_route_coverage(
                gpu_summaries, gpu_runtime, None),
            [],
        )
        with self.assertRaises(EvidenceError):
            _validate_phone_route_coverage(
                gpu_summaries, gpu_runtime, phone_summary)
        with self.assertRaises(EvidenceError):
            _validate_phone_route_coverage(
                {"gpu0": {"role": "PHONE"}},
                gpu_runtime,
                None,
            )

    def test_qualification_source_records_bind_full_a_and_b_roots(self):
        with tempfile.TemporaryDirectory(
                prefix="s40_qualification_") as directory:
            base = Path(directory)

            def root_record(name):
                root = (base / name).resolve()
                root.mkdir()
                payload = root / "raw.jsonl"
                payload.write_bytes(b'{"raw":1}\n')
                manifest = root / "EVIDENCE_BUNDLE.json"
                manifest.write_bytes(canonical_bytes({
                    "artifacts": [{
                        "path": payload.name,
                        "sha256": digest_file(payload),
                    }],
                }))
                records = {}
                for role in (
                        "artifact_snapshot",
                        "readiness_lock",
                        "fresh_snapshot",
                        "runtime_identity"):
                    path = root / f"{role}.json"
                    path.write_bytes(canonical_bytes({"role": role}))
                    records[role] = {
                        "path": str(path),
                        "sha256": digest_file(path),
                    }
                return {
                    **records,
                    "bundle_manifest_sha256": digest_file(manifest),
                    "bundle_root": str(root),
                }

            a_root = root_record("a")
            b_root = root_record("b")
            authority = {
                "a_chain": a_root,
                "current": b_root,
                "model_id": "model-b",
                "phase": "B_ONLY",
                "schema": "s40-route-qualification-authority-v1",
                "slot": "B",
            }
            rows = qualification_source_records(
                authority, "phone0", "model-b")
            self.assertEqual(len(rows), 12)
            self.assertEqual(
                len({row["path"] for row in rows}),
                len(rows),
            )
            self.assertEqual(
                {part for row in rows for part in (
                    "a_chain" if "::a_chain::" in row["role"]
                    else "current",
                )},
                {"a_chain", "current"},
            )

            authority["phase"] = "A_ONLY"
            authority["slot"] = "A"
            authority["model_id"] = "model-a"
            authority["current"] = a_root
            authority["a_chain"] = None
            rows = qualification_source_records(
                authority, "gpu0", "model-a")
            self.assertEqual(len(rows), 6)

    def test_qualification_source_records_fail_closed(self):
        with tempfile.TemporaryDirectory(
                prefix="s40_qualification_") as directory:
            root = (Path(directory) / "root").resolve()
            root.mkdir()
            payload = root / "raw.jsonl"
            payload.write_bytes(b'{"raw":1}\n')
            manifest = root / "EVIDENCE_BUNDLE.json"
            manifest.write_bytes(canonical_bytes({
                "artifacts": [{
                    "path": payload.name,
                    "sha256": digest_file(payload),
                }],
            }))
            records = {}
            for role in (
                    "artifact_snapshot",
                    "readiness_lock",
                    "fresh_snapshot",
                    "runtime_identity"):
                path = root / f"{role}.json"
                path.write_bytes(canonical_bytes({"role": role}))
                records[role] = {
                    "path": str(path),
                    "sha256": digest_file(path),
                }
            current = {
                **records,
                "bundle_manifest_sha256": digest_file(manifest),
                "bundle_root": str(root),
            }
            authority = {
                "a_chain": None,
                "current": current,
                "model_id": "model-a",
                "phase": "A_ONLY",
                "schema": "s40-route-qualification-authority-v1",
                "slot": "A",
            }

            mutations = []
            wrong_chain = copy.deepcopy(authority)
            wrong_chain["a_chain"] = copy.deepcopy(current)
            mutations.append(("wrong A chain", wrong_chain))

            wrong_model = copy.deepcopy(authority)
            wrong_model["model_id"] = "model-b"
            mutations.append(("wrong model", wrong_model))

            duplicate = copy.deepcopy(authority)
            duplicate["current"]["fresh_snapshot"] = copy.deepcopy(
                duplicate["current"]["artifact_snapshot"])
            mutations.append(("duplicate input", duplicate))

            wrong_digest = copy.deepcopy(authority)
            wrong_digest["current"]["runtime_identity"]["sha256"] = "f" * 64
            mutations.append(("wrong digest", wrong_digest))

            for label, value in mutations:
                with self.subTest(label=label), self.assertRaises(
                        EvidenceError):
                    qualification_source_records(
                        value, "gpu0", "model-a")

            outside = Path(directory) / "outside.json"
            outside.write_bytes(b"{}\n")
            escape_manifest = {
                "artifacts": [{
                    "path": "../outside.json",
                    "sha256": digest_file(outside),
                }],
            }
            manifest.write_bytes(canonical_bytes(escape_manifest))
            authority["current"]["bundle_manifest_sha256"] = digest_file(
                manifest)
            with self.assertRaises(EvidenceError):
                qualification_source_records(
                    authority, "gpu0", "model-a")

    def test_qualification_route_binds_distinct_route_and_phase_locks(self):
        with tempfile.TemporaryDirectory(
                prefix="s40_qualification_") as directory:
            root = (Path(directory) / "root").resolve()
            root.mkdir()
            route_lock_sha256 = "a" * 64
            phase_lock = root / "phase.lock.jsonl"
            phase_lock.write_bytes(b'{"phase":"lock"}\n')
            phase_lock_sha256 = digest_file(phase_lock)
            artifact = root / "artifact.json"
            artifact.write_bytes(canonical_bytes({
                "artifacts": [{"fixture": True}],
                "completed_ns": 20,
                "model_id": "model-a",
                "phase": "A_ONLY",
                "route_lock_sha256": route_lock_sha256,
                "schema": "s39-cp0-r1-artifact-snapshot-v2.3",
                "slot": "A",
                "started_ns": 10,
            }))
            readiness = root / "readiness.json"
            readiness.write_bytes(canonical_bytes({
                "artifact_snapshot_sha256": digest_file(artifact),
                "event_ns": 30,
                "phase": "A_ONLY",
                "phase_id": "phase-a",
                "schema": "s39-cp0-r1-readiness-lock-v2.3",
                "v2_2_phase_lock_sha256": phase_lock_sha256,
            }))
            fresh = root / "fresh.json"
            runtime = root / "runtime.json"
            fresh.write_bytes(b'{"fresh":true}\n')
            runtime.write_bytes(b'{"runtime":true}\n')
            manifest = root / "EVIDENCE_BUNDLE.json"
            manifest.write_bytes(canonical_bytes({
                "artifacts": [{
                    "path": phase_lock.name,
                    "sha256": phase_lock_sha256,
                }],
            }))
            authority = {
                "a_chain": None,
                "current": {
                    "artifact_snapshot": {
                        "path": str(artifact),
                        "sha256": digest_file(artifact),
                    },
                    "bundle_manifest_sha256": digest_file(manifest),
                    "bundle_root": str(root),
                    "fresh_snapshot": {
                        "path": str(fresh),
                        "sha256": digest_file(fresh),
                    },
                    "readiness_lock": {
                        "path": str(readiness),
                        "sha256": digest_file(readiness),
                    },
                    "runtime_identity": {
                        "path": str(runtime),
                        "sha256": digest_file(runtime),
                    },
                },
                "model_id": "model-a",
                "phase": "A_ONLY",
                "schema": "s40-route-qualification-authority-v1",
                "slot": "A",
            }
            route = {
                "artifact_certificate_path": str(artifact),
                "artifact_certificate_sha256": digest_file(artifact),
                "model_id": "model-a",
                "phase": "A_ONLY",
                "phase_lock_sha256": phase_lock_sha256,
                "qualification": authority,
                "readiness_lock_path": str(readiness),
                "readiness_lock_sha256": digest_file(readiness),
                "readiness_phase_id": "phase-a",
                "route_lock_sha256": route_lock_sha256,
                "slot": "A",
            }
            validate_qualification_route_identity(
                authority, route, "gpu0")

            for label, key, value in (
                    (
                        "route lock replaced by phase lock",
                        "route_lock_sha256",
                        phase_lock_sha256,
                    ),
                    (
                        "phase lock replaced by route lock",
                        "phase_lock_sha256",
                        route_lock_sha256,
                    ),
                    (
                        "wrong readiness phase",
                        "readiness_phase_id",
                        "phase-b",
                    )):
                mutated = copy.deepcopy(route)
                mutated[key] = value
                with self.subTest(label=label), self.assertRaises(
                        EvidenceError):
                    validate_qualification_route_identity(
                        authority, mutated, "gpu0")

    def test_route_qualification_summary_rejects_smoke_and_split_roots(self):
        models = ["model-a", "model-b"]
        qualifications = [{
            "a_chain_phase_id": None,
            "bundle_manifest_sha256": "1" * 64,
            "model_id": "model-a",
            "phase": "A_ONLY",
            "phase_id": "phase-a",
            "phase_lock_sha256": "2" * 64,
            "schema": "s40-route-qualification-derived-v1",
            "scope": "QUALIFIED_ROUTE",
            "status": "MODEL_A_QUALIFICATION_PASS",
            "v2_2_result_sha256": "3" * 64,
        }, {
            "a_chain_phase_id": "phase-a",
            "bundle_manifest_sha256": "4" * 64,
            "model_id": "model-b",
            "phase": "B_ONLY",
            "phase_id": "phase-b",
            "phase_lock_sha256": "5" * 64,
            "schema": "s40-route-qualification-derived-v1",
            "scope": "QUALIFIED_ROUTE",
            "status": "MODEL_B_QUALIFICATION_PASS",
            "v2_2_result_sha256": "6" * 64,
        }]
        summaries = {
            "gpu0": {"route_qualifications": qualifications},
            "phone0": {
                "route_qualifications": copy.deepcopy(qualifications),
            },
        }
        self.assertEqual(
            _validate_route_qualification_summaries(summaries, models),
            qualifications,
        )

        smoke = copy.deepcopy(summaries)
        smoke["gpu0"]["route_qualifications"][0]["scope"] = (
            "DESKTOP_SMOKE_ONLY")
        with self.assertRaises(EvidenceError):
            _validate_route_qualification_summaries(smoke, models)

        split = copy.deepcopy(summaries)
        split["phone0"]["route_qualifications"][1][
            "bundle_manifest_sha256"] = "7" * 64
        with self.assertRaises(EvidenceError):
            _validate_route_qualification_summaries(split, models)

        broken_chain = copy.deepcopy(summaries)
        broken_chain["gpu0"]["route_qualifications"][1][
            "a_chain_phase_id"] = "phase-foreign"
        with self.assertRaises(EvidenceError):
            _validate_route_qualification_summaries(
                broken_chain, models)


if __name__ == "__main__":
    unittest.main()
