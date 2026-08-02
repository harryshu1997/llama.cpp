#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
EXECUTORS = HERE.parent
S40 = EXECUTORS.parent
S39 = S40.parent / "s39_phone_model_switch_trace"
S22 = S40.parent / "s22_slo_overlap_pipeline"
for directory in (EXECUTORS, S39, S22):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from phone_gateway import (
    COMMAND_CLEANUP,
    COMMAND_DISCARD,
    COMMAND_DRAIN,
    COMMAND_EXECUTE,
    COMMAND_LOAD,
    COMMAND_REPLAY,
    COMMAND_UNLOAD,
    GatewayServer,
    PhoneRouteExecutor,
    RouteConnection,
    RouteSpec,
    SubprocessRouteSupervisor,
    canonical_bytes,
    control_argv,
    parse_command,
    parse_route_config,
    strict_json_loads,
)
from runtime_binding import (
    RuntimeBinding,
    RuntimeBindingError,
    process_start_time_ticks,
)
from stage_v3_client import (
    BatchResult,
    Hello,
    STAGE_V3_BASE_CAPABILITIES,
    STAGE_V3_CAP_IDENTITY,
    STAGE_V3_CAP_TERMINAL,
)


class FakeStatus:
    def __init__(self, active_sequences=0):
        self.active_sequences = active_sequences


class FakeStageNet:
    def __init__(
        self,
        stale_epoch=False,
        stop_error=None,
        remove_error=None,
    ):
        self.calls = []
        self.lock = threading.Lock()
        self.stale_epoch = stale_epoch
        self.stop_error = stop_error
        self.remove_error = remove_error
        self.stop_count = 0
        self.active_sequences = set()

    def batch(self, rows):
        rows = list(rows)
        with self.lock:
            self.calls.append(rows)
            self.active_sequences.update(row.seq_id for row in rows)
        return tuple(
            BatchResult(
                row.request_id,
                row.route_epoch - 1 if self.stale_epoch else row.route_epoch,
                row.seq_id,
                row.position,
                None,
                row.token + 1000,
            )
            for row in rows
        )

    def remove(self, seq_id, request_id, route_epoch):
        del request_id, route_epoch
        if self.remove_error is not None:
            raise self.remove_error
        self.active_sequences.discard(seq_id)
        return FakeStatus(len(self.active_sequences))

    def drain(self):
        return FakeStatus(len(self.active_sequences))

    def stop(self):
        self.stop_count += 1
        if self.stop_error is not None:
            raise self.stop_error

    def close(self):
        pass


def make_spec(model_id):
    digest = ("a" if model_id == "model-a" else "b") * 64
    return RouteSpec(
        model_id=model_id,
        model_sha256=digest,
        artifact_certificate_sha256="4" * 64,
        readiness_lock_sha256="5" * 64,
        readiness_phase_id=f"phase-{model_id}",
        relay_host="127.0.0.1",
        relay_port=10000,
        file_type=7,
        layer_start=0,
        layer_end=4,
        n_layer=4,
        n_embd=16,
        max_streams=8,
        n_batch=8,
        n_ubatch=8,
        batch_knee=8,
        gather_us=200000,
        queue_depth=32,
        prefill_chunk=4,
        phase="A_ONLY" if model_id == "model-a" else "B_ONLY",
        phase_lock_sha256="6" * 64,
        qualification={
            "model_id": model_id,
            "phase_id": f"phase-{model_id}",
            "schema": "s40-route-qualification-derived-v1",
            "scope": "QUALIFIED_ROUTE",
        },
        qualification_sha256="7" * 64,
        slot="A" if model_id == "model-a" else "B",
        load_argv=("load", model_id),
        unload_argv=("unload", model_id),
        a6000_identity="a6000",
        op15_boot_id="op15-boot",
        op12_boot_id="op12-boot",
        op15_shard_sha256="1" * 64,
        op12_shard_sha256="2" * 64,
        worker_sha256="3" * 64,
    )


def make_route_config(root: Path) -> tuple[Path, dict]:
    files = {}
    for role in (
        "a6000_ssh_control",
        "executor_bundle_manifest",
        "python",
        "ssh_config",
    ):
        path = root / role
        path.write_text(role + "\n", encoding="ascii")
        if role in ("a6000_ssh_control", "python"):
            path.chmod(0o755)
        files[role] = path
    manifest_sha256 = hashlib.sha256(
        files["executor_bundle_manifest"].read_bytes()
    ).hexdigest()
    spec = make_spec("model-a")
    route = {
        field: getattr(spec, field)
        for field in (
            "a6000_identity",
            "artifact_certificate_sha256",
            "batch_knee",
            "file_type",
            "gather_us",
            "layer_end",
            "layer_start",
            "max_streams",
            "model_id",
            "model_sha256",
            "n_batch",
            "n_embd",
            "n_layer",
            "n_ubatch",
            "op12_boot_id",
            "op12_shard_sha256",
            "op15_boot_id",
            "op15_shard_sha256",
            "prefill_chunk",
            "phase",
            "phase_lock_sha256",
            "qualification",
            "queue_depth",
            "readiness_lock_sha256",
            "readiness_phase_id",
            "relay_host",
            "relay_port",
            "slot",
            "worker_sha256",
        )
    }
    value = {
        "control": {
            "environment": {
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(root),
                "S40_EXECUTOR_BUNDLE": "1",
                "S40_EXECUTOR_BUNDLE_MANIFEST":
                    str(files["executor_bundle_manifest"]),
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
                for role, path in sorted(files.items())
            ],
        },
        "executor_id": "PHONE",
        "routes": [route],
        "schema": "s40-phone-route-config-v3",
    }
    path = root / "route-config.json"
    path.write_bytes(canonical_bytes(value))
    return path, value


class FakeSupervisor:
    def __init__(self, clients=None):
        self.clients = list(clients or [])
        self.opened = []
        self.closed = []

    def open(self, spec):
        client = self.clients.pop(0) if self.clients else FakeStageNet()
        self.opened.append(spec.model_id)
        return RouteConnection(
            client,
            Hello(
                spec.layer_start,
                spec.layer_end,
                spec.n_layer,
                spec.n_embd,
                spec.max_streams,
                1024,
                spec.n_batch,
                spec.n_ubatch,
                (
                    STAGE_V3_BASE_CAPABILITIES
                    | STAGE_V3_CAP_TERMINAL
                    | STAGE_V3_CAP_IDENTITY
                ),
                spec.file_type,
                spec.model_sha256,
            ),
            f"instance-{len(self.opened)}",
        )

    def close(self, spec, client):
        client.stop()
        client.close()
        self.closed.append(spec.model_id)


def make_executor(clients=None, initial_model="model-a"):
    specs = {
        "model-a": make_spec("model-a"),
        "model-b": make_spec("model-b"),
    }
    supervisor = FakeSupervisor(clients)
    executor = PhoneRouteExecutor(
        "PHONE",
        specs,
        supervisor,
        initial_model,
        timeout_s=2.0,
    )
    return executor, supervisor


def request_command(command_id, request_id, model_id="model-a"):
    prompt = [1, 2, 3, 4]
    return {
        "command_id": command_id,
        "controller_epoch": 0,
        "executor_id": "PHONE",
        "executor_instance_id": "instance-phone",
        "kind": COMMAND_EXECUTE,
        "max_output_tokens": 1,
        "model_id": model_id,
        "request": {
            "committed_output_tokens": [],
            "model_id": model_id,
            "owner_id": "PHONE",
            "ownership_epoch": 1,
            "position": len(prompt),
            "prompt_tokens": prompt,
            "publication_index": 0,
            "request_id": request_id,
            "state": 1,
        },
        "request_id": request_id,
        "schema": "llama-server-warm-tier-command-v3",
        "total_output_tokens": 8,
    }


def cleanup_command(command_id, request_id, model_id="model-a"):
    command = request_command(command_id, request_id, model_id)
    command["kind"] = COMMAND_CLEANUP
    command["max_output_tokens"] = 0
    command["total_output_tokens"] = 0
    return command


def empty_request():
    return {
        "committed_output_tokens": [],
        "model_id": "",
        "owner_id": "",
        "ownership_epoch": 0,
        "position": 0,
        "prompt_tokens": [],
        "publication_index": 0,
        "request_id": "",
        "state": 0,
    }


def model_command(command_id, kind, model_id):
    return {
        "command_id": command_id,
        "controller_epoch": 1,
        "executor_id": "PHONE",
        "executor_instance_id": "instance-phone",
        "kind": kind,
        "max_output_tokens": 0,
        "model_id": model_id,
        "request": empty_request(),
        "request_id": "",
        "schema": "llama-server-warm-tier-command-v3",
        "total_output_tokens": 0,
    }


class PhoneGatewayTests(unittest.TestCase):
    def setUp(self):
        authority = mock.patch(
            "phone_gateway.derive_route_qualification",
            side_effect=lambda _value, **kwargs: {
                "a_chain_phase_id": None,
                "bundle_manifest_sha256": "d" * 64,
                "model_id": kwargs["model_id"],
                "phase": kwargs["phase"],
                "phase_id": f"phase-{kwargs['model_id']}",
                "phase_lock_sha256": "6" * 64,
                "schema": "s40-route-qualification-derived-v1",
                "scope": "QUALIFIED_ROUTE",
                "status": (
                    "MODEL_A_QUALIFICATION_PASS"
                    if kwargs["phase"] == "A_ONLY"
                    else "MODEL_B_QUALIFICATION_PASS"
                ),
                "v2_2_result_sha256": "e" * 64,
            },
        )
        authority.start()
        self.addCleanup(authority.stop)

    def test_authentication_failure_is_fatal_before_command_read(self):
        class RejectingAuthenticator:
            calls = 0

            def authenticate(self, connection):
                del connection
                self.calls += 1
                raise RuntimeBindingError("rejected peer")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executor, _ = make_executor(initial_model=None)
            authenticator = RejectingAuthenticator()
            server = GatewayServer(
                root / "gateway.sock",
                executor,
                root / "evidence.jsonl",
                controller_authenticator=authenticator,
                route_config_sha256="1" * 64,
                run_id="run-1",
                runtime_binding=RuntimeBinding(
                    executor_id="PHONE",
                    executor_instance_id="instance-phone",
                    gateway_pid=os.getpid(),
                    gateway_start_time_ticks=process_start_time_ticks(),
                    runtime_config_path=str(root / "runtime.json"),
                    runtime_config_sha256="2" * 64,
                    runtime_config_device=1,
                    runtime_config_inode=1,
                ),
            )
            gateway_side, controller_side = socket.socketpair()
            controller_side.sendall(
                canonical_bytes(model_command(1, COMMAND_LOAD, "model-a"))
            )
            controller_side.shutdown(socket.SHUT_WR)
            server._serve_one(gateway_side)
            controller_side.close()
            self.assertEqual(authenticator.calls, 1)
            self.assertTrue(server._stopping.is_set())
            self.assertIsInstance(
                server._fatal_error,
                RuntimeBindingError,
            )
            self.assertEqual(executor.model_id, "")
            server._server.close()
            executor.close()
            server._evidence.close()

    def test_controller_binding_precedes_command_evidence(self):
        class RecordingAuthenticator:
            def __init__(self, path):
                self.path = path

            def authenticate(self, connection):
                del connection
                self.path.write_bytes(b"bound\n")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executor, _ = make_executor(initial_model=None)
            binding = root / "binding"
            server = GatewayServer(
                root / "gateway.sock",
                executor,
                root / "evidence.jsonl",
                controller_authenticator=RecordingAuthenticator(binding),
                route_config_sha256="1" * 64,
                run_id="run-1",
                runtime_binding=RuntimeBinding(
                    executor_id="PHONE",
                    executor_instance_id="instance-phone",
                    gateway_pid=os.getpid(),
                    gateway_start_time_ticks=process_start_time_ticks(),
                    runtime_config_path=str(root / "runtime.json"),
                    runtime_config_sha256="2" * 64,
                    runtime_config_device=1,
                    runtime_config_inode=1,
                ),
            )
            original_handle = executor.handle

            def bound_handle(command):
                self.assertEqual(binding.read_bytes(), b"bound\n")
                return original_handle(command)

            executor.handle = bound_handle
            gateway_side, controller_side = socket.socketpair()
            controller_side.sendall(
                canonical_bytes(model_command(1, COMMAND_LOAD, "model-a"))
            )
            controller_side.shutdown(socket.SHUT_WR)
            server._serve_one(gateway_side)
            result = strict_json_loads(
                controller_side.makefile("rb").read(),
                "gateway result",
            )
            self.assertTrue(result["success"])
            controller_side.close()
            self.assertIsNone(server._fatal_error)
            server._server.close()
            executor.close()
            server._evidence.close()

    def test_route_control_argv_and_environment_are_derived(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, value = make_route_config(root)
            executor_id, specs, _ = parse_route_config(path)
            self.assertEqual(executor_id, "PHONE")
            spec = specs["model-a"]
            self.assertEqual(
                control_argv(spec, "load"),
                (
                    str(root / "python"),
                    "-B",
                    "-s",
                    "-P",
                    str(root / "a6000_ssh_control"),
                    "--action",
                    "load",
                    "--config",
                    str(root / "ssh_config"),
                    "--model",
                    "model-a",
                ),
            )
            mutations = {
                "supplied argv": lambda item: item["routes"][0].__setitem__(
                    "load_argv",
                    ["/tmp/other"],
                ),
                "PATH": lambda item: item["control"]["environment"].__setitem__(
                    "PATH",
                    "/tmp:/usr/bin",
                ),
                "extra env": lambda item: item["control"][
                    "environment"
                ].__setitem__("HOME", "/tmp"),
                "bundle digest": lambda item: item["control"][
                    "environment"
                ].__setitem__("S40_EXECUTOR_BUNDLE_SHA256", "f" * 64),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    changed = copy.deepcopy(value)
                    mutate(changed)
                    candidate = root / f"{name.replace(' ', '-')}.json"
                    candidate.write_bytes(canonical_bytes(changed))
                    with self.assertRaises(Exception):
                        parse_route_config(candidate)

    def test_route_control_file_mutation_during_command_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _ = make_route_config(root)
            _, specs, _ = parse_route_config(path)
            spec = specs["model-a"]
            evidence = root / "route.jsonl"
            supervisor = SubprocessRouteSupervisor(1.0, evidence)

            def runner(argv, **kwargs):
                self.assertEqual(tuple(argv), control_argv(spec, "load"))
                self.assertEqual(kwargs["env"], dict(spec.control_env))
                (root / "executor_bundle_manifest").write_text(
                    "changed\n",
                    encoding="ascii",
                )
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    canonical_bytes({"success": True}),
                    b"",
                )

            with mock.patch(
                "phone_gateway.subprocess.run",
                side_effect=runner,
            ):
                with self.assertRaisesRegex(Exception, "changed"):
                    supervisor._run(spec, "load")
            supervisor._evidence.close()

    def test_sigterm_subprocess_unloads_and_removes_active_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / "active.json"
            active.write_bytes(canonical_bytes({
                "route_instance_id": "route-a",
                "schema": "test-active",
            }))
            socket_path = root / "gateway.sock"
            evidence = root / "commands.jsonl"
            route_evidence = root / "route.jsonl"
            fixture = HERE / "gateway_shutdown_fixture.py"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(fixture),
                    "--active",
                    str(active),
                    "--evidence",
                    str(evidence),
                    "--route-evidence",
                    str(route_evidence),
                    "--socket",
                    str(socket_path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
            deadline = time.monotonic() + 5.0
            while not socket_path.exists() and process.poll() is None:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            self.assertIsNone(process.poll())
            process.terminate()
            stdout, stderr = process.communicate(timeout=5.0)
            self.assertEqual(process.returncode, 0, stderr.decode("utf-8"))
            self.assertEqual(stdout, b"")
            self.assertEqual(stderr, b"")
            self.assertFalse(active.exists())
            rows = [
                strict_json_loads(raw + b"\n", "route cleanup")
                for raw in route_evidence.read_bytes().splitlines()
            ]
            self.assertEqual(
                rows,
                [{
                    "model_id": "model-a",
                    "route_instance_id": "route-a",
                    "schema": "s40-phone-route-unload-v1",
                    "success": True,
                }],
            )

    def test_relay_start_failure_unloads_loaded_route(self):
        spec = make_spec("model-a")
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "route.jsonl"
            supervisor = SubprocessRouteSupervisor(0.001, evidence)
            calls = []

            def run(run_spec, action):
                self.assertIs(run_spec, spec)
                calls.append(action)
                if action == "load":
                    return {
                        "a6000_identity": spec.a6000_identity,
                        "artifact_certificate_sha256":
                            spec.artifact_certificate_sha256,
                        "model_id": spec.model_id,
                        "model_sha256": spec.model_sha256,
                        "op12_boot_id": spec.op12_boot_id,
                        "op12_shard_sha256": spec.op12_shard_sha256,
                        "op15_boot_id": spec.op15_boot_id,
                        "op15_shard_sha256": spec.op15_shard_sha256,
                        "qualification_sha256":
                            spec.qualification_sha256,
                        "readiness_lock_sha256": spec.readiness_lock_sha256,
                        "readiness_phase_id": spec.readiness_phase_id,
                        "route_observation": {
                            "direct_peer": {},
                            "model_id": spec.model_id,
                            "phones": {"op12": {}, "op15": {}},
                            "route_instance_id": "route-1",
                            "schema":
                                "s40-phone-route-observation-v1",
                        },
                        "route_instance_id": "route-1",
                        "schema": "s40-phone-route-load-v3",
                        "success": True,
                        "worker_sha256": spec.worker_sha256,
                    }
                self.assertEqual(action, "rollback")
                return {
                    "model_id": spec.model_id,
                    "route_instance_id": "route-1",
                    "schema": "s40-phone-route-rollback-v1",
                    "success": True,
                }

            supervisor._run = run
            with mock.patch.object(
                __import__("phone_gateway").StageV3Client,
                "connect",
                side_effect=OSError("not ready"),
            ):
                with self.assertRaisesRegex(Exception, "was unloaded"):
                    supervisor.open(spec)
            supervisor._evidence.close()
            self.assertEqual(calls, ["load", "rollback"])
            rows = [
                strict_json_loads(line, "route row")
                for line in evidence.read_bytes().splitlines(keepends=True)
            ]
            self.assertEqual(
                [row["schema"] for row in rows],
                [
                    "s40-phone-route-load-v3",
                    "s40-phone-route-rollback-v1",
                ],
            )

    def test_two_controller_requests_form_one_wire_batch(self):
        client = FakeStageNet()
        executor, _ = make_executor([client])
        barrier = threading.Barrier(2)
        results = [None, None]
        errors = []

        def run(index):
            try:
                barrier.wait(timeout=1.0)
                results[index] = executor.handle(
                    request_command(index + 1, f"r{index + 1}")
                )
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3.0)
        self.assertFalse(errors)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0]), 8)
        self.assertEqual(
            {row.request_id for row in client.calls[0]},
            {1, 2},
        )
        self.assertTrue(all(result["success"] for result in results))
        self.assertEqual(
            [result["publications"][0]["token"] for result in results],
            [1004, 1004],
        )

        executor.handle(cleanup_command(3, "r1"))
        executor.handle(cleanup_command(4, "r2"))
        executor.close()

    def test_stale_epoch_and_duplicate_execute_fail_closed(self):
        client = FakeStageNet()
        spec = make_spec("model-a")
        spec = RouteSpec(
            **{
                **spec.__dict__,
                "max_streams": 1,
                "n_batch": 4,
                "n_ubatch": 4,
                "batch_knee": 4,
                "gather_us": 0,
                "queue_depth": 8,
            }
        )
        supervisor = FakeSupervisor([client])
        executor = PhoneRouteExecutor(
            "PHONE",
            {"model-a": spec},
            supervisor,
            "model-a",
            timeout_s=1.0,
        )
        first = request_command(1, "r1")
        self.assertTrue(executor.handle(first)["success"])
        stale = request_command(2, "r1")
        stale["request"]["ownership_epoch"] = 2
        with self.assertRaisesRegex(Exception, "stale ownership epoch"):
            executor.handle(stale)
        executor.handle(cleanup_command(3, "r1"))
        executor.close()

    def test_unload_reload_increments_epoch_and_rejects_stale_result(self):
        first = FakeStageNet()
        second = FakeStageNet(stale_epoch=True)
        executor, supervisor = make_executor([first, second])
        self.assertEqual(executor.route_epoch, 1)
        executor.handle(request_command(1, "r1"))
        executor.handle(cleanup_command(2, "r1"))
        executor.handle(model_command(3, COMMAND_DRAIN, "model-a"))
        executor.handle(model_command(4, COMMAND_UNLOAD, "model-a"))
        self.assertEqual(first.stop_count, 1)
        executor.handle(model_command(5, COMMAND_LOAD, "model-b"))
        self.assertEqual(executor.route_epoch, 2)
        with self.assertRaisesRegex(Exception, "stale or mismatched"):
            executor.handle(request_command(6, "r2", "model-b"))
        executor.close()
        self.assertEqual(supervisor.opened, ["model-a", "model-b"])
        self.assertEqual(supervisor.closed, ["model-a", "model-b"])

    def test_stop_failure_uses_cleanup_only_rollback(self):
        spec = make_spec("model-a")
        client = FakeStageNet(stop_error=OSError("send failed"))
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "route.jsonl"
            supervisor = SubprocessRouteSupervisor(1.0, evidence)
            supervisor._instances[id(client)] = "route-1"
            actions = []

            def run(run_spec, action):
                self.assertIs(run_spec, spec)
                actions.append(action)
                self.assertEqual(action, "rollback")
                return {
                    "model_id": spec.model_id,
                    "route_instance_id": "route-1",
                    "schema": "s40-phone-route-rollback-v1",
                    "success": True,
                }

            supervisor._run = run
            with self.assertRaisesRegex(Exception, "was rolled back"):
                supervisor.close(spec, client)
            supervisor._evidence.close()
            self.assertEqual(actions, ["rollback"])
            rows = [
                strict_json_loads(line, "rollback row")
                for line in evidence.read_bytes().splitlines(keepends=True)
            ]
            self.assertEqual(
                [row["schema"] for row in rows],
                ["s40-phone-route-rollback-v1"],
            )

    def test_close_continues_route_shutdown_after_remove_failure(self):
        client = FakeStageNet(remove_error=OSError("remove failed"))
        executor, supervisor = make_executor([client])
        executor.handle(request_command(1, "r1"))
        with self.assertRaisesRegex(
            Exception,
            "phone cleanup failed.*remove failed",
        ):
            executor.close()
        self.assertEqual(client.stop_count, 1)
        self.assertEqual(supervisor.closed, ["model-a"])
        self.assertIsNone(executor._active_spec)
        self.assertEqual(executor._sessions, {})

    def test_nonfinite_json_is_rejected(self):
        for raw in (b'{"value":NaN}\n', b'{"value":Infinity}\n'):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(Exception, "numeric constant"):
                    strict_json_loads(raw, "nonfinite")
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(Exception, "canonical JSON"):
                    canonical_bytes({"value": value})

    def test_drain_rejects_resident_execute_and_replay_without_state_loss(self):
        executor, _ = make_executor([FakeStageNet()])
        first = executor.handle(request_command(1, "r1"))
        self.assertTrue(first["success"])
        self.assertIn("r1", executor._sessions)
        executor.handle(model_command(2, COMMAND_DRAIN, "model-a"))

        execute = request_command(3, "r1")
        execute["request"]["committed_output_tokens"] = [
            first["publications"][0]["token"]
        ]
        execute["request"]["position"] += 1
        execute["request"]["publication_index"] += 1
        with self.assertRaisesRegex(Exception, "not accepting"):
            executor.handle(execute)

        replay = request_command(4, "r1")
        replay["kind"] = COMMAND_REPLAY
        replay["max_output_tokens"] = 0
        replay["total_output_tokens"] = 0
        with self.assertRaisesRegex(Exception, "not accepting"):
            executor.handle(replay)
        self.assertIn("r1", executor._sessions)
        executor.close()

    def test_parse_execute_replay_and_model_discard(self):
        execute = request_command(1, "r1")
        replay = request_command(2, "r1")
        replay["kind"] = COMMAND_REPLAY
        replay["max_output_tokens"] = 0
        replay["total_output_tokens"] = 0
        discard = model_command(3, COMMAND_DISCARD, "model-a")
        for command in (execute, replay, discard):
            parsed = parse_command(
                canonical_bytes(command),
                "PHONE",
                "instance-phone",
            )
            self.assertEqual(parsed["kind"], command["kind"])
        stale_instance = copy.deepcopy(execute)
        stale_instance["executor_instance_id"] = "stale-instance"
        with self.assertRaisesRegex(Exception, "executor instance mismatch"):
            parse_command(
                canonical_bytes(stale_instance),
                "PHONE",
                "instance-phone",
            )
        execute["request"]["prompt_tokens"][0] = -1
        with self.assertRaisesRegex(Exception, "token"):
            parse_command(
                canonical_bytes(execute),
                "PHONE",
                "instance-phone",
            )


if __name__ == "__main__":
    unittest.main()
