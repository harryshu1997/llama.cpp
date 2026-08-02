#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
EXECUTORS = HERE.parent
if str(EXECUTORS) not in sys.path:
    sys.path.insert(0, str(EXECUTORS))

from desktop_gateway import (
    DesktopGateway,
    DesktopExecutor,
    DesktopRouteSpec,
    ExecutableIdentity,
    SERVING_ENVELOPE,
    SubprocessCacheController,
    UrllibTransport,
    read_warm_tier_internal_token_from_env,
    validate_child_argv_template,
)
from runtime_binding import (
    RuntimeBinding,
    RuntimeBindingError,
    process_start_time_ticks,
)
from readiness_v23 import local_stat
from phone_gateway import (
    COMMAND_CLEANUP,
    COMMAND_DRAIN,
    COMMAND_EXECUTE,
    COMMAND_LOAD,
    COMMAND_REPLAY,
    COMMAND_UNLOAD,
)


class FakeHttp:
    def __init__(self, routes, active):
        self.routes = routes
        self.active = active
        self.prompts = []
        self.erases = []
        self.slot_uses = {}
        self.bad_cache = False
        self.instance = 1

    def model_row(self, model_id):
        spec = self.routes[model_id]
        model_offset = sorted(self.routes).index(model_id)
        process_id = 100 + self.instance + model_offset
        port = 8081 + model_offset
        result = {
            "id": model_id,
            "status": {
                "args": [],
                "value": "loaded" if self.active == model_id else "unloaded",
            },
        }
        if self.active == model_id:
            result["warm_tier_runtime"] = {
                "instance_id": f"{process_id}:{port}:{self.instance}",
                "port": port,
                "process_id": process_id,
                "schema": "llama-server-warm-tier-runtime-identity-v1",
            }
        return result

    def request(self, method, path, body):
        if method == "GET" and path == "/models":
            return 200, {
                "data": [self.model_row(model_id) for model_id in self.routes],
                "object": "list",
            }
        if method == "GET" and path.startswith("/props?model="):
            return 200, {"model_path": "/models/fake.gguf"}
        if method == "POST" and path == "/models/load":
            self.active = body["model"]
            self.instance += 1
            return 200, {"success": True}
        if method == "POST" and path == "/models/unload":
            if self.active != body["model"]:
                return 409, {"success": False}
            self.active = None
            return 200, {"success": True}
        if method == "POST" and path.startswith("/slots/"):
            self.erases.append((path, body))
            slot_id = int(path.split("/")[2].split("?")[0])
            self.slot_uses.pop(slot_id, None)
            return 200, {"id_slot": slot_id}
        if method == "POST" and path == "/completion":
            self.prompts.append(body)
            slot_id = body["id_slot"]
            reused = slot_id in self.slot_uses
            self.slot_uses[slot_id] = list(body["prompt"])
            cached = 0
            if reused and not self.bad_cache:
                cached = max(1, len(body["prompt"]) - 1)
            n_predict = body["n_predict"]
            return 200, {
                "id_slot": slot_id,
                "tokens": [] if n_predict == 0 else [1000 + len(self.prompts)],
                "tokens_cached": cached,
                "tokens_evaluated": len(body["prompt"]),
            }
        raise AssertionError((method, path, body))


class FakeProbe:
    def inspect(self, spec, process_id, model_path, port):
        argv = [
            str(port) if value == "{PORT}" else value
            for value in spec.child_argv_template
        ]
        return {
            "argv": argv,
            "artifact_certificate_sha256":
                spec.artifact_certificate_sha256,
            "backend": spec.backend,
            "child_argv_sha256": "e" * 64,
            "child_executable": {
                "bytes": spec.child_executable.bytes,
                "path": spec.child_executable.path,
                "sha256": spec.child_executable.sha256,
            },
            "device_uuid": None if spec.backend == "CPU" else spec.device_uuid,
            "device_name": (
                None if spec.backend == "CPU" else spec.device_name
            ),
            "device_memory_free_mib": (
                None
                if spec.backend == "CPU"
                else spec.minimum_free_device_memory_mib
            ),
            "device_memory_total_mib": (
                None if spec.backend == "CPU" else spec.device_memory_total_mib
            ),
            "host_boot_id": spec.host_boot_id,
            "model_path": model_path,
            "model_sha256": spec.model_sha256,
            "minimum_free_device_memory_mib":
                spec.minimum_free_device_memory_mib,
            "native_model_id": spec.native_model_id,
            "n_gpu_layers": spec.n_gpu_layers,
            "process_id": process_id,
            "process_start_ticks": 10000 + process_id,
            "port": port,
            "qualification": spec.qualification,
            "readiness_lock_sha256": spec.readiness_lock_sha256,
            "readiness_phase_id": spec.readiness_phase_id,
            "schema": "s40-desktop-runtime-probe-v3",
            "serving_envelope": SERVING_ENVELOPE,
            "slot_save_path": spec.slot_save_path,
        }

    def process_exited(self, process_id, process_start_ticks, timeout_s):
        del process_id, process_start_ticks, timeout_s
        return True


class NeverExitedProbe(FakeProbe):
    def process_exited(self, process_id, process_start_ticks, timeout_s):
        del process_id, process_start_ticks, timeout_s
        return False


class FakeCache:
    def __init__(self):
        self.calls = []

    def prepare(self, spec, regime):
        self.calls.append((spec.model_id, regime))
        return {
            "argv": ["cache-control", regime, spec.model_path],
            "completed_ns": 2,
            "exit_code": 0,
            "output": {
                "model_path": spec.model_path,
                "model_stat": spec.model_stat,
                "regime": regime,
                "schema": "s40-cache-control-result-v1",
                "success": True,
            },
            "schema": "s40-cache-control-evidence-v1",
            "started_ns": 1,
            "stderr": "",
            "success": True,
        }


def route(model_id, role="CUDA0"):
    model_path = "/models/fake.gguf"
    slot_save_path = f"/tmp/s40-slot-{model_id}"
    executable = ExecutableIdentity(
        "/tmp/llama-server",
        1,
        "e" * 64,
    )
    n_gpu_layers = "20" if role == "CUDA0" else "0"
    return DesktopRouteSpec(
        model_id=model_id,
        native_model_id=model_id,
        model_sha256=("a" if model_id == "a" else "b") * 64,
        model_path=model_path,
        artifact_certificate_sha256="c" * 64,
        model_stat={
            "ctime_ns": 1,
            "device_id": 1,
            "inode": 1,
            "mode": 0o100444,
            "mtime_ns": 1,
            "size": 1,
        },
        backend="CUDA" if role == "CUDA0" else "CPU",
        child_argv_template=(
            executable.path,
            "-m", model_path,
            "-ngl", n_gpu_layers,
            "--slot-save-path", slot_save_path,
            "--port", "{PORT}",
            "--ctx-size", "4096",
            "--batch-size", "2048",
            "--ubatch-size", "512",
            "--parallel", "8",
            "--cont-batching",
            "--flash-attn", "on",
            "--split-mode", "none",
            "--cache-type-k", "f16",
            "--cache-type-v", "f16",
        ),
        child_executable=executable,
        device_uuid="GPU-deadbeef" if role == "CUDA0" else "NONE",
        device_name=(
            "NVIDIA GeForce RTX 4060 Ti" if role == "CUDA0" else "NONE"
        ),
        device_memory_total_mib=16380 if role == "CUDA0" else 0,
        host_boot_id="desktop-boot",
        minimum_free_device_memory_mib=512 if role == "CUDA0" else 0,
        n_gpu_layers=n_gpu_layers,
        phase="A_ONLY" if model_id == "a" else "B_ONLY",
        phase_lock_sha256="f" * 64,
        qualification={
            "model_id": model_id,
            "phase": "A_ONLY" if model_id == "a" else "B_ONLY",
            "phase_id": f"phase-{model_id}",
            "schema": "s40-route-qualification-derived-v1",
            "scope": "QUALIFIED_ROUTE",
            "status": (
                "MODEL_A_QUALIFICATION_PASS"
                if model_id == "a"
                else "MODEL_B_QUALIFICATION_PASS"
            ),
        },
        readiness_lock_sha256="d" * 64,
        readiness_phase_id=f"phase-{model_id}",
        slot_save_path=slot_save_path,
        slots=(0, 1),
    )


class DesktopArgvTests(unittest.TestCase):
    def test_child_rejects_proxy_agent_and_tool_features(self):
        spec = route("a")
        mutations = (
            ("--ui-mcp-proxy",),
            ("--ui_mcp_proxy",),
            ("--webui-mcp-proxy",),
            ("--webui_mcp_proxy",),
            ("-ag",),
            ("--agent",),
            ("--tools", "exec_shell_command"),
            ("--ui-mcp-proxy=true",),
            ("--tools=exec_shell_command",),
        )
        for extra_argv in mutations:
            with self.subTest(extra_argv=extra_argv):
                with self.assertRaisesRegex(
                        Exception, "forbidden server feature"):
                    validate_child_argv_template(
                        [*spec.child_argv_template, *extra_argv],
                        spec.child_executable,
                        spec.model_path,
                        spec.n_gpu_layers,
                        spec.slot_save_path,
                    )


def command(
    command_id,
    kind,
    model_id,
    request_id="",
    committed=None,
    total=0,
):
    prompt = [1, 2, 3, 4] if request_id else []
    committed = list(committed or [])
    return {
        "command_id": command_id,
        "controller_epoch": 1,
        "executor_id": "GPU",
        "executor_instance_id": "instance-gpu",
        "kind": kind,
        "max_output_tokens": 1 if kind == COMMAND_EXECUTE else 0,
        "model_id": model_id,
        "request": {
            "committed_output_tokens": committed,
            "model_id": model_id if request_id else "",
            "owner_id": "GPU" if request_id else "",
            "ownership_epoch": 1 if request_id else 0,
            "position": len(prompt) + len(committed),
            "prompt_tokens": prompt,
            "publication_index": len(committed),
            "request_id": request_id,
            "state": 1 if request_id else 0,
        },
        "request_id": request_id,
        "schema": "llama-server-warm-tier-command-v3",
        "total_output_tokens": total,
    }


class DesktopGatewayTests(unittest.TestCase):
    def test_internal_token_file_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            token = root / "token"
            token.write_bytes(b"a" * 64 + b"\n")
            token.chmod(0o600)
            with mock.patch.dict(
                    os.environ,
                    {"LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE": str(token)},
                    clear=False):
                self.assertEqual(
                    read_warm_tier_internal_token_from_env(),
                    "a" * 64,
                )

                token.chmod(0o644)
                with self.assertRaisesRegex(
                        Exception, "internal token metadata"):
                    read_warm_tier_internal_token_from_env()
                token.chmod(0o600)

                token.write_bytes(b"g" * 64 + b"\n")
                with self.assertRaisesRegex(
                        Exception, "internal token"):
                    read_warm_tier_internal_token_from_env()
                token.write_bytes(b"a" * 64 + b"\n")

                link = root / "token-link"
                link.symlink_to(token)
                with mock.patch.dict(
                        os.environ,
                        {
                            "LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE":
                                str(link),
                        },
                        clear=False):
                    with self.assertRaisesRegex(
                            Exception, "internal token metadata"):
                        read_warm_tier_internal_token_from_env()

    def test_http_transport_marks_internal_requests(self):
        captured = {}

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, _limit):
                return b"{}"

        def fake_urlopen(request, timeout):
            captured.update(dict(request.header_items()))
            self.assertEqual(timeout, 1.0)
            return Response()

        transport = UrllibTransport(
            "http://127.0.0.1:8080",
            1.0,
            "a" * 64,
        )
        with mock.patch(
                "desktop_gateway.urllib.request.urlopen",
                side_effect=fake_urlopen):
            status, value = transport.request("GET", "/models", None)
        self.assertEqual(status, 200)
        self.assertEqual(value, {})
        self.assertEqual(
            captured.get("X-llama-warm-tier-internal"),
            "a" * 64,
        )

    def make_executor(self):
        routes = {"a": route("a"), "b": route("b")}
        http = FakeHttp(routes, "a")
        executor = DesktopExecutor(
            "GPU",
            "GPU",
            "SINGLE_ACTIVE",
            routes,
            http,
            ("a",),
            FakeCache(),
            "WARM_CACHE",
            runtime_probe=FakeProbe(),
        )
        return executor, http

    def test_authentication_failure_is_fatal_before_command_read(self):
        class RejectingAuthenticator:
            @staticmethod
            def authenticate(connection):
                del connection
                raise RuntimeBindingError("rejected peer")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executor, _ = self.make_executor()
            gateway = DesktopGateway(
                root / "gateway.sock",
                root / "evidence.jsonl",
                executor,
                2.0,
                controller_authenticator=RejectingAuthenticator(),
                executor_config_sha256="1" * 64,
                profile_lock_sha256=None,
                run_id="run-1",
                runtime_binding=RuntimeBinding(
                    executor_id="GPU",
                    executor_instance_id="instance-gpu",
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
                json.dumps(command(1, COMMAND_DRAIN, "a")).encode("ascii")
                + b"\n"
            )
            controller_side.shutdown(socket.SHUT_WR)
            gateway._serve_one(gateway_side)
            controller_side.close()
            self.assertTrue(gateway._stopping.is_set())
            self.assertIsInstance(
                gateway._fatal_error,
                RuntimeBindingError,
            )
            self.assertEqual(executor._draining, set())
            gateway._server.close()
            executor.close()
            gateway._evidence.close()

    def test_sigterm_after_socket_readiness_cleans_gateway(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            socket_path = root / "gateway.sock"
            evidence = root / "evidence.jsonl"
            fixture = HERE / "desktop_gateway_shutdown_fixture.py"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(fixture),
                    "--evidence",
                    str(evidence),
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
            deadline = time.monotonic() + 5
            while not socket_path.exists() and process.poll() is None:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            self.assertIsNone(process.poll())
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stderr.decode("utf-8"))
            self.assertEqual(stdout, b"")
            self.assertEqual(stderr, b"")
            self.assertFalse(socket_path.exists())
            rows = [
                json.loads(line)
                for line in evidence.read_text(encoding="ascii").splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                rows[-1]["schema"],
                "s40-desktop-cleanup-evidence-v2",
            )
            self.assertTrue(rows[-1]["success"])

    def test_one_token_quanta_reuse_one_native_slot(self):
        executor, http = self.make_executor()
        first = executor.handle(
            command(1, COMMAND_EXECUTE, "a", "r1", total=2)
        )
        self.assertFalse(first["request_complete"])
        token = first["publications"][0]["token"]
        second = executor.handle(
            command(2, COMMAND_EXECUTE, "a", "r1", [token], total=2)
        )
        self.assertTrue(second["request_complete"])
        self.assertEqual(
            [body["id_slot"] for body in http.prompts],
            [0, 0],
        )
        self.assertTrue(all(body["cache_prompt"] for body in http.prompts))
        self.assertTrue(all(body["n_predict"] == 1 for body in http.prompts))
        self.assertTrue(all(body["temperature"] == 0.0 for body in http.prompts))
        evidence = executor.take_execute_evidence(2)
        self.assertTrue(evidence["resident_session_reused"])
        self.assertFalse(evidence["full_history_per_token_reprefill"])
        self.assertGreater(evidence["tokens_cached"], 0)

    def test_empty_bootstrap_load_and_terminal_cleanup(self):
        routes = {"a": route("a"), "b": route("b")}
        http = FakeHttp(routes, None)
        executor = DesktopExecutor(
            "GPU",
            "GPU",
            "SINGLE_ACTIVE",
            routes,
            http,
            (),
            FakeCache(),
            "WARM_CACHE",
            runtime_probe=FakeProbe(),
        )
        self.assertEqual(executor.runtime_inventory(), [])
        loaded = executor.handle(command(1, COMMAND_LOAD, "a"))
        self.assertTrue(loaded["success"])
        lifecycle = executor.take_lifecycle_evidence(1)
        self.assertEqual(lifecycle["operation"], "LOAD")
        self.assertEqual(
            [row["status"] for row in lifecycle["preflight"]["models"]],
            ["unloaded", "unloaded"],
        )
        cleanup = executor.close()
        self.assertTrue(cleanup["success"])
        self.assertEqual(cleanup["remaining_active_models"], [])
        self.assertEqual(http.active, None)

    def test_real_warm_cache_control_is_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.gguf"
            model.write_bytes(b"x" * (1024 * 1024))
            spec = replace(
                route("a"),
                model_path=str(model),
                model_stat=local_stat(model),
            )
            result = SubprocessCacheController(10).prepare(
                spec,
                "WARM_CACHE",
            )
        self.assertTrue(result["success"])
        self.assertGreaterEqual(
            result["output"]["after"]["resident_ppm"],
            950_000,
        )

    def test_cache_helper_does_not_inherit_warm_tier_capability(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.gguf"
            model.write_bytes(b"x")
            spec = replace(
                route("a"),
                model_path=str(model),
                model_stat=local_stat(model),
            )
            output = {
                "after": {"resident_ppm": 1_000_000},
                "before": {"resident_ppm": 1_000_000},
                "completed_ns": 2,
                "comparison": "at_least",
                "limit_ppm": 950_000,
                "model_path": str(model),
                "model_stat": spec.model_stat,
                "regime": "WARM_CACHE",
                "schema": "s40-cache-control-result-v1",
                "started_ns": 1,
                "success": True,
            }
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    json.dumps(
                        output,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("ascii")
                    + b"\n"
                ),
                stderr=b"",
            )
            environment = {
                "LLAMA_SERVER_WARM_TIER_CONFIG": "/tmp/config",
                "LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE": "/tmp/token",
                "LLAMA_SERVER_WARM_TIER_CONFIG_EXTRA": "keep-config",
                "LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE_EXTRA":
                    "keep-token",
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                with mock.patch.object(
                        subprocess, "run", return_value=completed) as run:
                    SubprocessCacheController(10).prepare(
                        spec,
                        "WARM_CACHE",
                    )
        helper_environment = run.call_args.kwargs["env"]
        self.assertNotIn(
            "LLAMA_SERVER_WARM_TIER_CONFIG",
            helper_environment,
        )
        self.assertNotIn(
            "LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE",
            helper_environment,
        )
        self.assertEqual(
            helper_environment[
                "LLAMA_SERVER_WARM_TIER_CONFIG_EXTRA"
            ],
            "keep-config",
        )
        self.assertEqual(
            helper_environment[
                "LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE_EXTRA"
            ],
            "keep-token",
        )

    def test_replay_cleanup_and_model_reload(self):
        executor, http = self.make_executor()
        replay = command(1, COMMAND_REPLAY, "a", "r1")
        result = executor.handle(replay)
        self.assertTrue(result["has_replay_snapshot"])
        executor.handle(command(2, COMMAND_CLEANUP, "a", "r1"))
        self.assertEqual(
            http.erases,
            [
                (
                    "/slots/0?action=erase",
                    {"model": "a"},
                )
            ],
        )
        executor.handle(command(3, COMMAND_DRAIN, "a"))
        executor.handle(command(4, COMMAND_UNLOAD, "a"))
        executor.handle(command(5, COMMAND_LOAD, "b"))
        self.assertEqual(http.active, "b")
        self.assertEqual(executor._active_models, {"b"})

    def test_unload_rejects_native_child_that_did_not_exit(self):
        routes = {"a": route("a"), "b": route("b")}
        http = FakeHttp(routes, "a")
        executor = DesktopExecutor(
            "GPU",
            "GPU",
            "SINGLE_ACTIVE",
            routes,
            http,
            ("a",),
            FakeCache(),
            "WARM_CACHE",
            lifecycle_timeout_s=0.01,
            runtime_probe=NeverExitedProbe(),
        )
        executor.handle(command(1, COMMAND_DRAIN, "a"))
        with self.assertRaisesRegex(Exception, "child did not exit"):
            executor.handle(command(2, COMMAND_UNLOAD, "a"))
        self.assertIn("a", executor._active_models)
        self.assertIsNone(executor.take_lifecycle_evidence(2))

    def test_reused_slot_without_cache_evidence_fails_closed(self):
        executor, http = self.make_executor()
        first = executor.handle(
            command(1, COMMAND_EXECUTE, "a", "r1", total=2)
        )
        http.bad_cache = True
        token = first["publications"][0]["token"]
        with self.assertRaisesRegex(Exception, "fully re-prefilled"):
            executor.handle(
                command(2, COMMAND_EXECUTE, "a", "r1", [token], total=2)
            )

    def test_drain_rejects_resident_execute_and_replay_without_state_loss(self):
        executor, http = self.make_executor()
        first = executor.handle(
            command(1, COMMAND_EXECUTE, "a", "r1", total=2)
        )
        self.assertIn("r1", executor._sessions)
        executor.handle(command(2, COMMAND_DRAIN, "a"))

        token = first["publications"][0]["token"]
        with self.assertRaisesRegex(Exception, "route is draining"):
            executor.handle(
                command(3, COMMAND_EXECUTE, "a", "r1", [token], total=2)
            )
        with self.assertRaisesRegex(Exception, "route is draining"):
            executor.handle(command(4, COMMAND_REPLAY, "a", "r1"))

        self.assertIn("r1", executor._sessions)
        self.assertEqual(len(http.erases), 0)
        executor.close()

    def test_two_requests_receive_distinct_slots(self):
        executor, http = self.make_executor()
        first = executor.handle(
            command(1, COMMAND_EXECUTE, "a", "r1", total=1)
        )
        second = executor.handle(
            command(2, COMMAND_EXECUTE, "a", "r2", total=1)
        )
        self.assertTrue(first["request_complete"])
        self.assertTrue(second["request_complete"])
        self.assertEqual(
            [body["id_slot"] for body in http.prompts],
            [0, 1],
        )

    def test_static_partial_serves_both_and_rejects_lifecycle(self):
        routes = {"a": route("a"), "b": route("b")}
        http = FakeHttp(routes, "a")

        def both_rows():
            rows = []
            for model_id in routes:
                row = http.model_row(model_id)
                row["status"]["value"] = "loaded"
                if "warm_tier_runtime" not in row:
                    model_offset = sorted(routes).index(model_id)
                    process_id = 100 + http.instance + model_offset
                    port = 8081 + model_offset
                    row["warm_tier_runtime"] = {
                        "instance_id":
                            f"{process_id}:{port}:{http.instance}",
                        "port": port,
                        "process_id": process_id,
                        "schema":
                            "llama-server-warm-tier-runtime-identity-v1",
                    }
                rows.append(row)
            return rows

        original_request = http.request

        def request(method, path, body):
            if method == "GET" and path == "/models":
                return 200, {"data": both_rows(), "object": "list"}
            return original_request(method, path, body)

        http.request = request
        executor = DesktopExecutor(
            "GPU",
            "GPU",
            "DUAL_STATIC_PARTIAL",
            routes,
            http,
            ("a", "b"),
            FakeCache(),
            "WARM_CACHE",
            runtime_probe=FakeProbe(),
        )
        self.assertTrue(
            executor.handle(
                command(1, COMMAND_EXECUTE, "a", "ra", total=1)
            )["success"]
        )
        self.assertTrue(
            executor.handle(
                command(2, COMMAND_EXECUTE, "b", "rb", total=1)
            )["success"]
        )
        self.assertEqual(
            [body["model"] for body in http.prompts],
            ["a", "b"],
        )
        with self.assertRaisesRegex(Exception, "rejects model lifecycle"):
            executor.handle(command(3, COMMAND_DRAIN, "a"))

    def test_static_partial_rejects_two_aliases_of_one_native_child(self):
        routes = {"a": route("a"), "b": route("b")}
        http = FakeHttp(routes, "a")

        def request(method, path, body):
            if method == "GET" and path == "/models":
                row_a = http.model_row("a")
                row_b = http.model_row("b")
                row_a["status"]["value"] = "loaded"
                row_b["status"]["value"] = "loaded"
                row_b["warm_tier_runtime"] = row_a["warm_tier_runtime"]
                return 200, {"data": [row_a, row_b], "object": "list"}
            return FakeHttp.request(http, method, path, body)

        http.request = request
        with self.assertRaisesRegex(Exception, "distinct native children"):
            DesktopExecutor(
                "GPU",
                "GPU",
                "DUAL_STATIC_PARTIAL",
                routes,
                http,
                ("a", "b"),
                FakeCache(),
                "WARM_CACHE",
                runtime_probe=FakeProbe(),
            )


if __name__ == "__main__":
    unittest.main()
