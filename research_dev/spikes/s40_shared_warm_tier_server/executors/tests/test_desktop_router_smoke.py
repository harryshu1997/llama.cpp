#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import hashlib
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
EXECUTORS = HERE.parent
if str(EXECUTORS) not in sys.path:
    sys.path.insert(0, str(EXECUTORS))

import desktop_router_smoke as smoke
from desktop_gateway import parse_desktop_config, parse_desktop_smoke_config
from phone_gateway import (
    COMMAND_CLEANUP,
    COMMAND_EXECUTE,
    COMMAND_LOAD,
    canonical_bytes,
    parse_command,
)


class FakeSmokeExecutor:
    def __init__(self, *, fail_load=False, fail_cleanup=False):
        self.executor_id = "GPU"
        self.executor_instance_id = "instance-gpu"
        self.fail_load = fail_load
        self.fail_cleanup = fail_cleanup
        self.loaded = False
        self.command_ids = []
        self.execute_evidence = {}
        self.requests = set()
        self.close_calls = 0
        self.seen_executor_instance_id = None

    def handle(self, command):
        self.assert_typed_command(command)
        command_id = command["command_id"]
        self.command_ids.append(command_id)
        if command["kind"] == COMMAND_LOAD:
            if self.fail_load:
                return {"success": False}
            self.loaded = True
            return {"success": True}
        if command["kind"] == COMMAND_EXECUTE:
            request = command["request"]
            request_id = request["request_id"]
            reused = request_id in self.requests
            self.requests.add(request_id)
            self.execute_evidence[command_id] = {
                "execute_quantum_tokens": 1,
                "full_history_per_token_reprefill": False,
                "publication_count": 1,
                "resident_session_reused": reused,
            }
            return {
                "publications": [{
                    "position": request["position"],
                    "token": 1000 + command_id,
                }],
                "success": True,
            }
        if command["kind"] == COMMAND_CLEANUP:
            self.requests.discard(command["request_id"])
            return {"success": True}
        raise AssertionError(command)

    def assert_typed_command(self, command):
        instance_id = command.get("executor_instance_id")
        if type(instance_id) is not str or not instance_id:
            raise AssertionError("missing executor instance")
        if self.seen_executor_instance_id is None:
            self.seen_executor_instance_id = instance_id
        elif instance_id != self.seen_executor_instance_id:
            raise AssertionError("changed executor instance")
        parsed = parse_command(
            canonical_bytes(command),
            self.executor_id,
            instance_id,
        )
        if parsed != command:
            raise AssertionError("command did not round-trip")

    def take_execute_evidence(self, command_id):
        return self.execute_evidence.pop(command_id, None)

    def take_lifecycle_evidence(self, command_id):
        if command_id != 1 or not self.loaded:
            return None
        return {"operation": "LOAD"}

    def runtime_inventory(self):
        return [{"logical_model_id": "model-a"}] if self.loaded else []

    def close(self):
        self.close_calls += 1
        active = ["model-a"] if self.loaded else []
        unloaded = []
        if self.loaded and not self.fail_cleanup:
            unloaded.append({
                "instance_id": "123:8080:1",
                "logical_model_id": "model-a",
                "native_model_id": "native-a",
                "process_exited": True,
                "process_id": 123,
                "process_start_ticks": 456,
            })
            self.loaded = False
        problems = ["injected cleanup failure"] if self.fail_cleanup else []
        return {
            "completed_ns": 20,
            "initial_active_models": active,
            "initial_busy_requests": [],
            "initial_request_sessions": [],
            "problems": problems,
            "remaining_active_models": active if self.fail_cleanup else [],
            "remaining_request_sessions": [],
            "schema": "s40-desktop-cleanup-evidence-v1",
            "started_ns": 10,
            "success": not problems,
            "unloaded": unloaded,
        }


class DesktopRouterSmokeTests(unittest.TestCase):
    @staticmethod
    def prompts():
        return [[index + 1, index + 2] for index in range(8)]

    def test_b1_b8_geometry_and_global_command_ids(self):
        executor = FakeSmokeExecutor()
        result = smoke.run_smoke_session(
            executor,
            executor.executor_instance_id,
            "model-a",
            self.prompts(),
        )
        self.assertEqual(
            [row["batch_size"] for row in result["geometries"]],
            [1, 8],
        )
        self.assertEqual(sorted(executor.command_ids), list(range(1, 83)))
        self.assertEqual(executor.command_ids[0], 1)
        self.assertEqual(len(set(executor.command_ids)), 82)
        self.assertEqual(result["last_command_id"], 82)
        self.assertTrue(
            result["cleanup_evidence"]["unloaded"][0]["process_exited"]
        )
        self.assertEqual(executor.close_calls, 1)

    def test_load_failure_still_closes_and_fails(self):
        executor = FakeSmokeExecutor(fail_load=True)
        with self.assertRaisesRegex(Exception, "smoke load failed"):
            smoke.run_smoke_session(
                executor,
                executor.executor_instance_id,
                "model-a",
                self.prompts(),
            )
        self.assertEqual(executor.command_ids, [1])
        self.assertEqual(executor.close_calls, 1)

    def test_cleanup_failure_refuses_pass(self):
        executor = FakeSmokeExecutor(fail_cleanup=True)
        with self.assertRaisesRegex(Exception, "terminal cleanup failed"):
            smoke.run_smoke_session(
                executor,
                executor.executor_instance_id,
                "model-a",
                self.prompts(),
            )
        self.assertEqual(executor.close_calls, 1)

    def test_main_returns_nonzero_on_load_failure(self):
        executor = FakeSmokeExecutor(fail_load=True)
        route = SimpleNamespace(
            model_sha256="a" * 64,
            slots=tuple(range(8)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = [
                "desktop_router_smoke.py",
                "--config",
                str(root / "config.json"),
                "--model-id",
                "model-a",
                "--output",
                str(root / "output.json"),
                "--requests",
                str(root / "requests.jsonl"),
                "--requests-sha256",
                "b" * 64,
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    smoke,
                    "parse_desktop_smoke_config",
                    return_value=(
                        "GPU",
                        "GPU",
                        "SINGLE_ACTIVE",
                        "http://127.0.0.1:8080",
                        "WARM_CACHE",
                        {"model-a": route},
                        None,
                        SimpleNamespace(
                            path="/usr/bin/nvidia-smi",
                            bytes=1,
                            sha256="d" * 64,
                        ),
                        "c" * 64,
                    ),
                ),
                mock.patch.object(
                    smoke,
                    "load_prompts",
                    return_value=self.prompts(),
                ),
                mock.patch.object(
                    smoke,
                    "read_warm_tier_internal_token_from_env",
                    return_value="e" * 64,
                ),
                mock.patch.object(
                    smoke,
                    "DesktopExecutor",
                    return_value=executor,
                ),
                mock.patch.object(smoke, "write_exclusive") as writer,
            ):
                self.assertEqual(smoke.main(), 2)
                writer.assert_not_called()
        self.assertEqual(executor.close_calls, 1)

    def test_smoke_config_is_separate_from_qualified_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.gguf"
            model.write_bytes(b"model")
            child = root / "llama-server"
            child.write_bytes(b"server")
            child.chmod(0o755)
            nvidia_smi = root / "nvidia-smi"
            nvidia_smi.write_bytes(b"smi")
            nvidia_smi.chmod(0o755)
            slots = root / "slots"
            slots.mkdir()

            def identity(path):
                return {
                    "bytes": path.stat().st_size,
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            template = [
                str(child),
                "-m", str(model),
                "-ngl", "99",
                "--slot-save-path", str(slots),
                "--port", "{PORT}",
                "--ctx-size", "4096",
                "--batch-size", "2048",
                "--ubatch-size", "512",
                "--parallel", "8",
                "--split-mode", "none",
                "--cache-type-k", "f16",
                "--cache-type-v", "f16",
                "--flash-attn", "on",
                "--cont-batching",
            ]
            qualification = {
                "gpu_uuid": "GPU-test",
                "model_bytes": model.stat().st_size,
                "model_id": "model-a",
                "model_path": str(model),
                "model_sha256": hashlib.sha256(
                    model.read_bytes()
                ).hexdigest(),
                "phase": "DESKTOP_SMOKE",
                "phase_id": "smoke-a",
                "schema": "s40-desktop-smoke-authority-v1",
                "scope": "DESKTOP_SMOKE_ONLY",
            }
            config = {
                "base_url": "http://127.0.0.1:8080",
                "cache_regime": "WARM_CACHE",
                "executor_id": "GPU",
                "mode": "SINGLE_ACTIVE",
                "nvidia_smi": identity(nvidia_smi),
                "profile_lock_sha256": None,
                "role": "GPU",
                "routes": [{
                    "backend": "CUDA",
                    "child_argv_template": template,
                    "child_executable": identity(child),
                    "device_memory_total_mib": 16380,
                    "device_name": "NVIDIA GeForce RTX 4060 Ti",
                    "device_uuid": "GPU-test",
                    "host_boot_id": "boot",
                    "minimum_free_device_memory_mib": 512,
                    "model_id": "model-a",
                    "model_path": str(model),
                    "model_sha256": qualification["model_sha256"],
                    "native_model_id": "native-a",
                    "n_gpu_layers": "99",
                    "qualification": qualification,
                    "slot_save_path": str(slots),
                    "slots": list(range(8)),
                }],
                "schema": "s40-desktop-smoke-config-v1",
            }
            path = root / "smoke-config.json"
            path.write_bytes(canonical_bytes(config))
            with mock.patch(
                "desktop_gateway.derive_desktop_smoke_qualification",
                return_value={
                    "model_id": "model-a",
                    "phase": "DESKTOP_SMOKE",
                    "phase_id": "smoke-a",
                    "schema": "s40-desktop-smoke-derived-v1",
                    "scope": "DESKTOP_SMOKE_ONLY",
                    "status": "DESKTOP_B1_B8_SMOKE_AUTHORIZED",
                },
            ):
                parsed = parse_desktop_smoke_config(path)
            self.assertEqual(parsed[5]["model-a"].phase, "DESKTOP_SMOKE")
            with self.assertRaisesRegex(Exception, "desktop config schema"):
                parse_desktop_config(path)


if __name__ == "__main__":
    unittest.main()
