#!/usr/bin/env python3

import copy
import hashlib
import json
from pathlib import Path
import stat
import types
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "managed_plan_gate_v1.py"
gate = types.ModuleType("managed_plan_gate_v1")
gate.__file__ = str(SOURCE)
exec(compile(SOURCE.read_bytes(), str(SOURCE), "exec"), gate.__dict__)

PLACEHOLDER = "00000000-0000-0000-0000-000000000000"
CUDA_BOOT = "11111111-1111-4111-8111-111111111111"
PHONE_PLACEHOLDER = "00000000-0000-0000-0000-000000000002"
PHONE_BOOT = "22222222-2222-4222-8222-222222222222"
MANAGED = "/opt/s39/managed-runtime-launcher"
RUNTIME = "/opt/s39/cuda-route/llama-layersplit"


def stat_row(size):
    return {
        "build_id": None,
        "ctime_ns": 10,
        "device_id": 11,
        "inode": 12 + size,
        "mode": stat.S_IFREG | 0o755,
        "mtime_ns": 13,
        "size": size,
    }


def component(component_id, path, checksum, size):
    return {
        "bytes": size,
        "component_id": component_id,
        "path": path,
        "sha256": checksum,
        "stat": stat_row(size),
    }


def encode(plan, boot_id, launcher=MANAGED):
    raw = json.dumps(
        plan,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return [
        launcher,
        "--plan-json",
        raw,
        "--plan-sha256",
        hashlib.sha256(raw.encode("ascii")).hexdigest(),
        "--boot-id",
        boot_id,
    ]


def cuda_fixture():
    components = [
        component("cuda-route.bin", RUNTIME, "b" * 64, 101),
        component("cuda-route.lib", "/opt/s39/cuda-route/libllama.so", "c" * 64, 102),
    ]
    flags = {
        "--backend": "CUDA0",
        "--driver-batch": "64",
        "--driver-context": "512",
        "--driver-max-prefill": "64",
        "--layer-end": "40",
        "--layer-start": "0",
        "--mode": "monov3",
        "--model": "/models/Qwen3-14B-Q4_K_M.gguf",
        "--port": "39125",
    }
    route = {
        "argv": [
            RUNTIME,
            "--model",
            flags["--model"],
            "--mode",
            "monov3",
            "--backend",
            "CUDA0",
            "--layer-start",
            "0",
            "--layer-end",
            "40",
            "--port",
            "39125",
            "--driver-batch",
            "64",
            "--driver-context",
            "512",
            "--driver-max-prefill",
            "64",
        ],
        "cwd": "/opt/s39/cuda-route",
        "environment": {
            "CUDA_VISIBLE_DEVICES": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "LAYERSPLIT_MEMORY_CERT": "1",
            "LAYERSPLIT_MODEL_SHA256": "d" * 64,
            "LAYERSPLIT_PLACEMENT_CERT": "1",
            "LD_LIBRARY_PATH": "/opt/s39/cuda-route",
        },
        "kind": "local_exec",
    }
    expectation = {
        "android": None,
        "bound_route": copy.deepcopy(route),
        "bundle_id": "cuda_route",
        "components": components,
        "cuda_environment": copy.deepcopy(route["environment"]),
        "cuda_flags": flags,
        "endpoint": "cuda",
        "launcher_component_id": "cuda-route.bin",
        "managed_launcher": {"path": MANAGED, "sha256": "a" * 64},
        "mode": "local_cuda",
        "prospective_route": copy.deepcopy(route),
        "runtime_launcher": {
            "component_id": "cuda-route.bin",
            "path": RUNTIME,
            "sha256": "b" * 64,
        },
    }
    plan = {
        "android": None,
        "bundle_id": "cuda_route",
        "components": components,
        "endpoint": "cuda",
        "launcher_component_id": "cuda-route.bin",
        "mode": "local_cuda",
        "route": route,
        "schema": gate.PLAN_SCHEMA,
        "ssh": None,
    }
    return expectation, plan


def android_fixture():
    runtime = "/data/local/tmp/s39/op15/llama-layersplit"
    components = [
        component("op15.bin", runtime, "e" * 64, 201),
        component("op15.lib", "/data/local/tmp/s39/op15/libllama.so", "f" * 64, 202),
    ]
    prospective_route = {
        "devices": "GPUOpenCL",
        "driver_batch": 64,
        "driver_context": 512,
        "driver_max_prefill": 64,
        "dynamic_cut": True,
        "kind": "stagenet_worker",
        "kv_unified": True,
        "layer_end": 30,
        "layer_start": 0,
        "mode": "stagenet",
        "model_path": "/data/local/tmp/s39/weights.gguf",
        "model_sha256": "1" * 64,
        "n_gpu_layers": 999,
        "placement_cert": True,
        "port": 39127,
        "runtime_root": "/data/local/tmp/s39/op15",
    }
    bound_route = copy.deepcopy(prospective_route)
    android = {
        "adb_path": "/usr/lib/android-sdk/platform-tools/adb",
        "adb_port": 5038,
        "adb_selector": "3C15AU002CL00000",
        "adb_sha256": "2" * 64,
        "boot_id_source": "phase_fresh_snapshot",
        "physical_serial": "3C15AU002CL00000",
        "shutdown_timeout_ms": 30000,
        "startup_timeout_ms": 120000,
    }
    expectation = {
        "android": android,
        "bound_route": bound_route,
        "bundle_id": "op15_stagenet",
        "components": components,
        "cuda_environment": None,
        "cuda_flags": None,
        "endpoint": "op15",
        "launcher_component_id": "op15.bin",
        "managed_launcher": {"path": MANAGED, "sha256": "a" * 64},
        "mode": "android",
        "prospective_route": prospective_route,
        "runtime_launcher": {
            "component_id": "op15.bin",
            "path": runtime,
            "sha256": "e" * 64,
        },
    }
    prospective = {
        "android": android,
        "bundle_id": "op15_stagenet",
        "components": components,
        "endpoint": "op15",
        "launcher_component_id": "op15.bin",
        "mode": "android",
        "route": prospective_route,
        "schema": gate.PLAN_SCHEMA,
        "ssh": None,
    }
    bound = copy.deepcopy(prospective)
    bound["route"] = bound_route
    return expectation, prospective, bound


class ManagedPlanGateTests(unittest.TestCase):
    def validate_cuda(self, *, expectation=None, prospective=None, bound=None,
                      prospective_argv=None, bound_argv=None):
        base_expectation, base_plan = cuda_fixture()
        expectation = expectation or base_expectation
        prospective = prospective or base_plan
        bound = bound or copy.deepcopy(base_plan)
        return gate.validate_managed_plan_pair(
            prospective_argv or encode(prospective, PLACEHOLDER),
            bound_argv or encode(bound, CUDA_BOOT),
            expectation=expectation,
            prospective_boot_id=PLACEHOLDER,
            bound_boot_id=CUDA_BOOT,
        )

    def test_cuda_pair_passes_with_nested_local_exec(self):
        result = self.validate_cuda()
        self.assertEqual(result["status"], "MANAGED_PLAN_BOUNDARY_PASS")
        self.assertEqual(result["runtime_launcher_sha256"], "b" * 64)

    def test_android_pair_passes_with_explicit_null_ssh(self):
        expectation, prospective, bound = android_fixture()
        result = gate.validate_managed_plan_pair(
            encode(prospective, PHONE_PLACEHOLDER),
            encode(bound, PHONE_BOOT),
            expectation=expectation,
            prospective_boot_id=PHONE_PLACEHOLDER,
            bound_boot_id=PHONE_BOOT,
        )
        self.assertEqual(result["endpoint"], "op15")

    def test_flattened_cuda_argv_is_rejected(self):
        _expectation, plan = cuda_fixture()
        flattened = copy.deepcopy(plan["route"]["argv"])
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "length|shape"):
            self.validate_cuda(prospective_argv=flattened)

    def test_missing_ssh_is_rejected(self):
        _expectation, plan = cuda_fixture()
        del plan["ssh"]
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "E_KEYS"):
            self.validate_cuda(prospective=plan)

    def test_non_null_ssh_is_rejected(self):
        _expectation, plan = cuda_fixture()
        plan["ssh"] = {}
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "ssh"):
            self.validate_cuda(prospective=plan)

    def test_missing_boot_option_is_rejected(self):
        _expectation, plan = cuda_fixture()
        argv = encode(plan, PLACEHOLDER)[:-2]
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "length"):
            self.validate_cuda(prospective_argv=argv)

    def test_stale_bound_boot_is_rejected(self):
        expectation, plan = cuda_fixture()
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "E_STALE_BOOT"):
            gate.validate_managed_plan_pair(
                encode(plan, PLACEHOLDER),
                encode(plan, PLACEHOLDER),
                expectation=expectation,
                prospective_boot_id=PLACEHOLDER,
                bound_boot_id=PLACEHOLDER,
            )

    def test_wrong_real_boot_is_rejected(self):
        _expectation, plan = cuda_fixture()
        wrong = "33333333-3333-4333-8333-333333333333"
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "boot_id"):
            self.validate_cuda(bound_argv=encode(plan, wrong))

    def test_boot_leaked_inside_plan_is_rejected(self):
        _expectation, plan = cuda_fixture()
        plan["route"]["environment"]["STALE_BOOT"] = PLACEHOLDER
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "E_BOOT_INSIDE_PLAN"):
            self.validate_cuda(prospective=plan)

    def test_stale_placeholder_inside_managed_plan_is_rejected(self):
        expectation, plan = cuda_fixture()
        leaked_value = "GPU-" + PLACEHOLDER
        expectation["bound_route"]["environment"]["CUDA_VISIBLE_DEVICES"] = leaked_value
        expectation["cuda_environment"]["CUDA_VISIBLE_DEVICES"] = leaked_value
        expectation["prospective_route"]["environment"]["CUDA_VISIBLE_DEVICES"] = leaked_value
        prospective = copy.deepcopy(plan)
        prospective["route"] = copy.deepcopy(expectation["prospective_route"])
        bound = copy.deepcopy(plan)
        bound["route"] = copy.deepcopy(expectation["bound_route"])
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "E_BOOT_INSIDE_PLAN"):
            self.validate_cuda(
                expectation=expectation,
                prospective=prospective,
                bound=bound,
            )

    def test_missing_nested_cuda_flag_is_rejected(self):
        _expectation, plan = cuda_fixture()
        index = plan["route"]["argv"].index("--driver-batch")
        del plan["route"]["argv"][index:index + 2]
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "route"):
            self.validate_cuda(prospective=plan)

    def test_extra_nested_cuda_flag_is_rejected(self):
        expectation, plan = cuda_fixture()
        for route in (
            expectation["prospective_route"],
            expectation["bound_route"],
            plan["route"],
        ):
            route["argv"] += ["--unbound-extra", "YES"]
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "length|flags"):
            self.validate_cuda(expectation=expectation, prospective=plan)

    def test_extra_android_route_field_is_rejected(self):
        expectation, prospective, bound = android_fixture()
        for route in (
            expectation["prospective_route"],
            expectation["bound_route"],
            prospective["route"],
            bound["route"],
        ):
            route["unbound_extra"] = "YES"
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "E_KEYS"):
            gate.validate_managed_plan_pair(
                encode(prospective, PHONE_PLACEHOLDER),
                encode(bound, PHONE_BOOT),
                expectation=expectation,
                prospective_boot_id=PHONE_PLACEHOLDER,
                bound_boot_id=PHONE_BOOT,
            )

    def test_extra_cuda_environment_is_rejected(self):
        expectation, _plan = cuda_fixture()
        expectation["cuda_environment"]["LD_PRELOAD"] = "/tmp/unbound.so"
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "E_KEYS"):
            self.validate_cuda(expectation=expectation)

    def test_plan_digest_mutation_is_rejected(self):
        _expectation, plan = cuda_fixture()
        argv = encode(plan, PLACEHOLDER)
        argv[2] = argv[2].replace('"39125"', '"39126"', 1)
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "plan_sha256"):
            self.validate_cuda(prospective_argv=argv)

    def test_runtime_component_digest_mutation_is_rejected(self):
        _expectation, plan = cuda_fixture()
        plan["components"][0]["sha256"] = "9" * 64
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "components"):
            self.validate_cuda(prospective=plan)

    def test_runtime_launcher_path_mutation_is_rejected(self):
        _expectation, plan = cuda_fixture()
        plan["route"]["argv"][0] = "/tmp/unbound-launcher"
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "route"):
            self.validate_cuda(prospective=plan)

    def test_unapproved_bound_plan_change_is_rejected(self):
        _expectation, plan = cuda_fixture()
        bound = copy.deepcopy(plan)
        bound["bundle_id"] = "other"
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "bundle_id"):
            self.validate_cuda(prospective=plan, bound=bound)

    def test_noncanonical_inline_plan_is_rejected(self):
        _expectation, plan = cuda_fixture()
        raw = json.dumps(plan, separators=(", ", ": "))
        argv = [
            MANAGED,
            "--plan-json",
            raw,
            "--plan-sha256",
            hashlib.sha256(raw.encode("ascii")).hexdigest(),
            "--boot-id",
            PLACEHOLDER,
        ]
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "canonical"):
            self.validate_cuda(prospective_argv=argv)

    def test_wrong_managed_launcher_is_rejected(self):
        _expectation, plan = cuda_fixture()
        with self.assertRaisesRegex(gate.ManagedPlanGateError, "shape"):
            self.validate_cuda(
                prospective_argv=encode(
                    plan,
                    PLACEHOLDER,
                    "/tmp/other-launcher",
                )
            )


if __name__ == "__main__":
    unittest.main()
