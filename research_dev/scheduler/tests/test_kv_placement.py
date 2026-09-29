"""Layer placement, shared budgets and native CPU/GPU attention equivalence."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from dataclasses import replace

from research_dev.scheduler import DeviceMemoryCapacity, ModelManifest, RuntimePlacementSnapshot
from research_dev.scheduler._internal.kv_placement import plan_layer_kv, reserve_layer_kv
from research_dev.scheduler._internal.runtime_resources import (
    RuntimeMemoryLedger, RuntimeResourceError, RuntimeRemoteResidentOmissionProof,
    remote_resident_accounting,
)
from research_dev.scheduler.adapters.llama_server_contracts import (
    LlamaServerLaunchContract, _kv_cpu_layers, _kv_device_cells, _launch_contract_supports_execution,
)
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError
from research_dev.scheduler._internal.runtime_plan import RuntimePhoneShard
from research_dev.scheduler.campaigns.burstgpt.layer_kv_gate import native_phone_proofs

ROOT = Path(__file__).resolve().parents[3]
QWEN = ModelManifest.from_json(json.loads((ROOT / "research_dev/scheduler/campaigns/burstgpt/data/QWEN_MANIFEST.json").read_text()))
GIB = 1024**3


class KvPlacementTests(unittest.TestCase):
    def plan(self, **overrides):
        args = dict(context_size=32768, parallel=1, ubatch_size=512,
                    default_pool_by_layer={il: "cpu" if il < 24 else "gpu" for il in range(40)},
                    host_pool="cpu", kv_budget_by_pool={"cpu": 4 * GIB, "gpu": GIB})
        args.update(overrides)
        return plan_layer_kv(QWEN, **args)

    def test_exact_qwen_bytes_and_spill(self):
        plan = self.plan()
        self.assertEqual(dict(plan.bytes_by_pool), {"cpu": 4 * GIB, "gpu": GIB})
        self.assertEqual(plan.cpu_layers, tuple(range(32)))
        self.assertEqual(sum(row[2] for row in plan.layers), 5 * GIB)
        self.assertEqual(plan.plan_sha256, self.plan().plan_sha256)

    def test_no_spill_and_host_budget_exhaustion(self):
        self.assertEqual(self.plan(kv_budget_by_pool={"cpu": 3 * GIB, "gpu": 2 * GIB}).cpu_layers, tuple(range(24)))
        with self.assertRaisesRegex(RuntimeResourceError, "host budget"):
            self.plan(kv_budget_by_pool={"cpu": 3 * GIB, "gpu": GIB})

    def test_context_limit_and_bad_mapping(self):
        with self.assertRaisesRegex(RuntimeResourceError, "context limit"):
            self.plan(context_size=131072)
        with self.assertRaisesRegex(RuntimeResourceError, "every layer"):
            self.plan(default_pool_by_layer={0: "cpu"})

    def test_swa_not_spilled_and_shared_cache_counted_once(self):
        model = replace(QWEN, sliding_window=1024, sliding_window_pattern=tuple(il % 2 == 0 for il in range(40)))
        plan = plan_layer_kv(model, context_size=32768, parallel=1, ubatch_size=512,
            default_pool_by_layer={il: "gpu" for il in range(40)}, host_pool="cpu",
            kv_budget_by_pool={"cpu": 6 * GIB, "gpu": GIB}, shared_kv_sources={39: 37})
        self.assertTrue(all(il % 2 == 1 for il in plan.cpu_layers))
        self.assertEqual(next(row[2] for row in plan.layers if row[0] == 39), 0)

    def test_existing_ledger_prevents_double_spend(self):
        plan = self.plan()
        snapshot = RuntimePlacementSnapshot("kv-test", 0, 1, {
            pool: DeviceMemoryCapacity(pool, size, 0, 0) for pool, size in (("cpu", 6 * GIB), ("gpu", 2 * GIB))})
        ledger = RuntimeMemoryLedger()
        reservations = reserve_layer_kv(ledger, "first", plan, snapshot, host_pool="cpu",
            base_peak_bytes_by_pool={"cpu": GIB, "gpu": GIB}, prefill_policy="local-prefill")
        self.assertEqual(sum(row.reserved_bytes for row in reservations), 7 * GIB)
        with self.assertRaises(RuntimeResourceError):
            reserve_layer_kv(ledger, "second", plan, snapshot, host_pool="cpu",
                base_peak_bytes_by_pool={"cpu": GIB, "gpu": GIB}, prefill_policy="local-prefill")
        self.assertEqual(len(ledger.release_owner("first")), 2)

    def test_future_release_not_credited(self):
        with self.assertRaisesRegex(RuntimeResourceError, "verified ownership"):
            reserve_layer_kv(RuntimeMemoryLedger(), "kv", self.plan(), None, host_pool="cpu",
                base_peak_bytes_by_pool={"cpu": 12 * GIB}, prefill_policy="remote-prefill")
        with self.assertRaisesRegex(RuntimeResourceError, "decode-only"):
            reserve_layer_kv(RuntimeMemoryLedger(), "kv", self.plan(), None, host_pool="cpu",
                base_peak_bytes_by_pool={"cpu": 12 * GIB}, prefill_policy="restore-before-prefill",
                remote_accounting=object())

    def test_verified_credit_excludes_boundary_bytes_and_needs_recovery(self):
        plan = self.plan()
        accounting = remote_resident_accounting(
            artifact_sha256=QWEN.artifact_sha256, desktop_pool_id="cpu", layer_mask=1,
            desktop_weights_full_bytes=3 * GIB, omitted_bytes_planned=GIB,
            proof=RuntimeRemoteResidentOmissionProof(1, GIB, GIB - 4096, "validated"),
            phone_weights_bytes_by_session={"HTP0": GIB}, phone_workspace_bytes=GIB,
            kv_bytes_by_pool=dict(plan.bytes_by_pool), transition_peak_bytes_by_pool={"cpu": 0},
            live_available_bytes_by_pool={"cpu": 3 * GIB},
            reduced_allocation_bytes_by_pool={"cpu": 2 * GIB},
            recovery_required_bytes_by_pool={"cpu": 3 * GIB}, fallback_mode="teardown")
        snapshot = RuntimePlacementSnapshot("kv-credit", 0, 1, {
            pool: DeviceMemoryCapacity(pool, size, 0, 0)
            for pool, size in (("cpu", 8 * GIB), ("gpu", 2 * GIB), ("phone", 2 * GIB))})
        args = dict(host_pool="cpu", base_peak_bytes_by_pool={"cpu": 3 * GIB, "gpu": GIB, "phone": 2 * GIB},
                    prefill_policy="remote-prefill")
        reservations = reserve_layer_kv(RuntimeMemoryLedger(), "verified", plan, snapshot,
                                       remote_accounting=accounting, **args)
        amounts = {row.resource_id: row.reserved_bytes for row in reservations}
        self.assertEqual(amounts, {"cpu": 6 * GIB + 4096, "gpu": 2 * GIB, "phone": 2 * GIB})
        for invalid in (replace(accounting, recovery_feasible=False), replace(accounting, proof=None),
                        replace(accounting, artifact_sha256="sha256:" + "0" * 64)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(RuntimeResourceError, "verified ownership"):
                reserve_layer_kv(RuntimeMemoryLedger(), "invalid", plan, snapshot,
                                 remote_accounting=invalid, **args)

    def test_launch_identity(self):
        contract = LlamaServerLaunchContract("q", 32768, 1, 512, 512, 16, "cpu", "gpu", None, {})
        self.assertNotEqual(contract, replace(contract, kv_cpu_layers=(24, 25)))
        self.assertEqual(_kv_cpu_layers({"kv_cpu_layers": "25,24"}, QWEN), (24, 25))
        for value in ("-1", "40", "1,1", "1,", "1.5", 3):
            with self.subTest(value=value), self.assertRaises(PhysicalAdapterError):
                _kv_cpu_layers({"kv_cpu_layers": value}, QWEN)

    def test_split_prefix_memory_and_identity(self):
        kwargs = dict(context_size=32768, parallel=1, ubatch_size=512,
                      default_pool_by_layer={il: "cpu" if il < 24 else "gpu" for il in range(40)},
                      host_pool="cpu", kv_budget_by_pool={"cpu": 10 * GIB, "gpu": 10 * GIB})
        plain = plan_layer_kv(QWEN, **kwargs)
        split = plan_layer_kv(QWEN, **kwargs, kv_device_cells={il: 8192 for il in range(24, 40)})
        self.assertEqual(split.cpu_layers, tuple(range(24)))
        self.assertEqual(split.bytes_by_pool["gpu"], 16 * (8192 + 512) * 4096)
        self.assertEqual(split.bytes_by_pool["cpu"], 24 * 32768 * 4096 + 16 * (32768 - 8192 + 512) * 4096)
        self.assertEqual(sum(size for _, _, size in split.layers), sum(split.bytes_by_pool.values()))
        self.assertNotEqual(plain.plan_sha256, split.plan_sha256)
        changed = plan_layer_kv(QWEN, **kwargs, kv_device_cells={24: 16384})
        self.assertNotEqual(changed.plan_sha256, split.plan_sha256)
        with self.assertRaises(RuntimeResourceError):
            plan_layer_kv(QWEN, **{**kwargs, "kv_budget_by_pool": {"cpu": 10 * GIB, "gpu": 0}},
                          kv_device_cells={24: 8192})

    def test_split_prefix_launch_and_rejection(self):
        contract = LlamaServerLaunchContract("q", 32768, 1, 512, 512, 16, "cpu", "gpu", None, {})
        split = replace(contract, kv_device_cells=((24, 8192),))
        self.assertFalse(_launch_contract_supports_execution(contract, split))
        self.assertFalse(_launch_contract_supports_execution(split, replace(split, kv_device_cells=((24, 4096),))))
        self.assertEqual(_kv_device_cells({"kv_device_cells": "25:8192,24:8192"}, QWEN), ((24, 8192), (25, 8192)))
        for value in ("24", "24:-1", "24:1,24:2", "40:256", "24:256,", 7):
            with self.subTest(value=value), self.assertRaises(PhysicalAdapterError):
                _kv_device_cells({"kv_device_cells": value}, QWEN)
        for prefix in (-1, 257, 33024):
            with self.assertRaises(PhysicalAdapterError):
                replace(contract, kv_device_cells=((24, prefix),))


class KvNativeTests(unittest.TestCase):
    def test_physical_gate_execution_proof_is_request_scoped_and_complete(self):
        shard = RuntimePhoneShard("HTP0", "session://phone/HTP0", 1, QWEN.feed_forward_length,
            100, "sha256:" + "a" * 64, "sha256:" + "b" * 64, QWEN.artifact_sha256, 1)
        call = ("S41SERVERFFNCALL context=6b762d70686f6e65:0:3:0 request=2 layer=0 "
                "tokens=3 columns=17408 payload_bytes=30720")
        warmup = "S41SERVERFFNCALL request=1 layer=0 tokens=2 columns=17408 payload_bytes=20480"
        proof, = native_phone_proofs([warmup, call], (shard,), QWEN, "kv-phone", 2, 2)
        self.assertEqual((proof.calls, proof.rows, proof.payload_bytes), (1, 3, 30720))
        self.assertEqual(proof.session_generation, shard.session_generation)
        self.assertEqual(proof.operator_plan_sha256, shard.operator_plan_sha256)
        for lines in ([warmup], [call, call], [call.replace("layer=0", "layer=1")],
                      [call.replace("columns=17408", "columns=128")],
                      [call.replace("payload_bytes=30720", "payload_bytes=10240")]):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                native_phone_proofs(lines, (shard,), QWEN, "kv-phone", 2, 2)
        with self.assertRaises(ValueError):
            native_phone_proofs([call], (shard,), QWEN, "kv-phone", 2, 3)

    def test_native_mixed_placement_logits(self):
        probe = Path(os.environ.get("S42_LLAMA_BUILD_BIN", ROOT / "build-cpu/bin")) / "llama-ffn-remote-resident-probe"
        if not probe.exists():
            self.skipTest("build the native probe first")
        import numpy as np
        from research_dev.scheduler.tests.tiny_llama_gguf import write_tiny_llama_gguf
        gpu_layers = int(os.environ.get("S42_KV_TEST_GPU_LAYERS", "0"))
        with tempfile.TemporaryDirectory(prefix="s42-kv-native-") as directory:
            root = Path(directory)
            model = write_tiny_llama_gguf(root / "model.gguf", head_dim=64)
            for flash in ("off", "on"):
                logits = []
                traces = []
                for label, extra in (("default", []), ("mixed", ["--kv-cpu-layers", "0,2"])):
                    result, values = root / f"{flash}-{label}.json", root / f"{flash}-{label}.bin"
                    run = subprocess.run([str(probe), "--model", str(model), "--out", str(result),
                        "--logits", str(values), "--gpu-layers", str(gpu_layers), "--flash-attn", flash,
                        *extra], capture_output=True, text=True, timeout=120)
                    self.assertEqual(run.returncode, 0, run.stderr[-4000:])
                    data = json.loads(result.read_text())
                    self.assertTrue(data["context_created"], data["log"])
                    self.assertEqual(data["decode"]["status"], 0)
                    if label == "mixed":
                        rows = [row for row in data["log"] if "KV_PLACEMENT" in row]
                        self.assertEqual(len(rows), 4)
                        self.assertTrue(any("layer=0 device=CPU" in row for row in rows))
                        if gpu_layers:
                            self.assertTrue(any("layer=1 device=CUDA" in row for row in rows))
                    logits.append(np.fromfile(values, dtype=np.float32))
                    traces.append(data["decode"]["argmax"])
                self.assertEqual(*traces)
                # CPU and CUDA F16 reductions need not be bit-identical.
                self.assertTrue(np.allclose(*logits, atol=1e-3, rtol=1e-3), float(np.max(np.abs(logits[0] - logits[1]))))


if __name__ == "__main__":
    unittest.main()
