#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
for path in (REPO_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_residency_v1.profile_gpu_residency import (  # noqa: E402
    MIB,
    canonical,
    capacity_status,
    integrate_gpu_energy_uj,
    parse_allocations,
    rapl_delta_uj,
    select_smoke_rows,
    server_command,
    RunningServer,
)


class DynamicResidencyProfileTests(unittest.TestCase):
    def test_allocation_parser_requires_exact_layer_count(self) -> None:
        log = "\n".join([
            "load_tensors: offloaded 15/41 layers to GPU",
            "load_tensors: CUDA0 model buffer size = 10304.33 MiB",
            "llama_kv_cache: CUDA0 KV buffer size = 1344.00 MiB",
            "sched_reserve: CUDA0 compute buffer size = 354.00 MiB",
        ])
        self.assertEqual(parse_allocations(log, 15), {
            "compute_buffer_mib": 354.0,
            "gpu_kv_buffer_mib": 1344.0,
            "model_buffer_mib": 10304.33,
            "offloaded_layers": 15,
            "total_layers": 41,
        })
        with self.assertRaisesRegex(ValueError, "placement mismatch"):
            parse_allocations(log, 18)

    def test_capacity_gate_rejects_swap_before_gpu_reserve(self) -> None:
        stages = [{
            "gpu": {"memory_free_bytes": 600 * MIB},
            "process_memory": {"swap_bytes": 1},
        }]
        self.assertEqual(
            capacity_status(stages, 512 * MIB),
            "REJECTED_PROCESS_SWAP",
        )
        stages[0]["process_memory"]["swap_bytes"] = 0
        self.assertEqual(capacity_status(stages, 512 * MIB), "CAPACITY_PASS")
        self.assertEqual(
            capacity_status(stages, 700 * MIB),
            "REJECTED_GPU_RESERVE",
        )

    def test_server_command_binds_runtime_geometry(self) -> None:
        command = server_command(
            Path("/tmp/server"),
            Path("/tmp/model"),
            alias="model-a",
            layers=15,
            ctx_size=24576,
            parallel=4,
            batch_size=2048,
            port=18981,
        )
        self.assertIn("--kv-unified", command)
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "15")
        self.assertEqual(command[command.index("--ctx-size") + 1], "24576")
        self.assertEqual(command[command.index("--parallel") + 1], "4")
        self.assertEqual(command[command.index("--port") + 1], "18981")

    def test_cpu_only_command_has_no_cuda_binding(self) -> None:
        command = server_command(
            Path("/tmp/server"),
            Path("/tmp/model"),
            alias="model-b",
            layers=0,
            ctx_size=32768,
            parallel=8,
            batch_size=4096,
            port=18982,
        )
        self.assertEqual(command[command.index("--device") + 1], "none")
        self.assertNotIn("--flash-attn", command)
        self.assertEqual(parse_allocations("CPU model buffer", 0), {
            "compute_buffer_mib": 0,
            "gpu_kv_buffer_mib": 0,
            "model_buffer_mib": 0,
            "offloaded_layers": 0,
            "total_layers": None,
        })

    def test_canonical_record_hash_is_stable(self) -> None:
        value = {"schema": "test", "status": "CAPACITY_PASS"}
        self.assertEqual(
            hashlib.sha256(canonical(value)).hexdigest(),
            hashlib.sha256(canonical(dict(reversed(list(value.items())))))
            .hexdigest(),
        )

    def test_running_server_exposes_child_pid(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None and process.stderr is not None
        running = RunningServer(
            "test",
            process,
            Path("/tmp/stdout"),
            Path("/tmp/stderr"),
            process.stdout,
            process.stderr,
        )
        self.assertEqual(running.pid, process.pid)
        process.wait(timeout=10)
        process.stdout.close()
        process.stderr.close()

    def test_smoke_rows_bind_both_effective_models(self) -> None:
        rows = [
            {
                "input_tokens": 271,
                "model_id": "qwen3-14b-q4_k_m",
                "output_tokens": 41,
                "request_index": 50,
            },
            {
                "input_tokens": 16,
                "model_id": "gemma-4-12b-it-q8_0",
                "output_tokens": 9,
                "request_index": 52,
            },
        ]
        qwen, gemma = select_smoke_rows(rows)
        self.assertEqual(qwen["request_index"], 52)
        self.assertEqual(gemma["request_index"], 50)

    def test_energy_accounting_handles_wrap_and_sample_bounds(self) -> None:
        before = {
            "energy_uj": 900,
            "max_energy_range_uj": 1000,
            "monotonic_ns": 10,
        }
        after = {
            "energy_uj": 100,
            "max_energy_range_uj": 1000,
            "monotonic_ns": 20,
        }
        self.assertEqual(rapl_delta_uj(before, after), 200)
        rows = [
            {"monotonic_ns": 2_000_000_000, "power_mw": 10_000},
            {"monotonic_ns": 3_000_000_000, "power_mw": 20_000},
        ]
        self.assertEqual(
            integrate_gpu_energy_uj(rows, 1_000_000_000, 4_000_000_000),
            40_000_000,
        )


if __name__ == "__main__":
    unittest.main()
