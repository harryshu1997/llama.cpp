#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_residency_v1.analyze_prefetch_fence_run import (  # noqa: E402
    AnalysisError,
    analyze,
    canonical,
)


class PrefetchFenceAnalysisTests(unittest.TestCase):
    def write_inputs(self, root: Path, mode: str) -> dict[str, Path]:
        copied = 268_435_456 if mode == "prefetch" else 0
        chunks = 64 if mode == "prefetch" else 0
        qwen = {
            "metrics": {"duration_s": 30.0, "requests": 1},
            "paid_end_ns": 31_000_000_000,
            "paid_start_ns": 1_000_000_000,
            "prefetch_arm_ns": 1_000_000_001,
            "request_results": [{
                "input_tokens": 16,
                "output_tokens": 9,
                "request_index": 52,
                "tokens": [1, 2, 3],
            }],
            "schema": "s41-burstgpt-llama-server-result-v1",
            "server_command": [
                "/tmp/llama-server",
                "--n-gpu-layers", "15",
                "--device", "CUDA0",
            ],
            "server_energy": {
                "cpu_package_energy_j": 100.0,
                "gpu_board_energy_j": 50.0,
                "server_compute_device_energy_j": 150.0,
            },
            "status": "PASS",
        }
        phone = {
            "boundary": "paid_trace_interval",
            "schema": "s41-phone-energy-v3",
            "status": "PASS",
            "whole_phone_energy_j": 3.0,
        }
        bridge = {
            "prefetch_copied_bytes": copied,
            "prefetch_copied_chunks": chunks,
            "prefetch_copied_window_overrun_max_ms": 0.0,
            "prefetch_copied_window_overrun_p90_ms": 0.0,
            "prefetch_fence_calls": 100,
            "prefetch_fence_enabled": True,
            "prefetch_group_first_layer": 0,
            "prefetch_group_last_layer": 11,
            "reset_recoveries": 0,
            "status": "ok",
        }
        helper = " ".join([
            "PREFETCH_RESULT",
            "status=PASS",
            f"mode={mode}",
            "source_offset=15838752",
            "stage_bytes=268435456",
            "chunk_bytes=4194304",
            f"chunks_per_window={1 if mode == 'prefetch' else 0}",
            "fence_calls=100",
            "armed_calls=64",
            "copy_windows=64",
            "warmup_copy_calls=10",
            "warmup_copy_bytes=41943040",
            f"copied_bytes={copied}",
            f"copied_chunks={chunks}",
            "gpu_free_min_bytes=1073741824",
            "gpu_reserve_bytes=536870912",
            "source_resident_bytes=268435456",
            "source_pinned=true",
            "stream_priority=-5",
            "source_fnv64=7f94ae3ec50b0fdd",
            f"destination_fnv64={'7f94ae3ec50b0fdd' if mode == 'prefetch' else 'cbf29ce484222325'}",
            f"verified={'true' if mode == 'prefetch' else 'false'}",
            "adoptable=false",
        ]) + "\n"
        values = {
            "qwen": json.dumps(qwen),
            "phone": json.dumps(phone),
            "server": (
                "[ffn-split] connected layers=12 "
                "mask=0000000000000fff\n"
            ),
            "bridge": "FFNDMABUF " + json.dumps(bridge) + "\n",
            "helper": helper,
        }
        paths = {}
        for name, value in values.items():
            path = root / f"{name}.txt"
            path.write_text(value, encoding="ascii")
            paths[name] = path
        return paths

    def run_analysis(self, mode: str) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(Path(directory), mode)
            return analyze(
                mode=mode,
                qwen_path=paths["qwen"],
                phone_path=paths["phone"],
                server_path=paths["server"],
                bridge_path=paths["bridge"],
                helper_path=paths["helper"],
            )

    def test_prefetch_binds_verified_nonadoptable_bytes(self) -> None:
        result = self.run_analysis("prefetch")
        self.assertEqual(
            result["admission"],
            "FENCE_TRANSFER_PASS_NO_WEIGHT_ADOPTION",
        )
        self.assertTrue(result["gates"]["destination_bytes_verified"])
        self.assertFalse(result["weight_adoptable_by_gemma_executor"])
        self.assertEqual(result["energy"]["fleet_j"], 153.0)

    def test_observe_arm_copies_no_bytes(self) -> None:
        result = self.run_analysis("observe")
        self.assertEqual(result["helper"]["copied_bytes"], 0)
        self.assertIsNone(
            result["gates"]["transfer_completed_inside_phone_window"]
        )

    def test_record_hash_is_canonical(self) -> None:
        result = self.run_analysis("prefetch")
        claimed = result.pop("record_sha256")
        self.assertEqual(claimed, hashlib.sha256(canonical(result)).hexdigest())

    def test_gpu_reserve_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(Path(directory), "prefetch")
            text = paths["helper"].read_text(encoding="ascii")
            paths["helper"].write_text(
                text.replace(
                    "gpu_free_min_bytes=1073741824",
                    "gpu_free_min_bytes=1",
                ),
                encoding="ascii",
            )
            with self.assertRaisesRegex(AnalysisError, "GPU reserve"):
                analyze(
                    mode="prefetch",
                    qwen_path=paths["qwen"],
                    phone_path=paths["phone"],
                    server_path=paths["server"],
                    bridge_path=paths["bridge"],
                    helper_path=paths["helper"],
                )


if __name__ == "__main__":
    unittest.main()
