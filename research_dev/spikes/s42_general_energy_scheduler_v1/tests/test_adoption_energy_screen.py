#!/usr/bin/env python3
"""Tests for the matched adoption energy screen."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
DYNAMIC = ROOT / "dynamic_residency_v1"
TRACE = (
    ROOT.parent
    / "s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/"
      "burstgpt_gpu_cpu_op15_v1/REQUESTS_SEMANTIC_SOURCE.jsonl"
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load(
    "run_adoption_energy_screen",
    DYNAMIC / "run_adoption_energy_screen.py",
)
analyzer = load(
    "analyze_adoption_energy_screen_abba",
    DYNAMIC / "analyze_adoption_energy_screen_abba.py",
)


class AdoptionEnergyScreenTest(unittest.TestCase):
    def test_frozen_workload_geometry(self) -> None:
        rows = runner.load_requests(TRACE)
        qwen = runner.STAGE_INDICES + runner.TAIL_INDICES
        self.assertEqual(sum(rows[index]["input_tokens"] for index in qwen), 936)
        self.assertEqual(sum(rows[index]["output_tokens"] for index in qwen), 117)
        self.assertEqual(rows[runner.GEMMA_INDEX]["input_tokens"], 271)
        self.assertEqual(rows[runner.GEMMA_INDEX]["output_tokens"], 41)

    def test_fenced_result(self) -> None:
        line = (
            "S42_FENCED_TENSOR_RESULT status=PASS "
            "tensor=token_embd.weight source_offset=15838752 "
            "bytes=2013265920 chunk_bytes=4194304 "
            "chunks_per_window=9 fence_calls=120 armed_calls=54 "
            "copy_windows=54 copied_chunks=480 verified=true "
            "source_pinned=true adoptable=true "
            "gpu_free_min_bytes=995688448 "
            "gpu_reserve_bytes=536870912\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gemma.stderr"
            path.write_text(line, encoding="ascii")
            value = runner.parse_result(path)
        self.assertEqual(value["bytes"], 2_013_265_920)
        self.assertEqual(value["copy_windows"], 54)

    @staticmethod
    def fake_run(arm: str, repeat: int, fleet_j: float) -> dict:
        return {
            "arm": arm,
            "artifacts": {},
            "cpu_package_j": fleet_j * 0.5,
            "duration_s": 100.0 if arm == "control" else 90.0,
            "fleet_j": fleet_j,
            "gemma_ready_s": 80.0 if arm == "control" else 40.0,
            "gpu_board_j": fleet_j * 0.3,
            "phone_j": fleet_j * 0.05,
            "repeat_index": repeat,
            "request_token_hashes": {"50": "same", "52": "same"},
            "server_j": fleet_j * 0.8,
            "source_ready_s": 20.0,
            "transition": {"bytes": 2_013_265_920},
        }

    def test_positive_screen_still_requires_restore(self) -> None:
        rows = [
            self.fake_run("control", 1, 100.0),
            self.fake_run("dynamic", 1, 90.0),
            self.fake_run("dynamic", 2, 89.0),
            self.fake_run("control", 2, 101.0),
        ]
        with mock.patch.object(analyzer, "validate_run", side_effect=rows):
            value = analyzer.analyze({
                "control_r1": Path("a"),
                "dynamic_r1": Path("b"),
                "dynamic_r2": Path("c"),
                "control_r2": Path("d"),
            })
        self.assertEqual(
            value["admission"],
            "ENERGY_SCREEN_PASS_FALLBACK_AND_RESTORE_RECEIPTS_PENDING",
        )
        self.assertFalse(value["full_trace_authorized"])
        self.assertTrue(all(value["outcome_gates"].values()))

    def test_regression_blocks_full_trace(self) -> None:
        rows = [
            self.fake_run("control", 1, 100.0),
            self.fake_run("dynamic", 1, 102.0),
            self.fake_run("dynamic", 2, 103.0),
            self.fake_run("control", 2, 101.0),
        ]
        with mock.patch.object(analyzer, "validate_run", side_effect=rows):
            value = analyzer.analyze({
                "control_r1": Path("a"),
                "dynamic_r1": Path("b"),
                "dynamic_r2": Path("c"),
                "control_r2": Path("d"),
            })
        self.assertEqual(
            value["admission"], "ENERGY_SCREEN_FAIL_FULL_TRACE_BLOCKED"
        )
        self.assertFalse(value["full_trace_authorized"])


if __name__ == "__main__":
    unittest.main()
