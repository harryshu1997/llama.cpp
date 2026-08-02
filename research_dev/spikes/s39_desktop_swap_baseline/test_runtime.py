#!/usr/bin/env python3
"""Offline tests for CP0-D runtime arithmetic and strict helpers."""

from __future__ import annotations

from pathlib import Path
import argparse
import tempfile
import unittest


import run_desktop_baseline as runtime
import analyze_campaign as analysis


class RuntimeTests(unittest.TestCase):
    @staticmethod
    def fake_server(serving_profile: str) -> runtime.ServerProcess:
        runner = argparse.Namespace(
            models={"model": {"path": Path("/models/model.gguf")}},
            server=Path("/bin/llama-server"),
            serving_profile=serving_profile,
        )
        return runtime.ServerProcess(runner, "model", 8080, "test")

    def test_stock_default_command_omits_tuning(self) -> None:
        command = self.fake_server("stock_default").build_command()
        forbidden = {
            "--n-gpu-layers", "--split-mode", "--main-gpu", "--device",
            "--fit", "--ctx-size", "--parallel", "--batch-size",
            "--ubatch-size", "--flash-attn", "--cont-batching",
            "--kv-unified", "--no-cache-idle-slots", "--cache-type-k",
            "--cache-type-v",
        }
        self.assertFalse(forbidden & set(command))
        self.assertEqual(command[:5], [
            "/bin/llama-server", "--model", "/models/model.gguf",
            "--alias", "model",
        ])

    def test_fixed_profile_retains_explicit_envelope(self) -> None:
        command = self.fake_server("fixed_full_cuda").build_command()
        self.assertIn("--n-gpu-layers", command)
        self.assertIn("--parallel", command)
        self.assertIn("--cont-batching", command)

    def test_gpu_energy_integer_integral(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sampler = runtime.PowerSampler(
                Path(directory) / "power.jsonl", 0, lambda: None
            )
            sampler.rows = [
                {
                    "t_ns": index * 100_000_000,
                    "gpu_power_instant_mw": 10_000 + index * 1_000,
                }
                for index in range(30)
            ]
            result = sampler.integrate(100_000_000, 2_600_000_000)
            expected_nj = sum(
                (10_000 + index * 1_000) * 100_000_000 // 1_000
                for index in range(1, 26)
            )
            self.assertEqual(result["energy_nj"], expected_nj)
            sampler.writer.close()

    def test_energy_requires_bracketing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sampler = runtime.PowerSampler(
                Path(directory) / "power.jsonl", 0, lambda: None
            )
            sampler.rows = [
                {"t_ns": index * 1_000_000_000,
                 "gpu_power_instant_mw": 10_000 + index}
                for index in range(8)
            ]
            with self.assertRaises(runtime.RunError):
                sampler.integrate(-1, 4_000_000_000)
            sampler.writer.close()

    def test_energy_rejects_oversampled_constant_sensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sampler = runtime.PowerSampler(
                Path(directory) / "power.jsonl", 0, lambda: None
            )
            sampler.rows = [
                {"t_ns": index * 100_000_000, "gpu_power_instant_mw": 10_000}
                for index in range(31)
            ]
            with self.assertRaises(runtime.RunError):
                sampler.integrate(0, 3_000_000_000)
            sampler.writer.close()

    def test_proc_memory_parser_rejects_wrong_unit(self) -> None:
        with self.assertRaises(runtime.RunError):
            runtime.parse_kib("12 MB")

    def test_independent_integrator_matches_hand_result(self) -> None:
        rows = [
            {
                "t_ns": index * 100_000_000,
                "gpu_power_instant_mw": 20_000 + index * 100,
            }
            for index in range(30)
        ]
        result = analysis.integrate_power(rows, 100_000_000, 2_600_000_000)
        expected = sum(
            (20_000 + index * 100) * 100_000_000 // 1_000
            for index in range(1, 26)
        )
        self.assertEqual(result["energy_nj"], expected)

    def test_failed_run_uses_only_bracketed_completions(self) -> None:
        requests = [
            {"completion_ns": 1_000},
            {"completion_ns": 2_000},
            {"completion_ns": 3_000},
        ]
        samples = [{"t_ns": 0}, {"t_ns": 2_500}]
        self.assertEqual(
            analysis.bracketed_completion_end(requests, samples),
            (2_000, 2),
        )

    def test_failed_run_requires_a_bracketed_completion(self) -> None:
        with self.assertRaises(analysis.AnalysisError):
            analysis.bracketed_completion_end(
                [{"completion_ns": 2_000}],
                [{"t_ns": 1_000}],
            )

    def test_trailing_throughput_is_a_time_series(self) -> None:
        self.assertEqual(
            analysis.trailing_throughput([0, 5, 0, 0, 5, 0], 5),
            [0.0, 1.0, 1.0, 1.0, 2.0, 2.0],
        )

    def test_cumulative_energy_is_a_time_series(self) -> None:
        samples = [
            {
                "t_ns": index * 1_000_000_000,
                "gpu_power_instant_mw": 1_000,
            }
            for index in range(3)
        ]
        self.assertEqual(
            analysis.cumulative_energy_j(
                samples, 0, 2_000_000_000, 2
            ),
            [1.0, 2.0],
        )

    def test_graph_writers_create_nonempty_assets(self) -> None:
        run = {
            "completions": {
                "qwen3-8b-q8_0": [0, 2, 1, 0],
                "qwen3-14b-q4_k_m": [0, 0, 1, 1],
            },
            "cumulative_energy_j": [10.0, 30.0, 50.0, 60.0],
            "duration_s": 4,
            "energy_is_lower_bound": False,
            "model_throughput_rps": {
                "qwen3-8b-q8_0": 0.75,
                "qwen3-14b-q4_k_m": 0.25,
            },
            "power_w": [10.0, 80.0, 70.0, 20.0],
            "request_count": 4,
            "slo_met_count": 3,
            "stranded_request_count": 0,
            "switches_s": [1.5, 3.0],
            "throughput_series_rps": {
                "qwen3-8b-q8_0": [0.0, 0.4, 0.6, 0.6],
                "qwen3-14b-q4_k_m": [0.0, 0.0, 0.2, 0.4],
            },
            "throughput_window_s": 5,
            "throughput_rps": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            throughput_svg = Path(directory) / "throughput.svg"
            throughput_png = Path(directory) / "throughput.png"
            energy_svg = Path(directory) / "energy.svg"
            energy_png = Path(directory) / "energy.png"
            analysis.make_svg(run, run, throughput_svg)
            analysis.make_png(run, run, throughput_png)
            analysis.make_energy_svg(run, run, energy_svg)
            analysis.make_energy_png(run, run, energy_png)
            for path in (
                    throughput_svg, throughput_png, energy_svg, energy_png):
                self.assertGreater(path.stat().st_size, 1000)
            self.assertIn(
                "model throughput over time",
                throughput_svg.read_text(encoding="ascii"),
            )
            self.assertIn(
                "selected-GPU energy over time",
                energy_svg.read_text(encoding="ascii"),
            )


if __name__ == "__main__":
    unittest.main()
