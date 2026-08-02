#!/usr/bin/env python3

from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
S40 = HERE.parent
sys.path.insert(0, str(S40))

from evidence_common import EvidenceError  # noqa: E402
from resource_sampler import (  # noqa: E402
    collect_sample,
    cpu_utilization_milli_pct,
    parse_cpu_stat,
    parse_meminfo,
    parse_nvidia_csv,
    parse_process_stat,
    parse_process_status,
    process_cpu_utilization_milli_pct,
    sample_until_stopped,
)


class ResourceSamplerTests(unittest.TestCase):
    def test_nvidia_values_convert_without_float(self):
        row = parse_nvidia_csv(
            "GPU-test, 123, 456, 98.125\n",
            "GPU-test",
        )
        self.assertEqual(row["gpu_memory_used_bytes"], 123 * 1024 * 1024)
        self.assertEqual(row["gpu_memory_free_bytes"], 456 * 1024 * 1024)
        self.assertEqual(row["gpu_power_mw"], 98125)

    def test_nvidia_identity_is_load_bearing(self):
        with self.assertRaisesRegex(EvidenceError, "UUID mismatch"):
            parse_nvidia_csv("GPU-other, 1, 2, 3\n", "GPU-selected")

    def test_host_memory_counters_are_exact(self):
        row = parse_meminfo(
            "MemAvailable: 100 kB\n"
            "SwapTotal: 20 kB\n"
            "SwapFree: 15 kB\n"
        )
        self.assertEqual(row["system_mem_available_bytes"], 102400)
        self.assertEqual(row["system_swap_total_bytes"], 20480)

    def test_invalid_swap_is_rejected(self):
        with self.assertRaisesRegex(EvidenceError, "swap counters"):
            parse_meminfo(
                "MemAvailable: 100 kB\n"
                "SwapTotal: 20 kB\n"
                "SwapFree: 21 kB\n"
            )

    def test_cpu_utilization_uses_counter_delta(self):
        previous = parse_cpu_stat("cpu  10 0 10 80 0 0 0 0\n")
        current = parse_cpu_stat("cpu  20 0 20 160 0 0 0 0\n")
        self.assertEqual(cpu_utilization_milli_pct(previous, current), 20000)

    def test_process_rss_and_swap_are_required(self):
        row = parse_process_status("VmRSS: 10 kB\nVmSwap: 2 kB\n")
        self.assertEqual(row["process_rss_bytes"], 10240)
        self.assertEqual(row["process_swap_bytes"], 2048)
        with self.assertRaisesRegex(EvidenceError, "missing"):
            parse_process_status("VmRSS: 10 kB\n")

    def test_process_identity_and_cpu_ticks_are_parsed(self):
        fields = [
            "S", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19",
        ]
        ticks, start = parse_process_stat(
            "123 (server worker) " + " ".join(fields) + "\n")
        self.assertEqual(ticks, 23)
        self.assertEqual(start, 19)

    def test_process_cpu_uses_host_counter_domain(self):
        previous_host = (100, 80)
        current_host = (200, 160)
        self.assertEqual(
            process_cpu_utilization_milli_pct(
                previous_host, current_host, 10, 15),
            5_000,
        )

    def test_process_cpu_counter_regression_is_rejected(self):
        with self.assertRaisesRegex(EvidenceError, "counter delta"):
            process_cpu_utilization_milli_pct(
                (100, 80), (200, 160), 11, 10)

    def test_host_cpu_captures_child_work_when_controller_is_idle(self):
        previous_host = (100, 90)
        current_host = (200, 100)
        self.assertEqual(
            cpu_utilization_milli_pct(previous_host, current_host),
            90_000,
        )
        self.assertEqual(
            process_cpu_utilization_milli_pct(
                previous_host, current_host, 10, 10),
            0,
        )

    def test_sample_command_uses_the_explicit_executable(self):
        observed = []

        def runner(argv, **_kwargs):
            observed.append(argv)
            raise EvidenceError("sentinel")

        with self.assertRaisesRegex(EvidenceError, "sentinel"):
            collect_sample(
                "run",
                0,
                "GPU-test",
                2,
                "boot",
                1,
                None,
                None,
                Path("/captured/nvidia-smi"),
                runner,
            )
        self.assertEqual(observed[0][0], "/captured/nvidia-smi")

    def test_sampler_rejects_an_unbound_nvidia_smi(self):
        with tempfile.TemporaryDirectory(
                prefix="s40_sampler_") as directory:
            root = Path(directory)
            executable = root / "nvidia-smi"
            executable.write_bytes(b"#!/bin/sh\nexit 0\n")
            executable.chmod(0o700)
            with self.assertRaisesRegex(EvidenceError, "digest mismatch"):
                sample_until_stopped(
                    "run",
                    "GPU-test",
                    __import__("os").getpid(),
                    root / "output.jsonl",
                    root / "stop",
                    200,
                    executable,
                    "0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
