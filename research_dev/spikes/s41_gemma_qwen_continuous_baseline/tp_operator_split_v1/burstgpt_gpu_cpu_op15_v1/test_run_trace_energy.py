#!/usr/bin/env python3

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import run_trace


class EnergyTest(unittest.TestCase):
    @staticmethod
    def rows():
        seconds = 1_000_000_000
        values = [
            (0, 10_000, 900),
            (seconds, 20_000, 100),
            (2 * seconds, 20_000, 400),
            (3 * seconds, 10_000, 700),
        ]
        return [
            {
                "gpu": {"power_mw": power_mw, "sample_t_ns": t_ns},
                "rapl_package": {
                    "energy_uj": energy_uj,
                    "max_energy_range_uj": 1000,
                    "sample_t_ns": t_ns,
                },
            }
            for t_ns, power_mw, energy_uj in values
        ]

    def test_gpu_power_interpolation(self):
        energy_j = run_trace.integrate_power_samples(
            self.rows(), 500_000_000, 2_500_000_000
        )
        self.assertAlmostEqual(energy_j, 37.5)

    def test_rapl_wrap_and_interpolation(self):
        energy_j = run_trace.integrate_rapl_samples(
            self.rows(), 500_000_000, 2_500_000_000
        )
        self.assertAlmostEqual(energy_j, 0.00055)

    def test_missing_rapl_fails_closed(self):
        rows = self.rows()
        rows[1]["rapl_package"] = None
        with self.assertRaises(run_trace.RunError):
            run_trace.integrate_rapl_samples(rows, 0, 2_000_000_000)

    def test_stats_include_mean(self):
        self.assertEqual(run_trace.stats([1.0, 2.0, 6.0])["mean"], 3.0)

    def test_runtime_manifest_binds_executable(self):
        executable = Path(f"/proc/{os.getpid()}/exe").resolve()
        manifest = run_trace.process_runtime_manifest(
            os.getpid(), [executable.parent]
        )
        self.assertIn(str(executable), {row["path"] for row in manifest})
        self.assertTrue(all(len(row["sha256"]) == 64 for row in manifest))

    def test_runtime_manifest_scopes_deleted_mappings(self):
        with tempfile.TemporaryDirectory() as directory:
            mapped = Path(directory) / "mapping.bin"
            mapped.write_bytes(b"x" * 4096)
            code = (
                "import mmap,os,sys,time;"
                "f=open(sys.argv[1],'r+b');"
                "m=mmap.mmap(f.fileno(),0);"
                "os.unlink(sys.argv[1]);"
                "print('READY',flush=True);"
                "time.sleep(30)"
            )
            child = subprocess.Popen(
                [sys.executable, "-c", code, str(mapped)],
                stdout=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(child.stdout.readline().strip(), "READY")
                executable = Path(sys.executable).resolve()
                manifest = run_trace.process_runtime_manifest(
                    child.pid, [executable.parent]
                )
                self.assertIn(
                    str(Path(f"/proc/{child.pid}/exe").resolve()),
                    {row["path"] for row in manifest},
                )
                with self.assertRaises(run_trace.RunError):
                    run_trace.process_runtime_manifest(
                        child.pid, [Path(directory)]
                    )
            finally:
                child.terminate()
                child.wait(timeout=5)
                child.stdout.close()


if __name__ == "__main__":
    unittest.main()
