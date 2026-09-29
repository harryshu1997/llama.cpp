#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "kernel_energy_v1"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import energy_common  # noqa: E402
import materialize_profile  # noqa: E402
import model_switch_probe  # noqa: E402


class KernelEnergyToolsTest(unittest.TestCase):
    def test_power_integral(self) -> None:
        points = [(0, 2.0), (1_000_000_000, 4.0), (2_000_000_000, 2.0)]
        self.assertAlmostEqual(
            energy_common.integrate_power(
                points, 500_000_000, 1_500_000_000
            ),
            3.5,
        )

    def test_rapl_wrap(self) -> None:
        rows = [
            {
                "monotonic_ns": 0,
                "rapl_energy_uj": 900,
                "rapl_max_energy_range_uj": 1000,
            },
            {
                "monotonic_ns": 1_000_000_000,
                "rapl_energy_uj": 100,
                "rapl_max_energy_range_uj": 1000,
            },
            {
                "monotonic_ns": 2_000_000_000,
                "rapl_energy_uj": 300,
                "rapl_max_energy_range_uj": 1000,
            },
        ]
        self.assertAlmostEqual(
            energy_common.integrate_rapl(
                rows, 500_000_000, 1_500_000_000, "monotonic_ns"
            ),
            0.0002,
        )

    def test_markers(self) -> None:
        self.assertEqual(
            energy_common.parse_desktop_window(
                "ENERGY_WINDOW_START unix_ns=10 mode=ffn\n"
                "ENERGY_WINDOW_END unix_ns=20 mode=ffn\n"
            ),
            (10, 20),
        )
        self.assertEqual(
            energy_common.parse_phone_window(
                "PHONE_ENERGY_WINDOW_START phone_uptime_s=1.25 label=x\n"
                "PHONE_ENERGY_WINDOW_END phone_uptime_ns=2000000000 label=x\n"
            ),
            (1_250_000_000, 2_000_000_000),
        )

    def test_phone_energy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.tsv"
            path.write_text(
                "uptime_s\tusb_current_ua\tusb_voltage_uv\t"
                "battery_current_ma\tbattery_voltage_uv\t"
                "battery_charge_counter_uah\n"
                "1.0\t1000000\t5000000\t0\t4000000\t100\n"
                "2.0\t1000000\t5000000\t0\t4000000\t100\n"
                "3.0\t1000000\t5000000\t0\t4000000\t100\n",
                encoding="ascii",
            )
            rows = energy_common.read_phone_samples(path)
            result = energy_common.phone_energy_summary(
                rows, 1_000_000_000, 3_000_000_000
            )
        self.assertEqual(result["whole_phone_average_power_w"], 5.0)
        self.assertEqual(result["whole_phone_energy_j"], 10.0)

    def test_duplicate_phone_uptime_is_coalesced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.tsv"
            path.write_text(
                "uptime_s\tusb_current_ua\tusb_voltage_uv\t"
                "battery_current_ma\tbattery_voltage_uv\t"
                "battery_charge_counter_uah\n"
                "1.0\t1000000\t5000000\t0\t4000000\t100\n"
                "2.0\t1000000\t5000000\t0\t4000000\t100\n"
                "2.0\t2000000\t5000000\t0\t4000000\t100\n"
                "3.0\t1000000\t5000000\t0\t4000000\t100\n",
                encoding="ascii",
            )
            rows = energy_common.read_phone_samples(path)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["usb_input_w"], 7.5)

    def test_repeat_summary_reports_held_out_error(self) -> None:
        result = materialize_profile.summarize([100.0, 101.0, 99.0])
        self.assertEqual(result["median"], 100)
        self.assertEqual(result["repeat_count"], 3)
        self.assertLess(result["repeat_loo_max_error_ppm"], 20_000)

    def test_invalid_repeat_is_not_profile_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "case-r1.json",
                "case-r2.json",
                "case-r3.json",
                "case-r3.invalid-warm.json",
            ):
                root.joinpath(name).write_text("{}\n", encoding="ascii")
            rows = materialize_profile.exact_files(root, "case")
        self.assertEqual([path.name for path in rows], [
            "case-r1.json", "case-r2.json", "case-r3.json"
        ])

    def test_model_switch_requires_complete_cuda_offload(self) -> None:
        self.assertEqual(
            model_switch_probe.offloaded_layers("offloaded 41/41 layers to GPU"),
            "41/41",
        )
        with self.assertRaises(RuntimeError):
            model_switch_probe.offloaded_layers(
                "offloaded 40/41 layers to GPU"
            )

    def test_nonnegative_link_fit(self) -> None:
        fixed, slope = materialize_profile.fit_nonnegative_line([
            (100, 60),
            (200, 110),
            (300, 160),
        ])
        self.assertEqual(fixed, 10)
        self.assertEqual(slope, 0.5)

        fixed, slope = materialize_profile.fit_nonnegative_line([
            (100, 20),
            (200, 10),
        ])
        self.assertEqual(fixed, 20)
        self.assertEqual(slope, 0.0)


if __name__ == "__main__":
    unittest.main()
