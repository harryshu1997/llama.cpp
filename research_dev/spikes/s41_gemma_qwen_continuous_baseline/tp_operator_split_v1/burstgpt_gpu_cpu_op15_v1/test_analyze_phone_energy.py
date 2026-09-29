#!/usr/bin/env python3

from pathlib import Path
import tempfile
import unittest

import analyze_phone_energy


class PhoneEnergyTest(unittest.TestCase):
    def test_oplus_battery_current_is_milliamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.tsv"
            path.write_text(
                "uptime_s\tusb_current_ua\tusb_voltage_uv\t"
                "battery_current_ma\tbattery_voltage_uv\t"
                "battery_charge_counter_uah\n"
                "1.0\t500000\t5000000\t250\t4000000\t5000000\n"
                "2.0\t500000\t5000000\t250\t4000000\t4999931\n"
                "3.0\t500000\t5000000\t250\t4000000\t4999861\n",
                encoding="ascii",
            )
            rows = analyze_phone_energy.read_samples(path)
            self.assertEqual(rows[0]["battery_discharge_w"], 1.0)

    def test_legacy_header_requires_explicit_reinterpretation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.tsv"
            path.write_text(
                "uptime_s\tusb_current_ua\tusb_voltage_uv\t"
                "battery_current_ua\tbattery_voltage_uv\n"
                "1.0\t500000\t5000000\t250\t4000000\n"
                "2.0\t500000\t5000000\t250\t4000000\n"
                "3.0\t500000\t5000000\t250\t4000000\n",
                encoding="ascii",
            )
            with self.assertRaises(analyze_phone_energy.run_trace.RunError):
                analyze_phone_energy.read_samples(path)
            rows = analyze_phone_energy.read_samples(
                path, legacy_battery_current_ma=True
            )
            self.assertEqual(rows[0]["battery_discharge_w"], 1.0)

    def test_mapping_and_integration(self):
        before = {
            "boot_id": "b",
            "host_midpoint_ns": 10_000_000_000,
            "phone_uptime_ns": 5_000_000_000,
        }
        after = {
            "boot_id": "b",
            "host_midpoint_ns": 14_000_000_000,
            "phone_uptime_ns": 9_000_000_000,
        }
        rows = [
            {
                "battery_discharge_w": 1.0,
                "phone_uptime_ns": value,
                "usb_input_w": 2.0,
            }
            for value in (5_000_000_000, 7_000_000_000, 9_000_000_000)
        ]
        mapped, slope = analyze_phone_energy.map_samples(rows, before, after)
        self.assertEqual(slope, 1.0)
        self.assertAlmostEqual(
            analyze_phone_energy.integrate(
                mapped, "usb_input_w", 11_000_000_000, 13_000_000_000
            ),
            4.0,
        )
        self.assertAlmostEqual(
            analyze_phone_energy.integrate(
                mapped, "battery_discharge_w", 11_000_000_000, 13_000_000_000
            ),
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
