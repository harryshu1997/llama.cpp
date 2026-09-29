#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import attach_phone_energy  # noqa: E402


class AttachPhoneEnergyTest(unittest.TestCase):
    def test_case_integration(self) -> None:
        rows = [
            {
                "battery_discharge_w": 1.0,
                "host_ns": 0,
                "usb_input_w": 2.0,
            },
            {
                "battery_discharge_w": 1.0,
                "host_ns": 1_000_000_000,
                "usb_input_w": 2.0,
            },
            {
                "battery_discharge_w": 1.0,
                "host_ns": 2_000_000_000,
                "usb_input_w": 2.0,
            },
        ]
        case = {
            "case": {"case_id": "case-1"},
            "paid_end_ns": 1_500_000_000,
            "paid_start_ns": 500_000_000,
        }
        value = attach_phone_energy.case_energy(rows, case)
        self.assertAlmostEqual(value["usb_input_energy_j"], 2.0)
        self.assertAlmostEqual(value["battery_discharge_energy_j"], 1.0)
        self.assertAlmostEqual(value["whole_phone_energy_j"], 3.0)
        self.assertAlmostEqual(value["whole_phone_average_power_w"], 3.0)


if __name__ == "__main__":
    unittest.main()
