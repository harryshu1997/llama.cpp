#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
if str(S39) not in sys.path:
    sys.path.insert(0, str(S39))

from capture_phone_thermal import parse_thermal


VALID = (
    b"BOOT\tboot-1\n"
    b"ZONE\tgpuss-0\t31000\n"
    b"ZONE\tgpuss-1\t32500\n"
)


class CapturePhoneThermalTests(unittest.TestCase):
    def test_valid_capture(self):
        result = parse_thermal(VALID)
        self.assertEqual(result["device_boot_id"], "boot-1")
        self.assertEqual(result["gpu_max_millic"], 32500)

    def test_duplicate_zone_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "zone name"):
            parse_thermal(VALID + b"ZONE\tgpuss-0\t32000\n")

    def test_missing_boot_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            parse_thermal(b"ZONE\tgpuss-0\t31000\n")

    def test_non_integer_temperature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            parse_thermal(b"BOOT\tboot-1\nZONE\tgpuss-0\t31.0\n")

    def test_unknown_record_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            parse_thermal(VALID + b"OTHER\tvalue\n")


if __name__ == "__main__":
    unittest.main()
