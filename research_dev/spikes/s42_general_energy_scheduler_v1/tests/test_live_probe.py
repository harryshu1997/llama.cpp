#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_probe import (  # noqa: E402
    adb_server_ports,
    parse_adb_device,
    parse_gpu,
    phone_usb_speed_mbps,
)


class LiveProbeTests(unittest.TestCase):
    def test_parse_gpu(self) -> None:
        value = parse_gpu(
            "NVIDIA GeForce RTX 4060 Ti, GPU-1, 580.1, P8, 7.62, 0, 779\n"
        )
        self.assertEqual(value["uuid"], "GPU-1")
        self.assertEqual(value["memory_used_mib"], 779)

    def test_parse_adb_device(self) -> None:
        value = parse_adb_device(
            "List of devices attached\nserial0 device usb:2-1 product:x\n",
            "serial0",
        )
        self.assertEqual(value["state"], "device")

    def test_match_usb_bus_device_speed(self) -> None:
        lsusb = "Bus 002 Device 117: ID 22d9:2772 OPPO Electronics Corp.\n"
        tree = (
            "/:  Bus 002.Port 001: Dev 001, Class=root_hub, 20000M/x2\n"
            "    |__ Port 002: Dev 117, If 0, Class=Imaging, 5000M\n"
        )
        self.assertEqual(phone_usb_speed_mbps(lsusb, tree, "22d9:2772"), 5000)

    def test_does_not_match_other_bus(self) -> None:
        lsusb = "Bus 002 Device 117: ID 22d9:2772 OPPO Electronics Corp.\n"
        tree = (
            "/:  Bus 001.Port 001: Dev 001, Class=root_hub, 480M\n"
            "    |__ Port 002: Dev 117, If 0, Class=Imaging, 480M\n"
        )
        self.assertIsNone(phone_usb_speed_mbps(lsusb, tree, "22d9:2772"))

    def test_discovers_only_existing_adb_server_ports(self) -> None:
        output = (
            'LISTEN 0 4096 127.0.0.1:5040 0.0.0.0:* users:(("adb",pid=1))\n'
            'LISTEN 0 4096 127.0.0.1:5037 0.0.0.0:* users:(("adb",pid=2))\n'
            'LISTEN 0 128 127.0.0.1:5038 0.0.0.0:*\n'
        )
        self.assertEqual(adb_server_ports(output), [5037, 5040])


if __name__ == "__main__":
    unittest.main()
