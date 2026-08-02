#!/usr/bin/env python3
"""Tests for CP0-D file-cache measurement and control."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest


import cache_control


class CacheControlTests(unittest.TestCase):
    def test_warm_and_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(os.urandom(4 * 1024 * 1024))
            warmed = cache_control.warm_file(path, 256 * 1024)
            self.assertEqual(warmed["bytes"], path.stat().st_size)
            self.assertGreaterEqual(warmed["resident_ppm"], 950_000)

    def test_evict_keeps_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            content = os.urandom(4 * 1024 * 1024)
            path.write_bytes(content)
            cache_control.warm_file(path)
            result = cache_control.evict_file(path)
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(result["method"], "POSIX_FADV_DONTNEED_COMPLETE_FILE")

    def test_empty_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.bin"
            path.write_bytes(b"")
            with self.assertRaises(cache_control.CacheError):
                cache_control.resident_pages(path)


if __name__ == "__main__":
    unittest.main()
