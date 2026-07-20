#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

