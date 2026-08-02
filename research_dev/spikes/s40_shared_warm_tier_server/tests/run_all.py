#!/usr/bin/env python3

from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
EXPECTED_TESTS = 253


def build_suite():
    return unittest.defaultTestLoader.discover(
        str(HERE), pattern="test_*.py")


if __name__ == "__main__":
    suite = build_suite()
    count = suite.countTestCases()
    if count != EXPECTED_TESTS:
        print(
            f"test discovery mismatch: expected {EXPECTED_TESTS}, found {count}")
        raise SystemExit(2)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
