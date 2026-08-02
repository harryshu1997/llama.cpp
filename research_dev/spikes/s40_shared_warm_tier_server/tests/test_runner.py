#!/usr/bin/env python3

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "s40_run_all", HERE / "run_all.py")
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class RunnerTests(unittest.TestCase):
    def test_discovery_is_cwd_independent_and_nonempty(self):
        before = Path.cwd()
        try:
            with tempfile.TemporaryDirectory(prefix="s40_runner_") as directory:
                os.chdir(directory)
                suite = RUNNER.build_suite()
        finally:
            os.chdir(before)
        self.assertEqual(
            suite.countTestCases(),
            RUNNER.EXPECTED_TESTS,
        )
        self.assertGreater(suite.countTestCases(), 0)


if __name__ == "__main__":
    unittest.main()

