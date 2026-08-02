#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
EXECUTORS = HERE.parent
if str(EXECUTORS) not in sys.path:
    sys.path.insert(0, str(EXECUTORS))

from executor_bundle import (
    BundleError,
    build_executor_bundle,
    validate_executor_bundle,
    validate_runtime_environment,
)


class ExecutorBundleTests(unittest.TestCase):
    def build(self) -> tuple[tempfile.TemporaryDirectory, Path, dict]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        bundle = Path(temporary.name) / "bundle"
        result = build_executor_bundle(bundle)
        return temporary, bundle, result

    def test_build_validate_and_launch(self):
        _, bundle, result = self.build()
        validation = validate_executor_bundle(
            bundle,
            bundle / "MANIFEST.json",
            result["manifest_sha256"],
        )
        self.assertEqual(validation["status"], "PASS")
        self.assertEqual(
            result["python_argv_prefix"][1:4], ["-I", "-S", "-B"])
        self.assertNotIn("PYTHONPATH", result["environment"])
        environment = {
            "PATH": os.environ.get("PATH", ""),
            **result["environment"],
        }
        process = subprocess.run(
            [
                *result["python_argv_prefix"],
                str(bundle / "gateway_bridge.py"),
                "--help",
            ],
            cwd="/",
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr.decode("ascii"))

    def test_isolated_launcher_ignores_ambient_python_code(self):
        _, bundle, result = self.build()
        injected = bundle.parent / "injected"
        injected.mkdir()
        marker = bundle.parent / "loaded"
        (injected / "sitecustomize.py").write_text(
            f"open({str(marker)!r}, 'w').write('loaded')\n",
            encoding="ascii",
        )
        process = subprocess.run(
            [
                *result["python_argv_prefix"],
                str(bundle / "gateway_bridge.py"),
                "--help",
            ],
            cwd="/",
            env={
                **os.environ,
                **result["environment"],
                "PYTHONPATH": str(injected),
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        self.assertEqual(
            process.returncode, 0, process.stderr.decode("ascii"))
        self.assertFalse(marker.exists())

    def test_mutated_dependency_is_rejected_before_launch(self):
        _, bundle, result = self.build()
        dependency = bundle / "mixed_phase_batcher.py"
        dependency.write_bytes(dependency.read_bytes() + b"\n")
        process = subprocess.run(
            [
                *result["python_argv_prefix"],
                str(bundle / "phone_gateway.py"),
                "--help",
            ],
            cwd="/",
            env={
                "PATH": os.environ.get("PATH", ""),
                **result["environment"],
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        self.assertEqual(process.returncode, 2)
        self.assertIn(b"bundle file changed", process.stderr)
        self.assertNotIn(b"Traceback", process.stderr)

    def test_unlisted_file_and_bytecode_are_rejected(self):
        for name in ("extra.py", "__pycache__/extra.pyc"):
            with self.subTest(name=name):
                _, bundle, _ = self.build()
                path = bundle / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"x")
                with self.assertRaisesRegex(
                    BundleError,
                    "unexpected bundle file|contains bytecode",
                ):
                    validate_executor_bundle(
                        bundle,
                        bundle / "MANIFEST.json",
                    )

    def test_manifest_digest_and_entrypoint_are_bound(self):
        _, bundle, result = self.build()
        environment = result["environment"]
        with mock.patch.dict(os.environ, environment, clear=False):
            validate_runtime_environment(bundle / "phone_gateway.py")
            with self.assertRaisesRegex(BundleError, "environment"):
                validate_runtime_environment(EXECUTORS / "phone_gateway.py")
        changed = dict(environment)
        changed["S40_EXECUTOR_BUNDLE_SHA256"] = "0" * 64
        with mock.patch.dict(os.environ, changed, clear=False):
            with self.assertRaisesRegex(BundleError, "manifest digest"):
                validate_runtime_environment(bundle / "phone_gateway.py")


if __name__ == "__main__":
    unittest.main()
