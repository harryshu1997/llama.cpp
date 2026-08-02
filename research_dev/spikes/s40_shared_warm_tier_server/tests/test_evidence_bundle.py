#!/usr/bin/env python3

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
S40 = HERE.parent
sys.path.insert(0, str(S40))

from evidence_bundle import (  # noqa: E402
    BundleError,
    build_bundle,
    validate_bundle,
)


class EvidenceBundleTests(unittest.TestCase):
    def test_build_and_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            result = build_bundle(bundle)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(
                result["python_argv_prefix"][1:4], ["-I", "-S", "-B"])
            self.assertEqual(result["environment"], {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            })
            validate_bundle(
                bundle,
                bundle / "MANIFEST.json",
                result["manifest_sha256"],
            )

    def test_isolated_launcher_ignores_ambient_python_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            result = build_bundle(bundle)
            injected = root / "injected"
            injected.mkdir()
            marker = root / "loaded"
            (injected / "sitecustomize.py").write_text(
                f"open({str(marker)!r}, 'w').write('loaded')\n",
                encoding="ascii",
            )
            process = subprocess.run(
                [
                    *result["python_argv_prefix"],
                    str(bundle / "gpu_isolation.py"),
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

    def test_dependency_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            result = build_bundle(bundle)
            (bundle / "event_evidence.py").write_bytes(b"changed\n")
            with self.assertRaisesRegex(BundleError, "file changed"):
                validate_bundle(
                    bundle,
                    bundle / "MANIFEST.json",
                    result["manifest_sha256"],
                )

    def test_bytecode_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            build_bundle(bundle)
            cache = bundle / "__pycache__"
            cache.mkdir()
            (cache / "stale.pyc").write_bytes(b"stale")
            with self.assertRaisesRegex(
                    BundleError, "contains bytecode"):
                validate_bundle(bundle, bundle / "MANIFEST.json")


if __name__ == "__main__":
    unittest.main()
