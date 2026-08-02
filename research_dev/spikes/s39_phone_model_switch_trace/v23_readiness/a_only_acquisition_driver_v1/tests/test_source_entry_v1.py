#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import py_compile
import stat
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
ENTRYPOINT = HERE.parent / "run_a_only_acquisition_v1.py"


def load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "s39_a_only_source_entry_v1_test",
        ENTRYPOINT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load source entrypoint")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SourceEntryTests(unittest.TestCase):
    def test_entrypoint_source_is_regular_ascii(self):
        entrypoint = load_entrypoint()
        metadata = ENTRYPOINT.stat(follow_symlinks=False)
        self.assertTrue(stat.S_ISREG(metadata.st_mode))
        ENTRYPOINT.read_bytes().decode("ascii")
        self.assertIs(
            entrypoint.common.canonical_bytes,
            entrypoint.canonical_bytes,
        )

    def test_stale_bytecode_cannot_replace_self_contained_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "run.py"
            source.write_bytes(ENTRYPOINT.read_bytes())
            malicious = root / "malicious.py"
            marker = root / "marker.txt"
            malicious.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('pyc', encoding='ascii')\n",
                encoding="ascii",
            )
            cache = root / "__pycache__"
            cache.mkdir()
            cache_path = cache / (
                "run." + sys.implementation.cache_tag + ".pyc"
            )
            py_compile.compile(
                str(malicious),
                cfile=str(cache_path),
                dfile=str(source),
                doraise=True,
            )
            completed = subprocess.run(
                [sys.executable, "-I", "-B", str(source), "--help"],
                capture_output=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            self.assertFalse(marker.exists())
            self.assertIn(b"--command-plan", completed.stdout)


if __name__ == "__main__":
    unittest.main()
