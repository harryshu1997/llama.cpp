"""Runtime image identity recorded by the desktop calibration and remote-resident gate drivers."""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler.campaigns.burstgpt.desktop_parent_calibration import (  # noqa: E402
    _parse_ldd_libraries,
    _runtime_libraries_sha256,
    _sha256,
)


class RuntimeLibrariesTests(unittest.TestCase):
    def test_parser_keeps_only_real_files_inside_the_build_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bin"
            root.mkdir()
            real = root / "libllama.so.0.0.0"
            real.write_bytes(b"llama")
            os.symlink(real.name, root / "libllama.so.0")
            outside = Path(directory) / "libcuda.so.1"
            outside.write_bytes(b"cuda")
            text = "\n".join([
                "\tlinux-vdso.so.1 (0x00007fff)",
                f"\tlibllama.so.0 => {root / 'libllama.so.0'} (0x00007f00)",
                f"\tlibcuda.so.1 => {outside} (0x00007f01)",
                f"\tlibmissing.so.0 => {root / 'libmissing.so.0'} (0x00007f02)",
                "\tlibgone.so.1 => not found",
                "\t/lib64/ld-linux-x86-64.so.2 (0x00007f03)",
            ])
            found = _parse_ldd_libraries(text, root)
            self.assertEqual(found, {"libllama.so.0": real.resolve()})
            self.assertEqual(_parse_ldd_libraries(text, Path(directory) / "elsewhere"), {})

    def test_digest_record_names_real_files_and_reports_ldd_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bogus = Path(directory) / "llama-server"
            bogus.write_bytes(b"not an ELF image")
            record = _runtime_libraries_sha256(bogus)
            self.assertEqual(record["libraries"], {})
            self.assertIsNotNone(record["ldd_error"])
            self.assertEqual(
                _sha256(bogus), "sha256:" + hashlib.sha256(b"not an ELF image").hexdigest()
            )


if __name__ == "__main__":
    unittest.main()
