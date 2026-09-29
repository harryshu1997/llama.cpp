"""Native HELLO selection for a READY subset during another session's load."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


class ResidentRouterSubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        compiler = shutil.which("c++")
        if compiler is None or not sys.platform.startswith("linux"):
            raise unittest.SkipTest("native router test requires Linux and C++")
        cls.directory = tempfile.TemporaryDirectory(prefix="s42-router-subset-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.executable = Path(cls.directory.name) / "resident-router-subset"
        compiled = subprocess.run(
            [
                compiler, "-std=c++17", "-pthread", "-D__ANDROID__",
                "-I", str(ROOT),
                str(HERE / "native/resident_router_subset.cpp"),
                "-o", str(cls.executable),
            ],
            capture_output=True, text=True, timeout=60,
        )
        if compiled.returncode:
            raise AssertionError(compiled.stderr)

    def test_ready_subset_and_exact_handshake(self) -> None:
        for case in (
            "subset", "one", "missing", "partial",
            "artifact", "geometry", "width", "mask",
        ):
            with self.subTest(case=case):
                subprocess.run(
                    [str(self.executable), case],
                    check=True, capture_output=True, text=True, timeout=5,
                    cwd=self.directory.name,
                )


if __name__ == "__main__":
    unittest.main()
