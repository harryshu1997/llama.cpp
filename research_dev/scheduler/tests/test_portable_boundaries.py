"""Portable path: the import rule between its layers, ASCII sources, and independence from the spikes."""
from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEDULER = REPO_ROOT / "research_dev" / "scheduler"
PACKAGE = "research_dev.scheduler"
DRIVERS = sorted((SCHEDULER / "drivers").glob("*.py"))
PORTABLE = sorted((SCHEDULER / "campaigns" / "portable").glob("*.py"))
NEW_LOWER = [SCHEDULER / "configuration" / "site.py", SCHEDULER / "_internal" / "shared_layer.py",
             SCHEDULER / "adapters" / "phone_functionfs_bridge_session.py"]


def imported_modules(path: Path) -> set[str]:
    """Absolute dotted names of everything a source file imports from this package."""
    package = PACKAGE + "." + ".".join(path.relative_to(SCHEDULER).with_suffix("").parts[:-1])
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="ascii"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                names.add(base + ("." + node.module if node.module else ""))
            elif node.module:
                names.add(node.module)
    return {name for name in names if name.startswith(PACKAGE)}


class PortableBoundaryTests(unittest.TestCase):
    def test_drivers_sit_below_the_campaigns_and_beside_no_scheduler_internals(self):
        self.assertTrue(DRIVERS)
        for path in DRIVERS:
            for name in imported_modules(path):
                with self.subTest(file=path.name, imports=name):
                    self.assertFalse(name.startswith(PACKAGE + ".campaigns"), "drivers must not import campaigns")
                    self.assertFalse(name.startswith(PACKAGE + "._unified"), "drivers must not import _unified")
                    self.assertNotEqual(name, PACKAGE + ".scheduler")

    def test_lower_layers_never_import_the_portable_path(self):
        lower = [path for directory in ("configuration", "_internal", "adapters", "_unified")
                 for path in sorted((SCHEDULER / directory).rglob("*.py"))]
        self.assertTrue(len(lower) > 50)
        for path in lower:
            source = path.read_text(encoding="ascii")
            if "drivers" not in source and "portable" not in source:
                continue
            for name in imported_modules(path):
                with self.subTest(file=str(path.relative_to(SCHEDULER)), imports=name):
                    self.assertFalse(name.startswith(PACKAGE + ".drivers"))
                    self.assertFalse(name.startswith(PACKAGE + ".campaigns.portable"))

    def test_portable_campaign_does_not_reach_into_the_paper_campaign(self):
        self.assertTrue(PORTABLE)
        for path in PORTABLE:
            for name in imported_modules(path):
                with self.subTest(file=path.name, imports=name):
                    self.assertFalse(name.startswith(PACKAGE + ".campaigns.burstgpt"),
                                     "the portable path shares drivers and adapters, not the paper campaign's code")

    def test_new_sources_are_ascii(self):
        for path in (*DRIVERS, *PORTABLE, *NEW_LOWER):
            with self.subTest(file=str(path.relative_to(SCHEDULER))):
                path.read_text(encoding="ascii")

    def test_portable_path_imports_without_the_spikes(self):
        code = ("import builtins, sys\n"
                "real = builtins.__import__\n"
                "def guarded(name, *args, **kwargs):\n"
                "    if name.startswith('research_dev.spikes'):\n"
                "        raise ImportError('spikes are not part of the portable path: ' + name)\n"
                "    return real(name, *args, **kwargs)\n"
                "builtins.__import__ = guarded\n"
                "import research_dev.scheduler.campaigns.portable.__main__\n"
                "import research_dev.scheduler.drivers.phone_functionfs\n"
                "assert not [name for name in sys.modules if name.startswith('research_dev.spikes')]\n")
        done = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr[-600:])


if __name__ == "__main__":
    unittest.main()
