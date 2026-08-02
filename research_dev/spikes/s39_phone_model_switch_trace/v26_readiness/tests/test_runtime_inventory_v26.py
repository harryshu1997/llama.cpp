#!/usr/bin/env python3

from pathlib import Path
import tempfile
import unittest
import importlib.util


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inventory", HERE / "runtime_inventory_v26.py")
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "bundle"
        self.root.mkdir()
        self.file = self.root / "runtime.bin"
        self.file.write_bytes(b"runtime-v26")
        self.file.chmod(0o755)

    def tearDown(self):
        self.temp.cleanup()

    def test_secure_pin_reopens_same_inode(self):
        value = inventory.secure_local_pin(self.file, executable=True)
        self.assertEqual(value["bytes"], len(b"runtime-v26"))
        self.assertEqual(value["sha256"], inventory.sha256(b"runtime-v26"))

    def test_symlinked_ancestor_is_rejected(self):
        target = Path(self.temp.name) / "target"
        target.mkdir()
        link = Path(self.temp.name) / "link"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(inventory.InventoryError, "E_LOCAL_ANCESTOR"):
            inventory._secure_directory(link)

    def test_extra_file_does_not_change_secure_pin(self):
        extra = self.root / "extra"
        extra.write_bytes(b"unbound")
        first = inventory.secure_local_pin(self.file, executable=True)
        extra.write_bytes(b"changed")
        second = inventory.secure_local_pin(self.file, executable=True)
        self.assertEqual(first, second)

    def test_replaced_file_is_not_accepted_as_original(self):
        first = inventory.secure_local_pin(self.file, executable=True)
        replacement = self.root / "replacement"
        replacement.write_bytes(b"replacement")
        replacement.chmod(0o755)
        self.file.unlink()
        replacement.rename(self.file)
        second = inventory.secure_local_pin(self.file, executable=True)
        self.assertNotEqual(first["sha256"], second["sha256"])


if __name__ == "__main__":
    unittest.main()
