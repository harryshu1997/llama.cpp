#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_joint_capture_plan_v1",
    ROOT / "build_joint_capture_plan_v1.py",
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
joint = builder.joint


class JointPlanBuilderTests(unittest.TestCase):
    def test_command_and_plan_are_accepted_and_expand_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            producer = root / "producer"
            launch = root / "launch.json"
            history = root / "history.json"
            producer.write_bytes(b"#!/usr/bin/python3\n")
            producer.chmod(0o755)
            launch.write_bytes(b"{}\n")
            history.write_bytes(b"{}\n")
            mechanism = {
                "desktop": [["/bin/true"]],
                "op12": [["/bin/true"]],
                "op15": [["/bin/true"]],
            }
            mechanism_sha256 = joint.sha256_bytes(
                joint.canonical_bytes(mechanism)
            )
            phone = builder.command(
                producer,
                launch,
                history,
                mechanism_sha256,
                "phone",
                root,
            )
            cuda = builder.command(
                producer,
                launch,
                history,
                mechanism_sha256,
                "cuda",
                root,
            )
            value = {
                "commands": {"cuda": cuda, "phone": phone},
                "history": {
                    "bytes": len(history.read_bytes()),
                    "path": str(history),
                    "sha256": joint.sha256_bytes(history.read_bytes()),
                },
                "mechanism_commands": mechanism,
                "model_id": joint.MODEL_ID,
                "model_sha256": "a" * 64,
                "phase": joint.PHASE,
                "schema": joint.PLAN_SCHEMA,
            }
            plan = root / "capture-plan.json"
            plan.write_bytes(joint.canonical_bytes(value))
            validated, _ = joint.load_plan(plan)
            self.assertEqual(validated, value)
            captured = {
                (name, record["argv_index"]): record["path"]
                for name in ("phone", "cuda")
                for record in value["commands"][name]["executed_files"]
            }
            argv = joint.command_argv(
                value,
                "cuda",
                root / "cuda.result.json",
                "cp0-r1-v24-a-only-test",
                root,
                100,
                "b" * 64,
                captured,
            )
            self.assertEqual(argv[0], str(producer))
            self.assertIn(str(history), argv)
            self.assertNotIn("{output_path}", argv)
            self.assertNotIn("{phase_id}", argv)


if __name__ == "__main__":
    unittest.main()
