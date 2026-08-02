#!/usr/bin/env python3

from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest


DRIVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DRIVER_DIR))

import run_a_only_acquisition_v1 as driver


class NormalizationTests(unittest.TestCase):
    def test_corpus_row_does_not_require_event_timestamp(self):
        row = {
            "acquisition_id": "phase-1",
            "kind": "item",
            "phase": "A_ONLY",
            "phase_id": "phase-1",
            "role": "quality.corpus",
        }
        self.assertEqual(
            driver._normalized(row),
            {
                "acquisition_id": "phase-1",
                "kind": "item",
                "role": "quality.corpus",
            },
        )

    def test_launch_plan_must_be_captured_by_argv_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "producer.py"
            source.write_text(
                "#!/usr/bin/env python3\nimport json\n",
                encoding="ascii",
            )
            launch = root / "launch.json"
            launch.write_bytes(b"{}\n")
            template = [
                str(source),
                "--launch-plan",
                str(launch),
                "--output",
                "{output_path}",
                "--phase-id",
                "{phase_id}",
                "--pre-dir",
                "{pre_dir}",
                "--started",
                "{acquisition_started_ns}",
                "--plan",
                "{command_plan_sha256}",
            ]
            producer = {
                "argv_template": template,
                "executed_files": [{
                    "argv_index": 0,
                    "bytes": source.stat().st_size,
                    "path": str(source),
                    "sha256": driver.sha256_bytes(source.read_bytes()),
                }],
                "result_filename": "result.json",
                "timeout_seconds": 60,
            }
            with self.assertRaisesRegex(
                driver.ReadinessError,
                "executed_file_indexes",
            ):
                driver._validate_producer(producer, "joint_phone_cuda")

            producer["executed_files"].append({
                "argv_index": 2,
                "bytes": launch.stat().st_size,
                "path": str(launch),
                "sha256": driver.sha256_bytes(launch.read_bytes()),
            })
            driver._validate_producer(producer, "joint_phone_cuda")
            plan = {
                "producers": {
                    "joint_phone_cuda": copy.deepcopy(producer),
                    "cuda_monolithic": copy.deepcopy(producer),
                },
            }
            raw_dir = root / "raw"
            raw_dir.mkdir()
            captured = driver.capture_producers(plan, raw_dir)
            argv, _ = driver._producer_argv(
                producer,
                "joint_phone_cuda",
                raw_dir,
                root,
                "phase",
                1,
                "0" * 64,
                captured,
            )
            captured_launch = Path(argv[2])
            self.assertNotEqual(captured_launch, launch)
            self.assertEqual(captured_launch.read_bytes(), launch.read_bytes())
            self.assertEqual(captured_launch.stat().st_mode & 0o777, 0o644)


if __name__ == "__main__":
    unittest.main()
