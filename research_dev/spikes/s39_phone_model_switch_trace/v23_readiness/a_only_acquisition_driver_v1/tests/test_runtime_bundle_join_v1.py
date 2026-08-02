#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent / "run_a_only_acquisition_v1.py"
PRODUCTION_TESTS = (
    HERE.parent.parent / "production_v2" / "tests"
)
sys.path.insert(0, str(PRODUCTION_TESTS))

from test_runtime_bundle_regressions_v2 import (  # noqa: E402
    RuntimeBundleFixture,
    base,
    overlay,
)


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "s39_runtime_bundle_join_v1_test",
        DRIVER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load acquisition driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load_driver()


class RuntimeBundleJoinTests(unittest.TestCase):
    def fixture(self):
        fixture = RuntimeBundleFixture(self)
        args = fixture.overlay_args()
        identity, _ = base.read_canonical(args.runtime_bundle_identity)
        runtime, _ = base.read_canonical(args.base_runtime_identity)
        snapshot, snapshot_raw = base.read_canonical(
            args.runtime_bundle_snapshot
        )
        _, fresh_raw = base.read_canonical(args.runtime_bundle_fresh)
        manifest, _ = base.read_canonical(args.manifest)
        role_digests = {
            value["role"]: value["sha256"]
            for value in manifest["artifacts"]
        }
        raw_keys = driver.RAW_RUNTIME_PROCESS_KEYS
        raw_processes = [
            {
                key: copy.deepcopy(value)
                for key, value in process.items()
                if key in raw_keys
            }
            for process in identity["processes"]
        ]
        joint = {
            "runtime_processes": [
                process
                for process in raw_processes
                if process["bundle_id"] != "cuda_monolithic"
            ]
        }
        monolithic = {
            "runtime_process": next(
                process
                for process in raw_processes
                if process["bundle_id"] == "cuda_monolithic"
            )
        }
        return (
            fixture,
            args,
            runtime,
            snapshot,
            snapshot_raw,
            fresh_raw,
            role_digests,
            joint,
            monolithic,
        )

    def build(self, values):
        (
            fixture,
            args,
            runtime,
            snapshot,
            snapshot_raw,
            fresh_raw,
            role_digests,
            joint,
            monolithic,
        ) = values
        result = driver._build_runtime_bundle_identity(
            joint,
            monolithic,
            runtime,
            snapshot,
            snapshot_raw,
            fresh_raw,
            role_digests,
            fixture.phase_id,
            610,
            700,
        )
        return args, result

    def test_joined_identity_passes_independent_overlay(self):
        args, result = self.build(self.fixture())
        args.runtime_bundle_identity.write_bytes(base.canonical_bytes(result))
        verdict = overlay.validate(args)
        self.assertEqual(verdict["status"], "RUNTIME_BUNDLE_PROVENANCE_PASS")
        self.assertEqual(
            [value["bundle_id"] for value in result["processes"]],
            [
                "cuda_monolithic",
                "cuda_route",
                "op12_stagenet",
                "op15_direct_relay",
                "op15_stagenet",
            ],
        )

    def test_wrong_stagenet_pid_is_rejected(self):
        values = self.fixture()
        joint = values[-2]
        next(
            value
            for value in joint["runtime_processes"]
            if value["bundle_id"] == "op12_stagenet"
        )["pid"] += 1
        with self.assertRaisesRegex(ValueError, "worker_pid"):
            self.build(values)

    def test_wrong_launcher_path_is_rejected(self):
        values = self.fixture()
        values[-1]["runtime_process"]["launcher_path"] += ".other"
        with self.assertRaisesRegex(ValueError, "launcher_path"):
            self.build(values)

    def test_stale_observation_is_rejected(self):
        values = self.fixture()
        values[-1]["runtime_process"]["observed_ns"] = 609
        with self.assertRaisesRegex(ValueError, "E_RUNTIME_OBSERVED"):
            self.build(values)

    def test_duplicate_bundle_membership_is_rejected(self):
        values = self.fixture()
        joint = values[-2]
        next(
            value
            for value in joint["runtime_processes"]
            if value["bundle_id"] == "op12_stagenet"
        )["bundle_id"] = "cuda_route"
        with self.assertRaisesRegex(ValueError, "runtime_process_order"):
            self.build(values)


if __name__ == "__main__":
    unittest.main()
