#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parents[1]
V24_TEST = (
    HERE.parent
    / "v24_readiness"
    / "tests"
    / "test_v24_joint_authority.py"
)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cp0_r1_evidence_v25 as v25
import v25_common as common


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v24_tests = load_module("v25_bound_v24_joint_tests", V24_TEST)
v24 = v24_tests.evidence


class SwapPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = v24_tests.JointAuthorityFixture(
            Path(self.temporary.name)
        )
        self.evidence = copy.deepcopy(
            self.fixture.receipt["phone_evidence"]
        )
        self.minimum = self.fixture.contract["gates"][
            "phone_minimum_available_bytes"
        ]
        self.locked = {
            "op12": 8192,
            "op15": 4096,
        }
        for phone, swap in (("op12", 4096), ("op15", 2048)):
            for moment in ("before", "after"):
                self.evidence["raw_probes"][phone][moment][
                    "system_swap_used_bytes"
                ] = swap

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_v24_rejects_v25_stable_nonzero_swap(self) -> None:
        with self.assertRaises(v24.common.EvidenceError):
            v24._validate_phone_raw_probes(
                self.evidence,
                self.fixture.phone_launch,
                self.fixture.runtime,
                self.fixture.contract,
                self.fixture.started_ns,
                self.fixture.completed_ns,
            )

    def test_v25_accepts_only_after_preserving_v24_checks(self) -> None:
        validator = v25._v25_phone_probe_validator(
            v24._validate_phone_raw_probes,
            self.locked,
            self.minimum,
        )
        validator(
            self.evidence,
            self.fixture.phone_launch,
            self.fixture.runtime,
            self.fixture.contract,
            self.fixture.started_ns,
            self.fixture.completed_ns,
        )

    def test_swap_growth_fails(self) -> None:
        self.evidence["raw_probes"]["op12"]["after"][
            "system_swap_used_bytes"
        ] = 4097
        with self.assertRaises(common.EvidenceError):
            v25._validate_v25_phone_swap(
                self.evidence,
                self.locked,
                self.minimum,
            )

    def test_process_swap_fails(self) -> None:
        self.evidence["raw_probes"]["op15"]["after"][
            "process_swap_bytes"
        ] = 1
        with self.assertRaises(common.EvidenceError):
            v25._validate_v25_phone_swap(
                self.evidence,
                self.locked,
                self.minimum,
            )

    def test_headroom_fails(self) -> None:
        self.evidence["raw_probes"]["op15"]["before"][
            "available_bytes"
        ] = self.minimum - 1
        with self.assertRaises(common.EvidenceError):
            v25._validate_v25_phone_swap(
                self.evidence,
                self.locked,
                self.minimum,
            )


if __name__ == "__main__":
    unittest.main()
