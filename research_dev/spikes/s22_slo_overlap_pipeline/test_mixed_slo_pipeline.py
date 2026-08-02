#!/usr/bin/env python3

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from mixed_slo_pipeline import load_profiles, load_trace, require_numeric_override


class MixedSloInputTests(unittest.TestCase):
    def write(
        self,
        value: object,
        directory: Path | None = None,
        name: str = "input.json",
    ) -> Path:
        if directory is None:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            directory = Path(temporary.name)
        path = directory / name
        path.write_text(json.dumps(value), encoding="ascii")
        return path

    def test_profiles_load(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        evidence = self.write({"evidence": 1}, directory, "evidence.json")
        row = {
            "route_id": "R0", "head_name": "cuda", "offloaded_layers": 0,
            "fixed_us": 0, "p95_step_us": 1, "profiled_batch": 1,
            "max_active": 1, "gather_cap_us": 0,
            "wait_stages_per_step": 2,
            "evidence_sha256": "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest(),
            "evidence_path": evidence.name,
            "correctness_scope": "NUMERICALLY_UNCERTIFIED",
        }
        profiles, scopes = load_profiles(self.write({
            "schema": "s22-route-profiles-v1", "routes": [row],
        }, directory))
        self.assertEqual(profiles[0].route_id, "R0")
        self.assertEqual(scopes, {"R0": "NUMERICALLY_UNCERTIFIED"})

    def test_profile_digest_mismatch_rejected(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        evidence = self.write({"evidence": 1}, directory, "evidence.json")
        row = {
            "route_id": "R0", "head_name": "cuda", "offloaded_layers": 0,
            "fixed_us": 0, "p95_step_us": 1, "profiled_batch": 1,
            "max_active": 1, "gather_cap_us": 0,
            "wait_stages_per_step": 2,
            "evidence_sha256": "sha256:" + "0" * 64,
            "evidence_path": evidence.name,
            "correctness_scope": "NUMERICALLY_UNCERTIFIED",
        }
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            load_profiles(self.write({
                "schema": "s22-route-profiles-v1", "routes": [row],
            }, directory))

    def test_trace_sorts_by_arrival_priority_and_id(self) -> None:
        rows = [
            {"request_id": 3, "arrival_us": 1, "slo_us": 2, "steps": 1, "priority": 0},
            {"request_id": 2, "arrival_us": 0, "slo_us": 2, "steps": 1, "priority": 1},
            {"request_id": 1, "arrival_us": 0, "slo_us": 2, "steps": 1, "priority": 0},
        ]
        requests = load_trace(self.write({
            "schema": "s22-mixed-slo-trace-v1", "requests": rows,
        }))
        self.assertEqual([request.request_id for request in requests], [1, 2, 3])

    def test_duplicate_request_id_rejected(self) -> None:
        row = {"request_id": 1, "arrival_us": 0, "slo_us": 2, "steps": 1, "priority": 0}
        with self.assertRaisesRegex(ValueError, "unique"):
            load_trace(self.write({
                "schema": "s22-mixed-slo-trace-v1", "requests": [row, row],
            }))

    def test_numeric_route_requires_explicit_override(self) -> None:
        scopes = {"R0": "NUMERICALLY_UNCERTIFIED", "R1": "EXACT_POINT"}
        with self.assertRaisesRegex(ValueError, "mechanics-only override"):
            require_numeric_override(scopes, ["R0"], False)
        require_numeric_override(scopes, ["R0"], True)
        require_numeric_override(scopes, ["R1"], False)


if __name__ == "__main__":
    unittest.main()
