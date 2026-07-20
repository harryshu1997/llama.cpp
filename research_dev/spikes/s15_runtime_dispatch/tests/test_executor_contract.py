#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
S15 = HERE.parent
S14 = S15.parent / "s14_mixed_streaming_scheduler"
sys.path.insert(0, str(S15))
sys.path.insert(0, str(S14))

from power_frontier_policy import BoundaryCertificate  # noqa: E402
from executor_contract import (  # noqa: E402
    BOUNDARY_SCHEMA,
    ExecutionRequest,
    ExecutionResult,
    ExecutorError,
    RecordedExecutor,
    RecordedOutcome,
    recorded_key,
)

MANIFEST = "sha256:" + "ef" * 32
COHORT = "sha256:" + "de" * 32


def request(request_ids=("r0",), launch_id=1, timeout_us=1000,
            profile_id="sha256:" + "ab" * 32, route_epoch=1,
            residency_epoch=1, lease_epoch=1, device_boot_epoch=1,
            registry_generation=1) -> ExecutionRequest:
    return ExecutionRequest(
        launch_id=launch_id,
        route_id="op15-gemma-head-0-8",
        profile_id=profile_id,
        device_id="op15:3C15AU002CL00000",
        route_epoch=route_epoch,
        residency_epoch=residency_epoch,
        lease_epoch=lease_epoch,
        device_boot_epoch=device_boot_epoch,
        registry_generation=registry_generation,
        compatibility_key="gemma|decode|gemma-head-0-8",
        request_ids=tuple(request_ids),
        cohort_sha256=COHORT,
        input_manifest_sha256=MANIFEST,
        timeout_us=timeout_us,
        expected_boundary_schema=BOUNDARY_SCHEMA,
    )


def outcome(outcome_name="completed", delay=100, admit=None,
            profile_id="sha256:" + "ab" * 32) -> RecordedOutcome:
    return RecordedOutcome(
        outcome=outcome_name,
        finish_delay_us=delay,
        profile_id=profile_id,
        route_epoch=1,
        residency_epoch=1,
        device_boot_epoch=1,
        cohort_sha256=COHORT,
        input_manifest_sha256=MANIFEST,
        admit=admit or {},
    )


class ResultValidationTests(unittest.TestCase):
    def test_completed_needs_one_certificate_per_request(self) -> None:
        req = request(("r0", "r1"))
        missing = ExecutionResult(1, "completed", 100, BOUNDARY_SCHEMA,
                                  (BoundaryCertificate("r0", True, True, True, True),))
        with self.assertRaisesRegex(ExecutorError, "one certificate per request"):
            missing.validate(req)

    def test_completed_wrong_schema_is_rejected(self) -> None:
        req = request()
        bad = ExecutionResult(1, "completed", 100, "other-schema",
                              (BoundaryCertificate("r0", True, True, True, True),))
        with self.assertRaisesRegex(ExecutorError, "wrong boundary schema"):
            bad.validate(req)

    def test_non_completed_carries_no_certificate(self) -> None:
        req = request()
        bad = ExecutionResult(1, "timed_out", 100, BOUNDARY_SCHEMA,
                              (BoundaryCertificate("r0", True, True, True, True),))
        with self.assertRaisesRegex(ExecutorError, "no certificate"):
            bad.validate(req)

    def test_launch_id_must_match(self) -> None:
        req = request(launch_id=5)
        bad = ExecutionResult(6, "timed_out", 100, BOUNDARY_SCHEMA, ())
        with self.assertRaisesRegex(ExecutorError, "launch id"):
            bad.validate(req)


class RecordedExecutorTests(unittest.TestCase):
    def test_unknown_launch_times_out_and_is_not_fabricated(self) -> None:
        executor = RecordedExecutor({})
        result = executor.launch(request(), now_us=10)
        self.assertEqual(result.outcome, "timed_out")
        self.assertEqual(result.certificates, ())
        self.assertEqual(result.finish_us, 10 + 1000)

    def test_recorded_completion_returns_one_certificate_each(self) -> None:
        req = request(("r0", "r1"))
        script = {recorded_key(req.route_id, req.request_ids): outcome()}
        executor = RecordedExecutor(script)
        result = executor.launch(req, now_us=0)
        self.assertEqual(result.outcome, "completed")
        self.assertEqual(sorted(c.request_id for c in result.certificates), ["r0", "r1"])
        self.assertTrue(all(c.admitted() for c in result.certificates))

    def test_per_request_admit_false_produces_a_failing_boundary(self) -> None:
        req = request(("r0", "r1"))
        script = {recorded_key(req.route_id, req.request_ids): outcome(admit={"r1": False})}
        executor = RecordedExecutor(script)
        result = executor.launch(req, now_us=0)
        by_id = {c.request_id: c.admitted() for c in result.certificates}
        self.assertTrue(by_id["r0"])
        self.assertFalse(by_id["r1"])

    def test_identity_mismatch_is_rejected(self) -> None:
        req = request(profile_id="sha256:" + "cd" * 32)
        script = {recorded_key(req.route_id, req.request_ids): outcome(profile_id="sha256:" + "ab" * 32)}
        executor = RecordedExecutor(script)
        with self.assertRaisesRegex(ExecutorError, "identity does not match"):
            executor.launch(req, now_us=0)

    def test_lease_epoch_and_registry_generation_are_bound(self) -> None:
        for req in (request(lease_epoch=2), request(registry_generation=2)):
            script = {recorded_key(req.route_id, req.request_ids): outcome()}
            executor = RecordedExecutor(script)
            with self.assertRaisesRegex(ExecutorError, "identity does not match"):
                executor.launch(req, now_us=0)

    def test_recorded_input_manifest_identity_is_bound(self) -> None:
        req = request()
        wrong = outcome()
        object.__setattr__(wrong, "input_manifest_sha256", "sha256:" + "01" * 32)
        executor = RecordedExecutor({recorded_key(req.route_id, req.request_ids): wrong})
        with self.assertRaisesRegex(ExecutorError, "identity does not match"):
            executor.launch(req, now_us=0)

    def test_recorded_error_carries_no_certificate(self) -> None:
        req = request()
        script = {recorded_key(req.route_id, req.request_ids): outcome("error")}
        executor = RecordedExecutor(script)
        result = executor.launch(req, now_us=0)
        self.assertEqual((result.outcome, result.certificates), ("error", ()))


class NoDeviceIOTests(unittest.TestCase):
    def test_module_has_no_device_transport(self) -> None:
        source = (S15 / "executor_contract.py").read_text(encoding="ascii").lower()
        for banned in ("import subprocess", "from subprocess", "import socket",
                       "from socket", "os.system", ".popen(", "pty.spawn"):
            self.assertNotIn(banned, source)


if __name__ == "__main__":
    unittest.main()
