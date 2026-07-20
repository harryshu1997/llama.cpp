#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
S15 = HERE.parent
S14 = S15.parent / "s14_mixed_streaming_scheduler"
sys.path.insert(0, str(S15))
sys.path.insert(0, str(S14))

from executor_contract import BOUNDARY_SCHEMA, ExecutionRequest  # noqa: E402
import route_fixtures as F  # noqa: E402
from physical_executor import (  # noqa: E402
    PhysicalExecutor,
    PhysicalExecutorError,
    PhysicalRouteBinding,
    SESSION_SCHEMA,
    SessionTransport,
    SubprocessSessionTransport,
    TransportReply,
)
from power_frontier_policy import WorkItem  # noqa: E402
from route_registry import ReadyRouteRegistry  # noqa: E402
from runtime_dispatch import LaneBinding, MixedDispatchCoordinator  # noqa: E402


PROFILE = "sha256:" + "ab" * 32
BINARY = "sha256:" + "cd" * 32
MANIFEST = "sha256:" + "ef" * 32
COHORT = "sha256:" + "de" * 32
ROUTE = "op12-gemma-head-0-6"
DEVICE = "op12:5ae7a43d"


def request(launch_id=1, request_ids=("r0",), lease_epoch=7,
            registry_generation=9) -> ExecutionRequest:
    return ExecutionRequest(
        launch_id=launch_id,
        route_id=ROUTE,
        profile_id=PROFILE,
        device_id=DEVICE,
        route_epoch=11,
        residency_epoch=5,
        lease_epoch=lease_epoch,
        device_boot_epoch=3,
        registry_generation=registry_generation,
        compatibility_key="gemma|decode|gemma-head-0-6",
        request_ids=tuple(request_ids),
        cohort_sha256=COHORT,
        input_manifest_sha256=MANIFEST,
        timeout_us=1_000_000,
        expected_boundary_schema=BOUNDARY_SCHEMA,
    )


def binding(profile_id=PROFILE) -> PhysicalRouteBinding:
    return PhysicalRouteBinding(
        route_id=ROUTE,
        profile_id=profile_id,
        device_id=DEVICE,
        protocol_version=1,
        worker_binary_sha256=BINARY,
        worker_generation=4,
        first_session_id=1,
        device_boot_id="boot-op12",
        cohort_sha256=COHORT,
        input_manifest_sha256=MANIFEST,
        layer_range=(0, 6),
        expected_backend="HTP0",
    )


def record(req=None, session_id=1, outcome="completed") -> dict:
    req = req or request()
    result = {
        "schema": SESSION_SCHEMA,
        "protocol_version": 1,
        "launch_id": req.launch_id,
        "route_id": req.route_id,
        "profile_id": req.profile_id,
        "device_id": req.device_id,
        "route_epoch": req.route_epoch,
        "residency_epoch": req.residency_epoch,
        "lease_epoch": req.lease_epoch,
        "device_boot_epoch": req.device_boot_epoch,
        "registry_generation": req.registry_generation,
        "compatibility_key": req.compatibility_key,
        "request_ids": list(req.request_ids),
        "cohort_sha256": req.cohort_sha256,
        "input_manifest_sha256": req.input_manifest_sha256,
        "worker_binary_sha256": BINARY,
        "worker_generation": 4,
        "device_boot_id": "boot-op12",
        "layer_range": [0, 6],
        "session_id": session_id,
        "outcome": outcome,
        "boundary_schema": BOUNDARY_SCHEMA,
        "placement": {
            "status": "SCHEDULED_PLACEMENT_OK",
            "layer_start": 0,
            "layer_end": 6,
            "missing_buffer_compute_nodes": 0,
            "compute_by_op_and_buffer": {
                "MUL_MAT": {"HTP0": 100},
                "GET_ROWS": {"CPU": 1},
            },
        },
        "boundaries": [
            {
                "request_id": request_id,
                "identity_ok": True,
                "epoch_ok": True,
                "correctness_ok": True,
                "d2h_complete": True,
            }
            for request_id in req.request_ids
        ],
    }
    if outcome == "error":
        result["placement"] = None
        result["boundaries"] = []
    return result


def encoded(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


class FakeTransport(SessionTransport):
    def __init__(self, replies):
        self.replies = list(replies)
        self.payloads = []

    def exchange(self, req, payload):
        self.payloads.append(payload)
        return self.replies.pop(0)


class EchoTransport(SessionTransport):
    def exchange(self, req, payload):
        return TransportReply("reply", 10, encoded(record(req)))


class PhysicalExecutorTests(unittest.TestCase):
    def test_completed_record_returns_exact_certificates(self) -> None:
        req = request(request_ids=("r0", "r1"))
        transport = FakeTransport([TransportReply("reply", 123, encoded(record(req)))])
        result = PhysicalExecutor([binding()], transport).launch(req, 1000)
        self.assertEqual((result.outcome, result.finish_us), ("completed", 1123))
        self.assertEqual([cert.request_id for cert in result.certificates], ["r0", "r1"])
        sent = json.loads(transport.payloads[0])
        self.assertEqual((sent["lease_epoch"], sent["registry_generation"]), (7, 9))
        self.assertEqual(sent["input_manifest_sha256"], MANIFEST)

    def test_lease_and_registry_identity_are_load_bearing(self) -> None:
        req = request()
        for key, value in (("lease_epoch", 8), ("registry_generation", 10)):
            bad = record(req)
            bad[key] = value
            executor = PhysicalExecutor(
                [binding()], FakeTransport([TransportReply("reply", 1, encoded(bad))])
            )
            with self.assertRaisesRegex(PhysicalExecutorError, f"identity mismatch: {key}"):
                executor.launch(req, 0)

    def test_stale_session_id_is_rejected(self) -> None:
        req1 = request(launch_id=1)
        req2 = request(launch_id=2)
        transport = FakeTransport([
            TransportReply("reply", 1, encoded(record(req1, session_id=1))),
            TransportReply("reply", 1, encoded(record(req2, session_id=1))),
        ])
        executor = PhysicalExecutor([binding()], transport)
        executor.launch(req1, 0)
        with self.assertRaisesRegex(PhysicalExecutorError, "stale, duplicated, or has a gap"):
            executor.launch(req2, 1)

    def test_late_reply_is_a_timeout_without_admitting_its_record(self) -> None:
        bad = record()
        bad["route_id"] = "would-fail-if-parsed"
        executor = PhysicalExecutor(
            [binding()], FakeTransport([TransportReply("reply", 1_000_001, encoded(bad))])
        )
        result = executor.launch(request(), 5)
        self.assertEqual((result.outcome, result.finish_us), ("timed_out", 1_000_005))

    def test_undeclared_cpu_compute_is_rejected(self) -> None:
        bad = record()
        bad["placement"]["compute_by_op_and_buffer"]["MUL_MAT"] = {"CPU": 5}
        executor = PhysicalExecutor(
            [binding()], FakeTransport([TransportReply("reply", 1, encoded(bad))])
        )
        with self.assertRaisesRegex(PhysicalExecutorError, "undeclared backend fallback"):
            executor.launch(request(), 0)

    def test_wrong_boundary_ownership_is_rejected(self) -> None:
        bad = record()
        bad["boundaries"][0]["request_id"] = "foreign"
        executor = PhysicalExecutor(
            [binding()], FakeTransport([TransportReply("reply", 1, encoded(bad))])
        )
        with self.assertRaisesRegex(PhysicalExecutorError, "one certificate per request"):
            executor.launch(request(), 0)

    def test_duplicate_json_key_is_rejected(self) -> None:
        payload = encoded(record()).replace(b'{"boundaries"', b'{"schema":"duplicate","boundaries"', 1)
        executor = PhysicalExecutor([binding()], FakeTransport([TransportReply("reply", 1, payload)]))
        with self.assertRaisesRegex(PhysicalExecutorError, "duplicate JSON key 'schema'"):
            executor.launch(request(), 0)

    def test_timeout_and_transport_error_never_complete(self) -> None:
        for state, expected in (("timed_out", "timed_out"), ("error", "error")):
            executor = PhysicalExecutor([binding()], FakeTransport([TransportReply(state, 10)]))
            result = executor.launch(request(), 5)
            self.assertEqual((result.outcome, result.certificates), (expected, ()))

    def test_error_record_carries_no_completion_evidence(self) -> None:
        executor = PhysicalExecutor(
            [binding()], FakeTransport([TransportReply("reply", 10, encoded(record(outcome="error")))])
        )
        self.assertEqual(executor.launch(request(), 0).outcome, "error")
        bad = record(outcome="error")
        bad["boundaries"] = record()["boundaries"]
        executor = PhysicalExecutor(
            [binding()], FakeTransport([TransportReply("reply", 10, encoded(bad))])
        )
        with self.assertRaisesRegex(PhysicalExecutorError, "carries completion evidence"):
            executor.launch(request(), 0)

    def test_unknown_or_mismatched_binding_fails_closed(self) -> None:
        unknown = copy.copy(request())
        object.__setattr__(unknown, "route_id", "missing")
        result = PhysicalExecutor([binding()], FakeTransport([])).launch(unknown, 0)
        self.assertEqual(result.outcome, "error")
        wrong = request()
        object.__setattr__(wrong, "profile_id", "sha256:" + "ef" * 32)
        with self.assertRaisesRegex(PhysicalExecutorError, "binding does not match"):
            PhysicalExecutor([binding()], FakeTransport([])).launch(wrong, 0)
        wrong_manifest = request()
        object.__setattr__(wrong_manifest, "input_manifest_sha256", "sha256:" + "01" * 32)
        with self.assertRaisesRegex(PhysicalExecutorError, "input manifest"):
            PhysicalExecutor([binding()], FakeTransport([])).launch(wrong_manifest, 0)

    def test_subprocess_transport_persists_exact_bytes(self) -> None:
        req = request()
        reply_bytes = encoded(record(req))
        command = (
            sys.executable,
            "-c",
            f"import sys; sys.stdin.buffer.read(); sys.stdout.buffer.write({reply_bytes!r})",
        )
        with tempfile.TemporaryDirectory() as tmp:
            transport = SubprocessSessionTransport({ROUTE: command}, Path(tmp))
            result = PhysicalExecutor([binding()], transport).launch(req, 0)
            self.assertEqual(result.outcome, "completed")
            runs = list(Path(tmp).iterdir())
            self.assertEqual(len(runs), 1)
            self.assertTrue((runs[0] / "request.json").is_file())
            self.assertEqual((runs[0] / "stdout.bin").read_bytes(), reply_bytes)
            metadata = json.loads((runs[0] / "transport.json").read_bytes())
            self.assertEqual(metadata["state"], "reply")

    def test_measured_op12_route_completes_through_coordinator(self) -> None:
        snapshot = F.op12_head_snapshot()
        registry = ReadyRouteRegistry()
        registry.install([snapshot])
        executor = PhysicalExecutor([binding(snapshot.profile_id)], EchoTransport())
        lane = LaneBinding("op12", ROUTE, "phone", timeout_us=4_000_000,
                           cohort_sha256=COHORT, input_manifest_sha256=MANIFEST)
        coordinator = MixedDispatchCoordinator(registry, executor, (lane,), queue_capacity=4)
        item = WorkItem(
            "r0", "generation", "gemma-4-12b-it-f16", "gemma-head-0-6",
            "gemma-4-12b-it-f16|decode|gemma-head-0-6", 0, 10_000_000, 1,
        )
        coordinator.admit(item, 0)
        coordinator.dispatch(ROUTE, 0)
        self.assertEqual(coordinator.terminal_of("r0"), "completed_phone")
        self.assertEqual(registry.outstanding(ROUTE), 0)


if __name__ == "__main__":
    unittest.main()
