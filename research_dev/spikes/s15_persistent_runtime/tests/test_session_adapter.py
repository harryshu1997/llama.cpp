#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
S15 = SPIKE.parent / "s15_runtime_dispatch"
S14 = SPIKE.parent / "s14_mixed_streaming_scheduler"
sys.path[:0] = [str(SPIKE), str(S15), str(S14)]

from executor_contract import BOUNDARY_SCHEMA, ExecutionRequest, ExecutorError  # noqa: E402
from physical_executor import (  # noqa: E402
    PhysicalExecutor,
    PhysicalRouteBinding,
    SessionTransport,
    TransportReply,
)
from session_adapter import (  # noqa: E402
    PersistentWorkerCapability,
    SessionAdapterError,
    StageNetSessionAdapter,
    StageNetSessionBinding,
)


PROFILE = "sha256:" + "ab" * 32
BINARY = "sha256:" + "cd" * 32
COHORT = "sha256:" + "de" * 32
MANIFEST = "sha256:" + "ef" * 32
BOOT = "boot-op15"


def request(launch_id: int = 1, request_ids=("r0",)) -> ExecutionRequest:
    return ExecutionRequest(
        launch_id=launch_id,
        route_id="op15-head-0-8",
        profile_id=PROFILE,
        device_id="op15:serial",
        route_epoch=12,
        residency_epoch=1,
        lease_epoch=launch_id,
        device_boot_epoch=1,
        registry_generation=1,
        compatibility_key="gemma|decode|head-0-8",
        request_ids=tuple(request_ids),
        cohort_sha256=COHORT,
        input_manifest_sha256=MANIFEST,
        timeout_us=1_000_000,
        expected_boundary_schema=BOUNDARY_SCHEMA,
    )


def physical_binding(first_session_id: int = 1) -> PhysicalRouteBinding:
    return PhysicalRouteBinding(
        route_id="op15-head-0-8",
        profile_id=PROFILE,
        device_id="op15:serial",
        protocol_version=1,
        worker_binary_sha256=BINARY,
        worker_generation=2,
        first_session_id=first_session_id,
        device_boot_id=BOOT,
        cohort_sha256=COHORT,
        input_manifest_sha256=MANIFEST,
        layer_range=(0, 8),
        expected_backend="HTP0",
    )


def session_binding() -> StageNetSessionBinding:
    return StageNetSessionBinding(BINARY, 1, BOOT, (0, 8), 48)


def capability(detach_supported: bool = True) -> PersistentWorkerCapability:
    return PersistentWorkerCapability(BINARY, 2, detach_supported)


def adapter(capabilities=None) -> StageNetSessionAdapter:
    if capabilities is None:
        capabilities = (capability(),)
    return StageNetSessionAdapter(session_binding(), tuple(capabilities))


def cert(session_id: int = 1, session_end: str = "DETACH",
         steps_session: int = 32, steps_total: int = 32) -> dict:
    return {
        "schema": "ls-stagenet-session-v2",
        "proto_version": 2,
        "session_id": session_id,
        "session_end": session_end,
        "expected_backend": "HTP0",
        "worker_pid": 1234,
        "worker_boot_nonce": "0123456789abcdef",
        "device_boot_id": BOOT,
        "layer_start": 0,
        "layer_end": 8,
        "n_layer": 48,
        "steps_session": steps_session,
        "steps_total": steps_total,
        "reset_applied": session_end == "DETACH",
        "missing_buffer_compute_nodes": 0,
        "compute_by_op_and_buffer": {
            "MUL_MAT": {"HTP0": 100},
            "GET_ROWS": {"CPU": 1},
        },
        "placement_status": "SCHEDULED_PLACEMENT_OK",
    }


def encoded(value: dict) -> bytes:
    return (
        "SESSIONCERT "
        + json.dumps(value, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def boundaries(req: ExecutionRequest) -> list[dict]:
    return [
        {
            "request_id": request_id,
            "identity_ok": True,
            "epoch_ok": True,
            "correctness_ok": True,
            "d2h_complete": True,
        }
        for request_id in req.request_ids
    ]


class FakeTransport(SessionTransport):
    def __init__(self, payload: bytes):
        self.payload = payload

    def exchange(self, req, request_payload):
        return TransportReply("reply", 10, self.payload)


class SessionAdapterTests(unittest.TestCase):
    def test_adapter_failures_are_typed_executor_failures(self) -> None:
        self.assertTrue(issubclass(SessionAdapterError, ExecutorError))

    def test_two_detach_sessions_are_contiguous_and_reusable(self) -> None:
        value = adapter()
        req1 = request(1)
        self.assertEqual(value.begin("DETACH"), 1)
        first = value.accept(encoded(cert()), 0, False, req1, physical_binding(), boundaries(req1))
        self.assertTrue(value.reuse_permitted)
        value.release_after_detach()
        req2 = request(2, ("r1",))
        self.assertEqual(value.begin("DETACH"), 2)
        second_cert = cert(2, steps_session=4, steps_total=36)
        second = value.accept(
            encoded(second_cert), 0, False, req2, physical_binding(), boundaries(req2),
        )
        self.assertEqual((json.loads(first)["session_id"], json.loads(second)["session_id"]), (1, 2))

    def test_adapted_record_passes_unchanged_physical_executor(self) -> None:
        req = request(1, ("r0", "r1"))
        value = adapter()
        value.begin("DETACH")
        payload = value.accept(encoded(cert()), 0, False, req, physical_binding(), boundaries(req))
        result = PhysicalExecutor(
            [physical_binding()], FakeTransport(payload),
        ).launch(req, 100)
        self.assertEqual((result.outcome, result.finish_us), ("completed", 110))
        self.assertEqual([item.request_id for item in result.certificates], ["r0", "r1"])

    def test_duplicate_stale_and_gapped_session_ids_poison(self) -> None:
        for bad_id in (0, 2, 7):
            value = adapter()
            value.begin("DETACH")
            with self.subTest(session_id=bad_id), self.assertRaisesRegex(
                    SessionAdapterError, "session_id"):
                value.accept(
                    encoded(cert(bad_id)), 0, False, request(), physical_binding(), boundaries(request()),
                )
            self.assertEqual(value.state, "POISONED")

    def test_changed_pid_nonce_or_boot_is_rejected(self) -> None:
        for field, changed in (
            ("worker_pid", 4321),
            ("worker_boot_nonce", "fedcba9876543210"),
            ("device_boot_id", "other-boot"),
        ):
            value = adapter()
            req1 = request(1)
            value.begin("DETACH")
            value.accept(encoded(cert()), 0, False, req1, physical_binding(), boundaries(req1))
            value.release_after_detach()
            req2 = request(2, ("r1",))
            bad = cert(2, steps_session=1, steps_total=33)
            bad[field] = changed
            value.begin("DETACH")
            with self.subTest(field=field), self.assertRaises(SessionAdapterError):
                value.accept(encoded(bad), 0, False, req2, physical_binding(), boundaries(req2))
            self.assertEqual(value.state, "POISONED")

    def test_wrong_range_backend_missing_buffer_and_reset_are_rejected(self) -> None:
        mutations = (
            ("range", lambda item: item.__setitem__("layer_end", 9)),
            ("backend", lambda item: item.__setitem__("expected_backend", "GPUOpenCL")),
            ("missing", lambda item: item.__setitem__("missing_buffer_compute_nodes", 1)),
            ("reset", lambda item: item.__setitem__("reset_applied", False)),
            ("fallback", lambda item: item["compute_by_op_and_buffer"].update(
                {"MUL_MAT": {"CPU": 100}}
            )),
        )
        for label, mutate in mutations:
            value = adapter()
            bad = cert()
            mutate(bad)
            value.begin("DETACH")
            with self.subTest(label=label), self.assertRaises(SessionAdapterError):
                value.accept(encoded(bad), 0, False, request(), physical_binding(), boundaries(request()))
            self.assertEqual(value.state, "POISONED")

    def test_missing_cert_ack_or_error_eof_never_permits_reuse(self) -> None:
        cases = []
        cases.append((None, 0))
        cases.append((encoded(cert()), None))
        cases.append((encoded(cert()), 1))
        for ending in ("ERROR", "EOF", "STOP"):
            cases.append((encoded(cert(session_end=ending)), 0))
        for payload, ack in cases:
            value = adapter()
            value.begin("DETACH")
            with self.subTest(payload=payload, ack=ack), self.assertRaises(SessionAdapterError):
                value.accept(payload, ack, False, request(), physical_binding(), boundaries(request()))
            self.assertFalse(value.reuse_permitted)
            self.assertEqual(value.state, "POISONED")

    def test_worker_without_explicit_capability_cannot_detach(self) -> None:
        foreign = PersistentWorkerCapability("sha256:" + "01" * 32, 2, True)
        for capabilities in ((), (capability(False),), (foreign,)):
            value = adapter(capabilities)
            with self.assertRaisesRegex(SessionAdapterError, "no explicit"):
                value.begin("DETACH")
            self.assertEqual(value.state, "IDLE")

    def test_stop_allows_reset_false_and_terminates_adapter(self) -> None:
        value = adapter(())
        req = request()
        value.begin("STOP")
        stop = cert(session_end="STOP")
        stop["reset_applied"] = False
        payload = value.accept(encoded(stop), None, True, req, physical_binding(), boundaries(req))
        self.assertEqual(json.loads(payload)["session_id"], 1)
        self.assertEqual(value.state, "STOPPED")
        self.assertFalse(value.reuse_permitted)
        with self.assertRaisesRegex(SessionAdapterError, "state STOPPED"):
            value.begin("STOP")

        value = adapter(())
        value.begin("STOP")
        with self.assertRaisesRegex(SessionAdapterError, "did not terminate"):
            value.accept(encoded(stop), None, False, req, physical_binding(), boundaries(req))
        self.assertEqual(value.state, "POISONED")

        value = adapter()
        value.begin("DETACH")
        with self.assertRaisesRegex(SessionAdapterError, "unexpectedly terminated"):
            value.accept(encoded(cert()), 0, True, req, physical_binding(), boundaries(req))
        self.assertEqual(value.state, "POISONED")

    def test_detach_requires_explicit_release_before_next_session(self) -> None:
        value = adapter()
        req = request()
        value.begin("DETACH")
        value.accept(encoded(cert()), 0, False, req, physical_binding(), boundaries(req))
        with self.assertRaisesRegex(SessionAdapterError, "state DETACHED"):
            value.begin("DETACH")
        value.release_after_detach()
        self.assertEqual(value.begin("DETACH"), 2)

    def test_duplicate_cert_after_detach_poisons(self) -> None:
        value = adapter()
        req = request()
        value.begin("DETACH")
        value.accept(encoded(cert()), 0, False, req, physical_binding(), boundaries(req))
        with self.assertRaisesRegex(SessionAdapterError, "outside an active session"):
            value.accept(encoded(cert()), 0, False, req, physical_binding(), boundaries(req))
        self.assertEqual(value.state, "POISONED")

    def test_boundary_failure_prevents_reuse(self) -> None:
        value = adapter()
        req = request()
        bad = boundaries(req)
        bad[0]["d2h_complete"] = False
        value.begin("DETACH")
        with self.assertRaisesRegex(SessionAdapterError, "did not pass"):
            value.accept(encoded(cert()), 0, False, req, physical_binding(), bad)
        self.assertFalse(value.reuse_permitted)

    def test_unknown_fields_and_duplicate_keys_are_rejected(self) -> None:
        value = adapter()
        bad = cert()
        bad["unknown"] = 1
        value.begin("DETACH")
        with self.assertRaisesRegex(SessionAdapterError, "unknown fields"):
            value.accept(encoded(bad), 0, False, request(), physical_binding(), boundaries(request()))
        duplicate = encoded(cert()).replace(b'{"schema"', b'{"schema":"duplicate","schema"', 1)
        value = adapter()
        value.begin("DETACH")
        with self.assertRaisesRegex(SessionAdapterError, "duplicate"):
            value.accept(duplicate, 0, False, request(), physical_binding(), boundaries(request()))

    def test_wrong_physical_binding_poisoned_before_record(self) -> None:
        value = adapter()
        req = request()
        wrong = copy.copy(physical_binding())
        object.__setattr__(wrong, "device_boot_id", "other")
        value.begin("DETACH")
        with self.assertRaisesRegex(SessionAdapterError, "differs"):
            value.accept(encoded(cert()), 0, False, req, wrong, boundaries(req))
        self.assertEqual(value.state, "POISONED")


if __name__ == "__main__":
    unittest.main()
