#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
S15 = SPIKE.parent / "s15_runtime_dispatch"
S14 = SPIKE.parent / "s14_mixed_streaming_scheduler"
sys.path[:0] = [str(SPIKE), str(S15), str(S14)]

from executor_contract import BOUNDARY_SCHEMA, ExecutionRequest, ExecutorError  # noqa: E402
from persistent_transport import (  # noqa: E402
    PersistentPreparedTransport,
    PersistentTransportError,
)
from physical_executor import PhysicalExecutor, PhysicalRouteBinding  # noqa: E402


DIGEST = "sha256:" + "ab" * 32


def request(launch_id: int, timeout_us: int = 1_000_000) -> ExecutionRequest:
    return ExecutionRequest(
        launch_id=launch_id,
        route_id="route",
        profile_id=DIGEST,
        device_id="phone",
        route_epoch=1,
        residency_epoch=1,
        lease_epoch=1,
        device_boot_epoch=1,
        registry_generation=1,
        compatibility_key="model|decode|island",
        request_ids=(f"r{launch_id}",),
        cohort_sha256=DIGEST,
        input_manifest_sha256=DIGEST,
        timeout_us=timeout_us,
        expected_boundary_schema=BOUNDARY_SCHEMA,
    )


def payload(req: ExecutionRequest) -> bytes:
    return (
        json.dumps({"launch_id": req.launch_id}, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def binding() -> PhysicalRouteBinding:
    return PhysicalRouteBinding(
        route_id="route",
        profile_id=DIGEST,
        device_id="phone",
        protocol_version=1,
        worker_binary_sha256=DIGEST,
        worker_generation=1,
        first_session_id=1,
        device_boot_id="boot",
        cohort_sha256=DIGEST,
        input_manifest_sha256=DIGEST,
        layer_range=(0, 8),
        expected_backend="HTP0",
    )


class TransportFixture:
    def __init__(self, mode: str):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "artifacts"
        self.transport = PersistentPreparedTransport(
            (sys.executable, str(HERE / "persistent_launcher_fixture.py"), mode),
            self.root,
            ready_timeout_s=2,
        )

    def close(self) -> None:
        if self.transport.state not in ("FINALIZED",):
            self.transport.terminate()
            self.transport.finalize()
        self.tmp.cleanup()


class PersistentTransportTests(unittest.TestCase):
    def test_transport_failures_are_typed_executor_failures(self) -> None:
        self.assertTrue(issubclass(PersistentTransportError, ExecutorError))

    def test_two_exchanges_use_one_process_and_distinct_frames(self) -> None:
        fixture = TransportFixture("normal")
        try:
            replies = []
            for launch_id in (1, 2):
                req = request(launch_id)
                reply = fixture.transport.exchange(req, payload(req))
                self.assertEqual(reply.state, "reply")
                replies.append(json.loads(reply.payload))
            self.assertEqual(replies, [
                {"launch_id": 1, "sequence": 1},
                {"launch_id": 2, "sequence": 2},
            ])
            self.assertEqual(fixture.transport.state, "READY")
            fixture.transport.drain()
            self.assertEqual(fixture.transport.state, "DRAINING")
            fixture.transport.finalize()
            self.assertEqual(fixture.transport.state, "FINALIZED")
            self.assertTrue((fixture.root / "exchange-000001-launch-1/request.json").is_file())
            self.assertTrue((fixture.root / "exchange-000002-launch-2/reply.json").is_file())
        finally:
            fixture.close()

    def test_duplicate_reply_poisons_and_cannot_be_reused(self) -> None:
        fixture = TransportFixture("duplicate")
        try:
            req = request(1)
            reply = fixture.transport.exchange(req, payload(req))
            self.assertEqual(reply.state, "error")
            self.assertEqual(fixture.transport.state, "POISONED")
            with self.assertRaisesRegex(PersistentTransportError, "state POISONED"):
                fixture.transport.exchange(request(2), payload(request(2)))
        finally:
            fixture.close()

    def test_timeout_after_partial_reply_poisons(self) -> None:
        fixture = TransportFixture("partial")
        try:
            req = request(1, timeout_us=100_000)
            reply = fixture.transport.exchange(req, payload(req))
            self.assertEqual(reply.state, "timed_out")
            self.assertIn("timed out", fixture.transport.poison_reason)
            raw = fixture.root / "exchange-000001-launch-1/stdout.bin"
            self.assertEqual(raw.read_bytes(), b'{"partial":')
            with self.assertRaises(PersistentTransportError):
                fixture.transport.exchange(request(2), payload(request(2)))
        finally:
            fixture.close()

    def test_late_duplicate_cannot_be_reused_by_next_request(self) -> None:
        fixture = TransportFixture("late_duplicate")
        try:
            first = request(1)
            self.assertEqual(fixture.transport.exchange(first, payload(first)).state, "reply")
            second = request(2)
            reply = fixture.transport.exchange(second, payload(second))
            self.assertEqual(reply.state, "error")
            self.assertTrue(
                "launch_id" in fixture.transport.poison_reason
                or "cross-talk" in fixture.transport.poison_reason
            )
            self.assertEqual(fixture.transport.state, "POISONED")
        finally:
            fixture.close()

    def test_missing_reply_fails_closed(self) -> None:
        fixture = TransportFixture("missing")
        try:
            req = request(1, timeout_us=100_000)
            self.assertEqual(fixture.transport.exchange(req, payload(req)).state, "timed_out")
            self.assertEqual(fixture.transport.state, "POISONED")
        finally:
            fixture.close()

    def test_unsolicited_reply_is_request_reply_crosstalk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transport = None
            try:
                transport = PersistentPreparedTransport(
                    (sys.executable, str(HERE / "persistent_launcher_fixture.py"), "crosstalk"),
                    Path(tmp) / "artifacts",
                    ready_timeout_s=2,
                )
            except PersistentTransportError:
                return
            try:
                req = request(1)
                try:
                    reply = transport.exchange(req, payload(req))
                except PersistentTransportError:
                    pass
                else:
                    self.assertEqual(reply.state, "error")
            finally:
                transport.terminate()
                transport.finalize()

    def test_finalize_rejects_active_exchange(self) -> None:
        fixture = TransportFixture("slow")
        result = []
        req = request(1)
        thread = threading.Thread(
            target=lambda: result.append(fixture.transport.exchange(req, payload(req)))
        )
        try:
            thread.start()
            deadline = time.monotonic() + 1
            while fixture.transport.state != "ACTIVE" and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(fixture.transport.state, "ACTIVE")
            with self.assertRaisesRegex(PersistentTransportError, "active exchange"):
                fixture.transport.finalize()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result[0].state, "reply")
        finally:
            fixture.close()

    def test_process_stop_blocks_another_exchange(self) -> None:
        fixture = TransportFixture("stop")
        try:
            req = request(1)
            self.assertEqual(fixture.transport.exchange(req, payload(req)).state, "reply")
            deadline = time.monotonic() + 1
            while fixture.transport.state == "READY" and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(fixture.transport.state, "STOPPED")
            with self.assertRaisesRegex(PersistentTransportError, "state STOPPED"):
                fixture.transport.exchange(request(2), payload(request(2)))
        finally:
            fixture.close()

    def test_nonzero_drain_fails_after_persisting_process_artifacts(self) -> None:
        fixture = TransportFixture("nonzero")
        try:
            fixture.transport.drain()
            with self.assertRaisesRegex(PersistentTransportError, "exited nonzero"):
                fixture.transport.finalize()
            self.assertEqual(fixture.transport.state, "FINALIZED")
            metadata = json.loads((fixture.root / "process.json").read_text(encoding="ascii"))
            self.assertEqual(metadata["returncode"], 3)
            self.assertEqual(metadata["poison_reason"], "launcher exited nonzero")
        finally:
            fixture.close()

    def test_stderr_is_closed_by_marker_and_attributed_once(self) -> None:
        fixture = TransportFixture("diagnostic")
        try:
            for launch_id in (1, 2):
                req = request(launch_id)
                self.assertEqual(fixture.transport.exchange(req, payload(req)).state, "reply")
                path = fixture.root / f"exchange-{launch_id:06d}-launch-{launch_id}/stderr.bin"
                self.assertEqual(path.read_bytes(), f"diagnostic-{launch_id}\n".encode("ascii"))
        finally:
            fixture.close()

    def test_missing_wrong_duplicate_or_trailing_marker_poisons(self) -> None:
        for mode in ("missing_marker", "wrong_marker", "duplicate_marker", "trailing_stderr"):
            fixture = TransportFixture(mode)
            try:
                req = request(1, timeout_us=100_000)
                reply = fixture.transport.exchange(req, payload(req))
                with self.subTest(mode=mode):
                    self.assertNotEqual(reply.state, "reply")
                    self.assertEqual(fixture.transport.state, "POISONED")
            finally:
                fixture.close()

    def test_launch_id_must_be_contiguous(self) -> None:
        fixture = TransportFixture("normal")
        try:
            first = request(1)
            self.assertEqual(fixture.transport.exchange(first, payload(first)).state, "reply")
            with self.assertRaisesRegex(PersistentTransportError, "not contiguous"):
                fixture.transport.exchange(request(3), payload(request(3)))
            second = request(2)
            self.assertEqual(fixture.transport.exchange(second, payload(second)).state, "reply")
        finally:
            fixture.close()

    def test_noncanonical_request_is_rejected_before_write(self) -> None:
        fixture = TransportFixture("normal")
        try:
            with self.assertRaisesRegex(PersistentTransportError, "not canonical"):
                fixture.transport.exchange(request(1), b'{"launch_id": 1}\n')
            self.assertEqual(fixture.transport.state, "READY")
        finally:
            fixture.close()

    def test_persistent_adapter_records_pass_physical_executor_twice(self) -> None:
        fixture = TransportFixture("physical_adapter")
        try:
            executor = PhysicalExecutor([binding()], fixture.transport)
            first = executor.launch(request(1), 100)
            second = executor.launch(request(2), first.finish_us)
            self.assertEqual((first.outcome, second.outcome), ("completed", "completed"))
            self.assertEqual(
                [item.request_id for item in first.certificates + second.certificates],
                ["r1", "r2"],
            )
            fixture.transport.drain()
            fixture.transport.finalize()
        finally:
            fixture.close()


if __name__ == "__main__":
    unittest.main()
