#!/usr/bin/env python3

import copy
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
S40 = HERE.parent
sys.path.insert(0, str(S40))

from acquire_trace import (  # noqa: E402
    load_run_start_event,
    run_trace,
    validate_status,
)
from evidence_common import EvidenceError  # noqa: E402


def request(index: int, arrival_us: int) -> dict:
    return {
        "arrival_us": arrival_us,
        "event_id": f"r{index}",
        "input_tokens": 2,
        "model_id": "qwen3-8b-q8_0",
        "output_tokens": 8,
        "prompt_tokens": [1, 2],
        "request_index": index,
        "schema": "s39-cp0d-desktop-request-v1",
        "slo_us": 30_000_000,
        "source_input_tokens": 2,
        "source_model": "fixture",
        "source_output_tokens": 8,
        "source_t_us": 0,
    }


class FakeClock:
    def __init__(self, now: int = 0) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += int(seconds * 1_000_000_000)


class FakeTransport:
    def __init__(self, *, strand: set[str] | None = None) -> None:
        self.admissions = []
        self.strand = strand or set()

    def admit(self, value):
        self.admissions.append(copy.deepcopy(value))
        return {
            "arrival_order": value["arrival_order"],
            "controller_epoch": 0,
            "request_id": value["request_id"],
            "schema": "llama-server-warm-tier-admission-v1",
            "state": "ACTIVE",
        }

    def status(self, request_id):
        terminal = "STRANDED" if request_id in self.strand else "COMPLETED"
        tokens = [] if terminal == "STRANDED" else list(range(8))
        return {
            "committed_output_tokens": tokens,
            "controller_epoch": 0,
            "model": "qwen3-8b-q8_0",
            "owner_id": "gpu",
            "ownership_epoch": 1,
            "position": 2 + len(tokens),
            "publication_index": len(tokens),
            "request_id": request_id,
            "schema": "llama-server-warm-tier-request-status-v1",
            "state": terminal,
        }

    def finalize(self, reason):
        return {
            "controller_epoch": 0,
            "reason": reason,
            "schema": "llama-server-warm-tier-finalize-result-v1",
            "state": "FINALIZED",
        }

    def finalization_status(self):
        return {
            "controller_epoch": 0,
            "schema": "llama-server-warm-tier-finalize-status-v1",
            "state": "FINALIZED",
        }


class NeverTerminal(FakeTransport):
    def status(self, request_id):
        row = super().status(request_id)
        row["committed_output_tokens"] = []
        row["position"] = 2
        row["publication_index"] = 0
        row["state"] = "ACTIVE"
        return row

    def finalize(self, reason):
        row = super().finalize(reason)
        row["state"] = "DRAINING"
        return row

    def finalization_status(self):
        row = super().finalization_status()
        row["state"] = "DRAINING"
        return row


class LateCompletion(FakeTransport):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def status(self, request_id):
        self.clock.now += 2_000_000
        return super().status(request_id)


class AcquireTraceTests(unittest.TestCase):
    def test_current_cpp_run_start_is_accepted(self):
        event_log = HERE / "fixtures" / "core_success_v3.jsonl"
        row = load_run_start_event(event_log, "test-run")
        self.assertEqual(row["schema"], "s40-warm-tier-event-v3")
        self.assertEqual(row["schema_version"], 3)
        self.assertIsNone(row["command_id"])
        self.assertIsNone(row["command_kind"])

    def test_serial_admission_then_terminal_conservation(self):
        rows = [request(0, 10), request(1, 10), request(2, 20)]
        clock = FakeClock()
        transport = FakeTransport(strand={"r2"})
        result = run_trace(
            rows,
            0,
            transport,
            campaign_horizon_ns=1_000_000_000,
            drain_bound_ns=1_000_000_000,
            clock=clock,
            sleep=clock.sleep,
            poll_workers=3,
        )
        self.assertEqual(
            [row["arrival_order"] for row in transport.admissions],
            [0, 1, 2],
        )
        self.assertEqual(result["terminal_count"], 3)
        self.assertEqual(result["completed_count"], 2)
        self.assertEqual(result["stranded_count"], 1)

    def test_first_admission_deadline_is_fail_closed(self):
        rows = [request(0, 10)]
        clock = FakeClock(10_001)
        with self.assertRaisesRegex(EvidenceError, "deadline missed"):
            run_trace(
                rows,
                0,
                FakeTransport(),
                campaign_horizon_ns=1_000_000_000,
                drain_bound_ns=1_000_000_000,
                clock=clock,
                sleep=clock.sleep,
            )

    def test_active_drain_bound_is_fail_closed(self):
        row = request(0, 0)
        clock = FakeClock()
        with self.assertRaisesRegex(EvidenceError, "active-drain bound"):
            run_trace(
                [row],
                0,
                NeverTerminal(),
                campaign_horizon_ns=1_000_000,
                drain_bound_ns=1_000_000,
                clock=clock,
                sleep=clock.sleep,
                poll_interval_ns=1_000_000,
            )

    def test_late_completion_is_measured_not_stranded(self):
        row = request(0, 0)
        row["slo_us"] = 1
        clock = FakeClock()
        result = run_trace(
            [row],
            0,
            LateCompletion(clock),
            campaign_horizon_ns=1_000_000_000,
            drain_bound_ns=1_000_000_000,
            clock=clock,
            sleep=clock.sleep,
        )
        self.assertEqual(result["completed_count"], 1)
        self.assertEqual(result["stranded_count"], 0)
        self.assertEqual(result["finalization_reason"], "TRACE_COMPLETE")

    def test_completed_status_requires_exact_eight(self):
        expected = request(0, 0)
        status = FakeTransport().status("r0")
        status["committed_output_tokens"].pop()
        status["publication_index"] = 7
        status["position"] = 9
        with self.assertRaisesRegex(EvidenceError, "wrong token count"):
            validate_status(status, expected, None)

    def test_status_history_cannot_change(self):
        expected = request(0, 0)
        previous = FakeTransport().status("r0")
        previous["state"] = "ACTIVE"
        previous["committed_output_tokens"] = [1, 2]
        previous["publication_index"] = 2
        previous["position"] = 4
        current = copy.deepcopy(previous)
        current["committed_output_tokens"] = [1, 3, 4]
        current["publication_index"] = 3
        current["position"] = 5
        with self.assertRaisesRegex(EvidenceError, "history changed"):
            validate_status(current, expected, previous)


if __name__ == "__main__":
    unittest.main()
