#!/usr/bin/env python3

from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

import research_dev.scheduler as public

try:
    from .test_runtime_controller import (
        bindings,
        model,
        profile,
        request,
        snapshot,
        wait_acquired,
    )
except ImportError:
    from test_runtime_controller import (
        bindings,
        model,
        profile,
        request,
        snapshot,
        wait_acquired,
    )


def records(scheduler: public.UnifiedScheduler) -> list[dict[str, object]]:
    value = scheduler.runtime_decision_log()
    if value.get("schema") != "research-scheduler-decision-log-v1":
        raise AssertionError("decision log schema")
    return value["records"]


class RuntimeDecisionLogTests(unittest.TestCase):
    def test_every_attempt_records_all_candidates_and_selection(self) -> None:
        scheduler = public.UnifiedScheduler((profile(),), "enforce")
        ticket = scheduler.submit_runtime_request(
            request("decision"), model(), bindings(ready=False),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        decision_records = [
            row for row in records(scheduler)
            if row["event_kind"] == "DECISION"
        ]
        self.assertEqual(len(decision_records), 1)
        record = decision_records[0]
        candidates = {
            row["route_id"]: row for row in record["candidates"]
        }
        estimates = {
            row.route_id: row.to_json()
            for row in ticket.cost_estimates.estimates
        }
        self.assertEqual(set(candidates), set(estimates))
        self.assertEqual(
            candidates["phone-full"]["reason"], "EXECUTOR_NOT_READY"
        )
        self.assertEqual(record["selected"]["route_id"], ticket.decision.route_id)
        self.assertEqual(
            record["selected"]["executor"], ticket.binding.to_json()
        )
        self.assertEqual(
            record["selected"]["resource_leases"],
            [
                {
                    "lanes": list(lease.lanes),
                    "lease_id": lease.lease_id,
                    "predicted_end_us": lease.predicted_end_us,
                    "reserved_until_us": lease.reserved_until_us,
                    "resource_id": lease.resource_id,
                    "start_us": lease.start_us,
                    "token": lease.token,
                }
                for lease in ticket.decision.leases
            ],
        )
        selected_cost = next(
            row for row in ticket.cost_estimates.estimates
            if row.route_id == ticket.decision.route_id
        )
        self.assertEqual(record["selected"]["cost"], selected_cost.to_json())

    def test_fallback_records_form_one_attempt_chain(self) -> None:
        scheduler = public.UnifiedScheduler((profile(),), "enforce")
        original = request("fallback-log")
        scheduler.submit_runtime_request(
            original, model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        active = wait_acquired(scheduler, original.request_id)
        recovery = scheduler.fail_runtime_request(
            original.request_id,
            failed_at_us=active.decision.start_us + 1,
            reason="synthetic_pre_dispatch_failure",
            request=original,
            model=model(),
            bindings=bindings(),
            snapshot=snapshot(active.decision.start_us + 1),
        )
        self.assertIsNotNone(recovery.fallback)
        attempts = [
            row for row in records(scheduler)
            if row["event_kind"] in {"DECISION", "REPLAN", "FALLBACK"}
        ]
        self.assertEqual([row["attempt_index"] for row in attempts], [0, 1])
        self.assertEqual(attempts[1]["event_kind"], "FALLBACK")
        self.assertEqual(
            attempts[1]["previous_ticket_id"], attempts[0]["ticket_id"]
        )

    def test_replan_records_form_one_attempt_chain(self) -> None:
        scheduler = public.UnifiedScheduler((profile(),), "enforce")
        scheduler.submit_runtime_request(
            request("blocker"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        blocker = wait_acquired(scheduler, "blocker")
        queued_request = request("replanned", 1_001)
        scheduler.submit_runtime_request(
            queued_request, model(), bindings(),
            snapshot=snapshot(1_001), observed_at_us=1_001,
        )
        queued = scheduler.runtime_ticket("replanned")
        scheduler.extend_runtime_request(
            "blocker",
            at_us=blocker.decision.start_us,
            reserved_until_us=queued.decision.start_us + 1,
        )
        scheduler.complete_runtime_request(
            "blocker", queued.decision.start_us + 1
        )
        wake = scheduler.wait_runtime_request(
            "replanned", time.monotonic_ns()
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
        self.assertEqual(
            wake.dispatch_receipt.wake_reason,
            "predecessor_completion",
        )
        replanned = scheduler.replan_runtime_request(
            "replanned",
            queued_request,
            model(),
            bindings(),
            snapshot=snapshot(queued.decision.start_us + 1),
            observed_at_us=queued.decision.start_us + 1,
            reason=wake.dispatch_receipt.wake_reason,
        )
        attempts = [
            row for row in records(scheduler)
            if row["request_ids"] == ["replanned"]
            and row["event_kind"] in {"DECISION", "REPLAN"}
        ]
        self.assertEqual([row["event_kind"] for row in attempts], [
            "DECISION", "REPLAN"
        ])
        self.assertEqual(
            attempts[1]["previous_ticket_id"], attempts[0]["ticket_id"]
        )
        self.assertEqual(replanned.attempt_index, 1)

    def test_completion_emits_exactly_one_terminal_record(self) -> None:
        scheduler = public.UnifiedScheduler((profile(),), "enforce")
        scheduler.submit_runtime_request(
            request("terminal-log"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        active = wait_acquired(scheduler, "terminal-log")
        scheduler.complete_runtime_request(
            "terminal-log", active.decision.finish_us
        )
        terminal = [
            row for row in records(scheduler)
            if row["event_kind"] in {"COMPLETED", "FAILED", "CANCELLED"}
        ]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["event_kind"], "COMPLETED")
        with self.assertRaises(public.UnifiedScheduleError):
            scheduler.cancel_runtime_request(
                "terminal-log", active.decision.finish_us, "duplicate"
            )
        self.assertEqual(
            len([
                row for row in records(scheduler)
                if row["event_kind"]
                    in {"COMPLETED", "FAILED", "CANCELLED"}
            ]),
            1,
        )

    def test_candidate_catalog_is_recorded_once_per_attempt(self) -> None:
        scheduler = public.UnifiedScheduler((profile(),), "enforce")
        scheduler.submit_runtime_request(
            request("compact-log"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        active = wait_acquired(scheduler, "compact-log")
        scheduler.complete_runtime_request(
            "compact-log", active.decision.finish_us
        )
        rows = records(scheduler)
        self.assertGreater(len(rows[0]["candidates"]), 0)
        self.assertTrue(all(
            row["candidates"] == [] for row in rows[1:]
        ))
        self.assertTrue(all(
            row["selected"]["route_id"] == active.decision.route_id
            for row in rows
        ))

    def test_unsafe_failure_emits_one_failed_record_without_fallback(self) -> None:
        scheduler = public.UnifiedScheduler((profile(),), "enforce")
        original = request("unsafe-failure")
        scheduler.submit_runtime_request(
            original, model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        active = wait_acquired(scheduler, original.request_id)
        result = scheduler.fail_runtime_request(
            original.request_id,
            failed_at_us=active.decision.start_us + 1,
            reason="synthetic_after_dispatch_failure",
            request=original,
            model=model(),
            bindings=bindings(),
            snapshot=snapshot(active.decision.start_us + 1),
            physical_failure=public.RuntimeExecutionFailure(
                phase="response",
                retry_safe=False,
                execution_started=True,
            ),
        )
        self.assertIsNone(result.fallback)
        terminal = [
            row for row in records(scheduler)
            if row["request_ids"] == [original.request_id]
            and row["event_kind"] in {"COMPLETED", "FAILED", "CANCELLED"}
        ]
        self.assertEqual([row["event_kind"] for row in terminal], ["FAILED"])

    def test_failed_transaction_appends_nothing(self) -> None:
        scheduler = public.UnifiedScheduler((profile(),), "enforce")
        before = scheduler.runtime_decision_log_bytes()
        with self.assertRaises(public.UnifiedScheduleError):
            scheduler.submit_runtime_request(
                request("bad-transaction"),
                model(),
                (bindings()[1],),
                snapshot=snapshot(1_000),
                observed_at_us=1_000,
            )
        self.assertEqual(scheduler.runtime_decision_log_bytes(), before)

    def test_records_are_canonical_hash_chained_and_deterministic(self) -> None:
        outputs = []
        for _ in range(2):
            scheduler = public.UnifiedScheduler((profile(),), "enforce")
            scheduler.submit_runtime_request(
                request("deterministic"), model(), bindings(ready=False),
                snapshot=snapshot(1_000), observed_at_us=1_000,
            )
            outputs.append(scheduler.runtime_decision_log_bytes())
            scheduler.validate_runtime_decision_log(
                scheduler.runtime_decision_log()
            )
        self.assertEqual(outputs[0], outputs[1])

        scheduler = public.UnifiedScheduler((profile(),), "enforce")
        scheduler.submit_runtime_request(
            request("mutation"), model(), bindings(ready=False),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        altered = copy.deepcopy(scheduler.runtime_decision_log())
        altered["records"][0]["event_time_us"] += 1
        with self.assertRaises(public.DecisionLogError):
            scheduler.validate_runtime_decision_log(altered)

    def test_canonical_log_is_hash_seed_independent(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        code = "\n".join((
            "import sys",
            f"sys.path.insert(0, {str(tests_dir)!r})",
            "from test_runtime_controller import *",
            "s = UnifiedScheduler((profile(),), 'enforce')",
            "s.submit_runtime_request(request('seed'), model(), "
                "bindings(ready=False), snapshot=snapshot(1000), "
                "observed_at_us=1000)",
            "sys.stdout.buffer.write(s.runtime_decision_log_bytes())",
        ))
        outputs = []
        for seed in ("1", "9137"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = seed
            outputs.append(subprocess.check_output(
                [sys.executable, "-c", code],
                cwd=Path(__file__).resolve().parents[3],
                env=environment,
            ))
        self.assertEqual(outputs[0], outputs[1])

    def test_acquired_and_cancelled_are_scheduler_records(self) -> None:
        scheduler = public.UnifiedScheduler((profile(),), "enforce")
        ticket = scheduler.submit_runtime_request(
            request("cancel-log"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        scheduler.cancel_runtime_request(
            ticket.request.request_id,
            ticket.decision.start_us,
            "synthetic_cancel",
        )
        self.assertEqual(
            [row["event_kind"] for row in records(scheduler)],
            ["DECISION", "CANCELLED"],
        )


if __name__ == "__main__":
    unittest.main()
