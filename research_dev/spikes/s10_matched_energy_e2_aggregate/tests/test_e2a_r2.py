#!/usr/bin/env python3
"""Regressions for the E2A v2 semantic-chain repair."""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import aggregate  # noqa: E402
import e2a_canon as canon  # noqa: E402
import resolver  # noqa: E402
from test_e2a import _rechain, bundle  # noqa: E402

FIXTURES = ROOT / "fixtures"


def load_triplet(slot=0):
    return (
        canon.load_strict(FIXTURES / "manifest.json"),
        canon.load_strict(
            FIXTURES / f"outcomes/outcomes.slot{slot:02d}.json"),
        canon.load_strict(FIXTURES / f"timelines/tl.slot{slot:02d}.json"),
    )


def resolved_slot(root, slot=0):
    manifest = canon.load_strict(root / "manifest.json")
    outcomes = canon.load_strict(
        root / f"outcomes/outcomes.slot{slot:02d}.json")
    timeline = canon.load_strict(root / f"timelines/tl.slot{slot:02d}.json")
    resolved = resolver.resolve_requests(
        manifest, outcomes, timeline, root, f"slot{slot}", slot // 2)
    return manifest, timeline, resolved


class NonVacuousWorkGate(unittest.TestCase):

    def test_zero_prefix_threshold_is_refused(self):
        manifest = canon.load_strict(FIXTURES / "manifest.json")
        plan = canon.load_strict(FIXTURES / "plan.json")
        entry = manifest["entries"][0]
        entry["min_prefix_tokens"] = 0
        entry["entry_sha256"] = canon.digest(
            {key: value for key, value in entry.items()
             if key != "entry_sha256"})
        canon.seal(manifest)
        plan["request_set_manifest_sha256"] = manifest["record_sha256"]
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_manifest(manifest, plan)
        self.assertEqual(ctx.exception.code, "E_WORK_GATE")

    def test_unbounded_length_tolerance_is_refused(self):
        manifest = canon.load_strict(FIXTURES / "manifest.json")
        plan = canon.load_strict(FIXTURES / "plan.json")
        entry = manifest["entries"][0]
        entry["length_tolerance_tokens"] = \
            resolver.MAX_LENGTH_TOLERANCE_TOKENS + 1
        entry["entry_sha256"] = canon.digest(
            {key: value for key, value in entry.items()
             if key != "entry_sha256"})
        canon.seal(manifest)
        plan["request_set_manifest_sha256"] = manifest["record_sha256"]
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_manifest(manifest, plan)
        self.assertEqual(ctx.exception.code, "E_WORK_GATE")

    def test_raw_outcome_records_are_not_same_work_evidence(self):
        manifest, control, _timeline = load_triplet(0)
        _manifest, treatment, _timeline = load_triplet(1)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.check_same_work(control, treatment, manifest)
        self.assertEqual(ctx.exception.code, "E_UNRESOLVED_WORK")

    def test_forged_resolved_wrapper_is_not_same_work_evidence(self):
        manifest, control, _timeline = load_triplet(0)
        forged = {"record": control, "token_ids": {}, "timing": {}}
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.check_same_work(forged, forged, manifest)
        self.assertEqual(ctx.exception.code, "E_UNRESOLVED_WORK")

    def test_identical_all_tardy_runs_are_refused(self):
        for slot in (0, 1):
            with self.subTest(slot=slot):
                manifest, outcomes, timeline = load_triplet(slot)
                for entry in manifest["entries"]:
                    entry["slo_deadline_us"] = 1
                    entry["entry_sha256"] = canon.digest(
                        {key: value for key, value in entry.items()
                         if key != "entry_sha256"})
                canon.seal(manifest)
                outcomes["manifest_sha256"] = manifest["record_sha256"]
                for item, entry in zip(outcomes["outcomes"],
                                       manifest["entries"]):
                    item["manifest_entry_sha256"] = entry["entry_sha256"]
                    item["terminal_outcome"] = "tardy"
                canon.seal(outcomes)
                timeline["outcomes"] = {
                    "met": 0, "tardy": len(outcomes["outcomes"]),
                    "rejected": 0, "canceled": 0,
                }
                with self.assertRaises(resolver.ResolveError) as ctx:
                    resolver.resolve_requests(
                        manifest, outcomes, timeline, FIXTURES, "all-tardy",
                        slot // 2)
                self.assertEqual(ctx.exception.code, "E_SLO_COHORT")


class LifecycleCausality(unittest.TestCase):

    def test_resolved_requests_are_mandatory(self):
        timeline = canon.load_strict(FIXTURES / "timelines/tl.slot00.json")
        lifecycle = canon.load_strict(FIXTURES / "lifecycle/lc.slot00.json")
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(lifecycle, timeline)
        self.assertEqual(ctx.exception.code, "E_UNRESOLVED_WORK")

    def test_exec_must_cover_output_completion(self):
        manifest, timeline, resolved = resolved_slot(FIXTURES, 0)
        lifecycle = canon.load_strict(FIXTURES / "lifecycle/lc.slot00.json")
        last = max(value["last_token_us"]
                   for value in resolved.timing.values())
        action = next(item for item in lifecycle["actions"]
                      if item["action_kind"] == "EXEC")
        action["end_us"] = last - 2
        action["ack_us"] = last - 2
        lifecycle["entry_state_digest"] = resolver.lifecycle_state_digest(
            lifecycle["actions"], timeline["window_start_us"])
        lifecycle["exit_state_digest"] = resolver.lifecycle_state_digest(
            lifecycle["actions"], timeline["window_end_us"])
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(lifecycle, timeline, None, resolved)
        self.assertEqual(ctx.exception.code, "E_LIFECYCLE_CAUSAL")

    def test_release_cannot_precede_acquire_ack(self):
        _manifest, timeline, resolved = resolved_slot(FIXTURES, 0)
        lifecycle = canon.load_strict(FIXTURES / "lifecycle/lc.slot00.json")
        release = next(item for item in lifecycle["actions"]
                       if item["action_kind"] == "LEASE_RELEASE")
        start = timeline["window_start_us"]
        release.update({"enqueue_us": start + 500, "start_us": start + 500,
                        "end_us": start + 600, "ack_us": start + 600})
        lifecycle["entry_state_digest"] = resolver.lifecycle_state_digest(
            lifecycle["actions"], start)
        lifecycle["exit_state_digest"] = resolver.lifecycle_state_digest(
            lifecycle["actions"], timeline["window_end_us"])
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(lifecycle, timeline, None, resolved)
        self.assertEqual(ctx.exception.code, "E_LEASE_ORDER")

    def test_lease_must_cover_execution_and_cleanup(self):
        _manifest, timeline, resolved = resolved_slot(FIXTURES, 0)
        lifecycle = canon.load_strict(FIXTURES / "lifecycle/lc.slot00.json")
        release = next(item for item in lifecycle["actions"]
                       if item["action_kind"] == "LEASE_RELEASE")
        release.update({"enqueue_us": 25_950_000, "start_us": 25_950_000,
                        "end_us": 25_960_000, "ack_us": 25_960_000})
        lifecycle["entry_state_digest"] = resolver.lifecycle_state_digest(
            lifecycle["actions"], timeline["window_start_us"])
        lifecycle["exit_state_digest"] = resolver.lifecycle_state_digest(
            lifecycle["actions"], timeline["window_end_us"])
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(lifecycle, timeline, None, resolved)
        self.assertEqual(ctx.exception.code, "E_LEASE_COVERAGE")

    def test_nonclean_entry_state_is_refused(self):
        _manifest, timeline, resolved = resolved_slot(FIXTURES, 0)
        lifecycle = canon.load_strict(FIXTURES / "lifecycle/lc.slot00.json")
        prefetch = next(item for item in lifecycle["actions"]
                        if item["action_kind"] == "PREFETCH")
        prefetch["enqueue_us"] = timeline["window_start_us"] - 1
        lifecycle["entry_state_digest"] = resolver.lifecycle_state_digest(
            lifecycle["actions"], timeline["window_start_us"])
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(lifecycle, timeline, None, resolved)
        self.assertEqual(ctx.exception.code, "E_LIFECYCLE_STATE")

    def test_queue_submit_is_bound_to_arrival(self):
        _manifest, timeline, resolved = resolved_slot(FIXTURES, 0)
        lifecycle = canon.load_strict(FIXTURES / "lifecycle/lc.slot00.json")
        submit = next(item for item in lifecycle["actions"]
                      if item["action_id"] == "a.submit.0")
        submit["enqueue_us"] += 1
        submit["start_us"] += 1
        submit["end_us"] += 1
        submit["ack_us"] += 1
        canon.seal(lifecycle)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_lifecycle(lifecycle, timeline, None, resolved)
        self.assertEqual(ctx.exception.code, "E_ARRIVAL_BINDING")


class LedgerTimelineJoin(unittest.TestCase):

    def test_start_event_time_cannot_drift_from_window(self):
        plan = canon.load_strict(FIXTURES / "plan.json")
        ledger = canon.load_strict(FIXTURES / "ledger.json")
        start = next(entry for entry in ledger["entries"]
                     if entry["slot_index"] == 0 and
                     entry["entry_kind"] == "SLOT_ATTEMPT_START")
        start["monotonic_us"] += 1
        _rechain(ledger)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_ledger(ledger, plan)
        self.assertEqual(ctx.exception.code, "E_LEDGER_TIME_BINDING")

    def test_end_event_clock_cannot_drift_from_timeline(self):
        plan = canon.load_strict(FIXTURES / "plan.json")
        ledger = canon.load_strict(FIXTURES / "ledger.json")
        end = next(entry for entry in ledger["entries"]
                   if entry["slot_index"] == 0 and
                   entry["entry_kind"] == "SLOT_ATTEMPT_END")
        end["clock_epoch_id"] = "boot-other"
        _rechain(ledger)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_ledger(ledger, plan)
        self.assertEqual(ctx.exception.code, "E_CLOCK_EPOCH")

    def test_end_event_drain_must_match_lifecycle(self):
        with bundle() as work:
            ledger = work.load("ledger.json")
            end = next(entry for entry in ledger["entries"]
                       if entry["slot_index"] == 0 and
                       entry["entry_kind"] == "SLOT_ATTEMPT_END")
            end["drain_acknowledged_us"] += 1
            _rechain(ledger)
            work.write("ledger.json", ledger)
            work.reindex()
            with self.assertRaises(aggregate.AggregateError) as ctx:
                work.evaluate()
            self.assertEqual(ctx.exception.code, "E_LEDGER_TIME_BINDING")


class ArrivalAndArtifactBinding(unittest.TestCase):

    def test_overlapping_paid_windows_are_refused(self):
        first = canon.load_strict(FIXTURES / "timelines/tl.slot00.json")
        second = canon.load_strict(FIXTURES / "timelines/tl.slot01.json")
        resolved = {0: (first, None, None), 1: (second, None, None)}
        aggregate._check_global_slots(resolved)
        second["window_start_us"] = first["window_end_us"] - 1
        with self.assertRaises(aggregate.AggregateError) as ctx:
            aggregate._check_global_slots(resolved)
        self.assertEqual(ctx.exception.code, "E_WINDOW_OVERLAP")

    def test_reused_timeline_evidence_digest_is_refused(self):
        first = canon.load_strict(FIXTURES / "timelines/tl.slot00.json")
        second = canon.load_strict(FIXTURES / "timelines/tl.slot01.json")
        resolved = {0: (first, None, None), 1: (second, None, None)}
        aggregate._check_global_slots(resolved)
        second["raw_artifact_sha256"] = first["raw_artifact_sha256"]
        with self.assertRaises(aggregate.AggregateError) as ctx:
            aggregate._check_global_slots(resolved)
        self.assertEqual(ctx.exception.code, "E_EVIDENCE_REUSE")

    def test_slo_is_arrival_to_last_token_not_dispatch_to_last_token(self):
        manifest, outcomes, timeline = load_triplet(0)
        entry = manifest["entries"][0]
        entry["slo_deadline_us"] = 550_000
        entry["entry_sha256"] = canon.digest(
            {key: value for key, value in entry.items()
             if key != "entry_sha256"})
        canon.seal(manifest)
        outcomes["manifest_sha256"] = manifest["record_sha256"]
        outcomes["outcomes"][0]["manifest_entry_sha256"] = \
            entry["entry_sha256"]
        canon.seal(outcomes)
        with self.assertRaises(resolver.ResolveError) as ctx:
            resolver.resolve_requests(
                manifest, outcomes, timeline, FIXTURES, "arrival-slo", 0)
        self.assertEqual(ctx.exception.code, "E_SLO_OUTCOME")

    def test_one_path_cannot_carry_two_declared_digests(self):
        path = "manifest.json"
        actual = canon.sha256_bytes((FIXTURES / path).read_bytes())
        with resolver.resolution_context():
            resolver.read_once(path, actual, FIXTURES)
            with self.assertRaises(resolver.ResolveError) as ctx:
                resolver.read_once(path, "0" * 64, FIXTURES)
        self.assertEqual(ctx.exception.code, "E_PATH_DIGEST_CONFLICT")

    def test_output_path_cannot_be_reused_across_runs(self):
        with bundle() as work:
            control = work.load("outcomes/outcomes.slot00.json")
            treatment = work.load("outcomes/outcomes.slot01.json")
            treatment["outcomes"][0]["output_artifact_path"] = \
                control["outcomes"][0]["output_artifact_path"]
            treatment["outcomes"][0]["output_artifact_sha256"] = \
                control["outcomes"][0]["output_artifact_sha256"]
            outcome_digest = work.write(
                "outcomes/outcomes.slot01.json", treatment)

            ledger = work.load("ledger.json")
            end = next(entry for entry in ledger["entries"]
                       if entry["slot_index"] == 1 and
                       entry["entry_kind"] == "SLOT_ATTEMPT_END")
            end["request_outcome_record_sha256"] = \
                treatment["record_sha256"]
            _rechain(ledger)
            work.write("ledger.json", ledger)
            work.reindex()
            self.assertEqual(outcome_digest,
                             work.load("bundle.json")["slots"][1]
                             ["outcomes_sha256"])
            with self.assertRaises(resolver.ResolveError) as ctx:
                work.evaluate()
            self.assertEqual(ctx.exception.code, "E_EVIDENCE_PATH_REUSE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
