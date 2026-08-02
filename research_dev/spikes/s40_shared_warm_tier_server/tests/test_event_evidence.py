#!/usr/bin/env python3

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
S40 = HERE.parent
sys.path.insert(0, str(S40))
sys.path.insert(0, str(HERE))

from evidence_common import EvidenceError  # noqa: E402
from evidence_common import digest_file, read_jsonl  # noqa: E402
from event_evidence import (  # noqa: E402
    reduce_events,
    token_history_digest,
    validate_resources,
)
from fake_ledger import (  # noqa: E402
    Ledger,
    executor_failure,
    expected_request,
    precommit_rollback,
    resources,
    snapshot,
    successful_transition,
)


def resequence(rows):
    for index, row in enumerate(rows):
        row["sequence"] = index


class EventEvidenceTests(unittest.TestCase):
    def test_cpp_history_digest_known_vector(self):
        self.assertEqual(
            token_history_digest([1, 2, 3], [100, 101]),
            "b2c2611b132cd66f6ecc086dbea88824e5f1435f5aa2a8f026cf2a65e3285694",
        )

    def test_historical_core_fixture_is_rejected(self):
        fixture = HERE / "fixtures" / "core_transition.jsonl"
        self.assertEqual(
            digest_file(fixture),
            "17fc92c25e5bb36e8bb16173a1a371e61648595e281c9624fed4dd3982188634",
        )
        expected = [{
            "arrival_us": 0,
            "event_id": "r0",
            "input_tokens": 2,
            "model_id": "model-b",
            "output_tokens": 1,
            "prompt_tokens": [1, 2],
            "request_index": 0,
            "schema": "s39-cp0d-desktop-request-v1",
            "slo_us": 30_000_000,
            "source_input_tokens": 2,
            "source_model": "fixture",
            "source_output_tokens": 1,
            "source_t_us": 0,
        }]
        with self.assertRaisesRegex(
                EvidenceError,
                "runtime_config_sha256|key set mismatch|history digest mismatch"
                "|commit-complete|cleanup"):
            reduce_events(read_jsonl(fixture, "core_fixture"), expected)

    def test_historical_v2_core_fixture_is_rejected(self):
        fixture = HERE / "fixtures" / "core_success_v2.jsonl"
        self.assertEqual(
            digest_file(fixture),
            "f470dd6d2f6887c6c8e59ef9f1bae4ccef3d75efbaac19e84680a453e9eb2826",
        )
        with self.assertRaisesRegex(
                EvidenceError, "key set mismatch|schema mismatch"):
            reduce_events(
                read_jsonl(fixture, "historical_core_success"),
                [expected_request()],
            )

    def test_current_core_success_fixture_reduces(self):
        fixture = HERE / "fixtures" / "core_success_v3.jsonl"
        self.assertEqual(
            digest_file(fixture),
            "3c5a598d0c512b146c5686aee81ca031198f35659f2f220343c12a1ab5440a2e",
        )
        expected = [expected_request(
            request_id="r0",
            model_id="model-b",
            arrival_us=0,
        )]
        expected[0]["output_tokens"] = 2
        result = reduce_events(
            read_jsonl(fixture, "core_success"),
            expected,
            expected_runtime_config_sha256="b" * 64,
        )
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["completed_request_count"], 1)
        self.assertEqual(result["phase_duration_ns"]["cleanup"]["count"], 2)

    def test_current_core_failed_precommit_fixture_fails_closed(self):
        fixture = HERE / "fixtures" / "core_failed_precommit_v3.jsonl"
        self.assertEqual(
            digest_file(fixture),
            "aa72e15e0a2af251a7333742604f0e0eeb5ef01057502c39929aaa9142a22279",
        )
        expected = [expected_request(
            request_id="r0",
            model_id="model-b",
            arrival_us=0,
        )]
        expected[0]["output_tokens"] = 2
        events = read_jsonl(fixture, "core_failed_precommit")
        self.assertFalse(any(
            event["kind"] in {"ownership_commit", "run_end"}
            for event in events
        ))
        request_events = [
            event for event in events if event["request"] is not None
        ]
        self.assertEqual(request_events[-1]["request"]["owner_id"], "phone")
        with self.assertRaisesRegex(
                EvidenceError, "missing run bounds|invalid transition"):
            reduce_events(
                events,
                expected,
                expected_runtime_config_sha256="b" * 64,
            )

    def test_historical_v2_failed_precommit_fixture_is_rejected(self):
        fixture = HERE / "fixtures" / "core_failed_precommit_v2.jsonl"
        self.assertEqual(
            digest_file(fixture),
            "c06df9a88755b2e58de6bdcd709b6c850793957b6781e01b513d3ec6ac4bc118",
        )
        events = read_jsonl(fixture, "core_failed_precommit")
        with self.assertRaisesRegex(
                EvidenceError, "key set mismatch|schema mismatch"):
            reduce_events(
                events,
                [expected_request()],
                expected_runtime_config_sha256="b" * 64,
            )

    def test_current_cpp_binary_event_dump_reduces(self):
        binary = S40.parents[2] / "build-cuda" / "bin" \
            / "test-server-warm-tier"
        self.assertTrue(binary.is_file())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "events.jsonl"
            subprocess.run(
                [str(binary), "--dump-events", str(output)],
                check=True,
                timeout=30,
            )
            events = read_jsonl(output, "cpp_event_dump")
        self.assertEqual(events[0]["schema"], "s40-warm-tier-event-v3")
        self.assertEqual(events[0]["schema_version"], 3)
        self.assertIsNone(events[0]["command_id"])
        dispatch = next(
            event for event in events
            if event["kind"] == "request_dispatched")
        self.assertEqual(
            (dispatch["command_id"], dispatch["command_kind"]),
            (1, 0),
        )
        expected = [expected_request(
            request_id="r0",
            model_id="model-b",
            arrival_us=0,
        )]
        expected[0]["output_tokens"] = 2
        self.assertEqual(
            reduce_events(
                events,
                expected,
                expected_runtime_config_sha256="b" * 64,
            )["verdict"],
            "PASS",
        )

    def test_real_shaped_transition_reduces(self):
        events, expected = successful_transition()
        result = reduce_events(events, expected)
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["completed_request_count"], 1)
        self.assertEqual(result["stranded_request_count"], 0)
        self.assertEqual(result["model_completed_counts"], {"B": 1})
        self.assertEqual(result["ownership_commit_count"], 1)
        self.assertGreater(result["maximum_model_publication_gap_ns"], 0)
        self.assertEqual(result["phase_duration_ns"]["replay"]["count"], 1)
        self.assertEqual(result["phase_duration_ns"]["cleanup"]["count"], 1)
        self.assertGreaterEqual(
            result["maximum_global_token_publication_gap_ns"],
            result["maximum_intra_request_token_publication_gap_ns"],
        )

    def test_physical_reduction_uses_bound_trace_origin(self):
        events, expected = successful_transition()
        expected[0]["arrival_us"] = 4_999
        events[-1]["detail"] = "TRACE_COMPLETE"
        trace_start = {
            "active_drain_deadline_ns": 1_020_001_000,
            "campaign_horizon_ns": 900_001_000,
            "campaign_horizon_us": 900_000,
            "created_ns": 0,
            "drain_bound_us": 120_000,
            "event_log_run_start_ns": 0,
            "experiment_contract_sha256": "a" * 64,
            "host_boot_id": "boot-fixture",
            "requests_sha256": "c" * 64,
            "run_id": "fake-run",
            "runtime_config_sha256": "0" * 64,
            "schema": "s40-trace-start-v1",
            "trace_origin_ns": 1_000,
            "trace_start_lead_us": 1,
        }
        result = reduce_events(events, expected, trace_start=trace_start)
        self.assertEqual(result["controller_run_start_ns"], 0)
        self.assertEqual(result["trace_origin_ns"], 1_000)
        self.assertEqual(
            result["run_duration_ns"],
            events[-1]["t_ns"] - trace_start["trace_origin_ns"],
        )

    def test_trace_start_arithmetic_is_load_bearing(self):
        events, expected = successful_transition()
        events[-1]["detail"] = "TRACE_COMPLETE"
        trace_start = {
            "active_drain_deadline_ns": 1_020_001_001,
            "campaign_horizon_ns": 900_001_001,
            "campaign_horizon_us": 900_000,
            "created_ns": 0,
            "drain_bound_us": 120_000,
            "event_log_run_start_ns": 0,
            "experiment_contract_sha256": "a" * 64,
            "host_boot_id": "boot-fixture",
            "requests_sha256": "c" * 64,
            "run_id": "fake-run",
            "runtime_config_sha256": "0" * 64,
            "schema": "s40-trace-start-v1",
            "trace_origin_ns": 1_001,
            "trace_start_lead_us": 1,
        }
        with self.assertRaisesRegex(EvidenceError, "lead arithmetic"):
            reduce_events(events, expected, trace_start=trace_start)

    def test_runtime_config_digest_is_load_bearing(self):
        events, expected = successful_transition()
        events[-1]["runtime_config_sha256"] = "1" * 64
        with self.assertRaisesRegex(
                EvidenceError, "runtime config digest changed"):
            reduce_events(events, expected)
        events[-1]["runtime_config_sha256"] = "0" * 64
        with self.assertRaisesRegex(
                EvidenceError, "runtime config digest mismatch"):
            reduce_events(
                events,
                expected,
                expected_runtime_config_sha256="2" * 64,
            )

    def test_frozen_arrival_order_is_load_bearing(self):
        ledger = Ledger("arrival-order")
        ledger.add("run_start")
        expected = []
        for index in range(2):
            row = expected_request(
                request_id=f"r{index}",
                model_id="B",
                arrival_us=0,
            )
            row["request_index"] = index
            expected.append(row)
        for index in (1, 0):
            request_id = f"r{index}"
            queued = snapshot(
                request_id, "B", [1, 2], [], None, 0, "QUEUED")
            ledger.add(
                "request_arrived",
                model_id="B",
                request_id=request_id,
                request=queued,
            )
            terminal = snapshot(
                request_id, "B", [1, 2], [], None, 0, "STRANDED")
            ledger.add(
                "request_stranded",
                model_id="B",
                request_id=request_id,
                request=terminal,
                success=False,
            )
        ledger.add("run_end")
        with self.assertRaisesRegex(
                EvidenceError, "frozen arrival order mismatch"):
            reduce_events(ledger.rows, expected)

    def test_selected_gpu_energy_is_bracketed(self):
        events, expected = successful_transition()
        sample_rows = resources(events[0]["run_id"], events[-1]["t_ns"])
        result = reduce_events(events, expected, sample_rows)
        expected_energy = 100_000 * events[-1]["t_ns"] // 1_000
        self.assertEqual(result["energy"]["gpu_energy_nj"], expected_energy)
        self.assertEqual(
            result["energy"]["gpu_energy_scope"],
            "SELECTED_GPU_BOARD_DEVELOPMENT_ONLY",
        )
        self.assertFalse(result["energy"]["energy_claim_authorized"])
        self.assertEqual(result["energy"]["phone_energy"], "UNKNOWN")
        self.assertEqual(result["energy"]["total_system_energy"], "UNKNOWN")

    def test_energy_units_are_mw_times_ns_to_nj(self):
        rows = resources("energy-units", 500_000_000)
        final = copy.deepcopy(rows[-1])
        final["sequence"] = 2
        final["t_ns"] = 1_000_000_000
        rows.append(final)
        result = validate_resources(
            rows,
            "energy-units",
            0,
            1_000_000_000,
        )
        self.assertEqual(result["gpu_energy_nj"], 100_000_000_000)

    def test_zero_based_token_publication_is_required(self):
        events, expected = successful_transition()
        token = next(row for row in events if row["kind"] == "token_committed")
        token["publication_index"] = token["request"]["publication_index"]
        with self.assertRaisesRegex(EvidenceError, "duplicate or skipped"):
            reduce_events(events, expected)

    def test_raw_executor_publication_must_match_committed_token(self):
        events, expected = successful_transition()
        end = next(
            row for row in events
            if row["kind"] == "execute_end"
            and row["result_publications"]
        )
        end["result_publications"][0]["token"] += 1
        with self.assertRaisesRegex(
                EvidenceError, "committed/raw publication mismatch"):
            reduce_events(events, expected)

    def test_quarantined_executor_result_cannot_commit_output(self):
        events, expected = successful_transition()
        end = next(
            row for row in events
            if row["kind"] == "execute_end"
            and row["result_publications"]
        )
        end["command_disposition"] = "QUARANTINED"
        with self.assertRaisesRegex(
                EvidenceError,
                "publication is not from successful EXECUTE"
                "|quarantined result was committed"):
            reduce_events(events, expected)

    def test_raw_completion_marker_must_match_controller_completion(self):
        events, expected = successful_transition()
        completion = next(
            row for row in events if row["kind"] == "request_completed")
        end = next(
            row for row in events
            if row["command_id"] == completion["command_id"]
            and row["kind"] == "execute_end"
        )
        end["result_request_complete"] = False
        with self.assertRaisesRegex(
                EvidenceError, "completion missing from raw result"):
            reduce_events(events, expected)

    def test_position_is_derived_from_exact_history(self):
        events, expected = successful_transition()
        token = next(row for row in events if row["kind"] == "token_committed")
        token["request"]["position"] += 1
        with self.assertRaisesRegex(EvidenceError, "position mismatch"):
            reduce_events(events, expected)

    def test_negative_token_id_is_rejected(self):
        events, expected = successful_transition()
        arrival = next(
            row for row in events if row["kind"] == "request_arrived")
        arrival["request"]["prompt_tokens"][0] = -1
        with self.assertRaisesRegex(
                EvidenceError, "nonnegative int32 token"):
            reduce_events(events, expected)

    def test_dispatch_is_the_only_initial_owner_assignment(self):
        events, expected = successful_transition()
        arrival = next(row for row in events if row["kind"] == "request_arrived")
        arrival["request"]["owner_id"] = "PHONE"
        with self.assertRaisesRegex(EvidenceError, "unexpected owner"):
            reduce_events(events, expected)

    def test_precommit_owner_change_is_rejected(self):
        events, expected = successful_transition()
        replay = next(row for row in events if row["kind"] == "replay_end")
        replay["request"]["owner_id"] = "GPU"
        with self.assertRaisesRegex(EvidenceError, "history digest mismatch|frontier"):
            reduce_events(events, expected)

    def test_stale_controller_epoch_is_rejected(self):
        events, expected = successful_transition()
        index = next(
            index for index, row in enumerate(events)
            if row["kind"] == "load_begin")
        events[index]["controller_epoch"] = 0
        with self.assertRaisesRegex(EvidenceError, "stale controller epoch"):
            reduce_events(events, expected)

    def test_duplicate_ownership_commit_is_rejected(self):
        events, expected = successful_transition()
        index = next(
            index for index, row in enumerate(events)
            if row["kind"] == "ownership_commit")
        duplicate = copy.deepcopy(events[index])
        duplicate["t_ns"] = events[index + 1]["t_ns"]
        events.insert(index + 1, duplicate)
        resequence(events)
        with self.assertRaisesRegex(
                EvidenceError, "request committed twice|stale or skipped epoch"):
            reduce_events(events, expected)

    def test_missing_ownership_commit_complete_is_rejected(self):
        events, expected = successful_transition()
        events = [
            row for row in events
            if row["kind"] != "ownership_commit_complete"
        ]
        resequence(events)
        with self.assertRaisesRegex(
                EvidenceError,
                "interleaved before commit complete|commit-complete marker"):
            reduce_events(events, expected)

    def test_forged_ownership_commit_count_is_rejected(self):
        events, expected = successful_transition()
        complete = next(
            row for row in events
            if row["kind"] == "ownership_commit_complete")
        complete["detail"] = "2"
        with self.assertRaisesRegex(EvidenceError, "commit count mismatch"):
            reduce_events(events, expected)

    def test_event_cannot_interleave_atomic_commit_group(self):
        events, expected = successful_transition()
        index = next(
            index for index, row in enumerate(events)
            if row["kind"] == "ownership_commit_complete")
        injected = copy.deepcopy(events[0])
        injected["controller_epoch"] = 1
        injected["kind"] = "phone_telemetry"
        injected["t_ns"] = events[index - 1]["t_ns"]
        injected["detail"] = "{}"
        events.insert(index, injected)
        resequence(events)
        with self.assertRaisesRegex(
                EvidenceError, "interleaved before commit complete"):
            reduce_events(events, expected)

    def test_commit_complete_requires_target_ready(self):
        events, expected = successful_transition()
        events = [
            row for row in events
            if not (
                row["kind"] == "model_state_changed"
                and row["model_id"] == "B"
                and row["executor_id"] == "GPU"
                and row["state_after"] == "READY"
            )
        ]
        resequence(events)
        with self.assertRaisesRegex(EvidenceError, "target is not ready"):
            reduce_events(events, expected)

    def test_completion_cannot_hide_unpublished_tokens(self):
        events, expected = successful_transition()
        token_indices = [
            index for index, row in enumerate(events)
            if row["kind"] == "token_committed"
        ]
        del events[token_indices[-1]]
        resequence(events)
        with self.assertRaisesRegex(
                EvidenceError, "token count or publication mismatch"):
            reduce_events(events, expected)

    def test_unexplained_lifecycle_failure_cannot_pass(self):
        events, expected = successful_transition()
        load_end = next(
            row for row in events if row["kind"] == "load_end")
        load_end["success"] = False
        result = reduce_events(events, expected)
        self.assertEqual(result["verdict"], "FAIL_EVENT")

    def test_general_draining_to_ready_is_rejected(self):
        events, expected = successful_transition()
        state = next(
            row for row in events
            if row["kind"] == "model_state_changed"
            and row["model_id"] == "B"
            and row["executor_id"] == "PHONE"
            and row["state_after"] == "DRAINING")
        state["state_before"] = "DRAINING"
        state["state_after"] = "READY"
        with self.assertRaisesRegex(
                EvidenceError, "not a failed precommit epoch|broken chain"):
            reduce_events(events, expected)

    def test_failed_precommit_rollback_preserves_old_owner(self):
        events, expected = precommit_rollback()
        result = reduce_events(events, expected)
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["precommit_rollback_count"], 1)
        self.assertEqual(result["phase_duration_ns"]["discard"]["count"], 1)
        completion = next(
            row for row in events if row["kind"] == "request_completed")
        self.assertEqual(completion["request"]["owner_id"], "PHONE")
        self.assertEqual(completion["request"]["ownership_epoch"], 1)

    def test_rollback_without_discard_is_rejected(self):
        events, expected = precommit_rollback()
        events = [
            row for row in events
            if row["kind"] not in {"discard_begin", "discard_end"}
        ]
        resequence(events)
        with self.assertRaisesRegex(
                EvidenceError, "not a failed precommit epoch"):
            reduce_events(events, expected)

    def test_missing_cleanup_end_is_rejected(self):
        events, expected = successful_transition()
        events = [
            row for row in events if row["kind"] != "cleanup_end"
        ]
        resequence(events)
        with self.assertRaisesRegex(EvidenceError, "unterminated phase"):
            reduce_events(events, expected)

    def test_cleanup_failure_is_a_failed_verdict(self):
        events, expected = successful_transition()
        cleanup = next(row for row in events if row["kind"] == "cleanup_end")
        cleanup["success"] = False
        result = reduce_events(events, expected)
        self.assertEqual(result["verdict"], "FAIL_CLEANUP")

    def test_executor_failure_is_preserved_not_fabricated(self):
        events, expected = executor_failure()
        result = reduce_events(events, expected)
        self.assertEqual(result["verdict"], "FAIL_EXECUTOR")
        self.assertEqual(result["completed_request_count"], 0)
        self.assertEqual(result["stranded_request_count"], 1)

    def test_illegal_model_transition_is_rejected(self):
        events, expected = successful_transition()
        state = next(
            row for row in events
            if row["kind"] == "model_state_changed"
            and row["state_before"] == "READY"
        )
        state["state_after"] = "ABSENT"
        with self.assertRaisesRegex(EvidenceError, "invalid transition"):
            reduce_events(events, expected)

    def test_warm_ready_does_not_overwrite_gpu_publication(self):
        events, expected = successful_transition()
        result = reduce_events(events, expected)
        intent_t = next(
            row["t_ns"] for row in events
            if row["kind"] == "switch_intent_submitted")
        commit_complete_t = next(
            row["t_ns"] for row in events
            if row["kind"] == "ownership_commit_complete")
        self.assertEqual(
            result["maximum_model_publication_gap_ns"],
            commit_complete_t - intent_t,
        )

    def test_resource_start_bracket_is_required(self):
        events, expected = successful_transition()
        rows = resources(events[0]["run_id"], events[-1]["t_ns"])
        rows[0]["t_ns"] = 1
        with self.assertRaisesRegex(EvidenceError, "start bracket"):
            reduce_events(events, expected, rows)


if __name__ == "__main__":
    unittest.main()
