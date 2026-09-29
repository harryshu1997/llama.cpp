#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from research_dev.scheduler.campaigns.burstgpt.runner import (
    apply_named_replay_schedule,
    execution_fraction_history,
    persist_replay_schedule,
    scale_replay_arrivals,
)
from research_dev.scheduler.campaigns.burstgpt.offline_residency_gate import (
    OfflineResidencyGateError,
    _SessionCallDeltaTimeout,
    _SessionCallGapError,
    _retained_session_call_gap,
    _matched_session_call_gap,
    _matched_retained_call_rows,
    _drain_timeline,
    _require_online_coverage,
    _require_interruption_evidence,
    _select_gate_items,
    _stage_phase_interval,
    _stage_target_generation,
    _wait_session_call_deltas,
)


def item(index: int, request_id: str, arrival_us: int) -> dict[str, object]:
    return {
        "combined_index": index,
        "model_id": "synthetic-model",
        "row": {
            "arrival_us": arrival_us,
            "event_id": request_id,
        },
        "source": "synthetic",
        "source_index": index,
    }


class BurstGptReplayTests(unittest.TestCase):
    def test_completed_lifecycle_still_requires_both_coverage_thresholds(self):
        online = {
            model + "_weighted_assisted_coverage": {"coverage_ppm": 700_000}
            for model in ("qwen", "gemma")
        }
        _require_online_coverage(online)
        for model in ("qwen", "gemma"):
            key = model + "_weighted_assisted_coverage"
            online[key]["coverage_ppm"] = 699_999
            with self.subTest(model=model), self.assertRaisesRegex(
                OfflineResidencyGateError, model.title() + " fraction-weighted coverage"
            ):
                _require_online_coverage(online)
            online[key]["coverage_ppm"] = 700_000

    def test_reverse_stage_generation_comes_from_selected_session_map(self):
        shard = SimpleNamespace(session_id="selected", artifact_sha256="artifact")
        layout = SimpleNamespace(
            shards=(shard,), session_generation_by_id={"retained": 1, "selected": 3},
        )
        stage = SimpleNamespace(
            selected_session_id=shard.session_id, layout=SimpleNamespace(layout=layout),
        )
        self.assertFalse(hasattr(shard, "session_generation"))
        self.assertEqual(_stage_target_generation(stage), 3)
        layout.session_generation_by_id[shard.session_id] = 5
        self.assertEqual(_stage_target_generation(stage), 5)
        self.assertEqual(layout.session_generation_by_id["retained"], 1)
        del layout.session_generation_by_id[shard.session_id]
        with self.assertRaises(KeyError):
            _stage_target_generation(stage)

    def test_drain_timeline_separates_control_delay_and_loading(self):
        helpers = [
            {"kind": "REBIND_DRAIN_POLICY_BOUND", "request_id": "request",
             "drain_policy_sha256": "policy", "observed_at_us": 10},
            {"kind": "REBIND_QUIESCED", "request_id": "request",
             "drain_policy_sha256": "policy", "observed_at_us": 13},
        ]
        timings = [
            {"kind": "DECODE_BOUNDARY_OBSERVED", "request_id": "request",
             "observed_at_us": 11, "token_index": 5, "terminal": False},
            {"kind": "CONTROL_ISSUED", "request_id": "request",
             "observed_at_us": 12, "control": {"policy_hash": "policy"}},
        ]
        group = {"windows": [{"started_at_us": 2, "finished_at_us": 12,
                              "token_start": 1, "token_end": 5}]}
        stage = SimpleNamespace(
            selected_session_id="selected-session", verified_at_us=26,
            transition_receipts=(SimpleNamespace(started_us=15, finished_us=25),),
        )
        result = _drain_timeline(helpers, timings, group, stage)
        self.assertEqual(result["drain_to_quiesced_us"], 3)
        self.assertEqual(result["next_safe_boundary_observed_at_us"], 11)
        self.assertEqual(result["control_issued_at_us"], 12)
        self.assertEqual(result["replacement_loading_started_at_us"], 15)
        self.assertEqual(result["physical_ready_ack_at_us"], 25)
        self.assertEqual(result["ready_publication_at_us"], 26)
        self.assertEqual(result["shortened_window_tokens"], 4)

    @staticmethod
    def _equivalent_gap_fixture():
        artifact = "sha256:" + "3" * 64
        rows = [{
            "artifact_sha256": artifact,
            "session_id": "session-retained", "session_generation": 1,
            "calls": token * 6 + layer + 1,
            "monotonic_us": token * 500 + layer * 10,
            "layer": layer, "token_ordinal": token,
            "columns": 1024, "fraction_ppm": 1_000_000,
            "active_layer_mask": 63, "tokens": 1, "plan_generation": 1,
        } for token in range(80) for layer in range(6)]
        interval = {"load_authorized_monotonic_us": 35 * 500 + 15,
                    "ready_monotonic_us": 38 * 500 + 15}
        return rows, interval, artifact

    def test_equivalent_classes_do_not_mix_layer_and_token_gaps(self):
        rows, interval, artifact = self._equivalent_gap_fixture()
        with self.assertRaises(_SessionCallGapError):
            _retained_session_call_gap(
                rows, interval, artifact_sha256=artifact,
                session_id="session-retained", session_generation=1,
            )
        result = _matched_session_call_gap(rows, interval, "session-retained", 1)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["schema"], "s42-retained-session-call-gap-v2")
        self.assertEqual(len(result["classes"]), 11)
        self.assertTrue(all(len(row["reference_intervals"]) == 30
                            for row in result["classes"]))
        self.assertEqual({row["baseline_median_us"] for row in result["classes"]},
                         {10, 500})

    def test_equivalent_metric_rejects_real_stall(self):
        rows, interval, _artifact = self._equivalent_gap_fixture()
        for row in rows:
            if row["monotonic_us"] >= interval["load_authorized_monotonic_us"]:
                row["monotonic_us"] += 1_100
        result = _matched_session_call_gap(rows, interval, "session-retained", 1)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any(row["status"] == "FAIL" for row in result["classes"]))

    def test_equivalent_metric_requires_same_mask_and_fraction(self):
        for field, changed in (("active_layer_mask", 127), ("columns", 512),
                               ("fraction_ppm", 500_000)):
            with self.subTest(field=field):
                rows, interval, _artifact = self._equivalent_gap_fixture()
                for row in rows:
                    if not 34 <= row["token_ordinal"] <= 39:
                        row[field] = changed
                result = _matched_session_call_gap(rows, interval, "session-retained", 1)
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(all(row["status"] == "INSUFFICIENT"
                                    for row in result["classes"]))

    def test_equivalent_metric_can_use_exact_post_ready_reference(self):
        rows, interval, _artifact = self._equivalent_gap_fixture()
        for row in rows:
            if row["token_ordinal"] < 34:
                row["active_layer_mask"] = 127
        result = _matched_session_call_gap(rows, interval, "session-retained", 1)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(row["baseline_after_count"] > 0
                            for row in result["classes"]))

    def test_incomplete_gap_collection_never_becomes_a_passing_verdict(self):
        rows, interval, _artifact = self._equivalent_gap_fixture()
        for row in rows:
            if not 34 <= row["token_ordinal"] <= 39:
                row["columns"] = 512
        measured = _matched_session_call_gap(rows, interval, "session-retained", 1)
        _require_interruption_evidence([measured], collect_incomplete_reference=True)
        self.assertEqual(measured["status"], "FAIL")
        self.assertTrue(all(row["status"] == "INSUFFICIENT"
                            for row in measured["classes"]))
        with self.assertRaises(_SessionCallGapError):
            _require_interruption_evidence([measured])

    def test_gap_collection_does_not_defer_an_observed_stall(self):
        rows, interval, _artifact = self._equivalent_gap_fixture()
        for row in rows:
            if row["monotonic_us"] >= interval["load_authorized_monotonic_us"]:
                row["monotonic_us"] += 1_100
        measured = _matched_session_call_gap(rows, interval, "session-retained", 1)
        with self.assertRaises(_SessionCallGapError):
            _require_interruption_evidence([measured], collect_incomplete_reference=True)

    def test_absent_transition_calls_are_insufficient_not_a_measured_stall(self):
        rows, interval, _artifact = self._equivalent_gap_fixture()
        for observed in ([], rows[:60]):
            with self.subTest(call_count=len(observed)):
                measured = _matched_session_call_gap(
                    observed, interval, "session-retained", 1,
                )
                self.assertEqual(measured["status"], "INSUFFICIENT")
                self.assertEqual(measured["classes"], [])
                _require_interruption_evidence(
                    [measured], collect_incomplete_reference=True,
                )
                with self.assertRaises(_SessionCallGapError):
                    _require_interruption_evidence([measured])

    def test_native_phone_join_requires_exact_generation_and_counter(self):
        native = [{"artifact_sha256": "artifact", "tokens": 1,
                   "layer": 0, "columns": 128,
                   "contexts": [{"scheduler_request_id": "request", "plan_generation": 2}]}]
        group = {"windows": [{"applied_ack": {"plan_generation": 2},
                              "policy": {"layer_mask": 1, "columns": 128,
                                         "split_fraction_ppm": 1_000_000}}]}
        phone = [{"artifact_sha256": "artifact", "session_id": "retained",
                  "session_generation": 3, "calls": 8, "monotonic_us": 42}]
        sessions = {"retained": {"artifact_sha256": "artifact", "layer_mask": 1,
                                 "session_generation": 3}}
        args = (native, phone, group, sessions, {("retained", 3): 7}, "request", "artifact")
        self.assertEqual(_matched_retained_call_rows(*args)[0]["monotonic_us"], 42)
        phone[0]["session_generation"] = 2
        with self.assertRaisesRegex(OfflineResidencyGateError, "exact phone counter"):
            _matched_retained_call_rows(*args)

    @staticmethod
    def _call_gap_fixture():
        artifact = "sha256:" + "3" * 64
        rows = [
            {
                "artifact_sha256": artifact,
                "calls": index,
                "epoch_us": 1_000_000 - index * 1_000,
                "monotonic_us": index * 10,
                "session_generation": 1,
                "session_id": "session-retained",
            }
            for index in range(1, 36)
        ]
        interval = {
            "load_authorized_monotonic_us": 315,
            "ready_monotonic_us": 335,
        }
        return rows, interval, artifact

    def test_retained_gap_uses_phone_monotonic_clock(self) -> None:
        rows, interval, artifact = self._call_gap_fixture()
        result = _retained_session_call_gap(
            rows, interval, artifact_sha256=artifact,
            session_id="session-retained", session_generation=1,
        )
        self.assertEqual(result["baseline_interval_count"], 30)
        self.assertEqual(result["baseline_median_inter_call_us"], 10)
        self.assertEqual(result["maximum_transition_inter_call_us"], 10)
        self.assertEqual(result["clock"], "phone_monotonic_us")

    def test_retained_gap_failure_preserves_measurements(self) -> None:
        rows, interval, artifact = self._call_gap_fixture()
        for row in rows[31:]:
            row["monotonic_us"] += 30
        with self.assertRaises(_SessionCallGapError) as raised:
            _retained_session_call_gap(
                rows, interval, artifact_sha256=artifact,
                session_id="session-retained", session_generation=1,
            )
        measurement = raised.exception.call_gap_measurement
        self.assertEqual(measurement["bound_us"], 20)
        self.assertEqual(measurement["maximum_transition_inter_call_us"], 40)
        self.assertEqual(len(measurement["reference_calls"]), 31)

    def test_retained_gap_rejects_sampled_or_foreign_generation_calls(self) -> None:
        rows, interval, artifact = self._call_gap_fixture()
        rows[31]["session_generation"] = 2
        with self.assertRaisesRegex(
            OfflineResidencyGateError, "lost a call event"
        ):
            _retained_session_call_gap(
                rows, interval, artifact_sha256=artifact,
                session_id="session-retained", session_generation=1,
            )

    def test_session_call_wait_returns_when_request_completes(self) -> None:
        artifact = "sha256:" + "1" * 64
        rig = SimpleNamespace(phone_residency_call_events=[{
            "artifact_sha256": artifact,
            "calls": 4,
            "session_generation": 1,
            "session_id": "HTP0",
        }])

        totals = _wait_session_call_deltas(
            rig,
            artifact,
            {("HTP0", 1): 4},
            {"HTP0": 1},
            minimum_delta=1,
            request_completed=lambda: True,
            timeout_s=1,
        )

        self.assertEqual(totals, {("HTP0", 1): 4})

    def test_session_call_timeout_carries_baselines_and_totals(self) -> None:
        artifact = "sha256:" + "2" * 64
        rig = SimpleNamespace(phone_residency_call_events=[{
            "artifact_sha256": artifact,
            "calls": 5,
            "session_generation": 1,
            "session_id": "HTP0",
        }])

        with self.assertRaises(_SessionCallDeltaTimeout) as raised:
            _wait_session_call_deltas(
                rig,
                artifact,
                {("HTP0", 1): 4, ("HTP1", 1): 7},
                {"HTP0": 1, "HTP1": 1},
                minimum_delta=2,
                request_completed=lambda: False,
                timeout_s=0.001,
            )

        self.assertEqual(
            raised.exception.call_diagnostics["baselines"],
            [
                {
                    "calls": 4,
                    "session_generation": 1,
                    "session_id": "HTP0",
                },
                {
                    "calls": 7,
                    "session_generation": 1,
                    "session_id": "HTP1",
                },
            ],
        )
        self.assertEqual(
            raised.exception.call_diagnostics["observed_totals"],
            [
                {
                    "calls": 5,
                    "session_generation": 1,
                    "session_id": "HTP0",
                },
                {
                    "calls": 0,
                    "session_generation": 1,
                    "session_id": "HTP1",
                },
            ],
        )

    def test_stage_interval_reads_generation_from_layout(self) -> None:
        shard = SimpleNamespace(
            session_id="HTP1",
            artifact_sha256="sha256:" + "1" * 64,
        )
        stage = SimpleNamespace(
            selected_session_id="HTP1",
            layout=SimpleNamespace(layout=SimpleNamespace(
                shards=(shard,),
                session_generation_by_id={"HTP1": 3},
            )),
        )
        events = tuple(
            {
                "artifact_sha256": shard.artifact_sha256,
                "component": "resident-manager",
                "epoch_us": epoch_us,
                "monotonic_us": epoch_us + 100,
                "phase": phase,
                "session_generation": 3,
                "session_id": "HTP1",
            }
            for phase, epoch_us in (
                ("LOAD_AUTHORIZED", 10),
                ("VERIFIED", 20),
                ("READY", 30),
            )
        )

        interval = _stage_phase_interval(events, stage)

        self.assertEqual(interval["session_generation"], 3)
        self.assertEqual(interval["ready_epoch_us"], 30)
        self.assertEqual(interval["ready_monotonic_us"], 130)

    def test_residency_gate_uses_only_selected_replay_requests(self) -> None:
        models = SimpleNamespace(
            expected_qwen=SimpleNamespace(model_id="qwen"),
            expected_gemma=SimpleNamespace(model_id="gemma"),
        )
        selected = [
            {
                "model_id": "qwen",
                "row": {"input_tokens": 100, "output_tokens": 50},
                "source": "large",
            },
            {
                "model_id": "gemma",
                "row": {"input_tokens": 80, "output_tokens": 60},
                "source": "large",
            },
            {
                "model_id": "llama",
                "row": {"input_tokens": 90, "output_tokens": 70},
                "source": "overlay",
            },
        ]

        large, qwen, gemma = _select_gate_items(models, selected)

        self.assertEqual(len(large), 2)
        self.assertIs(qwen, large[0])
        self.assertIs(gemma, large[1])

    def test_scaled_arrivals_preserve_order_and_simultaneous_gaps(
        self,
    ) -> None:
        selected = [
            item(34, "request-a", 10_000_000),
            item(35, "request-b", 10_000_000),
            item(36, "request-c", 24_050_000),
        ]

        scaled, schedule = scale_replay_arrivals(selected, 60)

        self.assertEqual(
            [row["row"]["arrival_us"] for row in scaled],
            [1_000_000, 1_000_000, 844_000_000],
        )
        self.assertEqual(schedule["source_span_us"], 14_050_000)
        self.assertEqual(schedule["replay_span_us"], 843_000_000)
        self.assertEqual(schedule["selected_indices"], [34, 35, 36])
        self.assertEqual(
            scaled[1]["row"]["source_arrival_us"], 10_000_000
        )
        self.assertEqual(
            scaled[1]["row"]["replay_arrival_us"], 1_000_000
        )

    def test_schedule_hash_is_deterministic(self) -> None:
        selected = [
            item(34, "request-a", 10_000_000),
            item(35, "request-b", 10_100_000),
        ]

        _, first = scale_replay_arrivals(selected, 60)
        _, second = scale_replay_arrivals(selected, 60)
        _, different = scale_replay_arrivals(selected, 61)

        self.assertEqual(
            first["schedule_sha256"], second["schedule_sha256"]
        )
        self.assertNotEqual(
            first["schedule_sha256"], different["schedule_sha256"]
        )

    def test_persisted_schedule_contains_source_and_replay_arrivals(
        self,
    ) -> None:
        _, schedule = scale_replay_arrivals(
            [item(34, "request-a", 10_000_000)], 60
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "REPLAY_SCHEDULE.json"

            persist_replay_schedule(path, schedule)
            persisted = json.loads(path.read_text(encoding="ascii"))

        self.assertEqual(persisted, schedule)
        self.assertEqual(
            persisted["schedule"][0],
            {
                "combined_request_index": 34,
                "replay_arrival_us": 1_000_000,
                "request_id": "request-a",
                "source_arrival_us": 10_000_000,
            },
        )

    def test_named_sparse_schedule_preserves_source_work(self) -> None:
        merged = [
            item(34, "request-a", 10_000_000),
            item(35, "request-b", 10_100_000),
            item(36, "request-c", 10_200_000),
        ]
        value = {
            "arrivals": [
                {
                    "combined_request_index": 36,
                    "replay_arrival_us": 1_000_000,
                },
                {
                    "combined_request_index": 34,
                    "replay_arrival_us": 21_000_000,
                },
            ],
            "schema": "research-scheduler-burstgpt-replay-v1",
            "trace_name": "synthetic_sparse_locality",
        }

        selected, schedule = apply_named_replay_schedule(merged, value)
        repeated, repeated_schedule = apply_named_replay_schedule(
            merged, value
        )

        self.assertEqual(
            [row["combined_index"] for row in selected], [36, 34]
        )
        self.assertEqual(
            [row["row"]["source_arrival_us"] for row in selected],
            [10_200_000, 10_000_000],
        )
        self.assertEqual(
            [row["row"]["replay_arrival_us"] for row in selected],
            [1_000_000, 21_000_000],
        )
        self.assertEqual(selected, repeated)
        self.assertEqual(
            schedule["schedule_sha256"],
            repeated_schedule["schedule_sha256"],
        )
        self.assertEqual(
            schedule["trace_name"], "synthetic_sparse_locality"
        )

    def test_fraction_history_is_bound_to_the_executed_ticket(self) -> None:
        imported = {
            "final_policy": {"split_fraction_ppm": 750_000},
            "grouped_observation_sha256": "sha256:" + "1" * 64,
            "request_id": "request-a",
            "ticket_id": "request-a:attempt:0",
            "windows": [{
                "completed_phone_calls": 4,
                "failure_reason": None,
                "output_valid": True,
                "policy": {"split_fraction_ppm": 750_000},
                "window_role": "exploration",
            }],
        }
        desktop = SimpleNamespace(
            execution_mode="desktop",
            initial_split_fraction_ppm=0,
        )

        history = execution_fraction_history(
            desktop,
            "request-a:attempt:1",
            [imported],
            0,
        )

        self.assertEqual(history["initial_split_fraction_ppm"], 0)
        self.assertEqual(history["selected_split_fraction_ppm"], 0)
        self.assertEqual(history["explored_split_fractions_ppm"], [])
        self.assertEqual(
            history["physically_executed_split_fractions_ppm"], [0]
        )
        self.assertEqual(
            history["phone_executed_split_fractions_ppm"], []
        )

    def test_fraction_history_uses_one_shared_cohort_observation(self) -> None:
        shared = {
            "final_policy": {"split_fraction_ppm": 750_000},
            "grouped_observation_sha256": "sha256:" + "2" * 64,
            "request_id": "request-leader",
            "ticket_id": "request-leader:attempt:1",
            "windows": [
                {
                    "cohort_id": "cohort-a",
                    "cohort_member_request_ids": [
                        "request-leader", "request-follower",
                    ],
                    "completed_phone_calls": 0,
                    "energy_owner_request_id": "request-leader",
                    "failure_reason": None,
                    "output_valid": True,
                    "policy": {"split_fraction_ppm": 0},
                    "window_role": "baseline",
                },
                {
                    "cohort_id": "cohort-a",
                    "cohort_member_request_ids": [
                        "request-leader", "request-follower",
                    ],
                    "completed_phone_calls": 8,
                    "energy_owner_request_id": "request-leader",
                    "failure_reason": None,
                    "output_valid": True,
                    "policy": {"split_fraction_ppm": 500_000},
                    "window_role": "exploration",
                },
                {
                    "cohort_id": None,
                    "cohort_member_request_ids": None,
                    "completed_phone_calls": 8,
                    "energy_owner_request_id": None,
                    "failure_reason": None,
                    "output_valid": True,
                    "policy": {"split_fraction_ppm": 750_000},
                    "window_role": "exploitation",
                },
            ],
        }
        adaptive = SimpleNamespace(
            execution_mode="adaptive-split",
            initial_split_fraction_ppm=0,
        )
        cohort = SimpleNamespace(
            cohort_id="cohort-a",
            leader_request_id="request-leader",
            member_request_ids=("request-leader", "request-follower"),
        )

        history = execution_fraction_history(
            adaptive,
            "request-follower:attempt:1",
            [shared],
            8,
            request_id="request-follower",
            decode_cohort=cohort,
        )

        self.assertEqual(
            history["adaptive_observation_scope"],
            "decode_cohort_shared",
        )
        self.assertEqual(
            history["adaptive_observation_ticket_id"],
            "request-leader:attempt:1",
        )
        self.assertEqual(
            history["physically_executed_split_fractions_ppm"],
            [0, 500_000],
        )
        self.assertEqual(
            history["phone_executed_split_fractions_ppm"],
            [500_000],
        )
        self.assertEqual(
            history["selected_split_fraction_ppm"], 500_000
        )


if __name__ == "__main__":
    unittest.main()
