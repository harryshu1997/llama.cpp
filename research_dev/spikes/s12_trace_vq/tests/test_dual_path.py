#!/usr/bin/env python3

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


SPIKE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIKE))

from dual_path_vq import (  # noqa: E402
    DualPathReplay,
    overlap_us,
    run_dual_path_config,
    s11_head_usb_result_bytes,
    s11_head_wifi_input_bytes,
    transfer_us,
    validate_dual_path_config,
    validate_dual_profile_evidence,
)
from profile_coverage import prepare_trace  # noqa: E402
from s12lib import S12Error, load_json, load_jsonl, profile_rows_by_batch, validate_profile  # noqa: E402


CONFIG_PATH = SPIKE / "configs" / "dual_path_shadow_fixture.json"
PROFILE_PATH = SPIKE / "profiles" / "s11_batched_route.json"
TRACE_PATH = SPIKE / "fixtures" / "varied_arrivals.jsonl"


def profile():
    return validate_profile(load_json(PROFILE_PATH))


def config():
    return validate_dual_path_config(copy.deepcopy(load_json(CONFIG_PATH)))


def coverage(limit: int | None = None):
    records = load_jsonl(TRACE_PATH)
    if limit is not None:
        records = records[:limit]
    return prepare_trace(records, profile(), "semi_synthetic_shape_shadow")


def replay(policy: str, cfg=None, count: int | None = None):
    p = profile()
    return DualPathReplay(
        policy,
        config() if cfg is None else cfg,
        coverage(count),
        profile_rows_by_batch(p),
        p["model"],
    ).run()


class DualPathContractTests(unittest.TestCase):
    def test_paths_must_be_distinct(self):
        cfg = copy.deepcopy(load_json(CONFIG_PATH))
        cfg["transport"]["usb_p2h"]["domain_id"] = cfg["transport"]["wifi_h2p"]["domain_id"]
        with self.assertRaisesRegex(S12Error, "distinct paths and domains"):
            validate_dual_path_config(cfg)

    def test_link_independence_is_topology_only_with_one_context(self):
        independent_cfg = config()
        serialized_cfg = copy.deepcopy(independent_cfg)
        serialized_cfg["transport"]["wifi_usb_independent"] = False
        independent = replay("fixed_phone", independent_cfg)
        serialized = replay("fixed_phone", serialized_cfg)
        self.assertEqual(independent["dual_path_overlap_us"], 0)
        self.assertEqual(serialized["dual_path_overlap_us"], 0)
        self.assertEqual(serialized["makespan_us"], independent["makespan_us"])
        self.assertEqual(independent["path_independence_status"], "TOPOLOGY_ASSUMPTION_UNMEASURED")
        self.assertEqual(serialized["path_independence_status"], "SERIALIZED_ABLATION_CONTROL")

    def test_unsupported_dynamic_residency_policy_is_rejected(self):
        cfg = copy.deepcopy(load_json(CONFIG_PATH))
        cfg["policies"].append("memory_admission_triggered")
        with self.assertRaisesRegex(S12Error, "unsupported policy"):
            validate_dual_path_config(cfg)

    def test_multiple_kv_owning_groups_are_rejected(self):
        cfg = copy.deepcopy(load_json(CONFIG_PATH))
        cfg["transport"]["phone_inflight_group_limit"] = 2
        with self.assertRaises(S12Error):
            validate_dual_path_config(cfg)

    def test_direction_is_fail_closed(self):
        cfg = copy.deepcopy(load_json(CONFIG_PATH))
        cfg["transport"]["wifi_h2p"]["direction"] = "PHONE_TO_HOST"
        with self.assertRaisesRegex(S12Error, "unsupported value"):
            validate_dual_path_config(cfg)

    def test_bool_cannot_impersonate_rate(self):
        cfg = copy.deepcopy(load_json(CONFIG_PATH))
        cfg["transport"]["usb_p2h"]["bytes_per_s"] = True
        with self.assertRaises(S12Error):
            validate_dual_path_config(cfg)

    def test_transfer_duration_uses_integer_ceiling(self):
        path = {
            "bytes_per_s": 3,
            "fixed_latency_us": 7,
        }
        self.assertEqual(transfer_us(1, path), 333341)

    def test_head_input_and_cut_result_are_not_conflated(self):
        p = profile()
        row = profile_rows_by_batch(p)[8]
        wifi = s11_head_wifi_input_bytes(8, p["model"])
        usb = s11_head_usb_result_bytes(row, p["model"])
        self.assertEqual(wifi, 1236)
        self.assertEqual(usb, 3809312)
        self.assertNotEqual(wifi, usb)

    def test_profile_timing_must_match_hashed_summary(self):
        value = profile()
        value["rows"][0]["phone_stage_us"] = 1
        with self.assertRaisesRegex(S12Error, "profile evidence mismatch"):
            validate_dual_profile_evidence(value)

    def test_background_timeline_cannot_extend_past_horizon(self):
        cfg = copy.deepcopy(load_json(CONFIG_PATH))
        cfg["horizon_us"] = 10
        with self.assertRaisesRegex(S12Error, "timestamp exceeds horizon"):
            validate_dual_path_config(cfg)


class DualPathReplayTests(unittest.TestCase):
    def test_old_s12_replay_is_not_the_dual_path_replay(self):
        result = run_dual_path_config(CONFIG_PATH)
        self.assertEqual(result["schema"], "s12-dual-path-replay-v1")
        self.assertNotEqual(
            result["deterministic_replay_sha256"],
            "sha256:b952f880eec584909f620939fa03a8dc7f20b734a6574fac1b4587f522bca734",
        )

    def test_phone_completion_requires_all_four_phases(self):
        result = replay("fixed_phone")
        outcomes = {row["event_id"]: row for row in result["outcomes"]}
        for decision in result["decisions"]:
            phases = decision["phase_us_by_step"]
            self.assertEqual(len(phases), 4)
            previous_tail = None
            for step in phases:
                ordered = [
                    step["wifi_start"],
                    step["wifi_finish"],
                    step["compute_start"],
                    step["compute_finish"],
                    step["usb_start"],
                    step["usb_finish"],
                    step["tail_start"],
                    step["tail_finish"],
                ]
                self.assertEqual(ordered, sorted(ordered))
                if previous_tail is not None:
                    self.assertGreaterEqual(step["wifi_start"], previous_tail)
                previous_tail = step["tail_finish"]
            for request_id in decision["request_ids"]:
                self.assertEqual(outcomes[request_id]["terminal"], "completed_phone")
                self.assertEqual(outcomes[request_id]["finish_us"], phases[-1]["tail_finish"])

    def test_wifi_and_usb_overlap(self):
        result = replay("fixed_phone")
        self.assertEqual(result["dual_path_overlap_us"], 0)
        self.assertTrue(result["wifi_usb_independent"])
        for decision in result["decisions"]:
            wifi = [
                (step["wifi_start"], step["wifi_finish"])
                for step in decision["phase_us_by_step"]
            ]
            usb = [
                (step["usb_start"], step["usb_finish"])
                for step in decision["phase_us_by_step"]
            ]
            self.assertEqual(overlap_us(wifi, usb), 0)

    def test_fixed_latency_is_charged_once_per_quantum(self):
        cfg = config()
        result = replay("fixed_phone", cfg)
        expected_wifi = sum(
            transfer_us(value, cfg["transport"]["wifi_h2p"])
            for decision in result["decisions"]
            for value in decision["wifi_input_bytes_by_step"]
        )
        expected_usb = sum(
            transfer_us(value, cfg["transport"]["usb_p2h"])
            for decision in result["decisions"]
            for value in decision["usb_result_bytes_by_step"]
        )
        self.assertEqual(result["wifi_h2p"]["busy_us"], expected_wifi)
        self.assertEqual(result["usb_p2h"]["busy_us"], expected_usb)

    def test_compute_never_overlaps_unmeasured_links(self):
        result = replay("fixed_phone")
        decisions = result["decisions"]
        compute = [
            (step["compute_start"], step["compute_finish"])
            for row in decisions
            for step in row["phase_us_by_step"]
        ]
        wifi = [
            (step["wifi_start"], step["wifi_finish"])
            for row in decisions
            for step in row["phase_us_by_step"]
        ]
        usb = [
            (step["usb_start"], step["usb_finish"])
            for row in decisions
            for step in row["phase_us_by_step"]
        ]
        self.assertEqual(overlap_us(compute, wifi), 0)
        self.assertEqual(overlap_us(compute, usb), 0)

    def test_each_policy_has_one_static_host_residency(self):
        phone = replay("fixed_phone")
        server = replay("causal_server_batch")
        self.assertEqual(phone["host_residency_mode"], "TAIL_ONLY")
        self.assertEqual(server["host_residency_mode"], "FULL_MODEL")
        self.assertEqual({row["route"] for row in phone["decisions"]}, {"A0_OP15"})
        self.assertEqual({row["route"] for row in server["decisions"]}, {"SERVER_ONLY"})
        self.assertEqual(phone["host_residency_transition_status"], "NONE_STATIC_FOR_REPLAY")
        self.assertEqual(server["host_residency_transition_status"], "NONE_STATIC_FOR_REPLAY")
        self.assertEqual(phone["host_resident_hbm_relief_mib"], 920)
        self.assertEqual(server["host_resident_hbm_relief_mib"], 0)
        self.assertEqual(phone["tail_vs_full_residency_delta_mib"], 920)
        self.assertEqual(server["tail_vs_full_residency_delta_mib"], 920)

    def test_policy_result_keys_are_uniform(self):
        results = run_dual_path_config(CONFIG_PATH)["policy_results"]
        self.assertEqual({tuple(sorted(result)) for result in results}, {tuple(sorted(results[0]))})

    def test_phone_groups_do_not_interleave_one_context(self):
        result = replay("fixed_phone")
        decisions = result["decisions"]
        self.assertGreater(len(decisions), 1)
        for previous, current in zip(decisions, decisions[1:]):
            self.assertGreaterEqual(
                current["phase_us_by_step"][0]["wifi_start"],
                previous["phase_us_by_step"][-1]["tail_finish"],
            )

    def test_path_byte_ledgers_equal_completed_transfers(self):
        result = replay("fixed_phone")
        wifi = sum(row["wifi_input_bytes"] for row in result["decisions"])
        usb = sum(row["usb_result_bytes"] for row in result["decisions"])
        self.assertEqual(result["wifi_h2p"]["payload_bytes_completed"], wifi)
        self.assertEqual(result["usb_p2h"]["payload_bytes_completed"], usb)
        for path in (result["wifi_h2p"], result["usb_p2h"]):
            self.assertEqual(
                path["payload_bytes_completed"],
                path["payload_bytes_useful"] + path["payload_bytes_wasted"],
            )
            self.assertEqual(path["payload_bytes_wasted"], 0)

    def test_buffers_remain_bounded(self):
        cfg = config()
        result = replay("fixed_phone", cfg)
        self.assertLessEqual(
            result["phone_ingress_buffer_peak_bytes"],
            cfg["transport"]["phone_ingress_buffer_bytes"],
        )
        self.assertLessEqual(
            result["phone_result_buffer_peak_bytes"],
            cfg["transport"]["phone_result_buffer_bytes"],
        )
        self.assertLessEqual(
            result["host_result_buffer_peak_bytes"],
            cfg["transport"]["host_result_buffer_bytes"],
        )

    def test_horizon_refuses_groups_that_cannot_finish(self):
        cfg = config()
        cfg["horizon_us"] = 600000
        cfg["background_hbm_mib_timeline"] = [{"t_us": 0, "used_mib": 0}]
        result = replay("fixed_phone", cfg, count=8)
        self.assertGreater(result["timed_out"], 0)
        self.assertEqual(result["discarded_inflight_groups"], 0)
        self.assertEqual(result["horizon_release"]["phone_inflight_groups"], 0)
        self.assertEqual(result["terminal_conservation"], 8)

    def test_tiny_ingress_buffer_backpressures_without_overflow(self):
        cfg = config()
        cfg["transport"]["phone_ingress_buffer_bytes"] = 1
        result = replay("fixed_phone", cfg, count=1)
        self.assertEqual(result["timed_out"], 1)
        self.assertEqual(result["phone_route_batches"], 0)
        self.assertEqual(result["discarded_inflight_groups"], 0)
        self.assertEqual(result["phone_ingress_buffer_peak_bytes"], 0)
        self.assertEqual(result["wifi_h2p"]["payload_bytes_completed"], 0)

    def test_smaller_batch_fallback_preserves_feasible_work(self):
        cfg = config()
        cfg["transport"]["phone_result_buffer_bytes"] = 500000
        cfg["transport"]["host_result_buffer_bytes"] = 500000
        result = replay("fixed_phone", cfg, count=2)
        batches = [row["batch_size"] for row in result["decisions"]]
        self.assertEqual(batches, [1, 1])
        self.assertEqual(result["completed_phone"], 2)

    def test_trace_arrival_past_horizon_is_rejected(self):
        cfg = config()
        cfg["horizon_us"] = 10
        cfg["background_hbm_mib_timeline"] = [{"t_us": 0, "used_mib": 0}]
        p = profile()
        with self.assertRaisesRegex(S12Error, "arrival exceeds replay horizon"):
            DualPathReplay(
                "fixed_phone",
                cfg,
                coverage(),
                profile_rows_by_batch(p),
                p["model"],
            )

    def test_slower_usb_increases_fixed_phone_makespan(self):
        fast_cfg = config()
        fast_cfg["horizon_us"] = 10000000
        slow_cfg = copy.deepcopy(fast_cfg)
        slow_cfg["transport"]["usb_p2h"]["bytes_per_s"] //= 8
        fast = replay("fixed_phone", fast_cfg, count=1)
        slow = replay("fixed_phone", slow_cfg, count=1)
        self.assertEqual(slow["completed_phone"], fast["completed_phone"])
        self.assertGreater(slow["makespan_us"], fast["makespan_us"])

    def test_strict_rows_remain_unprofiled(self):
        cfg = config()
        cfg["trace_mode"] = "strict_real"
        p = profile()
        strict = prepare_trace(load_jsonl(TRACE_PATH), p, "strict_real")
        result = DualPathReplay(
            "fixed_phone",
            cfg,
            strict,
            profile_rows_by_batch(p),
            p["model"],
        ).run()
        self.assertEqual(result["unprofiled"], 12)
        self.assertEqual(result["completed_phone"], 0)
        self.assertEqual(
            result["claim_scope"],
            "SYNTHETIC_ONLY_DUAL_PATH_MECHANICS_NO_PERFORMANCE_CLAIM",
        )

    def test_replay_is_byte_deterministic_in_process(self):
        first = run_dual_path_config(CONFIG_PATH)
        second = run_dual_path_config(CONFIG_PATH)
        self.assertEqual(
            first["deterministic_replay_sha256"],
            second["deterministic_replay_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
