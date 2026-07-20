#!/usr/bin/env python3

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock


SPIKE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SPIKE))

from policies import memory_admission_choice, offline_server_plan  # noqa: E402
from profile_coverage import prepare_trace  # noqa: E402
from s12lib import (  # noqa: E402
    S12Error,
    load_json,
    load_jsonl,
    profile_rows_by_batch,
    read_jsonl_snapshot,
    require_int,
    sha256_bytes,
    strict_json_loads,
    validate_profile,
)
from vq_sim import CausalReplay, run_config, run_offline, validate_config  # noqa: E402


PROFILE_PATH = SPIKE / "profiles" / "s11_batched_route.json"
TRACE_PATH = SPIKE / "fixtures" / "varied_arrivals.jsonl"
STRICT_CONFIG = SPIKE / "configs" / "strict_fixture.json"
SHADOW_CONFIG = SPIKE / "configs" / "shadow_fixture.json"


def profile():
    return validate_profile(load_json(PROFILE_PATH))


def exact_request():
    record = copy.deepcopy(load_jsonl(TRACE_PATH)[0])
    model = profile()["model"]
    record["provenance"] = "real"
    record["input_tokens"] = model["prompt_tokens"]
    record["output_tokens"] = model["generated_tokens"]
    record["source_fields"] = {
        "s11_chat": True,
        "s11_context_tokens": model["context_tokens"],
        "s11_host_model_sha256": model["host_model_sha256"],
        "s11_model_id": model["model_id"],
        "s11_prompt_sha256": model["prompt_sha256"],
    }
    return record


class StrictInputTests(unittest.TestCase):
    def test_duplicate_json_key_rejected(self):
        with self.assertRaisesRegex(S12Error, "duplicate JSON key"):
            strict_json_loads('{"a":1,"a":2}')

    def test_trace_snapshot_parses_and_hashes_one_read(self):
        first = b'{"generation":1}\n'
        mutated = b'{"generation":2}\n'
        with mock.patch.object(Path, "read_bytes", side_effect=[first, mutated]) as read:
            records, digest = read_jsonl_snapshot(Path("mutable-trace.jsonl"))
        self.assertEqual(read.call_count, 1)
        self.assertEqual(records, [{"generation": 1}])
        self.assertEqual(digest, sha256_bytes(first))

    def test_bool_is_not_integer(self):
        with self.assertRaises(S12Error):
            require_int("x", True)

    def test_config_bool_integer_rejected(self):
        cfg = copy.deepcopy(load_json(SHADOW_CONFIG))
        cfg["horizon_us"] = True
        with self.assertRaises(S12Error):
            validate_config(cfg)

    def test_profile_validates_and_binds_local_artifacts(self):
        validate_profile(load_json(PROFILE_PATH), base_dir=REPO, verify_artifacts=True)

    def test_profile_activation_formula_mutation_rejected(self):
        value = copy.deepcopy(load_json(PROFILE_PATH))
        value["rows"][2]["activation_bytes"] += 1
        with self.assertRaisesRegex(S12Error, "formula mismatch"):
            validate_profile(value)

    def test_profile_digest_mutation_rejected_when_verified(self):
        value = copy.deepcopy(load_json(PROFILE_PATH))
        value["rows"][0]["summary_sha256"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(S12Error, "digest mismatch"):
            validate_profile(value, base_dir=REPO, verify_artifacts=True)


class CoverageTests(unittest.TestCase):
    def test_strict_fixture_has_zero_eligible_rows(self):
        result = prepare_trace(load_jsonl(TRACE_PATH), profile(), "strict_real")
        self.assertEqual(result["profiled_requests"], 0)
        self.assertEqual(result["unprofiled_requests"], 12)

    def test_shape_alone_is_not_eligible(self):
        record = copy.deepcopy(load_jsonl(TRACE_PATH)[0])
        record["provenance"] = "real"
        result = prepare_trace([record], profile(), "strict_real")
        self.assertFalse(result["prepared_requests"][0]["profile_eligible"])
        self.assertIn("E_MODEL_ID", result["prepared_requests"][0]["reasons"])

    def test_exact_identity_is_eligible(self):
        result = prepare_trace([exact_request()], profile(), "strict_real")
        self.assertEqual(result["profiled_requests"], 1)
        self.assertTrue(result["prepared_requests"][0]["profile_eligible"])

    def test_integer_one_cannot_impersonate_chat_boolean(self):
        record = exact_request()
        record["source_fields"]["s11_chat"] = 1
        result = prepare_trace([record], profile(), "strict_real")
        self.assertFalse(result["prepared_requests"][0]["profile_eligible"])
        self.assertIn("E_CHAT", result["prepared_requests"][0]["reasons"])

    def test_shadow_is_explicitly_synthetic_only(self):
        result = prepare_trace(load_jsonl(TRACE_PATH), profile(), "semi_synthetic_shape_shadow")
        self.assertEqual(result["profiled_requests"], 12)
        self.assertEqual(
            result["claim_scope"],
            "SYNTHETIC_ONLY_NO_REAL_TRACE_PERFORMANCE_CLAIM",
        )
        self.assertTrue(
            all(req["provenance"] == "semi_synthetic" for req in result["prepared_requests"])
        )

    def test_duplicate_event_id_rejected(self):
        records = [exact_request(), exact_request()]
        with self.assertRaisesRegex(S12Error, "duplicate event_id"):
            prepare_trace(records, profile(), "strict_real")

    def test_unordered_trace_rejected(self):
        records = [exact_request(), exact_request()]
        records[0]["event_id"] = "r0"
        records[0]["t_us"] = 10
        records[1]["event_id"] = "r1"
        records[1]["t_us"] = 0
        with self.assertRaisesRegex(S12Error, "must already be ordered"):
            prepare_trace(records, profile(), "strict_real")


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.rows = profile_rows_by_batch(profile())

    def test_memory_policy_never_uses_phone_when_server_fits(self):
        choice = memory_admission_choice(
            8,
            True,
            tuple(sorted(self.rows)),
            26000,
            self.rows,
            True,
        )
        self.assertEqual(choice.route, "SERVER_ONLY")
        self.assertTrue(choice.server_admission_feasible)

    def test_memory_policy_uses_phone_only_on_infeasible_server(self):
        choice = memory_admission_choice(
            8,
            True,
            tuple(sorted(self.rows)),
            22500,
            self.rows,
            True,
        )
        self.assertEqual(choice.route, "A0_OP15")
        self.assertEqual(choice.batch_size, 1)
        self.assertFalse(choice.server_admission_feasible)

    def test_offline_reference_reads_future_and_batches(self):
        groups = offline_server_plan(
            [0, 100000],
            self.rows,
            26000,
            [{"t_us": 0, "used_mib": 0}],
            16,
            1000000,
            24,
        )
        self.assertEqual(groups, [{"batch_size": 2, "finish_us": 266957, "start_us": 100000}])


class ReplayTests(unittest.TestCase):
    def test_strict_trace_is_unprofiled_not_given_latency(self):
        result = run_config(STRICT_CONFIG)
        for row in result["policy_results"]:
            self.assertEqual(row["unprofiled"], 12)
            self.assertEqual(row["completed_server"] + row["completed_phone"], 0)
            self.assertEqual(row["terminal_conservation"], 12)
            self.assertEqual(row["energy"]["status"], "NOT_RUN")

    def test_shadow_all_policies_conserve_terminals(self):
        result = run_config(SHADOW_CONFIG)
        self.assertEqual(len(result["policy_results"]), 4)
        for row in result["policy_results"]:
            self.assertEqual(row["terminal_conservation"], row["requests"])
            self.assertEqual(row["unprofiled"], 0)
            self.assertEqual(row["energy"]["status"], "NOT_RUN")

    def test_memory_phone_decisions_are_admission_triggered(self):
        result = run_config(SHADOW_CONFIG, ["memory_admission_triggered"])
        row = result["policy_results"][0]
        phone = [decision for decision in row["decisions"] if decision["route"] == "A0_OP15"]
        self.assertGreater(len(phone), 0)
        self.assertEqual(row["memory_phone_selections"], len(phone))
        self.assertTrue(all(not decision["server_admission_feasible"] for decision in phone))

    def test_activation_bytes_are_exact(self):
        result = run_config(SHADOW_CONFIG, ["memory_admission_triggered"])
        row = result["policy_results"][0]
        expected = sum(
            decision["batch_size"] * (28 + 3) * 3840 * 4
            for decision in row["decisions"]
            if decision["route"] == "A0_OP15"
        )
        self.assertEqual(row["activation_bytes_total"], expected)

    def test_causal_policy_does_not_read_future_arrivals(self):
        cfg = validate_config(copy.deepcopy(load_json(SHADOW_CONFIG)))
        coverage = prepare_trace(load_jsonl(TRACE_PATH), profile(), "semi_synthetic_shape_shadow")
        prefix = copy.deepcopy(coverage)
        prefix["prepared_requests"] = prefix["prepared_requests"][:4]
        prefix["total_requests"] = 4
        prefix["profiled_requests"] = 4
        full_sim = CausalReplay("memory_admission_triggered", cfg, coverage, self.rows())
        prefix_sim = CausalReplay("memory_admission_triggered", cfg, prefix, self.rows())
        full = full_sim.run()
        short = prefix_sim.run()
        self.assertEqual(full["decisions"][0], short["decisions"][0])

    def rows(self):
        return profile_rows_by_batch(profile())

    def test_finite_queue_overflow_is_terminal(self):
        cfg = validate_config(copy.deepcopy(load_json(SHADOW_CONFIG)))
        cfg["server_queue_limit"] = 1
        cfg["batch_hold_us"] = 300000
        coverage = prepare_trace(load_jsonl(TRACE_PATH), profile(), "semi_synthetic_shape_shadow")
        result = CausalReplay("causal_server_batch", cfg, coverage, self.rows()).run()
        self.assertGreater(result["rejected_queue_full"], 0)
        self.assertEqual(result["terminal_conservation"], result["requests"])

    def test_hbm_increase_during_active_route_fails_closed(self):
        cfg = validate_config(copy.deepcopy(load_json(SHADOW_CONFIG)))
        cfg["background_hbm_mib_timeline"] = [
            {"t_us": 0, "used_mib": 0},
            {"t_us": 100000, "used_mib": 4000},
        ]
        coverage = prepare_trace([load_jsonl(TRACE_PATH)[0]], profile(), "semi_synthetic_shape_shadow")
        with self.assertRaisesRegex(S12Error, "oversubscribed an active route"):
            CausalReplay("causal_server_batch", cfg, coverage, self.rows()).run()

    def test_hbm_change_at_exact_completion_uses_half_open_route(self):
        cfg = validate_config(copy.deepcopy(load_json(SHADOW_CONFIG)))
        cfg["background_hbm_mib_timeline"] = [
            {"t_us": 0, "used_mib": 0},
            {"t_us": 209619, "used_mib": 4000},
        ]
        coverage = prepare_trace([load_jsonl(TRACE_PATH)[0]], profile(), "semi_synthetic_shape_shadow")
        result = CausalReplay("causal_server_batch", cfg, coverage, self.rows()).run()
        self.assertEqual(result["completed_server"], 1)
        self.assertEqual(result["timed_out"], 0)

    def test_offline_is_marked_clairvoyant(self):
        result = run_config(SHADOW_CONFIG, ["server_only_optimized"])
        self.assertTrue(result["policy_results"][0]["clairvoyant"])
        self.assertTrue(result["policy_results"][0]["is_offline_upper_bound"])

    def test_offline_peak_includes_rising_background_during_active_route(self):
        cfg = validate_config(copy.deepcopy(load_json(SHADOW_CONFIG)))
        cfg["background_hbm_mib_timeline"] = [
            {"t_us": 0, "used_mib": 1000},
            {"t_us": 100000, "used_mib": 2000},
        ]
        coverage = prepare_trace([load_jsonl(TRACE_PATH)[0]], profile(), "semi_synthetic_shape_shadow")
        result = run_offline(cfg, coverage, self.rows())
        self.assertEqual(result["completed_server"], 1)
        self.assertEqual(result["peak_a6000_hbm_mib"], 25254)

    def test_background_memory_is_accounted_without_dispatch(self):
        result = run_config(STRICT_CONFIG, ["causal_server_batch"])
        self.assertEqual(result["policy_results"][0]["peak_a6000_hbm_mib"], 3500)

    def test_duplicate_selected_policy_rejected(self):
        with self.assertRaisesRegex(S12Error, "non-empty and unique"):
            run_config(SHADOW_CONFIG, ["fixed_phone", "fixed_phone"])

    def test_replay_is_byte_deterministic_in_process(self):
        first = run_config(SHADOW_CONFIG)
        second = run_config(SHADOW_CONFIG)
        self.assertEqual(
            first["deterministic_replay_sha256"],
            second["deterministic_replay_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
