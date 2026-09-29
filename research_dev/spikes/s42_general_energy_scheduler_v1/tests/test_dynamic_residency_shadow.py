#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
for path in (REPO_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_residency_v1.shadow_replay import (  # noqa: E402
    DEFAULT_ADOPTION_ENERGY_SCREEN,
    DEFAULT_AGGREGATE,
    DEFAULT_GPU_CAPACITY,
    DEFAULT_GPU_DIAGNOSTIC,
    DEFAULT_GPU_TENSOR_BUNDLE,
    DEFAULT_PHONE_PLAN,
    DEFAULT_TRACE,
    ShadowReplayError,
    build_shadow,
    canonical,
    read_object,
    read_trace,
    sha256,
)
from research_dev.scheduler import PhoneResidencyPlan  # noqa: E402


RESULT = (
    ROOT
    / "dynamic_residency_v1/results/DYNAMIC_RESIDENCY_SHADOW_V1.json"
)


def build() -> dict[str, object]:
    return build_shadow(
        read_object(DEFAULT_AGGREGATE),
        read_trace(DEFAULT_TRACE),
        PhoneResidencyPlan.from_json(read_object(DEFAULT_PHONE_PLAN)),
        read_object(DEFAULT_GPU_CAPACITY),
        read_object(DEFAULT_GPU_DIAGNOSTIC),
        read_object(DEFAULT_GPU_TENSOR_BUNDLE),
        read_object(DEFAULT_ADOPTION_ENERGY_SCREEN),
        aggregate_hash=sha256(DEFAULT_AGGREGATE),
        trace_hash=sha256(DEFAULT_TRACE),
        phone_plan_hash=sha256(DEFAULT_PHONE_PLAN),
        gpu_capacity_hash=sha256(DEFAULT_GPU_CAPACITY),
        gpu_diagnostic_hash=sha256(DEFAULT_GPU_DIAGNOSTIC),
        gpu_tensor_bundle_hash=sha256(DEFAULT_GPU_TENSOR_BUNDLE),
        adoption_energy_screen_hash=sha256(DEFAULT_ADOPTION_ENERGY_SCREEN),
    )


class DynamicResidencyShadowTests(unittest.TestCase):
    def test_checked_in_shadow_matches_compiler(self) -> None:
        self.assertEqual(read_object(RESULT), build())

    def test_shadow_binds_static_fallback_and_equal_work(self) -> None:
        result = build()
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["mode"], "SHADOW_ONLY")
        self.assertEqual(
            result["baseline"]["accounted_fleet_energy_saving_pct_vs_cpu_overflow"],
            25.537300794962192,
        )
        self.assertEqual(result["workload"]["requests"], 74)
        self.assertEqual(result["workload"]["input_tokens"], 33_843)
        self.assertEqual(result["workload"]["output_tokens"], 11_605)
        self.assertEqual(
            result["workload"]["effective_model_requests"],
            {
                "gemma-4-12b-f16-proxy": 17,
                "qwen3-14b-f16-proxy": 57,
            },
        )
        self.assertEqual(
            result["workload"]["held_for_static_gemma_phase"], 17
        )

    def test_phone_capacity_keeps_mandatory_reserve(self) -> None:
        result = build()
        memory = result["phone_memory"]
        self.assertEqual(memory["resident_weight_bytes"], 9_673_170_944)
        self.assertEqual(
            memory["stageable_bytes_beyond_reserve"], 248_168_448
        )
        self.assertEqual(
            memory["stageable_shortfall_for_smallest_slice_bytes"],
            2_960_478_208,
        )
        diagnostic = memory["atomic_rotation_diagnostic"]
        self.assertEqual(
            diagnostic["strict_policy_reason"], "MEASUREMENT_REQUIRED"
        )
        self.assertEqual(
            diagnostic["capacity_only_reason"],
            "ATOMIC_STAGING_MEMORY",
        )
        self.assertFalse(result["gates"]["phone_fourth_known_slice_fits"])

    def test_missing_gpu_receipts_prevent_incremental_claim(self) -> None:
        result = build()
        self.assertEqual(
            result["gpu_shadow"]["status"],
            "BLOCKED_CONTENTION_ENERGY_REGRESSION",
        )
        self.assertIsNone(
            result["gpu_shadow"]["predicted_bubble_coverage_ppm"]
        )
        self.assertEqual(result["gpu_shadow"]["admitted_backfills"], 0)
        self.assertIsNone(
            result["shadow_accounting"]["incremental_energy_saving_j"]
        )

    def test_adoption_energy_screen_blocks_whole_request_overlap(self) -> None:
        result = build()
        screen = result["gpu_shadow"]["adoption_energy_screen_abba"]
        self.assertEqual(
            screen["admission"], "ENERGY_SCREEN_FAIL_FULL_TRACE_BLOCKED"
        )
        self.assertAlmostEqual(
            screen["changes"]["fleet_j_change_pct"], 15.325699423288919
        )
        self.assertAlmostEqual(
            screen["changes"]["duration_s_change_pct"], 24.162759100976672
        )
        self.assertTrue(result["gates"]["adoption_energy_screen_repeated"])
        self.assertTrue(
            result["gates"]["adoption_transition_full_boundary"]
        )
        self.assertFalse(result["gates"]["adoption_energy_screen_positive"])
        self.assertEqual(result["gpu_shadow"]["admitted_backfills"], 0)

    def test_dual_residency_capacity_is_not_an_energy_claim(self) -> None:
        result = build()
        candidate = result["gpu_shadow"]["capacity_candidate"]
        self.assertTrue(result["gates"]["dual_residency_capacity"])
        self.assertEqual(candidate["qwen_gpu_layers"], 15)
        self.assertEqual(candidate["gemma_gpu_layers"], 1)
        self.assertEqual(candidate["process_swap_max_bytes"], 0)
        self.assertGreater(candidate["free_beyond_reserve_bytes"], 0)
        self.assertEqual(candidate["status"], "CAPACITY_ONLY")
        self.assertFalse(result["gates"]["dynamic_incremental_energy_claim"])

    def test_exact_tensor_slices_expose_nonatomic_transition(self) -> None:
        result = build()
        slices = result["gpu_shadow"]["exact_tensor_slice_bytes"]
        constraint = result["gpu_shadow"][
            "physical_transition_constraint"
        ]
        self.assertTrue(result["gates"]["exact_gpu_tensor_slice_manifests"])
        self.assertFalse(result["gates"]["strict_atomic_gpu_staging"])
        self.assertTrue(
            result["gates"]["fallback_backed_transition_contract"]
        )
        self.assertFalse(result["gates"]["fallback_ready_route_receipt"])
        self.assertEqual(slices["qwen_gpu_18_raw_bytes"], 12_786_807_808)
        self.assertEqual(slices["gemma_gpu_1_raw_bytes"], 2_013_281_280)
        self.assertEqual(
            slices["transition"]["qwen_evicted_layer_ids"], [23, 24, 25]
        )
        self.assertEqual(
            constraint["staging_shortfall_bytes"], 2_238_709_760
        )
        self.assertTrue(
            constraint["gemma_after_qwen15_atomic_stage_fits"]
        )
        self.assertEqual(
            [
                row["mode"]
                for row in constraint["required_transition_sequence"]
            ],
            result["policy"]["transition_modes"][::-1],
        )

    def test_service_abba_is_bound_but_not_admitted(self) -> None:
        result = build()
        diagnostic = result["gpu_shadow"]["service_energy_abba"]
        self.assertEqual(
            diagnostic["status"],
            "REPEATED_SERVICE_DIRECTION_PASS_NO_ADMISSION",
        )
        self.assertAlmostEqual(
            diagnostic["changes"]["server_energy_pct"],
            -8.655677702193309,
        )
        self.assertAlmostEqual(
            diagnostic["changes"]["wall_service_pct"],
            -7.247610533708205,
        )
        self.assertTrue(result["gates"]["dual_residency_abba_repeated"])
        self.assertFalse(result["gates"]["dual_residency_exact_output"])
        self.assertTrue(
            result["gates"]["dual_residency_repeated_service_directional"]
        )
        self.assertFalse(
            diagnostic["screen_gates"]["qwen_first_token_not_regressed"]
        )
        self.assertIsNone(
            result["shadow_accounting"]["incremental_energy_saving_j"]
        )

    def test_record_hash_is_canonical(self) -> None:
        result = build()
        claimed = result.pop("record_sha256")
        self.assertEqual(
            claimed, hashlib.sha256(canonical(result)).hexdigest()
        )

    def test_tampered_work_or_aggregate_fails_closed(self) -> None:
        aggregate = read_object(DEFAULT_AGGREGATE)
        rows = read_trace(DEFAULT_TRACE)
        plan = PhoneResidencyPlan.from_json(read_object(DEFAULT_PHONE_PLAN))
        capacity = read_object(DEFAULT_GPU_CAPACITY)
        diagnostic = read_object(DEFAULT_GPU_DIAGNOSTIC)
        tensor_bundle = read_object(DEFAULT_GPU_TENSOR_BUNDLE)
        adoption_screen = read_object(DEFAULT_ADOPTION_ENERGY_SCREEN)
        changed_rows = deepcopy(rows)
        changed_rows[0]["output_tokens"] += 1
        with self.assertRaisesRegex(ShadowReplayError, "trace geometry"):
            build_shadow(
                aggregate,
                changed_rows,
                plan,
                capacity,
                diagnostic,
                tensor_bundle,
                adoption_screen,
                aggregate_hash=sha256(DEFAULT_AGGREGATE),
                trace_hash=sha256(DEFAULT_TRACE),
                phone_plan_hash=sha256(DEFAULT_PHONE_PLAN),
                gpu_capacity_hash=sha256(DEFAULT_GPU_CAPACITY),
                gpu_diagnostic_hash=sha256(DEFAULT_GPU_DIAGNOSTIC),
                gpu_tensor_bundle_hash=sha256(DEFAULT_GPU_TENSOR_BUNDLE),
                adoption_energy_screen_hash=sha256(
                    DEFAULT_ADOPTION_ENERGY_SCREEN
                ),
            )
        changed_aggregate = deepcopy(aggregate)
        changed_aggregate["status"] = "FAIL"
        with self.assertRaisesRegex(ShadowReplayError, "aggregate identity"):
            build_shadow(
                changed_aggregate,
                rows,
                plan,
                capacity,
                diagnostic,
                tensor_bundle,
                adoption_screen,
                aggregate_hash=sha256(DEFAULT_AGGREGATE),
                trace_hash=sha256(DEFAULT_TRACE),
                phone_plan_hash=sha256(DEFAULT_PHONE_PLAN),
                gpu_capacity_hash=sha256(DEFAULT_GPU_CAPACITY),
                gpu_diagnostic_hash=sha256(DEFAULT_GPU_DIAGNOSTIC),
                gpu_tensor_bundle_hash=sha256(DEFAULT_GPU_TENSOR_BUNDLE),
                adoption_energy_screen_hash=sha256(
                    DEFAULT_ADOPTION_ENERGY_SCREEN
                ),
            )
        changed_capacity = deepcopy(capacity)
        changed_capacity["configuration"]["qwen_gpu_layers"] = 14
        with self.assertRaisesRegex(
            ShadowReplayError, "capacity artifact identity"
        ):
            build_shadow(
                aggregate,
                rows,
                plan,
                changed_capacity,
                diagnostic,
                tensor_bundle,
                adoption_screen,
                aggregate_hash=sha256(DEFAULT_AGGREGATE),
                trace_hash=sha256(DEFAULT_TRACE),
                phone_plan_hash=sha256(DEFAULT_PHONE_PLAN),
                gpu_capacity_hash=sha256(DEFAULT_GPU_CAPACITY),
                gpu_diagnostic_hash=sha256(DEFAULT_GPU_DIAGNOSTIC),
                gpu_tensor_bundle_hash=sha256(DEFAULT_GPU_TENSOR_BUNDLE),
                adoption_energy_screen_hash=sha256(
                    DEFAULT_ADOPTION_ENERGY_SCREEN
                ),
            )


if __name__ == "__main__":
    unittest.main()
