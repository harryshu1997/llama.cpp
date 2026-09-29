#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[4]
S42_ROOT = REPO_ROOT / "research_dev/spikes/s42_general_energy_scheduler_v1"
SCHEDULER_TEST_ROOT = REPO_ROOT / "research_dev/scheduler/tests"
sys.path[:0] = [
    str(SCHEDULER_TEST_ROOT),
    str(S42_ROOT / "tests"),
    str(S42_ROOT),
    str(REPO_ROOT),
]

import test_policy as legacy  # noqa: E402
import research_dev.scheduler as unified_scheduler  # noqa: E402
from research_dev.scheduler import (  # noqa: E402
    AccountingContract,
    AccountingKind,
    AccountingScope,
    CandidateSet,
    EnergyComponent,
    FeatureRange,
    MetricEstimate,
    ModelIdentity,
    QualityClass,
    Request,
    ResourceRequirement,
    RouteMaturity,
    RuntimeResourceRequirement,
    SchedulerContractError,
    UnitKind,
    adapt_legacy_profile,
    adapt_legacy_request,
    canonical_json,
    make_work_set_hash,
)


MODEL = ModelIdentity(
    model_id="gemma4-12b",
    model_hash="sha256:" + "a" * 64,
    architecture="gemma4",
    weight_format="q8_0",
    weight_bytes=12_000_000_000,
)


def legacy_bundle(*, energy_status: str = "measured"):
    return legacy.profile([
        legacy.baseline(status=energy_status),
        legacy.offload(status=energy_status),
    ])


class LegacyGoldenTests(unittest.TestCase):
    def test_stage0_policy_matrix_is_frozen(self) -> None:
        expected = {
            "capacity": ("offload", "MINIMUM_SERVER_CAPACITY_COST", 0, 900, 80),
            "control": ("baseline", "CONTROL_BASELINE", 0, 1000, 100),
            "enforce": ("offload", "VERIFIED_ENERGY_SAVING", 0, 900, 80),
            "shadow": ("offload", "SHADOW_FASTEST_QUALIFIED", 0, 900, 80),
        }
        actual = {}
        for mode in sorted(expected):
            scheduler = unified_scheduler.UnifiedScheduler(
                (legacy_bundle(),), mode
            )
            decision = scheduler.schedule(legacy.request())
            actual[mode] = (
                decision.route_id,
                decision.reason,
                decision.start_us,
                decision.finish_us,
                decision.energy_uj,
            )
        self.assertEqual(actual, expected)

    def test_work_set_hash_encoding_is_frozen(self) -> None:
        self.assertEqual(
            make_work_set_hash(("r0",)),
            "sha256:e184c2c729bccd7bf7b04e0ce9d3031b952413b0751d8dfc8925645e4cab9b44",
        )


class MixedTraceIntegrationTests(unittest.TestCase):
    MODEL_COUNTS = {
        "gemma-4-12b-it-q4_0": 17,
        "gemma-4-e2b-it-q8_0-vlm": 10,
        "llama-3.2-1b-instruct-q4_0": 10,
        "qwen3-0.6b-q8_0": 10,
        "qwen3-14b-q4_k_m": 57,
        "qwen3-8b-q8_0": 10,
    }
    GPU_MODELS = {
        "gemma-4-12b-it-q4_0",
        "qwen3-14b-q4_k_m",
        "qwen3-8b-q8_0",
    }

    def profile(self) -> unified_scheduler.ProfileBundle:
        routes = []
        for model_id in sorted(self.MODEL_COUNTS):
            resource_id = "gpu" if model_id in self.GPU_MODELS else "cpu"
            routes.append({
                "route_id": f"{model_id}-baseline",
                "workload_id": model_id,
                "granularity": "task",
                "baseline": True,
                "resource_slots": {resource_id: 1},
                "latency": {
                    "cost_us": {
                        "kind": "affine_tokens_v1",
                        "fixed": 1000,
                        "input_token": 50,
                        "output_token": 100,
                    },
                    "ucb_add_us": 0,
                    "sample_count": 1,
                    "measured": True,
                },
                "energy": {
                    "status": "unknown",
                    "cost_uj": None,
                    "lower_error_ppm": 0,
                    "upper_error_ppm": 0,
                },
                "overlap": {"status": "not_applicable"},
                "quality_class": "exact",
                "placement_verified": True,
                "resident": True,
                "server_busy_ppm": 1_000_000,
                "server_memory_bytes": 1,
                "evidence_ids": ["test-only-mixed-trace-profile"],
            })
        return unified_scheduler.ProfileBundle.from_json({
            "schema": unified_scheduler.PROFILE_SCHEMA,
            "profile_id": "test-only-mixed-trace-profile",
            "resources": [
                {
                    "resource_id": "cpu",
                    "kind": "cpu",
                    "capacity": 3,
                    "ready": True,
                    "identity": "test-cpu",
                },
                {
                    "resource_id": "gpu",
                    "kind": "gpu",
                    "capacity": 1,
                    "ready": True,
                    "identity": "test-gpu",
                },
            ],
            "trace_workload_map": {
                model_id: model_id for model_id in self.MODEL_COUNTS
            },
            "policy": {},
            "routes": routes,
        })

    def replay(self) -> list[unified_scheduler.Decision]:
        path = S42_ROOT / "mixed_model_trace_v1/REQUESTS_MIXED_114.jsonl"
        trace = unified_scheduler.load_mixed_model_trace(
            path,
            {model_id: model_id for model_id in self.MODEL_COUNTS},
        )
        scheduler = unified_scheduler.UnifiedScheduler(
            (self.profile(),), "control"
        )
        return [scheduler.schedule(request) for request in trace.requests]

    def test_unified_scheduler_replays_all_mixed_requests(self) -> None:
        first = self.replay()
        second = self.replay()
        self.assertEqual(len(first), 114)
        self.assertEqual(
            Counter(decision.route_id for decision in first),
            Counter({
                f"{model_id}-baseline": count
                for model_id, count in self.MODEL_COUNTS.items()
            }),
        )
        self.assertEqual(
            [unified_scheduler.decision_to_json(decision) for decision in first],
            [unified_scheduler.decision_to_json(decision) for decision in second],
        )


class CanonicalCompatibilityTests(unittest.TestCase):
    def test_legacy_profile_materializes_exact_candidate_set(self) -> None:
        request = legacy.request()
        unit = adapt_legacy_request(request, MODEL)
        candidates = adapt_legacy_profile(legacy_bundle(), request, unit)
        routes = {route.route_id: route for route in candidates.routes}

        self.assertEqual(candidates.unit, unit)
        self.assertEqual(routes["baseline"].latency_us.mean, 1000)
        self.assertEqual(routes["offload"].latency_us.mean, 900)
        self.assertEqual(routes["offload"].energy_uj.mean, 80)
        self.assertEqual(
            {resource.resource_id for resource in routes["offload"].resources},
            {"phone", "server", "usb"},
        )
        self.assertEqual(
            {lease.resource_id for lease in routes["offload"].phase_leases},
            {"phone", "server", "usb"},
        )
        self.assertEqual(routes["offload"].rejection_reason(unit), None)

    def test_legacy_unknown_energy_remains_unknown(self) -> None:
        request = legacy.request()
        unit = adapt_legacy_request(request, MODEL)
        candidates = adapt_legacy_profile(
            legacy_bundle(energy_status="unknown"), request, unit
        )
        self.assertTrue(all(not route.energy for route in candidates.routes))
        self.assertTrue(
            all(route.maturity == RouteMaturity.PREDICTED for route in candidates.routes)
        )

    def test_gpu_route_memory_is_separate_from_cpu_compute(self) -> None:
        request = legacy.request()
        unit = adapt_legacy_request(request, MODEL)
        profile = legacy.profile([
            legacy.gpu_baseline(),
            legacy.task_route(
                "phone",
                latency_us=800,
                energy_uj=80,
                resources={"phone": 1, "usb": 1},
            ),
        ])
        candidates = adapt_legacy_profile(profile, request, unit)
        gpu = candidates.routes[0]
        self.assertEqual(
            {resource.resource_id for resource in gpu.resources}, {"gpu"}
        )
        self.assertEqual(
            {(memory.resource_id, memory.pool_id) for memory in gpu.memory},
            {("gpu", "device-memory")},
        )

    def test_measured_bge_profile_adapts_without_policy_changes(self) -> None:
        path = (
            S42_ROOT
            / "small_model_phone_v1/results/4060ti_op15_20260808"
            / "SCHEDULER_PROFILE_CUDA_EPOCH_OPEN.json"
        )
        _, profile = unified_scheduler.load_profile_bundle(path)
        request = unified_scheduler.Request(
            request_id="bge-stage1",
            workload_id="bge-small-en-v1.5-q8-batch32-resident",
            arrival_us=0,
            deadline_us=1_000_000_000,
            input_tokens=1088,
            output_tokens=1,
            quality_requirement="bounded_numeric",
            features={"batch32_groups": 1},
            semantics=unified_scheduler.RequestSemantics(
                kv_owner="none",
                full_logits_required=False,
                sampler_location="none",
            ),
        )
        model = replace(
            MODEL,
            model_id="bge-small-en-v1.5",
            architecture="bert",
            weight_bytes=36_685_152,
        )
        unit = adapt_legacy_request(request, model)
        candidates = adapt_legacy_profile(profile, request, unit)
        routes = {route.route_id: route for route in candidates.routes}
        self.assertEqual(set(routes), {"desktop-cuda", "phone-adreno"})
        self.assertEqual(
            {(item.resource_id, item.pool_id) for item in routes["desktop-cuda"].memory},
            {("cuda0", "device-memory")},
        )
        self.assertEqual(
            {item.resource_id for item in routes["phone-adreno"].residency},
            {"op15-adreno", "usb-token-rpc"},
        )

    def test_exact_applicability_rejects_new_shape(self) -> None:
        request = legacy.request()
        unit = adapt_legacy_request(request, MODEL)
        route = adapt_legacy_profile(legacy_bundle(), request, unit).routes[0]
        changed = Request(
            "r0",
            "work",
            0,
            10_000,
            11,
            2,
            "exact",
            semantics=request.semantics,
        )
        changed_unit = adapt_legacy_request(changed, MODEL)
        self.assertEqual(
            route.rejection_reason(changed_unit), "FEATURE_OUT_OF_DOMAIN"
        )

    def test_adapter_rejects_request_unit_shape_mismatch(self) -> None:
        request = legacy.request()
        unit = adapt_legacy_request(request, MODEL)
        changed = Request(
            "r0",
            "work",
            0,
            10_000,
            11,
            2,
            "exact",
            semantics=request.semantics,
        )
        with self.assertRaisesRegex(
            SchedulerContractError, "does not describe"
        ):
            adapt_legacy_profile(legacy_bundle(), changed, unit)

    def test_exact_applicability_rejects_model_hash_change(self) -> None:
        request = legacy.request()
        unit = adapt_legacy_request(request, MODEL)
        route = adapt_legacy_profile(legacy_bundle(), request, unit).routes[0]
        changed_model = replace(MODEL, model_hash="sha256:" + "b" * 64)
        changed_unit = adapt_legacy_request(request, changed_model)
        self.assertEqual(
            route.rejection_reason(changed_unit), "MODEL_HASH_OUT_OF_DOMAIN"
        )

    def test_semantic_quality_accepts_numeric_variability(self) -> None:
        routes = [
            legacy.baseline(),
            legacy.offload(quality="approximate"),
        ]
        request = legacy.request(quality="approximate")
        unit = adapt_legacy_request(request, MODEL)
        candidate = adapt_legacy_profile(
            legacy.profile(routes), request, unit
        ).routes[1]
        self.assertEqual(candidate.quality.quality_class, QualityClass.SEMANTIC)
        self.assertTrue(candidate.quality.finite_output_required)
        self.assertTrue(candidate.quality.nonempty_output_required)
        self.assertIsNone(candidate.rejection_reason(unit))

    def test_legacy_non_request_route_requires_explicit_accounting(self) -> None:
        request = legacy.request()
        unit = adapt_legacy_request(
            request,
            MODEL,
            unit_kind=UnitKind.COHORT,
            member_request_ids=("r0", "r1"),
            unit_id="cohort-0",
        )
        with self.assertRaisesRegex(
            SchedulerContractError, "explicit accounting"
        ):
            adapt_legacy_profile(legacy_bundle(), request, unit)

    def test_non_additive_cohort_is_bound_to_exact_work_set(self) -> None:
        request = legacy.request()
        unit = adapt_legacy_request(
            request,
            MODEL,
            unit_kind=UnitKind.COHORT,
            member_request_ids=("r0", "r1"),
            unit_id="cohort-0",
        )
        accounting = AccountingContract(
            scope=AccountingScope.COHORT,
            kind=AccountingKind.NON_ADDITIVE_COHORT_TOTAL,
            boundary_id="fleet-cohort-boundary-v1",
            work_set_hash=unit.work_set_hash,
        )
        profile = legacy_bundle()
        candidates = adapt_legacy_profile(
            profile,
            request,
            unit,
            accounting_by_route={
                route.route_id: accounting for route in profile.routes
            },
        )
        self.assertTrue(
            all(route.rejection_reason(unit) is None for route in candidates.routes)
        )
        other_unit = replace(
            unit,
            member_request_ids=("r0", "r2"),
            work_set_hash=make_work_set_hash(("r0", "r2")),
        )
        self.assertEqual(
            candidates.routes[0].rejection_reason(other_unit),
            "WORK_SET_OUT_OF_DOMAIN",
        )

    def test_candidate_set_rejects_cross_scope_accounting(self) -> None:
        request = legacy.request()
        unit = adapt_legacy_request(request, MODEL)
        candidates = adapt_legacy_profile(legacy_bundle(), request, unit)
        route = candidates.routes[0]
        wrong = AccountingContract(
            scope=AccountingScope.PHYSICAL_UBATCH,
            kind=AccountingKind.EXCLUSIVE_UNIT_TOTAL,
            boundary_id="fleet-request-boundary-v1",
            work_set_hash=unit.work_set_hash,
        )
        changed = replace(
            route,
            energy=(EnergyComponent(route.energy[0].estimate_uj, wrong),),
        )
        with self.assertRaisesRegex(
            SchedulerContractError, "ACCOUNTING_SCOPE_MISMATCH"
        ):
            CandidateSet("bad-scope", unit, (changed, candidates.routes[1]))

    def test_route_can_separate_request_energy_and_cuda_tail(self) -> None:
        request = legacy.request()
        unit = adapt_legacy_request(request, MODEL, epoch_id="cuda-epoch-7")
        candidates = adapt_legacy_profile(legacy_bundle(), request, unit)
        route = candidates.routes[0]
        tail = EnergyComponent(
            estimate_uj=MetricEstimate(313_015_000, 330_000_000, 3, True),
            accounting=AccountingContract(
                scope=AccountingScope.EPOCH,
                kind=AccountingKind.EPOCH_OBLIGATION,
                boundary_id="physical-boundary-1",
                epoch_id="cuda-epoch-7",
            ),
        )
        changed = replace(route, energy=route.energy + (tail,))
        self.assertIsNone(changed.rejection_reason(unit))
        self.assertEqual(changed.energy_uj.mean, 313_015_100)

    def test_route_energy_components_require_one_boundary(self) -> None:
        request = legacy.request()
        unit = adapt_legacy_request(request, MODEL, epoch_id="cuda-epoch-7")
        route = adapt_legacy_profile(legacy_bundle(), request, unit).routes[0]
        tail = EnergyComponent(
            MetricEstimate(100, 120, 1, True),
            AccountingContract(
                AccountingScope.EPOCH,
                AccountingKind.EPOCH_OBLIGATION,
                "different-boundary",
                epoch_id="cuda-epoch-7",
            ),
        )
        with self.assertRaisesRegex(SchedulerContractError, "one fleet boundary"):
            replace(route, energy=route.energy + (tail,))

    def test_candidate_routes_require_one_energy_boundary(self) -> None:
        request = legacy.request()
        unit = adapt_legacy_request(request, MODEL)
        candidates = adapt_legacy_profile(legacy_bundle(), request, unit)
        candidate = candidates.routes[1]
        changed_accounting = replace(
            candidate.energy[0].accounting,
            boundary_id="different-fleet-boundary",
        )
        changed = replace(
            candidate,
            energy=(EnergyComponent(
                candidate.energy[0].estimate_uj,
                changed_accounting,
            ),),
        )
        with self.assertRaisesRegex(
            SchedulerContractError, "different fleet-energy boundaries"
        ):
            CandidateSet(
                candidates.profile_id,
                unit,
                (candidates.routes[0], changed),
            )

    def test_canonical_encoding_is_deterministic(self) -> None:
        request = legacy.request()
        unit = adapt_legacy_request(request, MODEL)
        left = canonical_json(unit)
        right = canonical_json(replace(unit, features=dict(reversed(
            list(unit.features.items())
        ))))
        self.assertEqual(left, right)

    def test_feature_ranges_reject_empty_intervals(self) -> None:
        with self.assertRaisesRegex(SchedulerContractError, "empty"):
            FeatureRange(10, 9)


if __name__ == "__main__":
    unittest.main()
