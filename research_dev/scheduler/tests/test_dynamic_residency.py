#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    DeviceMemoryCapacity,
    DynamicFallbackServiceContract,
    DynamicFallbackServiceReceipt,
    DynamicPlacementLease,
    DynamicResidencyCandidate,
    DynamicResidencyError,
    DynamicResidencyReceipt,
    DynamicResidencySnapshot,
    DynamicWeightPlacement,
    DynamicWeightPlacementSpec,
    MetricEstimate,
    ProfileBundle,
    UnifiedScheduleError,
    UnifiedScheduler,
    apply_dynamic_residency_receipt,
    canonical_sha256,
    select_dynamic_residency_transition,
)


def digest(character: str) -> str:
    return "sha256:" + character * 64


def metric(
    mean: int,
    upper: int,
    lower: int,
    *,
    measured: bool = True,
) -> MetricEstimate:
    return MetricEstimate(
        mean=mean,
        upper=upper,
        lower=lower,
        sample_count=3,
        measured=measured,
    )


def placement_spec(
    placement_id: str,
    model_id: str,
    character: str,
    resident_bytes: int,
    resource_id: str = "gpu-vram",
) -> DynamicWeightPlacementSpec:
    if resource_id == "gpu-vram":
        execution_resource = "gpu-compute"
        runtime_binding = "cuda0"
    elif resource_id == "desktop-ram":
        execution_resource = "cpu"
        runtime_binding = "numa0"
    else:
        execution_resource = "op15-htp"
        runtime_binding = "HTP0"
    return DynamicWeightPlacementSpec(
        placement_id=placement_id,
        slice_id=f"{model_id}-layers",
        model_id=model_id,
        model_hash=digest(character),
        weight_hash=digest(character),
        resource_id=resource_id,
        resident_bytes=resident_bytes,
        execution_resource_ids=(execution_resource,),
        runtime_binding_ids=(runtime_binding,),
        evidence_ids=(f"{model_id}-placement-v1",),
    )


def snapshot(
    *,
    capacity_bytes: int = 1_000,
    occupied_bytes: int = 600,
    reserve_bytes: int = 100,
    active_leases: int = 0,
    minimum_resident_until_us: int = 100,
) -> DynamicResidencySnapshot:
    old = DynamicWeightPlacement(
        spec=placement_spec("qwen-gpu", "qwen", "a", 400),
        generation=1,
        resident_since_us=0,
        minimum_resident_until_us=minimum_resident_until_us,
        active_leases=active_leases,
    )
    return DynamicResidencySnapshot(
        snapshot_id="snapshot-1",
        epoch_key=digest("e"),
        generation=1,
        captured_at_us=100,
        valid_until_us=10_000,
        memory={
            "gpu-vram": DeviceMemoryCapacity(
                "gpu-vram",
                capacity_bytes,
                occupied_bytes,
                reserve_bytes,
            ),
        },
        placements={old.placement_id: old},
    )


def candidate(
    source: DynamicResidencySnapshot,
    candidate_id: str = "gemma-prefetch",
    *,
    resident_bytes: int = 300,
    expected_reuse_count: int = 4,
    minimum_reuse_count: int = 2,
    baseline_energy: tuple[int, int, int] = (1_000, 1_050, 950),
    resident_energy: tuple[int, int, int] = (500, 550, 450),
    load_energy: tuple[int, int, int] = (100, 120, 90),
    eviction_energy: tuple[int, int, int] = (20, 30, 15),
    measured: bool = True,
) -> DynamicResidencyCandidate:
    return DynamicResidencyCandidate(
        candidate_id=candidate_id,
        source_snapshot_id=source.snapshot_id,
        source_snapshot_sha256=canonical_sha256(source.to_json()),
        source_generation=source.generation,
        source_epoch_key=source.epoch_key,
        target=placement_spec("gemma-gpu", "gemma", "b", resident_bytes),
        evict_placement_ids=("qwen-gpu",),
        transition_resource_ids=("gpu-dma",),
        expected_reuse_count=expected_reuse_count,
        minimum_reuse_count=minimum_reuse_count,
        minimum_residency_us=500,
        latest_ready_us=1_000,
        energy_boundary_id="cpu-package+gpu-board+whole-phone",
        accounting_scope="burstgpt-horizon",
        baseline_latency_us=metric(1_000, 1_100, 900, measured=measured),
        resident_latency_us=metric(500, 550, 450, measured=measured),
        load_latency_us=metric(100, 120, 90, measured=measured),
        eviction_latency_us=metric(50, 60, 40, measured=measured),
        baseline_energy_uj=metric(*baseline_energy, measured=measured),
        resident_energy_uj=metric(*resident_energy, measured=measured),
        load_energy_uj=metric(*load_energy, measured=measured),
        eviction_energy_uj=metric(*eviction_energy, measured=measured),
        evidence_ids=("matched-dynamic-residency-v1",),
    )


def fallback_snapshot() -> DynamicResidencySnapshot:
    base = snapshot(capacity_bytes=999)
    qwen_cpu = DynamicWeightPlacement(
        spec=placement_spec(
            "qwen-cpu", "qwen", "a", 200, "desktop-ram"
        ),
        generation=1,
        resident_since_us=0,
        minimum_resident_until_us=0,
    )
    gemma_phone = DynamicWeightPlacement(
        spec=placement_spec(
            "gemma-phone", "gemma", "b", 200, "op15-dram"
        ),
        generation=1,
        resident_since_us=0,
        minimum_resident_until_us=0,
    )
    return DynamicResidencySnapshot(
        snapshot_id=base.snapshot_id,
        epoch_key=base.epoch_key,
        generation=base.generation,
        captured_at_us=base.captured_at_us,
        valid_until_us=base.valid_until_us,
        memory={
            **base.memory,
            "desktop-ram": DeviceMemoryCapacity(
                "desktop-ram", 2_000, 500, 100
            ),
            "op15-dram": DeviceMemoryCapacity(
                "op15-dram", 2_000, 500, 100
            ),
        },
        placements={
            **base.placements,
            qwen_cpu.placement_id: qwen_cpu,
            gemma_phone.placement_id: gemma_phone,
        },
    )


def workspace_snapshot() -> DynamicResidencySnapshot:
    source = snapshot()
    return DynamicResidencySnapshot(
        snapshot_id=source.snapshot_id,
        epoch_key=source.epoch_key,
        generation=source.generation,
        captured_at_us=source.captured_at_us,
        valid_until_us=source.valid_until_us,
        memory={
            **source.memory,
            "desktop-ram": DeviceMemoryCapacity(
                "desktop-ram", 3_000, 500, 200
            ),
        },
        placements=source.placements,
    )


def fallback_contract(
    *,
    measured: bool = True,
    valid_until_us: int = 2_000,
) -> DynamicFallbackServiceContract:
    return DynamicFallbackServiceContract(
        fallback_id="cpu-op15-ready-route",
        route_id="qwen-cpu+gemma-op15",
        route_hash=digest("c"),
        placement_ids=("qwen-cpu", "gemma-phone"),
        execution_resource_ids=("cpu", "op15-htp"),
        model_hashes={"gemma": digest("b"), "qwen": digest("a")},
        ready_at_us=150,
        valid_until_us=valid_until_us,
        service_latency_us=metric(80, 100, 60, measured=measured),
        service_energy_uj=metric(30, 40, 20, measured=measured),
        restore_latency_us=metric(60, 80, 50, measured=measured),
        restore_energy_uj=metric(10, 20, 5, measured=measured),
        energy_boundary_id="cpu-package+gpu-board+whole-phone",
        accounting_scope="burstgpt-horizon",
        evidence_ids=("fallback-ready-and-restore-v1",),
    )


def fallback_candidate(
    source: DynamicResidencySnapshot,
    *,
    contract: DynamicFallbackServiceContract | None = None,
) -> DynamicResidencyCandidate:
    return replace(
        candidate(source),
        transition_resource_ids=("gpu-dma", "cpu", "op15-htp"),
        transition_mode=(
            "DRAIN_OR_EVICT_BEFORE_STAGE_WITH_READY_FALLBACK"
        ),
        fallback_contract=(
            fallback_contract() if contract is None else contract
        ),
    )


def select(
    source: DynamicResidencySnapshot,
    rows: tuple[DynamicResidencyCandidate, ...] | None = None,
):
    return select_dynamic_residency_transition(
        source,
        (candidate(source),) if rows is None else rows,
        now_us=200,
        transition_resource_ready_us=200,
    )


def result_snapshot(
    source: DynamicResidencySnapshot,
    decision,
    completed_at_us: int = 370,
) -> DynamicResidencySnapshot:
    assert decision.target is not None
    target = DynamicWeightPlacement(
        spec=decision.target,
        generation=decision.target_generation,
        resident_since_us=completed_at_us,
        minimum_resident_until_us=(
            completed_at_us + decision.target_minimum_residency_us
        ),
    )
    return DynamicResidencySnapshot(
        snapshot_id="snapshot-2",
        epoch_key=decision.target_epoch_key,
        generation=decision.target_generation,
        captured_at_us=completed_at_us,
        valid_until_us=20_000,
        memory={
            resource_id: DeviceMemoryCapacity(
                resource_id,
                source.memory[resource_id].capacity_bytes,
                occupied_bytes,
                source.memory[resource_id].reserve_bytes,
            )
            for resource_id, occupied_bytes
            in decision.occupied_bytes_after.items()
        },
        placements={
            **{
                placement_id: placement
                for placement_id, placement in source.placements.items()
                if placement_id not in decision.evict_placement_ids
            },
            target.placement_id: target,
        },
    )


def receipt(source, decision, result, status: str = "READY"):
    completed_at_us = (
        result.captured_at_us if status == "READY" else 250
    )
    return DynamicResidencyReceipt(
        transition_id=decision.transition_id,
        decision_sha256=decision.decision_sha256,
        status=status,
        source_snapshot_id=source.snapshot_id,
        source_generation=source.generation,
        source_epoch_key=source.epoch_key,
        started_at_us=200,
        completed_at_us=completed_at_us,
        result_snapshot=result,
        evidence_ids=("dynamic-transition-receipt-v1",),
        failure_reason=None if status == "READY" else "LOAD_FAILED",
        fallback_receipt=(
            None
            if decision.fallback_contract is None
            else DynamicFallbackServiceReceipt(
                fallback_id=decision.fallback_contract.fallback_id,
                contract_sha256=(
                    decision.fallback_contract.contract_sha256
                ),
                route_hash=decision.fallback_contract.route_hash,
                placement_ids=decision.fallback_contract.placement_ids,
                acquired_at_us=decision.transition_start_us,
                ready_at_us=decision.fallback_contract.ready_at_us,
                released_at_us=completed_at_us,
                served_request_ids=("request-42",),
                evidence_ids=("fallback-held-through-transition-v1",),
            )
        ),
    )


def scheduler_profile() -> ProfileBundle:
    return ProfileBundle.from_json({
        "schema": "s42-general-scheduler-profile-v1",
        "profile_id": "dynamic-residency-test",
        "resources": [
            {
                "resource_id": resource_id,
                "kind": kind,
                "capacity": 1,
                "ready": True,
                "identity": resource_id,
            }
            for resource_id, kind in (
                ("cpu", "cpu"),
                ("gpu-compute", "gpu"),
                ("gpu-dma", "gpu_dma"),
                ("op15-htp", "htp"),
            )
        ],
        "routes": [{
            "route_id": "cpu-baseline",
            "workload_id": "work",
            "granularity": "task",
            "baseline": True,
            "resource_slots": {"cpu": 1},
            "latency": {
                "cost_us": {
                    "kind": "affine_tokens_v1",
                    "fixed": 1_000,
                    "input_token": 0,
                    "output_token": 0,
                },
                "ucb_add_us": 0,
                "sample_count": 3,
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
            "evidence_ids": ["dynamic-residency-test"],
        }],
        "trace_workload_map": {"work": "work"},
        "policy": {
            "energy_saving_ppm": 50_000,
            "latency_limit_ppm": 2_000_000,
        },
    })
class DynamicResidencyTests(unittest.TestCase):
    def test_snapshot_and_candidate_round_trip(self) -> None:
        source = snapshot()
        row = candidate(source)
        self.assertEqual(
            DynamicResidencySnapshot.from_json(source.to_json()), source
        )
        self.assertEqual(
            DynamicResidencyCandidate.from_json(row.to_json()), row
        )

    def test_energy_positive_atomic_transition(self) -> None:
        source = snapshot()
        decision = select(source)
        self.assertEqual(decision.candidate_id, "gemma-prefetch")
        self.assertEqual(
            decision.reason, "ENERGY_POSITIVE_RESIDENCY_TRANSITION"
        )
        self.assertEqual(decision.ready_upper_us, 380)
        self.assertEqual(decision.energy_saving_lower_uj, 250)
        self.assertEqual(decision.energy_saving_ppm, 263_157)
        self.assertEqual(decision.occupied_bytes_after, {"gpu-vram": 500})
        self.assertEqual(
            [action.kind for action in decision.actions],
            ["PREFETCH", "VERIFY", "DRAIN", "EVICT", "PUBLISH"],
        )

    def test_atomic_staging_requires_peak_memory(self) -> None:
        source = snapshot(capacity_bytes=999)
        decision = select(source)
        self.assertIsNone(decision.candidate_id)
        self.assertEqual(
            decision.rejected,
            (("gemma-prefetch", "ATOMIC_STAGING_MEMORY"),),
        )

    def test_transition_workspace_uses_snapshot_memory(self) -> None:
        source = workspace_snapshot()
        fits = replace(
            candidate(source),
            transition_workspace_bytes={"desktop-ram": 2_300},
        )
        self.assertEqual(
            DynamicResidencyCandidate.from_json(fits.to_json()), fits
        )
        decision = select(source, (fits,))
        self.assertEqual(
            decision.transition_workspace_bytes,
            {"desktop-ram": 2_300},
        )
        self.assertEqual(
            decision.to_json()["transition_workspace_bytes"],
            {"desktop-ram": 2_300},
        )

        too_large = replace(
            fits, transition_workspace_bytes={"desktop-ram": 2_301}
        )
        rejected = select(source, (too_large,))
        self.assertEqual(
            rejected.rejected,
            (("gemma-prefetch", "TRANSITION_WORKSPACE_MEMORY"),),
        )

    def test_transition_workspace_resource_and_size_fail_closed(self) -> None:
        source = workspace_snapshot()
        missing = replace(
            candidate(source),
            transition_workspace_bytes={"unknown-ram": 1},
        )
        decision = select(source, (missing,))
        self.assertEqual(
            decision.rejected,
            ((
                "gemma-prefetch",
                "TRANSITION_MEMORY_RESOURCE_MISSING",
            ),),
        )
        with self.assertRaisesRegex(
            DynamicResidencyError, "workspace bytes"
        ):
            replace(
                candidate(source),
                transition_workspace_bytes={"desktop-ram": 0},
            )

    def test_fallback_backed_transition_uses_drain_first_order(self) -> None:
        source = fallback_snapshot()
        row = fallback_candidate(source)
        self.assertEqual(
            DynamicResidencyCandidate.from_json(row.to_json()), row
        )
        decision = select(source, (row,))
        self.assertEqual(decision.candidate_id, "gemma-prefetch")
        self.assertEqual(
            decision.reason,
            "ENERGY_POSITIVE_FALLBACK_BACKED_RESIDENCY_TRANSITION",
        )
        self.assertEqual(decision.ready_upper_us, 380)
        self.assertEqual(decision.recovery_upper_us, 460)
        self.assertEqual(decision.energy_saving_lower_uj, 190)
        self.assertEqual(
            [action.kind for action in decision.actions],
            [
                "FALLBACK_ACQUIRE",
                "FALLBACK_ACQUIRE",
                "DRAIN",
                "EVICT",
                "PREFETCH",
                "VERIFY",
                "PUBLISH",
                "FALLBACK_RELEASE",
                "FALLBACK_RELEASE",
            ],
        )

    def test_fallback_must_cover_each_affected_model(self) -> None:
        source = fallback_snapshot()
        incomplete = replace(
            fallback_contract(), model_hashes={"qwen": digest("a")}
        )
        decision = select(source, (fallback_candidate(
            source, contract=incomplete
        ),))
        self.assertEqual(
            decision.rejected,
            (("gemma-prefetch", "FALLBACK_MODEL_COVERAGE"),),
        )

    def test_fallback_resources_must_be_inside_transition_lease(self) -> None:
        source = fallback_snapshot()
        row = replace(
            fallback_candidate(source),
            transition_resource_ids=("gpu-dma",),
        )
        decision = select(source, (row,))
        self.assertEqual(
            decision.rejected,
            (("gemma-prefetch", "FALLBACK_RESOURCE_NOT_LEASED"),),
        )

    def test_fallback_measurements_and_recovery_window_fail_closed(self) -> None:
        source = fallback_snapshot()
        cases = (
            (
                fallback_contract(measured=False),
                "MEASUREMENT_REQUIRED",
            ),
            (
                fallback_contract(valid_until_us=459),
                "FALLBACK_EXPIRES_BEFORE_RECOVERY",
            ),
        )
        for contract, reason in cases:
            with self.subTest(reason=reason):
                decision = select(source, (fallback_candidate(
                    source, contract=contract
                ),))
                self.assertEqual(
                    decision.rejected,
                    (("gemma-prefetch", reason),),
                )

    def test_final_memory_capacity_is_checked(self) -> None:
        source = snapshot(capacity_bytes=1_200)
        decision = select(source, (candidate(source, resident_bytes=1_000),))
        self.assertIsNone(decision.candidate_id)
        self.assertIn(
            ("gemma-prefetch", "MEMORY_CAPACITY"), decision.rejected
        )

    def test_reuse_must_amortize_transition(self) -> None:
        source = snapshot()
        decision = select(source, (candidate(
            source, expected_reuse_count=1, minimum_reuse_count=2
        ),))
        self.assertEqual(
            decision.rejected,
            (("gemma-prefetch", "REUSE_NOT_AMORTIZED"),),
        )

    def test_eviction_hysteresis_and_leases_fail_closed(self) -> None:
        for source, reason in (
            (snapshot(minimum_resident_until_us=300), "EVICTION_HYSTERESIS"),
            (snapshot(active_leases=1), "EVICTION_LEASED"),
        ):
            with self.subTest(reason=reason):
                decision = select(source)
                self.assertIn(("gemma-prefetch", reason), decision.rejected)

    def test_transition_energy_is_inside_gate(self) -> None:
        source = snapshot()
        row = candidate(
            source,
            resident_energy=(700, 750, 650),
            load_energy=(150, 170, 140),
            eviction_energy=(50, 60, 40),
        )
        decision = select(source, (row,))
        self.assertEqual(
            decision.rejected,
            (("gemma-prefetch", "ENERGY_REGRESSION"),),
        )

    def test_unmeasured_candidate_fails_closed(self) -> None:
        source = snapshot()
        decision = select(source, (candidate(source, measured=False),))
        self.assertEqual(
            decision.rejected,
            (("gemma-prefetch", "MEASUREMENT_REQUIRED"),),
        )

    def test_stale_candidate_fails_closed(self) -> None:
        source = snapshot()
        row = replace(candidate(source), source_generation=2)
        decision = select(source, (row,))
        self.assertEqual(
            decision.rejected,
            (("gemma-prefetch", "SOURCE_EPOCH_MISMATCH"),),
        )

    def test_candidate_binds_exact_source_snapshot(self) -> None:
        source = snapshot()
        row = candidate(source)
        refreshed = replace(source, captured_at_us=1)
        decision = select(refreshed, (row,))
        self.assertEqual(
            decision.rejected,
            (("gemma-prefetch", "SOURCE_EPOCH_MISMATCH"),),
        )

    def test_snapshot_must_cover_complete_transition(self) -> None:
        source = replace(snapshot(), valid_until_us=380)
        decision = select(source)
        self.assertEqual(
            decision.rejected,
            (("gemma-prefetch", "SNAPSHOT_EXPIRES_BEFORE_READY"),),
        )
        with self.assertRaisesRegex(DynamicResidencyError, "expired"):
            select_dynamic_residency_transition(
                snapshot(),
                (candidate(snapshot()),),
                now_us=10_000,
                transition_resource_ready_us=10_000,
            )

    def test_energy_first_selection_chooses_larger_net_saving(self) -> None:
        source = snapshot()
        smaller = candidate(
            source,
            "smaller-saving",
            resident_energy=(600, 650, 550),
        )
        larger = candidate(source, "larger-saving")
        decision = select(source, (smaller, larger))
        self.assertEqual(decision.candidate_id, "larger-saving")

    def test_success_receipt_publishes_one_new_epoch(self) -> None:
        source = snapshot()
        decision = select(source)
        result = result_snapshot(source, decision)
        transition_receipt = receipt(source, decision, result)
        self.assertEqual(
            DynamicResidencyReceipt.from_json(
                transition_receipt.to_json()
            ),
            transition_receipt,
        )
        self.assertEqual(
            apply_dynamic_residency_receipt(
                source, decision, transition_receipt
            ),
            result,
        )

    def test_fallback_receipt_covers_success_and_failed_restore(self) -> None:
        source = fallback_snapshot()
        decision = select(source, (fallback_candidate(source),))
        result = result_snapshot(source, decision)
        successful = receipt(source, decision, result)
        self.assertEqual(
            DynamicResidencyReceipt.from_json(successful.to_json()),
            successful,
        )
        self.assertEqual(
            apply_dynamic_residency_receipt(source, decision, successful),
            result,
        )
        failed = receipt(source, decision, source, status="FAILED")
        self.assertIs(
            apply_dynamic_residency_receipt(source, decision, failed),
            source,
        )

    def test_fallback_receipt_identity_and_window_are_required(self) -> None:
        source = fallback_snapshot()
        decision = select(source, (fallback_candidate(source),))
        result = result_snapshot(source, decision)
        valid = receipt(source, decision, result)
        assert valid.fallback_receipt is not None
        cases = (
            (replace(valid, fallback_receipt=None), "omitted"),
            (
                replace(
                    valid,
                    fallback_receipt=replace(
                        valid.fallback_receipt, route_hash=digest("d")
                    ),
                ),
                "identity mismatch",
            ),
            (
                replace(
                    valid,
                    fallback_receipt=replace(
                        valid.fallback_receipt,
                        released_at_us=valid.completed_at_us + 1,
                    ),
                ),
                "does not cover",
            ),
        )
        for changed, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(DynamicResidencyError, reason):
                    apply_dynamic_residency_receipt(
                        source, decision, changed
                    )

    def test_failed_atomic_transition_preserves_source_epoch(self) -> None:
        source = snapshot()
        decision = select(source)
        failed = receipt(source, decision, source, status="FAILED")
        self.assertIs(
            apply_dynamic_residency_receipt(source, decision, failed),
            source,
        )

    def test_receipt_hash_and_source_generation_are_bound(self) -> None:
        source = snapshot()
        decision = select(source)
        result = result_snapshot(source, decision)
        valid = receipt(source, decision, result)
        for changed, reason in (
            (replace(valid, decision_sha256=digest("f")), "decision hash"),
            (replace(valid, source_generation=2), "receipt is stale"),
        ):
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(DynamicResidencyError, reason):
                    apply_dynamic_residency_receipt(
                        source, decision, changed
                    )

    def test_success_outside_timing_envelope_is_rejected(self) -> None:
        source = snapshot()
        decision = select(source)
        late_result = result_snapshot(source, decision, completed_at_us=381)
        late = receipt(source, decision, late_result)
        with self.assertRaisesRegex(DynamicResidencyError, "timing"):
            apply_dynamic_residency_receipt(source, decision, late)

    def test_unified_scheduler_leases_transition_resource(self) -> None:
        source = snapshot()
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        schedule = unified.schedule_dynamic_residency(
            (candidate(source),), now_us=200
        )
        self.assertEqual(schedule.decision.candidate_id, "gemma-prefetch")
        self.assertEqual(
            {lease.resource_id for lease in schedule.leases}, {"gpu-dma"}
        )
        self.assertEqual({lease.start_us for lease in schedule.leases}, {200})
        self.assertEqual(
            {lease.reserved_until_us for lease in schedule.leases}, {380}
        )
        result = result_snapshot(source, schedule.decision)
        self.assertEqual(
            unified.complete_dynamic_residency(
                schedule, receipt(source, schedule.decision, result)
            ),
            result,
        )
        self.assertEqual(unified.dynamic_residency_snapshot, result)

    def test_unified_scheduler_leases_fallback_through_recovery(self) -> None:
        source = fallback_snapshot()
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        schedule = unified.schedule_dynamic_residency(
            (fallback_candidate(source),), now_us=200
        )
        self.assertEqual(
            {lease.resource_id for lease in schedule.leases},
            {"cpu", "gpu-dma", "op15-htp"},
        )
        self.assertEqual(
            {lease.reserved_until_us for lease in schedule.leases},
            {460},
        )
        self.assertEqual(schedule.decision.ready_upper_us, 380)
        self.assertEqual(schedule.decision.recovery_upper_us, 460)
        result = result_snapshot(source, schedule.decision)
        self.assertEqual(
            unified.complete_dynamic_residency(
                schedule,
                receipt(source, schedule.decision, result),
            ),
            result,
        )

    def test_unified_scheduler_waits_for_fallback_ready_receipt(self) -> None:
        source = fallback_snapshot()
        contract = replace(fallback_contract(), ready_at_us=250)
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        schedule = unified.schedule_dynamic_residency(
            (fallback_candidate(source, contract=contract),), now_us=200
        )
        self.assertEqual({lease.start_us for lease in schedule.leases}, {250})
        self.assertEqual(
            {lease.reserved_until_us for lease in schedule.leases}, {510}
        )
        self.assertEqual(schedule.decision.ready_upper_us, 430)
        self.assertEqual(schedule.decision.recovery_upper_us, 510)

    def test_unified_scheduler_executes_two_epoch_capacity_sequence(self) -> None:
        source = fallback_snapshot()
        qwen_fallback = replace(
            fallback_contract(),
            placement_ids=("qwen-cpu",),
            execution_resource_ids=("cpu",),
            model_hashes={"qwen": digest("a")},
        )
        qwen_downsize = replace(
            candidate(source),
            target=placement_spec(
                "qwen-gpu-15", "qwen", "a", 300
            ),
            transition_resource_ids=("gpu-dma", "cpu"),
            transition_mode=(
                "DRAIN_OR_EVICT_BEFORE_STAGE_WITH_READY_FALLBACK"
            ),
            fallback_contract=qwen_fallback,
        )
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        first = unified.schedule_dynamic_residency(
            (qwen_downsize,), now_us=200
        )
        first_result = result_snapshot(source, first.decision)
        unified.complete_dynamic_residency(
            first,
            receipt(source, first.decision, first_result),
        )
        self.assertEqual(
            set(first_result.placements),
            {"gemma-phone", "qwen-cpu", "qwen-gpu-15"},
        )

        add_gemma = replace(
            candidate(first_result),
            evict_placement_ids=(),
        )
        second = unified.schedule_dynamic_residency(
            (add_gemma,), now_us=400
        )
        self.assertEqual(
            second.decision.transition_mode,
            "ATOMIC_STAGE_BEFORE_EVICT",
        )
        self.assertEqual(
            [action.kind for action in second.decision.actions],
            ["PREFETCH", "VERIFY", "PUBLISH"],
        )
        self.assertEqual(
            second.decision.occupied_bytes_after["gpu-vram"], 800
        )

    def test_pending_transition_binds_schedule_and_blocks_refresh(self) -> None:
        source = snapshot()
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        schedule = unified.schedule_dynamic_residency(
            (candidate(source),), now_us=200
        )
        refreshed = replace(source, captured_at_us=201)
        with self.assertRaisesRegex(
            UnifiedScheduleError, "blocked by a transition"
        ):
            unified.update_dynamic_residency(refreshed)
        result = result_snapshot(source, schedule.decision)
        transition_receipt = receipt(
            source, schedule.decision, result
        )
        with self.assertRaisesRegex(
            UnifiedScheduleError, "transition is not pending"
        ):
            unified.complete_dynamic_residency(
                replace(schedule, leases=()), transition_receipt
            )
        self.assertEqual(
            unified.complete_dynamic_residency(
                schedule, transition_receipt
            ),
            result,
        )

    def test_transition_waits_for_existing_dma_lease(self) -> None:
        source = snapshot()
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        unified.reserve_external_resource(
            "gpu-dma", "model-copy", 200, 300
        )
        schedule = unified.schedule_dynamic_residency(
            (candidate(source),), now_us=200
        )
        self.assertEqual(
            {lease.start_us for lease in schedule.leases}, {300}
        )
        self.assertEqual(schedule.decision.ready_upper_us, 480)

    def test_revoked_transition_resource_fails_closed(self) -> None:
        source = snapshot()
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        unified.set_resource_ready("gpu-dma", False, 0)
        schedule = unified.schedule_dynamic_residency(
            (candidate(source),), now_us=200
        )
        self.assertIsNone(schedule.decision.candidate_id)
        self.assertEqual(schedule.leases, ())
        self.assertIn(
            ("gemma-prefetch", "RESOURCE_NOT_READY"),
            schedule.decision.rejected,
        )

    def test_direct_epoch_change_requires_transition_receipt(self) -> None:
        source = snapshot()
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        decision = select(source)
        result = result_snapshot(source, decision)
        with self.assertRaisesRegex(
            UnifiedScheduleError, "requires a receipt"
        ):
            unified.update_dynamic_residency(result)

    def test_executor_placement_lease_blocks_transition(self) -> None:
        source = snapshot()
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        lease = unified.acquire_dynamic_placements(
            ("qwen-gpu",), owner_id="request-17", now_us=200
        )
        self.assertIsInstance(lease, DynamicPlacementLease)
        current = unified.dynamic_residency_snapshot
        assert current is not None
        self.assertEqual(current.placements["qwen-gpu"].active_leases, 1)
        with self.assertRaisesRegex(
            UnifiedScheduleError, "active placement leases"
        ):
            unified.schedule_dynamic_residency(
                (candidate(current),), now_us=200
            )
        unified.release_dynamic_placements(lease, 250)
        current = unified.dynamic_residency_snapshot
        assert current is not None
        self.assertEqual(current.placements["qwen-gpu"].active_leases, 0)
        with self.assertRaisesRegex(
            UnifiedScheduleError, "lease is not active"
        ):
            unified.release_dynamic_placements(lease, 250)

    def test_executor_placement_acquisition_is_fail_closed(self) -> None:
        source = snapshot()
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        with self.assertRaisesRegex(
            UnifiedScheduleError, "placement is absent"
        ):
            unified.acquire_dynamic_placements(
                ("absent",), owner_id="request-17", now_us=200
            )
        transition = unified.schedule_dynamic_residency(
            (candidate(source),), now_us=200
        )
        self.assertIsNotNone(transition.decision.candidate_id)
        with self.assertRaisesRegex(
            UnifiedScheduleError, "blocked by a residency transition"
        ):
            unified.acquire_dynamic_placements(
                ("qwen-gpu",), owner_id="request-18", now_us=200
            )

    def test_target_execution_resource_must_be_registered(self) -> None:
        source = snapshot()
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        row = candidate(source)
        row = replace(
            row,
            target=replace(
                row.target, execution_resource_ids=("absent-gpu",)
            ),
        )
        with self.assertRaisesRegex(
            UnifiedScheduleError, "target execution resource is absent"
        ):
            unified.schedule_dynamic_residency((row,), now_us=200)


if __name__ == "__main__":
    unittest.main()
