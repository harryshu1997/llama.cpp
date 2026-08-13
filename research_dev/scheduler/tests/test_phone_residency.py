#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    DeviceMemoryCapacity,
    DynamicResidencySnapshot,
    DynamicWeightPlacement,
    DynamicWeightPlacementSpec,
    MetricEstimate,
    PhoneArmGroup,
    PhoneOffloadCandidate,
    PhoneResidencyError,
    PhoneResidencyPlan,
    PhoneResidencySnapshot,
    PhoneSessionPlan,
    PhoneSessionReceipt,
    ProfileBundle,
    ResidentPhoneSlice,
    UnifiedScheduleError,
    UnifiedScheduler,
    build_arm_signal,
    select_energy_positive_offload,
    validate_session_transition,
)


def digest(character: str) -> str:
    return "sha256:" + character * 64


def resident_slice(
    slice_id: str,
    weight_character: str,
    resident_bytes: int,
    physical_m_min: int = 1,
    physical_m_max: int = 512,
) -> ResidentPhoneSlice:
    return ResidentPhoneSlice(
        slice_id=slice_id,
        model_id="gemma-f16",
        model_hash=digest("a"),
        operator_family=slice_id,
        weight_hash=digest(weight_character),
        resident_bytes=resident_bytes,
        physical_m_min=physical_m_min,
        physical_m_max=physical_m_max,
        evidence_ids=("physical-op15-v1",),
    )


def plan() -> PhoneResidencyPlan:
    return PhoneResidencyPlan(
        plan_id="op15-three-session-v1",
        phone_serial="phone",
        memory_resource_id="op15-dram",
        shared_compute_resource_id="op15-htp",
        transport_resource_ids=("op15-functionfs", "desktop-usb-root"),
        memory_capacity_bytes=10_000,
        minimum_available_bytes=2_000,
        reset_generation=4,
        sessions=(
            PhoneSessionPlan(
                "htp0",
                "HTP0",
                3_200,
                (resident_slice("ffn", "b", 3_100),),
            ),
            PhoneSessionPlan(
                "htp1",
                "HTP1",
                3_200,
                (resident_slice("projections", "c", 2_200, 12),),
            ),
            PhoneSessionPlan(
                "htp2",
                "HTP2",
                3_200,
                (resident_slice("head", "d", 1_100, 1, 8),),
            ),
        ),
    )


def snapshot(
    residency_plan: PhoneResidencyPlan,
    *,
    state_overrides: dict[str, str] | None = None,
    mem_available_bytes: int = 2_500,
) -> PhoneResidencySnapshot:
    overrides = state_overrides or {}
    receipts = {}
    for index, session in enumerate(residency_plan.sessions, start=1):
        receipts[session.session_id] = PhoneSessionReceipt(
            session_id=session.session_id,
            compute_backend=session.compute_backend,
            state=overrides.get(session.session_id, "WARM"),
            generation=index,
            reset_generation=residency_plan.reset_generation,
            worker_hash=digest("e"),
            allocated_bytes=session.resident_bytes,
            slice_weight_hashes={
                row.slice_id: row.weight_hash for row in session.slices
            },
            last_transition_us=100,
        )
    return PhoneResidencySnapshot(
        snapshot_id="phone-snapshot-v1",
        plan_id=residency_plan.plan_id,
        captured_at_us=100,
        mem_available_bytes=mem_available_bytes,
        sessions=receipts,
    )


def scheduler_profile() -> ProfileBundle:
    return ProfileBundle.from_json({
        "schema": "s42-general-scheduler-profile-v1",
        "profile_id": "phone-residency-test",
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
                ("op15-htp", "phone_accelerator"),
                ("op15-functionfs", "phone_transport"),
                ("desktop-usb-root", "usb_root"),
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
            "evidence_ids": ["test-phone-residency"],
        }],
        "trace_workload_map": {"work": "work"},
        "policy": {
            "energy_saving_ppm": 50_000,
            "latency_limit_ppm": 2_000_000,
        },
    })


def dynamic_phone_snapshot(
    residency_plan: PhoneResidencyPlan,
) -> DynamicResidencySnapshot:
    placements = {}
    for index, session in enumerate(residency_plan.sessions, start=1):
        for row in session.slices:
            placement_id = f"{session.session_id}:{row.slice_id}"
            spec = DynamicWeightPlacementSpec(
                placement_id=placement_id,
                slice_id=row.slice_id,
                model_id=row.model_id,
                model_hash=row.model_hash,
                weight_hash=row.weight_hash,
                resource_id=residency_plan.memory_resource_id,
                resident_bytes=row.resident_bytes,
                execution_resource_ids=(
                    residency_plan.shared_compute_resource_id,
                ),
                runtime_binding_ids=(
                    session.session_id,
                    session.compute_backend,
                ),
                evidence_ids=row.evidence_ids,
            )
            placements[placement_id] = DynamicWeightPlacement(
                spec=spec,
                generation=1,
                resident_since_us=0,
                minimum_resident_until_us=0,
            )
    return DynamicResidencySnapshot(
        snapshot_id="dynamic-phone-snapshot-v1",
        epoch_key=digest("f"),
        generation=1,
        captured_at_us=100,
        valid_until_us=5_000,
        memory={
            residency_plan.memory_resource_id: DeviceMemoryCapacity(
                residency_plan.memory_resource_id,
                residency_plan.memory_capacity_bytes,
                7_500,
                residency_plan.minimum_available_bytes,
            ),
        },
        placements=placements,
    )


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


def candidate(
    candidate_id: str,
    slice_id: str,
    offload_units: int,
    *,
    host: tuple[int, int, int] = (800, 850, 750),
    phone: tuple[int, int, int] = (700, 780, 650),
    split_energy: tuple[int, int, int] = (760, 800, 720),
    additional_slice_ids: tuple[str, ...] = (),
    execution_mode: str = "parallel_split",
) -> PhoneOffloadCandidate:
    return PhoneOffloadCandidate(
        candidate_id=candidate_id,
        slice_id=slice_id,
        offload_units=offload_units,
        energy_boundary_id="cpu-package+gpu-board+whole-phone",
        accounting_scope="physical_ubatch",
        baseline_latency_us=metric(1_000, 1_100, 900),
        host_remainder_us=metric(*host),
        phone_path_us=metric(*phone),
        baseline_energy_uj=metric(1_000, 1_050, 950),
        split_energy_uj=metric(*split_energy),
        evidence_ids=("matched-energy-v1",),
        additional_slice_ids=additional_slice_ids,
        execution_mode=execution_mode,
    )


class PhoneResidencyTests(unittest.TestCase):
    def test_plan_round_trip_and_separate_mapping_limits(self) -> None:
        residency_plan = plan()
        self.assertEqual(residency_plan.resident_bytes, 6_400)
        self.assertEqual(
            PhoneResidencyPlan.from_json(residency_plan.to_json()),
            residency_plan,
        )

    def test_aggregate_memory_reserve_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            PhoneResidencyError, "violates memory reserve"
        ):
            PhoneResidencyPlan(
                plan_id="too-large",
                phone_serial="phone",
                memory_resource_id="op15-dram",
                shared_compute_resource_id="op15-htp",
                transport_resource_ids=(
                    "op15-functionfs",
                    "desktop-usb-root",
                ),
                memory_capacity_bytes=10_000,
                minimum_available_bytes=2_000,
                reset_generation=0,
                sessions=(
                    PhoneSessionPlan(
                        "htp0",
                        "HTP0",
                        9_000,
                        (resident_slice("oversized", "f", 8_500),),
                    ),
                ),
            )

    def test_per_session_mapping_limit_fails_closed(self) -> None:
        with self.assertRaisesRegex(PhoneResidencyError, "mapping limit"):
            PhoneSessionPlan(
                "htp0",
                "HTP0",
                3_200,
                (resident_slice("oversized", "f", 3_201),),
            )

    def test_all_sessions_arm_the_same_compute_resource(self) -> None:
        residency_plan = plan()
        residency_snapshot = snapshot(residency_plan)
        signals = [
            build_arm_signal(
                residency_plan,
                residency_snapshot,
                request_id="request",
                route_id="route",
                slice_id=slice_id,
                physical_m=physical_m,
                armed_at_us=200,
                execute_not_before_us=300,
                deadline_us=2_000,
            )
            for slice_id, physical_m in (
                ("ffn", 4),
                ("projections", 48),
                ("head", 4),
            )
        ]
        self.assertEqual(
            {row.shared_compute_resource_id for row in signals},
            {"op15-htp"},
        )
        self.assertEqual(
            [row.compute_backend for row in signals],
            ["HTP0", "HTP1", "HTP2"],
        )

    def test_arm_requires_a_warm_idle_session(self) -> None:
        residency_plan = plan()
        residency_snapshot = snapshot(
            residency_plan, state_overrides={"htp1": "EXECUTING"}
        )
        with self.assertRaisesRegex(PhoneResidencyError, "not warm"):
            build_arm_signal(
                residency_plan,
                residency_snapshot,
                request_id="request",
                route_id="route",
                slice_id="projections",
                physical_m=48,
                armed_at_us=200,
                execute_not_before_us=300,
                deadline_us=2_000,
            )

    def test_runtime_memory_floor_fails_closed(self) -> None:
        residency_plan = plan()
        with self.assertRaisesRegex(PhoneResidencyError, "memory reserve"):
            snapshot(
                residency_plan, mem_available_bytes=1_999
            ).validate_against(residency_plan)

    def test_snapshot_rejects_non_receipt_without_attribute_error(self) -> None:
        with self.assertRaisesRegex(PhoneResidencyError, "sessions are invalid"):
            PhoneResidencySnapshot(
                snapshot_id="invalid",
                plan_id="plan",
                captured_at_us=1,
                mem_available_bytes=1,
                sessions={"htp0": object()},
            )

    def test_session_state_machine(self) -> None:
        for source, target in (
            ("UNLOADED", "LOADING"),
            ("LOADING", "HASHED"),
            ("HASHED", "WARM"),
            ("WARM", "ARMED"),
            ("ARMED", "EXECUTING"),
            ("EXECUTING", "WARM"),
        ):
            validate_session_transition(source, target)
        with self.assertRaisesRegex(PhoneResidencyError, "invalid"):
            validate_session_transition("UNLOADED", "EXECUTING")

    def test_energy_first_selection_does_not_maximize_offload(self) -> None:
        residency_plan = plan()
        decision = select_energy_positive_offload(
            residency_plan,
            snapshot(residency_plan),
            (
                candidate(
                    "wider",
                    "ffn",
                    8_192,
                    split_energy=(790, 820, 750),
                ),
                candidate(
                    "balanced",
                    "ffn",
                    6_144,
                    split_energy=(760, 800, 720),
                ),
            ),
            request_id="request",
            route_id="route",
            physical_m=4,
            now_us=200,
            phone_resource_ready_us=200,
            deadline_us=2_000,
        )
        self.assertEqual(decision.candidate_id, "balanced")
        self.assertEqual(decision.offload_units, 6_144)
        assert decision.arm_signal is not None
        self.assertEqual(decision.arm_signal.session_id, "htp0")

    def test_composite_candidate_arms_disjoint_resident_sessions(self) -> None:
        residency_plan = plan()
        decision = select_energy_positive_offload(
            residency_plan,
            snapshot(residency_plan),
            (candidate(
                "qwen-full-layers",
                "ffn",
                12,
                additional_slice_ids=("head",),
            ),),
            request_id="request",
            route_id="qwen-full-ffn",
            physical_m=4,
            now_us=200,
            phone_resource_ready_us=200,
            deadline_us=2_000,
        )
        self.assertEqual(decision.candidate_id, "qwen-full-layers")
        self.assertIsInstance(decision.arm_signal, PhoneArmGroup)
        assert isinstance(decision.arm_signal, PhoneArmGroup)
        self.assertEqual(
            [signal.session_id for signal in decision.arm_signal.signals],
            ["htp0", "htp2"],
        )
        self.assertEqual(
            {signal.shared_compute_resource_id
             for signal in decision.arm_signal.signals},
            {"op15-htp"},
        )

    def test_full_replacement_uses_latency_gate_without_join_gate(self) -> None:
        residency_plan = plan()
        decision = select_energy_positive_offload(
            residency_plan,
            snapshot(residency_plan),
            (candidate(
                "qwen-full-replacement",
                "ffn",
                12,
                host=(1, 1, 1),
                phone=(700, 780, 650),
                execution_mode="full_replacement",
            ),),
            request_id="request",
            route_id="qwen-full-ffn",
            physical_m=1,
            now_us=200,
            phone_resource_ready_us=200,
            deadline_us=2_000,
        )
        self.assertEqual(decision.candidate_id, "qwen-full-replacement")
        self.assertEqual(decision.split_latency_upper_us, 780)
        self.assertEqual(decision.exposed_join_wait_upper_us, 0)

    def test_full_replacement_still_obeys_latency_gate(self) -> None:
        residency_plan = plan()
        decision = select_energy_positive_offload(
            residency_plan,
            snapshot(residency_plan),
            (candidate(
                "slow-full-replacement",
                "ffn",
                12,
                host=(1, 1, 1),
                phone=(1_100, 1_200, 1_000),
                execution_mode="full_replacement",
            ),),
            request_id="request",
            route_id="qwen-full-ffn",
            physical_m=1,
            now_us=200,
            phone_resource_ready_us=200,
            deadline_us=2_000,
        )
        self.assertIsNone(decision.candidate_id)
        self.assertIn(
            ("slow-full-replacement", "LATENCY_REGRESSION"),
            decision.rejected,
        )

    def test_unknown_execution_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(PhoneResidencyError, "execution mode"):
            candidate(
                "bad-mode",
                "ffn",
                12,
                execution_mode="unknown",
            )

    def test_queue_delay_rejects_exposed_phone_work(self) -> None:
        residency_plan = plan()
        decision = select_energy_positive_offload(
            residency_plan,
            snapshot(residency_plan),
            (candidate("queued", "ffn", 6_144),),
            request_id="request",
            route_id="route",
            physical_m=4,
            now_us=200,
            phone_resource_ready_us=700,
            deadline_us=2_000,
        )
        self.assertIsNone(decision.candidate_id)
        self.assertIn(
            decision.rejected[0][1],
            {"LATENCY_REGRESSION", "PHONE_EXPOSED"},
        )

    def test_candidate_specific_resource_readiness(self) -> None:
        residency_plan = plan()
        decision = select_energy_positive_offload(
            residency_plan,
            snapshot(residency_plan),
            (
                candidate("unavailable", "ffn", 6_144),
                candidate("ready", "ffn", 5_632),
            ),
            request_id="request",
            route_id="route",
            physical_m=4,
            now_us=200,
            phone_resource_ready_us={"unavailable": None, "ready": 200},
            deadline_us=2_000,
        )
        self.assertEqual(decision.candidate_id, "ready")
        self.assertIn(
            ("unavailable", "RESOURCE_NOT_READY"), decision.rejected
        )

    def test_unified_scheduler_leases_htp_and_usb_as_one_phone_path(self) -> None:
        residency_plan = plan()
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            phone_residency_plan=residency_plan,
            phone_residency_snapshot=snapshot(residency_plan),
        )
        first = unified.schedule_phone_offload(
            (candidate("ffn-call", "ffn", 6_144),),
            request_id="request-1",
            route_id="route",
            physical_m=4,
            now_us=200,
            deadline_us=3_000,
        )
        self.assertEqual(
            {lease.resource_id for lease in first.leases},
            {
                "op15-htp",
                "op15-functionfs",
                "desktop-usb-root",
            },
        )
        self.assertEqual({lease.start_us for lease in first.leases}, {200})
        self.assertEqual(
            {lease.reserved_until_us for lease in first.leases}, {980}
        )

        second = unified.schedule_phone_offload(
            (candidate("projection-call", "projections", 2_048),),
            request_id="request-2",
            route_id="route",
            physical_m=48,
            now_us=200,
            deadline_us=3_000,
            latency_limit_ppm=2_000_000,
            maximum_join_wait_ppm=1_000_000,
        )
        assert second.decision.arm_signal is not None
        self.assertEqual(
            second.decision.arm_signal.execute_not_before_us, 980
        )
        self.assertEqual(
            {lease.start_us for lease in second.leases}, {980}
        )

    def test_dynamic_phone_placement_is_leased_until_release(self) -> None:
        residency_plan = plan()
        dynamic = dynamic_phone_snapshot(residency_plan)
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            phone_residency_plan=residency_plan,
            phone_residency_snapshot=snapshot(residency_plan),
            dynamic_residency_snapshot=dynamic,
        )
        schedule = unified.schedule_phone_offload(
            (candidate("ffn-call", "ffn", 6_144),),
            request_id="request-1",
            route_id="route",
            physical_m=4,
            now_us=200,
            deadline_us=3_000,
        )
        self.assertEqual(schedule.placement_ids, ("htp0:ffn",))
        current = unified.dynamic_residency_snapshot
        assert current is not None
        self.assertEqual(
            current.placements["htp0:ffn"].active_leases, 1
        )
        with self.assertRaisesRegex(
            UnifiedScheduleError, "active placement leases"
        ):
            unified.schedule_dynamic_residency((), now_us=200)
        unified.release_phone_offload(schedule, 900)
        current = unified.dynamic_residency_snapshot
        assert current is not None
        self.assertEqual(
            current.placements["htp0:ffn"].active_leases, 0
        )
        with self.assertRaisesRegex(
            UnifiedScheduleError, "schedule is not active"
        ):
            unified.release_phone_offload(schedule, 900)

    def test_unified_scheduler_rejects_revoked_phone_resource(self) -> None:
        residency_plan = plan()
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            phone_residency_plan=residency_plan,
            phone_residency_snapshot=snapshot(residency_plan),
        )
        unified.set_resource_ready("op15-htp", False, 0)
        result = unified.schedule_phone_offload(
            (candidate("ffn-call", "ffn", 6_144),),
            request_id="request",
            route_id="route",
            physical_m=4,
            now_us=200,
            deadline_us=3_000,
        )
        self.assertIsNone(result.decision.candidate_id)
        self.assertEqual(result.leases, ())
        self.assertIn(
            ("ffn-call", "RESOURCE_NOT_READY"), result.decision.rejected
        )

    def test_missing_energy_lcb_cannot_enforce(self) -> None:
        residency_plan = plan()
        row = candidate("candidate", "ffn", 6_144)
        row = PhoneOffloadCandidate(
            candidate_id=row.candidate_id,
            slice_id=row.slice_id,
            offload_units=row.offload_units,
            energy_boundary_id=row.energy_boundary_id,
            accounting_scope=row.accounting_scope,
            baseline_latency_us=row.baseline_latency_us,
            host_remainder_us=row.host_remainder_us,
            phone_path_us=row.phone_path_us,
            baseline_energy_uj=MetricEstimate(1_000, 1_050, 3, True),
            split_energy_uj=row.split_energy_uj,
            evidence_ids=row.evidence_ids,
        )
        decision = select_energy_positive_offload(
            residency_plan,
            snapshot(residency_plan),
            (row,),
            request_id="request",
            route_id="route",
            physical_m=4,
            now_us=200,
            phone_resource_ready_us=200,
            deadline_us=2_000,
        )
        self.assertEqual(
            decision.rejected,
            (("candidate", "BASELINE_ENERGY_LCB_MISSING"),),
        )


if __name__ == "__main__":
    unittest.main()
