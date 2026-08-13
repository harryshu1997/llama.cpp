#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    MetricEstimate,
    PhoneArbiterError,
    PhoneArbiterQueue,
    PhoneArbiterWindow,
    PhoneOffloadCandidate,
    PhoneResidencyPlan,
    PhoneResidencySnapshot,
    PhoneSessionPlan,
    PhoneSessionReceipt,
    PhoneReadyWork,
    ProfileBundle,
    ResidentPhoneSlice,
    UnifiedScheduleError,
    UnifiedScheduler,
    canonical_sha256,
    select_phone_arbiter_work,
)


def digest(character: str) -> str:
    return "sha256:" + character * 64


def metric(mean: int, upper: int, lower: int) -> MetricEstimate:
    return MetricEstimate(
        mean=mean,
        upper=upper,
        lower=lower,
        sample_count=3,
        measured=True,
    )


def resident_slice(
    slice_id: str,
    model_id: str,
    character: str,
) -> ResidentPhoneSlice:
    return ResidentPhoneSlice(
        slice_id=slice_id,
        model_id=model_id,
        model_hash=digest("a" if model_id == "qwen" else "b"),
        operator_family="dense-ffn",
        weight_hash=digest(character),
        resident_bytes=3_000,
        physical_m_min=1,
        physical_m_max=16,
        evidence_ids=(slice_id + "-physical",),
    )


def residency_plan() -> PhoneResidencyPlan:
    return PhoneResidencyPlan(
        plan_id="op15-qwen-gemma-v1",
        phone_serial="op15",
        memory_resource_id="op15-dram",
        shared_compute_resource_id="op15-htp",
        transport_resource_ids=("op15-functionfs", "desktop-usb-root"),
        memory_capacity_bytes=10_000,
        minimum_available_bytes=1_000,
        reset_generation=4,
        sessions=(
            PhoneSessionPlan(
                "htp0",
                "HTP0",
                3_200,
                (resident_slice("gemma-ffn", "gemma", "c"),),
            ),
            PhoneSessionPlan(
                "htp1",
                "HTP1",
                3_200,
                (resident_slice("qwen-low", "qwen", "d"),),
            ),
            PhoneSessionPlan(
                "htp2",
                "HTP2",
                3_200,
                (resident_slice("qwen-high", "qwen", "e"),),
            ),
        ),
    )


def residency_snapshot(plan: PhoneResidencyPlan) -> PhoneResidencySnapshot:
    receipts = {
        session.session_id: PhoneSessionReceipt(
            session_id=session.session_id,
            compute_backend=session.compute_backend,
            state="WARM",
            generation=index,
            reset_generation=plan.reset_generation,
            worker_hash=digest("f"),
            allocated_bytes=session.resident_bytes,
            slice_weight_hashes={
                row.slice_id: row.weight_hash for row in session.slices
            },
            last_transition_us=50,
        )
        for index, session in enumerate(plan.sessions, start=1)
    }
    return PhoneResidencySnapshot(
        snapshot_id="op15-snapshot-1",
        plan_id=plan.plan_id,
        captured_at_us=100,
        mem_available_bytes=1_500,
        sessions=receipts,
    )


def candidate(
    work_id: str,
    slice_id: str,
    *,
    additional: tuple[str, ...] = (),
    phone_upper_us: int = 40,
    execution_mode: str = "parallel_split",
) -> PhoneOffloadCandidate:
    return PhoneOffloadCandidate(
        candidate_id=work_id,
        slice_id=slice_id,
        additional_slice_ids=additional,
        offload_units=1,
        execution_mode=execution_mode,
        energy_boundary_id="cpu-package+gpu-board+whole-phone",
        accounting_scope="fp16-burstgpt",
        baseline_latency_us=metric(100, 110, 90),
        host_remainder_us=metric(45, 50, 40),
        phone_path_us=metric(30, phone_upper_us, 25),
        baseline_energy_uj=metric(1_000, 1_050, 950),
        split_energy_uj=metric(650, 700, 600),
        evidence_ids=(work_id + "-energy",),
    )


def work(
    work_id: str,
    model_id: str,
    priority: str,
    *,
    sequence_index: int = 0,
    predecessor: str | None = None,
    phone_upper_us: int = 40,
    valid_until_us: int = 270,
) -> PhoneReadyWork:
    is_qwen = model_id == "qwen"
    return PhoneReadyWork(
        work_id=work_id,
        pipeline_id=model_id + "-request",
        request_id=model_id + "-request-1",
        route_id=model_id + "-phone-route",
        model_id=model_id,
        priority_class=priority,
        sequence_index=sequence_index,
        predecessor_output_receipt_id=predecessor,
        ready_receipt_id=work_id + "-ready",
        ready_at_us=110,
        valid_until_us=valid_until_us,
        deadline_us=1_000,
        physical_m=1,
        runtime_verified=True,
        candidate=candidate(
            work_id,
            "qwen-low" if is_qwen else "gemma-ffn",
            additional=("qwen-high",) if is_qwen else (),
            phone_upper_us=phone_upper_us,
            execution_mode="full_replacement" if is_qwen else "parallel_split",
        ),
        evidence_ids=(work_id + "-ready-evidence",),
    )


def window(
    snapshot: PhoneResidencySnapshot,
    plan: PhoneResidencyPlan,
    **overrides: object,
) -> PhoneArbiterWindow:
    values = {
        "window_id": "qwen-gap-1",
        "phone_snapshot_id": snapshot.snapshot_id,
        "phone_snapshot_sha256": canonical_sha256(snapshot.to_json()),
        "plan_id": plan.plan_id,
        "reset_generation": plan.reset_generation,
        "protected_owner_id": "qwen-request-1",
        "protected_model_id": "qwen",
        "captured_at_us": 100,
        "valid_until_us": 280,
        "protected_ready_lower_us": 300,
        "guard_us": 20,
        "desktop_memory_available_bytes": 4_000,
        "desktop_memory_reserve_bytes": 2_000,
        "swap_in_delta_pages": 0,
        "swap_out_delta_pages": 0,
        "maximum_swap_io_pages": 0,
        "duplicate_hot_slice_ids": (),
        "runtime_verified": True,
        "evidence_ids": ("physical-gap-v1",),
    }
    values.update(overrides)
    return PhoneArbiterWindow(**values)


def queue(
    snapshot: PhoneResidencySnapshot,
    plan: PhoneResidencyPlan,
    rows: tuple[PhoneReadyWork, ...],
    *,
    next_by_pipeline: dict[str, int] | None = None,
    completed: tuple[str, ...] = (),
    queue_id: str = "phone-queue-1",
    captured_at_us: int = 100,
    valid_until_us: int = 280,
) -> PhoneArbiterQueue:
    return PhoneArbiterQueue(
        queue_id=queue_id,
        phone_snapshot_id=snapshot.snapshot_id,
        phone_snapshot_sha256=canonical_sha256(snapshot.to_json()),
        plan_id=plan.plan_id,
        reset_generation=plan.reset_generation,
        captured_at_us=captured_at_us,
        valid_until_us=valid_until_us,
        next_sequence_by_pipeline=(
            next_by_pipeline
            if next_by_pipeline is not None
            else {row.pipeline_id: row.sequence_index for row in rows}
        ),
        completed_output_receipt_ids=completed,
        ready_work=rows,
    )


def scheduler_profile() -> ProfileBundle:
    return ProfileBundle.from_json({
        "schema": "s42-general-scheduler-profile-v1",
        "profile_id": "phone-arbiter-test",
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
            "evidence_ids": ["phone-arbiter-test"],
        }],
        "trace_workload_map": {"work": "work"},
        "policy": {
            "energy_saving_ppm": 50_000,
            "latency_limit_ppm": 2_000_000,
        },
    })


class PhoneArbiterTests(unittest.TestCase):
    def test_protected_work_precedes_filler(self) -> None:
        plan = residency_plan()
        snapshot = residency_snapshot(plan)
        rows = (
            work("gemma-0", "gemma", "filler"),
            work("qwen-0", "qwen", "protected"),
        )
        decision = select_phone_arbiter_work(
            plan,
            snapshot,
            window(snapshot, plan),
            queue(snapshot, plan, rows),
            now_us=120,
            resource_ready_us=120,
        )
        self.assertEqual(decision.work_id, "qwen-0")
        self.assertEqual(decision.reason, "PROTECTED_PHONE_WORK")
        self.assertEqual(decision.priority_class, "protected")
        self.assertIn(("gemma-0", "LOWER_PHONE_PRIORITY"), decision.rejected)

    def test_filler_uses_only_a_safe_measured_gap(self) -> None:
        plan = residency_plan()
        snapshot = residency_snapshot(plan)
        decision = select_phone_arbiter_work(
            plan,
            snapshot,
            window(snapshot, plan),
            queue(
                snapshot,
                plan,
                (work("gemma-0", "gemma", "filler"),),
            ),
            now_us=120,
            resource_ready_us=120,
        )
        self.assertEqual(decision.work_id, "gemma-0")
        self.assertEqual(decision.reason, "GAP_FILLING_PHONE_WORK")
        self.assertEqual(decision.phone_finish_upper_us, 160)
        self.assertEqual(decision.safe_end_us, 270)
        self.assertEqual(decision.slack_us, 110)

    def test_long_filler_is_rejected_before_protected_work(self) -> None:
        plan = residency_plan()
        snapshot = residency_snapshot(plan)
        row = work(
            "gemma-long",
            "gemma",
            "filler",
            phone_upper_us=160,
            valid_until_us=1_000,
        )
        decision = select_phone_arbiter_work(
            plan,
            snapshot,
            window(snapshot, plan, valid_until_us=260),
            queue(snapshot, plan, (row,), valid_until_us=1_000),
            now_us=120,
            resource_ready_us=120,
            latency_limit_ppm=2_000_000,
            maximum_join_wait_ppm=1_000_000,
        )
        self.assertIsNone(decision.work_id)
        self.assertEqual(
            dict(decision.rejected)["gemma-long"],
            "PROTECTED_WORK_GUARD",
        )

    def test_completion_receipt_releases_the_remaining_phone_window(self) -> None:
        plan = residency_plan()
        snapshot = residency_snapshot(plan)
        rows = (
            work(
                "gemma-0",
                "gemma",
                "filler",
                valid_until_us=500,
            ),
            work(
                "qwen-0",
                "qwen",
                "protected",
                valid_until_us=500,
            ),
        )
        completed_window = window(
            snapshot,
            plan,
            captured_at_us=150,
            valid_until_us=500,
            protected_ready_lower_us=None,
            protected_completion_receipt_id="qwen-complete-1",
            protected_completion_at_us=145,
        )
        decision = select_phone_arbiter_work(
            plan,
            snapshot,
            completed_window,
            queue(
                snapshot,
                plan,
                rows,
                captured_at_us=150,
                valid_until_us=500,
            ),
            now_us=160,
            resource_ready_us=160,
        )
        self.assertEqual(decision.work_id, "gemma-0")
        self.assertEqual(decision.reason, "POST_PROTECTED_PHONE_WORK")
        self.assertEqual(decision.safe_end_us, 500)
        self.assertEqual(
            decision.protected_completion_receipt_id,
            "qwen-complete-1",
        )
        self.assertIn(
            ("qwen-0", "PROTECTED_OWNER_COMPLETED"),
            decision.rejected,
        )

    def test_completion_window_requires_a_timestamped_receipt(self) -> None:
        plan = residency_plan()
        snapshot = residency_snapshot(plan)
        invalid = (
            {
                "protected_ready_lower_us": None,
                "protected_completion_receipt_id": "qwen-complete-1",
            },
            {
                "protected_ready_lower_us": None,
                "protected_completion_at_us": 100,
            },
            {
                "protected_ready_lower_us": 300,
                "protected_completion_receipt_id": "qwen-complete-1",
                "protected_completion_at_us": 100,
            },
            {
                "protected_ready_lower_us": None,
                "protected_completion_receipt_id": "qwen-complete-1",
                "protected_completion_at_us": 101,
            },
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(PhoneArbiterError):
                    window(snapshot, plan, **overrides)

    def test_memory_swap_and_duplicate_hot_sets_fail_closed(self) -> None:
        plan = residency_plan()
        snapshot = residency_snapshot(plan)
        row = work("gemma-0", "gemma", "filler")
        cases = (
            ({"desktop_memory_available_bytes": 1_000}, "DESKTOP_MEMORY_RESERVE"),
            ({"swap_out_delta_pages": 1}, "DESKTOP_SWAP_ACTIVITY"),
            (
                {"duplicate_hot_slice_ids": ("gemma-layer-0",)},
                "DUPLICATE_HOT_WEIGHT_SLICE",
            ),
        )
        for overrides, reason in cases:
            with self.subTest(reason=reason):
                decision = select_phone_arbiter_work(
                    plan,
                    snapshot,
                    window(snapshot, plan, **overrides),
                    queue(snapshot, plan, (row,)),
                    now_us=120,
                    resource_ready_us=120,
                )
                self.assertIsNone(decision.work_id)
                self.assertEqual(dict(decision.rejected)[row.work_id], reason)

    def test_unified_scheduler_advances_only_on_ordered_receipt(self) -> None:
        plan = residency_plan()
        snapshot = residency_snapshot(plan)
        scheduler = UnifiedScheduler(
            (scheduler_profile(),),
            "enforce",
            phone_residency_plan=plan,
            phone_residency_snapshot=snapshot,
        )
        first = work("gemma-0", "gemma", "filler")
        first_schedule = scheduler.schedule_phone_arbiter_work(
            window(snapshot, plan),
            queue(snapshot, plan, (first,)),
            now_us=120,
        )
        self.assertEqual(first_schedule.decision.work_id, "gemma-0")
        self.assertEqual(
            first_schedule.phone_schedule.decision.arm_signal.deadline_us,
            first_schedule.decision.safe_end_us,
        )
        scheduler.release_phone_arbiter_work(
            first_schedule, 150, "gemma-output-0"
        )

        second = work(
            "gemma-1",
            "gemma",
            "filler",
            sequence_index=1,
            predecessor="gemma-output-0",
        )
        second_window = window(
            snapshot,
            plan,
            window_id="qwen-gap-2",
            captured_at_us=150,
            valid_until_us=280,
        )
        second_queue = queue(
            snapshot,
            plan,
            (second,),
            next_by_pipeline={"gemma-request": 1},
            completed=("gemma-output-0",),
            queue_id="phone-queue-2",
            captured_at_us=150,
        )
        second_schedule = scheduler.schedule_phone_arbiter_work(
            second_window,
            second_queue,
            now_us=160,
        )
        scheduler.release_phone_arbiter_work(
            second_schedule, 190, "gemma-output-1"
        )
        with self.assertRaisesRegex(
            UnifiedScheduleError, "pipeline sequence is stale"
        ):
            scheduler.schedule_phone_arbiter_work(
                second_window,
                second_queue,
                now_us=200,
            )


if __name__ == "__main__":
    unittest.main()
