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
    DynamicResidencyCandidate,
    DynamicResidencySnapshot,
    DynamicWeightPlacement,
    DynamicWeightPlacementSpec,
    GpuBackfillCandidate,
    GpuBackfillError,
    GpuBackfillSchedule,
    GpuBubbleWindow,
    MetricEstimate,
    ProfileBundle,
    UnifiedScheduleError,
    UnifiedScheduler,
    canonical_sha256,
    select_gpu_backfill,
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
    return DynamicWeightPlacementSpec(
        placement_id=placement_id,
        slice_id=f"{model_id}-layers",
        model_id=model_id,
        model_hash=digest(character),
        weight_hash=digest(character),
        resource_id=resource_id,
        resident_bytes=resident_bytes,
        execution_resource_ids=(
            "gpu-compute" if resource_id == "gpu-vram" else "cpu",
        ),
        runtime_binding_ids=(
            "cuda0" if resource_id == "gpu-vram" else "cpu-process",
        ),
        evidence_ids=(f"{model_id}-placement-v1",),
    )


def snapshot() -> DynamicResidencySnapshot:
    placements = {
        "qwen-gpu": DynamicWeightPlacement(
            spec=placement_spec("qwen-gpu", "qwen", "a", 400),
            generation=1,
            resident_since_us=0,
            minimum_resident_until_us=100,
        ),
        "gemma-gpu": DynamicWeightPlacement(
            spec=placement_spec("gemma-gpu", "gemma", "b", 200),
            generation=1,
            resident_since_us=0,
            minimum_resident_until_us=100,
        ),
    }
    return DynamicResidencySnapshot(
        snapshot_id="snapshot-1",
        epoch_key=digest("e"),
        generation=1,
        captured_at_us=100,
        valid_until_us=800,
        memory={
            "gpu-vram": DeviceMemoryCapacity(
                "gpu-vram", 1_000, 700, 100
            ),
        },
        placements=placements,
    )


def bubble(
    source: DynamicResidencySnapshot,
    *,
    valid_until_us: int = 800,
) -> GpuBubbleWindow:
    return GpuBubbleWindow(
        bubble_id="qwen-bubble-1",
        fence_receipt_id="qwen-stage-fence-1",
        source_snapshot_id=source.snapshot_id,
        source_snapshot_sha256=canonical_sha256(source.to_json()),
        source_generation=source.generation,
        source_epoch_key=source.epoch_key,
        gpu_resource_id="gpu-compute",
        protected_owner_id="qwen-request-1",
        protected_model_id="qwen",
        captured_at_us=source.captured_at_us,
        valid_until_us=valid_until_us,
        protected_ready_lower_us=900,
        guard_us=20,
        runtime_verified=True,
        evidence_ids=("qwen-runtime-fence-v1",),
    )


def candidate(
    candidate_id: str = "gemma-filler",
    *,
    measured: bool = True,
    avoided_energy: tuple[int, int, int] = (500, 550, 450),
    backfill_energy: tuple[int, int, int] = (200, 230, 180),
) -> GpuBackfillCandidate:
    return GpuBackfillCandidate(
        candidate_id=candidate_id,
        work_id="gemma-request-42",
        model_id="gemma",
        gpu_resource_id="gpu-compute",
        workspace_resource_id="gpu-vram",
        workspace_bytes=100,
        required_placement_ids=("gemma-gpu",),
        additional_resource_ids=(),
        deadline_us=700,
        energy_boundary_id="cpu-package+gpu-board+whole-phone",
        accounting_scope="burstgpt-horizon",
        service_latency_us=metric(100, 120, 90, measured=measured),
        restore_latency_us=metric(20, 30, 15, measured=measured),
        avoided_energy_uj=metric(*avoided_energy, measured=measured),
        backfill_energy_uj=metric(*backfill_energy, measured=measured),
        evidence_ids=("matched-gpu-backfill-v1",),
    )


def dynamic_candidate(
    source: DynamicResidencySnapshot,
) -> DynamicResidencyCandidate:
    return DynamicResidencyCandidate(
        candidate_id="llama-prefetch",
        source_snapshot_id=source.snapshot_id,
        source_snapshot_sha256=canonical_sha256(source.to_json()),
        source_generation=source.generation,
        source_epoch_key=source.epoch_key,
        target=placement_spec("llama-gpu", "llama", "c", 100),
        evict_placement_ids=("qwen-gpu",),
        transition_resource_ids=("gpu-dma",),
        expected_reuse_count=4,
        minimum_reuse_count=2,
        minimum_residency_us=500,
        latest_ready_us=700,
        energy_boundary_id="cpu-package+gpu-board+whole-phone",
        accounting_scope="burstgpt-horizon",
        baseline_latency_us=metric(1_000, 1_100, 900),
        resident_latency_us=metric(500, 550, 450),
        load_latency_us=metric(100, 120, 90),
        eviction_latency_us=metric(50, 60, 40),
        baseline_energy_uj=metric(1_000, 1_050, 950),
        resident_energy_uj=metric(500, 550, 450),
        load_energy_uj=metric(100, 120, 90),
        eviction_energy_uj=metric(20, 30, 15),
        evidence_ids=("matched-dynamic-residency-v1",),
    )


def scheduler_profile() -> ProfileBundle:
    return ProfileBundle.from_json({
        "schema": "s42-general-scheduler-profile-v1",
        "profile_id": "gpu-backfill-test",
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
            "evidence_ids": ["gpu-backfill-test"],
        }],
        "trace_workload_map": {"work": "work"},
        "policy": {
            "energy_saving_ppm": 50_000,
            "latency_limit_ppm": 2_000_000,
        },
    })


class GpuBackfillTests(unittest.TestCase):
    def test_energy_positive_resident_filler(self) -> None:
        source = snapshot()
        decision = select_gpu_backfill(
            source,
            bubble(source),
            (candidate(),),
            now_us=200,
            resource_ready_us=200,
        )
        self.assertEqual(decision.candidate_id, "gemma-filler")
        self.assertEqual(decision.reason, "ENERGY_POSITIVE_GPU_BACKFILL")
        self.assertEqual(decision.work_finish_upper_us, 320)
        self.assertEqual(decision.restore_finish_upper_us, 350)
        self.assertEqual(decision.slack_after_guard_us, 430)
        self.assertEqual(decision.energy_saving_lower_uj, 220)
        self.assertEqual(decision.energy_saving_ppm, 488_888)

    def test_bubble_validity_is_a_hard_finish_boundary(self) -> None:
        source = snapshot()
        decision = select_gpu_backfill(
            source,
            bubble(source, valid_until_us=340),
            (candidate(),),
            now_us=200,
            resource_ready_us=200,
        )
        self.assertEqual(
            decision.rejected,
            (("gemma-filler", "BUBBLE_TOO_SHORT"),),
        )

    def test_residency_and_workspace_fail_closed(self) -> None:
        source = snapshot()
        rows = (
            replace(candidate("missing"), required_placement_ids=("absent",)),
            replace(candidate("wrong-model"), model_id="qwen"),
            replace(candidate("too-large"), workspace_bytes=201),
        )
        decision = select_gpu_backfill(
            source,
            bubble(source),
            rows,
            now_us=200,
            resource_ready_us=200,
        )
        self.assertEqual(
            dict(decision.rejected),
            {
                "missing": "RESIDENCY_MISSING",
                "wrong-model": "RESIDENCY_MODEL_MISMATCH",
                "too-large": "WORKSPACE_MEMORY",
            },
        )

    def test_required_weights_must_share_gpu_memory_pool(self) -> None:
        source = snapshot()
        with_cpu_memory = DynamicResidencySnapshot(
            snapshot_id=source.snapshot_id,
            epoch_key=source.epoch_key,
            generation=source.generation,
            captured_at_us=source.captured_at_us,
            valid_until_us=source.valid_until_us,
            memory={
                **source.memory,
                "cpu-ram": DeviceMemoryCapacity(
                    "cpu-ram", 10_000, 1_000, 1_000
                ),
            },
            placements=source.placements,
        )
        row = replace(candidate(), workspace_resource_id="cpu-ram")
        decision = select_gpu_backfill(
            with_cpu_memory,
            bubble(with_cpu_memory),
            (row,),
            now_us=200,
            resource_ready_us=200,
        )
        self.assertEqual(
            decision.rejected,
            (("gemma-filler", "RESIDENCY_RESOURCE_MISMATCH"),),
        )

    def test_required_weights_must_bind_gpu_execution_resource(self) -> None:
        source = snapshot()
        placements = dict(source.placements)
        gemma = placements["gemma-gpu"]
        placements["gemma-gpu"] = replace(
            gemma,
            spec=replace(
                gemma.spec,
                execution_resource_ids=("different-gpu",),
            ),
        )
        changed = replace(source, placements=placements)
        decision = select_gpu_backfill(
            changed,
            bubble(changed),
            (candidate(),),
            now_us=200,
            resource_ready_us=200,
        )
        self.assertEqual(
            decision.rejected,
            (("gemma-filler", "RESIDENCY_EXECUTION_MISMATCH"),),
        )

    def test_unmeasured_or_energy_negative_filler_is_rejected(self) -> None:
        source = snapshot()
        rows = (
            candidate("unmeasured", measured=False),
            candidate(
                "energy-negative",
                avoided_energy=(200, 220, 180),
                backfill_energy=(200, 230, 180),
            ),
        )
        decision = select_gpu_backfill(
            source,
            bubble(source),
            rows,
            now_us=200,
            resource_ready_us=200,
        )
        self.assertEqual(
            dict(decision.rejected),
            {
                "unmeasured": "MEASUREMENT_REQUIRED",
                "energy-negative": "ENERGY_REGRESSION",
            },
        )

    def test_unverified_runtime_bubble_leaves_gpu_idle(self) -> None:
        source = snapshot()
        unverified = replace(bubble(source), runtime_verified=False)
        decision = select_gpu_backfill(
            source,
            unverified,
            (candidate(),),
            now_us=200,
            resource_ready_us=200,
        )
        self.assertIsNone(decision.candidate_id)
        self.assertEqual(
            decision.rejected,
            (("gemma-filler", "BUBBLE_UNVERIFIED"),),
        )

    def test_energy_first_selection_uses_larger_lower_bound_saving(self) -> None:
        source = snapshot()
        smaller = candidate(
            "smaller",
            avoided_energy=(400, 450, 350),
            backfill_energy=(200, 230, 180),
        )
        larger = candidate("larger")
        decision = select_gpu_backfill(
            source,
            bubble(source),
            (smaller, larger),
            now_us=200,
            resource_ready_us=200,
        )
        self.assertEqual(decision.candidate_id, "larger")

    def test_expired_or_stale_bubble_is_rejected(self) -> None:
        source = snapshot()
        with self.assertRaisesRegex(GpuBackfillError, "expired"):
            select_gpu_backfill(
                source,
                bubble(source),
                (candidate(),),
                now_us=800,
                resource_ready_us=800,
            )
        stale = replace(bubble(source), source_generation=2)
        with self.assertRaisesRegex(GpuBackfillError, "epoch mismatch"):
            select_gpu_backfill(
                source,
                stale,
                (candidate(),),
                now_us=200,
                resource_ready_us=200,
            )

    def test_bubble_binds_exact_residency_snapshot(self) -> None:
        source = snapshot()
        old_bubble = bubble(source)
        refreshed = replace(source, captured_at_us=1)
        with self.assertRaisesRegex(
            GpuBackfillError, "residency snapshot mismatch"
        ):
            select_gpu_backfill(
                refreshed,
                old_bubble,
                (candidate(),),
                now_us=200,
                resource_ready_us=200,
            )

    def test_unified_scheduler_leases_and_releases_resident_weights(self) -> None:
        source = snapshot()
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        schedule = unified.schedule_gpu_backfill(
            bubble(source), (candidate(),), now_us=200
        )
        self.assertIsInstance(schedule, GpuBackfillSchedule)
        self.assertEqual(
            {lease.resource_id for lease in schedule.leases},
            {"gpu-compute"},
        )
        current = unified.dynamic_residency_snapshot
        assert current is not None
        self.assertEqual(current.placements["gemma-gpu"].active_leases, 1)
        with self.assertRaisesRegex(
            UnifiedScheduleError, "blocked by placement leases"
        ):
            unified.update_dynamic_residency(
                replace(current, captured_at_us=250)
            )
        with self.assertRaisesRegex(
            UnifiedScheduleError, "active placement leases"
        ):
            unified.schedule_dynamic_residency(
                (dynamic_candidate(current),), now_us=200
            )
        unified.release_gpu_backfill(schedule, 330)
        current = unified.dynamic_residency_snapshot
        assert current is not None
        self.assertEqual(current.placements["gemma-gpu"].active_leases, 0)
        with self.assertRaisesRegex(
            UnifiedScheduleError, "schedule is not active"
        ):
            unified.release_gpu_backfill(schedule, 330)

    def test_gpu_completion_must_stay_inside_reserved_envelope(self) -> None:
        source = snapshot()
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        schedule = unified.schedule_gpu_backfill(
            bubble(source), (candidate(),), now_us=200
        )
        for actual_end_us in (199, 351):
            with self.subTest(actual_end_us=actual_end_us):
                with self.assertRaisesRegex(
                    UnifiedScheduleError, "decision envelope"
                ):
                    unified.release_gpu_backfill(
                        schedule, actual_end_us
                    )
        unified.release_gpu_backfill(schedule, 330)

    def test_gpu_queue_can_make_bubble_infeasible(self) -> None:
        source = snapshot()
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        unified.reserve_external_resource(
            "gpu-compute", "protected-work", 200, 700
        )
        schedule = unified.schedule_gpu_backfill(
            bubble(source),
            (replace(candidate(), deadline_us=900),),
            now_us=200,
        )
        self.assertIsNone(schedule.decision.candidate_id)
        self.assertEqual(schedule.leases, ())
        self.assertIn(
            ("gemma-filler", "BUBBLE_TOO_SHORT"),
            schedule.decision.rejected,
        )

    def test_pending_residency_transition_blocks_fast_path(self) -> None:
        source = snapshot()
        unified = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        transition = unified.schedule_dynamic_residency(
            (dynamic_candidate(source),), now_us=200
        )
        self.assertIsNotNone(transition.decision.candidate_id)
        with self.assertRaisesRegex(
            UnifiedScheduleError, "blocked by a residency transition"
        ):
            unified.schedule_gpu_backfill(
                bubble(source), (candidate(),), now_us=200
            )


if __name__ == "__main__":
    unittest.main()
