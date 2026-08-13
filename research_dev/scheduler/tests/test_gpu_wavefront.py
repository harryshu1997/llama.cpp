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
    DynamicResidencySnapshot,
    DynamicWeightPlacement,
    DynamicWeightPlacementSpec,
    GpuBackfillCandidate,
    GpuBubbleWindow,
    GpuReadyChunk,
    GpuWavefrontSnapshot,
    MetricEstimate,
    ProfileBundle,
    UnifiedScheduleError,
    UnifiedScheduler,
    canonical_sha256,
    select_gpu_wavefront_backfill,
)


def digest(character: str) -> str:
    return "sha256:" + character * 64


def metric(mean: int, upper: int, lower: int) -> MetricEstimate:
    return MetricEstimate(
        mean=mean,
        upper=upper,
        lower=lower,
        sample_count=4,
        measured=True,
    )


def placement_spec(
    placement_id: str,
    slice_id: str,
    model_id: str,
    character: str,
    resource_id: str,
    resident_bytes: int,
    execution_resource_id: str,
) -> DynamicWeightPlacementSpec:
    return DynamicWeightPlacementSpec(
        placement_id=placement_id,
        slice_id=slice_id,
        model_id=model_id,
        model_hash=digest(character),
        weight_hash=digest(character),
        resource_id=resource_id,
        resident_bytes=resident_bytes,
        execution_resource_ids=(execution_resource_id,),
        runtime_binding_ids=(placement_id + "-binding",),
        evidence_ids=(placement_id + "-receipt",),
    )


def residency_snapshot() -> DynamicResidencySnapshot:
    specs = (
        placement_spec(
            "qwen-gpu",
            "qwen-layers-25-39",
            "qwen",
            "a",
            "gpu-vram",
            400,
            "gpu-compute",
        ),
        placement_spec(
            "gemma-gpu",
            "gemma-layer-47",
            "gemma",
            "b",
            "gpu-vram",
            200,
            "gpu-compute",
        ),
        placement_spec(
            "qwen-cpu",
            "qwen-full-model",
            "qwen",
            "c",
            "cpu-ram",
            3_000,
            "cpu",
        ),
        placement_spec(
            "gemma-cpu",
            "gemma-full-model",
            "gemma",
            "d",
            "cpu-ram",
            3_000,
            "cpu",
        ),
        placement_spec(
            "qwen-phone",
            "qwen-phone-ffn",
            "qwen",
            "e",
            "phone-ram",
            1_500,
            "op15-htp",
        ),
        placement_spec(
            "gemma-phone",
            "gemma-phone-ffn",
            "gemma",
            "f",
            "phone-ram",
            1_500,
            "op15-htp",
        ),
    )
    placements = {
        spec.placement_id: DynamicWeightPlacement(
            spec=spec,
            generation=1,
            resident_since_us=0,
            minimum_resident_until_us=100,
        )
        for spec in specs
    }
    return DynamicResidencySnapshot(
        snapshot_id="dual-model-residency-1",
        epoch_key=digest("9"),
        generation=1,
        captured_at_us=100,
        valid_until_us=800,
        memory={
            "cpu-ram": DeviceMemoryCapacity(
                "cpu-ram", 10_000, 6_000, 1_000
            ),
            "gpu-vram": DeviceMemoryCapacity(
                "gpu-vram", 1_000, 700, 100
            ),
            "phone-ram": DeviceMemoryCapacity(
                "phone-ram", 5_000, 3_000, 1_000
            ),
        },
        placements=placements,
    )


def bubble(
    source: DynamicResidencySnapshot,
    *,
    bubble_id: str = "qwen-phone-tail-1",
) -> GpuBubbleWindow:
    return GpuBubbleWindow(
        bubble_id=bubble_id,
        fence_receipt_id=bubble_id + "-receipt",
        source_snapshot_id=source.snapshot_id,
        source_snapshot_sha256=canonical_sha256(source.to_json()),
        source_generation=source.generation,
        source_epoch_key=source.epoch_key,
        gpu_resource_id="gpu-compute",
        protected_owner_id="qwen-request-31",
        protected_model_id="qwen",
        captured_at_us=source.captured_at_us,
        valid_until_us=800,
        protected_ready_lower_us=900,
        guard_us=20,
        runtime_verified=True,
        evidence_ids=("qwen-ffn-fence-physical-v1",),
    )


def candidate(
    chunk_id: str,
    *,
    model_id: str = "gemma",
    service: tuple[int, int, int] = (100, 120, 90),
    avoided_energy: tuple[int, int, int] = (500, 550, 450),
    backfill_energy: tuple[int, int, int] = (200, 230, 180),
) -> GpuBackfillCandidate:
    return GpuBackfillCandidate(
        candidate_id=chunk_id,
        work_id=chunk_id + "-work",
        model_id=model_id,
        gpu_resource_id="gpu-compute",
        workspace_resource_id="gpu-vram",
        workspace_bytes=100,
        required_placement_ids=(model_id + "-gpu",),
        additional_resource_ids=("cpu", "op15-htp"),
        deadline_us=760,
        energy_boundary_id="cpu-package+gpu-board+whole-phone",
        accounting_scope="fp16-burstgpt-74",
        service_latency_us=metric(*service),
        restore_latency_us=metric(20, 30, 15),
        avoided_energy_uj=metric(*avoided_energy),
        backfill_energy_uj=metric(*backfill_energy),
        evidence_ids=(chunk_id + "-matched-energy",),
    )


def chunk(
    chunk_id: str,
    *,
    model_id: str = "gemma",
    pipeline_id: str = "gemma-request-50",
    sequence_index: int = 0,
    predecessor: str | None = None,
    runtime_verified: bool = True,
    row: GpuBackfillCandidate | None = None,
    ready_at_us: int = 150,
) -> GpuReadyChunk:
    return GpuReadyChunk(
        chunk_id=chunk_id,
        pipeline_id=pipeline_id,
        model_id=model_id,
        sequence_index=sequence_index,
        layer_start=47 if model_id == "gemma" else 25,
        layer_end=48 if model_id == "gemma" else 40,
        token_count=1,
        ready_receipt_id=chunk_id + "-ready",
        input_buffer_id=chunk_id + "-activation",
        input_buffer_sha256=digest("7"),
        predecessor_output_receipt_id=predecessor,
        ready_at_us=ready_at_us,
        valid_until_us=780,
        producer_resource_ids=("cpu", "op15-htp"),
        runtime_verified=runtime_verified,
        candidate=row or candidate(chunk_id, model_id=model_id),
        evidence_ids=(chunk_id + "-activation-receipt",),
    )


def wavefront(
    source: DynamicResidencySnapshot,
    chunks: tuple[GpuReadyChunk, ...],
    *,
    next_gemma: int = 0,
    completed: tuple[str, ...] = (),
    wavefront_id: str = "wavefront-1",
) -> GpuWavefrontSnapshot:
    return GpuWavefrontSnapshot(
        wavefront_id=wavefront_id,
        source_snapshot_id=source.snapshot_id,
        source_snapshot_sha256=canonical_sha256(source.to_json()),
        source_generation=source.generation,
        source_epoch_key=source.epoch_key,
        captured_at_us=source.captured_at_us,
        valid_until_us=800,
        next_sequence_by_pipeline={
            "gemma-request-50": next_gemma,
            "qwen-request-31": 0,
        },
        completed_output_receipt_ids=completed,
        ready_chunks=chunks,
    )


def scheduler_profile() -> ProfileBundle:
    return ProfileBundle.from_json({
        "schema": "s42-general-scheduler-profile-v1",
        "profile_id": "gpu-wavefront-test",
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
                ("op15-htp", "phone"),
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
            "evidence_ids": ["gpu-wavefront-test"],
        }],
        "trace_workload_map": {"work": "work"},
        "policy": {
            "energy_saving_ppm": 50_000,
            "latency_limit_ppm": 2_000_000,
        },
    })


class GpuWavefrontTests(unittest.TestCase):
    def test_selects_ready_opposite_model_chunk(self) -> None:
        source = residency_snapshot()
        decision = select_gpu_wavefront_backfill(
            source,
            bubble(source),
            wavefront(source, (chunk("gemma-chunk-0"),)),
            now_us=200,
            resource_ready_us=200,
        )
        self.assertEqual(decision.chunk_id, "gemma-chunk-0")
        self.assertEqual(decision.pipeline_id, "gemma-request-50")
        self.assertEqual(decision.layer_start, 47)
        self.assertEqual(decision.layer_end, 48)
        self.assertEqual(decision.ready_queue_depth, 1)
        self.assertEqual(decision.gpu_work_coverage_ppm, 206_896)
        self.assertEqual(decision.envelope_coverage_ppm, 258_620)
        self.assertEqual(decision.backfill.energy_saving_lower_uj, 220)

    def test_protected_model_and_unverified_input_fail_closed(self) -> None:
        source = residency_snapshot()
        rows = (
            chunk(
                "qwen-chunk-0",
                model_id="qwen",
                pipeline_id="qwen-request-31",
            ),
            chunk("gemma-unverified", runtime_verified=False),
        )
        decision = select_gpu_wavefront_backfill(
            source,
            bubble(source),
            wavefront(source, rows),
            now_us=200,
            resource_ready_us=200,
        )
        self.assertIsNone(decision.chunk_id)
        self.assertEqual(
            dict(decision.backfill.rejected),
            {
                "qwen-chunk-0": "PROTECTED_MODEL_CHUNK",
                "gemma-unverified": "READY_CHUNK_UNVERIFIED",
            },
        )

    def test_pipeline_head_and_predecessor_are_exact(self) -> None:
        source = residency_snapshot()
        future = chunk(
            "gemma-chunk-1",
            sequence_index=1,
            predecessor="gemma-output-0",
        )
        decision = select_gpu_wavefront_backfill(
            source,
            bubble(source),
            wavefront(
                source,
                (future,),
                next_gemma=0,
                completed=("gemma-output-0",),
            ),
            now_us=200,
            resource_ready_us=200,
        )
        self.assertEqual(
            decision.backfill.rejected,
            (("gemma-chunk-1", "PIPELINE_NOT_HEAD"),),
        )

    def test_coverage_objective_fills_more_of_the_safe_window(self) -> None:
        source = residency_snapshot()
        energy_row = candidate(
            "gemma-energy",
            service=(80, 100, 70),
            avoided_energy=(900, 950, 850),
            backfill_energy=(200, 230, 180),
        )
        coverage_row = candidate(
            "gemma-coverage",
            service=(250, 300, 220),
            avoided_energy=(500, 550, 450),
            backfill_energy=(200, 230, 180),
        )
        rows = (
            chunk(
                "gemma-energy",
                pipeline_id="gemma-request-50",
                row=energy_row,
            ),
            chunk(
                "gemma-coverage",
                pipeline_id="gemma-request-51",
                row=coverage_row,
            ),
        )
        state = GpuWavefrontSnapshot(
            wavefront_id="wavefront-choice",
            source_snapshot_id=source.snapshot_id,
            source_snapshot_sha256=canonical_sha256(source.to_json()),
            source_generation=source.generation,
            source_epoch_key=source.epoch_key,
            captured_at_us=100,
            valid_until_us=800,
            next_sequence_by_pipeline={
                "gemma-request-50": 0,
                "gemma-request-51": 0,
            },
            completed_output_receipt_ids=(),
            ready_chunks=rows,
        )
        coverage = select_gpu_wavefront_backfill(
            source,
            bubble(source),
            state,
            now_us=200,
            resource_ready_us=200,
        )
        energy = select_gpu_wavefront_backfill(
            source,
            bubble(source),
            state,
            now_us=200,
            resource_ready_us=200,
            objective="energy_then_coverage",
        )
        self.assertEqual(coverage.chunk_id, "gemma-coverage")
        self.assertEqual(energy.chunk_id, "gemma-energy")

    def test_unified_scheduler_advances_only_after_output_receipt(self) -> None:
        source = residency_snapshot()
        scheduler = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        first_state = wavefront(source, (chunk("gemma-chunk-0"),))
        first = scheduler.schedule_gpu_wavefront_backfill(
            bubble(source), first_state, now_us=200
        )
        self.assertEqual(
            {lease.resource_id for lease in first.leases},
            {"cpu", "gpu-compute", "op15-htp"},
        )
        with self.assertRaisesRegex(
            UnifiedScheduleError, "already has active GPU work"
        ):
            scheduler.schedule_gpu_wavefront_backfill(
                bubble(source), first_state, now_us=200
            )
        scheduler.release_gpu_wavefront_backfill(
            first, 330, "gemma-output-0"
        )
        refreshed = scheduler.dynamic_residency_snapshot
        assert refreshed is not None

        stale = wavefront(
            refreshed,
            (),
            next_gemma=0,
            completed=("gemma-output-0",),
            wavefront_id="wavefront-stale",
        )
        with self.assertRaisesRegex(
            UnifiedScheduleError, "sequence is stale"
        ):
            scheduler.schedule_gpu_wavefront_backfill(
                bubble(source), stale, now_us=350
            )

        second_chunk = chunk(
            "gemma-chunk-1",
            sequence_index=1,
            predecessor="gemma-output-0",
            ready_at_us=340,
        )
        second_state = wavefront(
            refreshed,
            (second_chunk,),
            next_gemma=1,
            completed=("gemma-output-0",),
            wavefront_id="wavefront-2",
        )
        second = scheduler.schedule_gpu_wavefront_backfill(
            bubble(refreshed, bubble_id="qwen-phone-tail-2"),
            second_state,
            now_us=350,
        )
        scheduler.release_gpu_wavefront_backfill(
            second, 480, "gemma-output-1"
        )
        current = scheduler.dynamic_residency_snapshot
        assert current is not None
        self.assertEqual(current.placements["gemma-gpu"].active_leases, 0)

    def test_abort_clears_wavefront_and_placement_leases(self) -> None:
        source = residency_snapshot()
        scheduler = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        state = wavefront(source, (chunk("gemma-chunk-0"),))
        schedule = scheduler.schedule_gpu_wavefront_backfill(
            bubble(source), state, now_us=200
        )
        scheduler.abort_gpu_wavefront_backfill(schedule, 351)
        current = scheduler.dynamic_residency_snapshot
        assert current is not None
        self.assertEqual(current.placements["gemma-gpu"].active_leases, 0)
        with self.assertRaisesRegex(
            UnifiedScheduleError, "schedule is not active"
        ):
            scheduler.release_gpu_wavefront_backfill(
                schedule, 330, "gemma-output-0"
            )

        retry = scheduler.schedule_gpu_wavefront_backfill(
            bubble(current, bubble_id="qwen-phone-tail-retry"),
            wavefront(
                current,
                (chunk("gemma-chunk-retry", ready_at_us=352),),
                wavefront_id="wavefront-retry",
            ),
            now_us=352,
        )
        self.assertEqual(retry.decision.sequence_index, 0)
        scheduler.release_gpu_wavefront_backfill(
            retry, 482, "gemma-output-retry"
        )

    def test_producer_contention_leaves_gpu_idle(self) -> None:
        source = residency_snapshot()
        scheduler = UnifiedScheduler(
            (scheduler_profile(),),
            "control",
            dynamic_residency_snapshot=source,
        )
        scheduler.reserve_external_resource(
            "op15-htp", "protected-qwen", 200, 800
        )
        schedule = scheduler.schedule_gpu_wavefront_backfill(
            bubble(source),
            wavefront(source, (chunk("gemma-chunk-0"),)),
            now_us=200,
        )
        self.assertIsNone(schedule.decision.chunk_id)
        self.assertEqual(
            schedule.decision.backfill.rejected,
            (("gemma-chunk-0", "RESOURCE_NOT_READY"),),
        )

    def test_ready_chunk_binds_exact_residency_snapshot(self) -> None:
        source = residency_snapshot()
        state = wavefront(source, (chunk("gemma-chunk-0"),))
        refreshed = replace(source, captured_at_us=101)
        with self.assertRaisesRegex(
            ValueError, "residency snapshot mismatch"
        ):
            select_gpu_wavefront_backfill(
                refreshed,
                bubble(refreshed),
                state,
                now_us=200,
                resource_ready_us=200,
            )


if __name__ == "__main__":
    unittest.main()
