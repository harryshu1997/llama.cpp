#!/usr/bin/env python3

from __future__ import annotations

import unittest

from research_dev.scheduler import (
    DeviceMemoryCapacity,
    MetricEstimate,
    RuntimePlacementCandidate,
    RuntimePlacementDecision,
    RuntimePlacementError,
    RuntimePlacementPlanner,
    RuntimePlacementSnapshot,
)


WORK_SET = "sha256:" + "a" * 64


def estimate(mean: int, lower: int, upper: int) -> MetricEstimate:
    return MetricEstimate(
        mean=mean,
        lower=lower,
        upper=upper,
        sample_count=2,
        measured=True,
    )


def candidate(
    candidate_id: str,
    *,
    gpu_bytes: int,
    latency: MetricEstimate,
    energy: MetricEstimate,
    status: str = "measured",
    placement_verified: bool = True,
    workload_verified: bool = True,
) -> RuntimePlacementCandidate:
    return RuntimePlacementCandidate(
        candidate_id=candidate_id,
        workload_id="two-model-burstgpt",
        work_set_sha256=WORK_SET,
        energy_boundary_id="cpu-package+gpu-board+whole-phone",
        additional_bytes={
            "cuda0-vram": gpu_bytes,
            "host-ram": 20_000,
        },
        runtime_bindings={
            "gemma_gpu_layers": 25,
            "qwen_gpu_layers": 18,
            "schedule_mode": "two-phase-switch",
        },
        latency_us=latency,
        fleet_energy_uj=energy,
        status=status,
        placement_verified=placement_verified,
        workload_verified=workload_verified,
        evidence_ids=(candidate_id + "-r1", candidate_id + "-r2"),
    )


def snapshot() -> RuntimePlacementSnapshot:
    return RuntimePlacementSnapshot(
        snapshot_id="live-capacity-r1",
        captured_at_us=1_000,
        valid_until_us=11_000,
        capacities={
            "cuda0-vram": DeviceMemoryCapacity(
                "cuda0-vram", 100_000, 10_000, 5_000
            ),
            "host-ram": DeviceMemoryCapacity(
                "host-ram", 200_000, 20_000, 20_000
            ),
        },
    )


class RuntimePlacementPlannerTests(unittest.TestCase):
    def test_selects_verified_energy_and_latency_improvement(self) -> None:
        baseline = candidate(
            "server-only",
            gpu_bytes=70_000,
            latency=estimate(2_779_000, 2_778_000, 2_780_000),
            energy=estimate(
                300_000_000, 299_000_000, 301_000_000
            ),
        )
        qualified = candidate(
            "server-plus-phone",
            gpu_bytes=70_000,
            latency=estimate(2_578_000, 2_559_000, 2_596_000),
            energy=estimate(
                224_000_000, 222_000_000, 226_000_000
            ),
        )
        full_gpu = candidate(
            "full-gpu",
            gpu_bytes=100_000,
            latency=estimate(2_000_000, 1_900_000, 2_100_000),
            energy=estimate(
                200_000_000, 190_000_000, 210_000_000
            ),
        )
        cohort_only = candidate(
            "cohort-only",
            gpu_bytes=70_000,
            latency=estimate(2_400_000, 2_300_000, 2_500_000),
            energy=estimate(
                210_000_000, 200_000_000, 220_000_000
            ),
            workload_verified=False,
        )
        decision = RuntimePlacementPlanner().plan(
            candidates=(baseline, qualified, full_gpu, cohort_only),
            baseline_candidate_id="server-only",
            snapshot=snapshot(),
            now_us=2_000,
            minimum_energy_saving_ppm=200_000,
            maximum_latency_ppm=999_999,
        )
        self.assertEqual(decision.selected.candidate_id, "server-plus-phone")
        self.assertGreaterEqual(
            decision.conservative_energy_saving_ppm, 200_000
        )
        self.assertLess(decision.conservative_latency_change_ppm, 0)
        self.assertIn(
            ("full-gpu", "CAPACITY:cuda0-vram"), decision.rejected
        )
        self.assertIn(
            ("cohort-only", "WORKLOAD_UNVERIFIED"), decision.rejected
        )
        self.assertEqual(
            RuntimePlacementDecision.from_json(decision.to_json()), decision
        )

    def test_stale_snapshot_fails_closed(self) -> None:
        baseline = candidate(
            "server-only",
            gpu_bytes=70_000,
            latency=estimate(2_779_000, 2_778_000, 2_780_000),
            energy=estimate(
                300_000_000, 299_000_000, 301_000_000
            ),
        )
        qualified = candidate(
            "server-plus-phone",
            gpu_bytes=70_000,
            latency=estimate(2_578_000, 2_559_000, 2_596_000),
            energy=estimate(
                224_000_000, 222_000_000, 226_000_000
            ),
        )
        with self.assertRaisesRegex(RuntimePlacementError, "stale"):
            RuntimePlacementPlanner().plan(
                candidates=(baseline, qualified),
                baseline_candidate_id="server-only",
                snapshot=snapshot(),
                now_us=11_000,
                minimum_energy_saving_ppm=200_000,
                maximum_latency_ppm=999_999,
            )

    def test_energy_gate_falls_back_to_measured_baseline(self) -> None:
        baseline = candidate(
            "server-only",
            gpu_bytes=70_000,
            latency=estimate(2_779_000, 2_778_000, 2_780_000),
            energy=estimate(
                300_000_000, 299_000_000, 301_000_000
            ),
        )
        weak = candidate(
            "weak-saving",
            gpu_bytes=70_000,
            latency=estimate(2_578_000, 2_559_000, 2_596_000),
            energy=estimate(
                250_000_000, 245_000_000, 250_000_000
            ),
        )
        decision = RuntimePlacementPlanner().plan(
            candidates=(baseline, weak),
            baseline_candidate_id="server-only",
            snapshot=snapshot(),
            now_us=2_000,
            minimum_energy_saving_ppm=200_000,
            maximum_latency_ppm=999_999,
        )
        self.assertEqual(decision.selected.candidate_id, "server-only")
        self.assertEqual(
            decision.selection_reason,
            "BASELINE_FALLBACK_NO_ADMISSIBLE_ALTERNATIVE",
        )
        self.assertEqual(decision.conservative_energy_saving_ppm, 0)
        self.assertEqual(decision.conservative_latency_change_ppm, 0)
        self.assertEqual(decision.rejected, (("weak-saving", "ENERGY_GATE"),))
        self.assertEqual(
            RuntimePlacementDecision.from_json(decision.to_json()), decision
        )

    def test_decision_hash_detects_mutation(self) -> None:
        baseline = candidate(
            "server-only",
            gpu_bytes=70_000,
            latency=estimate(2_779_000, 2_778_000, 2_780_000),
            energy=estimate(
                300_000_000, 299_000_000, 301_000_000
            ),
        )
        qualified = candidate(
            "server-plus-phone",
            gpu_bytes=70_000,
            latency=estimate(2_578_000, 2_559_000, 2_596_000),
            energy=estimate(
                224_000_000, 222_000_000, 226_000_000
            ),
        )
        decision = RuntimePlacementPlanner().plan(
            candidates=(baseline, qualified),
            baseline_candidate_id="server-only",
            snapshot=snapshot(),
            now_us=2_000,
            minimum_energy_saving_ppm=200_000,
            maximum_latency_ppm=999_999,
        ).to_json()
        decision["selected"]["runtime_bindings"]["qwen_gpu_layers"] = 17
        with self.assertRaisesRegex(RuntimePlacementError, "hash mismatch"):
            RuntimePlacementDecision.from_json(decision)


if __name__ == "__main__":
    unittest.main()
