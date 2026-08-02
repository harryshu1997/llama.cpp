#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fixed_diamond_runtime import (
    CapacityLedger,
    integrate_nvml,
    load_profiles,
    nearest_rank,
    parse_route_delays,
    select_rows,
    summarize_batch_events,
)
from workloads import deterministic_three_class


class RuntimeSummaryTests(unittest.TestCase):
    def test_nearest_rank_p95(self) -> None:
        self.assertEqual(nearest_rank(list(range(1, 21)), 0.95), 19)
        self.assertIsNone(nearest_rank([], 0.95))

    def test_nvml_trapezoid_and_edges(self) -> None:
        samples = [
            {"time_ns": 200_000_000, "power_w": 10.0},
            {"time_ns": 800_000_000, "power_w": 10.0},
        ]
        report = integrate_nvml(samples, 1_000_000_000)
        self.assertAlmostEqual(report["energy_j"], 10.0)
        self.assertEqual(report["max_sample_gap_ns"], 600_000_000)

    def test_batch_summary_counts_mixed_convergence(self) -> None:
        events = [{
            "batch_size": 2,
            "compute_us": 100,
            "max_queue_us": 20,
            "dispatch_reason": "BATCH_KNEE",
            "contributing_routes": ["R1", "R2"],
            "contributing_upstreams": ["cuda-prefix", "op12-prefix"],
            "status": "OK",
        }]
        summary = summarize_batch_events(events)
        self.assertEqual(summary["mean_batch"], 2.0)
        self.assertEqual(summary["mixed_route_batches"], 1)
        self.assertEqual(summary["mixed_upstream_batches"], 1)
        self.assertEqual(summary["summed_compute_us"], 100)


class CapacityTests(unittest.TestCase):
    def test_shared_op15_and_tail_capacity_is_conserved(self) -> None:
        ledger = CapacityLedger({
            "cuda-prefix": 1,
            "cuda-mid": 1,
            "op12-prefix": 1,
            "op15-mid": 1,
            "cuda-tail": 1,
        })
        self.assertTrue(ledger.try_reserve(1, 1, "R1"))
        self.assertFalse(ledger.try_reserve(2, 2, "R2"))
        ledger.release(1, 1)
        self.assertTrue(ledger.try_reserve(2, 2, "R2"))
        ledger.release(2, 2)
        self.assertFalse(ledger.pinned)
        self.assertFalse(any(ledger.active.values()))

    def test_route_delays_change_arrivals_without_changing_source(self) -> None:
        trace = deterministic_three_class()
        delays = parse_route_delays(["R1:100", "R2:200"])
        rows = select_rows(trace, ["R1", "R2"], 1, 1.0, delays)
        self.assertEqual(
            [(row.source["route_hint"], row.arrival_us) for row in rows],
            [("R1", 100), ("R2", 200)],
        )
        self.assertEqual(rows[0].source["arrival_us"], 0)


class ProfileTests(unittest.TestCase):
    def test_profile_loader_reopens_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence.json"
            evidence.write_text("{}\n", encoding="ascii")
            digest = "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
            resources = {
                "R0": ["cuda-prefix", "cuda-mid", "cuda-tail"],
                "R1": ["cuda-prefix", "op15-mid", "cuda-tail"],
                "R2": ["op12-prefix", "op15-mid", "cuda-tail"],
            }
            profiles = []
            for rank, route_id in enumerate(("R0", "R1", "R2")):
                profiles.append({
                    "route_id": route_id,
                    "offload_rank": rank,
                    "fixed_us": 0,
                    "prefill_token_us": 10 + rank,
                    "decode_step_us": 10 + rank,
                    "profiled_batch": 1,
                    "max_active": 1,
                    "gather_cap_us": 100,
                    "resources": resources[route_id],
                    "evidence_sha256": digest,
                })
            path = root / "profiles.json"
            path.write_text(json.dumps({
                "schema": "s24-fixed-route-profiles-v1",
                "profiles": profiles,
                "resource_capacities": {
                    "cuda-prefix": 1,
                    "cuda-mid": 1,
                    "op12-prefix": 1,
                    "op15-mid": 1,
                    "cuda-tail": 1,
                },
                "source_reports": [{
                    "path": "evidence.json",
                    "sha256": digest,
                }],
            }), encoding="ascii")
            loaded, capacities, sources = load_profiles(path)
            self.assertEqual([profile.route_id for profile in loaded], ["R0", "R1", "R2"])
            self.assertIsInstance(loaded[0].resources, tuple)
            self.assertEqual(capacities["cuda-tail"], 1)
            self.assertEqual(sources[0]["sha256"], digest)


if __name__ == "__main__":
    unittest.main()
