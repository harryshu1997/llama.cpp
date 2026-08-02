#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cut_selector import build_selection, canonical_bytes, sha256_file
from selected_profiles import ProfileError, load_bundle, write_bundle
from test_cut_selector import WORKERS, measurement


def event(batch: int) -> dict:
    return {"status": "OK", "batch_size": batch, "compute_us": 100}


def calibration(selection_path: Path) -> dict:
    active = {
        "R0": {"cuda-prefix", "cuda-mid", "cuda-tail"},
        "R2": {"op12-prefix", "op15-mid", "cuda-tail"},
    }
    ranges = {
        "cuda-prefix": (0, 6),
        "cuda-mid": (6, 8),
        "op12-prefix": (0, 1),
        "op15-mid": (1, 8),
        "cuda-tail": (8, 48),
    }
    points = []
    for route in ("R2", "R0"):
        for batch in (1, 4, 24, 32):
            for rep in range(2):
                points.append({
                    "route_id": route,
                    "batch_size": batch,
                    "rep": rep,
                    "wall_us": 1000 * batch + rep,
                    "max_latency_us": 1000 * batch + rep,
                    "cuda_work_us": (100 if route == "R2" else 300) * batch + rep,
                    "output_tokens": [1, 2, 3, 4],
                    "events": {
                        name: [event(batch) for _ in range(4)] if name in active[route] else []
                        for name in WORKERS
                    },
                })
    return {
        "schema": "s31-selected-route-calibration-v1",
        "status": "CALIBRATION_COMPLETE",
        "selected_cut": 1,
        "selection": {
            "path": selection_path.name,
            "sha256": "sha256:" + sha256_file(selection_path),
        },
        "shape": {"input_tokens": 1, "output_steps": 4, "context": 16},
        "batches": [1, 4, 24, 32],
        "reps": 2,
        "gather_us": 5000,
        "measurement_order": [
            "R2:B32", "R2:B24", "R2:B4", "R2:B1",
            "R0:B32", "R0:B24", "R0:B4", "R0:B1",
        ],
        "capacities": {name: 32 for name in WORKERS},
        "workers": {
            name: {
                "layer_start": ranges[name][0],
                "layer_end": ranges[name][1],
                "max_streams": 32,
            }
            for name in WORKERS
        },
        "points": points,
    }


class SelectedProfileTests(unittest.TestCase):
    def fixture(self, root: Path):
        measurements = []
        for cut, values in {1: (300, 400), 2: (500, 300)}.items():
            path = root / f"cut-{cut}.json"
            path.write_bytes(canonical_bytes(measurement(cut, *values, 2000 + cut)))
            measurements.append(path)
        selection = root / "selection.json"
        selection.write_bytes(canonical_bytes(build_selection(measurements, 5000, (1, 2))))
        source = root / "calibration.json"
        source.write_bytes(canonical_bytes(calibration(selection)))
        return source, selection

    def test_profile_binds_cut_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _selection = self.fixture(root)
            profile = root / "profiles.json"
            write_bundle(source, profile)
            routes, capacities, reserve, bundle, cut = load_bundle(profile)
            self.assertEqual(cut, 1)
            self.assertEqual(bundle["route_layers"]["R2"][0], ["op12-prefix", 0, 1])
            self.assertEqual([point.batch_size for point in routes[1].points], [1, 24, 32])
            self.assertEqual(capacities["op15-mid"], 32)
            self.assertEqual(reserve["cuda-tail"], 0)

    def test_worker_range_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _selection = self.fixture(root)
            value = __import__("json").loads(source.read_text(encoding="ascii"))
            value["workers"]["op15-mid"]["layer_start"] = 2
            source.write_bytes(canonical_bytes(value))
            with self.assertRaisesRegex(ProfileError, "worker identity"):
                write_bundle(source, root / "profiles.json")

    def test_forged_selection_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, selection = self.fixture(root)
            value = __import__("json").loads(selection.read_text(encoding="ascii"))
            value["selected"]["cut_layer"] = 2
            selection.write_bytes(canonical_bytes(value))
            source_value = __import__("json").loads(source.read_text(encoding="ascii"))
            source_value["selection"]["sha256"] = "sha256:" + sha256_file(selection)
            source.write_bytes(canonical_bytes(source_value))
            with self.assertRaisesRegex(ProfileError, "canonical"):
                write_bundle(source, root / "profiles.json")


if __name__ == "__main__":
    unittest.main()
