#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cut_selector import (
    CutSelectionError,
    build_selection,
    canonical_bytes,
    load_selection,
    validate_measurement,
)
from dynamic_cut_adapter import expected_routes


WORKERS = ("cuda-prefix", "cuda-mid", "op12-prefix", "op15-mid", "cuda-tail")


def measurement(cut: int, op12_us: int, op15_us: int, route_us: int) -> dict:
    ranges = {
        "cuda-prefix": (0, 6),
        "cuda-mid": (6, 8),
        "op12-prefix": (0, cut),
        "op15-mid": (cut, 8),
        "cuda-tail": (8, 48),
    }
    repetitions = []
    for rep in range(2):
        repetitions.append({
            "rep": rep,
            "wall_us": route_us + rep,
            "max_latency_us": route_us + rep,
            "completed_requests": 32,
            "output_tokens": [1, 2, 3, 4],
            "events": {
                name: ([{
                    "status": "OK",
                    "batch_size": 32,
                    "compute_us": (
                        op12_us if name == "op12-prefix"
                        else op15_us if name == "op15-mid"
                        else 100
                    ) + step + rep,
                } for step in range(4)] if name in {
                    "op12-prefix", "op15-mid", "cuda-tail",
                } else [])
                for name in WORKERS
            },
        })
    return {
        "schema": "s31-cut-measurement-v1",
        "status": "MEASUREMENT_COMPLETE",
        "candidate": {
            "cut_layer": cut,
            "op12_layers": [0, cut],
            "op15_layers": [cut, 8],
            "cuda_tail_layers": [8, 48],
        },
        "shape": {
            "input_tokens": 1,
            "output_steps": 4,
            "context": 16,
            "batch_size": 32,
        },
        "gather_us": 50000,
        "launch_evidence": {
            "activation_relay": "DESKTOP_DIRECT_WIFI",
            "phone_session_env": "sha256:" + "a" * 64,
            "desktop_session_env": "sha256:" + "b" * 64,
            "op12_model_sha256": "c" * 64,
            "op15_model_sha256": "c" * 64,
            "desktop_model_sha256": "c" * 64,
        },
        "workers": {
            name: {
                "layer_start": ranges[name][0],
                "layer_end": ranges[name][1],
                "max_streams": 32,
            }
            for name in WORKERS
        },
        "repetitions": repetitions,
        "final_software_state": {
            "runner_pins": {},
            "software_leases": {name: {} for name in WORKERS},
        },
    }


class CutSelectorTests(unittest.TestCase):
    def test_expected_routes_move_only_phone_cut(self) -> None:
        routes = expected_routes(3)
        self.assertEqual(routes["R2"], (
            ("op12-prefix", 0, 3),
            ("op15-mid", 3, 8),
            ("cuda-tail", 8, 48),
        ))
        self.assertEqual(routes["R0"][2], ("cuda-tail", 8, 48))

    def test_selector_minimizes_measured_bottleneck(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                2: measurement(2, 300, 700, 1200),
                3: measurement(3, 450, 500, 1100),
                4: measurement(4, 700, 350, 1300),
            }
            paths = []
            for cut, value in values.items():
                path = root / f"cut-{cut}.json"
                path.write_bytes(canonical_bytes(value))
                paths.append(path)
            result = build_selection(paths, 2_000, (2, 3, 4))
            self.assertEqual(result["selected"]["cut_layer"], 3)
            self.assertEqual(result["selected"]["phone_bottleneck_p95_us"], 504)

    def test_selection_replays_from_bound_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for cut, values in {1: (300, 400), 2: (500, 300)}.items():
                path = root / f"cut-{cut}.json"
                path.write_bytes(canonical_bytes(measurement(cut, *values, 1000 + cut)))
                paths.append(path)
            selection = root / "selection.json"
            selection.write_bytes(canonical_bytes(build_selection(paths, 2000, (1, 2))))
            self.assertEqual(load_selection(selection)["selected"]["cut_layer"], 1)
            value = json.loads(selection.read_text(encoding="ascii"))
            value["selected"]["cut_layer"] = 2
            selection.write_bytes(canonical_bytes(value))
            with self.assertRaisesRegex(CutSelectionError, "canonical"):
                load_selection(selection)

    def test_wrong_worker_range_is_rejected(self) -> None:
        value = measurement(3, 450, 500, 1100)
        value["workers"]["op15-mid"]["layer_start"] = 2
        with self.assertRaisesRegex(CutSelectionError, "HELLO"):
            validate_measurement(value, 2_000)

    def test_mixed_model_identity_is_rejected(self) -> None:
        value = measurement(3, 450, 500, 1100)
        value["launch_evidence"]["op15_model_sha256"] = "d" * 64
        with self.assertRaisesRegex(CutSelectionError, "launch evidence"):
            validate_measurement(value, 2_000)

    def test_non_hex_model_identity_is_rejected(self) -> None:
        value = measurement(3, 450, 500, 1100)
        for field in (
            "op12_model_sha256",
            "op15_model_sha256",
            "desktop_model_sha256",
        ):
            value["launch_evidence"][field] = "z" * 64
        with self.assertRaisesRegex(CutSelectionError, "launch evidence"):
            validate_measurement(value, 2_000)

    def test_sub_b32_event_is_rejected(self) -> None:
        value = measurement(3, 450, 500, 1100)
        value["repetitions"][0]["events"]["op12-prefix"][0]["batch_size"] = 31
        with self.assertRaisesRegex(CutSelectionError, "invalid event"):
            validate_measurement(value, 2_000)

    def test_retained_lease_is_rejected(self) -> None:
        value = measurement(3, 450, 500, 1100)
        value["final_software_state"]["software_leases"]["op12-prefix"] = {"0": 1}
        with self.assertRaisesRegex(CutSelectionError, "retained"):
            validate_measurement(value, 2_000)

    def test_missing_declared_cut_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cut-3.json"
            path.write_bytes(canonical_bytes(measurement(3, 450, 500, 1100)))
            with self.assertRaisesRegex(CutSelectionError, "declared cuts"):
                build_selection([path], 2_000, (2, 3))

    def test_route_slo_is_load_bearing(self) -> None:
        with self.assertRaisesRegex(CutSelectionError, "exceeds"):
            validate_measurement(measurement(3, 450, 500, 1100), 1_000)

    def test_json_is_ascii_and_canonical(self) -> None:
        encoded = canonical_bytes(measurement(3, 450, 500, 1100))
        self.assertEqual(encoded, encoded.decode("ascii").encode("ascii"))
        self.assertEqual(json.loads(encoded)["candidate"]["cut_layer"], 3)


if __name__ == "__main__":
    unittest.main()
