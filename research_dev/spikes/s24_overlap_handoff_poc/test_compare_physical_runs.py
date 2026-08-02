#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from array import array
from pathlib import Path

from compare_physical_runs import ComparisonError, compare_runs


def write_activation(path: Path, values: list[float], worker: str) -> dict:
    payload = array("f", values).tobytes()
    path.write_bytes(payload)
    return {
        "worker": worker,
        "layer_end": 8,
        "position": 0,
        "elements": len(values),
        "bytes": len(payload),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "l2_norm": 1.0,
        "minimum": min(values),
        "maximum": max(values),
        "path": str(path),
    }


def write_run(path: Path, activation: dict, tokens: list[int], route: str) -> None:
    path.write_text(json.dumps({
        "schema": "s24-fixed-diamond-physical-v1",
        "status": "RUN_COMPLETE",
        "runtime": {
            "requests": [{
                "request_id": 1,
                "prompt_length": 1,
                "output_steps": len(tokens),
                "output_tokens": tokens,
                "route_id": route,
                "boundary_activations": [activation],
            }],
        },
    }), encoding="ascii")


class ComparisonTests(unittest.TestCase):
    def test_exact_pair_passes_screen_but_not_quality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref_activation = write_activation(root / "ref.f32", [1.0, 2.0], "cuda-prefix")
            cand_activation = write_activation(root / "cand.f32", [1.0, 2.0], "op12-prefix")
            reference = root / "reference.json"
            candidate = root / "candidate.json"
            write_run(reference, ref_activation, [10, 11], "R0")
            write_run(candidate, cand_activation, [10, 11], "R2")
            report = compare_runs(reference, candidate, "r2-vs-r0", 0.005)
            self.assertTrue(report["boundary_gate_pass"])
            self.assertTrue(report["token_screen_pass"])
            self.assertFalse(report["quality_certified"])

    def test_activation_hash_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref_activation = write_activation(root / "ref.f32", [1.0, 2.0], "cuda-prefix")
            cand_path = root / "cand.f32"
            cand_activation = write_activation(cand_path, [1.0, 2.0], "op12-prefix")
            reference = root / "reference.json"
            candidate = root / "candidate.json"
            write_run(reference, ref_activation, [10], "R0")
            write_run(candidate, cand_activation, [10], "R2")
            cand_path.write_bytes(b"broken")
            with self.assertRaisesRegex(ComparisonError, "size mismatch"):
                compare_runs(reference, candidate, "mutated", 0.005)


if __name__ == "__main__":
    unittest.main()
