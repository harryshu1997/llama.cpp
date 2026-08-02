#!/usr/bin/env python3

import unittest

from verify_cp4 import GateError, convergence_evidence, finite_boundary


def event(routes, upstreams, batch_size=None):
    if batch_size is None:
        batch_size = len(routes)
    return {
        "routes": list(routes),
        "upstream_workers": list(upstreams),
        "batch_size": batch_size,
    }


def fixed_streams():
    return {
        "cuda-prefix": [],
        "cuda-mid": [],
        "op12-prefix": [],
        "op15-mid": [],
        "cuda-tail": [],
    }


class VerifyCp4Tests(unittest.TestCase):
    def runs(self):
        op15 = fixed_streams()
        op15["op15-mid"] = [
            event(("R1", "R2"), ("cuda-prefix", "op12-prefix")),
        ]
        tail = fixed_streams()
        tail["cuda-tail"] = [
            event(
                ("R0", "R1", "R2"),
                ("cuda-mid", "op15-mid", "op15-mid"),
            ),
        ]
        b4 = fixed_streams()
        b4["op12-prefix"] = [event(("R2",) * 4, ("TOKEN_SOURCE",) * 4, 4)]
        b4["op15-mid"] = [event(("R2",) * 4, ("op12-prefix",) * 4, 4)]
        b4["cuda-tail"] = [event(("R2",) * 4, ("op15-mid",) * 4, 4)]
        return {
            "r1-r2-shared-op15": {"batch_events": op15},
            "r0-r1-r2-shared-tail": {"batch_events": tail},
            "r2-b4": {"batch_events": b4},
        }

    def test_accepts_cross_source_and_shared_tail_batches(self):
        result = convergence_evidence(self.runs())
        self.assertEqual(result["op15"]["mixed_source_batches"], 1)
        self.assertEqual(result["cuda_tail"]["mixed_route_batches"], 1)
        self.assertEqual(
            result["r2_b4"]["maximum_physical_batch_by_worker"],
            {"op12-prefix": 4, "op15-mid": 4, "cuda-tail": 4},
        )

    def test_rejects_route_interleaving_without_mixed_source_batch(self):
        runs = self.runs()
        runs["r1-r2-shared-op15"]["batch_events"]["op15-mid"] = [
            event(("R1",), ("cuda-prefix",)),
            event(("R2",), ("op12-prefix",)),
        ]
        with self.assertRaisesRegex(GateError, "mixed-source"):
            convergence_evidence(runs)

    def test_boundary_values_must_be_finite_numbers(self):
        self.assertTrue(finite_boundary({
            "l2_norm": 1.0,
            "minimum": -2.0,
            "maximum": 3.0,
        }))
        self.assertFalse(finite_boundary({
            "l2_norm": float("inf"),
            "minimum": -2.0,
            "maximum": 3.0,
        }))


if __name__ == "__main__":
    unittest.main()
