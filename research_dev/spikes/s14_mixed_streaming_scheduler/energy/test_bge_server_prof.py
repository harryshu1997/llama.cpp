#!/usr/bin/env python3

from __future__ import annotations

import unittest

import bge_server_prof as prof


class ExecutedGraphBoundTests(unittest.TestCase):
    def test_batch_one_has_no_masked_excess(self) -> None:
        result = prof.executed_graph_bound({32: 31}, [1])
        point = result["points"][0]
        self.assertEqual(point["total_tokens"], 31)
        self.assertEqual(point["masked_excess_gflops"], 0.0)

    def test_physical_batch_uses_total_token_attention(self) -> None:
        result = prof.executed_graph_bound({32: 31}, [1, 4])
        b1, b4 = result["points"]
        self.assertEqual(b4["total_tokens"], 124)
        self.assertGreater(b4["masked_excess_gflops"], 0.0)
        self.assertGreater(b4["executed_fwd_gflops"], b4["useful_fwd_gflops"])
        self.assertGreater(b4["weight_only_ai_upper_bound"], b1["weight_only_ai_upper_bound"])

    def test_bound_refuses_regime_classification(self) -> None:
        result = prof.executed_graph_bound({128: 132}, [2])
        self.assertEqual(result["classification"], "UNCLASSIFIED_USE_MEASURED_KNEE")
        self.assertEqual(result["points"][0]["regime"], "UNCLASSIFIED_USE_MEASURED_KNEE")
        self.assertIn("upper bound", result["note"])


if __name__ == "__main__":
    unittest.main()
