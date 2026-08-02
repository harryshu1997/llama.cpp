#!/usr/bin/env python3

from __future__ import annotations

import math
import unittest

from mixed_prefill_decode import (
    DECODE_TOKEN,
    PROMPT,
    GateError,
    activation_metrics,
    evaluate,
    run_mixed,
    run_serial,
)

from stage_v3_client import BatchResult, Hello, ProtocolError, Status


class FakeClient:
    def __init__(self, bias: float = 0.0, nonfinite: bool = False):
        self.bias = bias
        self.nonfinite = nonfinite
        self.active: dict[int, tuple[int, int, int]] = {}
        self.batch_sizes: list[int] = []
        self._hello = None

    def hello(self) -> Hello:
        self._hello = Hello(0, 8, 48, 4, 4, 64, 64, 64, 15)
        return self._hello

    def status(self) -> Status:
        return Status(len(self.active), 4, False)

    def batch(self, rows):
        self.batch_sizes.append(len(rows))
        results = []
        for row in rows:
            state = self.active.get(row.seq_id)
            if state is None:
                if row.position != 0:
                    raise ProtocolError("new sequence must start at zero")
                state = (row.request_id, row.route_epoch, 0)
            if state != (row.request_id, row.route_epoch, row.position):
                raise ProtocolError("lineage mismatch")
            self.active[row.seq_id] = (
                row.request_id, row.route_epoch, row.position + 1,
            )
            value = float("nan") if self.nonfinite else (
                float(row.token + row.position) + self.bias * len(rows)
            )
            results.append(BatchResult(
                row.request_id, row.route_epoch, row.seq_id, row.position,
                (value, value + 1.0, value + 2.0, value + 3.0), None,
            ))
        return tuple(results)

    def remove(self, seq_id, request_id, route_epoch) -> Status:
        if self.active.get(seq_id, ())[:2] != (request_id, route_epoch):
            raise ProtocolError("remove lineage mismatch")
        del self.active[seq_id]
        return self.status()


class MetricsTests(unittest.TestCase):
    def test_identical(self) -> None:
        metrics = activation_metrics((1.0, 2.0), (1.0, 2.0))
        self.assertEqual(metrics["rel_l2"], 0.0)
        self.assertAlmostEqual(metrics["cosine"], 1.0)

    def test_reject_nonfinite(self) -> None:
        with self.assertRaises(GateError):
            activation_metrics((1.0, 2.0), (math.nan, 2.0))

    def test_reject_shape(self) -> None:
        with self.assertRaises(GateError):
            activation_metrics((1.0,), (1.0, 2.0))


class ExecutionTests(unittest.TestCase):
    def test_serial_and_mixed(self) -> None:
        reference = run_serial(FakeClient())
        treatment_client = FakeClient()
        treatment, event, _ = run_mixed(treatment_client)
        metrics, passed = evaluate(reference, treatment)
        self.assertTrue(passed)
        self.assertEqual(len(metrics), 5)
        self.assertEqual(event["physical_batch_size"], 5)
        self.assertEqual(
            [row["phase"] for row in event["rows"]],
            ["decode", "prefill", "prefill", "prefill", "prefill"],
        )
        self.assertEqual(treatment_client.batch_sizes, [4, 5])

    def test_batch_dependent_output_fails_numeric_gate(self) -> None:
        reference = run_serial(FakeClient())
        treatment, _, _ = run_mixed(FakeClient(bias=1000.0))
        _, passed = evaluate(reference, treatment)
        self.assertFalse(passed)

    def test_nonfinite_fails_closed(self) -> None:
        with self.assertRaises(GateError):
            run_mixed(FakeClient(nonfinite=True))

    def test_frozen_tokens(self) -> None:
        self.assertEqual(len(PROMPT), 4)
        self.assertIs(type(DECODE_TOKEN), int)


if __name__ == "__main__":
    unittest.main()
