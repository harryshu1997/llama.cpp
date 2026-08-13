#!/usr/bin/env python3

from __future__ import annotations

import unittest

from research_dev.scheduler import (
    OperatorSplitError,
    ParallelSplitShapeMeasurement,
    balance_parallel_split,
)


def measurement(
    tokens: int,
    calls: int,
    rpc_us: int,
    compute_us: int,
    host_us: int,
) -> ParallelSplitShapeMeasurement:
    return ParallelSplitShapeMeasurement(
        tokens=tokens,
        calls=calls,
        total_columns=15360,
        phone_columns=6144,
        phone_rpc_us=rpc_us,
        phone_compute_us=compute_us,
        host_us=host_us,
        evidence_ids=("fixed-width-abba",),
    )


class ParallelSplitBalanceTests(unittest.TestCase):
    def test_balances_measured_shapes_and_falls_back_elsewhere(self) -> None:
        balance = balance_parallel_split(
            (
                measurement(1, 100, 4300, 2900, 7400),
                measurement(5, 100, 8220, 6082, 7565),
                measurement(8, 1000, 9033, 6164, 7655),
                measurement(11, 10, 10761, 6935, 7824),
            ),
            candidate_columns=(5120, 5632, 6144),
            alternate_columns=(5632,),
            physical_max_tokens=16,
            policy_max_tokens=512,
            resident_phone_columns=6144,
            column_quantum=1024,
            minimum_saving_ppm=10_000,
        )
        selected = {
            row.tokens: row.selected_columns for row in balance.estimates
        }
        self.assertEqual(selected, {1: 6144, 5: 5632, 8: 5120, 11: 5120})
        self.assertEqual(
            balance.table,
            "4:6144,5:5632,7:6144,8:5120,10:6144,"
            "11:5120,16:6144,512:0",
        )
        self.assertGreater(balance.predicted_saving_ppm, 0)

    def test_rejects_measurement_without_host_remainder(self) -> None:
        with self.assertRaisesRegex(
            OperatorSplitError, "must leave host work"
        ):
            ParallelSplitShapeMeasurement(
                tokens=1,
                calls=1,
                total_columns=6144,
                phone_columns=6144,
                phone_rpc_us=1,
                phone_compute_us=1,
                host_us=1,
                evidence_ids=("evidence",),
            )


if __name__ == "__main__":
    unittest.main()
