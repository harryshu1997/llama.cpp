#!/usr/bin/env python3

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from workloads import (
    MECHANICS_SCOPE,
    OBSERVED_SCOPE,
    WorkloadError,
    content_digest,
    dense_mechanics,
    deterministic_three_class,
    load_s23,
    observed_context_cohort,
    validate,
)


HERE = Path(__file__).resolve().parent
S23_TRACE = HERE.parent / "s23_dense_trace_runtime/burstgpt-dense-60.json"


class WorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s23 = load_s23(S23_TRACE)

    def test_deterministic_trace_has_four_requests_per_route(self) -> None:
        value = deterministic_three_class()
        counts = {
            route_id: sum(
                row["route_hint"] == route_id for row in value["requests"]
            )
            for route_id in ("R0", "R1", "R2")
        }
        self.assertEqual(counts, {"R0": 4, "R1": 4, "R2": 4})
        validate(value)

    def test_dense_mechanics_preserves_21_17_22_arrivals(self) -> None:
        value = dense_mechanics(self.s23)
        self.assertEqual(value["scope"], MECHANICS_SCOPE)
        self.assertEqual(
            [
                sum(row["arrival_us"] == arrival for row in value["requests"])
                for arrival in (0, 1_000_000, 2_000_000)
            ],
            [21, 17, 22],
        )
        self.assertTrue(all(
            row["input_tokens"] == 1 and row["output_steps"] == 4
            for row in value["requests"]
        ))

    def test_observed_context_cohort_is_exactly_28(self) -> None:
        value = observed_context_cohort(self.s23)
        self.assertEqual(value["scope"], OBSERVED_SCOPE)
        self.assertEqual(len(value["requests"]), 28)
        self.assertTrue(all(
            row["observed_input_tokens"] + row["observed_output_tokens"]
            <= 600
            for row in value["requests"]
        ))
        self.assertEqual(
            sum(row["input_tokens"] for row in value["requests"]),
            12_586,
        )
        self.assertEqual(
            sum(row["output_steps"] for row in value["requests"]),
            958,
        )

    def test_rehashed_proxy_mutation_fails_contract(self) -> None:
        value = dense_mechanics(self.s23)
        mutated = copy.deepcopy(value)
        mutated["requests"][0]["input_tokens"] = 2
        mutated["trace_hash"] = content_digest(mutated)
        with self.assertRaisesRegex(WorkloadError, "mechanics trace"):
            validate(mutated)

    def test_unhashed_mutation_fails_digest(self) -> None:
        value = deterministic_three_class()
        value["requests"][0]["arrival_us"] = 1
        with self.assertRaisesRegex(WorkloadError, "content digest"):
            validate(value)


if __name__ == "__main__":
    unittest.main()
