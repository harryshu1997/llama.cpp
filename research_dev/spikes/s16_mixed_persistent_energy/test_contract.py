#!/usr/bin/env python3

from __future__ import annotations

import copy
import unittest

from experiment_contract import (
    BATCH, FULL_SCHEDULE, LOW_COHORTS, N_GEN, ContractError, canonical,
    integrate_power, parse_power_jsonl, strict_object, summarize,
    validate_result_record,
)


REFERENCE = list(range(N_GEN))


def result_record() -> dict:
    return {
        "batch_size": BATCH,
        "elapsed_us": 1000,
        "host_pid": 12,
        "launch_id": 1,
        "n_gen": N_GEN,
        "outcome": "completed",
        "request_count": BATCH,
        "route_wall_us": 900,
        "schema": "layersplit-persistent-result-v1",
        "session_end": "DETACH",
        "token_ids": [REFERENCE] * BATCH,
    }


def summary_rows() -> list[dict]:
    rows = []
    for label, pair in FULL_SCHEDULE:
        energy = 2_000_000_000_000 if label == "P0" else 1_500_000_000_000
        rows.append({
            "label": label,
            "pair": pair,
            "gemma_batches": LOW_COHORTS,
            "gemma_requests": LOW_COHORTS * BATCH,
            "gemma_tokens": LOW_COHORTS * BATCH * N_GEN,
            "all_tokens_match": True,
            "all_low_inside_bge": True,
            "low_route_wall_us": [1_000_000] * LOW_COHORTS,
            "bge": {
                "encodes": 100,
                "latency_samples_us": [100 if label == "P0" else 104] * 10,
            },
            "power": {
                "energy_nj": energy,
                "uncertainty_nj": 100_000_000_000,
            },
        })
    return rows


class ContractTests(unittest.TestCase):
    def test_strict_object_rejects_duplicate(self) -> None:
        with self.assertRaisesRegex(ContractError, "duplicate key"):
            strict_object(b'{"a":1,"a":2}\n', "fixture")

    def test_result_record_accepts_exact(self) -> None:
        validate_result_record(result_record(), REFERENCE, "P0", 1, "DETACH")

    def test_result_record_rejects_one_token(self) -> None:
        value = result_record()
        value["token_ids"][0] = [99] * N_GEN
        with self.assertRaisesRegex(ContractError, "tokens differ"):
            validate_result_record(value, REFERENCE, "P0", 1, "DETACH")

    def test_power_reopen_and_quality(self) -> None:
        rows = [
            {
                "t_us": index * 100_000,
                "power_mw": 100_000 + index % 2,
                "power_limit_mw": 300_000,
                "util_milli_pct": 50_000,
                "pstate": "P2",
            }
            for index in range(112)
        ]
        payload = b"".join(canonical(row) for row in rows)
        reopened = parse_power_jsonl(payload)
        result = integrate_power(reopened, 100_000, 10_600_000, True)
        self.assertGreaterEqual(result["independent_updates"], 100)
        self.assertEqual(result["max_sample_gap_us"], 100_000)
        self.assertGreater(result["energy_nj"], 0)

    def test_power_rejects_oversampled_constant_sensor(self) -> None:
        rows = [
            {
                "t_us": index * 100_000,
                "power_mw": 100_000,
                "power_limit_mw": 300_000,
                "util_milli_pct": 50_000,
                "pstate": "P2",
            }
            for index in range(112)
        ]
        with self.assertRaisesRegex(ContractError, "independent updates"):
            integrate_power(rows, 100_000, 10_600_000, True)

    def test_summary_passes_absolute_low_slo(self) -> None:
        result = summarize(summary_rows(), True)
        self.assertTrue(result["high_priority_gate"])
        self.assertTrue(result["selected_gpu_board_relief_gate"])
        self.assertTrue(result["overall_pass"])

    def test_summary_rejects_high_priority_regression(self) -> None:
        rows = summary_rows()
        for row in rows:
            if row["label"] == "P2":
                row["bge"]["latency_samples_us"] = [110] * 10
        result = summarize(rows, True)
        self.assertFalse(result["high_priority_gate"])
        self.assertFalse(result["overall_pass"])

    def test_summary_rejects_low_slo_miss(self) -> None:
        rows = summary_rows()
        rows[0]["low_route_wall_us"][0] = 5_000_001
        with self.assertRaisesRegex(ContractError, "absolute SLO"):
            summarize(rows, True)

    def test_summary_rejects_schedule_mutation(self) -> None:
        rows = summary_rows()
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaisesRegex(ContractError, "schedule"):
            summarize(rows, True)

    def test_energy_uncertainty_can_block_claim(self) -> None:
        rows = copy.deepcopy(summary_rows())
        for row in rows:
            row["power"]["uncertainty_nj"] = 400_000_000_000
        result = summarize(rows, True)
        self.assertFalse(result["selected_gpu_board_relief_gate"])
        self.assertFalse(result["overall_pass"])


if __name__ == "__main__":
    unittest.main()

