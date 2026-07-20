#!/usr/bin/env python3

from __future__ import annotations

import copy
import unittest

from experiment_contract import (
    BATCHES_PER_ROUND, CONTROL_BATCH, FULL_ROUNDS, FULL_SCHEDULE, N_GEN,
    OP12_EXCHANGE_CREDIT, PHONE_BATCH, ContractError, summarize,
    validate_result_record,
)


REFERENCE = list(range(N_GEN))


def result(batch: int) -> dict:
    return {
        "batch_size": batch,
        "elapsed_us": 1000,
        "host_pid": 12,
        "launch_id": 1,
        "n_gen": N_GEN,
        "outcome": "completed",
        "request_count": batch,
        "route_wall_us": 900,
        "schema": "layersplit-persistent-result-v1",
        "session_end": "DETACH",
        "token_ids": [REFERENCE] * batch,
    }


def rows() -> list[dict]:
    output = []
    for label, pair in FULL_SCHEDULE:
        row = {
            "label": label,
            "pair": pair,
            "rounds": FULL_ROUNDS,
            "gemma_requests": FULL_ROUNDS * BATCHES_PER_ROUND * CONTROL_BATCH,
            "gemma_tokens": FULL_ROUNDS * BATCHES_PER_ROUND * CONTROL_BATCH * N_GEN,
            "all_tokens_match": True,
            "all_low_inside_bge": True,
            "selected_gpu_peak_memory_mib": 40000,
            "bge": {"encodes": 1000, "latency_samples_us": [100] * 20},
            "power": {
                "energy_nj": 2_000_000_000_000 if label == "P0" else 1_700_000_000_000,
                "uncertainty_nj": 50_000_000_000,
            },
        }
        if label == "P0":
            row.update({
                "control_route_wall_us": [500_000] * FULL_ROUNDS * BATCHES_PER_ROUND,
                "control_windows_us": [[index * 1000, index * 1000 + 900]
                                       for index in range(FULL_ROUNDS * BATCHES_PER_ROUND)],
            })
        else:
            op12_count = min(FULL_ROUNDS, OP12_EXCHANGE_CREDIT)
            op15_count = FULL_ROUNDS * BATCHES_PER_ROUND - op12_count
            row.update({
                "phone_route_wall_us": {
                    "op15": [3_000_000] * op15_count,
                    "op12": [9_000_000] * op12_count,
                },
                "phone_windows_us": {
                    "op15": [[index * 1000, index * 1000 + 900]
                             for index in range(op15_count)],
                    "op12": [[index * 1000, index * 1000 + 950]
                             for index in range(op12_count)],
                },
                "round_overlap_us": [900] * op12_count,
                "route_assignments": [
                    ["op15", "op12" if index < op12_count else "op15"]
                    for index in range(FULL_ROUNDS)
                ],
                "group_completion_us": [[3_000_000, 9_000_000]] * FULL_ROUNDS,
            })
        output.append(row)
    return output


class ContractTests(unittest.TestCase):
    def test_accepts_b32_control(self) -> None:
        validate_result_record(result(CONTROL_BATCH), REFERENCE, CONTROL_BATCH, 1, "DETACH")

    def test_accepts_b32_phone(self) -> None:
        validate_result_record(result(PHONE_BATCH), REFERENCE, PHONE_BATCH, 1, "DETACH")

    def test_rejects_wrong_batch(self) -> None:
        with self.assertRaisesRegex(ContractError, "identity"):
            validate_result_record(
                result(CONTROL_BATCH * 2), REFERENCE, CONTROL_BATCH, 1, "DETACH")

    def test_rejects_one_wrong_token(self) -> None:
        value = result(PHONE_BATCH)
        value["token_ids"][0] = [99] * N_GEN
        with self.assertRaisesRegex(ContractError, "tokens differ"):
            validate_result_record(value, REFERENCE, PHONE_BATCH, 1, "DETACH")

    def test_summary_accepts_two_phone_overlap(self) -> None:
        summary = summarize(rows(), True)
        self.assertTrue(summary["high_priority_gate"])
        self.assertTrue(summary["selected_gpu_board_relief_gate"])

    def test_summary_rejects_nonoverlap(self) -> None:
        value = rows()
        next(row for row in value if row["label"] == "P4")["round_overlap_us"][0] = 0
        with self.assertRaisesRegex(ContractError, "did not overlap"):
            summarize(value, True)

    def test_summary_rejects_op12_slo(self) -> None:
        value = rows()
        next(row for row in value if row["label"] == "P4")["phone_route_wall_us"]["op12"][0] = 12_000_001
        with self.assertRaisesRegex(ContractError, "SLO"):
            summarize(value, True)

    def test_energy_gate_requires_ten_percent(self) -> None:
        value = copy.deepcopy(rows())
        for row in value:
            if row["label"] == "P4":
                row["power"]["energy_nj"] = 1_900_000_000_000
        self.assertFalse(summarize(value, True)["selected_gpu_board_relief_gate"])


if __name__ == "__main__":
    unittest.main()
