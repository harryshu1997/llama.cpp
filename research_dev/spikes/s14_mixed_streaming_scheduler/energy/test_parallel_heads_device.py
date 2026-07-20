#!/usr/bin/env python3

from __future__ import annotations

import copy
import unittest

import parallel_heads_device as P


def cert(start: int, end: int, buffer: str) -> dict:
    return {
        "status": "SCHEDULED_PLACEMENT_OK",
        "layer_start": start,
        "layer_end": end,
        "missing_buffer_compute_nodes": 0,
        "compute_by_buffer_type": {buffer: 10},
        "compute_by_op_and_buffer": {"MUL_MAT": {buffer: 10}},
    }


def thermal() -> dict:
    return {"valid": True, "sensors_millic": {"nsphmx-0": 40_000}, "max_millic": 40_000}


class ParallelChecksTests(unittest.TestCase):
    def inputs(self):
        rows = [
            {"stream_index": 0, "batch_size": 2, "token_ids": [1]},
            {"stream_index": 1, "batch_size": 2, "token_ids": [1]},
        ]
        thermals = {name: {"start": thermal(), "end": thermal()} for name in ("op15", "op12")}
        return rows, cert(0, 6, "HTP0"), cert(0, 6, "HTP0"), cert(6, 48, "CUDA0"), thermals

    def test_accepts_complete_result(self) -> None:
        rows, op15, op12, host, thermals = self.inputs()
        self.assertTrue(all(P.validate_result(0, rows, [1], 2, op15, op12, host, thermals).values()))

    def test_rejects_one_wrong_stream(self) -> None:
        rows, op15, op12, host, thermals = self.inputs()
        rows[1]["token_ids"] = [2]
        self.assertFalse(P.validate_result(0, rows, [1], 2, op15, op12, host, thermals)["all_tokens_match_reference"])

    def test_rejects_host_cpu_gemm(self) -> None:
        rows, op15, op12, host, thermals = self.inputs()
        host = copy.deepcopy(host)
        host["compute_by_op_and_buffer"] = {"MUL_MAT": {"CPU": 10}}
        self.assertFalse(P.validate_result(0, rows, [1], 2, op15, op12, host, thermals)["host_tail_cert_ok"])

    def test_rejects_wrong_phone_range(self) -> None:
        rows, op15, op12, host, thermals = self.inputs()
        op12["layer_end"] = 8
        self.assertFalse(P.validate_result(0, rows, [1], 2, op15, op12, host, thermals)["op12_cert_ok"])


if __name__ == "__main__":
    unittest.main()
