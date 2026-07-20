#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import op12_profile_adapter as A


class OP12AdapterTests(unittest.TestCase):
    def test_requires_seven_profiles(self) -> None:
        with self.assertRaisesRegex(A.OP12ProfileError, "seven profiles"):
            A.load_op12_head_route([], 1)

    def test_rejects_duplicate_json_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"schema":1,"schema":2}', encoding="ascii")
            with self.assertRaisesRegex(A.OP12ProfileError, "duplicate key"):
                A._load(path)

    def test_rejects_overlapping_intervals(self) -> None:
        with self.assertRaisesRegex(A.OP12ProfileError, "intervals overlap"):
            A._check_disjoint([(1, 4), (3, 5)])

    def test_accepts_disjoint_intervals(self) -> None:
        A._check_disjoint([(5, 8), (1, 4), (8, 9)])

    def test_placement_rejects_cpu_gemm(self) -> None:
        row = {
            "layer_range": [0, 8], "batch": 1, "n_requests_measured": 8,
            "placement_status": "SCHEDULED_PLACEMENT_OK", "missing_buffer_compute_nodes": 0,
            "host_returncode": 0, "token_match_vs_mono": True,
            "all_tokens_match_vs_mono": True,
            "token_ids": [1], "token_ids_by_request": [[1] for _ in range(8)],
            "placement_cert": {
                "status": "SCHEDULED_PLACEMENT_OK", "layer_start": 0, "layer_end": 8,
                "compute_by_op_and_buffer": {"MUL_MAT": {"CPU": 1}},
            },
        }
        with self.assertRaisesRegex(A.OP12ProfileError, "undeclared CPU work"):
            A._placement(row, 8, 8, [1])

    def test_placement_recomputes_token_booleans(self) -> None:
        row = {
            "layer_range": [0, 8], "batch": 1, "n_requests_measured": 1,
            "placement_status": "SCHEDULED_PLACEMENT_OK", "missing_buffer_compute_nodes": 0,
            "host_returncode": 0, "token_match_vs_mono": True,
            "all_tokens_match_vs_mono": True, "token_ids": [2],
            "token_ids_by_request": [[2]],
            "placement_cert": {
                "status": "SCHEDULED_PLACEMENT_OK", "layer_start": 0, "layer_end": 8,
                "compute_by_op_and_buffer": {"MUL_MAT": {"HTP0": 1}},
            },
        }
        with self.assertRaisesRegex(A.OP12ProfileError, "stored token evidence"):
            A._placement(row, 1, 8, [1])

    def test_thermal_rejects_forged_maximum(self) -> None:
        data = {"thermal": {
            "start_max_millic": 60_000, "end_max_millic": 85_000,
            "start": {"valid": True, "sensors_millic": {"x": 50_000}, "max_millic": 40_000},
            "end": {"valid": True, "sensors_millic": {"x": 50_000}, "max_millic": 50_000},
        }}
        with self.assertRaisesRegex(A.OP12ProfileError, "thermal envelope"):
            A._thermal(data)

    def test_raw_evidence_recomputes_median(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cert = {"status": "SCHEDULED_PLACEMENT_OK"}
            host = root / "host.stderr"
            phone = root / "phone.log"
            rows = [
                {"request_index": 0, "batch_size": 1, "token_ids": [1],
                 "stage_a_us": 10, "host_us": 20, "request_wall_us": 30},
                {"request_index": 1, "batch_size": 1, "token_ids": [1],
                 "stage_a_us": 11, "host_us": 21, "request_wall_us": 31},
            ]
            host.write_text("".join("ROUTEJSON " + json.dumps(row) + "\n" for row in rows), encoding="ascii")
            phone.write_text("PLACEMENTCERT " + json.dumps(cert) + "\n", encoding="ascii")
            row = {"stage_a_us_p50": 11, "host_us_p50": 21,
                   "request_wall_us_p50": 999, "placement_cert": cert}
            with self.assertRaisesRegex(A.OP12ProfileError, "stored timing"):
                A._raw_evidence({".stderr": host, ".log": phone}, row, [1], 2)

    def test_raw_evidence_rejects_token_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host = root / "host.stderr"
            phone = root / "phone.log"
            host.write_text(
                'ROUTEJSON {"request_index":0,"batch_size":1,"token_ids":[2],'
                '"stage_a_us":10,"host_us":20,"request_wall_us":30}\n',
                encoding="ascii",
            )
            phone.write_text('PLACEMENTCERT {"status":"SCHEDULED_PLACEMENT_OK"}\n', encoding="ascii")
            with self.assertRaisesRegex(A.OP12ProfileError, "token correctness"):
                A._raw_evidence(
                    {".stderr": host, ".log": phone},
                    {"placement_cert": {"status": "SCHEDULED_PLACEMENT_OK"}},
                    [1], 1,
                )


if __name__ == "__main__":
    unittest.main()
