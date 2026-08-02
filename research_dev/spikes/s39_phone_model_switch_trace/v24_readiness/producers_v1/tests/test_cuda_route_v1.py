#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "cuda_route_v1.py"
SPEC = importlib.util.spec_from_file_location("cuda_route_v1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
route = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(route)


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def batch(self, rows):
        self.calls.append(rows)
        return [row[4] + 1 for row in rows]


def history_fixture() -> dict:
    requests = [
        {
            "item_index": item_index,
            "prompt_sha256": f"{item_index + 1:064x}",
            "prompt_utf8_base64": "QQ==",
            "prompt_utf8_bytes": 1,
            "request_id": item_index % 8 + 1,
            "seq_id": item_index % 8,
            "token_ids": [1000 + item_index, 2000 + item_index],
        }
        for item_index in range(64)
    ]
    groups = []
    for group_index in range(8):
        items = list(range(group_index * 8, group_index * 8 + 8))
        prefill_rows = [
            {
                "item_index": item_index,
                "position": position,
                "request_id": requests[item_index]["request_id"],
                "seq_id": requests[item_index]["seq_id"],
                "token_id": requests[item_index]["token_ids"][position],
            }
            for position in range(2)
            for item_index in items
        ]
        groups.append({
            "decode_calls": [
                {
                    "call_index": 1 + ordinal,
                    "continuation_input_ordinal": ordinal,
                    "continuation_output_ordinal": ordinal + 1,
                    "rows": [
                        {
                            "item_index": item_index,
                            "position": 2 + ordinal,
                            "request_id": requests[item_index]["request_id"],
                            "seq_id": requests[item_index]["seq_id"],
                        }
                        for item_index in items
                    ],
                }
                for ordinal in range(7)
            ],
            "group_index": group_index,
            "item_indices": items,
            "prefill_partitions": [{
                "call_index": 0,
                "rows": prefill_rows,
            }],
        })
    return {
        "mechanics_b8": groups[0],
        "quality_groups": groups,
        "requests": requests,
    }


class CudaRouteTests(unittest.TestCase):
    def test_persisted_canonical_all64_history_loads(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "results"
            / "prephase_20260726T0915Z"
            / "token-history.json"
        )
        value, raw = route.load_histories(path)
        self.assertEqual(
            route.sha256(raw),
            "3755944451dac26ee046f6fc89b1ebd5ea060150d0447d739707138bdcb93314",
        )
        self.assertEqual(value["prefill_chunking"], "WHOLE_POSITION_WAVES_MAX_64_ROWS")

    def test_exact_group_uses_final_prefill_plus_seven_decode_calls(self):
        history = history_fixture()
        client = FakeClient()
        continuations, calls, receipt = route.run_history_group(
            client,
            history,
            history["quality_groups"][0],
            list(range(1001, 1009)),
            9,
            0,
        )
        self.assertEqual(len(client.calls), 8)
        self.assertEqual([call["phase"] for call in calls], ["prefill"] + ["decode"] * 7)
        self.assertEqual(len(receipt["call_receipts"]), 8)
        self.assertTrue(all(len(tokens) == 8 for tokens in continuations))
        self.assertEqual(
            continuations[0][0],
            history["requests"][0]["token_ids"][-1] + 1,
        )

    def test_group_mutations_fail_closed(self):
        history = history_fixture()
        route.validate_history_group(
            history["quality_groups"][0],
            0,
            history["requests"],
        )
        history["quality_groups"][0]["decode_calls"][0][
            "continuation_output_ordinal"
        ] = 2
        with self.assertRaisesRegex(route.CaptureError, "decode_calls"):
            route.validate_history_group(
                history["quality_groups"][0],
                0,
                history["requests"],
            )

    def test_v24_contract_constants_are_exact(self):
        self.assertEqual(
            route.SCHEMA,
            "s39-cp0-r1-v24-cuda-route-raw-v1",
        )
        self.assertEqual(
            route.PLAN_SCHEMA,
            "s39-cp0-r1-v24-cuda-route-launch-v1",
        )
        self.assertEqual(route.N_CTX_SEQ, 512)
        self.assertEqual(route.SERVING_ENVELOPE["n_ctx_seq"], 512)


if __name__ == "__main__":
    unittest.main()
