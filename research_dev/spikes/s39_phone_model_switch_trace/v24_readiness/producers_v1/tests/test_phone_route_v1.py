#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "phone_route_v1.py"
SPEC = importlib.util.spec_from_file_location("phone_route_v1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
phone = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phone)


def requests() -> list[dict]:
    result = []
    for item_index in range(64):
        length = 10 + item_index % 8
        result.append({
            "item_index": item_index,
            "prompt_sha256": f"{item_index:064x}",
            "prompt_utf8_base64": "QQ==",
            "prompt_utf8_bytes": 1,
            "request_id": item_index % 8 + 1,
            "seq_id": item_index % 8,
            "token_ids": list(range(item_index + 1, item_index + 1 + length)),
        })
    return result


def group(values: list[dict], group_index: int) -> dict:
    item_indices = list(range(group_index * 8, group_index * 8 + 8))
    rows = []
    for item_index in item_indices:
        request = values[item_index]
        for position, token_id in enumerate(request["token_ids"]):
            rows.append(
                (
                    position,
                    item_index,
                    request["request_id"],
                    request["seq_id"],
                    token_id,
                )
            )
    rows.sort(key=lambda value: (value[0], value[1]))
    partitions = []
    current = []
    for position in sorted({row[0] for row in rows}):
        wave = [row for row in rows if row[0] == position]
        if current and len(current) + len(wave) > 64:
            partitions.append(current)
            current = []
        current.extend(wave)
    if current:
        partitions.append(current)
    prefill = [
        {
            "call_index": call_index,
            "rows": [
                {
                    "item_index": item_index,
                    "position": position,
                    "request_id": request_id,
                    "seq_id": seq_id,
                    "token_id": token_id,
                }
                for position, item_index, request_id, seq_id, token_id in partition
            ],
        }
        for call_index, partition in enumerate(partitions)
    ]
    decode = [
        {
            "call_index": len(prefill) + ordinal,
            "continuation_input_ordinal": ordinal,
            "continuation_output_ordinal": ordinal + 1,
            "rows": [
                {
                    "item_index": item_index,
                    "position": len(values[item_index]["token_ids"]) + ordinal,
                    "request_id": values[item_index]["request_id"],
                    "seq_id": values[item_index]["seq_id"],
                }
                for item_index in item_indices
            ],
        }
        for ordinal in range(7)
    ]
    return {
        "decode_calls": decode,
        "group_index": group_index,
        "item_indices": item_indices,
        "prefill_partitions": prefill,
    }


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def batch(self, rows):
        self.calls.append(copy.deepcopy(rows))
        return [row[4] + 1 for row in rows]


class PhoneRouteTests(unittest.TestCase):
    def test_persisted_canonical_all64_history_loads(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "results"
            / "prephase_20260726T0915Z"
            / "token-history.json"
        )
        raw = path.read_bytes()
        self.assertEqual(
            phone.sha256(raw),
            "3755944451dac26ee046f6fc89b1ebd5ea060150d0447d739707138bdcb93314",
        )
        value, reopened = phone.load_histories(
            path,
            phone.sha256(raw),
            phone.MODEL_SHA256,
        )
        self.assertEqual(reopened, raw)
        self.assertEqual(value["prefill_chunking"], "WHOLE_POSITION_WAVES_MAX_64_ROWS")

    def setUp(self) -> None:
        self.requests = requests()
        self.group = group(self.requests, 0)
        self.history = {
            "quality_groups": [self.group],
            "requests": self.requests,
        }

    def test_exact_group_runs_seven_decode_calls(self):
        client = FakeClient()
        continuations, calls, frames, receipt = phone.run_history_group(
            client,
            self.history,
            self.group,
            list(range(1001, 1009)),
            17,
            23,
        )
        prefill_count = len(self.group["prefill_partitions"])
        self.assertEqual(len(calls), prefill_count + 7)
        self.assertEqual(len(client.calls), prefill_count + 7)
        self.assertEqual(len(frames), prefill_count + 7)
        self.assertEqual(
            set(calls[0]),
            {
                "call_index",
                "n_seqs",
                "n_tokens",
                "phase",
                "positions",
                "request_ids",
                "seq_ids",
            },
        )
        self.assertEqual(calls[0]["request_ids"],
                         [row["request_id"]
                          for row in self.group["prefill_partitions"][0]["rows"]])
        self.assertEqual([frame["call_index"] for frame in frames],
                         list(range(23, 23 + prefill_count + 7)))
        self.assertTrue(all(len(tokens) == 8 for tokens in continuations))
        self.assertEqual(receipt["continuations"], continuations)
        self.assertEqual(receipt["item_indices"], list(range(8)))

    def test_final_prefill_output_is_first_continuation(self):
        client = FakeClient()
        continuations, _, _, _ = phone.run_history_group(
            client,
            self.history,
            self.group,
            list(range(1001, 1009)),
            17,
            0,
        )
        for seq_id, tokens in enumerate(continuations):
            final_prompt_token = self.requests[seq_id]["token_ids"][-1]
            self.assertEqual(tokens, list(range(final_prompt_token + 1,
                                                final_prompt_token + 9)))

    def test_group_validator_rejects_call_mutation(self):
        bad = copy.deepcopy(self.group)
        bad["decode_calls"][0]["call_index"] += 1
        with self.assertRaisesRegex(phone.CaptureError, "history.*decode"):
            phone.validate_history_group(bad, 0, self.requests)

    def test_group_validator_rejects_row_order_mutation(self):
        bad = copy.deepcopy(self.group)
        bad["prefill_partitions"][0]["rows"].reverse()
        with self.assertRaisesRegex(phone.CaptureError, "history.*prefill"):
            phone.validate_history_group(bad, 0, self.requests)

    def test_runtime_rejects_request_identity_mutation(self):
        bad = copy.deepcopy(self.group)
        bad["decode_calls"][0]["rows"][0]["request_id"] = 8
        with self.assertRaisesRegex(phone.CaptureError, "E_GROUP_DECODE_REQUEST"):
            phone.run_history_group(
                FakeClient(),
                self.history,
                bad,
                list(range(1001, 1009)),
                17,
                0,
            )

    def test_mechanics_and_bridge_use_one_based_request_ids(self):
        continuations, calls, _, _ = phone.run_history_group(
            FakeClient(),
            self.history,
            self.group,
            list(range(1001, 1009)),
            17,
            0,
        )
        rows = phone.make_mechanics_rows(
            self.requests[:8],
            continuations,
            calls,
            "a" * 64,
            1000,
        )
        self.assertEqual(
            [row["request_id"] for row in rows[1:]],
            list(range(1, 9)),
        )
        bridge = phone.make_bridge_rows(rows, 2000, "phase-1")
        normalized = {
            "acquisition_id": "phase-1",
            **{key: value for key, value in rows[1].items() if key != "event_ns"},
            "role": "model.qwen3-14b-q4_k_m.mechanics.phone",
        }
        self.assertEqual(
            bridge[0]["phone_request_sha256"],
            phone.sha256(phone.canonical_bytes(normalized)),
        )

    def test_bridge_rejects_zero_based_request_ids(self):
        rows = [{"kind": "meta"}] + [
            {"request_id": request_id}
            for request_id in range(8)
        ]
        with self.assertRaisesRegex(phone.CaptureError, "bridge.request_ids"):
            phone.make_bridge_rows(rows, 2000, "phase-1")


if __name__ == "__main__":
    unittest.main()
