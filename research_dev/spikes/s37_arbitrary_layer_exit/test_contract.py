#!/usr/bin/env python3

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from arbitrary_topology import CUTS, PHONE_WORKERS, route_specs
from contract import ContractError, SCHEMA, validate, with_digest


def event(active_range: list[int]) -> dict[str, object]:
    return {"active_range": active_range}


def valid_result() -> dict[str, object]:
    signature = [532, 532, 532, 532]
    rows = []
    request_id = 1
    for route_id, phone, cut in route_specs():
        repetitions = []
        for _ in range(2):
            repetitions.append({
                "batch_size": 8,
                "tokens": list(signature),
                "request_ids": list(range(request_id, request_id + 8)),
                "events": {
                    phone: [event([0, cut])],
                    "tail": [event([cut, 48])],
                },
            })
            request_id += 8
        rows.append({
            "route_id": route_id,
            "phone": phone,
            "cut": cut,
            "repetitions": repetitions,
        })
    return with_digest({
        "schema": SCHEMA,
        "status": "PASS",
        "physical": True,
        "token_signature": signature,
        "workers": {},
        "routes": rows,
        "final_state": {
            "route_pins": {},
            "software_leases": {"cuda": 0, "op12": 0, "op15": 0, "tail": 0},
            "active_sequences": {"cuda": 0, "op12": 0, "op15": 0, "tail": 0},
        },
        "final_workers": {},
    })


class ContractTests(unittest.TestCase):
    def test_route_specs_cover_every_joint_cut(self) -> None:
        self.assertEqual(len(route_specs()), len(CUTS) * len(PHONE_WORKERS))
        self.assertEqual(
            {(phone, cut) for _route, phone, cut in route_specs()},
            {(phone, cut) for phone in PHONE_WORKERS for cut in CUTS},
        )

    def test_valid_result_passes(self) -> None:
        validate(valid_result())

    def test_missing_route_fails(self) -> None:
        value = valid_result()
        unhashed = dict(value)
        del unhashed["result_hash"]
        unhashed["routes"] = unhashed["routes"][:-1]
        with self.assertRaises(ContractError):
            validate(with_digest(unhashed))

    def test_wrong_tail_range_fails(self) -> None:
        value = valid_result()
        unhashed = copy.deepcopy(value)
        del unhashed["result_hash"]
        unhashed["routes"][0]["repetitions"][0]["events"]["tail"][0][
            "active_range"
        ] = [5, 48]
        with self.assertRaises(ContractError):
            validate(with_digest(unhashed))

    def test_token_mismatch_fails(self) -> None:
        value = valid_result()
        unhashed = copy.deepcopy(value)
        del unhashed["result_hash"]
        unhashed["routes"][0]["repetitions"][0]["tokens"][0] = 7
        with self.assertRaises(ContractError):
            validate(with_digest(unhashed))

    def test_live_state_fails(self) -> None:
        value = valid_result()
        unhashed = copy.deepcopy(value)
        del unhashed["result_hash"]
        unhashed["final_state"]["active_sequences"]["op12"] = 1
        with self.assertRaises(ContractError):
            validate(with_digest(unhashed))


if __name__ == "__main__":
    unittest.main()
