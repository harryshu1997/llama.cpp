#!/usr/bin/env python3

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiles import (  # noqa: E402
    EXPECTED_ROUTES,
    ProfileError,
    SCHEMA,
    validate_profile_bundle,
    with_profile_hash,
)


def bundle() -> dict:
    rows = []
    for route_id, (device, cut) in EXPECTED_ROUTES.items():
        rows.append({
            "route_id": route_id,
            "device": device,
            "cut": cut,
            "batch_points": [
                {
                    "batch_size": 8,
                    "repetitions": 2,
                    "p95_latency_us": 900,
                    "max_latency_us": 1_000,
                    "token_consistent": True,
                },
                {
                    "batch_size": 32,
                    "repetitions": 2,
                    "p95_latency_us": 1_100,
                    "max_latency_us": 1_200,
                    "token_consistent": True,
                },
            ],
            "predicted_p95_us": 1_200,
            "safety_margin_us": 240,
            "measured": True,
            "eligible": True,
        })
    return with_profile_hash({
        "schema": SCHEMA,
        "physical": True,
        "model_scope": "GEMMA4_12B_F16_LOGICAL_MODEL",
        "routes": rows,
    })


def rehash(value: dict) -> dict:
    value = copy.deepcopy(value)
    value.pop("profile_hash", None)
    return with_profile_hash(value)


class ProfileTests(unittest.TestCase):
    def test_valid_bundle_returns_five_profiles(self) -> None:
        self.assertEqual(len(validate_profile_bundle(bundle())), 5)

    def test_hash_mutation_is_rejected(self) -> None:
        value = bundle()
        value["routes"][0]["cut"] = 8
        with self.assertRaises(ProfileError):
            validate_profile_bundle(value)

    def test_identity_mutation_is_rejected_after_rehash(self) -> None:
        value = bundle()
        value["routes"][0]["device"] = "op12"
        with self.assertRaises(ProfileError):
            validate_profile_bundle(rehash(value))

    def test_one_repetition_is_rejected(self) -> None:
        value = bundle()
        value["routes"][0]["batch_points"][0]["repetitions"] = 1
        with self.assertRaises(ProfileError):
            validate_profile_bundle(rehash(value))

    def test_unmeasured_eligible_route_is_rejected(self) -> None:
        value = bundle()
        value["routes"][0]["measured"] = False
        with self.assertRaises(ProfileError):
            validate_profile_bundle(rehash(value))


if __name__ == "__main__":
    unittest.main()

