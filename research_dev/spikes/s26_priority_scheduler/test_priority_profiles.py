#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from priority_profiles import (
    PriorityProfileError,
    canonical_bytes,
    derive_bundle,
    load_bundle,
)


class PriorityProfileTests(unittest.TestCase):
    def test_physical_derivation(self) -> None:
        bundle = derive_bundle()
        by_route = {row["route_id"]: row for row in bundle["routes"]}
        r0_b1, r0_b4 = by_route["R0"]["points"]
        r2_b1, r2_b4 = by_route["R2"]["points"]
        self.assertEqual((r0_b1["duration_us"], r0_b1["cuda_work_us"]), (349395, 212995))
        self.assertEqual((r0_b4["duration_us"], r0_b4["cuda_work_us"]), (319699, 313949))
        self.assertEqual((r2_b1["duration_us"], r2_b1["cuda_work_us"]), (1418965, 155690))
        self.assertEqual((r2_b4["duration_us"], r2_b4["cuda_work_us"]), (2203547, 208875))
        self.assertEqual(r0_b4["cuda_work_us"] - r2_b4["cuda_work_us"], 105074)

    def test_canonical_bundle_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "profiles.json"
            path.write_bytes(canonical_bytes(derive_bundle()))
            routes, capacities, reserve, _bundle = load_bundle(path)
        self.assertEqual([route.route_id for route in routes], ["R0", "R2"])
        self.assertEqual(routes[0].target().batch_size, 4)
        self.assertEqual(routes[1].target().batch_size, 4)
        self.assertEqual(capacities["cuda-tail"], 8)
        self.assertEqual(reserve["cuda-tail"], 4)

    def test_profile_value_mutation_is_rejected(self) -> None:
        value = derive_bundle()
        value["routes"][1]["points"][1]["duration_us"] -= 1
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "profiles.json"
            path.write_bytes(canonical_bytes(value))
            with self.assertRaisesRegex(PriorityProfileError, "canonical derived"):
                load_bundle(path)

    def test_noncanonical_bytes_are_rejected(self) -> None:
        value = derive_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "profiles.json"
            path.write_text(json.dumps(value, indent=2), encoding="ascii")
            with self.assertRaisesRegex(PriorityProfileError, "canonical derived"):
                load_bundle(path)

    def test_duplicate_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "profiles.json"
            path.write_text('{"schema":"a","schema":"b"}\n', encoding="ascii")
            with self.assertRaisesRegex(PriorityProfileError, "duplicate JSON key"):
                load_bundle(path)


if __name__ == "__main__":
    unittest.main()
