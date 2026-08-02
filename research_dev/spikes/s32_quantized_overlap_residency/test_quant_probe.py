#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest


PATH = pathlib.Path(__file__).with_name("quant_probe.py")
SPEC = importlib.util.spec_from_file_location("s32_quant_probe", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class QuantProbeTests(unittest.TestCase):
    def test_nearest_rank(self):
        self.assertEqual(MOD.nearest_rank([3, 1, 2, 4], 50, 100), 2)
        self.assertEqual(MOD.nearest_rank([3, 1, 2, 4], 95, 100), 4)

    def test_nearest_rank_rejects_empty(self):
        with self.assertRaises(ValueError):
            MOD.nearest_rank([], 95, 100)

    def test_summarize_rejects_bool_and_nonpositive(self):
        for values in ([1, True], [1, 0], [1, -1]):
            with self.assertRaises(ValueError):
                MOD.summarize(values)

    def test_parse_memory(self):
        row = MOD.parse_kib_fields("VmRSS:\t123 kB\nVmSwap:\t0 kB\n")
        self.assertEqual(row, {"VmRSS": 123, "VmSwap": 0})

    def test_parse_memory_rejects_bad_unit(self):
        with self.assertRaises(ValueError):
            MOD.parse_kib_fields("VmRSS: 123 MB\n")

    def test_resource_failures(self):
        self.assertEqual(MOD.resource_failures({"process_kib": {"VmSwap": 0}}), [])
        self.assertEqual(
            MOD.resource_failures({"process_kib": {"VmSwap": 1}}),
            ["PHONE_SWAP_NONZERO"],
        )

    def test_resource_failures_rejects_float(self):
        with self.assertRaises(ValueError):
            MOD.resource_failures({"process_kib": {"VmSwap": 0.0}})

    def test_compare_sequences(self):
        result = MOD.compare_sequences(((1.0, 2.0),), ((1.0, 2.0),))
        self.assertTrue(result[0]["byte_equal"])
        self.assertEqual(result[0]["rel_l2"], 0.0)

    def test_numeric_gate(self):
        self.assertTrue(MOD.numeric_pass(
            [{"rel_l2": 0.005, "cosine": 0.999}], 0.005, 0.999))
        self.assertFalse(MOD.numeric_pass(
            [{"rel_l2": 0.0051, "cosine": 0.9999}], 0.005, 0.999))
        self.assertFalse(MOD.numeric_pass([], 0.005, 0.999))

    def test_validate_phone_capacity(self):
        hello = MOD.Hello(0, 2, 48, 3840, 31, 256, 256, 256, 15)
        with self.assertRaises(MOD.ProtocolError):
            MOD.validate_hello(hello, 0, 2, 32, False)

    def test_validate_range(self):
        hello = MOD.Hello(0, 3, 48, 3840, 32, 256, 256, 256, 15)
        with self.assertRaises(MOD.ProtocolError):
            MOD.validate_hello(hello, 0, 2, 32, False)


if __name__ == "__main__":
    unittest.main()
