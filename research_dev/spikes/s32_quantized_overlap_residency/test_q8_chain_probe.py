#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest


PATH = pathlib.Path(__file__).with_name("q8_chain_probe.py")
SPEC = importlib.util.spec_from_file_location("s32_q8_chain_probe", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def hello(start, end, terminal=False, streams=32):
    capabilities = 0x0F | (0x10 if terminal else 0)
    return MOD.Hello(start, end, 48, 3840, streams, 16, 256, 256, capabilities)


class ChainProbeTests(unittest.TestCase):
    def test_topology(self):
        MOD.validate_topology(
            hello(0, 4), hello(4, 16), hello(16, 48, True), hello(0, 16), 32,
        )

    def test_rejects_gap(self):
        with self.assertRaises(MOD.ProtocolError):
            MOD.validate_topology(
                hello(0, 4), hello(5, 16), hello(16, 48, True), hello(0, 16), 32,
            )

    def test_rejects_small_worker(self):
        with self.assertRaises(MOD.ProtocolError):
            MOD.validate_topology(
                hello(0, 4, streams=31), hello(4, 16),
                hello(16, 48, True), hello(0, 16), 32,
            )

    def test_token_rows(self):
        rows = MOD.rows_for_tokens([2, 3], 4, 100)
        self.assertEqual((rows[1].request_id, rows[1].seq_id, rows[1].position, rows[1].token), (101, 1, 4, 3))

    def test_initial_tokens_are_distinct(self):
        self.assertEqual(MOD.initial_tokens(4), [2, 3, 4, 5])
        with self.assertRaises(ValueError):
            MOD.initial_tokens(0)

    def test_hidden_rows_preserve_lineage(self):
        result = MOD.BatchResult(9, 10, 1, 2, (1.0,) * 3840, None)
        rows = MOD.rows_for_hidden([result], [7], 2)
        self.assertEqual((rows[0].request_id, rows[0].route_epoch, rows[0].hidden), (9, 10, result.hidden))

    def test_rejects_terminal_as_hidden(self):
        result = MOD.BatchResult(9, 10, 1, 2, None, 7)
        with self.assertRaises(MOD.ProtocolError):
            MOD.require_hidden([result], 1, "test")


if __name__ == "__main__":
    unittest.main()
