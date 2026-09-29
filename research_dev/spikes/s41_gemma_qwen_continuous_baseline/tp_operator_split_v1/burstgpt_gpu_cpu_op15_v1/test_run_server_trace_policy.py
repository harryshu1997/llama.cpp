#!/usr/bin/env python3

import unittest

import run_server_trace
import run_trace


class SplitPolicyTest(unittest.TestCase):
    def test_i1_policy_is_bound(self):
        run_server_trace.validate_split_policy(
            "i1-balanced",
            11136,
            run_server_trace.BALANCED_SPLIT_POLICY,
        )

    def test_i2_policy_is_bound(self):
        run_server_trace.validate_split_policy(
            "i2-decode-rebalance",
            11136,
            run_server_trace.DECODE_REBALANCE_SPLIT_POLICY,
        )

    def test_i2_r1_policy_is_bound(self):
        run_server_trace.validate_split_policy(
            "i2-r1-decode-rebalance",
            11136,
            run_server_trace.DECODE_REBALANCE_R1_SPLIT_POLICY,
        )

    def test_i2_r2_policy_is_bound(self):
        run_server_trace.validate_split_policy(
            "i2-r2-decode-rebalance",
            11136,
            run_server_trace.DECODE_REBALANCE_R2_SPLIT_POLICY,
        )

    def test_i3_policy_is_bound(self):
        run_server_trace.validate_split_policy(
            "i3-hidden-wait",
            11136,
            run_server_trace.HIDDEN_WAIT_SPLIT_POLICY,
        )

    def test_table_substitution_fails(self):
        with self.assertRaises(run_trace.RunError):
            run_server_trace.validate_split_policy(
                "i2-decode-rebalance",
                11136,
                run_server_trace.BALANCED_SPLIT_POLICY,
            )

    def test_width_substitution_fails(self):
        with self.assertRaises(run_trace.RunError):
            run_server_trace.validate_split_policy(
                "i2-decode-rebalance",
                11776,
                run_server_trace.DECODE_REBALANCE_SPLIT_POLICY,
            )

    def test_unknown_identity_fails(self):
        with self.assertRaises(run_trace.RunError):
            run_server_trace.validate_split_policy(
                "candidate",
                11136,
                run_server_trace.DECODE_REBALANCE_SPLIT_POLICY,
            )


if __name__ == "__main__":
    unittest.main()
