#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
if str(S39) not in sys.path:
    sys.path.insert(0, str(S39))

from validate_qwen25_route import (
    ACTIVATION_BYTES,
    B1_GENERATED_TOKENS,
    GENERATED_TOKENS,
    GateError,
    MODEL_SHA256,
    PROMPT_TOKENS,
    REQUESTS,
    STEPS,
    cpu_ops,
    parse_json,
    validate_event,
    validate_relay,
    validate_route_report,
)


def event(index):
    phase = "prefill" if index == 0 else "decode"
    return {
        "batch_size": REQUESTS,
        "decode_rows": 0 if index == 0 else REQUESTS,
        "prefill_rows": REQUESTS if index == 0 else 0,
        "mixed_phase": False,
        "phases": [phase] * REQUESTS,
        "positions": [0 if index == 0 else index] * REQUESTS,
        "sequence_ids": list(range(REQUESTS)),
        "request_ids": list(range(6001, 6001 + REQUESTS)),
        "release_reason": "BATCH_KNEE",
    }


def route_report():
    return {
        "schema": "s39-direct-order-route-v1",
        "verdict": "PASS",
        "status": "DIRECT_ORDER_MECHANICS_PASS",
        "order": "sorted",
        "tokens_exact": True,
        "token_checks": REQUESTS * STEPS,
        "permutation": list(range(REQUESTS)),
        "configuration": {
            "requests": REQUESTS,
            "steps": STEPS,
            "batch_knee": REQUESTS,
            "prompt_tokens": list(PROMPT_TOKENS),
            "expected_tokens": list(GENERATED_TOKENS),
        },
        "worker": {
            "layer_start": 0,
            "layer_end": 48,
            "n_layer": 48,
            "n_embd": 5120,
            "file_type": 2,
            "model_sha256": MODEL_SHA256,
            "max_streams": REQUESTS,
        },
        "batch_events": [event(index) for index in range(STEPS)],
        "requests": [
            {
                "request_id": 6001 + index,
                "sequence_id": index,
                "tokens": list(GENERATED_TOKENS),
                "slo_met": True,
            }
            for index in range(REQUESTS)
        ],
        "transport": {
            "direct_activation_payload_bytes": ACTIVATION_BYTES,
            "host_activation_payload_bytes": 0,
            "mode": "OP15_TO_OP12_DIRECT_WIFI",
        },
    }


def relay_record():
    return {
        "schema": "ls-stage-direct-relay-v1",
        "status": "DIRECT_RELAY_OK",
        "run_rc": 0,
        "layer_start": 0,
        "cut_layer": 32,
        "layer_end": 48,
        "n_layer": 48,
        "n_embd": 5120,
        "file_type": 2,
        "model_sha256": MODEL_SHA256,
        "batches": STEPS,
        "rows": REQUESTS * STEPS,
        "activation_payload_bytes": ACTIVATION_BYTES,
        "host_activation_payload_bytes": 0,
    }


class Qwen25RouteValidatorTests(unittest.TestCase):
    def test_valid_route_report(self):
        self.assertEqual(validate_route_report(route_report()), "MATCHED_B32")

    def test_mismatched_b1_oracle_is_a_diagnostic(self):
        report = route_report()
        report["configuration"]["expected_tokens"] = list(B1_GENERATED_TOKENS)
        report["tokens_exact"] = False
        report["verdict"] = "FAIL"
        report["status"] = "DIRECT_ORDER_FAIL"
        self.assertEqual(
            validate_route_report(report),
            "MISMATCHED_B1_DIAGNOSTIC",
        )

    def test_noncanonical_sequence_is_rejected(self):
        report = route_report()
        report["batch_events"][2]["sequence_ids"][0] = 9
        with self.assertRaisesRegex(GateError, "sequence_ids"):
            validate_route_report(report)

    def test_cuda_token_mutation_is_rejected(self):
        report = route_report()
        report["requests"][17]["tokens"][-1] = 17689
        with self.assertRaisesRegex(GateError, "request\\[17\\].tokens"):
            validate_route_report(report)

    def test_host_activation_is_rejected(self):
        report = route_report()
        report["transport"]["host_activation_payload_bytes"] = ACTIVATION_BYTES
        with self.assertRaisesRegex(GateError, "host_activation"):
            validate_route_report(report)

    def test_valid_relay(self):
        validate_relay(relay_record())

    def test_wrong_cut_is_rejected(self):
        record = relay_record()
        record["cut_layer"] = 30
        with self.assertRaisesRegex(GateError, "cut_layer"):
            validate_relay(record)

    def test_cpu_ops_are_derived_by_operation(self):
        placement = {
            "compute_by_op_and_buffer": {
                "GET_ROWS": {"CPU": 32},
                "MUL_MAT": {"OpenCL": 64},
            }
        }
        self.assertEqual(cpu_ops(placement), {"GET_ROWS"})
        placement["compute_by_op_and_buffer"]["MUL_MAT"]["CPU"] = 1
        self.assertEqual(cpu_ops(placement), {"GET_ROWS", "MUL_MAT"})

    def test_duplicate_json_key_is_rejected(self):
        with self.assertRaisesRegex(GateError, "duplicate JSON key"):
            parse_json('{"a":1,"a":1}', "fixture")


if __name__ == "__main__":
    unittest.main()
