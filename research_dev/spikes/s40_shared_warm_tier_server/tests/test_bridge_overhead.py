#!/usr/bin/env python3

import copy
import base64
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
S40 = HERE.parent
sys.path.insert(0, str(S40))

from bridge_overhead import (  # noqa: E402
    summarize,
    validate_combined_measurement,
    validate_combined_transport_gate,
    validate_measurement,
    validate_native_measurement,
    validate_native_transport_gate,
)
from evidence_common import EvidenceError  # noqa: E402


def fixture():
    rows = []
    for mode, base in (
            ("DIRECT_SOCKET", 100),
            ("FRESH_PYTHON_BRIDGE", 1_100)):
        for fanout in (1, 8):
            for sample in range(2):
                rows.append({
                    "batch_index": sample,
                    "batch_makespan_ns": base + 10,
                    "command_bytes": 100,
                    "fanout": fanout,
                    "item_index": 0,
                    "latency_ns": base + sample,
                    "mode": mode,
                    "result_bytes": 100,
                })
    return {
        "executor_bundle_manifest_sha256": "a" * 64,
        "host_boot_id": "boot",
        "python_executable_sha256": "b" * 64,
        "rows": rows,
        "schema": "s40-bridge-overhead-v1",
        "summary": summarize(rows),
    }


def native_fixture(fanout, latency=120):
    rows = []
    command_id = 1
    for batch in range(2):
        for item in range(fanout):
            started = 1_000 + command_id * 1_000
            rows.append({
                "batch_index": batch,
                "batch_makespan_ns": latency + 20,
                "command_bytes": 100,
                "command_id": command_id,
                "completed_ns": started + latency,
                "item_index": item,
                "latency_ns": latency,
                "result_bytes": 100,
                "started_ns": started,
            })
            command_id += 1
    return {
        "fanout": fanout,
        "rows": rows,
        "sample_count": len(rows),
        "schema": "s40-native-unix-executor-bench-v1",
        "socket_path": "/tmp/noop.sock",
        "transport": "UNIX_SOCKET",
    }


def combined_fixture(native_latency=120):
    legacy = fixture()
    binary = "/tmp/test-warm-tier-executors"
    invocations = []
    for fanout in (1, 8):
        native = native_fixture(fanout, native_latency)
        stdout = __import__("json").dumps(
            native,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii") + b"\n"
        invocations.append({
            "argv": [
                binary,
                "--unix-bench",
                native["socket_path"],
                str(native["sample_count"]),
                str(fanout),
                "1234",
                "5678",
            ],
            "completed_ns": 200,
            "exit_code": 0,
            "fanout": fanout,
            "peer_pid": 1234,
            "peer_start_time_ticks": 5678,
            "started_ns": 100,
            "stderr_base64": "",
            "stdout_base64": base64.b64encode(stdout).decode("ascii"),
        })
    return {
        "executor_bundle_manifest_sha256":
            legacy["executor_bundle_manifest_sha256"],
        "host_boot_id": legacy["host_boot_id"],
        "native_bench_binary": {
            "bytes": 123,
            "path": binary,
            "sha256": "c" * 64,
        },
        "native_invocations": invocations,
        "python_executable_sha256": legacy["python_executable_sha256"],
        "rows": legacy["rows"],
        "schema": "s40-transport-overhead-v2",
        "summary": legacy["summary"],
    }


class BridgeOverheadTests(unittest.TestCase):
    def test_valid_fixture_recomputes_incremental_p95(self):
        result = validate_measurement(fixture(), minimum_samples=2)
        self.assertEqual(
            result["incremental_p95_ns"],
            {"1": 1_000, "8": 1_000},
        )

    def test_summary_mutation_is_rejected(self):
        value = fixture()
        value["summary"][0]["latency_p95_ns"] += 1
        with self.assertRaisesRegex(EvidenceError, "summary mismatch"):
            validate_measurement(value, minimum_samples=2)

    def test_duplicate_sample_is_rejected(self):
        value = fixture()
        value["rows"].append(copy.deepcopy(value["rows"][0]))
        value["summary"] = summarize(value["rows"])
        with self.assertRaisesRegex(EvidenceError, "duplicate sample"):
            validate_measurement(value, minimum_samples=2)

    def test_minimum_sample_count_is_enforced(self):
        with self.assertRaisesRegex(EvidenceError, "insufficient samples"):
            validate_measurement(fixture(), minimum_samples=3)

    def test_native_rows_are_reduced_and_gated(self):
        row = validate_native_measurement(
            native_fixture(8), minimum_samples=16)
        self.assertEqual(row["latency_p95_ns"], 120)
        result = validate_native_transport_gate(
            [native_fixture(1), native_fixture(8)],
            fixture(),
            fastest_desktop_execute_ns=100_000,
            minimum_samples=2,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["threshold_ns"], 5_000)

    def test_native_latency_is_derived_from_timestamps(self):
        value = native_fixture(1)
        value["rows"][0]["latency_ns"] += 1
        with self.assertRaisesRegex(EvidenceError, "latency mismatch"):
            validate_native_measurement(value, minimum_samples=2)

    def test_native_requires_full_batches_and_unique_commands(self):
        value = native_fixture(8)
        value["rows"].pop()
        value["sample_count"] -= 1
        with self.assertRaisesRegex(EvidenceError, "row count mismatch"):
            validate_native_measurement(value, minimum_samples=2)
        value = native_fixture(1)
        value["rows"][1]["command_id"] = value["rows"][0]["command_id"]
        with self.assertRaisesRegex(EvidenceError, "duplicate command"):
            validate_native_measurement(value, minimum_samples=2)

    def test_native_gate_fails_material_overhead(self):
        result = validate_native_transport_gate(
            [native_fixture(1, 20_000), native_fixture(8, 20_000)],
            fixture(),
            fastest_desktop_execute_ns=100_000,
            minimum_samples=2,
        )
        self.assertFalse(result["performance_claim_authorized"])
        self.assertEqual(
            result["status"], "FAIL_MATERIAL_TRANSPORT_OVERHEAD")

    def test_combined_artifact_recomputes_native_and_direct(self):
        value = combined_fixture()
        result = validate_combined_measurement(value, minimum_samples=2)
        self.assertEqual(result["status"], "MEASURED")
        gate = validate_combined_transport_gate(
            value,
            fastest_desktop_execute_ns=100_000,
            minimum_samples=2,
        )
        self.assertEqual(gate["status"], "PASS")

    def test_combined_artifact_rejects_binary_or_argv_mutation(self):
        value = combined_fixture()
        value["native_bench_binary"]["sha256"] = "not-a-digest"
        with self.assertRaisesRegex(EvidenceError, "SHA-256"):
            validate_combined_measurement(value, minimum_samples=2)
        value = combined_fixture()
        value["native_invocations"][0]["argv"][-1] = "8"
        with self.assertRaisesRegex(EvidenceError, "argv mismatch"):
            validate_combined_measurement(value, minimum_samples=2)

    def test_combined_artifact_rejects_noncanonical_or_foreign_socket(self):
        value = combined_fixture()
        raw = base64.b64decode(
            value["native_invocations"][0]["stdout_base64"])
        value["native_invocations"][0]["stdout_base64"] = (
            base64.b64encode(b" " + raw).decode("ascii")
        )
        with self.assertRaisesRegex(EvidenceError, "not canonical"):
            validate_combined_measurement(value, minimum_samples=2)
        value = combined_fixture()
        second = __import__("json").loads(base64.b64decode(
            value["native_invocations"][1]["stdout_base64"]))
        second["socket_path"] = "/tmp/other.sock"
        value["native_invocations"][1]["stdout_base64"] = (
            base64.b64encode(
                __import__("json").dumps(
                    second,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii") + b"\n"
            ).decode("ascii")
        )
        value["native_invocations"][1]["argv"][2] = "/tmp/other.sock"
        with self.assertRaisesRegex(EvidenceError, "gateway socket mismatch"):
            validate_combined_measurement(value, minimum_samples=2)


if __name__ == "__main__":
    unittest.main()
