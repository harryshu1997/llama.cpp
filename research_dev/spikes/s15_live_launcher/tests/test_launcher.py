#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
LIVE = HERE.parent
S15 = LIVE.parent / "s15_runtime_dispatch"
sys.path[:0] = [str(LIVE), str(S15)]

import build_manifest as builder  # noqa: E402
from input_manifest import InputManifestError, canonical, load_manifest, validate_launch_manifest  # noqa: E402
import physical_launcher as launcher  # noqa: E402


MANIFEST_PAYLOAD = builder.build()
MANIFEST = load_manifest(MANIFEST_PAYLOAD)
COHORT = json.loads(launcher.COHORT_PATH.read_text(encoding="ascii"))
COHORT_IDS = tuple(item["event_id"] for item in COHORT["requests"])


def request() -> dict:
    return {
        "schema": "s15-physical-request-v1",
        "command": "EXECUTE",
        "protocol_version": 1,
        "launch_id": 1,
        "route_id": "op15-gemma-head-0-8",
        "profile_id": "sha256:947f1fd95b3f1a7c881b71d793bf3177fc9612a46d959fcef987da0026531cae",
        "device_id": "op15:3C15AU002CL00000",
        "route_epoch": 12,
        "residency_epoch": 1,
        "lease_epoch": 1,
        "device_boot_epoch": 1,
        "registry_generation": 1,
        "compatibility_key": "gemma-4-12b-it-f16|decode|gemma-head-0-8",
        "request_ids": list(COHORT_IDS),
        "cohort_sha256": launcher.EXPECTED_COHORT,
        "input_manifest_sha256": launcher.EXPECTED_INPUT,
        "timeout_us": 180_000_000,
        "expected_boundary_schema": "s15-boundary-certificate-v1",
        "worker_binary_sha256": "sha256:2494f191515ca576cacd8c411de79fa8985f719cba2a11cd6371aa6e1024be9f",
        "worker_generation": 1,
        "device_boot_id": "test-boot",
        "layer_range": [0, 8],
    }


def fixed_request() -> dict:
    value = request()
    value["request_ids"] = list(MANIFEST.request_ids)
    value["input_manifest_sha256"] = MANIFEST.sha256
    return value


def route_rows(tokens=None) -> list[dict]:
    tokens = tokens or [1, 2, 3, 4, 5, 6, 7, 8]
    return [
        {
            "stream_index": index,
            "status": "ok",
            "batch_size": 32,
            "generated_tokens": 8,
            "token_ids": list(tokens),
        }
        for index in range(32)
    ]


def cert() -> dict:
    return {
        "status": "SCHEDULED_PLACEMENT_OK",
        "layer_start": 0,
        "layer_end": 8,
        "missing_buffer_compute_nodes": 0,
        "compute_by_op_and_buffer": {
            "MUL_MAT": {"HTP0": 100},
            "GET_ROWS": {"CPU": 8},
        },
    }


def stderr_rows(rows: list[dict]) -> bytes:
    return b"".join(b"ROUTEJSON " + canonical(row) for row in rows) + \
        b"DRIVER_DONE " + canonical({"status": "ok", "requests": 32})


def phone_cert(value: dict) -> bytes:
    return b"PLACEMENTCERT " + canonical(value)


class ManifestTests(unittest.TestCase):
    def test_authoritative_cohort_and_input_files_validate(self) -> None:
        cohort, inputs, request_ids, prompt = launcher.load_workload()
        self.assertEqual(request_ids, COHORT_IDS)
        self.assertEqual(prompt, "Explain batching.")
        self.assertEqual(cohort["execution_target"]["batch"], 32)
        self.assertEqual(inputs["input_manifest_hash"],
                         "sha256:c28651deee12f50b95085cc8699cae2ec2ea71e04298b6adfb0a6fe50098bbc3")

    def test_manifest_binds_all_32_exact_inputs(self) -> None:
        self.assertEqual(len(MANIFEST.request_ids), 32)
        self.assertEqual(MANIFEST.repeated_input, b"Explain batching.")
        self.assertEqual(validate_launch_manifest(MANIFEST, fixed_request()), b"Explain batching.")

    def test_manifest_mutation_and_unknown_field_are_rejected(self) -> None:
        value = json.loads(MANIFEST_PAYLOAD)
        value["requests"][0]["input_sha256"] = "sha256:" + "00" * 32
        with self.assertRaisesRegex(InputManifestError, "digest mismatch"):
            load_manifest(canonical(value))
        value = json.loads(MANIFEST_PAYLOAD)
        value["unknown"] = 1
        with self.assertRaisesRegex(InputManifestError, "unknown fields"):
            load_manifest(canonical(value))

    def test_missing_request_and_nonidentical_input_are_rejected(self) -> None:
        bad = fixed_request()
        bad["request_ids"] = bad["request_ids"][:-1]
        with self.assertRaisesRegex(InputManifestError, "request IDs"):
            validate_launch_manifest(MANIFEST, bad)
        value = json.loads(MANIFEST_PAYLOAD)
        value["requests"][0]["input_b64"] = "ZGlmZmVyZW50"
        value["requests"][0]["input_sha256"] = \
            "sha256:9d6f965ac832e40a5df6c06afe983e3b449c07b843ff51ce76204de05c690d11"
        with self.assertRaisesRegex(InputManifestError, "identical input"):
            load_manifest(canonical(value))

    def test_duplicate_keys_are_rejected(self) -> None:
        payload = MANIFEST_PAYLOAD.replace(b'{"repeat_constraint"',
                                           b'{"schema":"duplicate","repeat_constraint"', 1)
        with self.assertRaisesRegex(InputManifestError, "duplicate JSON key"):
            load_manifest(payload)


class RequestTests(unittest.TestCase):
    def test_wrong_route_profile_and_device_are_rejected(self) -> None:
        for key in ("route_id", "profile_id", "device_id"):
            bad = request()
            bad[key] = "wrong"
            with self.subTest(key=key), self.assertRaisesRegex(
                    launcher.LauncherError, f"identity mismatch: {key}"):
                launcher.validate_request(bad)

    def test_stale_epochs_are_rejected(self) -> None:
        for key in ("route_epoch", "residency_epoch", "lease_epoch",
                    "device_boot_epoch", "registry_generation"):
            bad = request()
            bad[key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(
                    launcher.LauncherError, "stale or unsupported epoch"):
                launcher.validate_request(bad)

    def test_missing_and_duplicate_request_keys_are_rejected(self) -> None:
        bad = request()
        del bad["route_id"]
        with self.assertRaisesRegex(launcher.LauncherError, "missing or unknown"):
            launcher.validate_request(bad)
        payload = canonical(request()).replace(
            b'{"cohort_sha256"', b'{"cohort_sha256":"dup","cohort_sha256"', 1)
        with self.assertRaisesRegex(launcher.LauncherError, "duplicate JSON key"):
            launcher.strict_object(payload, "request")

    def test_wrong_cohort_or_input_artifact_is_rejected(self) -> None:
        for key in ("cohort_sha256", "input_manifest_sha256"):
            bad = request()
            bad[key] = "sha256:" + "00" * 32
            with self.subTest(key=key), self.assertRaisesRegex(
                    launcher.LauncherError, f"identity mismatch: {key}"):
                launcher.validate_request(bad, COHORT_IDS)

    def test_request_ids_must_exactly_match_cohort(self) -> None:
        bad = request()
        bad["request_ids"] = bad["request_ids"][:-1]
        with self.assertRaisesRegex(launcher.LauncherError, "frozen cohort"):
            launcher.validate_request(bad, COHORT_IDS)


class EvidenceTests(unittest.TestCase):
    def completed(self, rows=None, placement=None):
        rows = rows if rows is not None else route_rows()
        placement = placement if placement is not None else cert()
        return launcher.validate_completion(
            stderr_rows(rows), phone_cert(placement), [1, 2, 3, 4, 5, 6, 7, 8]
        )

    def test_valid_raw_evidence_produces_32_boundaries(self) -> None:
        placement, rows = self.completed()
        result = launcher.completed_record(request(), placement)
        self.assertEqual(len(rows), 32)
        self.assertEqual(len(result["boundaries"]), 32)
        self.assertTrue(all(value["d2h_complete"] for value in result["boundaries"]))

    def test_token_mismatch_is_rejected(self) -> None:
        rows = route_rows()
        rows[7]["token_ids"][-1] = 99
        with self.assertRaisesRegex(launcher.LauncherError, "correctness"):
            self.completed(rows=rows)

    def test_partial_stream_set_is_rejected(self) -> None:
        with self.assertRaisesRegex(launcher.LauncherError, "complete"):
            self.completed(rows=route_rows()[:-1])

    def test_cpu_fallback_is_rejected(self) -> None:
        placement = cert()
        placement["compute_by_op_and_buffer"]["MUL_MAT"] = {"CPU": 1}
        with self.assertRaisesRegex(launcher.LauncherError, "fallback"):
            self.completed(placement=placement)

    def test_error_record_has_no_completion_evidence(self) -> None:
        result = launcher.error_record(request())
        self.assertEqual((result["outcome"], result["placement"], result["boundaries"]),
                         ("error", None, []))


class ReadyProcessTests(unittest.TestCase):
    def test_phone_style_readiness_on_stdout_is_observed(self) -> None:
        code = "import sys,time; print('[stagenet] listening', flush=True); time.sleep(.05)"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = launcher.ReadyProcess(
                [sys.executable, "-c", code], None, root / "out", root / "err",
                b"[stagenet] listening", ("stdout", "stderr"),
            )
            process.wait_ready(1)
            self.assertEqual(process.wait(1), 0)

    def test_thermal_stream_requires_the_exact_discovered_sensor_set(self) -> None:
        monitor = object.__new__(launcher.ThermalMonitor)
        monitor._expected_names = frozenset(("nsphmx-0", "nsphmx-1"))
        self.assertIsNone(monitor._parse(b"THERMAL nsphmx-0=30000\n"))
        self.assertIsNone(monitor._parse(b"THERMAL nsphmx-0=30000 renamed=31000\n"))
        value = monitor._parse(b"THERMAL nsphmx-0=30000 nsphmx-1=31000\n")
        self.assertEqual(value["max_millic"], 31000)

    def test_preflight_helpers_cannot_inherit_launcher_stdin(self) -> None:
        source = (LIVE / "physical_launcher.py").read_text(encoding="ascii")
        monitor_block = source[source.index("class ThermalMonitor"):source.index("def run_reference")]
        run_block = source[source.index("def run(args:"):source.index("def adb(")]
        self.assertIn("stdin=subprocess.DEVNULL", monitor_block)
        self.assertIn("stdin=subprocess.DEVNULL", run_block)


if __name__ == "__main__":
    unittest.main()
