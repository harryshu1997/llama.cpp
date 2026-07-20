#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_cohort as build
import validate_cohort as validator


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")


class CohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build.main()
        cls.cohort = json.loads(build.COHORT.read_text(encoding="ascii"))
        cls.inputs = json.loads(build.INPUT_MANIFEST.read_text(encoding="ascii"))

    def validate_mutation(self, cohort=None, inputs=None, trace_manifest=None) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            cohort_path = directory / "cohort.json"
            inputs_path = directory / "inputs.json"
            trace_manifest_path = build.TRACE_MANIFEST
            write_json(cohort_path, self.cohort if cohort is None else cohort)
            write_json(inputs_path, self.inputs if inputs is None else inputs)
            if trace_manifest is not None:
                trace_manifest_path = directory / "trace.manifest.json"
                write_json(trace_manifest_path, trace_manifest)
            validator.validate(
                cohort_path,
                inputs_path,
                trace_manifest_path=trace_manifest_path,
            )

    def test_frozen_artifacts_validate(self) -> None:
        validated = validator.validate()
        self.assertEqual(len(validated["requests"]), 32)

    def test_byte_deterministic(self) -> None:
        before = (build.COHORT.read_bytes(), build.INPUT_MANIFEST.read_bytes())
        build.main()
        self.assertEqual(before, (build.COHORT.read_bytes(), build.INPUT_MANIFEST.read_bytes()))

    def test_validator_does_not_import_builder(self) -> None:
        source = (build.HERE / "validate_cohort.py").read_text(encoding="ascii")
        self.assertNotIn("import build_cohort", source)

    def test_duplicate_source_json_key_rejected(self) -> None:
        with self.assertRaisesRegex(validator.ValidationError, "duplicate JSON key"):
            validator.strict_value('{"event_id":"a","event_id":"b"}')

    def test_manifest_row_count_recomputed(self) -> None:
        manifest = json.loads(build.TRACE_MANIFEST.read_text(encoding="ascii"))
        manifest["output_row_count"] += 1
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "manifest.json"
            write_json(path, manifest)
            cohort = copy.deepcopy(self.cohort)
            cohort["source"]["manifest_sha256"] = "sha256:" + build.sha256_file(path)
            cohort["cohort_hash"] = build.content_address(cohort, "cohort_hash")
            with self.assertRaisesRegex(validator.ValidationError, "row count"):
                self.validate_mutation(cohort=cohort, trace_manifest=manifest)

    def test_duplicate_request_rejected(self) -> None:
        cohort = copy.deepcopy(self.cohort)
        cohort["requests"][1] = cohort["requests"][0]
        cohort["cohort_hash"] = build.content_address(cohort, "cohort_hash")
        with self.assertRaisesRegex(validator.ValidationError, "payload binding"):
            self.validate_mutation(cohort=cohort)

    def test_changed_source_fields_rejected(self) -> None:
        cohort = copy.deepcopy(self.cohort)
        cohort["requests"][0]["observed_input_tokens"] += 1
        cohort["cohort_hash"] = build.content_address(cohort, "cohort_hash")
        with self.assertRaisesRegex(validator.ValidationError, "source replay"):
            self.validate_mutation(cohort=cohort)

    def test_nonidentical_payload_rejected(self) -> None:
        inputs = copy.deepcopy(self.inputs)
        inputs["request_payloads"][3]["payload_sha256"] = "sha256:" + "0" * 64
        inputs["input_manifest_hash"] = build.content_address(inputs, "input_manifest_hash")
        cohort = copy.deepcopy(self.cohort)
        cohort["execution_target"]["input_manifest_hash"] = inputs["input_manifest_hash"]
        cohort["cohort_hash"] = build.content_address(cohort, "cohort_hash")
        with self.assertRaisesRegex(validator.ValidationError, "payload binding"):
            self.validate_mutation(cohort=cohort, inputs=inputs)

    def test_observed_priority_claim_rejected(self) -> None:
        cohort = copy.deepcopy(self.cohort)
        cohort["synthetic_sidecar"]["provenance"] = "real"
        cohort["cohort_hash"] = build.content_address(cohort, "cohort_hash")
        with self.assertRaisesRegex(validator.ValidationError, "sidecar"):
            self.validate_mutation(cohort=cohort)

    def test_b31_rejected(self) -> None:
        cohort = copy.deepcopy(self.cohort)
        cohort["requests"].pop()
        cohort["cohort_hash"] = build.content_address(cohort, "cohort_hash")
        with self.assertRaisesRegex(validator.ValidationError, "count"):
            self.validate_mutation(cohort=cohort)

    def test_forged_profile_rejected(self) -> None:
        cohort = copy.deepcopy(self.cohort)
        cohort["execution_target"]["conservative_duration_us"] -= 1_000_000
        cohort["cohort_hash"] = build.content_address(cohort, "cohort_hash")
        with self.assertRaisesRegex(validator.ValidationError, "duration binding"):
            self.validate_mutation(cohort=cohort)

    def test_forged_schedule_rejected(self) -> None:
        cohort = copy.deepcopy(self.cohort)
        cohort["admission_schedule"]["predicted_slack_us"] += 1
        cohort["cohort_hash"] = build.content_address(cohort, "cohort_hash")
        with self.assertRaisesRegex(validator.ValidationError, "SLO schedule"):
            self.validate_mutation(cohort=cohort)


if __name__ == "__main__":
    unittest.main()
