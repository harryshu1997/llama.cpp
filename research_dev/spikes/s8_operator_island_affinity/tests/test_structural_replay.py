#!/usr/bin/python3
"""Focused fail-closed tests for the S8 structural replay."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import structural_replay as replayer  # noqa: E402


CONFIGS = ROOT / "configs"
DAGS = ROOT / "service_dags"
NORMALIZER = ROOT / "normalize_trace.py"


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def burst_source():
    rows = [
        ["0.0", "s0", "1", "GPT-4", "10", "2", "12", "Conversation log"],
        ["1.0", "", "2", "ChatGPT", "3", "1", "4", "API log"],
        ["2.0", "s2", "3", "GPT-4", "5", "4", "9", "Conversation log"],
    ]
    header = [
        "Timestamp",
        "Session ID",
        "Elapsed time",
        "Model",
        "Request tokens",
        "Response tokens",
        "Total tokens",
        "Log Type",
    ]
    lines = [header, *rows]
    return b"\n".join(",".join(row).encode("ascii") for row in lines) + b"\n"


def request(event_id, t_us, provenance="semi_synthetic", source="mix"):
    return {
        "schema_version": 1,
        "event_id": event_id,
        "source": source,
        "provenance": provenance,
        "t_us": t_us,
        "service": "api_generation",
        "model_class": "large_text_generation",
        "session_id": None,
        "input_tokens": 1,
        "output_tokens": 1,
        "images": 0,
        "audio_ms": 0,
        "retrieved_chunks": 0,
        "cache_keys": [],
        "observed_latency_us": None,
        "priority_class": None,
        "deadline_us": None,
        "priority_provenance": "none",
        "deadline_provenance": "none",
        "source_fields": {},
    }


class StructuralReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._base_tmp = tempfile.TemporaryDirectory()
        cls.base = pathlib.Path(cls._base_tmp.name)
        cls.source = cls.base / "BurstGPT_3.csv"
        cls.config = cls.base / "burstgpt.config.json"
        cls.run_dir = cls.base / "normalized"

        source_data = burst_source()
        cls.source.write_bytes(source_data)
        config = json.loads(
            (CONFIGS / "burstgpt.config.json").read_text(encoding="ascii")
        )
        config["origin"].update(
            {
                "bytes": len(source_data),
                "sha256": digest(source_data),
                "line_count": 4,
                "record_count": 3,
            }
        )
        cls.config.write_bytes(canonical(config) + b"\n")
        process = subprocess.run(
            [
                sys.executable,
                str(NORMALIZER),
                "--source-file",
                str(cls.source),
                "--config",
                str(cls.config),
                "--scenario",
                "median",
                "--output-dir",
                str(cls.run_dir),
            ],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if process.returncode != 0:
            raise AssertionError(process.stderr)

    @classmethod
    def tearDownClass(cls):
        cls._base_tmp.cleanup()

    @contextmanager
    def case(self, copy_dags=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            run_dir = root / "normalized"
            source = root / self.source.name
            config = root / self.config.name
            shutil.copytree(self.run_dir, run_dir)
            shutil.copy2(self.source, source)
            shutil.copy2(self.config, config)
            dag_dir = DAGS
            if copy_dags:
                dag_dir = root / "service_dags"
                shutil.copytree(DAGS, dag_dir)
            yield {
                "root": root,
                "run_dir": run_dir,
                "source": source,
                "config": config,
                "dag_dir": dag_dir,
                "trace": "burstgpt-v2.median.jsonl",
            }

    def replay(self, case):
        return replayer.replay(
            case["run_dir"],
            case["trace"],
            case["config"],
            case["dag_dir"],
            case["source"],
        )

    def assert_replay_error(self, case, code):
        with self.assertRaises(replayer.ReplayError) as context:
            self.replay(case)
        self.assertEqual(context.exception.code, code)

    def reseal_trace(self, case, trace_data):
        run_dir = case["run_dir"]
        trace_path = run_dir / case["trace"]
        sidecar_path = run_dir / case["trace"].replace(
            ".jsonl", ".manifest.json"
        )
        artifact_path = run_dir / "normalize.artifact.json"
        trace_path.write_bytes(trace_data)
        sidecar = json.loads(sidecar_path.read_text(encoding="ascii"))
        sidecar["output_sha256"] = digest(trace_data)
        artifact = json.loads(artifact_path.read_text(encoding="ascii"))
        matches = [
            output
            for output in artifact["outputs"]
            if output["path"] == case["trace"]
        ]
        self.assertEqual(len(matches), 1)
        matches[0]["sha256"] = digest(trace_data)
        matches[0]["sidecar_manifest_sha256"] = digest(canonical(sidecar))
        replay_preimage = "\n".join(
            output["sha256"]
            for output in sorted(artifact["outputs"], key=lambda item: item["path"])
        ).encode("ascii")
        artifact["deterministic_replay_sha256"] = digest(replay_preimage)
        sidecar_path.write_bytes(canonical(sidecar) + b"\n")
        artifact_path.write_bytes(canonical(artifact) + b"\n")

    def rewrite_records(self, case, mutate):
        trace_path = case["run_dir"] / case["trace"]
        records = [
            json.loads(line)
            for line in trace_path.read_text(encoding="ascii").splitlines()
        ]
        mutate(records)
        trace_data = b"".join(canonical(record) + b"\n" for record in records)
        self.reseal_trace(case, trace_data)

    def test_direct_normalizer_output_and_demand(self):
        with self.case() as case:
            result = self.replay(case)
        self.assertEqual(result["kind"], "structural_replay")
        self.assertEqual(result["record_count"], 3)
        self.assertEqual(
            result["total_demand"],
            {
                "input_tokens": 18,
                "output_tokens": 7,
                "images": 0,
                "audio_ms": 0,
                "retrieved_chunks": 0,
            },
        )
        by_service = {row["service"]: row for row in result["services"]}
        self.assertEqual(by_service["api_generation"]["request_count"], 1)
        self.assertEqual(
            by_service["conversation_generation"]["demand"]["input_tokens"],
            15,
        )

    def test_replay_is_byte_deterministic_and_structural_only(self):
        with self.case() as case:
            first = replayer.canonical_json(self.replay(case))
            second = replayer.canonical_json(self.replay(case))
        self.assertEqual(first, second)
        result = json.loads(first)
        forbidden = ("energy", "latency", "power", "throughput", "duration")

        def visit(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    self.assertFalse(any(word in key.lower() for word in forbidden))
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(result)

    def test_renormalization_byte_compare_passes(self):
        with self.case() as case:
            replayer.verify_renormalization(
                case["run_dir"],
                case["source"],
                case["config"],
            )

    def test_resealed_trace_fails_renormalization(self):
        with self.case() as case:
            self.rewrite_records(
                case,
                lambda records: records[0].__setitem__(
                    "input_tokens", records[0]["input_tokens"] + 1
                ),
            )
            with self.assertRaises(replayer.ReplayError) as context:
                replayer.verify_renormalization(
                    case["run_dir"],
                    case["source"],
                    case["config"],
                )
            self.assertEqual(context.exception.code, "E_RENORMALIZATION")

    def test_trace_hash_drift_is_rejected(self):
        with self.case() as case:
            trace_path = case["run_dir"] / case["trace"]
            trace_path.write_bytes(trace_path.read_bytes() + b" ")
            self.assert_replay_error(case, "E_TRACE_HASH")

    def test_wrong_raw_source_pin_is_rejected(self):
        with self.case() as case:
            case["source"].write_bytes(case["source"].read_bytes() + b"\n")
            self.assert_replay_error(case, "E_SOURCE_PIN")

    def test_duplicate_event_id_is_rejected(self):
        with self.case() as case:
            self.rewrite_records(
                case,
                lambda records: records[1].__setitem__(
                    "event_id", records[0]["event_id"]
                ),
            )
            self.assert_replay_error(case, "E_DUPLICATE_EVENT_ID")

    def test_duplicate_json_key_is_rejected(self):
        with self.case() as case:
            trace_path = case["run_dir"] / case["trace"]
            lines = trace_path.read_bytes().splitlines()
            record = json.loads(lines[0])
            needle = canonical({"event_id": record["event_id"]})[1:-1]
            replacement = needle + b"," + needle
            self.assertEqual(lines[0].count(needle), 1)
            lines[0] = lines[0].replace(needle, replacement)
            self.reseal_trace(case, b"\n".join(lines) + b"\n")
            self.assert_replay_error(case, "E_JSON_DUPLICATE_KEY")

    def test_disordered_trace_is_rejected(self):
        with self.case() as case:
            trace_path = case["run_dir"] / case["trace"]
            lines = trace_path.read_bytes().splitlines()
            lines[0], lines[1] = lines[1], lines[0]
            self.reseal_trace(case, b"\n".join(lines) + b"\n")
            self.assert_replay_error(case, "E_ORDER")

    def test_unknown_service_is_rejected(self):
        with self.case() as case:
            self.rewrite_records(
                case,
                lambda records: records[0].__setitem__("service", "unknown"),
            )
            self.assert_replay_error(case, "E_SERVICE_UNKNOWN")

    def test_dag_provenance_mismatch_is_rejected(self):
        with self.case(copy_dags=True) as case:
            dag_path = case["dag_dir"] / "api_generation.json"
            dag = json.loads(dag_path.read_text(encoding="ascii"))
            dag["provenance"] = "synthetic"
            dag_path.write_bytes(canonical(dag) + b"\n")
            self.assert_replay_error(case, "E_DAG_MISMATCH")

    def test_malformed_sidecar_and_artifact_are_rejected(self):
        for filename, field in (
            ("burstgpt-v2.median.manifest.json", "output_sha256"),
            ("normalize.artifact.json", "outputs"),
        ):
            with self.subTest(filename=filename), self.case() as case:
                path = case["run_dir"] / filename
                value = json.loads(path.read_text(encoding="ascii"))
                del value[field]
                path.write_bytes(canonical(value) + b"\n")
                self.assert_replay_error(case, "E_SCHEMA")

    def test_extra_artifact_input_is_rejected(self):
        with self.case() as case:
            path = case["run_dir"] / "normalize.artifact.json"
            artifact = json.loads(path.read_text(encoding="ascii"))
            artifact["inputs"].append(
                {
                    "role": "ignored",
                    "path": "ignored.bin",
                    "sha256": "sha256:" + "0" * 64,
                }
            )
            path.write_bytes(canonical(artifact) + b"\n")
            self.assert_replay_error(case, "E_ARTIFACT_BINDING")

    def test_bad_window_arithmetic_is_rejected(self):
        with self.case() as case:
            sidecar_path = case["run_dir"] / "burstgpt-v2.median.manifest.json"
            artifact_path = case["run_dir"] / "normalize.artifact.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="ascii"))
            sidecar["window"]["t_start_us"] += 1
            artifact = json.loads(artifact_path.read_text(encoding="ascii"))
            artifact["outputs"][0]["sidecar_manifest_sha256"] = digest(
                canonical(sidecar)
            )
            sidecar_path.write_bytes(canonical(sidecar) + b"\n")
            artifact_path.write_bytes(canonical(artifact) + b"\n")
            self.assert_replay_error(case, "E_MANIFEST_SEMANTICS")

    def test_bad_deterministic_replay_hash_is_rejected(self):
        with self.case() as case:
            path = case["run_dir"] / "normalize.artifact.json"
            artifact = json.loads(path.read_text(encoding="ascii"))
            artifact["deterministic_replay_sha256"] = "sha256:" + "0" * 64
            path.write_bytes(canonical(artifact) + b"\n")
            self.assert_replay_error(case, "E_ARTIFACT_BINDING")

    def test_record_gate_a_constants_are_bound_to_config(self):
        with self.case() as case:
            self.rewrite_records(
                case,
                lambda records: records[0].__setitem__("images", 1),
            )
            self.assert_replay_error(case, "E_GATE_A_FIELDS")

    def test_timestamp_at_window_end_is_rejected(self):
        with self.case() as case:
            self.rewrite_records(
                case,
                lambda records: records[-1].__setitem__("t_us", 900000000),
            )
            self.assert_replay_error(case, "E_ORDER")

    def test_resealed_provenance_cannot_override_config(self):
        with self.case(copy_dags=True) as case:
            self.rewrite_records(
                case,
                lambda records: [
                    record.__setitem__("provenance", "real_decomposed")
                    for record in records
                ],
            )
            sidecar_path = case["run_dir"] / "burstgpt-v2.median.manifest.json"
            artifact_path = case["run_dir"] / "normalize.artifact.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="ascii"))
            sidecar["provenance"] = "real_decomposed"
            artifact = json.loads(artifact_path.read_text(encoding="ascii"))
            artifact["outputs"][0]["sidecar_manifest_sha256"] = digest(
                canonical(sidecar)
            )
            sidecar_path.write_bytes(canonical(sidecar) + b"\n")
            artifact_path.write_bytes(canonical(artifact) + b"\n")
            for dag_path in case["dag_dir"].glob("*.json"):
                dag = json.loads(dag_path.read_text(encoding="ascii"))
                dag["provenance"] = "real_decomposed"
                dag_path.write_bytes(canonical(dag) + b"\n")
            self.assert_replay_error(case, "E_SOURCE_PIN")

    def test_window_metric_is_recomputed(self):
        with self.case() as case:
            sidecar_path = case["run_dir"] / "burstgpt-v2.median.manifest.json"
            artifact_path = case["run_dir"] / "normalize.artifact.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="ascii"))
            sidecar["window"]["metric_value"] = 1
            artifact = json.loads(artifact_path.read_text(encoding="ascii"))
            artifact["outputs"][0]["sidecar_manifest_sha256"] = digest(
                canonical(sidecar)
            )
            sidecar_path.write_bytes(canonical(sidecar) + b"\n")
            artifact_path.write_bytes(canonical(artifact) + b"\n")
            self.assert_replay_error(case, "E_MANIFEST_SEMANTICS")

    def test_quantile_window_requires_rank(self):
        with self.case() as case:
            sidecar_path = case["run_dir"] / "burstgpt-v2.median.manifest.json"
            artifact_path = case["run_dir"] / "normalize.artifact.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="ascii"))
            del sidecar["window"]["quantile_rank"]
            artifact = json.loads(artifact_path.read_text(encoding="ascii"))
            artifact["outputs"][0]["sidecar_manifest_sha256"] = digest(
                canonical(sidecar)
            )
            sidecar_path.write_bytes(canonical(sidecar) + b"\n")
            artifact_path.write_bytes(canonical(artifact) + b"\n")
            self.assert_replay_error(case, "E_MANIFEST_SEMANTICS")

    def test_timestamp_policy_is_bound_to_config(self):
        with self.case() as case:
            sidecar_path = case["run_dir"] / "burstgpt-v2.median.manifest.json"
            artifact_path = case["run_dir"] / "normalize.artifact.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="ascii"))
            sidecar["window"]["timestamp_policy"] = "sort_stable"
            artifact = json.loads(artifact_path.read_text(encoding="ascii"))
            artifact["outputs"][0]["sidecar_manifest_sha256"] = digest(
                canonical(sidecar)
            )
            sidecar_path.write_bytes(canonical(sidecar) + b"\n")
            artifact_path.write_bytes(canonical(artifact) + b"\n")
            self.assert_replay_error(case, "E_MANIFEST_SEMANTICS")

    def test_filters_are_frozen_empty(self):
        with self.case() as case:
            sidecar_path = case["run_dir"] / "burstgpt-v2.median.manifest.json"
            artifact_path = case["run_dir"] / "normalize.artifact.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="ascii"))
            sidecar["filters"] = ["exclude_zero_output"]
            sidecar["exclusion_counts"] = {"exclude_zero_output": 0}
            artifact = json.loads(artifact_path.read_text(encoding="ascii"))
            artifact["outputs"][0]["sidecar_manifest_sha256"] = digest(
                canonical(sidecar)
            )
            sidecar_path.write_bytes(canonical(sidecar) + b"\n")
            artifact_path.write_bytes(canonical(artifact) + b"\n")
            self.assert_replay_error(case, "E_MANIFEST_SEMANTICS")

    def test_quantile_rank_uses_frozen_formula(self):
        with self.case() as case:
            sidecar_path = case["run_dir"] / "burstgpt-v2.median.manifest.json"
            artifact_path = case["run_dir"] / "normalize.artifact.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="ascii"))
            sidecar["window"]["n_nonempty_bins"] = 10
            sidecar["window"]["quantile_rank"] = 1
            artifact = json.loads(artifact_path.read_text(encoding="ascii"))
            artifact["outputs"][0]["sidecar_manifest_sha256"] = digest(
                canonical(sidecar)
            )
            sidecar_path.write_bytes(canonical(sidecar) + b"\n")
            artifact_path.write_bytes(canonical(artifact) + b"\n")
            self.assert_replay_error(case, "E_MANIFEST_SEMANTICS")

    def test_normalization_seed_is_forbidden(self):
        with self.case() as case:
            path = case["run_dir"] / "normalize.artifact.json"
            artifact = json.loads(path.read_text(encoding="ascii"))
            artifact["seed"] = 7
            path.write_bytes(canonical(artifact) + b"\n")
            self.assert_replay_error(case, "E_ARTIFACT_BINDING")

    def test_run_id_is_recomputed(self):
        with self.case() as case:
            path = case["run_dir"] / "normalize.artifact.json"
            artifact = json.loads(path.read_text(encoding="ascii"))
            artifact["run_id"] = "normalize-forged"
            path.write_bytes(canonical(artifact) + b"\n")
            self.assert_replay_error(case, "E_ARTIFACT_BINDING")

    def test_missing_sibling_output_is_rejected(self):
        with self.case() as case:
            path = case["run_dir"] / "normalize.artifact.json"
            artifact = json.loads(path.read_text(encoding="ascii"))
            artifact["outputs"].append(
                {
                    "path": "burstgpt-v2.zzz.jsonl",
                    "sha256": "sha256:" + "0" * 64,
                    "sidecar_manifest_sha256": "sha256:" + "0" * 64,
                }
            )
            artifact["outputs"].sort(key=lambda output: output["path"])
            replay_preimage = "\n".join(
                output["sha256"] for output in artifact["outputs"]
            ).encode("ascii")
            artifact["deterministic_replay_sha256"] = digest(replay_preimage)
            path.write_bytes(canonical(artifact) + b"\n")
            self.assert_replay_error(case, "E_IO")

    def test_noncanonical_real_row_id_is_rejected(self):
        with self.case() as case:
            self.rewrite_records(
                case,
                lambda records: records[0].__setitem__(
                    "event_id", "burstgpt-v2:000"
                ),
            )
            self.assert_replay_error(case, "E_EVENT_ID")

    def test_dag_cycle_is_rejected(self):
        with self.case(copy_dags=True) as case:
            path = case["dag_dir"] / "api_generation.json"
            dag = json.loads(path.read_text(encoding="ascii"))
            dag["nodes"][0]["depends_on"] = [dag["nodes"][-1]["node_id"]]
            path.write_bytes(canonical(dag) + b"\n")
            self.assert_replay_error(case, "E_DAG")

    def test_empty_trace_is_rejected(self):
        with self.case() as case:
            self.reseal_trace(case, b"")
            self.assert_replay_error(case, "E_EMPTY_TRACE")

    def test_record_bound_is_rejected_before_record_parsing(self):
        trace = b"{}\n" * (replayer.MAX_RECORDS + 1)
        with self.assertRaises(replayer.ReplayError) as context:
            replayer.parse_trace(trace, None, {}, {})
        self.assertEqual(context.exception.code, "E_RECORD_LIMIT")

    def test_mixed_order_key_contract(self):
        self.assertEqual(
            replayer.order_key(request("mix:2:ragpulse:7", 11)),
            (11, 2, 7),
        )
        for event_id in (
            "mix:02:ragpulse:7",
            "mix:2:ragpulse:07",
            "mix:\u0662:ragpulse:7",
            "mix:2:ragpulse:\u0667",
        ):
            with self.subTest(event_id=event_id):
                with self.assertRaises(replayer.ReplayError) as context:
                    replayer.order_key(request(event_id, 11))
                self.assertEqual(context.exception.code, "E_EVENT_ID")

    def test_mixed_rank_order_is_rejected(self):
        records = [
            request("mix:1:a:0", 10),
            request("mix:0:b:1", 10),
        ]
        trace = b"".join(canonical(record) + b"\n" for record in records)
        validator = replayer.load_validator("request.schema.json")
        with self.assertRaises(replayer.ReplayError) as context:
            replayer.parse_trace(
                trace,
                validator,
                {
                    "source": "mix",
                    "provenance": "semi_synthetic",
                    "gate_a": {
                        "audio_ms": 0,
                        "deadline_provenance": "none",
                        "deadline_us": None,
                        "images": 0,
                        "observed_latency_us": None,
                        "priority_class": None,
                        "priority_provenance": "none",
                    },
                },
                {
                    "provenance": "semi_synthetic",
                    "window": {
                        "t_start_us": 0,
                        "t_end_us": 900000000,
                    },
                },
            )
        self.assertEqual(context.exception.code, "E_ORDER")


if __name__ == "__main__":
    unittest.main()
