#!/usr/bin/env python3
"""Focused fail-closed tests for the S8 Gate-A normalizer."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import pathlib
import py_compile
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import normalize_trace as normalizer  # noqa: E402


CONFIGS = ROOT / "configs"


def load_config(name):
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def physical_line_count(data):
    return sum(1 for _ in io.BytesIO(data))


def pin_config(config, data, record_count):
    result = copy.deepcopy(config)
    result["origin"]["bytes"] = len(data)
    result["origin"]["sha256"] = "sha256:" + hashlib.sha256(data).hexdigest()
    result["origin"]["line_count"] = physical_line_count(data)
    result["origin"]["record_count"] = record_count
    return result


def write_case(root, data, config):
    source = root / "source.data"
    config_path = root / "config.json"
    source.write_bytes(data)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return source, config_path


def args(source, config, output=None, scenario="all", scan_only=False, force=False):
    return Namespace(
        source_file=str(source),
        config=str(config),
        output_dir=None if output is None else str(output),
        scenario=scenario,
        scan_only=scan_only,
        force=force,
    )


def burst_data(rows, newline=b"\r\n", header=None):
    fields = [
        "Timestamp",
        "Session ID",
        "Elapsed time",
        "Model",
        "Request tokens",
        "Response tokens",
        "Total tokens",
        "Log Type",
    ]
    encoded = [(",".join(header or fields)).encode("ascii")]
    encoded.extend((",".join(row)).encode("ascii") for row in rows)
    return newline.join(encoded) + newline


def rag_record(timestamp, input_tokens, output_tokens, session):
    return {
        "timestamp": str(timestamp),
        "input_length": input_tokens,
        "output_length": output_tokens,
        "session_id": session,
        "hash_ids": {
            "sys_prompt": [1],
            "passages_ids": [2, 3],
            "history": [4],
            "web_search": [],
            "user_input": [5],
        },
    }


def rag_data(records, terminal_blank=True):
    body = b"\n".join(
        json.dumps(record, separators=(",", ":")).encode("ascii")
        for record in records
    )
    return body + (b"\n\n" if terminal_blank else b"\n")


class PositiveNormalization(unittest.TestCase):
    def test_direct_cli_is_the_only_certifying_entrypoint(self):
        records = [
            rag_record(1000, 60, 40, "later"),
            rag_record(100, 1, 0, "first"),
            rag_record(200, 1, 0, "second"),
        ]
        data = rag_data(records)
        config = pin_config(load_config("ragpulse.config.json"), data, len(records))
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, config_path = write_case(root, data, config)
            output = root / "out"
            process = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "normalize_trace.py"),
                    "--source-file",
                    str(source),
                    "--config",
                    str(config_path),
                    "--scenario",
                    "all",
                    "--output-dir",
                    str(output),
                ],
                capture_output=True,
                text=True,
                env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertEqual(len(result["outputs"]), 4)
            artifact = json.loads(
                (output / "normalize.artifact.json").read_text(encoding="ascii")
            )
            self.assertEqual(
                artifact["gate_results"],
                {
                    "atomic_run_publish": True,
                    "direct_source_execution": True,
                    "source_snapshot_verified": True,
                },
            )

    def test_burst_windows_are_exact_and_deterministic(self):
        rows = [
            ["0.0", "s0", "1", "GPT-4", "10", "10", "20", "Conversation log"],
            ["100.0", "s1", "2", "ChatGPT", "5", "5", "10", "Conversation log"],
            ["900.0", "", "3", "GPT-4", "5", "0", "5", "API log"],
            ["1800.0", "", "4", "GPT-4", "80", "20", "100", "API log"],
            ["2700.0", "s4", "5", "ChatGPT", "30", "20", "50", "Conversation log"],
            ["3600.0", "", "6", "GPT-4", "150", "50", "200", "API log"],
        ]
        data = burst_data(rows)
        config = pin_config(load_config("burstgpt.config.json"), data, len(rows))
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, config_path = write_case(root, data, config)
            first = normalizer._normalize_impl(args(source, config_path, root / "out1"))
            second = normalizer._normalize_impl(args(source, config_path, root / "out2"))

            self.assertEqual(first["scenarios"]["low"]["bin_index"], 1)
            self.assertEqual(first["scenarios"]["median"]["bin_index"], 3)
            self.assertEqual(first["scenarios"]["high"]["bin_index"], 4)
            self.assertEqual(first["scenarios"]["burst"]["bin_index"], 4)
            for scenario in normalizer.SCENARIOS:
                out1 = root / "out1" / f"burstgpt-v2.{scenario}.jsonl"
                out2 = root / "out2" / f"burstgpt-v2.{scenario}.jsonl"
                side1 = root / "out1" / f"burstgpt-v2.{scenario}.manifest.json"
                side2 = root / "out2" / f"burstgpt-v2.{scenario}.manifest.json"
                self.assertEqual(out1.read_bytes(), out2.read_bytes())
                self.assertEqual(side1.read_bytes(), side2.read_bytes())
                self.assertTrue(out1.read_bytes().endswith(b"\n"))
                self.assertNotIn(b"\r", out1.read_bytes())
                manifest = json.loads(side1.read_text(encoding="ascii"))
                self.assertEqual(
                    manifest["output_sha256"],
                    "sha256:" + hashlib.sha256(out1.read_bytes()).hexdigest(),
                )
                expected_sidecar_hash = normalizer.sha256_bytes(
                    normalizer.canonical_json(manifest)
                )
                output_info = next(
                    output
                    for output in first["outputs"]
                    if output["scenario"] == scenario
                )
                self.assertEqual(
                    output_info["sidecar_manifest_sha256"],
                    expected_sidecar_hash,
                )

            median = json.loads(
                (root / "out1" / "burstgpt-v2.median.jsonl").read_text(
                    encoding="ascii"
                )
            )
            self.assertEqual(median["event_id"], "burstgpt-v2:4")
            self.assertEqual(median["t_us"], 0)
            self.assertEqual(median["source_fields"]["elapsed_time"], 5)

    def test_rag_stable_sort_and_single_terminal_blank(self):
        records = [
            rag_record(1000, 60, 40, "later"),
            rag_record(100, 1, 0, "first"),
            rag_record(200, 1, 0, "second"),
            rag_record(1800, 25, 25, "middle"),
        ]
        data = rag_data(records)
        config = pin_config(load_config("ragpulse.config.json"), data, len(records))
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, config_path = write_case(root, data, config)
            result = normalizer._normalize_impl(args(source, config_path, root / "out"))
            self.assertEqual(result["terminal_blank_lines"], 1)
            self.assertEqual(result["input_nonmonotonic_pairs"], 1)
            lines = (
                root / "out" / "ragpulse.low.jsonl"
            ).read_text(encoding="ascii").splitlines()
            parsed = [json.loads(line) for line in lines]
            self.assertEqual(
                [record["event_id"] for record in parsed],
                ["ragpulse:1", "ragpulse:2"],
            )
            self.assertEqual(
                parsed[0]["cache_keys"],
                [
                    "sys_prompt:1",
                    "passages_ids:2",
                    "passages_ids:3",
                    "history:4",
                    "user_input:5",
                ],
            )
            sidecar = json.loads(
                (root / "out" / "ragpulse.low.manifest.json").read_text(
                    encoding="ascii"
                )
            )
            self.assertEqual(sidecar["window"]["timestamp_policy"], "sort_stable")
            self.assertEqual(sidecar["window"]["input_nonmonotonic_pairs"], 1)
            artifact = json.loads(
                (root / "out" / "normalize.artifact.json").read_text(
                    encoding="ascii"
                )
            )
            self.assertEqual(len(artifact["outputs"]), 4)
            replay_preimage = "\n".join(
                output["sha256"]
                for output in sorted(
                    artifact["outputs"],
                    key=lambda output: output["path"],
                )
            ).encode("ascii")
            self.assertEqual(
                artifact["deterministic_replay_sha256"],
                normalizer.sha256_bytes(replay_preimage),
            )
            self.assertEqual(
                result["artifact_manifest_sha256"],
                normalizer.sha256_bytes(normalizer.canonical_json(artifact)),
            )

    def test_scan_only_writes_nothing(self):
        rows = [
            ["0.0", "", "1", "GPT-4", "1", "1", "2", "API log"],
        ]
        data = burst_data(rows)
        config = pin_config(load_config("burstgpt.config.json"), data, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, config_path = write_case(root, data, config)
            result = normalizer._normalize_impl(
                args(source, config_path, scan_only=True)
            )
            self.assertEqual(result["status"], "scan_only")
            self.assertEqual(result["record_count"], 1)
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["config.json", "source.data"],
            )


class FailClosedInputs(unittest.TestCase):
    def test_imported_public_entrypoint_is_rejected(self):
        with self.assertRaises(normalizer.NormalizeError) as context:
            normalizer.normalize(Namespace())
        self.assertEqual(context.exception.code, "E_EXECUTION_PROVENANCE")

    def test_direct_pyc_entrypoint_is_rejected(self):
        rows = [["0.0", "", "1", "GPT-4", "1", "1", "2", "API log"]]
        data = burst_data(rows)
        config = pin_config(load_config("burstgpt.config.json"), data, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, config_path = write_case(root, data, config)
            pyc_path = root / "runner.pyc"
            py_compile.compile(
                str(ROOT / "normalize_trace.py"),
                cfile=str(pyc_path),
                doraise=True,
            )
            output = root / "out"
            process = subprocess.run(
                [
                    sys.executable,
                    str(pyc_path),
                    "--source-file",
                    str(source),
                    "--config",
                    str(config_path),
                    "--scenario",
                    "median",
                    "--output-dir",
                    str(output),
                ],
                capture_output=True,
                text=True,
                env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(process.returncode, 1)
            self.assertIn("E_EXECUTION_PROVENANCE", process.stderr)
            self.assertFalse(output.exists())

    def assert_error(self, data, config, code, scan_only=True):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, config_path = write_case(root, data, config)
            with self.assertRaises(normalizer.NormalizeError) as context:
                normalizer._normalize_impl(
                    args(
                        source,
                        config_path,
                        root / "out",
                        scan_only=scan_only,
                    )
                )
            self.assertEqual(context.exception.code, code)
            self.assertFalse((root / "out").exists())

    def test_bom_is_rejected_before_hash_trust(self):
        rows = [["0.0", "", "1", "GPT-4", "1", "1", "2", "API log"]]
        data = b"\xef\xbb\xbf" + burst_data(rows)
        config = pin_config(load_config("burstgpt.config.json"), data, 1)
        self.assert_error(data, config, "E_BOM")

    def test_lone_cr_is_rejected(self):
        rows = [["0.0", "", "1", "GPT-4", "1", "1", "2", "API log"]]
        complete = burst_data(rows, newline=b"\n")
        data = complete[:-1] + b"\r"
        config = pin_config(load_config("burstgpt.config.json"), data, 1)
        self.assert_error(data, config, "E_NEWLINE")

    def test_csv_blank_line_is_rejected(self):
        rows = [["0.0", "", "1", "GPT-4", "1", "1", "2", "API log"]]
        data = burst_data(rows) + b"\r\n"
        config = pin_config(load_config("burstgpt.config.json"), data, 2)
        self.assert_error(data, config, "E_BLANK_ROW")

    def test_rag_interior_blank_is_rejected(self):
        first = json.dumps(rag_record(0, 1, 1, "a")).encode("ascii")
        second = json.dumps(rag_record(1, 1, 1, "b")).encode("ascii")
        data = first + b"\n\n" + second + b"\n\n"
        config = pin_config(load_config("ragpulse.config.json"), data, 3)
        self.assert_error(data, config, "E_BLANK_ROW")

    def test_rag_multiple_terminal_blanks_are_rejected(self):
        data = json.dumps(rag_record(0, 1, 1, "a")).encode("ascii") + b"\n\n\n"
        config = pin_config(load_config("ragpulse.config.json"), data, 2)
        self.assert_error(data, config, "E_BLANK_ROW")

    def test_duplicate_json_key_is_rejected(self):
        data = (
            b'{"timestamp":"0","timestamp":"1","input_length":1,'
            b'"output_length":1,"session_id":"s","hash_ids":'
            b'{"sys_prompt":[1],"passages_ids":[],"history":[2],'
            b'"web_search":[],"user_input":[3]}}\n\n'
        )
        config = pin_config(load_config("ragpulse.config.json"), data, 1)
        self.assert_error(data, config, "E_JSON_DUPLICATE_KEY")

    def test_bool_is_not_an_integer(self):
        record = rag_record(0, 1, 1, "a")
        record["input_length"] = True
        data = rag_data([record])
        config = pin_config(load_config("ragpulse.config.json"), data, 1)
        self.assert_error(data, config, "E_JSON_TYPE")

    def test_signed_zero_is_rejected_by_integer_grammar(self):
        data = (
            b'{"timestamp":"0","input_length":-0,"output_length":1,'
            b'"session_id":"s","hash_ids":{"sys_prompt":[1],'
            b'"passages_ids":[],"history":[2],"web_search":[],'
            b'"user_input":[3]}}\n\n'
        )
        config = pin_config(load_config("ragpulse.config.json"), data, 1)
        self.assert_error(data, config, "E_INTEGER")

    def test_burst_timestamp_decrease_is_rejected(self):
        rows = [
            ["2.0", "", "1", "GPT-4", "1", "1", "2", "API log"],
            ["1.0", "", "1", "GPT-4", "1", "1", "2", "API log"],
        ]
        data = burst_data(rows)
        config = pin_config(load_config("burstgpt.config.json"), data, 2)
        self.assert_error(data, config, "E_TIMESTAMP_ORDER")

    def test_source_hash_mismatch_is_rejected(self):
        rows = [["0.0", "", "1", "GPT-4", "1", "1", "2", "API log"]]
        data = burst_data(rows)
        config = pin_config(load_config("burstgpt.config.json"), data, 1)
        config["origin"]["sha256"] = "sha256:" + "0" * 64
        self.assert_error(data, config, "E_SOURCE_HASH")

    def test_header_mismatch_is_rejected(self):
        rows = [["0.0", "", "1", "GPT-4", "1", "1", "2", "API log"]]
        bad_header = [
            "Timestamp",
            "Session",
            "Elapsed time",
            "Model",
            "Request tokens",
            "Response tokens",
            "Total tokens",
            "Log Type",
        ]
        data = burst_data(rows, header=bad_header)
        config = pin_config(load_config("burstgpt.config.json"), data, 1)
        self.assert_error(data, config, "E_HEADER")

    def test_unsupported_schema_version_is_rejected(self):
        rows = [["0.0", "", "1", "GPT-4", "1", "1", "2", "API log"]]
        data = burst_data(rows)
        config = pin_config(load_config("burstgpt.config.json"), data, 1)
        config["schema_version"] = 2
        self.assert_error(data, config, "E_CONFIG")

    def test_terminal_blank_exception_is_ragpulse_only(self):
        data = rag_data([rag_record(0, 1, 1, "a")])
        config = pin_config(load_config("ragpulse.config.json"), data, 1)
        config["source"] = "other-source"
        self.assert_error(data, config, "E_CONFIG")

    def test_unsupported_json_source_fields_are_rejected(self):
        data = rag_data([rag_record(0, 1, 1, "a")])
        config = pin_config(load_config("ragpulse.config.json"), data, 1)
        config["mapping"]["source_fields"]["session"] = {
            "from": "session_id",
            "type": "string",
            "transform": "raw",
        }
        self.assert_error(data, config, "E_CONFIG")

    def test_existing_output_is_not_overwritten(self):
        rows = [["0.0", "", "1", "GPT-4", "1", "1", "2", "API log"]]
        data = burst_data(rows)
        config = pin_config(load_config("burstgpt.config.json"), data, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, config_path = write_case(root, data, config)
            output = root / "out"
            normalizer._normalize_impl(
                args(source, config_path, output, scenario="median")
            )
            before = (output / "burstgpt-v2.median.jsonl").read_bytes()
            with self.assertRaises(normalizer.NormalizeError) as context:
                normalizer._normalize_impl(
                    args(source, config_path, output, scenario="median")
                )
            self.assertEqual(context.exception.code, "E_EXISTS")
            self.assertEqual(
                (output / "burstgpt-v2.median.jsonl").read_bytes(),
                before,
            )
            self.assertEqual(
                [path for path in output.iterdir() if path.name.startswith(".")],
                [],
            )

    def test_stage_failure_leaves_no_partial_run(self):
        rows = [["0.0", "", "1", "GPT-4", "1", "1", "2", "API log"]]
        data = burst_data(rows)
        config = pin_config(load_config("burstgpt.config.json"), data, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, config_path = write_case(root, data, config)
            output = root / "out"
            original = normalizer._write_bytes_fsync
            calls = 0

            def fail_second(path, payload):
                nonlocal calls
                calls += 1
                if calls == 2:
                    normalizer.fail("E_TEST_INJECTED", "injected stage failure")
                original(path, payload)

            with mock.patch.object(
                normalizer,
                "_write_bytes_fsync",
                side_effect=fail_second,
            ):
                with self.assertRaises(normalizer.NormalizeError) as context:
                    normalizer._normalize_impl(
                        args(source, config_path, output, scenario="median")
                    )
            self.assertEqual(context.exception.code, "E_TEST_INJECTED")
            self.assertFalse(output.exists())
            self.assertEqual(
                [path for path in root.iterdir() if path.name.startswith(".out.")],
                [],
            )

    def test_verified_snapshot_is_the_only_parse_source(self):
        original_rows = [
            ["0.0", "", "1", "GPT-4", "7", "1", "8", "API log"],
        ]
        replacement_rows = [
            ["0.0", "", "1", "GPT-4", "999", "1", "1000", "API log"],
        ]
        original_data = burst_data(original_rows)
        replacement_data = burst_data(replacement_rows)
        config = pin_config(load_config("burstgpt.config.json"), original_data, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, config_path = write_case(root, original_data, config)
            output = root / "out"
            real_copy = normalizer.copy_source_snapshot

            def copy_then_replace(source_path, snapshot_path):
                real_copy(source_path, snapshot_path)
                source_path.write_bytes(replacement_data)

            with mock.patch.object(
                normalizer,
                "copy_source_snapshot",
                side_effect=copy_then_replace,
            ):
                normalizer._normalize_impl(
                    args(source, config_path, output, scenario="median")
                )
            record = json.loads(
                (output / "burstgpt-v2.median.jsonl").read_text(encoding="ascii")
            )
            self.assertEqual(record["input_tokens"], 7)
            manifest = json.loads(
                (output / "burstgpt-v2.median.manifest.json").read_text(
                    encoding="ascii"
                )
            )
            self.assertEqual(manifest["source_sha256"], config["origin"]["sha256"])

    def test_code_change_before_publish_is_rejected(self):
        rows = [["0.0", "", "1", "GPT-4", "1", "1", "2", "API log"]]
        data = burst_data(rows)
        config = pin_config(load_config("burstgpt.config.json"), data, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, config_path = write_case(root, data, config)
            output = root / "out"
            with mock.patch.object(
                normalizer,
                "normalizer_version",
                return_value="sha256:" + "0" * 64,
            ):
                with self.assertRaises(normalizer.NormalizeError) as context:
                    normalizer._normalize_impl(
                        args(source, config_path, output, scenario="median")
                    )
            self.assertEqual(context.exception.code, "E_CODE_CHANGED")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
