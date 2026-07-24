#!/usr/bin/env python3

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
S39 = HERE.parent
SCRIPT = S39 / "build_replay_intents.py"

SPEC = importlib.util.spec_from_file_location("s39_build_replay_intents", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def load(path):
    return json.loads(path.read_text(encoding="ascii"))


def write(path, value):
    path.write_bytes(canonical_bytes(value))


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


class ReplayIntentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="s39_replay_test_")
        base = Path(self.temporary.name)
        self.root = base / "s39"
        shutil.copytree(
            S39,
            self.root,
            ignore=shutil.ignore_patterns("__pycache__", "replay_*.json*"),
        )
        source_s8 = S39.parent / "s8_operator_island_affinity"
        copied_s8 = base / "s8_operator_island_affinity"
        (copied_s8 / "configs").mkdir(parents=True)
        shutil.copy2(source_s8 / "normalize_trace.py", copied_s8 / "normalize_trace.py")
        shutil.copy2(
            source_s8 / "configs" / "burstgpt.config.json",
            copied_s8 / "configs" / "burstgpt.config.json",
        )
        self.output = base / "output"

    def tearDown(self):
        self.temporary.cleanup()

    def build(self):
        return REPLAY.build(
            root=self.root,
            selector_path=self.root / "ACTIVE_TRACE.json",
            shard_manifest_path=self.root / "SHARD_MANIFEST.json",
            output_dir=self.output,
        )

    def bundle(self):
        return self.root / "bundle_frequent"

    def refresh_links(self):
        bundle = self.bundle()
        requests_path = bundle / "requests.jsonl"
        assignment_path = bundle / "model_assignment.json"
        assignment = load(assignment_path)
        requests_raw = requests_path.read_bytes()
        assignment["source_trace_sha256"] = sha256(requests_raw)
        write(assignment_path, assignment)
        assignment_raw = assignment_path.read_bytes()
        manifest_path = bundle / "manifest.json"
        manifest = load(manifest_path)
        manifest["outputs"]["requests"].update(
            {
                "bytes": len(requests_raw),
                "records": requests_raw.count(b"\n"),
                "sha256": sha256(requests_raw),
            }
        )
        manifest["outputs"]["model_assignment"].update(
            {
                "bytes": len(assignment_raw),
                "records": 1,
                "sha256": sha256(assignment_raw),
            }
        )
        write(manifest_path, manifest)

    def mutate_request(self, index, callback):
        path = self.bundle() / "requests.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]
        callback(rows, index)
        path.write_bytes(b"".join(canonical_bytes(row) for row in rows))
        self.refresh_links()

    def assert_rejected(self, fragment):
        with self.assertRaisesRegex(REPLAY.ReplayError, fragment):
            self.build()

    def test_frozen_trace_derives_five_windows_and_nine_changes(self):
        manifest = self.build()
        self.assertEqual(manifest["derived"]["promotion_windows"], 5)
        self.assertEqual(manifest["derived"]["target_changes"], 9)
        intents = [
            json.loads(line)
            for line in (self.output / "replay_intents.jsonl")
            .read_text(encoding="ascii")
            .splitlines()
        ]
        self.assertEqual(
            [record["t_us"] for record in intents],
            [
                60_000_000,
                129_000_000,
                359_000_000,
                421_000_000,
                498_000_000,
                546_000_000,
                911_000_000,
                989_000_000,
                1_154_000_000,
            ],
        )
        self.assertEqual(
            [record["kind"] for record in intents],
            ["PROMOTE", "DEMOTE"] * 4 + ["PROMOTE"],
        )
        self.assertEqual(len({record["intent_id"] for record in intents}), 9)
        self.assertEqual(len({record["window_id"] for record in intents}), 5)

    def test_output_binds_all_input_artifacts(self):
        result = self.build()
        expected = {
            "active_trace_sha256": sha256(
                (self.root / "ACTIVE_TRACE.json").read_bytes()
            ),
            "bundle_manifest_sha256": sha256(
                (self.bundle() / "manifest.json").read_bytes()
            ),
            "model_assignment_sha256": sha256(
                (self.bundle() / "model_assignment.json").read_bytes()
            ),
            "replay_builder_sha256": sha256(
                (self.root / "build_replay_intents.py").read_bytes()
            ),
            "requests_sha256": sha256(
                (self.bundle() / "requests.jsonl").read_bytes()
            ),
            "shard_manifest_sha256": sha256(
                (self.root / "SHARD_MANIFEST.json").read_bytes()
            ),
        }
        self.assertEqual(result["inputs"], expected)
        raw = (self.output / "replay_intents.jsonl").read_bytes()
        self.assertEqual(result["output"]["sha256"], sha256(raw))
        self.assertEqual(result["output"]["bytes"], len(raw))

    def test_tampered_request_artifact_is_rejected(self):
        path = self.bundle() / "requests.jsonl"
        raw = bytearray(path.read_bytes())
        raw[10] ^= 1
        path.write_bytes(raw)
        self.assert_rejected("SHA-256 mismatch")

    def test_tampered_builder_artifact_is_rejected(self):
        path = self.root / "build_frequent_trace.py"
        path.write_bytes(path.read_bytes() + b"\n")
        self.assert_rejected("local SHA-256 mismatch")

    def test_nonmonotonic_trace_is_rejected_after_rebinding(self):
        path = self.bundle() / "requests.jsonl"
        rows = path.read_bytes().splitlines(keepends=True)
        rows[0], rows[1] = rows[1], rows[0]
        path.write_bytes(b"".join(rows))
        self.refresh_links()
        self.assert_rejected("nonmonotonic")

    def test_unknown_source_model_is_rejected_after_rebinding(self):
        self.mutate_request(
            0,
            lambda rows, index: rows[index]["source_fields"].__setitem__(
                "model", "UnknownModel"
            ),
        )
        self.assert_rejected("unknown model")

    def test_unbound_model_artifact_is_rejected(self):
        path = self.bundle() / "model_assignment.json"
        assignment = load(path)
        assignment["mappings"]["GPT-4"]["artifact"]["sha256"] = "0" * 64
        write(path, assignment)
        self.refresh_links()
        self.assert_rejected("artifact is not bound")

    def test_noncanonical_model_id_is_rejected(self):
        path = self.bundle() / "model_assignment.json"
        assignment = load(path)
        assignment["mappings"]["GPT-4"]["model_id"] = "Qwen3/14B"
        write(path, assignment)
        self.refresh_links()
        self.assert_rejected("model_id")

    def test_manifest_cannot_forge_declared_switch_count(self):
        path = self.bundle() / "manifest.json"
        manifest = load(path)
        manifest["statistics"]["qualifying_switch_cycles"] = 6
        write(path, manifest)
        self.assert_rejected("statistics do not match")

    def test_float_cannot_pass_integer_statistic(self):
        path = self.bundle() / "manifest.json"
        manifest = load(path)
        manifest["statistics"]["qualifying_switch_cycles"] = 5.0
        write(path, manifest)
        self.assert_rejected("statistics do not match")

    def test_float_cannot_pass_shard_artifact_size(self):
        path = self.root / "SHARD_MANIFEST.json"
        manifest = load(path)
        manifest["models"]["qwen3-14b-q4_k_m"]["source"]["bytes"] = 9_001_752_960.0
        write(path, manifest)
        self.assert_rejected("expected integer")

    def test_duplicate_selector_key_is_rejected(self):
        (self.root / "ACTIVE_TRACE.json").write_text(
            '{"schema_version":1,"schema_version":1}\n',
            encoding="ascii",
        )
        self.assert_rejected("duplicate JSON key")

    def test_boolean_cannot_pass_schema_version(self):
        path = self.root / "ACTIVE_TRACE.json"
        selector = load(path)
        selector["schema_version"] = True
        write(path, selector)
        self.assert_rejected("expected integer")

    def test_selector_path_escape_is_rejected(self):
        path = self.root / "ACTIVE_TRACE.json"
        selector = load(path)
        selector["active_bundle"] = "../bundle_frequent"
        write(path, selector)
        self.assert_rejected("unsafe")

    def test_unknown_request_field_is_rejected_after_rebinding(self):
        self.mutate_request(
            0,
            lambda rows, index: rows[index].__setitem__("unexpected", 1),
        )
        self.assert_rejected("keys differ")

    def test_noncanonical_request_bytes_are_rejected_after_rebinding(self):
        path = self.bundle() / "requests.jsonl"
        rows = path.read_text(encoding="ascii").splitlines()
        first = json.loads(rows[0])
        rows[0] = json.dumps(first, sort_keys=True)
        path.write_text("\n".join(rows) + "\n", encoding="ascii")
        self.refresh_links()
        self.assert_rejected("not canonical JSON")

    def test_cli_failure_is_nonzero_without_traceback(self):
        path = self.root / "ACTIVE_TRACE.json"
        selector = load(path)
        selector["active_profile"] = "wrong"
        write(path, selector)
        process = subprocess.run(
            [
                sys.executable,
                str(self.root / "build_replay_intents.py"),
                "--root",
                str(self.root),
                "--selector",
                str(path),
                "--shard-manifest",
                str(self.root / "SHARD_MANIFEST.json"),
                "--output",
                str(self.output),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("S39_REPLAY_ERROR:", process.stderr)
        self.assertNotIn("Traceback", process.stderr)

    def test_cross_process_output_is_byte_identical(self):
        outputs = []
        for seed in ("0", "1", "42", "12345"):
            output = Path(self.temporary.name) / f"determinism-{seed}"
            process = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--root",
                    str(S39),
                    "--selector",
                    str(S39 / "ACTIVE_TRACE.json"),
                    "--shard-manifest",
                    str(S39 / "SHARD_MANIFEST.json"),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": seed,
                },
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            outputs.append(
                (
                    (output / "replay_intents.jsonl").read_bytes(),
                    (output / "replay_manifest.json").read_bytes(),
                )
            )
        self.assertTrue(all(value == outputs[0] for value in outputs[1:]))


if __name__ == "__main__":
    unittest.main()
