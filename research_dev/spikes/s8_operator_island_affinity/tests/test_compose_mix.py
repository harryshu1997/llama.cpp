#!/usr/bin/python3
"""Fail-closed and determinism tests for the S8 mix-v1 composer.

Runs under the project venv (no jsonschema dependency): the composer validates
its config and components structurally in code. The composer is always invoked
as a subprocess so its direct-source-execution certification holds.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
NORMALIZER = ROOT / "normalize_trace.py"
COMPOSER = ROOT / "compose_mix.py"
ENV = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def burst_source():
    header = [
        "Timestamp", "Session ID", "Elapsed time", "Model",
        "Request tokens", "Response tokens", "Total tokens", "Log Type",
    ]
    rows = [
        ["0.0", "s0", "1", "GPT-4", "10", "2", "12", "Conversation log"],
        ["1.0", "", "2", "ChatGPT", "3", "1", "4", "API log"],
        ["2.0", "s2", "3", "GPT-4", "5", "4", "9", "Conversation log"],
    ]
    return b"\n".join(",".join(row).encode("ascii") for row in [header, *rows]) + b"\n"


def rag_source():
    def row(ts, il, ol, sid, pas):
        return json.dumps(
            {
                "timestamp": ts, "input_length": il, "output_length": ol,
                "session_id": sid,
                "hash_ids": {
                    "sys_prompt": [1], "passages_ids": pas, "history": [9],
                    "web_search": [], "user_input": [7],
                },
            },
            separators=(",", ":"), ensure_ascii=True,
        )
    lines = [row("0", 8, 4, "r0", [11, 12]), row("1", 6, 2, "r1", []), row("2", 5, 5, "r2", [13])]
    return ("\n".join(lines) + "\n\n").encode("ascii")


def normalize_component(tmp, name, cfg_name, source_bytes, line_count, record_count):
    src = tmp / name
    src.write_bytes(source_bytes)
    cfg = json.loads((CONFIGS / cfg_name).read_text(encoding="ascii"))
    cfg["origin"].update(
        {
            "bytes": len(source_bytes),
            "sha256": digest(source_bytes),
            "line_count": line_count,
            "record_count": record_count,
        }
    )
    cfg_path = tmp / cfg_name
    cfg_path.write_bytes(canonical(cfg) + b"\n")
    run_dir = tmp / (name + ".run")
    proc = subprocess.run(
        [
            sys.executable, str(NORMALIZER), "--source-file", str(src),
            "--config", str(cfg_path), "--scenario", "median", "--output-dir", str(run_dir),
        ],
        capture_output=True, text=True, env=ENV,
    )
    if proc.returncode != 0:
        raise AssertionError("normalizer failed: " + proc.stderr)
    return run_dir


class ComposeMix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(cls._tmp.name)
        cls.b_run = normalize_component(base, "BurstGPT_3.csv", "burstgpt.config.json", burst_source(), 4, 3)
        cls.r_run = normalize_component(base, "0_trace.jsonl", "ragpulse.config.json", rag_source(), 4, 3)
        cls.b_trace = (cls.b_run / "burstgpt-v2.median.jsonl").read_bytes()
        cls.b_side = json.loads((cls.b_run / "burstgpt-v2.median.manifest.json").read_bytes())
        cls.r_trace = (cls.r_run / "ragpulse.median.jsonl").read_bytes()
        cls.r_side = json.loads((cls.r_run / "ragpulse.median.manifest.json").read_bytes())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def base_config(self):
        return {
            "schema_version": 1,
            "provenance": "semi_synthetic",
            "mix_id": "mix-v1",
            "streams": [
                {
                    "rank": 0, "source": "burstgpt-v2",
                    "source_revision": self.b_side["source_revision"],
                    "component_trace": "burstgpt-v2.median.jsonl",
                    "scale_num": 1, "scale_den": 1, "offset_us": 0,
                    "input_output_sha256": digest(self.b_trace),
                    "input_manifest_sha256": digest(canonical(self.b_side)),
                },
                {
                    "rank": 1, "source": "ragpulse",
                    "source_revision": self.r_side["source_revision"],
                    "component_trace": "ragpulse.median.jsonl",
                    "scale_num": 1, "scale_den": 1, "offset_us": 500,
                    "input_output_sha256": digest(self.r_trace),
                    "input_manifest_sha256": digest(canonical(self.r_side)),
                },
            ],
        }

    def run_compose(self, config, out_name="mix.run", components=None):
        tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        b_run = tmp / "b.run"
        r_run = tmp / "r.run"
        shutil.copytree(self.b_run, b_run)
        shutil.copytree(self.r_run, r_run)
        cfg_path = tmp / "mix-v1.config.json"
        cfg_path.write_bytes(canonical(config) + b"\n")
        if components is None:
            components = {0: b_run, 1: r_run}
        argv = [sys.executable, str(COMPOSER), "--config", str(cfg_path)]
        for rank, path in components.items():
            argv += ["--component", f"{rank}:{path}"]
        argv += ["--output-dir", str(tmp / out_name)]
        proc = subprocess.run(argv, capture_output=True, text=True, env=ENV)
        return proc, tmp, {"b_run": b_run, "r_run": r_run}

    def assert_fail(self, proc, code):
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertTrue(proc.stderr.startswith(code + ":"), proc.stderr)

    # --- happy path + determinism -----------------------------------------
    def test_compose_succeeds_and_is_deterministic(self):
        proc, tmp, runs = self.run_compose(self.base_config(), out_name="mix.a")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        first = json.loads(proc.stdout)
        self.assertEqual(first["output_row_count"], 6)
        argv = [
            sys.executable, str(COMPOSER), "--config", str(tmp / "mix-v1.config.json"),
            "--component", f"0:{runs['b_run']}", "--component", f"1:{runs['r_run']}",
            "--output-dir", str(tmp / "mix.b"),
        ]
        proc2 = subprocess.run(argv, capture_output=True, text=True, env=ENV)
        self.assertEqual(proc2.returncode, 0, proc2.stderr)
        second = json.loads(proc2.stdout)
        self.assertEqual(first["output_sha256"], second["output_sha256"])
        self.assertEqual(first["artifact_run_id"], second["artifact_run_id"])
        # published files exist and hashes are self-consistent
        trace = (tmp / "mix.a" / "mix-v1.jsonl").read_bytes()
        self.assertEqual(digest(trace), first["output_sha256"])

    def test_offset_interleaves_streams(self):
        proc, tmp, _ = self.run_compose(self.base_config())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = (tmp / "mix.run" / "mix-v1.jsonl").read_text(encoding="ascii").splitlines()
        ids = [json.loads(line)["event_id"] for line in lines]
        self.assertEqual(ids[0], "mix:0:burstgpt-v2:0")
        self.assertEqual(ids[1], "mix:1:ragpulse:0")  # offset 500 places it after burst t=0
        provs = {json.loads(line)["provenance"] for line in lines}
        self.assertEqual(provs, {"semi_synthetic"})

    # --- config fail-closed -----------------------------------------------
    def test_single_stream_config_is_rejected(self):
        config = self.base_config()
        config["streams"] = config["streams"][:1]
        proc, _, _ = self.run_compose(config)
        self.assert_fail(proc, "E_CONFIG")

    def test_duplicate_rank_is_rejected(self):
        config = self.base_config()
        config["streams"][1]["rank"] = 0
        proc, _, _ = self.run_compose(config)
        self.assert_fail(proc, "E_CONFIG")

    def test_nonascending_rank_is_rejected(self):
        config = self.base_config()
        config["streams"][0]["rank"] = 5
        config["streams"][1]["rank"] = 2
        proc, _, _ = self.run_compose(config)
        self.assert_fail(proc, "E_CONFIG")

    def test_duplicate_component_binding_is_rejected(self):
        config = self.base_config()
        config["streams"][1]["input_output_sha256"] = config["streams"][0]["input_output_sha256"]
        config["streams"][1]["input_manifest_sha256"] = config["streams"][0]["input_manifest_sha256"]
        proc, _, _ = self.run_compose(config)
        self.assert_fail(proc, "E_CONFIG")

    def test_wrong_provenance_is_rejected(self):
        config = self.base_config()
        config["provenance"] = "real"
        proc, _, _ = self.run_compose(config)
        self.assert_fail(proc, "E_CONFIG")

    # --- component-pin fail-closed ----------------------------------------
    def test_component_trace_hash_mismatch_is_rejected(self):
        config = self.base_config()
        config["streams"][0]["input_output_sha256"] = digest(b"tampered")
        proc, _, _ = self.run_compose(config)
        self.assert_fail(proc, "E_COMPONENT_PIN")

    def test_component_manifest_hash_mismatch_is_rejected(self):
        config = self.base_config()
        config["streams"][1]["input_manifest_sha256"] = digest(b"tampered")
        proc, _, _ = self.run_compose(config)
        self.assert_fail(proc, "E_COMPONENT_PIN")

    def test_source_revision_mismatch_is_rejected(self):
        config = self.base_config()
        config["streams"][0]["source_revision"] = "wrong-rev"
        proc, _, _ = self.run_compose(config)
        self.assert_fail(proc, "E_COMPONENT_PIN")

    def test_missing_component_for_rank_is_rejected(self):
        tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        b_run = tmp / "b.run"
        shutil.copytree(self.b_run, b_run)
        cfg_path = tmp / "mix-v1.config.json"
        cfg_path.write_bytes(canonical(self.base_config()) + b"\n")
        argv = [
            sys.executable, str(COMPOSER), "--config", str(cfg_path),
            "--component", f"0:{b_run}", "--output-dir", str(tmp / "mix.run"),
        ]
        proc = subprocess.run(argv, capture_output=True, text=True, env=ENV)
        self.assert_fail(proc, "E_ARG")

    def test_overflow_scale_is_rejected(self):
        config = self.base_config()
        config["streams"][1]["scale_num"] = 9007199254740991
        proc, _, _ = self.run_compose(config)
        self.assert_fail(proc, "E_OVERFLOW")

    def test_output_dir_exists_is_rejected(self):
        proc, tmp, runs = self.run_compose(self.base_config(), out_name="mix.run")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        argv = [
            sys.executable, str(COMPOSER), "--config", str(tmp / "mix-v1.config.json"),
            "--component", f"0:{runs['b_run']}", "--component", f"1:{runs['r_run']}",
            "--output-dir", str(tmp / "mix.run"),
        ]
        proc2 = subprocess.run(argv, capture_output=True, text=True, env=ENV)
        self.assert_fail(proc2, "E_EXISTS")

    def test_tampered_component_trace_is_rejected(self):
        config = self.base_config()
        tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        b_run = tmp / "b.run"
        r_run = tmp / "r.run"
        shutil.copytree(self.b_run, b_run)
        shutil.copytree(self.r_run, r_run)
        # mutate a component trace byte-for-byte without updating the committed hash
        trace_path = b_run / "burstgpt-v2.median.jsonl"
        data = trace_path.read_bytes()
        trace_path.write_bytes(data[:-1] + b" \n")
        cfg_path = tmp / "mix-v1.config.json"
        cfg_path.write_bytes(canonical(config) + b"\n")
        argv = [
            sys.executable, str(COMPOSER), "--config", str(cfg_path),
            "--component", f"0:{b_run}", "--component", f"1:{r_run}",
            "--output-dir", str(tmp / "mix.run"),
        ]
        proc = subprocess.run(argv, capture_output=True, text=True, env=ENV)
        self.assert_fail(proc, "E_COMPONENT_PIN")

    def test_imported_execution_is_rejected(self):
        script = (
            "import sys; sys.path.insert(0, %r); import compose_mix; "
            "sys.exit(compose_mix.main(['--config','/dev/null','--output-dir','/dev/null']))"
            % str(ROOT)
        )
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=ENV)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("E_EXECUTION_PROVENANCE", proc.stderr)


if __name__ == "__main__":
    unittest.main()
