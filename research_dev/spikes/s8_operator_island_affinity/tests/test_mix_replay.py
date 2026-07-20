#!/usr/bin/python3
"""Fail-closed tests for the S8 mixed (semi_synthetic) structural replay.

Requires jsonschema 4.10.3, so run under /usr/bin/python3 (not the venv):
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m unittest tests.test_mix_replay
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
from contextlib import contextmanager


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import structural_replay as replayer  # noqa: E402

CONFIGS = ROOT / "configs"
DAGS = ROOT / "service_dags"
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
                "timestamp": ts, "input_length": il, "output_length": ol, "session_id": sid,
                "hash_ids": {
                    "sys_prompt": [1], "passages_ids": pas, "history": [9],
                    "web_search": [], "user_input": [7],
                },
            },
            separators=(",", ":"), ensure_ascii=True,
        )
    lines = [row("0", 8, 4, "r0", [11, 12]), row("1", 6, 2, "r1", []), row("2", 5, 5, "r2", [13])]
    return ("\n".join(lines) + "\n\n").encode("ascii")


def normalize_component(base, name, cfg_name, source_bytes, line_count, record_count):
    src = base / name
    src.write_bytes(source_bytes)
    cfg = json.loads((CONFIGS / cfg_name).read_text(encoding="ascii"))
    cfg["origin"].update(
        {
            "bytes": len(source_bytes), "sha256": digest(source_bytes),
            "line_count": line_count, "record_count": record_count,
        }
    )
    cfg_path = base / cfg_name
    cfg_path.write_bytes(canonical(cfg) + b"\n")
    run_dir = base / (name + ".run")
    proc = subprocess.run(
        [
            sys.executable, str(NORMALIZER), "--source-file", str(src), "--config", str(cfg_path),
            "--scenario", "median", "--output-dir", str(run_dir),
        ],
        capture_output=True, text=True, env=ENV,
    )
    if proc.returncode != 0:
        raise AssertionError("normalizer failed: " + proc.stderr)
    return run_dir


class MixReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(cls._tmp.name)
        cls.b_run = normalize_component(base, "BurstGPT_3.csv", "burstgpt.config.json", burst_source(), 4, 3)
        cls.r_run = normalize_component(base, "0_trace.jsonl", "ragpulse.config.json", rag_source(), 4, 3)
        b_trace = (cls.b_run / "burstgpt-v2.median.jsonl").read_bytes()
        b_side = json.loads((cls.b_run / "burstgpt-v2.median.manifest.json").read_bytes())
        r_trace = (cls.r_run / "ragpulse.median.jsonl").read_bytes()
        r_side = json.loads((cls.r_run / "ragpulse.median.manifest.json").read_bytes())
        cls.mix_config = {
            "schema_version": 1, "provenance": "semi_synthetic", "mix_id": "mix-v1",
            "streams": [
                {
                    "rank": 0, "source": "burstgpt-v2", "source_revision": b_side["source_revision"],
                    "component_trace": "burstgpt-v2.median.jsonl", "scale_num": 1, "scale_den": 1, "offset_us": 0,
                    "input_output_sha256": digest(b_trace), "input_manifest_sha256": digest(canonical(b_side)),
                },
                {
                    "rank": 1, "source": "ragpulse", "source_revision": r_side["source_revision"],
                    "component_trace": "ragpulse.median.jsonl", "scale_num": 1, "scale_den": 1, "offset_us": 500,
                    "input_output_sha256": digest(r_trace), "input_manifest_sha256": digest(canonical(r_side)),
                },
            ],
        }
        cls.mix_config_path = base / "mix-v1.config.json"
        cls.mix_config_path.write_bytes(canonical(cls.mix_config) + b"\n")
        cls.mix_run = base / "mix.run"
        proc = subprocess.run(
            [
                sys.executable, str(COMPOSER), "--config", str(cls.mix_config_path),
                "--component", f"0:{cls.b_run}", "--component", f"1:{cls.r_run}",
                "--output-dir", str(cls.mix_run),
            ],
            capture_output=True, text=True, env=ENV,
        )
        if proc.returncode != 0:
            raise AssertionError("compose failed: " + proc.stderr)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @contextmanager
    def case(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            mix_run = root / "mix.run"
            b_run = root / "b.run"
            r_run = root / "r.run"
            shutil.copytree(self.mix_run, mix_run)
            shutil.copytree(self.b_run, b_run)
            shutil.copytree(self.r_run, r_run)
            config_path = root / "mix-v1.config.json"
            config_path.write_bytes(canonical(self.mix_config) + b"\n")
            yield {
                "mix_run": mix_run, "config": config_path,
                "components": {0: b_run, 1: r_run},
            }

    def replay(self, case):
        return replayer.replay_mix(case["mix_run"], case["config"], case["components"], DAGS)

    def assert_error(self, case, code):
        with self.assertRaises(replayer.ReplayError) as ctx:
            self.replay(case)
        self.assertEqual(ctx.exception.code, code, ctx.exception.message)

    def reseal(self, case):
        """Recompute the full hash chain so structural checks fire, not E_TRACE_HASH."""
        run = case["mix_run"]
        trace = (run / "mix-v1.jsonl").read_bytes()
        sidecar = json.loads((run / "mix-v1.manifest.json").read_bytes())
        artifact = json.loads((run / "normalize.artifact.json").read_bytes())
        out_sha = digest(trace)
        sidecar["output_sha256"] = out_sha
        sidecar["output_row_count"] = trace.count(b"\n")
        sidecar_bytes = canonical(sidecar)
        artifact["outputs"][0]["sha256"] = out_sha
        artifact["outputs"][0]["sidecar_manifest_sha256"] = digest(sidecar_bytes)
        artifact["deterministic_replay_sha256"] = digest(out_sha.encode("ascii"))
        run_preimage = canonical({
            "mix_id": self.mix_config["mix_id"], "config_hash": artifact["config_hash"],
            "code_version": artifact["code_version"], "output_sha256": out_sha,
        })
        artifact["run_id"] = "mix-" + hashlib.sha256(run_preimage).hexdigest()[:24]
        (run / "mix-v1.manifest.json").write_bytes(sidecar_bytes + b"\n")
        (run / "normalize.artifact.json").write_bytes(canonical(artifact) + b"\n")

    def mutate_records(self, case, fn):
        run = case["mix_run"]
        lines = [json.loads(x) for x in (run / "mix-v1.jsonl").read_text().splitlines()]
        fn(lines)
        (run / "mix-v1.jsonl").write_bytes(b"".join(canonical(r) + b"\n" for r in lines))
        self.reseal(case)

    # --- happy path -------------------------------------------------------
    def test_replay_passes_and_accounts_demand(self):
        with self.case() as case:
            result = self.replay(case)
        self.assertEqual(result["kind"], "structural_replay_mix")
        self.assertEqual(result["record_count"], 6)
        self.assertEqual(result["total_demand"]["input_tokens"], 37)
        self.assertEqual(result["total_demand"]["output_tokens"], 18)
        self.assertEqual(result["total_demand"]["retrieved_chunks"], 3)
        counts = {s["service"]: s["request_count"] for s in result["services"]}
        self.assertEqual(counts, {"api_generation": 1, "conversation_generation": 2, "rag_qa": 3})
        ranks = {b["rank"]: b["record_count"] for b in result["bindings"]["streams"]}
        self.assertEqual(ranks, {0: 3, 1: 3})
        self.assertFalse(result["bindings"]["recompose_verified"])

    # --- hash chain fail-closed -------------------------------------------
    def test_trace_hash_drift_is_rejected(self):
        with self.case() as case:
            path = case["mix_run"] / "mix-v1.jsonl"
            data = path.read_bytes()
            path.write_bytes(data[:-1] + b" \n")  # drift bytes, do NOT reseal
            self.assert_error(case, "E_TRACE_HASH")

    def test_component_tamper_is_rejected(self):
        with self.case() as case:
            comp = case["components"][0] / "burstgpt-v2.median.jsonl"
            data = comp.read_bytes()
            comp.write_bytes(data[:-1] + b" \n")
            self.assert_error(case, "E_COMPONENT_PIN")

    def test_missing_component_is_rejected(self):
        with self.case() as case:
            case["components"].pop(1)
            self.assert_error(case, "E_COMPONENT_MISSING")

    # --- structural fail-closed (with reseal) -----------------------------
    def test_disordered_trace_is_rejected(self):
        with self.case() as case:
            self.mutate_records(case, lambda rows: rows.reverse())
            self.assert_error(case, "E_ORDER")

    def test_wrong_record_provenance_is_rejected(self):
        with self.case() as case:
            def fn(rows):
                rows[0]["provenance"] = "real"
            self.mutate_records(case, fn)
            self.assert_error(case, "E_SOURCE_PIN")

    def test_unknown_service_is_rejected(self):
        with self.case() as case:
            def fn(rows):
                rows[0]["service"] = "no_such_service"
            self.mutate_records(case, fn)
            self.assert_error(case, "E_SERVICE_UNKNOWN")

    def test_source_not_matching_stream_is_rejected(self):
        with self.case() as case:
            def fn(rows):
                rows[0]["source"] = "ragpulse"  # rank 0 is burstgpt-v2
            self.mutate_records(case, fn)
            self.assert_error(case, "E_SOURCE_PIN")

    def test_sidecar_stream_tamper_is_rejected(self):
        with self.case() as case:
            run = case["mix_run"]
            sidecar = json.loads((run / "mix-v1.manifest.json").read_bytes())
            sidecar["streams"][1]["offset_us"] = 999  # differs from committed config
            sidecar_bytes = canonical(sidecar)
            artifact = json.loads((run / "normalize.artifact.json").read_bytes())
            artifact["outputs"][0]["sidecar_manifest_sha256"] = digest(sidecar_bytes)
            (run / "mix-v1.manifest.json").write_bytes(sidecar_bytes + b"\n")
            (run / "normalize.artifact.json").write_bytes(canonical(artifact) + b"\n")
            self.assert_error(case, "E_MANIFEST_SEMANTICS")

    def test_artifact_run_id_tamper_is_rejected(self):
        with self.case() as case:
            run = case["mix_run"]
            artifact = json.loads((run / "normalize.artifact.json").read_bytes())
            artifact["run_id"] = "mix-deadbeefdeadbeefdeadbeef"
            (run / "normalize.artifact.json").write_bytes(canonical(artifact) + b"\n")
            self.assert_error(case, "E_ARTIFACT_BINDING")

    def test_config_drift_is_rejected(self):
        with self.case() as case:
            # change the committed config without recomposing -> config hash drift
            config = json.loads(case["config"].read_bytes())
            config["streams"][1]["offset_us"] = 12345
            case["config"].write_bytes(canonical(config) + b"\n")
            self.assert_error(case, "E_CONFIG_BINDING")


if __name__ == "__main__":
    unittest.main()
