#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
EXECUTORS = HERE.parent
S40 = EXECUTORS.parent
S39 = S40.parent / "s39_phone_model_switch_trace"
V23_TESTS = S39 / "v23_readiness" / "tests"
for path in (EXECUTORS, S39, S39 / "tests", S39 / "v23_readiness", V23_TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from executor_bundle import build_executor_bundle
from phone_gateway import GatewayError
import qualification_authority as authority
import cp0_r1_evidence_v22 as v22
from test_cp0_r1_evidence_v22 import V22PhaseFixture
from test_v23_readiness import ReadinessFixture


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


class QualificationAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.bundle = self.root / "executor-bundle"
        self.bundle_result = build_executor_bundle(self.bundle)

    def qualification_root(self, fixture, name):
        root = self.root / name
        shutil.copytree(fixture.phase.root, root)
        records = {}
        for key, source in fixture.paths_out.items():
            path = root / "v23" / f"{key}.json"
            path.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, path)
            records[key] = {
                "path": str(path),
                "sha256": digest(path),
            }
        return {
            "artifact_snapshot": records["artifact"],
            "bundle_manifest_sha256": digest(root / "EVIDENCE_BUNDLE.json"),
            "bundle_root": str(root),
            "fresh_snapshot": records["fresh"],
            "readiness_lock": records["lock"],
            "runtime_identity": records["runtime"],
        }

    def authority(self, fixture, name="current"):
        current = self.qualification_root(fixture, name)
        return {
            "a_chain": None,
            "current": current,
            "model_id": fixture.model["model_id"],
            "phase": "A_ONLY",
            "schema": authority.AUTHORITY_SCHEMA,
            "slot": "A",
        }

    def b_fixture(self, a_fixture):
        a_result = v22.evaluate_root(
            a_fixture.phase.root,
            v22.MANIFEST_NAME,
            a_fixture.v22_contract,
            a_fixture.v22_contract_raw,
            a_fixture.candidate,
            a_fixture.candidate_raw,
            a_fixture.parent,
            a_fixture.corpus,
            [],
        )
        phase = V22PhaseFixture(
            self.root / "b-phase",
            "B_ONLY",
            a_fixture.v22_contract,
            a_fixture.v22_contract_raw,
            a_fixture.parent,
            a_fixture.candidate,
            a_fixture.candidate_raw,
            [a_result],
            [a_fixture.route],
            frozen_corpus=a_fixture.corpus,
        )
        fixture = ReadinessFixture(self)
        fixture.phase = phase
        fixture.manifest = json.loads(
            (phase.root / v22.MANIFEST_NAME).read_bytes()
        )
        fixture.model = fixture.candidate["models"][1]
        fixture.route = phase.lock
        fixture.paths = {
            "cuda": fixture.route["cuda_model_path"],
            "op15": fixture.route["op15_shard_path"],
            "op12": fixture.route["op12_shard_path"],
            "op15_worker": "/data/local/tmp/s39-v23/llama-layersplit",
            "op12_worker": "/data/local/tmp/s39-v23/llama-layersplit",
        }
        fixture.worker_digests = {
            "op15_worker": "3" * 64,
            "op12_worker": "4" * 64,
        }
        fixture.stats = {
            endpoint: {
                "ctime_ns": 200_000_000_000 + index,
                "device_id": index + 11,
                "inode": 2000 + index,
                "mode": 33188,
                "mtime_ns": 190_000_000_000 + index,
                "size": (
                    fixture.model["artifact"]["bytes"]
                    if endpoint == "cuda"
                    else 50_000_000
                    if endpoint.endswith("_worker")
                    else fixture.route[f"{endpoint}_shard_bytes"]
                ),
            }
            for index, endpoint in enumerate(
                ("cuda", "op15", "op12", "op15_worker", "op12_worker")
            )
        }
        transfer_role = (
            f"model.{fixture.model['model_id']}.route_transfer"
        )
        fixture.direct_payload_bytes = sum(
            row["payload_bytes"]
            for row in phase.rows[transfer_role]
            if row["kind"] == "transfer"
        )
        fixture.artifact = fixture.make_artifact()
        fixture.artifact["phase"] = "B_ONLY"
        fixture.artifact["slot"] = "B"
        fixture.artifact_raw = authority.canonical_bytes(fixture.artifact)
        phase_lock_sha = next(
            item["sha256"]
            for item in fixture.manifest["artifacts"]
            if item["role"] == "phase.lock"
        )
        fixture.lock = {
            "artifact_snapshot_sha256":
                hashlib.sha256(fixture.artifact_raw).hexdigest(),
            "event_ns": phase.opened_ns + 400,
            "phase": "B_ONLY",
            "phase_id": phase.phase_id,
            "schema": "s39-cp0-r1-readiness-lock-v2.3",
            "v2_2_phase_lock_sha256": phase_lock_sha,
        }
        fixture.lock_raw = authority.canonical_bytes(fixture.lock)
        fixture.fresh = fixture.make_fresh()
        fixture.fresh["phase"] = "B_ONLY"
        fixture.fresh["phase_id"] = phase.phase_id
        fixture.fresh_raw = authority.canonical_bytes(fixture.fresh)
        fixture.runtime = fixture.make_runtime()
        fixture.runtime["phase"] = "B_ONLY"
        fixture.runtime["phase_id"] = phase.phase_id
        fixture.write()
        return fixture

    def validate(self, value, fixture):
        return authority.validate_route_authority(
            value,
            expected_model_id=fixture.model["model_id"],
            expected_phase="A_ONLY",
            expected_slot="A",
            expected_artifact_certificate_sha256=
                value["current"]["artifact_snapshot"]["sha256"],
            expected_readiness_lock_sha256=
                value["current"]["readiness_lock"]["sha256"],
            executor_bundle=self.bundle,
            executor_bundle_manifest_sha256=
                self.bundle_result["manifest_sha256"],
        )

    def test_full_v23_and_v22_raw_roots_are_re_evaluated(self):
        fixture = ReadinessFixture(self)
        value = self.authority(fixture)
        result = self.validate(value, fixture)
        self.assertEqual(result["scope"], "QUALIFIED_ROUTE")
        self.assertEqual(result["status"], "MODEL_A_QUALIFICATION_PASS")
        self.assertEqual(result["phase_id"], fixture.phase.phase_id)
        self.assertEqual(
            result["phase_lock_sha256"],
            json.loads(
                fixture.paths_out["lock"].read_bytes()
            )["v2_2_phase_lock_sha256"],
        )
        self.assertRegex(result["v2_2_result_sha256"], r"^[0-9a-f]{64}$")

    def test_b_only_re_evaluates_full_a_chain_before_current_root(self):
        a_fixture = ReadinessFixture(self)
        b_fixture = self.b_fixture(a_fixture)
        a_root = self.qualification_root(a_fixture, "valid-a-chain")
        current = self.qualification_root(b_fixture, "valid-b-current")
        value = {
            "a_chain": a_root,
            "current": current,
            "model_id": b_fixture.model["model_id"],
            "phase": "B_ONLY",
            "schema": authority.AUTHORITY_SCHEMA,
            "slot": "B",
        }
        result = authority.validate_route_authority(
            value,
            expected_model_id=b_fixture.model["model_id"],
            expected_phase="B_ONLY",
            expected_slot="B",
            expected_artifact_certificate_sha256=
                current["artifact_snapshot"]["sha256"],
            expected_readiness_lock_sha256=
                current["readiness_lock"]["sha256"],
            executor_bundle=self.bundle,
            executor_bundle_manifest_sha256=
                self.bundle_result["manifest_sha256"],
        )
        self.assertEqual(result["status"], "MODEL_B_QUALIFICATION_PASS")
        self.assertEqual(result["phase_id"], b_fixture.phase.phase_id)
        self.assertEqual(
            result["a_chain_phase_id"],
            a_fixture.phase.phase_id,
        )

    def test_self_consistent_summary_without_raw_root_is_rejected(self):
        fixture = ReadinessFixture(self)
        value = self.authority(fixture)
        fake = self.root / "fake"
        fake.mkdir()
        value["current"]["bundle_root"] = str(fake)
        value["current"]["bundle_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(GatewayError, "manifest"):
            self.validate(value, fixture)

    def test_missing_or_mutated_raw_root_is_rejected(self):
        for mutation in ("missing", "mutated"):
            with self.subTest(mutation=mutation):
                fixture = ReadinessFixture(self)
                value = self.authority(fixture, f"raw-{mutation}")
                manifest = Path(value["current"]["bundle_root"]) / "EVIDENCE_BUNDLE.json"
                if mutation == "missing":
                    manifest.unlink()
                else:
                    role = (
                        f"model.{fixture.model['model_id']}."
                        "mechanics.phone"
                    )
                    row = next(
                        item
                        for item in fixture.manifest["artifacts"]
                        if item["role"] == role
                    )
                    target = Path(value["current"]["bundle_root"]) / row["path"]
                    target.write_bytes(target.read_bytes() + b" ")
                with self.assertRaisesRegex(GatewayError, "manifest|changed"):
                    self.validate(value, fixture)

    def test_mutated_frozen_evaluator_dependency_is_rejected(self):
        fixture = ReadinessFixture(self)
        value = self.authority(fixture)
        evaluator = self.bundle / authority.EVALUATOR_RELATIVE
        evaluator.write_bytes(evaluator.read_bytes() + b"\n")
        with self.assertRaisesRegex(
            Exception,
            "bundle file changed",
        ):
            self.validate(value, fixture)

    def test_fake_readiness_lock_is_rejected_even_when_rehashed(self):
        fixture = ReadinessFixture(self)
        value = self.authority(fixture)
        lock = Path(value["current"]["readiness_lock"]["path"])
        raw = lock.read_bytes().replace(b'"phase":"A_ONLY"', b'"phase":"B_ONLY"')
        lock.write_bytes(raw)
        value["current"]["readiness_lock"]["sha256"] = digest(lock)
        with self.assertRaisesRegex(GatewayError, "artifact certificate|evaluator"):
            self.validate(value, fixture)

    def test_wrong_phase_and_legacy_status_only_result_are_rejected(self):
        fixture = ReadinessFixture(self)
        for mutation in ("phase", "legacy"):
            with self.subTest(mutation=mutation):
                value = self.authority(fixture, f"wrong-{mutation}")
                if mutation == "phase":
                    value["phase"] = "B_ONLY"
                else:
                    value["status"] = "MODEL_A_QUALIFICATION_PASS"
                    value["result_path"] = "/tmp/status-only.json"
                with self.assertRaisesRegex(
                    GatewayError,
                    "fields do not match schema|phase",
                ):
                    self.validate(value, fixture)

    def test_stale_a_chain_is_checked_before_b_evaluation(self):
        a_fixture = ReadinessFixture(self)
        current_fixture = ReadinessFixture(self)
        a_root = self.qualification_root(a_fixture, "a-chain")
        current = self.qualification_root(current_fixture, "b-current")
        manifest = Path(a_root["bundle_root"]) / "EVIDENCE_BUNDLE.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")
        value = {
            "a_chain": a_root,
            "current": current,
            "model_id": "qwen3-8b-q8_0",
            "phase": "B_ONLY",
            "schema": authority.AUTHORITY_SCHEMA,
            "slot": "B",
        }
        with self.assertRaisesRegex(GatewayError, "A chain.*manifest|manifest"):
            authority.validate_route_authority(
                value,
                expected_model_id="qwen3-8b-q8_0",
                expected_phase="B_ONLY",
                expected_slot="B",
                expected_artifact_certificate_sha256=
                    current["artifact_snapshot"]["sha256"],
                expected_readiness_lock_sha256=
                    current["readiness_lock"]["sha256"],
                executor_bundle=self.bundle,
                executor_bundle_manifest_sha256=
                    self.bundle_result["manifest_sha256"],
            )

    def test_a_chain_mutation_during_b_evaluation_is_rejected(self):
        a_fixture = ReadinessFixture(self)
        b_fixture = self.b_fixture(a_fixture)
        a_root = self.qualification_root(a_fixture, "changing-a-chain")
        current = self.qualification_root(b_fixture, "changing-b-current")
        value = {
            "a_chain": a_root,
            "current": current,
            "model_id": b_fixture.model["model_id"],
            "phase": "B_ONLY",
            "schema": authority.AUTHORITY_SCHEMA,
            "slot": "B",
        }
        real_run = subprocess.run
        calls = 0

        def run_and_mutate(*args, **kwargs):
            nonlocal calls
            result = real_run(*args, **kwargs)
            calls += 1
            if calls == 2:
                path = Path(a_root["runtime_identity"]["path"])
                path.write_bytes(path.read_bytes() + b" ")
            return result

        with (
            mock.patch.object(
                authority.subprocess,
                "run",
                side_effect=run_and_mutate,
            ),
            self.assertRaisesRegex(
                GatewayError,
                "qualification input changed|A chain changed during evaluation",
            ),
        ):
            authority.validate_route_authority(
                value,
                expected_model_id=b_fixture.model["model_id"],
                expected_phase="B_ONLY",
                expected_slot="B",
                expected_artifact_certificate_sha256=
                    current["artifact_snapshot"]["sha256"],
                expected_readiness_lock_sha256=
                    current["readiness_lock"]["sha256"],
                executor_bundle=self.bundle,
                executor_bundle_manifest_sha256=
                    self.bundle_result["manifest_sha256"],
            )

    def test_wrong_evaluator_status_is_rejected(self):
        fixture = ReadinessFixture(self)
        root = self.qualification_root(fixture, "wrong-status")
        raw = authority.canonical_bytes({
            "derived": {},
            "schema": "s39-cp0-r1-readiness-result-v2.3",
            "status": "MODEL_A_QUALIFICATION_PASS",
        })
        with self.assertRaisesRegex(GatewayError, "evaluator status"):
            authority._parse_evaluator_output(
                raw,
                authority._validate_root(root, "root"),
                "A_ONLY",
                fixture.model["model_id"],
            )

    def test_desktop_smoke_scope_cannot_authorize_a_route(self):
        model = self.root / "model.gguf"
        with model.open("xb") as output:
            output.truncate(8709518112)
        model_sha = (
            "408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6"
        )
        smoke = {
            "gpu_uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
            "model_bytes": model.stat().st_size,
            "model_id": "qwen3-8b-q8_0",
            "model_path": str(model),
            "model_sha256": model_sha,
            "phase": "DESKTOP_SMOKE",
            "phase_id": "smoke-b-1",
            "schema": authority.DESKTOP_SMOKE_SCHEMA,
            "scope": authority.DESKTOP_SMOKE_SCOPE,
        }
        with mock.patch.object(
            authority,
            "file_sha256",
            return_value=model_sha,
        ):
            result = authority.validate_desktop_smoke_authority(
                smoke,
                expected_model_id="qwen3-8b-q8_0",
                expected_model_path=str(model),
                executor_bundle=self.bundle,
                executor_bundle_manifest_sha256=
                    self.bundle_result["manifest_sha256"],
            )
        self.assertEqual(result["scope"], "DESKTOP_SMOKE_ONLY")
        with self.assertRaisesRegex(GatewayError, "key set|schema"):
            authority.validate_route_authority(
                smoke,
                expected_model_id="qwen3-8b-q8_0",
                expected_phase="B_ONLY",
                expected_slot="B",
                expected_artifact_certificate_sha256="0" * 64,
                expected_readiness_lock_sha256="1" * 64,
                executor_bundle=self.bundle,
                executor_bundle_manifest_sha256=
                    self.bundle_result["manifest_sha256"],
            )

    def test_coordinated_smoke_config_and_authority_mutation_is_rejected(self):
        model = self.root / "other.gguf"
        model.write_bytes(b"other")
        smoke = {
            "gpu_uuid": "GPU-attacker",
            "model_bytes": model.stat().st_size,
            "model_id": "attacker-model",
            "model_path": str(model),
            "model_sha256": digest(model),
            "phase": "DESKTOP_SMOKE",
            "phase_id": "smoke-attacker-1",
            "schema": authority.DESKTOP_SMOKE_SCHEMA,
            "scope": authority.DESKTOP_SMOKE_SCOPE,
        }
        with self.assertRaisesRegex(
            GatewayError,
            "outside experiment contract",
        ):
            authority.validate_desktop_smoke_authority(
                smoke,
                expected_model_id="attacker-model",
                expected_model_path=str(model),
                executor_bundle=self.bundle,
                executor_bundle_manifest_sha256=
                    self.bundle_result["manifest_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
