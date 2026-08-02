#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
V24 = HERE.parent
PRODUCTION_TESTS = (
    V24 / "production_plan_v1" / "tests" / "test_identity_binding_v1.py"
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evidence = load("test_v24_identity_evidence", V24 / "cp0_r1_evidence_v24.py")
binder_tests = load("test_v24_identity_fixture", PRODUCTION_TESTS)
binding = binder_tests.binding
common = evidence.common


def canonical(value) -> bytes:
    return common.canonical_bytes(value)


class AuthorityFixture:
    def __init__(self, case: unittest.TestCase):
        self.source = binder_tests.Fixture(case)
        self.root = self.source.root
        model_sha256 = binder_tests.sha(b"model")
        self.source.candidate_path.write_bytes(canonical({
            "models": [{
                "artifact": {
                    "bytes": 5,
                    "sha256": model_sha256,
                },
                "model_id": evidence.MODEL_ID,
                "slot": "A",
            }],
            "schema": "s39-cp0-r1-candidate-v1",
        }))
        self.source.history_path.write_bytes(canonical({
            "candidate_sha256": common.sha256_file(
                self.source.candidate_path
            ),
            "model_sha256": model_sha256,
            "schema": "s39-cp0-r1-token-history-v2.4",
        }))
        runtime = self.source.read(
            self.source.prospective_paths["runtime_plan"]
        )
        runtime["candidate_sha256"] = common.sha256_file(
            self.source.candidate_path
        )
        runtime["token_history"]["model_sha256"] = model_sha256
        self.source.prospective_paths["runtime_plan"].write_bytes(
            canonical(runtime)
        )
        joint = self.source.read(
            self.source.prospective_paths["joint_capture_plan"]
        )
        joint["history"] = self.source.record(self.source.history_path)
        self.source.prospective_paths["joint_capture_plan"].write_bytes(
            canonical(joint)
        )
        prospective = self.source.prospective_root
        prospective["candidate"] = self.source.record(
            self.source.candidate_path
        )
        prospective["token_history"] = self.source.record(
            self.source.history_path
        )
        prospective["model_sha256"] = model_sha256
        prospective["artifacts"] = {
            name: self.source.record(path)
            for name, path in sorted(self.source.prospective_paths.items())
        }
        self.source.prospective_root_path.write_bytes(canonical(prospective))
        self.source.phase_lock["runtime_bundle_plan_sha256"] = (
            common.sha256_file(
                self.source.prospective_paths["runtime_plan"]
            )
        )
        self.source.phase_lock_path.write_bytes(
            canonical(self.source.phase_lock)
        )

        self.run_root = self.root / "run"
        self.bound_paths = {
            "cuda_route_launch":
                self.run_root / "bound" / "cuda-route-launch.json",
            "joint_capture_plan":
                self.run_root / "bound" / "joint-capture-plan.json",
            "phone_route_launch":
                self.run_root / "bound" / "phone-route-launch.json",
            "runtime_plan":
                self.run_root / "bound" / "runtime-bundle-plan.json",
        }
        self.receipt_path = (
            self.run_root / "bound" / "identity-binding-receipt.json"
        )
        self.bound_root_path = (
            self.run_root / "bound" / "bound-runtime-root.json"
        )
        self.receipt, self.bound_root = binding.bind(
            prospective_root_path=self.source.prospective_root_path,
            contract_path=self.source.contract_path,
            preparation_path=self.source.preparation_path,
            phase_lock_path=self.source.phase_lock_path,
            prospective_paths=self.source.prospective_paths,
            bound_paths=self.bound_paths,
            receipt_output=self.receipt_path,
            bound_root_output=self.bound_root_path,
            clock_ns=binder_tests.Clock(),
        )
        self.fresh_path = self.source.write(
            "fresh.json",
            {"started_ns": 400},
        )
        self.stage_receipt_path = (
            self.run_root / "receipts" / "identity_binding" / "receipt.json"
        )
        self.stage_receipt_path.parent.mkdir(parents=True)
        self.stage_receipt = {
            "argv": self._stage_argv(),
            "completed_ns": 320,
            "returncode": 0,
            "schema": evidence.ORCHESTRATION_RECEIPT_SCHEMA,
            "stage": "identity_binding",
            "started_ns": 290,
        }
        self.stage_receipt_path.write_bytes(canonical(self.stage_receipt))
        prospective_mechanism = self.source.read(
            self.source.prospective_paths["phone_route_launch"]
        )["mechanism_commands"]
        self.orchestration = {
            "identity_attestation": {
                "bound_root_sha256": common.sha256_file(
                    self.bound_root_path
                ),
                "identity_binding_receipt_sha256": common.sha256_file(
                    self.receipt_path
                ),
                "schema":
                    "s39-cp0-r1-v24-identity-binding-attestation-v1",
                "status": "POST_REBOOT_IDENTITY_BINDING_PASS",
            },
            "inputs": {
                "prospective_root": self.source.record(
                    self.source.prospective_root_path
                ),
            },
            "mechanism_commands_sha256": common.sha256_bytes(
                canonical(prospective_mechanism)
            ),
            "run_root": str(self.run_root),
            "stage_intervals": {
                "identity_binding": {
                    "completed_ns": 320,
                    "started_ns": 290,
                },
            },
        }

    def _stage_argv(self) -> list[str]:
        argv = [
            "/usr/bin/python3",
            "-B",
            "/executed/identity_binding.py",
            "--prospective-root",
            str(self.source.prospective_root_path),
            "--contract",
            str(self.source.contract_path),
            "--preparation",
            str(self.source.preparation_path),
            "--phase-lock",
            str(self.source.phase_lock_path),
        ]
        for name, path in self.source.prospective_paths.items():
            argv.extend([
                f"--prospective-{name.replace('_', '-')}",
                str(path),
            ])
        for name, path in self.bound_paths.items():
            argv.extend([
                f"--bound-{name.replace('_', '-')}",
                str(path),
            ])
        argv.extend([
            "--receipt",
            str(self.receipt_path),
            "--bound-root",
            str(self.bound_root_path),
        ])
        return argv

    def kwargs(self) -> dict:
        return {
            "prospective_root_path": self.source.prospective_root_path,
            "bound_root_path": self.bound_root_path,
            "identity_binding_receipt_path": self.receipt_path,
            "identity_binding_stage_receipt_path": self.stage_receipt_path,
            "contract_path": self.source.contract_path,
            "candidate_path": self.source.candidate_path,
            "runtime_plan_path": self.bound_paths["runtime_plan"],
            "token_history_path": self.source.history_path,
            "tokenizer_plan_path": self.source.tokenizer_path,
            "preparation_path": self.source.preparation_path,
            "phase_lock_path": self.source.phase_lock_path,
            "fresh_path": self.fresh_path,
            "contract": self.source.contract,
            "contract_raw": self.source.contract_path.read_bytes(),
            "candidate_raw": self.source.candidate_path.read_bytes(),
            "orchestration": self.orchestration,
        }

    def republish_bound_artifact(self, name: str, value: dict) -> None:
        self.bound_paths[name].write_bytes(canonical(value))
        self.receipt["outputs"][name] = self.source.record(
            self.bound_paths[name]
        )
        self.receipt_path.write_bytes(canonical(self.receipt))
        self.bound_root["artifacts"] = copy.deepcopy(
            self.receipt["outputs"]
        )
        self.bound_root["identity_binding_receipt"] = self.source.record(
            self.receipt_path
        )
        self.bound_root_path.write_bytes(canonical(self.bound_root))


class IdentityAuthorityTests(unittest.TestCase):
    def test_real_binder_output_passes_final_authority(self):
        fixture = AuthorityFixture(self)
        result = evidence.validate_identity_binding(**fixture.kwargs())
        self.assertEqual(result["status"], "V2_4_IDENTITY_BINDING_PASS")
        self.assertEqual(
            result["mechanism_commands_sha256"],
            fixture.receipt["mechanism_commands_sha256"],
        )

    def test_prospective_root_lineage_mutation_is_rejected(self):
        fixture = AuthorityFixture(self)
        fixture.receipt["prospective_root_sha256"] = "0" * 64
        fixture.receipt_path.write_bytes(canonical(fixture.receipt))
        with self.assertRaisesRegex(
            common.EvidenceError,
            "identity.receipt.prospective_root",
        ):
            evidence.validate_identity_binding(**fixture.kwargs())

    def test_orchestration_prospective_digest_mutation_is_rejected(self):
        fixture = AuthorityFixture(self)
        fixture.orchestration["inputs"]["prospective_root"]["sha256"] = (
            "0" * 64
        )
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_PROSPECTIVE_PLAN_SHA256",
        ):
            evidence.validate_identity_binding(**fixture.kwargs())

    def test_binding_interval_after_fresh_snapshot_is_rejected(self):
        fixture = AuthorityFixture(self)
        fixture.receipt["completed_ns"] = 400
        fixture.receipt_path.write_bytes(canonical(fixture.receipt))
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_IDENTITY_BINDING_ORDER",
        ):
            evidence.validate_identity_binding(**fixture.kwargs())

    def test_realized_mechanism_digest_mutation_is_rejected(self):
        fixture = AuthorityFixture(self)
        fixture.receipt["mechanism_commands_sha256"] = "0" * 64
        fixture.receipt_path.write_bytes(canonical(fixture.receipt))
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_IDENTITY_REALIZED_MECHANISM",
        ):
            evidence.validate_identity_binding(**fixture.kwargs())

    def test_embedded_sentinel_in_bound_output_is_rejected(self):
        fixture = AuthorityFixture(self)
        phone = fixture.source.read(
            fixture.bound_paths["phone_route_launch"]
        )
        phone["codec"]["argv"].append("tcp://0.0.0.15:9000")
        fixture.republish_bound_artifact("phone_route_launch", phone)
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_SENTINEL_LEAKAGE",
        ):
            evidence.validate_identity_binding(**fixture.kwargs())

    def test_bound_joint_launch_path_mutation_is_rejected(self):
        fixture = AuthorityFixture(self)
        joint = fixture.source.read(
            fixture.bound_paths["joint_capture_plan"]
        )
        command = joint["commands"]["phone"]
        command["argv_template"][command["launch_plan_argv_index"]] = (
            str(fixture.bound_paths["cuda_route_launch"])
        )
        fixture.republish_bound_artifact("joint_capture_plan", joint)
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_BOUND_JOINT_PATH",
        ):
            evidence.validate_identity_binding(**fixture.kwargs())

    def test_bound_inline_plan_digest_mutation_is_rejected(self):
        fixture = AuthorityFixture(self)
        phone = fixture.source.read(
            fixture.bound_paths["phone_route_launch"]
        )
        argv = phone["processes"]["op15_direct_relay"]["argv"]
        digest_index = argv.index("--plan-sha256") + 1
        argv[digest_index] = "0" * 64
        phone["mechanism_commands"] = evidence._phone_mechanism_matrix(phone)
        fixture.republish_bound_artifact("phone_route_launch", phone)
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_BOUND_CUDA_MECHANISM|plan_sha256",
        ):
            evidence.validate_identity_binding(**fixture.kwargs())

    def test_bound_root_artifact_projection_mutation_is_rejected(self):
        fixture = AuthorityFixture(self)
        fixture.bound_root["artifacts"]["cuda_route_launch"]["sha256"] = (
            "0" * 64
        )
        fixture.bound_root_path.write_bytes(canonical(fixture.bound_root))
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_IDENTITY_ATTESTED_ROOT",
        ):
            evidence.validate_identity_binding(**fixture.kwargs())

    def test_bound_root_receipt_lineage_mutation_is_rejected(self):
        fixture = AuthorityFixture(self)
        fixture.bound_root["identity_binding_receipt"]["sha256"] = "0" * 64
        fixture.bound_root_path.write_bytes(canonical(fixture.bound_root))
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_IDENTITY_ATTESTED_ROOT",
        ):
            evidence.validate_identity_binding(**fixture.kwargs())

    def test_binder_stdout_receipt_attestation_mutation_is_rejected(self):
        fixture = AuthorityFixture(self)
        fixture.orchestration["identity_attestation"][
            "identity_binding_receipt_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_IDENTITY_ATTESTED_RECEIPT",
        ):
            evidence.validate_identity_binding(**fixture.kwargs())

    def test_binder_stdout_root_attestation_mutation_is_rejected(self):
        fixture = AuthorityFixture(self)
        fixture.orchestration["identity_attestation"][
            "bound_root_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_IDENTITY_ATTESTED_ROOT",
        ):
            evidence.validate_identity_binding(**fixture.kwargs())

    def test_identity_stage_bound_path_mutation_is_rejected(self):
        fixture = AuthorityFixture(self)
        option = "--bound-phone-route-launch"
        index = fixture.stage_receipt["argv"].index(option) + 1
        fixture.stage_receipt["argv"][index] = str(
            fixture.bound_paths["cuda_route_launch"]
        )
        fixture.stage_receipt_path.write_bytes(
            canonical(fixture.stage_receipt)
        )
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_IDENTITY_STAGE_BOUND: phone_route_launch",
        ):
            evidence.validate_identity_binding(**fixture.kwargs())

    def test_bound_phone_network_mutation_is_rejected(self):
        fixture = AuthorityFixture(self)
        phone = fixture.source.read(
            fixture.bound_paths["phone_route_launch"]
        )
        phone["phones"]["op12"]["local_ipv4"] = "192.0.2.99"
        fixture.republish_bound_artifact("phone_route_launch", phone)
        with self.assertRaisesRegex(
            common.EvidenceError,
            "E_BOUND_PHONE_NETWORK",
        ):
            evidence.validate_identity_binding(**fixture.kwargs())

    def test_downstream_execution_binds_both_subproducer_plans(self):
        fixture = AuthorityFixture(self)
        identity = evidence.validate_identity_binding(**fixture.kwargs())
        cuda_receipt = {
            "mechanism_commands_sha256":
                identity["mechanism_commands_sha256"],
        }
        joint_receipt = {
            "capture_plan_sha256":
                identity["artifact_sha256s"]["joint_capture_plan"],
            "mechanism_commands_sha256":
                identity["mechanism_commands_sha256"],
            "subproducer_bindings": {
                "cuda": {
                    "launch_plan_sha256":
                        identity["artifact_sha256s"]["cuda_route_launch"],
                },
                "phone": {
                    "launch_plan_sha256":
                        identity["artifact_sha256s"]["phone_route_launch"],
                },
            },
        }
        evidence.validate_bound_execution_bindings(
            cuda_receipt=cuda_receipt,
            joint_receipt=joint_receipt,
            identity=identity,
        )
        for name in ("cuda", "phone"):
            mutated = copy.deepcopy(joint_receipt)
            mutated["subproducer_bindings"][name][
                "launch_plan_sha256"
            ] = "0" * 64
            with self.subTest(name=name), self.assertRaisesRegex(
                common.EvidenceError,
                f"E_JOINT_BOUND_SUBPRODUCER: {name}",
            ):
                evidence.validate_bound_execution_bindings(
                    cuda_receipt=cuda_receipt,
                    joint_receipt=mutated,
                    identity=identity,
                )

    def test_legacy_v1_orchestration_plan_is_rejected(self):
        fixture = AuthorityFixture(self)
        legacy = {
            "inputs": {},
            "mechanism_commands_sha256": "0" * 64,
            "model_id": evidence.MODEL_ID,
            "model_sha256": "0" * 64,
            "phase": evidence.PHASE,
            "phase_id": "cp0-r1-v24-a-only-legacy",
            "python": {},
            "run_root": str(fixture.run_root),
            "schema": "s39-cp0-r1-v24-a-only-orchestration-plan-v1",
            "stages": {},
        }
        legacy_path = fixture.source.write("legacy-plan.json", legacy)
        with self.assertRaisesRegex(
            common.EvidenceError,
            "orchestration.plan.schema",
        ):
            evidence.validate_orchestration_provenance(
                orchestration_plan_path=legacy_path,
                bundle_root=fixture.run_root / "raw-bundle",
                contract_path=fixture.source.contract_path,
                candidate_path=fixture.source.candidate_path,
                contract=fixture.source.contract,
                contract_raw=fixture.source.contract_path.read_bytes(),
                candidate_raw=fixture.source.candidate_path.read_bytes(),
            )

    def test_cuda_bound_schema_uses_current_v24_schema(self):
        self.assertEqual(
            evidence.BOUND_ARTIFACT_SCHEMAS["cuda_route_launch"],
            "s39-cp0-r1-v24-cuda-route-launch-v1",
        )
        self.assertNotEqual(
            evidence.BOUND_ARTIFACT_SCHEMAS["cuda_route_launch"],
            "s39-cp0-r1-a-only-cuda-route-launch-v1",
        )


if __name__ == "__main__":
    unittest.main()
