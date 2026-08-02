#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SUBJECT_ROOT = HERE.parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


binding = load("test_identity_binding", SUBJECT_ROOT / "identity_binding_v1.py")
common = binding.common


def canonical(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class Clock:
    def __init__(self):
        self.value = 290

    def __call__(self):
        self.value += 10
        return self.value


class Fixture:
    def __init__(self, case: unittest.TestCase):
        temporary = tempfile.TemporaryDirectory()
        case.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.contract = {
            "devices": {
                "cuda": {
                    "host": "zhihao-Z690-C-ac",
                    "uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
                },
            },
            "schema": "s39-cp0-r1-evidence-contract-v2.4",
        }
        self.contract_path = self.write("contract.json", self.contract)
        self.candidate_path = self.write(
            "candidate.json",
            {"schema": "s39-cp0-r1-candidate-v1"},
        )
        self.history_path = self.write(
            "history.json",
            {"schema": "s39-cp0-r1-token-history-v2.4"},
        )
        self.tokenizer_path = self.write(
            "tokenizer.json",
            {"schema": "s39-cp0-r1-a-only-tokenizer-plan-v2"},
        )
        self.mono_path = self.write(
            "mono.json",
            {"schema": "s39-cp0-r1-v24-cuda-monolithic-launch-v1"},
        )
        self.spec_path = self.write(
            "spec.json",
            {"schema": "s39-cp0-r1-v24-prospective-runtime-spec-v1"},
        )
        unbound12 = common.UNBOUND_PHONE_NETWORK["op12"]["local_ipv4"]
        unbound15 = common.UNBOUND_PHONE_NETWORK["op15"]["local_ipv4"]
        interface = common.UNBOUND_PHONE_NETWORK["op12"]["interface"]
        processes = {
            "op12_stagenet": {
                "argv": ["/bin/echo", "--interface", interface, "--peer", unbound15],
            },
            "op15_direct_relay": {
                "argv": self.inline_argv({
                    "route": {
                        "head_host": unbound12,
                        "listen_host": unbound15,
                        "runtime_argv": [
                            "/bin/echo",
                            "--head",
                            f"{unbound12}:9000",
                        ],
                    },
                    "schema": "s39-managed-runtime-launch-plan-v1",
                }),
            },
            "op15_stagenet": {
                "argv": ["/bin/echo", "--interface", interface, "--peer", unbound12],
            },
        }
        probes = {
            "op12": {
                "after_argv": ["/bin/echo", interface, unbound12, unbound15],
                "before_argv": ["/bin/echo", interface, unbound12, unbound15],
            },
            "op15": {
                "after_argv": ["/bin/echo", interface, unbound15, unbound12],
                "before_argv": ["/bin/echo", interface, unbound15, unbound12],
            },
        }
        mechanism = {
            "desktop": [
                ["/bin/echo", f"desktop-{index}"]
                for index in range(9)
            ],
            "op12": [
                processes["op12_stagenet"]["argv"],
                probes["op12"]["before_argv"],
                probes["op12"]["after_argv"],
            ],
            "op15": [
                processes["op15_stagenet"]["argv"],
                processes["op15_direct_relay"]["argv"],
                probes["op15"]["before_argv"],
                probes["op15"]["after_argv"],
            ],
        }
        self.cuda = {
            "mechanism_commands": mechanism,
            "schema": binding.ARTIFACT_SCHEMAS["cuda_route_launch"],
        }
        self.phone = {
            "codec": {"argv": mechanism["desktop"][0]},
            "mechanism_commands": mechanism,
            "phones": {
                "op12": {
                    "boot_id": common.UNBOUND_BOOT_IDS["op12"],
                    "direct_peer_ipv4": unbound15,
                    **common.UNBOUND_PHONE_NETWORK["op12"],
                },
                "op15": {
                    "boot_id": common.UNBOUND_BOOT_IDS["op15"],
                    "direct_peer_ipv4": unbound12,
                    **common.UNBOUND_PHONE_NETWORK["op15"],
                },
            },
            "probes": probes,
            "processes": processes,
            "schema": binding.ARTIFACT_SCHEMAS["phone_route_launch"],
        }
        self.prospective_paths = {
            "cuda_route_launch": self.write("cuda.json", self.cuda),
            "phone_route_launch": self.write("phone.json", self.phone),
        }
        prospective_mechanism_sha = sha(canonical(mechanism))
        joint = {
            "commands": {
                name: self.joint_command(
                    self.prospective_paths[f"{name}_route_launch"],
                    prospective_mechanism_sha,
                )
                for name in ("cuda", "phone")
            },
            "history": {
                "bytes": self.history_path.stat().st_size,
                "path": str(self.history_path),
                "sha256": sha(self.history_path.read_bytes()),
            },
            "mechanism_commands": mechanism,
            "schema": binding.ARTIFACT_SCHEMAS["joint_capture_plan"],
        }
        self.prospective_paths["joint_capture_plan"] = self.write(
            "joint.json",
            joint,
        )
        runtime = {
            "candidate_sha256": sha(self.candidate_path.read_bytes()),
            "contract_sha256": sha(self.contract_path.read_bytes()),
            "cuda_monolithic_launch": self.read(self.mono_path),
            "schema": binding.ARTIFACT_SCHEMAS["runtime_plan"],
            "token_history": {
                "artifact_path": str(self.history_path),
                "model_sha256": sha(b"model"),
                "tokenizer_plan_sha256": sha(self.tokenizer_path.read_bytes()),
            },
        }
        self.prospective_paths["runtime_plan"] = self.write(
            "runtime.json",
            runtime,
        )
        self.preparation = {
            "before_boot_ids": {
                "op12": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                "op15": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            },
            "completed_ns": 100,
            "devices": {
                "cuda": {
                    "host_boot_id": "11111111-1111-4111-8111-111111111111",
                },
                "op12": {
                    "boot_id": "22222222-2222-4222-8222-222222222222",
                    "interface": "wlan0",
                    "local_ipv4": "192.0.2.12",
                },
                "op15": {
                    "boot_id": "33333333-3333-4333-8333-333333333333",
                    "interface": "wlan0",
                    "local_ipv4": "192.0.2.15",
                },
            },
            "schema": "s39-cp0-r1-reboot-preparation-v2.4",
        }
        self.preparation_path = self.write("preparation.json", self.preparation)
        self.phase_lock = {
            "contract_sha256": sha(self.contract_path.read_bytes()),
            "device_boot_ids": {
                "cuda": self.preparation["devices"]["cuda"]["host_boot_id"],
                "op12": self.preparation["devices"]["op12"]["boot_id"],
                "op15": self.preparation["devices"]["op15"]["boot_id"],
            },
            "event_ns": 200,
            "phase": common.PHASE,
            "phase_id": "cp0-r1-v24-a-only-binding-test",
            "preparation_sha256": sha(self.preparation_path.read_bytes()),
            "runtime_bundle_plan_sha256": sha(
                self.prospective_paths["runtime_plan"].read_bytes()
            ),
            "schema": "s39-cp0-r1-phase-lock-v2.4",
        }
        self.phase_lock_path = self.write("phase-lock.json", self.phase_lock)
        self.prospective_root = {
            "acquisition_ready": False,
            "artifacts": {
                name: self.record(path)
                for name, path in sorted(self.prospective_paths.items())
            },
            "candidate": self.record(self.candidate_path),
            "contract": self.record(self.contract_path),
            "cuda_monolithic_launch": self.record(self.mono_path),
            "desktop_control": {
                "cuda_ssh_target": common.CUDA_SSH_TARGET,
                "phone_adb_port": common.PHONE_ADB_PORT,
            },
            "identity_placeholders": common.UNBOUND_BOOT_IDS,
            "model_id": common.MODEL_ID,
            "model_sha256": runtime["token_history"]["model_sha256"],
            "network_placeholders": common.UNBOUND_PHONE_NETWORK,
            "phase": common.PHASE,
            "schema": binding.PROSPECTIVE_SCHEMA,
            "spec": self.record(self.spec_path),
            "status": "POST_REBOOT_IDENTITY_BINDING_REQUIRED",
            "token_history": self.record(self.history_path),
            "tokenizer_plan": self.record(self.tokenizer_path),
        }
        self.prospective_root_path = self.write(
            "prospective-root.json",
            self.prospective_root,
        )

    def write(self, name: str, value) -> Path:
        path = self.root / name
        path.write_bytes(canonical(value))
        return path

    @staticmethod
    def read(path: Path):
        return json.loads(path.read_text(encoding="ascii"))

    @staticmethod
    def record(path: Path):
        raw = path.read_bytes()
        return {"bytes": len(raw), "path": str(path), "sha256": sha(raw)}

    @staticmethod
    def joint_command(path: Path, mechanism_sha: str):
        raw = path.read_bytes()
        argv = [
            "/bin/echo",
            "--mechanism-commands-sha256",
            mechanism_sha,
            "--launch-plan",
            str(path),
        ]
        return {
            "argv_template": argv,
            "executed_files": [{
                "argv_index": 4,
                "bytes": len(raw),
                "path": str(path),
                "sha256": sha(raw),
            }],
            "launch_plan_argv_index": 4,
            "launch_plan_sha256": sha(raw),
        }

    @staticmethod
    def inline_argv(plan):
        compact = json.dumps(
            plan,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return [
            "/bin/echo",
            "--plan-json",
            compact,
            "--plan-sha256",
            sha(compact.encode("ascii")),
        ]

    def bind(self):
        bound = {
            name: self.root / "bound" / f"{name}.json"
            for name in binding.ARTIFACT_SCHEMAS
        }
        receipt_path = self.root / "bound" / "receipt.json"
        root_path = self.root / "bound" / "root.json"
        receipt, root = binding.bind(
            prospective_root_path=self.prospective_root_path,
            contract_path=self.contract_path,
            preparation_path=self.preparation_path,
            phase_lock_path=self.phase_lock_path,
            prospective_paths=self.prospective_paths,
            bound_paths=bound,
            receipt_output=receipt_path,
            bound_root_output=root_path,
            clock_ns=Clock(),
        )
        return bound, receipt, root

    def refresh_root_and_lock(self):
        self.prospective_root["artifacts"] = {
            name: self.record(path)
            for name, path in sorted(self.prospective_paths.items())
        }
        self.prospective_root_path.write_bytes(canonical(self.prospective_root))
        self.phase_lock["runtime_bundle_plan_sha256"] = sha(
            self.prospective_paths["runtime_plan"].read_bytes()
        )
        self.phase_lock_path.write_bytes(canonical(self.phase_lock))


class IdentityBindingTests(unittest.TestCase):
    def test_rebinds_actual_route_commands_and_joint_digest(self):
        fixture = Fixture(self)
        prospective_root_raw = fixture.prospective_root_path.read_bytes()
        bound, receipt, root = fixture.bind()
        phone = fixture.read(bound["phone_route_launch"])
        cuda = fixture.read(bound["cuda_route_launch"])
        joint = fixture.read(bound["joint_capture_plan"])
        all_argv = [
            *(
                command["argv"]
                for command in phone["processes"].values()
            ),
            *(
                probe[key]
                for probe in phone["probes"].values()
                for key in ("before_argv", "after_argv")
            ),
        ]
        flattened = [item for argv in all_argv for item in argv]
        self.assertIn("192.0.2.12", flattened)
        self.assertIn("192.0.2.15", flattened)
        for sentinel in (
            "0.0.0.12",
            "0.0.0.15",
            "UNBOUND_AFTER_REBOOT",
        ):
            self.assertFalse(any(sentinel in item for item in flattened))
        relay_argv = phone["processes"]["op15_direct_relay"]["argv"]
        inline = json.loads(relay_argv[relay_argv.index("--plan-json") + 1])
        self.assertEqual(inline["route"]["head_host"], "192.0.2.12")
        self.assertEqual(
            inline["route"]["runtime_argv"][-1],
            "192.0.2.12:9000",
        )
        self.assertEqual(
            relay_argv[relay_argv.index("--plan-sha256") + 1],
            sha(
                relay_argv[relay_argv.index("--plan-json") + 1].encode("ascii")
            ),
        )
        self.assertEqual(cuda["mechanism_commands"], phone["mechanism_commands"])
        self.assertEqual(joint["mechanism_commands"], phone["mechanism_commands"])
        realized_sha = sha(canonical(phone["mechanism_commands"]))
        self.assertEqual(receipt["mechanism_commands_sha256"], realized_sha)
        self.assertEqual(root["mechanism_commands_sha256"], realized_sha)
        attestation = binding._attestation(receipt, root)
        self.assertEqual(
            attestation,
            {
                "bound_root_sha256": sha(canonical(root)),
                "identity_binding_receipt_sha256": sha(canonical(receipt)),
                "schema": binding.ATTESTATION_SCHEMA,
                "status": "POST_REBOOT_IDENTITY_BINDING_PASS",
            },
        )
        for command in joint["commands"].values():
            option = command["argv_template"].index(
                "--mechanism-commands-sha256"
            )
            self.assertEqual(command["argv_template"][option + 1], realized_sha)
        self.assertEqual(
            fixture.prospective_root_path.read_bytes(),
            prospective_root_raw,
        )

    def test_pre_reboot_identity_in_prospective_plan_is_rejected(self):
        fixture = Fixture(self)
        phone = fixture.read(fixture.prospective_paths["phone_route_launch"])
        phone["phones"]["op12"]["boot_id"] = (
            fixture.preparation["before_boot_ids"]["op12"]
        )
        fixture.prospective_paths["phone_route_launch"].write_bytes(canonical(phone))
        fixture.refresh_root_and_lock()
        with self.assertRaisesRegex(
            common.ProductionError,
            "E_PROSPECTIVE_IDENTITY_NOT_UNBOUND",
        ):
            fixture.bind()

    def test_stale_network_value_in_prospective_metadata_is_rejected(self):
        fixture = Fixture(self)
        phone = fixture.read(fixture.prospective_paths["phone_route_launch"])
        phone["phones"]["op12"]["local_ipv4"] = "192.0.2.99"
        fixture.prospective_paths["phone_route_launch"].write_bytes(canonical(phone))
        fixture.refresh_root_and_lock()
        with self.assertRaisesRegex(
            common.ProductionError,
            "E_PROSPECTIVE_NETWORK_NOT_UNBOUND",
        ):
            fixture.bind()

    def test_reused_post_reboot_identity_is_rejected(self):
        fixture = Fixture(self)
        fixture.preparation["before_boot_ids"]["op12"] = (
            fixture.preparation["devices"]["op12"]["boot_id"]
        )
        fixture.preparation_path.write_bytes(canonical(fixture.preparation))
        fixture.phase_lock["preparation_sha256"] = sha(
            fixture.preparation_path.read_bytes()
        )
        fixture.phase_lock_path.write_bytes(canonical(fixture.phase_lock))
        with self.assertRaisesRegex(
            common.ProductionError,
            "E_REBOOT_IDENTITY_REUSE",
        ):
            fixture.bind()

    def test_embedded_sentinel_that_is_not_a_declared_token_is_rejected(self):
        fixture = Fixture(self)
        phone = fixture.read(fixture.prospective_paths["phone_route_launch"])
        phone["phones"]["op12"]["device"] = "tcp://0.0.0.15:9000"
        fixture.prospective_paths["phone_route_launch"].write_bytes(canonical(phone))
        fixture.refresh_root_and_lock()
        with self.assertRaisesRegex(
            common.ProductionError,
            "E_SENTINEL_LEAKAGE",
        ):
            fixture.bind()

    def test_stale_inline_plan_digest_is_rejected(self):
        fixture = Fixture(self)
        phone = fixture.read(fixture.prospective_paths["phone_route_launch"])
        argv = phone["processes"]["op15_direct_relay"]["argv"]
        digest_index = argv.index("--plan-sha256") + 1
        argv[digest_index] = sha(b"stale")
        fixture.prospective_paths["phone_route_launch"].write_bytes(canonical(phone))
        fixture.refresh_root_and_lock()
        with self.assertRaisesRegex(
            common.ProductionError,
            "plan_sha256",
        ):
            fixture.bind()

    def test_unpaired_inline_plan_option_is_rejected(self):
        fixture = Fixture(self)
        phone = fixture.read(fixture.prospective_paths["phone_route_launch"])
        argv = phone["processes"]["op15_direct_relay"]["argv"]
        option_index = argv.index("--plan-sha256")
        del argv[option_index:option_index + 2]
        fixture.prospective_paths["phone_route_launch"].write_bytes(canonical(phone))
        fixture.refresh_root_and_lock()
        with self.assertRaisesRegex(
            common.ProductionError,
            "E_INLINE_PLAN_OPTIONS",
        ):
            fixture.bind()


if __name__ == "__main__":
    unittest.main()
