#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
ORCHESTRATION = HERE.parent
V24 = ORCHESTRATION.parent
S39 = V24.parent
for path in (ORCHESTRATION, V24):
    sys.path.insert(0, str(path))

import orchestration_v1 as subject
import build_contract_v24
import cp0_r1_evidence_v24 as evidence


def marker(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


FAKE_SOURCE = br'''#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import sys


def value(name):
    index = sys.argv.index(name)
    return sys.argv[index + 1]


def write(path, schema, phase_id=None):
    data = {"schema": schema}
    if phase_id is not None:
        data["phase"] = "A_ONLY"
        data["phase_id"] = phase_id
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


stage = value("--fake-stage")
phase_id = value("--phase-id") if "--phase-id" in sys.argv else None
if stage == "artifact_root":
    data = {
        "candidate_sha256": sha(value("--candidate")),
        "completed_ns": 200,
        "components": [{
            "bytes": 9001752960,
            "component_id": "model.cuda",
            "endpoint": "cuda",
            "kind": "model_weight",
            "path": "/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf",
            "sha256": "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0",
            "stat": {
                "ctime_ns": 1,
                "device_id": 1,
                "inode": 1,
                "mode": 33188,
                "mtime_ns": 1,
                "size": 9001752960,
            },
        }],
        "contract_sha256": sha(value("--contract")),
        "inventories": [{"endpoint": "cuda"}],
        "model_id": "qwen3-14b-q4_k_m",
        "phase_scope": "PRE_REBOOT_OUTSIDE_PHASE",
        "runtime_bundle_plan_sha256": sha(value("--runtime-plan")),
        "schema": "s39-cp0-r1-artifact-root-v2.4",
        "started_ns": 100,
    }
    Path(value("--output")).parent.mkdir(parents=True, exist_ok=True)
    Path(value("--output")).write_text(
        json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
elif stage == "preparation":
    write(value("--output"), "s39-cp0-r1-reboot-preparation-v2.4")
elif stage == "phase_lock":
    data = {
        "artifact_root_sha256": sha(value("--root")),
        "candidate_sha256": sha(value("--candidate")),
        "contract_sha256": sha(value("--contract")),
        "device_boot_ids": {
            "cuda": "11111111-1111-4111-8111-111111111111",
            "op12": "22222222-2222-4222-8222-222222222222",
            "op15": "33333333-3333-4333-8333-333333333333",
        },
        "event_ns": 400,
        "model_id": "qwen3-14b-q4_k_m",
        "phase": "A_ONLY",
        "phase_id": phase_id,
        "preparation_sha256": sha(value("--preparation")),
        "quality_corpus_sha256": sha(value("--quality-corpus")),
        "runtime_bundle_plan_sha256": sha(value("--runtime-plan")),
        "schema": "s39-cp0-r1-phase-lock-v2.4",
    }
    Path(value("--output")).parent.mkdir(parents=True, exist_ok=True)
    Path(value("--output")).write_text(
        json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
elif stage == "identity_binding":
    pairs = (
        ("--prospective-cuda-route-launch", "--bound-cuda-route-launch"),
        ("--prospective-joint-capture-plan", "--bound-joint-capture-plan"),
        ("--prospective-phone-route-launch", "--bound-phone-route-launch"),
        ("--prospective-runtime-plan", "--bound-runtime-plan"),
    )
    outputs = {}
    for source, destination in pairs:
        target = Path(value(destination))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(value(source)).read_bytes())
        name = destination.removeprefix("--bound-").replace("-", "_")
        outputs[name] = {
            "bytes": target.stat().st_size,
            "path": str(target),
            "sha256": sha(target),
        }
    joint = json.loads(Path(value("--bound-joint-capture-plan")).read_text())
    mechanism_sha256 = hashlib.sha256(
        (
            json.dumps(
                joint["mechanism_commands"],
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    ).hexdigest()
    receipt = {
        "mechanism_commands_sha256": mechanism_sha256,
        "outputs": outputs,
        "schema": "s39-cp0-r1-v24-identity-binding-receipt-v1",
    }
    receipt_path = Path(value("--receipt"))
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    root = {
        "artifacts": outputs,
        "identity_binding_receipt": {
            "bytes": receipt_path.stat().st_size,
            "path": str(receipt_path),
            "sha256": sha(receipt_path),
        },
        "mechanism_commands_sha256": mechanism_sha256,
        "schema": "s39-cp0-r1-v24-bound-runtime-root-v1",
    }
    root_path = Path(value("--bound-root"))
    root_path.write_text(
        json.dumps(root, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "bound_root_sha256": sha(root_path),
        "identity_binding_receipt_sha256": sha(receipt_path),
        "schema": "s39-cp0-r1-v24-identity-binding-attestation-v1",
        "status": "POST_REBOOT_IDENTITY_BINDING_PASS",
    }, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
elif stage == "fresh_readiness":
    root = value("--root")
    lock = value("--phase-lock")
    preparation = value("--preparation")
    data = {
        "artifact_root_sha256": sha(root),
        "completed_ns": 600,
        "component_stats": [{
            "component_id": "model.cuda",
            "endpoint": "cuda",
            "path": "/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf",
            "stat": {
                "ctime_ns": 1,
                "device_id": 1,
                "inode": 1,
                "mode": 33188,
                "mtime_ns": 1,
                "size": 9001752960,
            },
        }],
        "devices": {
            "cuda": {
                "gpu_uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
                "host": "zhihao-Z690-C-ac",
                "host_boot_id": "11111111-1111-4111-8111-111111111111",
                "pci_bus_id": "0000:01:00.0",
                "system_swap_used_bytes": 0,
            },
            "op12": {
                "available_bytes": 1000000000,
                "boot_id": "22222222-2222-4222-8222-222222222222",
                "device": "OP595DL1",
                "model": "CPH2583",
                "product": "CPH2583",
                "serial": "5ae7a43d",
                "system_swap_used_bytes": 0,
                "thermal_status": 0,
            },
            "op15": {
                "available_bytes": 1000000000,
                "boot_id": "33333333-3333-4333-8333-333333333333",
                "device": "OP611FL1",
                "model": "CPH2749",
                "product": "CPH2749",
                "serial": "3C15AU002CL00000",
                "system_swap_used_bytes": 0,
                "thermal_status": 0,
            },
        },
        "inventories": [{"endpoint": "cuda"}],
        "phase": "A_ONLY",
        "phase_id": phase_id,
        "phase_lock_sha256": sha(lock),
        "preparation_sha256": sha(preparation),
        "runtime_bundle_plan_sha256": sha(value("--runtime-plan")),
        "schema": "s39-cp0-r1-fast-fresh-readiness-v2.4",
        "started_ns": 500,
    }
    Path(value("--output")).parent.mkdir(parents=True, exist_ok=True)
    Path(value("--output")).write_text(
        json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
elif stage == "readiness_projection":
    print(json.dumps({
        "schema": "s39-cp0-r1-v24-pre-acquisition-projection-v1",
        "status": "V2_4_PRE_ACQUISITION_READINESS_PASS",
    }, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
elif stage == "cuda_monolithic":
    if sha(value("--histories")) != value("--histories-sha256"):
        raise SystemExit(4)
    if sha(value("--launch-plan")) != value("--launch-plan-sha256"):
        raise SystemExit(4)
    write(value("--output"), "s39-cp0-r1-v24-cuda-monolithic-raw-v1", phase_id)
elif stage == "joint_phone_cuda":
    if sha(value("--capture-plan")) != value("--capture-plan-sha256"):
        raise SystemExit(4)
    write(value("--output"), "s39-cp0-r1-v24-joint-phone-cuda-raw-v1", phase_id)
elif stage == "fan_in":
    write(value("--runtime"), "s39-cp0-r1-runtime-identity-v2.4", phase_id)
    write(value("--acquisition"), "s39-cp0-r1-a-only-acquisition-v2.4", phase_id)
    root = Path(value("--bundle-root"))
    root.mkdir(parents=True, exist_ok=True)
    write(root / "EVIDENCE_BUNDLE_V2_4.json", "fake-raw-manifest-v1", phase_id)
elif stage == "authority":
    print(json.dumps({
        "schema": "s39-cp0-r1-evidence-result-v2.4",
        "status": "MODEL_A_QUALIFICATION_PASS_V2_4",
    }, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
else:
    raise SystemExit(3)
'''

AUTHORITY_SOURCE = br'''#!/usr/bin/env python3
from fake_authority_support import run

run()
'''

AUTHORITY_SUPPORT = br'''import json


def run():
    print(json.dumps({
        "schema": "s39-cp0-r1-evidence-result-v2.4",
        "status": "MODEL_A_QUALIFICATION_PASS_V2_4",
    }, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
'''


class RecordingRunner:
    def __init__(self, fail_stage: str | None = None, omit_output: str | None = None):
        self.fail_stage = fail_stage
        self.omit_output = omit_output
        self.calls: list[str] = []
        self.argvs: list[list[str]] = []
        self.real = subject.SubprocessRunner()

    def run(self, argv, *, cwd, env, timeout):
        stage = argv[argv.index("--fake-stage") + 1]
        self.calls.append(stage)
        self.argvs.append(list(argv))
        if stage == self.fail_stage:
            return subprocess.CompletedProcess(argv, 2, b"", b"forced failure\n")
        if stage == self.omit_output:
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        return self.real.run(argv, cwd=cwd, env=env, timeout=timeout)


class BundleRootRunner(RecordingRunner):
    def run(self, argv, *, cwd, env, timeout):
        stage = argv[argv.index("--fake-stage") + 1]
        if stage == "fan_in":
            bundle_root = Path(argv[argv.index("--bundle-root") + 1])
            if bundle_root.exists():
                raise AssertionError("bundle root existed before fan-in")
        return super().run(argv, cwd=cwd, env=env, timeout=timeout)


class BoundMutationRunner(RecordingRunner):
    def __init__(self, trigger_stage: str, target_option: str):
        super().__init__()
        self.trigger_stage = trigger_stage
        self.target_option = target_option
        self.bound_paths: dict[str, Path] = {}

    def run(self, argv, *, cwd, env, timeout):
        stage = argv[argv.index("--fake-stage") + 1]
        if stage == "identity_binding":
            for option in (
                "--bound-cuda-route-launch",
                "--bound-joint-capture-plan",
                "--bound-phone-route-launch",
                "--bound-runtime-plan",
            ):
                self.bound_paths[option] = Path(argv[argv.index(option) + 1])
        completed = super().run(argv, cwd=cwd, env=env, timeout=timeout)
        if stage == self.trigger_stage:
            path = self.bound_paths[self.target_option]
            value = json.loads(path.read_text(encoding="ascii"))
            value["post_binding_mutation"] = True
            path.write_text(
                json.dumps(
                    value,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="ascii",
            )
        return completed


class CoherentBoundMutationRunner(BoundMutationRunner):
    def __init__(self):
        super().__init__(
            "never",
            "--bound-joint-capture-plan",
        )

    @staticmethod
    def record(path: Path):
        return {
            "bytes": path.stat().st_size,
            "path": str(path),
            "sha256": subject.sha256_bytes(path.read_bytes()),
        }

    def run(self, argv, *, cwd, env, timeout):
        completed = super().run(argv, cwd=cwd, env=env, timeout=timeout)
        stage = argv[argv.index("--fake-stage") + 1]
        if stage != "identity_binding":
            return completed
        joint_path = self.bound_paths["--bound-joint-capture-plan"]
        joint = json.loads(joint_path.read_text(encoding="ascii"))
        joint["coherent_post_binding_mutation"] = True
        joint_path.write_text(
            json.dumps(
                joint,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="ascii",
        )
        receipt_path = Path(argv[argv.index("--receipt") + 1])
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        receipt["outputs"]["joint_capture_plan"] = self.record(joint_path)
        receipt_path.write_text(
            json.dumps(
                receipt,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="ascii",
        )
        root_path = Path(argv[argv.index("--bound-root") + 1])
        root = json.loads(root_path.read_text(encoding="ascii"))
        root["artifacts"]["joint_capture_plan"] = self.record(joint_path)
        root["identity_binding_receipt"] = self.record(receipt_path)
        root_path.write_text(
            json.dumps(
                root,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="ascii",
        )
        return completed


class JointInvocationMutationRunner(BoundMutationRunner):
    def __init__(self):
        super().__init__(
            "never",
            "--bound-joint-capture-plan",
        )

    def run(self, argv, *, cwd, env, timeout):
        stage = argv[argv.index("--fake-stage") + 1]
        if stage == "joint_phone_cuda":
            path = self.bound_paths["--bound-joint-capture-plan"]
            value = json.loads(path.read_text(encoding="ascii"))
            value["mutation_at_joint_invocation"] = True
            path.write_text(
                json.dumps(
                    value,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="ascii",
            )
        return super().run(argv, cwd=cwd, env=env, timeout=timeout)


class Clock:
    def __init__(self):
        self.value = 10_000

    def __call__(self):
        self.value += 10
        return self.value


class Fixture:
    def __init__(self, case: unittest.TestCase):
        temporary = tempfile.TemporaryDirectory()
        case.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.fake = self.root / "fake_stage.py"
        self.fake.write_bytes(FAKE_SOURCE)
        self.fake.chmod(0o755)
        self.authority = self.root / "fake_authority.py"
        self.authority.write_bytes(AUTHORITY_SOURCE)
        self.authority.chmod(0o755)
        self.authority_support = self.root / "fake_authority_support.py"
        self.authority_support.write_bytes(AUTHORITY_SUPPORT)
        source_raw = self.fake.read_bytes()
        source_pin = {
            "bytes": len(source_raw),
            "path": self.fake.name,
            "sha256": subject.sha256_bytes(source_raw),
        }
        authority_raw = self.authority.read_bytes()
        authority_support_raw = self.authority_support.read_bytes()
        authority_pin = {
            "bytes": len(authority_raw),
            "path": self.authority.name,
            "sha256": subject.sha256_bytes(authority_raw),
        }
        authority_support_pin = {
            "bytes": len(authority_support_raw),
            "path": self.authority_support.name,
            "sha256": subject.sha256_bytes(authority_support_raw),
        }
        self.phase_id = "cp0-r1-v24-a-only-test"
        self.model_sha = marker("model-a")
        self._write_inputs(source_pin, authority_pin, authority_support_pin)
        self.config_path = self.root / "config.json"
        self.plan_path = self.root / "plan.json"
        self.run_root = self.root / "run"
        self.config = self._config()
        self.config_path.write_bytes(subject.canonical_bytes(self.config))
        subject.build_plan(self.config_path, self.plan_path)

    def write(self, name: str, value: dict) -> Path:
        path = self.inputs / f"{name}.json"
        path.write_bytes(subject.canonical_bytes(value))
        return path

    def _write_inputs(
        self,
        source_pin: dict,
        authority_pin: dict,
        authority_support_pin: dict,
    ) -> None:
        candidate = {
            "models": [
                {
                    "artifact": {"sha256": self.model_sha},
                    "model_id": subject.MODEL_ID,
                    "slot": "A",
                }
            ],
            "schema": "s39-cp0-r1-candidate-v1",
        }
        self.paths = {}
        self.paths["candidate"] = self.write("candidate", candidate)
        corpus_raw = b"".join(
            subject.canonical_bytes(
                {
                    "answer": "A",
                    "choices": ["A", "B", "C", "D"],
                    "item_index": index,
                    "question": f"Question {index}",
                }
            )
            for index in range(64)
        )
        self.paths["quality_corpus"] = self.inputs / "quality_corpus.jsonl"
        self.paths["quality_corpus"].write_bytes(corpus_raw)
        contract = {
            "exit_authority": {
                "entrypoint": copy.deepcopy(authority_pin),
                "support": {
                    "fake_support": copy.deepcopy(authority_support_pin),
                },
            },
            "orchestration_requirements": {
                "source_programs": {
                    name: copy.deepcopy(source_pin)
                    for name in subject.ORCHESTRATION_SOURCE_STAGES
                },
                "stage_support": {
                    name: ["production_common"]
                    for name in subject.ORCHESTRATION_SOURCE_STAGES
                },
                "support": {
                    "production_common": copy.deepcopy(authority_support_pin),
                },
            },
            "producer_requirements": {
                "source_programs": {
                    "cuda_monolithic": copy.deepcopy(source_pin),
                    "joint_phone_cuda": copy.deepcopy(source_pin),
                }
            },
            "quality_corpus": {
                "bytes": len(corpus_raw),
                "path": "inputs/quality_corpus.jsonl",
                "sha256": subject.sha256_bytes(corpus_raw),
            },
            "schema": "s39-cp0-r1-evidence-contract-v2.4",
        }
        self.paths["contract"] = self.write("contract", contract)
        history = {
            "model_sha256": self.model_sha,
            "schema": "s39-cp0-r1-token-history-v2.4",
        }
        self.paths["token_history"] = self.write("token_history", history)
        self.paths["tokenizer_plan"] = self.write(
            "tokenizer_plan",
            {"schema": "s39-cp0-r1-a-only-tokenizer-plan-v2"},
        )
        mono_launch = {
            "model_sha256": self.model_sha,
            "schema": "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
        }
        self.paths["cuda_monolithic_launch"] = self.write(
            "cuda_monolithic_launch",
            mono_launch,
        )
        self.paths["cuda_route_launch"] = self.write(
            "cuda_route_launch",
            {"schema": "s39-cp0-r1-v24-cuda-route-launch-v1"},
        )
        self.paths["phone_route_launch"] = self.write(
            "phone_route_launch",
            {"schema": "s39-cp0-r1-v24-phone-route-launch-v1"},
        )
        joint = {
            "commands": {
                "cuda": {
                    "launch_plan_sha256": subject.sha256_bytes(
                        self.paths["cuda_route_launch"].read_bytes()
                    )
                },
                "phone": {
                    "launch_plan_sha256": subject.sha256_bytes(
                        self.paths["phone_route_launch"].read_bytes()
                    )
                },
            },
            "history": {
                "sha256": subject.sha256_bytes(
                    self.paths["token_history"].read_bytes()
                )
            },
            "mechanism_commands": {"desktop": [["true"]], "op12": [["true"]], "op15": [["true"]]},
            "schema": "s39-cp0-r1-v24-joint-capture-plan-v1",
        }
        self.paths["joint_capture_plan"] = self.write("joint_capture_plan", joint)
        source_component = {
            "bytes": source_pin["bytes"],
            "component_id": "",
            "sha256": source_pin["sha256"],
        }
        captures = []
        components = []
        for index, kind in enumerate(
            ("artifact_root", "cuda_monolithic", "fast_fresh_readiness", "joint_phone_cuda")
        ):
            component_id = f"capture-{index}"
            components.append({**source_component, "component_id": component_id})
            captures.append({"component_id": component_id, "kind": kind})
        runtime = {
            "candidate_sha256": subject.sha256_bytes(self.paths["candidate"].read_bytes()),
            "capture_entrypoints": captures,
            "components": components,
            "contract_sha256": subject.sha256_bytes(self.paths["contract"].read_bytes()),
            "cuda_monolithic_launch": mono_launch,
            "schema": "s39-cp0-r1-runtime-bundle-plan-v2.4",
            "token_history": {
                "artifact_path": str(self.paths["token_history"]),
                "model_sha256": self.model_sha,
                "tokenizer_plan_path": str(self.paths["tokenizer_plan"]),
                "tokenizer_plan_sha256": subject.sha256_bytes(
                    self.paths["tokenizer_plan"].read_bytes()
                ),
            },
        }
        self.paths["runtime_plan"] = self.write("runtime_plan", runtime)
        prospective = {
            "acquisition_ready": False,
            "artifacts": {
                name: {
                    "bytes": path.stat().st_size,
                    "path": str(path),
                    "sha256": subject.sha256_bytes(path.read_bytes()),
                }
                for name, path in (
                    ("cuda_route_launch", self.paths["cuda_route_launch"]),
                    ("joint_capture_plan", self.paths["joint_capture_plan"]),
                    ("phone_route_launch", self.paths["phone_route_launch"]),
                    ("runtime_plan", self.paths["runtime_plan"]),
                )
            },
            "model_id": subject.MODEL_ID,
            "phase": subject.PHASE,
            "schema": "s39-cp0-r1-v24-prospective-runtime-root-v1",
        }
        self.paths["prospective_root"] = self.write(
            "prospective_root",
            prospective,
        )

    @staticmethod
    def _base(stage: str, args: list[str]) -> dict:
        return {
            "argv_template": ["--fake-stage", stage, *args],
            "cwd": "/tmp",
            "entrypoint": "",
            "environment": {
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "support_files": [],
            "timeout_seconds": 30,
        }

    def _commands(self) -> dict:
        common = ["--phase-id", "{phase_id}"]
        values = {
            "artifact_root": self._base(
                "artifact_root",
                [
                    "--output",
                    "{artifact_root}",
                    "--contract",
                    "{contract}",
                    "--candidate",
                    "{candidate}",
                    "--runtime-plan",
                    "{runtime_plan}",
                    "--history",
                    "{token_history}",
                    "--tokenizer-plan",
                    "{tokenizer_plan}",
                ],
            ),
            "preparation": self._base(
                "preparation",
                [
                    "--output",
                    "{preparation}",
                    "--contract",
                    "{contract}",
                    "--root",
                    "{artifact_root}",
                    "--runtime-plan",
                    "{runtime_plan}",
                ],
            ),
            "phase_lock": self._base(
                "phase_lock",
                [
                    "--output",
                    "{phase_lock}",
                    *common,
                    "--contract",
                    "{contract}",
                    "--candidate",
                    "{candidate}",
                    "--root",
                    "{artifact_root}",
                    "--preparation",
                    "{preparation}",
                    "--quality-corpus",
                    "{quality_corpus}",
                    "--runtime-plan",
                    "{runtime_plan}",
                ],
            ),
            "identity_binding": self._base(
                "identity_binding",
                [
                    "--prospective-root",
                    "{prospective_root}",
                    "--contract",
                    "{contract}",
                    "--preparation",
                    "{preparation}",
                    "--phase-lock",
                    "{phase_lock}",
                    "--prospective-cuda-route-launch",
                    "{cuda_route_launch}",
                    "--bound-cuda-route-launch",
                    "{bound_cuda_route_launch}",
                    "--prospective-joint-capture-plan",
                    "{joint_capture_plan}",
                    "--bound-joint-capture-plan",
                    "{bound_joint_capture_plan}",
                    "--prospective-phone-route-launch",
                    "{phone_route_launch}",
                    "--bound-phone-route-launch",
                    "{bound_phone_route_launch}",
                    "--prospective-runtime-plan",
                    "{runtime_plan}",
                    "--bound-runtime-plan",
                    "{bound_runtime_plan}",
                    "--receipt",
                    "{identity_binding_receipt}",
                    "--bound-root",
                    "{bound_root}",
                ],
            ),
            "fresh_readiness": self._base(
                "fresh_readiness",
                [
                    "--output",
                    "{fresh}",
                    *common,
                    "--contract",
                    "{contract}",
                    "--root",
                    "{artifact_root}",
                    "--preparation",
                    "{preparation}",
                    "--phase-lock",
                    "{phase_lock}",
                    "--runtime-plan",
                    "{bound_runtime_plan}",
                ],
            ),
            "readiness_projection": self._base(
                "readiness_projection",
                [
                    "--started",
                    "{acquisition_started_ns}",
                    "--contract",
                    "{contract}",
                    "--candidate",
                    "{candidate}",
                    "--runtime-plan",
                    "{bound_runtime_plan}",
                    "--tokenizer-plan",
                    "{tokenizer_plan}",
                    "--history",
                    "{token_history}",
                    "--root",
                    "{artifact_root}",
                    "--preparation",
                    "{preparation}",
                    "--phase-lock",
                    "{phase_lock}",
                    "--fresh",
                    "{fresh}",
                ],
            ),
            "cuda_monolithic": self._base(
                "cuda_monolithic",
                [
                    "--output",
                    "{cuda_monolithic}",
                    *common,
                    "--pre-dir",
                    "{pre_dir}",
                    "--started",
                    "{acquisition_started_ns}",
                    "--plan",
                    "{orchestration_plan_sha256}",
                    "--mechanism-commands-sha256",
                    "{mechanism_commands_sha256}",
                    "--model-sha256",
                    "{model_sha256}",
                    "--histories",
                    "{token_history}",
                    "--histories-sha256",
                    "{token_history_sha256}",
                    "--launch-plan",
                    "{cuda_monolithic_launch}",
                    "--launch-plan-sha256",
                    "{cuda_monolithic_launch_sha256}",
                    "--execute",
                    "--confirm",
                    "RUN_V24_CUDA_MONOLITHIC_A_ONLY",
                ],
            ),
            "joint_phone_cuda": self._base(
                "joint_phone_cuda",
                [
                    "--capture-plan",
                    "{bound_joint_capture_plan}",
                    "--capture-plan-sha256",
                    "{bound_joint_capture_plan_sha256}",
                    "--output",
                    "{joint_phone_cuda}",
                    *common,
                    "--pre-dir",
                    "{pre_dir}",
                    "--acquisition-started-ns",
                    "{acquisition_started_ns}",
                    "--command-plan-sha256",
                    "{orchestration_plan_sha256}",
                    "--execute",
                    "--confirm",
                    "RUN_V24_JOINT_PHONE_CUDA_A_ONLY",
                ],
            ),
            "fan_in": self._base(
                "fan_in",
                [
                    "--runtime",
                    "{runtime_identity}",
                    "--acquisition",
                    "{acquisition}",
                    "--bundle-root",
                    "{bundle_root}",
                    "--pre-dir",
                    "{pre_dir}",
                    "--started",
                    "{acquisition_started_ns}",
                    "--contract",
                    "{contract}",
                    "--candidate",
                    "{candidate}",
                    "--runtime-plan",
                    "{bound_runtime_plan}",
                    "--tokenizer-plan",
                    "{tokenizer_plan}",
                    "--history",
                    "{token_history}",
                    "--root",
                    "{artifact_root}",
                    "--preparation",
                    "{preparation}",
                    "--phase-lock",
                    "{phase_lock}",
                    "--fresh",
                    "{fresh}",
                    "--mono",
                    "{cuda_monolithic}",
                    "--joint",
                    "{joint_phone_cuda}",
                ],
            ),
            "authority": self._base(
                "authority",
                [
                    "--contract",
                    "{contract}",
                    "--candidate",
                    "{candidate}",
                    "--runtime-plan",
                    "{bound_runtime_plan}",
                    "--tokenizer-plan",
                    "{tokenizer_plan}",
                    "--token-history",
                    "{token_history}",
                    "--artifact-root",
                    "{artifact_root}",
                    "--preparation",
                    "{preparation}",
                    "--phase-lock",
                    "{phase_lock}",
                    "--fresh",
                    "{fresh}",
                    "--runtime-identity",
                    "{runtime_identity}",
                    "--acquisition",
                    "{acquisition}",
                    "--bundle-root",
                    "{bundle_root}",
                    "--orchestration-plan",
                    "{orchestration_plan}",
                    "--prospective-root",
                    "{prospective_root}",
                    "--bound-root",
                    "{bound_root}",
                    "--identity-binding-receipt",
                    "{identity_binding_receipt}",
                    "--identity-binding-stage-receipt",
                    "{identity_binding_stage_receipt}",
                ],
            ),
        }
        for value in values.values():
            value["entrypoint"] = str(self.fake)
        for stage in subject.ORCHESTRATION_SOURCE_STAGES:
            values[stage]["support_files"] = [str(self.authority_support)]
        values["authority"]["entrypoint"] = str(self.authority)
        values["authority"]["support_files"] = [str(self.authority_support)]
        return values

    def _config(self) -> dict:
        return {
            "commands": self._commands(),
            "inputs": {name: str(path) for name, path in self.paths.items()},
            "model_id": subject.MODEL_ID,
            "phase": subject.PHASE,
            "phase_id": self.phase_id,
            "run_root": str(self.run_root),
            "schema": subject.CONFIG_SCHEMA,
        }


class OrchestrationTests(unittest.TestCase):
    def assert_authority_provenance(self, fixture: Fixture) -> dict:
        contract, contract_raw = subject.read_canonical(
            fixture.paths["contract"],
            "contract",
        )
        _, candidate_raw = subject.read_canonical(
            fixture.paths["candidate"],
            "candidate",
        )
        return evidence.validate_orchestration_provenance(
            orchestration_plan_path=fixture.plan_path,
            bundle_root=fixture.run_root / "raw-bundle",
            contract_path=fixture.paths["contract"],
            candidate_path=fixture.paths["candidate"],
            contract=contract,
            contract_raw=contract_raw,
            candidate_raw=candidate_raw,
        )

    def test_preflight_is_byte_deterministic_and_runs_no_commands(self):
        fixture = Fixture(self)
        first = subject.canonical_bytes(subject.preflight(fixture.plan_path))
        second = subject.canonical_bytes(subject.preflight(fixture.plan_path))
        self.assertEqual(first, second)
        self.assertIn(
            b'"status":"NO_MODEL_PREFLIGHT_PASS_STAGED_IDENTITY_BINDING_REQUIRED"',
            first,
        )
        self.assertIn(b'"op12.boot_id"', first)
        self.assertFalse(fixture.run_root.exists())

    def test_four_contract_pinned_source_records_validate(self):
        fixture = Fixture(self)
        plan, _ = subject.load_plan(fixture.plan_path)
        pins = fixture.config["commands"]
        for stage in subject.ORCHESTRATION_SOURCE_STAGES:
            self.assertEqual(
                plan["stages"][stage]["entrypoint"]["path"],
                pins[stage]["entrypoint"],
            )
        self.assertEqual(
            subject.preflight(fixture.plan_path)["unresolved_provenance"],
            ["cuda.boot_id", "op12.boot_id", "op15.boot_id"],
        )

    def test_orchestration_source_file_mutation_is_rejected(self):
        fixture = Fixture(self)
        fixture.fake.write_bytes(fixture.fake.read_bytes() + b"\n")
        with self.assertRaisesRegex(subject.OrchestrationError, "E_VALUE"):
            subject.preflight(fixture.plan_path)

    def test_orchestration_source_path_mutation_is_rejected(self):
        fixture = Fixture(self)
        alternate = fixture.root / "alternate.py"
        alternate.write_bytes(fixture.fake.read_bytes())
        plan = subject.parse_json(fixture.plan_path.read_bytes(), "plan")
        plan["stages"]["preparation"]["entrypoint"] = subject.file_record(
            alternate,
            "alternate",
        )
        fixture.plan_path.write_bytes(subject.canonical_bytes(plan))
        with self.assertRaisesRegex(subject.OrchestrationError, "E_SOURCE_PIN_PATH"):
            subject.preflight(fixture.plan_path)

    def test_orchestration_source_stat_mutation_is_rejected(self):
        fixture = Fixture(self)
        plan = subject.parse_json(fixture.plan_path.read_bytes(), "plan")
        plan["stages"]["phase_lock"]["entrypoint"]["stat"]["mtime_ns"] += 1
        fixture.plan_path.write_bytes(subject.canonical_bytes(plan))
        with self.assertRaisesRegex(subject.OrchestrationError, "E_VALUE"):
            subject.preflight(fixture.plan_path)

    def test_orchestration_source_digest_mutation_is_rejected(self):
        fixture = Fixture(self)
        plan = subject.parse_json(fixture.plan_path.read_bytes(), "plan")
        plan["stages"]["fan_in"]["entrypoint"]["sha256"] = marker("mutated")
        fixture.plan_path.write_bytes(subject.canonical_bytes(plan))
        with self.assertRaisesRegex(subject.OrchestrationError, "E_VALUE"):
            subject.preflight(fixture.plan_path)

    def test_missing_orchestration_support_is_rejected(self):
        fixture = Fixture(self)
        plan = subject.parse_json(fixture.plan_path.read_bytes(), "plan")
        plan["stages"]["readiness_projection"]["support_files"] = []
        fixture.plan_path.write_bytes(subject.canonical_bytes(plan))
        with self.assertRaisesRegex(
            subject.OrchestrationError,
            "E_ORCHESTRATION_SUPPORT_PINS",
        ):
            subject.preflight(fixture.plan_path)

    def test_final_authority_rederives_four_source_provenances(self):
        fixture = Fixture(self)
        subject.run_acquisition(
            fixture.plan_path,
            runner=RecordingRunner(),
            now_ns=Clock(),
            test_only=True,
        )
        result = self.assert_authority_provenance(fixture)
        self.assertEqual(
            result["status"],
            "V2_4_ORCHESTRATION_PROVENANCE_PASS",
        )
        self.assertEqual(
            set(result["source_sha256s"]),
            set(subject.ORCHESTRATION_SOURCE_STAGES),
        )

    def test_final_authority_rejects_source_file_mutation(self):
        fixture = Fixture(self)
        subject.run_acquisition(
            fixture.plan_path,
            runner=RecordingRunner(),
            now_ns=Clock(),
            test_only=True,
        )
        mutated = bytearray(fixture.fake.read_bytes())
        mutated[0] ^= 1
        fixture.fake.write_bytes(mutated)
        current = subject.file_record(fixture.fake, "mutated_source")
        plan = subject.parse_json(fixture.plan_path.read_bytes(), "plan")
        for stage in subject.ORCHESTRATION_SOURCE_STAGES:
            plan["stages"][stage]["entrypoint"]["stat"] = copy.deepcopy(
                current["stat"]
            )
        fixture.plan_path.write_bytes(subject.canonical_bytes(plan))
        with self.assertRaisesRegex(
            evidence.common.EvidenceError,
            "E_ORCHESTRATION_LIVE_SHA256",
        ):
            self.assert_authority_provenance(fixture)

    def test_final_authority_rejects_source_path_mutation(self):
        fixture = Fixture(self)
        subject.run_acquisition(
            fixture.plan_path,
            runner=RecordingRunner(),
            now_ns=Clock(),
            test_only=True,
        )
        alternate = fixture.root / "alternate.py"
        alternate.write_bytes(fixture.fake.read_bytes())
        plan = subject.parse_json(fixture.plan_path.read_bytes(), "plan")
        plan["stages"]["preparation"]["entrypoint"] = subject.file_record(
            alternate,
            "alternate",
        )
        fixture.plan_path.write_bytes(subject.canonical_bytes(plan))
        with self.assertRaisesRegex(
            evidence.common.EvidenceError,
            "E_ORCHESTRATION_SOURCE_PATH",
        ):
            self.assert_authority_provenance(fixture)

    def test_final_authority_rejects_source_stat_mutation(self):
        fixture = Fixture(self)
        subject.run_acquisition(
            fixture.plan_path,
            runner=RecordingRunner(),
            now_ns=Clock(),
            test_only=True,
        )
        plan = subject.parse_json(fixture.plan_path.read_bytes(), "plan")
        plan["stages"]["phase_lock"]["entrypoint"]["stat"]["mtime_ns"] += 1
        fixture.plan_path.write_bytes(subject.canonical_bytes(plan))
        with self.assertRaisesRegex(
            evidence.common.EvidenceError,
            "E_ORCHESTRATION_SOURCE_STAT",
        ):
            self.assert_authority_provenance(fixture)

    def test_final_authority_rejects_source_digest_mutation(self):
        fixture = Fixture(self)
        subject.run_acquisition(
            fixture.plan_path,
            runner=RecordingRunner(),
            now_ns=Clock(),
            test_only=True,
        )
        plan = subject.parse_json(fixture.plan_path.read_bytes(), "plan")
        plan["stages"]["fan_in"]["entrypoint"]["sha256"] = marker("mutated")
        fixture.plan_path.write_bytes(subject.canonical_bytes(plan))
        with self.assertRaisesRegex(
            evidence.common.EvidenceError,
            "E_ORCHESTRATION_SOURCE_SHA256",
        ):
            self.assert_authority_provenance(fixture)

    def test_final_authority_rejects_captured_support_mutation(self):
        fixture = Fixture(self)
        subject.run_acquisition(
            fixture.plan_path,
            runner=RecordingRunner(),
            now_ns=Clock(),
            test_only=True,
        )
        captured = (
            fixture.run_root
            / "executed"
            / "readiness_projection"
            / "support"
            / f"000-{fixture.authority_support.name}"
        )
        captured.write_bytes(captured.read_bytes() + b"\n")
        with self.assertRaisesRegex(
            evidence.common.EvidenceError,
            "E_ORCHESTRATION_EXECUTED_SOURCE",
        ):
            self.assert_authority_provenance(fixture)

    def test_full_fake_sequence_runs_in_contract_order(self):
        fixture = Fixture(self)
        runner = RecordingRunner()
        result = subject.run_acquisition(
            fixture.plan_path,
            runner=runner,
            now_ns=Clock(),
            test_only=True,
        )
        self.assertEqual(runner.calls, list(subject.STAGE_ORDER))
        self.assertEqual(
            result["status"],
            "TEST_ONLY_SEQUENCE_PASS_NOT_ACQUISITION_EVIDENCE",
        )
        mono_argv = runner.argvs[runner.calls.index("cuda_monolithic")]
        self.assertEqual(
            mono_argv[mono_argv.index("--histories") + 1],
            str(fixture.paths["token_history"]),
        )
        joint_argv = runner.argvs[runner.calls.index("joint_phone_cuda")]
        self.assertEqual(
            joint_argv[joint_argv.index("--capture-plan") + 1],
            str(fixture.run_root / "bound" / "joint-capture-plan.json"),
        )

    def test_fan_in_exclusively_creates_bundle_root(self):
        fixture = Fixture(self)
        subject.run_acquisition(
            fixture.plan_path,
            runner=BundleRootRunner(),
            now_ns=Clock(),
            test_only=True,
        )
        self.assertTrue((fixture.run_root / "raw-bundle").is_dir())

    def test_no_producer_launch_before_fresh_projection_passes(self):
        fixture = Fixture(self)
        runner = RecordingRunner(fail_stage="readiness_projection")
        with self.assertRaisesRegex(subject.OrchestrationError, "E_STAGE_EXIT"):
            subject.run_acquisition(
                fixture.plan_path,
                runner=runner,
                now_ns=Clock(),
                test_only=True,
            )
        self.assertEqual(
            runner.calls,
            list(
                subject.STAGE_ORDER[
                    : subject.STAGE_ORDER.index("readiness_projection") + 1
                ]
            ),
        )
        self.assertNotIn("cuda_monolithic", runner.calls)
        self.assertNotIn("joint_phone_cuda", runner.calls)

    def test_bound_runtime_mutation_before_fresh_prevents_stage_invocation(self):
        fixture = Fixture(self)
        plan_path = fixture.plan_path
        runner = RecordingRunner()
        original = subject._pin_bound_outputs

        def pin_then_mutate(paths, attestation):
            pins = original(paths, attestation)
            path = paths["bound_runtime_plan"]
            value = json.loads(path.read_text(encoding="ascii"))
            value["post_binding_mutation"] = True
            path.write_text(
                json.dumps(
                    value,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="ascii",
            )
            return pins

        with mock.patch.object(
            subject,
            "_pin_bound_outputs",
            side_effect=pin_then_mutate,
        ):
            with self.assertRaisesRegex(
                subject.OrchestrationError,
                "E_BOUND_MUTATION: fresh_readiness.runtime_plan",
            ):
                subject.run_acquisition(
                    plan_path,
                    runner=runner,
                    now_ns=Clock(),
                    test_only=True,
                )
        self.assertNotIn("fresh_readiness", runner.calls)

    def test_bound_joint_mutation_before_joint_prevents_stage_invocation(self):
        fixture = Fixture(self)
        plan_path = fixture.plan_path
        runner = BoundMutationRunner(
            "cuda_monolithic",
            "--bound-joint-capture-plan",
        )
        with self.assertRaisesRegex(
            subject.OrchestrationError,
            "E_BOUND_MUTATION: joint_phone_cuda.joint_capture_plan",
        ):
            subject.run_acquisition(
                plan_path,
                runner=runner,
                now_ns=Clock(),
                test_only=True,
            )
        self.assertNotIn("joint_phone_cuda", runner.calls)

    def test_coherent_post_binder_rewrite_fails_pipe_attestation(self):
        fixture = Fixture(self)
        runner = CoherentBoundMutationRunner()
        with self.assertRaisesRegex(
            subject.OrchestrationError,
            "E_BOUND_ATTESTATION: root",
        ):
            subject.run_acquisition(
                fixture.plan_path,
                runner=runner,
                now_ns=Clock(),
                test_only=True,
            )
        self.assertNotIn("fresh_readiness", runner.calls)
        self.assertNotIn("joint_phone_cuda", runner.calls)

    def test_mutation_at_joint_invocation_is_refused_by_expected_digest(self):
        fixture = Fixture(self)
        runner = JointInvocationMutationRunner()
        with self.assertRaisesRegex(
            subject.OrchestrationError,
            "E_STAGE_EXIT: joint_phone_cuda",
        ):
            subject.run_acquisition(
                fixture.plan_path,
                runner=runner,
                now_ns=Clock(),
                test_only=True,
            )
        self.assertIn("joint_phone_cuda", runner.calls)
        output = Path(
            next(
                argv[argv.index("--output") + 1]
                for argv in runner.argvs
                if argv[argv.index("--fake-stage") + 1] == "joint_phone_cuda"
            )
        )
        self.assertFalse(output.exists())

    def test_no_authority_before_both_producer_outputs_exist(self):
        fixture = Fixture(self)
        runner = RecordingRunner(omit_output="joint_phone_cuda")
        with self.assertRaisesRegex(subject.OrchestrationError, "E_READ|E_PRODUCER"):
            subject.run_acquisition(
                fixture.plan_path,
                runner=runner,
                now_ns=Clock(),
                test_only=True,
            )
        self.assertIn("cuda_monolithic", runner.calls)
        self.assertIn("joint_phone_cuda", runner.calls)
        self.assertNotIn("fan_in", runner.calls)
        self.assertNotIn("authority", runner.calls)

    def test_input_drift_is_rejected_before_preflight(self):
        fixture = Fixture(self)
        fixture.paths["token_history"].write_bytes(
            subject.canonical_bytes(
                {
                    "model_sha256": marker("changed"),
                    "schema": "s39-cp0-r1-token-history-v2.4",
                }
            )
        )
        with self.assertRaisesRegex(subject.OrchestrationError, "E_VALUE"):
            subject.preflight(fixture.plan_path)

    def test_missing_immutable_joint_launch_binding_is_rejected(self):
        fixture = Fixture(self)
        config = copy.deepcopy(fixture.config)
        launch = fixture.paths["joint_capture_plan"]
        value = subject.parse_json(launch.read_bytes(), "joint")
        value["commands"]["cuda"]["launch_plan_sha256"] = marker("wrong")
        launch.write_bytes(subject.canonical_bytes(value))
        config_path = fixture.root / "bad-config.json"
        output = fixture.root / "bad-plan.json"
        config_path.write_bytes(subject.canonical_bytes(config))
        with self.assertRaisesRegex(
            subject.OrchestrationError,
            "prospective.artifacts.joint_capture_plan.sha256",
        ):
            subject.build_plan(config_path, output)

    def test_staged_layout_is_consumable_by_real_v24_producer_loaders(self):
        fixture = Fixture(self)
        runner = RecordingRunner()
        subject.run_acquisition(
            fixture.plan_path,
            runner=runner,
            now_ns=Clock(),
            test_only=True,
        )

        def load(name: str, path: Path):
            spec = importlib.util.spec_from_file_location(name, path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        producers = V24 / "producers_v1"
        mono = load("test_v24_mono_loader", producers / "cuda_monolithic_v1.py")
        cuda = load("test_v24_cuda_loader", producers / "cuda_route_v1.py")
        phone = load("test_v24_phone_loader", producers / "phone_route_v1.py")
        pre = fixture.run_root / "pre"
        mono_lock, _ = mono.load_phase(pre, fixture.phase_id)
        phone_lock, phone_lock_raw = phone.load_phase(pre, fixture.phase_id)
        _, cuda_device, cuda_lock = cuda.load_base_evidence(pre, fixture.phase_id)
        self.assertEqual(mono_lock, phone_lock)
        self.assertEqual(cuda_lock, phone_lock)
        self.assertEqual(cuda_device["gpu_uuid"], cuda.CUDA_UUID)
        phone_plan = {
            "phones": {
                "op12": {
                    "boot_id": "22222222-2222-4222-8222-222222222222",
                    "device": "OP595DL1",
                    "model": "CPH2583",
                    "product": "CPH2583",
                    "serial": "5ae7a43d",
                },
                "op15": {
                    "boot_id": "33333333-3333-4333-8333-333333333333",
                    "device": "OP611FL1",
                    "model": "CPH2749",
                    "product": "CPH2749",
                    "serial": "3C15AU002CL00000",
                },
            }
        }
        devices = phone.load_v24_fresh_devices(
            pre,
            fixture.phase_id,
            phone_lock,
            phone_lock_raw,
            phone_plan,
        )
        self.assertEqual(set(devices), {"op12", "op15"})
        staged_corpus = (pre / "quality_corpus.jsonl").read_bytes()
        self.assertEqual(
            subject.sha256_bytes(staged_corpus),
            phone_lock["quality_corpus_sha256"],
        )

    def test_real_v24_authority_entrypoint_and_support_paths_validate(self):
        contract = build_contract_v24.build_contract()
        contract_root = S39
        entrypoint = contract_root / contract["exit_authority"]["entrypoint"]["path"]
        supports = [
            subject.file_record(
                contract_root / record["path"],
                f"real_support.{name}",
            )
            for name, record in sorted(
                contract["exit_authority"]["support"].items()
            )
        ]
        value = {
            "argv_template": [
                "--contract",
                "{contract}",
                "--candidate",
                "{candidate}",
                "--runtime-plan",
                "{bound_runtime_plan}",
                "--tokenizer-plan",
                "{tokenizer_plan}",
                "--token-history",
                "{token_history}",
                "--artifact-root",
                "{artifact_root}",
                "--preparation",
                "{preparation}",
                "--phase-lock",
                "{phase_lock}",
                "--fresh",
                "{fresh}",
                "--runtime-identity",
                "{runtime_identity}",
                "--acquisition",
                "{acquisition}",
                "--bundle-root",
                "{bundle_root}",
                "--orchestration-plan",
                "{orchestration_plan}",
                "--prospective-root",
                "{prospective_root}",
                "--bound-root",
                "{bound_root}",
                "--identity-binding-receipt",
                "{identity_binding_receipt}",
                "--identity-binding-stage-receipt",
                "{identity_binding_stage_receipt}",
            ],
            "cwd": str(S39.parent.parent.parent),
            "entrypoint": subject.file_record(entrypoint, "real_authority"),
            "environment": {
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "support_files": sorted(supports, key=lambda row: row["path"]),
            "timeout_seconds": 60,
        }
        subject._validate_stage(value, "authority", contract, contract_root)


if __name__ == "__main__":
    unittest.main()
