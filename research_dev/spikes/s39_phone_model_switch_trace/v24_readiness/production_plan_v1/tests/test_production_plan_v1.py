#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import runpy
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SUBJECT_ROOT = HERE.parent
V24 = SUBJECT_ROOT.parent
S39 = V24.parent
REPO = S39.parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load("test_production_common", SUBJECT_ROOT / "production_common_v1.py")
preparation = load("test_preparation", SUBJECT_ROOT / "preparation_v1.py")
phase_lock = load("test_phase_lock", SUBJECT_ROOT / "phase_lock_v1.py")
fan_in = load("test_fan_in", SUBJECT_ROOT / "fan_in_v1.py")
materializer = load("test_materializer", SUBJECT_ROOT / "materialize_config_v1.py")
builder = load("test_contract_builder", V24 / "build_contract_v24.py")


def marker(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def canonical(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


class Clock:
    def __init__(self, start=200):
        self.value = start

    def __call__(self):
        self.value += 10
        return self.value


class Monotonic:
    def __init__(self, step=0.1):
        self.value = 0.0
        self.step = step

    def __call__(self):
        value = self.value
        self.value += self.step
        return value


class FakeRunner:
    def __init__(self, delayed=False, transient_offline=False):
        self.delayed = delayed
        self.transient_offline = transient_offline
        self.status_calls = {"3C15AU002CL00000": 0, "5ae7a43d": 0}
        self.calls = []

    def run(self, argv, *, timeout):
        del timeout
        self.calls.append(list(argv))
        if argv[0] == "ssh":
            if "SWAP_USED_KB" in argv[-1]:
                stdout = (
                    b"HOST=zhihao-Z690-C-ac\n"
                    b"BOOT_ID=11111111-1111-4111-8111-111111111111\n"
                    b"SWAP_USED_KB=4096\n"
                    b"GPU_UUID=GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08\n"
                    b"PCI_BUS_ID=0000:01:00.0\n"
                )
            else:
                stdout = (
                    b"zhihao-Z690-C-ac\n"
                    b"NVIDIA GeForce RTX 4060 Ti, "
                    b"GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08, 16380\n"
                    b"MODEL_BYTES=9001752960\n"
                    + f"MODEL_SHA256={marker('model')}\n".encode("ascii")
                )
            return subprocess.CompletedProcess(argv, 0, stdout, b"")
        port = argv[argv.index("-P") + 1]
        if "devices" in argv:
            stdout = b"List of devices attached\n"
            if port == "5038":
                stdout += (
                    b"3C15AU002CL00000 device product:CPH2749 "
                    b"model:CPH2749 device:OP611FL1\n"
                    b"5ae7a43d device product:CPH2583 "
                    b"model:CPH2583 device:OP595DL1\n"
                )
            return subprocess.CompletedProcess(argv, 0, stdout, b"")
        serial = argv[argv.index("-s") + 1]
        tail = argv[-1]
        if tail == "reboot" or tail == "wait-for-device":
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        if tail == "cat /proc/sys/kernel/random/boot_id":
            boot = (
                b"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa\n"
                if serial.startswith("3C")
                else b"bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb\n"
            )
            return subprocess.CompletedProcess(argv, 0, boot, b"")
        if "BOOT_COMPLETED=" in tail:
            self.status_calls[serial] += 1
            if self.transient_offline and self.status_calls[serial] == 1:
                return subprocess.CompletedProcess(argv, 1, b"", b"device offline\n")
            ready = not self.delayed or self.status_calls[serial] > 1
            identity = {
                "3C15AU002CL00000": ("CPH2749", "OP611FL1", "c"),
                "5ae7a43d": ("CPH2583", "OP595DL1", "d"),
            }[serial]
            stdout = (
                f"BOOT_ID={identity[2] * 8}-{identity[2] * 4}-4"
                f"{identity[2] * 3}-8{identity[2] * 3}-{identity[2] * 12}\n"
                f"BOOT_COMPLETED={'1' if ready else '0'}\n"
                f"PRODUCT={identity[0]}\n"
                f"MODEL={identity[0]}\n"
                f"DEVICE={identity[1]}\n"
                "INTERFACE=wlan0\n"
                f"LOCAL_IPV4={'10.0.0.15' if serial.startswith('3C') else '10.0.0.12'}\n"
                "MEM_AVAILABLE_KB=1048576\n"
                "SWAP_USED_KB=2048\n"
                "THERMAL_STATUS=0\n"
            ).encode("ascii")
            return subprocess.CompletedProcess(argv, 0, stdout, b"")
        if "SHARD_BYTES=" in tail:
            phone = "op15" if serial.startswith("3C") else "op12"
            identity = {
                "op15": ("CPH2749", "OP611FL1", "c", 6730922144, marker("op15")),
                "op12": ("CPH2583", "OP595DL1", "d", 4274027712, marker("op12")),
            }[phone]
            stdout = (
                f"{identity[0]}\n{identity[0]}\n{identity[1]}\n"
                f"{identity[2] * 8}-{identity[2] * 4}-4"
                f"{identity[2] * 3}-8{identity[2] * 3}-{identity[2] * 12}\n"
                f"SHARD_BYTES={identity[3]}\nSHARD_SHA256={identity[4]}\n"
            ).encode("ascii")
            return subprocess.CompletedProcess(argv, 0, stdout, b"")
        raise AssertionError(argv)


class Fixture:
    def __init__(self, case):
        temporary = tempfile.TemporaryDirectory()
        case.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.contract = {
            "devices": {
                "cuda": {
                    "host": "zhihao-Z690-C-ac",
                    "uuid": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
                },
                "op12": {
                    "device": "OP595DL1",
                    "model": "CPH2583",
                    "product": "CPH2583",
                    "serial": "5ae7a43d",
                },
                "op15": {
                    "device": "OP611FL1",
                    "model": "CPH2749",
                    "product": "CPH2749",
                    "serial": "3C15AU002CL00000",
                },
            },
            "gates": {
                "artifact_root_maximum_age_ns": 10_000,
                "phone_minimum_available_bytes": 512 * 1024 * 1024,
            },
            "incumbent_route_lock": {
                "backend": "GPUOpenCL",
                "cut_layer": 30,
                "op12_shard_sha256": marker("op12"),
                "op12_stored_layers": [24, 40],
                "op15_shard_sha256": marker("op15"),
                "op15_stored_layers": [0, 32],
            },
            "model_geometry": {
                common.MODEL_ID: {
                    "activation_dtype": "F32",
                    "activation_element_bytes": 4,
                    "cuda_model_path": "/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf",
                    "hidden_size": 5120,
                    "known_shards": {
                        "op12": {
                            "bytes": 4274027712,
                            "path": "/data/local/tmp/model/weights.gguf",
                            "sha256": marker("op12"),
                        },
                        "op15": {
                            "bytes": 6730922144,
                            "path": "/data/local/tmp/model/weights.gguf",
                            "sha256": marker("op15"),
                        },
                    },
                }
            },
            "phase_protocol": {"clock_id": "HOST_MONOTONIC_RAW", "phase": "A_ONLY"},
            "quality_corpus": {},
            "raw_predicate_contract": {"sha256": marker("raw-contract")},
            "schema": "s39-cp0-r1-evidence-contract-v2.4",
            "serving_envelope": {
                "batch": 8,
                "kv_type_k": "f16",
                "kv_type_v": "f16",
                "max_streams": 8,
                "n_batch": 64,
                "n_ctx_seq": 512,
                "n_ubatch": 64,
                "sampler": "greedy",
            },
        }
        self.candidate = {
            "models": [{
                "artifact": {"bytes": 9001752960, "sha256": marker("model")},
                "model_id": common.MODEL_ID,
                "n_layer": 40,
                "slot": "A",
            }],
            "schema": "s39-cp0-r1-candidate-v1",
        }
        self.plan = {
            "schema": "s39-cp0-r1-runtime-bundle-plan-v2.4",
        }
        self.plan_path = self.write("runtime.json", self.plan)
        self.root_value = {
            "completed_ns": 100,
            "components": [{
                "bytes": 9001752960,
                "component_id": "model.cuda",
                "endpoint": "cuda",
                "kind": "model_weight",
                "path": "/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf",
                "sha256": marker("model"),
                "stat": {},
            }],
            "runtime_bundle_plan_sha256": hashlib.sha256(
                self.plan_path.read_bytes()
            ).hexdigest(),
            "schema": "s39-cp0-r1-artifact-root-v2.4",
        }
        self.artifact_path = self.write("artifact.json", self.root_value)
        self.contract_path = self.root / "contract.json"
        self.candidate_path = self.write("candidate.json", self.candidate)
        corpus_rows = [
            {
                "choices": ["A", "B", "C", "D"],
                "dataset": "cais/mmlu",
                "dataset_revision": marker("revision")[:40],
                "expected_answer": "A",
                "item_index": index,
                "question": f"Question {index}",
                "source_row": index,
                "subject": "subject",
            }
            for index in range(64)
        ]
        self.corpus_raw = b"".join(canonical(value) for value in corpus_rows)
        self.corpus_path = self.root / "corpus.jsonl"
        self.corpus_path.write_bytes(self.corpus_raw)
        self.contract["quality_corpus"] = {
            "bytes": len(self.corpus_raw),
            "sha256": hashlib.sha256(self.corpus_raw).hexdigest(),
        }
        self.contract_path.write_bytes(canonical(self.contract))

    def write(self, name, value):
        path = self.root / name
        path.write_bytes(canonical(value))
        return path


class ProductionPlanTests(unittest.TestCase):
    def test_repo_root_is_exact(self):
        self.assertEqual(materializer.REPO, REPO)
        self.assertTrue((materializer.REPO / "AGENTS.md").is_file())
        self.assertTrue((materializer.REPO / "ggml").is_dir())

    def test_int64_and_v23_operational_bindings_fail_closed(self):
        with self.assertRaisesRegex(common.ProductionError, "E_INTEGER"):
            common.integer(1 << 63, "overflow")
        with self.assertRaisesRegex(common.ProductionError, "E_INTEGER"):
            common.integer(True, "bool")
        with self.assertRaisesRegex(materializer.common.ProductionError, "E_STALE_V23"):
            materializer._no_v23(
                {"entrypoint": "/tmp/v23_readiness/stale.py"},
                "runtime",
            )

    def test_missing_inputs_are_named_once_in_sorted_order(self):
        root = Path("/nonexistent/v24")
        with self.assertRaisesRegex(
            materializer.common.ProductionError,
            r"E_PRODUCTION_INPUTS_MISSING: alpha=.*alpha,beta=.*beta",
        ):
            materializer._missing(
                {"beta": root / "beta", "alpha": root / "alpha"}
            )

    def test_contract_drives_exact_stage_support_sets(self):
        contract = builder.build_contract()
        authority, stages = materializer._support_paths(contract)
        requirements = contract["orchestration_requirements"]
        for stage, names in requirements["stage_support"].items():
            expected = sorted(
                str((S39 / requirements["support"][name]["path"]).resolve())
                for name in names
            )
            self.assertEqual(stages[stage], expected)
        self.assertEqual(
            authority,
            [
                str((S39 / record["path"]).resolve())
                for _, record in sorted(contract["exit_authority"]["support"].items())
            ],
        )

    def test_topology_rejects_wrong_direct_peer_target_and_adb(self):
        values = {
            "contract": {"phase_protocol": {"phase": "A_ONLY"}},
            "runtime_plan": {"phase": "A_ONLY"},
            "phone_route_launch": {
                "mechanism_commands": {"desktop": [["adb", "-P", "5038"]]},
                "phones": {
                    "op12": {
                        "boot_id": common.UNBOUND_BOOT_IDS["op12"],
                        "direct_peer_ipv4": common.UNBOUND_PHONE_NETWORK["op15"]["local_ipv4"],
                        **common.UNBOUND_PHONE_NETWORK["op12"],
                    },
                    "op15": {
                        "boot_id": common.UNBOUND_BOOT_IDS["op15"],
                        "direct_peer_ipv4": common.UNBOUND_PHONE_NETWORK["op12"]["local_ipv4"],
                        **common.UNBOUND_PHONE_NETWORK["op15"],
                    },
                },
            },
            "cuda_route_launch": {},
            "joint_capture_plan": {"phase": "A_ONLY"},
            "prospective_root": {
                "desktop_control": {
                    "cuda_ssh_target": common.CUDA_SSH_TARGET,
                    "phone_adb_port": common.PHONE_ADB_PORT,
                },
                "identity_placeholders": common.UNBOUND_BOOT_IDS,
                "network_placeholders": common.UNBOUND_PHONE_NETWORK,
            },
        }
        materializer._validate_topology(values)
        for field, mutate, error in (
            (
                "peer",
                lambda item: item["phone_route_launch"]["phones"]["op15"].update(
                    direct_peer_ipv4="127.0.0.1"
                ),
                "direct.op15_to_op12",
            ),
            (
                "target",
                lambda item: item["prospective_root"]["desktop_control"].update(
                    cuda_ssh_target="wrong@host"
                ),
                "E_CUDA_SSH_TARGET_UNBOUND",
            ),
            (
                "adb",
                lambda item: item["phone_route_launch"].update(
                    mechanism_commands={"desktop": [["adb", "-P", "5037"]]}
                ),
                "E_PHONE_ADB_PORT_UNBOUND",
            ),
        ):
            with self.subTest(field=field):
                mutated = copy.deepcopy(values)
                mutate(mutated)
                with self.assertRaisesRegex(materializer.common.ProductionError, error):
                    materializer._validate_topology(mutated)

    def test_preparation_polls_delayed_boot_and_preserves_swap_baselines(self):
        fixture = Fixture(self)
        runner = FakeRunner(delayed=True, transient_offline=True)
        output = fixture.root / "preparation.json"
        value = preparation.prepare(
            output=output,
            contract_path=fixture.contract_path,
            artifact_root_path=fixture.artifact_path,
            runtime_plan_path=fixture.plan_path,
            confirmation=preparation.CONFIRMATION,
            timeout_seconds=60,
            runner=runner,
            clock_ns=Clock(),
            monotonic=Monotonic(),
            sleep=lambda _: None,
            poll_seconds=0.1,
        )
        self.assertGreaterEqual(runner.status_calls["3C15AU002CL00000"], 2)
        self.assertGreaterEqual(runner.status_calls["5ae7a43d"], 2)
        self.assertEqual(value["devices"]["cuda"]["system_swap_used_bytes"], 4096 * 1024)
        self.assertEqual(value["devices"]["op15"]["system_swap_used_bytes"], 2048 * 1024)
        self.assertEqual(value["before_boot_ids"], {
            "op15": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "op12": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        })
        self.assertEqual(value["devices"]["op15"]["local_ipv4"], "10.0.0.15")

    def test_preparation_boot_poll_timeout_is_fatal(self):
        fixture = Fixture(self)
        runner = FakeRunner(delayed=True)
        with self.assertRaisesRegex(preparation.common.ProductionError, "E_PHONE_BOOT_TIMEOUT"):
            preparation.prepare(
                output=fixture.root / "preparation.json",
                contract_path=fixture.contract_path,
                artifact_root_path=fixture.artifact_path,
                runtime_plan_path=fixture.plan_path,
                confirmation=preparation.CONFIRMATION,
                timeout_seconds=60,
                runner=runner,
                clock_ns=Clock(),
                monotonic=Monotonic(step=61),
                sleep=lambda _: None,
            )

    def test_phase_lock_raw_roles_are_deterministic_and_ordered(self):
        def run_once(suffix):
            fixture = Fixture(self)
            preparation_value = {
                "artifact_root_sha256": hashlib.sha256(
                    fixture.artifact_path.read_bytes()
                ).hexdigest(),
                "completed_ns": 200,
                "devices": {
                    "cuda": {"host_boot_id": "11111111-1111-4111-8111-111111111111"},
                    "op12": {"boot_id": "dddddddd-dddd-4ddd-8ddd-dddddddddddd"},
                    "op15": {"boot_id": "cccccccc-cccc-4ccc-8ccc-cccccccccccc"},
                },
                "runtime_bundle_plan_sha256": hashlib.sha256(
                    fixture.plan_path.read_bytes()
                ).hexdigest(),
                "schema": "s39-cp0-r1-reboot-preparation-v2.4",
            }
            preparation_path = fixture.write("preparation.json", preparation_value)
            pre_dir = fixture.root / f"pre-{suffix}"
            pre_dir.mkdir()
            output = pre_dir / "phase_lock.jsonl"
            runner = FakeRunner()
            phase_lock.materialize(
                output=output,
                phase_id="cp0-r1-v24-a-only-test",
                contract_path=fixture.contract_path,
                candidate_path=fixture.candidate_path,
                artifact_root_path=fixture.artifact_path,
                preparation_path=preparation_path,
                quality_corpus_path=fixture.corpus_path,
                runtime_plan_path=fixture.plan_path,
                pre_dir=pre_dir,
                confirmation=phase_lock.CONFIRMATION,
                timeout_seconds=60,
                runner=runner,
                clock_ns=Clock(start=990),
            )
            raws = {
                path.name: path.read_bytes()
                for path in sorted((pre_dir / "raw").iterdir())
            }
            return output.read_bytes(), raws

        first_lock, first = run_once("a")
        second_lock, second = run_once("b")
        self.assertEqual(first_lock, second_lock)
        self.assertEqual(first, second)
        lock_event = json.loads(first_lock)["event_ns"]
        preflight_events = [
            json.loads(line)["event_ns"]
            for line in first["phase-preflight.jsonl"].splitlines()
        ]
        route_event = json.loads(first["route-lock.jsonl"])["event_ns"]
        corpus_events = [
            json.loads(line)["event_ns"]
            for line in first["quality-corpus.jsonl"].splitlines()
        ]
        self.assertLessEqual(max(corpus_events + [route_event]), lock_event)
        self.assertLess(lock_event, min(preflight_events))

    def test_fan_in_accepts_precreated_empty_bundle_root(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        bundle = root / "bundle"
        bundle.mkdir()
        missing = root / "missing.json"
        kwargs = {
            "runtime_output": root / "runtime.json",
            "acquisition_output": root / "acquisition.json",
            "bundle_root": bundle,
            "pre_dir": root / "pre",
            "acquisition_started_ns": 1,
            "contract_path": missing,
            "candidate_path": missing,
            "runtime_plan_path": missing,
            "artifact_root_path": missing,
            "preparation_path": missing,
            "phase_lock_path": missing,
            "fresh_path": missing,
            "cuda_monolithic_path": missing,
            "joint_phone_cuda_path": missing,
        }
        with self.assertRaisesRegex(fan_in.common.ProductionError, "E_READ"):
            fan_in.fan_in(**kwargs)
        (bundle / "occupied").write_bytes(b"x")
        with self.assertRaisesRegex(fan_in.common.ProductionError, "E_BUNDLE_ROOT_NOT_EMPTY"):
            fan_in.fan_in(**kwargs)

    def test_stale_valid_pyc_cannot_override_mutated_support_source(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / "production_common_v1.py"
        stage = root / "phase_lock_v1.py"
        shutil.copy2(SUBJECT_ROOT / "production_common_v1.py", source)
        shutil.copy2(SUBJECT_ROOT / "phase_lock_v1.py", stage)
        metadata = source.stat()
        py_compile.compile(str(source), doraise=True)
        raw = source.read_bytes()
        self.assertIn(b"PHONE_ADB_PORT = 5038", raw)
        mutated = raw.replace(b"PHONE_ADB_PORT = 5038", b"PHONE_ADB_PORT = 5999")
        self.assertEqual(len(mutated), len(raw))
        source.write_bytes(mutated)
        os.utime(source, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        namespace = runpy.run_path(str(stage), run_name="stale_pyc_probe")
        self.assertEqual(namespace["common"].PHONE_ADB_PORT, 5999)


if __name__ == "__main__":
    unittest.main()
