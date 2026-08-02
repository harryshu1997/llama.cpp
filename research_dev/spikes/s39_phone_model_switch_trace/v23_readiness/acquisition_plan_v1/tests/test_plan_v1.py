#!/usr/bin/env python3

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
PLAN_DIR = HERE.parent
sys.path.insert(0, str(PLAN_DIR))

import plan_common_v1 as common


def marker(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class NestedPlanFixture:
    def __init__(self, root):
        self.root = root
        self.model_sha256 = marker("model")
        self.contract_sha256 = marker("contract")
        self.candidate_sha256 = marker("candidate")
        self.entrypoints = {}
        for name in ("joint", "monolithic", "phone", "cuda"):
            path = root / f"{name}.py"
            path.write_text("#!/usr/bin/python3 -I\nimport json\n", encoding="ascii")
            path.chmod(0o755)
            self.entrypoints[name] = path
        self.relay_probe = root / "remote-android-process-probe.py"
        self.relay_probe.write_text(
            "#!/usr/bin/python3 -I\nimport json\n",
            encoding="ascii",
        )
        self.relay_probe.chmod(0o755)
        self.adb = root / "adb"
        self.adb.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.adb.chmod(0o755)
        self.matrix = {
            "desktop": [[f"/bin/tool-{index}"] for index in range(9)],
            "op12": [["/bin/op12"]],
            "op15": [["/bin/op15"]],
        }
        self.histories_path = root / "histories.json"
        self.write(self.histories_path, {
            "histories": [[index, index + 1] for index in range(8)],
            "history_width": 2,
            "model_id": common.MODEL_ID,
            "model_sha256": self.model_sha256,
            "request_ids": list(range(8)),
            "route_epoch": 7,
            "schema": common.HISTORY_SCHEMA,
        })
        self.phone_path = root / "phone-launch.json"
        self.cuda_path = root / "cuda-launch.json"
        self.monolithic_path = root / "monolithic-launch.json"
        common_launch = {
            "expected_file_type": 15,
            "expected_max_streams": 8,
            "expected_n_batch": 64,
            "expected_n_ctx_seq": 256,
            "expected_n_embd": 5120,
            "expected_n_layer": 40,
            "expected_n_ubatch": 64,
            "model_id": common.MODEL_ID,
            "model_sha256": self.model_sha256,
        }
        history_sha256 = marker_bytes(self.histories_path.read_bytes())
        self.write(self.phone_path, {
            **common_launch,
            "codec": {},
            "history_path": str(self.histories_path),
            "history_sha256": history_sha256,
            "mechanism_commands": copy.deepcopy(self.matrix),
            "phones": {},
            "probes": {},
            "processes": {},
            "quality_corpus_content_sha256": marker("corpus"),
            "relay_process_probe": {
                "argv": [
                    str(self.relay_probe),
                    "--adb",
                    str(self.adb),
                    "--adb-port",
                    "5038",
                    "--adb-selector",
                    "192.0.2.15:5555",
                    "--adb-sha256",
                    marker_bytes(self.adb.read_bytes()),
                ],
                "cwd": str(root),
                "environment": {},
                "expected_argv": ["/data/local/tmp/relay", "--listen", "9001"],
                "expected_executable_path": "/data/local/tmp/relay",
                "expected_port": 9001,
                "launcher_bytes": self.relay_probe.stat().st_size,
                "launcher_sha256": marker_bytes(
                    self.relay_probe.read_bytes()
                ),
                "timeout_ms": 1000,
            },
            "relay_host": "127.0.0.1",
            "relay_port": 9001,
            "route_epoch": 7,
            "schema": common.PHONE_LAUNCH_SCHEMA,
        })
        self.write(self.cuda_path, {
            **common_launch,
            "codec": {},
            "expected_capabilities": 63,
            "history_path": str(self.histories_path),
            "history_sha256": history_sha256,
            "host": "127.0.0.1",
            "io_timeout_ms": 1000,
            "mechanism_commands": copy.deepcopy(self.matrix),
            "model_artifact": {},
            "nvidia_smi": {},
            "port": 9002,
            "quality_corpus_content_sha256": marker("corpus"),
            "route_epoch": 7,
            "schema": common.CUDA_LAUNCH_SCHEMA,
            "worker": {},
        })
        self.write(self.monolithic_path, {
            **common_launch,
            "command": copy.deepcopy(self.matrix["desktop"][8]),
            "cwd": str(root),
            "env": {"LC_ALL": "C"},
            "expected_capabilities": 63,
            "host": "127.0.0.1",
            "io_timeout_ms": 1000,
            "mechanism_commands": copy.deepcopy(self.matrix),
            "mechanism_commands_sha256": marker_bytes(
                common.canonical_bytes(self.matrix)
            ),
            "port": 9003,
            "schema": common.MONOLITHIC_LAUNCH_SCHEMA,
            "shutdown_timeout_ms": 1000,
            "startup_timeout_ms": 1000,
        })
        self.capture_path = root / "capture-plan.json"
        self.write(self.capture_path, {
            "commands": {
                "cuda": self.command(
                    self.entrypoints["cuda"],
                    {
                        "--histories": self.histories_path,
                        "--launch-plan": self.cuda_path,
                    },
                    "cuda.json",
                ),
                "phone": self.command(
                    self.entrypoints["phone"],
                    {
                        "--histories": self.histories_path,
                        "--launch-plan": self.phone_path,
                    },
                    "phone.json",
                ),
            },
            "mechanism_commands": copy.deepcopy(self.matrix),
            "model_id": common.MODEL_ID,
            "model_sha256": self.model_sha256,
            "phase": common.PHASE,
            "schema": common.JOINT_PLAN_SCHEMA,
        })
        self.command_path = root / "command-plan.json"
        self.command_plan = {
            "candidate_sha256": self.candidate_sha256,
            "contract_sha256": self.contract_sha256,
            "mechanism_commands": copy.deepcopy(self.matrix),
            "model_id": common.MODEL_ID,
            "model_sha256": self.model_sha256,
            "outputs": {
                **common.PAYLOAD_ROLES,
                "runtime_bundle_identity": "runtime_bundle_identity.json",
                "runtime_identity": "runtime_identity.json",
            },
            "phase": common.PHASE,
            "producers": {
                "cuda_monolithic": self.command(
                    self.entrypoints["monolithic"],
                    {
                        "--histories": self.histories_path,
                        "--launch-plan": self.monolithic_path,
                    },
                    "monolithic.json",
                ),
                "joint_phone_cuda": self.command(
                    self.entrypoints["joint"],
                    {"--capture-plan": self.capture_path},
                    "joint.json",
                ),
            },
            "schema": common.COMMAND_PLAN_SCHEMA,
        }
        self.write(self.command_path, self.command_plan)

    @staticmethod
    def write(path, value):
        path.write_bytes(common.canonical_bytes(value))

    @staticmethod
    def record(path, index):
        raw = path.read_bytes()
        return {
            "argv_index": index,
            "bytes": len(raw),
            "path": str(path),
            "sha256": marker_bytes(raw),
        }

    def command(self, entrypoint, files, result):
        argv = [str(entrypoint)]
        records = [self.record(entrypoint, 0)]
        for flag, path in sorted(files.items()):
            argv.extend([flag, str(path)])
            records.append(self.record(path, len(argv) - 1))
        argv.extend([
            "--output",
            "{output_path}",
            "--phase-id",
            "{phase_id}",
            "--pre-dir",
            "{pre_dir}",
            "--acquisition-started-ns",
            "{acquisition_started_ns}",
            "--command-plan-sha256",
            "{command_plan_sha256}",
        ])
        records.sort(key=lambda value: value["argv_index"])
        return {
            "argv_template": argv,
            "executed_files": records,
            "result_filename": result,
            "timeout_seconds": 60,
        }


def marker_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


class AcquisitionPlanV1Tests(unittest.TestCase):
    def test_inline_plan_requires_matching_digest(self):
        raw = '{"schema":"test"}'
        with self.assertRaisesRegex(
            common.PlanError,
            "plan-sha256.count",
        ):
            common.parse_inline_plan(
                ["/launcher", "--plan-json", raw],
                "command",
            )
        with self.assertRaisesRegex(
            common.PlanError,
            "plan-sha256",
        ):
            common.parse_inline_plan(
                [
                    "/launcher",
                    "--plan-json",
                    raw,
                    "--plan-sha256",
                    "0" * 64,
                ],
                "command",
            )
        self.assertEqual(
            common.parse_inline_plan(
                [
                    "/launcher",
                    "--plan-json",
                    raw,
                    "--plan-sha256",
                    hashlib.sha256(raw.encode("ascii")).hexdigest(),
                ],
                "command",
            ),
            {"schema": "test"},
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def test_frozen_candidate_contract_and_roles(self):
        contract, contract_raw, candidate_raw = common.validate_frozen_contract()
        self.assertEqual(common.sha256_bytes(contract_raw), common.CONTRACT_SHA256)
        self.assertEqual(common.sha256_bytes(candidate_raw), common.CANDIDATE_SHA256)
        roles = contract["phase_protocol"]["phase_roles"]["A_ONLY"]
        self.assertEqual(
            set(roles) - {
                "phase.lock",
                "phase.preflight",
                "quality.corpus",
                f"model.{common.MODEL_ID}.route_lock",
            },
            set(common.PAYLOAD_ROLES),
        )

    def test_existing_worker_path_is_exact(self):
        self.assertEqual(
            common.WORKER_PATH,
            "/data/local/tmp/s39-active-warm/v1/"
            "runtime-qwen-partial-v1/llama-layersplit",
        )

    def test_historical_executable_only_readiness_cannot_authorize(self):
        common.verify_historical_readiness_sources()
        with self.assertRaisesRegex(
            common.PlanError,
            "E_PRODUCTION_NOT_READY.*runtime_bundle",
        ):
            common.build_plan()

    def test_unfrozen_producers_are_named_blockers(self):
        blockers = common.production_blockers()
        self.assertIn(
            "absent:research_dev/spikes/s39_phone_model_switch_trace/"
            "v23_readiness/a_only_acquisition_driver_v1/"
            "A_ONLY_COMMAND_PLAN_V1.json",
            blockers,
        )
        self.assertIn(
            "not_frozen:research_dev/spikes/s39_phone_model_switch_trace/"
            "v23_readiness/a_only_acquisition_driver_v1/producers_v1/"
            "joint_phone_cuda_v1.py",
            blockers,
        )
        self.assertIn(
            "not_frozen:research_dev/spikes/s39_phone_model_switch_trace/"
            "v23_readiness/a_only_acquisition_driver_v1/producers_v1/"
            "cuda_monolithic_v1.py",
            blockers,
        )

    def test_builder_refuses_without_partial_outputs_or_traceback(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(PLAN_DIR / "build_a_only_plan_v1.py"),
                "--output-dir",
                str(self.root),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("A_ONLY_PLAN_BUILD_REFUSED", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_validator_refuses_without_traceback(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(PLAN_DIR / "validate_a_only_plan_v1.py"),
                "--plan",
                str(self.root / "plan.json"),
                "--manifest",
                str(self.root / "manifest.txt"),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("A_ONLY_PLAN_VALIDATE_REFUSED", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_manifest_rejects_duplicate_escape_and_pycache(self):
        digest = "a" * 64
        invalid = (
            f"{digest}  a.py\n{digest}  a.py\n",
            f"{digest}  ../a.py\n",
            f"{digest}  x/__pycache__/a.pyc\n",
            f"{digest}  z.py\n{digest}  a.py\n",
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(common.PlanError):
                    common.parse_manifest(raw.encode("ascii"))

    def test_unbound_python_import_is_rejected(self):
        safe = self.root / "safe.py"
        unsafe = self.root / "unsafe.py"
        safe.write_text("import json\nfrom pathlib import Path\n", encoding="ascii")
        unsafe.write_text("import local_runtime_support\n", encoding="ascii")
        common.verify_no_unbound_python_imports([safe])
        with self.assertRaisesRegex(common.PlanError, "E_UNBOUND_IMPORT"):
            common.verify_no_unbound_python_imports([unsafe])

    def test_driver_spec_rejects_mutations(self):
        driver = self.root / "driver.py"
        driver.write_text("#!/usr/bin/python3 -I\n", encoding="ascii")
        driver.chmod(0o755)
        spec = {
            "argv_template": [
                str(driver),
                "--phase-id",
                "{phase_id}",
                "--pre",
                "{pre_dir}",
                "--output",
                "{output_dir}",
            ],
            "executed_files": [common.executed_file(driver, 0)],
            "timeout_seconds": 60,
        }
        common.validate_driver_spec(spec, "driver")
        cases = []
        value = copy.deepcopy(spec)
        value["executed_files"][0]["argv_index"] = 99
        cases.append(value)
        value = copy.deepcopy(spec)
        value["executed_files"][0]["sha256"] = "bad"
        cases.append(value)
        value = copy.deepcopy(spec)
        value["timeout_seconds"] = 0
        cases.append(value)
        value = copy.deepcopy(spec)
        value["unknown"] = True
        cases.append(value)
        for index, value in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaises(common.PlanError):
                    common.validate_driver_spec(value, "driver")

    def test_driver_specs_exact_bind_all_executable_file_flags(self):
        source = self.root / "source.py"
        source.write_text("#!/usr/bin/python3 -I\n", encoding="ascii")
        for name in ("artifact", "fresh", "acquisition"):
            with self.subTest(name=name):
                argv = [str(source)]
                records = [common.executed_file(source, 0)]
                for flag in sorted(common.DRIVER_FILE_FLAGS[name]):
                    argv.extend([flag, str(source)])
                    records.append(common.executed_file(source, len(argv) - 1))
                spec = {
                    "argv_template": argv,
                    "executed_files": records,
                    "timeout_seconds": 60,
                }
                common.validate_driver_spec(spec, name)
                missing = copy.deepcopy(spec)
                missing["executed_files"].pop()
                with self.assertRaisesRegex(
                    common.PlanError,
                    "executed_file_indexes",
                ):
                    common.validate_driver_spec(missing, name)
                duplicate_flag = copy.deepcopy(spec)
                duplicate_flag["argv_template"].extend(
                    [sorted(common.DRIVER_FILE_FLAGS[name])[0], str(source)]
                )
                with self.assertRaisesRegex(common.PlanError, "count"):
                    common.validate_driver_spec(duplicate_flag, name)
                extra = self.root / f"{name}-extra.json"
                extra.write_text("{}\n", encoding="ascii")
                unbound = copy.deepcopy(spec)
                unbound["argv_template"].extend(
                    ["--extra-input", str(extra)]
                )
                with self.assertRaisesRegex(
                    common.PlanError,
                    "executed_file_indexes",
                ):
                    common.validate_driver_spec(unbound, name)
                bound = copy.deepcopy(unbound)
                bound["executed_files"].append(
                    common.executed_file(
                        extra,
                        len(bound["argv_template"]) - 1,
                    )
                )
                common.validate_driver_spec(bound, name)

    def test_nested_execution_graph_validates_all_plans(self):
        fixture = NestedPlanFixture(self.root)
        graph = common.validate_nested_execution_graph(
            fixture.command_path,
            fixture.contract_sha256,
            fixture.candidate_sha256,
        )
        self.assertEqual(graph["command_matrix"], fixture.matrix)
        self.assertEqual(
            set(graph["plan_paths"]),
            {
                fixture.command_path,
                fixture.capture_path,
                fixture.histories_path,
                fixture.phone_path,
                fixture.cuda_path,
                fixture.monolithic_path,
            },
        )
        self.assertTrue(set(fixture.entrypoints.values()).issubset(graph["executed_paths"]))
        self.assertEqual(
            graph["runtime_dependency_paths"],
            [fixture.relay_probe, fixture.adb],
        )

    def test_nested_execution_graph_rejects_matrix_drift(self):
        fixture = NestedPlanFixture(self.root)
        phone = json.loads(fixture.phone_path.read_text(encoding="ascii"))
        phone["mechanism_commands"]["op12"][0].append("--drift")
        fixture.write(fixture.phone_path, phone)
        capture = json.loads(fixture.capture_path.read_text(encoding="ascii"))
        command = capture["commands"]["phone"]
        record = next(
            item
            for item in command["executed_files"]
            if item["path"] == str(fixture.phone_path)
        )
        record["bytes"] = fixture.phone_path.stat().st_size
        record["sha256"] = marker_bytes(fixture.phone_path.read_bytes())
        fixture.write(fixture.capture_path, capture)
        top = json.loads(fixture.command_path.read_text(encoding="ascii"))
        record = next(
            item
            for item in top["producers"]["joint_phone_cuda"]["executed_files"]
            if item["path"] == str(fixture.capture_path)
        )
        record["bytes"] = fixture.capture_path.stat().st_size
        record["sha256"] = marker_bytes(fixture.capture_path.read_bytes())
        fixture.write(fixture.command_path, top)
        with self.assertRaisesRegex(common.PlanError, "phone_launch.mechanism_commands"):
            common.validate_nested_execution_graph(
                fixture.command_path,
                fixture.contract_sha256,
                fixture.candidate_sha256,
            )

    def test_nested_execution_graph_rejects_non_shared_histories(self):
        fixture = NestedPlanFixture(self.root)
        alternate = self.root / "alternate-histories.json"
        alternate.write_bytes(fixture.histories_path.read_bytes())
        capture = json.loads(fixture.capture_path.read_text(encoding="ascii"))
        command = capture["commands"]["cuda"]
        flag_index = command["argv_template"].index("--histories")
        value_index = flag_index + 1
        command["argv_template"][value_index] = str(alternate)
        record = next(
            item
            for item in command["executed_files"]
            if item["argv_index"] == value_index
        )
        record.update(self._binding_values(alternate, value_index))
        fixture.write(fixture.capture_path, capture)
        self._refresh_capture_binding(fixture)
        with self.assertRaisesRegex(common.PlanError, "E_SHARED_HISTORIES_PATH"):
            common.validate_nested_execution_graph(
                fixture.command_path,
                fixture.contract_sha256,
                fixture.candidate_sha256,
            )

    def test_nested_execution_graph_rejects_nested_schema_mutation(self):
        fixture = NestedPlanFixture(self.root)
        monolithic = json.loads(
            fixture.monolithic_path.read_text(encoding="ascii")
        )
        monolithic["schema"] = "wrong"
        fixture.write(fixture.monolithic_path, monolithic)
        top = json.loads(fixture.command_path.read_text(encoding="ascii"))
        command = top["producers"]["cuda_monolithic"]
        record = next(
            item
            for item in command["executed_files"]
            if item["path"] == str(fixture.monolithic_path)
        )
        record.update(self._binding_values(
            fixture.monolithic_path,
            record["argv_index"],
        ))
        fixture.write(fixture.command_path, top)
        with self.assertRaisesRegex(common.PlanError, "monolithic_launch.schema"):
            common.validate_nested_execution_graph(
                fixture.command_path,
                fixture.contract_sha256,
                fixture.candidate_sha256,
            )

    def test_nested_execution_graph_requires_every_file_flag_binding(self):
        cases = (
            ("top-joint", "top", "joint_phone_cuda", "--capture-plan"),
            ("top-mono-history", "top", "cuda_monolithic", "--histories"),
            ("top-mono-launch", "top", "cuda_monolithic", "--launch-plan"),
            ("joint-phone-history", "joint", "phone", "--histories"),
            ("joint-phone-launch", "joint", "phone", "--launch-plan"),
            ("joint-cuda-history", "joint", "cuda", "--histories"),
            ("joint-cuda-launch", "joint", "cuda", "--launch-plan"),
        )
        for label, level, command_name, flag in cases:
            with self.subTest(label=label):
                root = self.root / label
                root.mkdir()
                fixture = NestedPlanFixture(root)
                if level == "top":
                    plan_path = fixture.command_path
                    plan = json.loads(plan_path.read_text(encoding="ascii"))
                    command = plan["producers"][command_name]
                else:
                    plan_path = fixture.capture_path
                    plan = json.loads(plan_path.read_text(encoding="ascii"))
                    command = plan["commands"][command_name]
                value_index = command["argv_template"].index(flag) + 1
                command["executed_files"] = [
                    record
                    for record in command["executed_files"]
                    if record["argv_index"] != value_index
                ]
                fixture.write(plan_path, plan)
                if level == "joint":
                    self._refresh_capture_binding(fixture)
                with self.assertRaisesRegex(
                    common.PlanError,
                    "executed_file_indexes|E_EXECUTED_FILE_MISSING",
                ):
                    common.validate_nested_execution_graph(
                        fixture.command_path,
                        fixture.contract_sha256,
                        fixture.candidate_sha256,
                    )

    def test_nested_relay_probe_source_and_adb_are_digest_bound(self):
        for name in ("relay-probe", "adb"):
            with self.subTest(name=name):
                root = self.root / name
                root.mkdir()
                fixture = NestedPlanFixture(root)
                target = (
                    fixture.relay_probe
                    if name == "relay-probe"
                    else fixture.adb
                )
                target.write_bytes(target.read_bytes() + b"changed\n")
                with self.assertRaisesRegex(
                    common.PlanError,
                    "E_SOURCE_SHA256",
                ):
                    common.validate_nested_execution_graph(
                        fixture.command_path,
                        fixture.contract_sha256,
                        fixture.candidate_sha256,
                    )

    def test_manifest_root_rejects_digest_and_symlink_drift(self):
        source_root = self.root / "root"
        source_dir = source_root / "sources"
        source_dir.mkdir(parents=True)
        source = source_dir / "entry.py"
        source.write_bytes(b"one\n")
        manifest = source_root / "manifest.txt"
        relative = "sources/entry.py"
        manifest.write_text(
            f"{marker_bytes(source.read_bytes())}  {relative}\n",
            encoding="ascii",
        )
        common.validate_manifest_at_root(manifest, source_root, [relative])
        binding = {
            "source_manifest_path": str(manifest),
            "source_manifest_sha256": marker_bytes(manifest.read_bytes()),
            "source_root": str(source_root),
        }
        with (
            mock.patch.object(common, "REPO_ROOT", source_root),
            mock.patch.object(common, "MANIFEST_PATH", manifest),
        ):
            common.validate_bound_source_manifest(binding, [source])
            mutated = copy.deepcopy(binding)
            mutated["source_manifest_sha256"] = marker("wrong")
            with self.assertRaisesRegex(
                common.PlanError,
                "source_manifest_sha256",
            ):
                common.validate_bound_source_manifest(mutated, [source])
        original_manifest = manifest.read_bytes()
        with self.assertRaisesRegex(
            common.PlanError,
            "E_MANIFEST_SOURCE_SET",
        ):
            common.validate_manifest_at_root(
                manifest,
                source_root,
                [relative, "sources/missing.py"],
            )
        extra = source_root / "extra.py"
        extra.write_bytes(b"extra\n")
        manifest.write_bytes(
            (
                f"{marker_bytes(extra.read_bytes())}  extra.py\n"
            ).encode("ascii")
            + original_manifest
        )
        with self.assertRaisesRegex(
            common.PlanError,
            "E_MANIFEST_SOURCE_SET",
        ):
            common.validate_manifest_at_root(
                manifest,
                source_root,
                [relative],
            )
        manifest.write_bytes(original_manifest)
        source.write_bytes(b"two\n")
        with self.assertRaisesRegex(common.PlanError, "E_SOURCE_SHA256"):
            common.validate_manifest_at_root(manifest, source_root, [relative])
        source.write_bytes(b"one\n")
        source.unlink()
        outside = self.root / "outside.py"
        outside.write_bytes(b"one\n")
        source.symlink_to(outside)
        with self.assertRaisesRegex(common.PlanError, "E_SOURCE_SYMLINK"):
            common.validate_manifest_at_root(manifest, source_root, [relative])
        source.unlink()
        source_dir.rmdir()
        outside_dir = self.root / "outside"
        outside_dir.mkdir()
        (outside_dir / "entry.py").write_bytes(b"one\n")
        source_dir.symlink_to(outside_dir, target_is_directory=True)
        with self.assertRaisesRegex(common.PlanError, "E_SOURCE_SYMLINK"):
            common.validate_manifest_at_root(manifest, source_root, [relative])

    def test_source_binding_field_names_are_exact(self):
        self.assertEqual(
            common.SOURCE_BINDING_KEYS,
            {
                "source_manifest_path",
                "source_manifest_sha256",
                "source_root",
            },
        )

    @staticmethod
    def _binding_values(path, index):
        raw = path.read_bytes()
        return {
            "argv_index": index,
            "bytes": len(raw),
            "path": str(path),
            "sha256": marker_bytes(raw),
        }

    def _refresh_capture_binding(self, fixture):
        top = json.loads(fixture.command_path.read_text(encoding="ascii"))
        record = next(
            item
            for item in top["producers"]["joint_phone_cuda"]["executed_files"]
            if item["path"] == str(fixture.capture_path)
        )
        record.update(self._binding_values(
            fixture.capture_path,
            record["argv_index"],
        ))
        fixture.write(fixture.command_path, top)

    def test_digest_and_symlink_drift_are_rejected(self):
        source = self.root / "source"
        source.write_bytes(b"one\n")
        digest = common.sha256_bytes(source.read_bytes())
        common.verify_digest(source, digest)
        source.write_bytes(b"two\n")
        with self.assertRaisesRegex(common.PlanError, "E_SOURCE_SHA256"):
            common.verify_digest(source, digest)
        link = self.root / "link"
        link.symlink_to(source)
        with self.assertRaisesRegex(common.PlanError, "E_SOURCE_OPEN"):
            common.read_regular(link)


if __name__ == "__main__":
    unittest.main()
