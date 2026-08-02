#!/usr/bin/env python3

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "materialize_a_only_runtime_inputs_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "materialize_a_only_runtime_inputs_v1",
    SOURCE,
)
materialize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materialize)
PLAN_COMMON_SOURCE = SOURCE.parent / "plan_common_v1.py"
PLAN_COMMON_SPEC = importlib.util.spec_from_file_location(
    "plan_common_v1_for_materializer_test",
    PLAN_COMMON_SOURCE,
)
plan_common = importlib.util.module_from_spec(PLAN_COMMON_SPEC)
PLAN_COMMON_SPEC.loader.exec_module(plan_common)


def load_module(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


class Fixture:
    def __init__(self, test):
        self.temporary = tempfile.TemporaryDirectory()
        test.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.model_raw = b"test-model\n"
        self.model_sha256 = materialize.sha256(self.model_raw)
        self.model = self.root / "model" / "model.gguf"
        self._write(self.model, self.model_raw, False)
        self.constants = mock.patch.multiple(
            materialize,
            MODEL_BYTES=len(self.model_raw),
            MODEL_PATH=str(self.model),
            MODEL_SHA256=self.model_sha256,
        )
        self.constants.start()
        test.addCleanup(self.constants.stop)

        self.contract = self.root / "contract.json"
        self._json(
            self.contract,
            {"schema": "s39-cp0-r1-evidence-contract-v2.3"},
        )
        contract_sha256 = materialize.sha256(self.contract.read_bytes())
        self.candidate = self.root / "candidate.json"
        self._json(
            self.candidate,
            {
                "contract_sha256": contract_sha256,
                "models": [
                    {
                        "artifact": {
                            "bytes": len(self.model_raw),
                            "sha256": self.model_sha256,
                        },
                        "model_id": materialize.MODEL_ID,
                        "slot": "A",
                    }
                ],
            },
        )
        self.histories = self.root / "histories.json"
        self._json(
            self.histories,
            {
                "histories": [[index, index + 100] for index in range(8)],
                "history_width": 2,
                "model_id": materialize.MODEL_ID,
                "model_sha256": self.model_sha256,
                "request_ids": list(range(8)),
                "route_epoch": 7,
                "schema": materialize.HISTORY_SCHEMA,
            },
        )
        self.corpus = self.root / "quality-corpus.jsonl"
        corpus = bytearray()
        for index in range(64):
            corpus.extend(materialize.canonical_bytes({
                "acquisition_id": "phase",
                "choices": ["A", "B", "C", "D"],
                "dataset": "mmlu",
                "dataset_revision": "test",
                "expected_answer": index % 4,
                "item_index": index,
                "kind": "item",
                "phase": materialize.PHASE,
                "phase_id": "phase",
                "question": f"question-{index}",
                "role": "quality.corpus",
                "source_row": index,
                "subject": "test",
            }))
        self._write(self.corpus, bytes(corpus), False)

        self.sources = {}
        for key in materialize.SOURCE_KEYS:
            executable = key in materialize.EXECUTABLE_SOURCE_KEYS
            body = b"support\n"
            if key == "phone_producer":
                body = b'#!/usr/bin/env python3\nparser.add_argument("--histories")\n'
            elif executable:
                body = b"#!/usr/bin/env python3\n"
            path = self.root / "sources" / f"{key}.py"
            self._write(path, body, executable)
            self.sources[key] = str(path)

        self.runtime = {}
        runtime_names = {
            "adb_path": self.root / "tools" / "adb",
            "codec_path": self.root / "tools" / "codec",
            "cuda_launcher_path": self.root / "tools" / "cuda-launcher",
            "cuda_runtime_path": self.root / "cuda-route-runtime" / "cuda-route",
            "model_path": self.model,
            "monolithic_launcher_path": self.root / "tools" / "mono-launcher",
            "monolithic_runtime_path": self.root / "cuda-mono-runtime" / "cuda-mono",
            "nvidia_smi_path": self.root / "tools" / "nvidia-smi",
        }
        for key, path in runtime_names.items():
            if key != "model_path":
                self._write(path, f"#!/bin/sh\n# {key}\n".encode("ascii"), True)
            self.runtime[key] = str(path)
        self.cuda_route_library = self.root / "cuda-route-runtime" / "cuda-route.so"
        self.cuda_mono_library = self.root / "cuda-mono-runtime" / "cuda-mono.so"
        self._write(self.cuda_route_library, b"route-lib\n", False)
        self._write(self.cuda_mono_library, b"mono-lib\n", False)

        self.op12_root = self.root / "op12-runtime"
        self.op15_root = self.root / "op15-runtime"
        self.op12_worker = self.op12_root / "llama-layersplit"
        self.op15_worker = self.op15_root / "stage" / "llama-layersplit"
        self.op15_relay = self.op15_root / "relay" / "direct-relay"
        self.spec = self._make_spec()
        self.spec_path = self.root / "input.json"
        self.write_spec()

    def _write(self, path, raw, executable):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o755 if executable else 0o644)

    def _json(self, path, value):
        self._write(path, materialize.canonical_bytes(value), False)

    def _component(self, bundle, endpoint, component, path, role, raw=None):
        if raw is None:
            raw = (component + "\n").encode("ascii")
        path = Path(path)
        if path.is_file():
            metadata = path.stat(follow_symlinks=False)
            component_stat = {
                "build_id": None,
                **materialize.stat_record(metadata),
            }
        else:
            component_stat = {
                "build_id": None,
                "ctime_ns": 100,
                "device_id": 1,
                "inode": 1000 + len(component),
                "mode": (
                    0o100755 if role == "executable" else 0o100644
                ),
                "mtime_ns": 100,
                "size": len(raw),
            }
        return {
            "bundle_id": bundle,
            "bytes": len(raw),
            "component_id": component,
            "endpoint": endpoint,
            "path": str(path),
            "role": role,
            "sha256": materialize.sha256(raw),
            "stat": component_stat,
        }

    def _bundle(self, bundle, endpoint, launcher, required, role):
        return {
            "bundle_id": bundle,
            "endpoint": endpoint,
            "launcher_component_id": launcher,
            "process_role": role,
            "required_component_ids": sorted(required),
        }

    def _make_spec(self):
        cuda_route_runtime = Path(self.runtime["cuda_runtime_path"]).read_bytes()
        cuda_mono_runtime = Path(self.runtime["monolithic_runtime_path"]).read_bytes()
        route_lib = self.cuda_route_library.read_bytes()
        mono_lib = self.cuda_mono_library.read_bytes()
        components = [
            self._component(
                "cuda_monolithic", "cuda", "cuda_monolithic.launcher",
                self.runtime["monolithic_runtime_path"], "executable",
                cuda_mono_runtime,
            ),
            self._component(
                "cuda_monolithic", "cuda", "cuda_monolithic.library",
                self.cuda_mono_library, "shared_library", mono_lib,
            ),
            self._component(
                "cuda_route", "cuda", "cuda_route.launcher",
                self.runtime["cuda_runtime_path"], "executable",
                cuda_route_runtime,
            ),
            self._component(
                "cuda_route", "cuda", "cuda_route.library",
                self.cuda_route_library, "backend_library", route_lib,
            ),
            self._component(
                "op12_stagenet", "op12", "op12.launcher",
                self.op12_worker, "executable",
            ),
            self._component(
                "op12_stagenet", "op12", "op12.library",
                self.op12_root / "lib.so", "shared_library",
            ),
            self._component(
                "op15_direct_relay", "op15", "op15.relay",
                self.op15_relay, "executable",
            ),
            self._component(
                "op15_direct_relay", "op15", "op15.relay.library",
                self.op15_root / "relay" / "relay-lib.so", "shared_library",
            ),
            self._component(
                "op15_stagenet", "op15", "op15.launcher",
                self.op15_worker, "executable",
            ),
            self._component(
                "op15_stagenet", "op15", "op15.library",
                self.op15_root / "stage" / "stage-lib.so", "backend_library",
            ),
        ]
        bundles = [
            self._bundle(
                "cuda_monolithic", "cuda", "cuda_monolithic.launcher",
                ["cuda_monolithic.launcher", "cuda_monolithic.library"],
                "cuda_monolithic",
            ),
            self._bundle(
                "cuda_route", "cuda", "cuda_route.launcher",
                ["cuda_route.launcher", "cuda_route.library"], "cuda_route",
            ),
            self._bundle(
                "op12_stagenet", "op12", "op12.launcher",
                ["op12.launcher", "op12.library"], "stagenet_worker",
            ),
            self._bundle(
                "op15_direct_relay", "op15", "op15.relay",
                ["op15.relay", "op15.relay.library"], "direct_relay",
            ),
            self._bundle(
                "op15_stagenet", "op15", "op15.launcher",
                ["op15.launcher", "op15.library"], "stagenet_worker",
            ),
        ]
        shard_stat = {
            "ctime_ns": 100,
            "device_id": 1,
            "inode": 999,
            "mode": 0o100644,
            "mtime_ns": 100,
            "size": 123,
        }
        worker_route = {
            "devices": "HTP0",
            "driver_batch": 64,
            "driver_context": 256,
            "driver_max_prefill": 64,
            "dynamic_cut": True,
            "kv_unified": True,
            "n_gpu_layers": 99,
            "placement_cert": True,
        }
        telemetry = {
            "direct_peer_local_port": 13001,
            "direct_peer_port": 13002,
            "max_gpu_millic": 90000,
            "min_available_bytes": 1024,
        }
        return {
            "candidate_path": str(self.candidate),
            "contract_path": str(self.contract),
            "histories_path": str(self.histories),
            "network": {
                "cuda_monolithic_port": 12002,
                "cuda_route_port": 12001,
                "op12_stage_port": 12004,
                "op15_stage_port": 12005,
                "relay_host": "127.0.0.1",
                "relay_port": 12003,
                "relay_tail_source_port": 12006,
            },
            "phones": {
                "op12": {
                    "adb_selector": "192.0.2.12:5555",
                    "device": "op12-device",
                    "interface": "wlan0",
                    "local_ipv4": "10.0.0.12",
                    "model": "op12-model",
                    "product": "op12-product",
                    "relay_path": str(self.op12_root / "unused-relay"),
                    "serial": "op12-physical",
                    "shard_bytes": 123,
                    "shard_stat": shard_stat,
                    "worker_path": str(self.op12_worker),
                },
                "op15": {
                    "adb_selector": "192.0.2.15:5555",
                    "device": "op15-device",
                    "interface": "wlan0",
                    "local_ipv4": "10.0.0.15",
                    "model": "op15-model",
                    "product": "op15-product",
                    "relay_path": str(self.op15_relay),
                    "serial": "op15-physical",
                    "shard_bytes": 123,
                    "shard_stat": shard_stat,
                    "worker_path": str(self.op15_worker),
                },
            },
            "quality_corpus_path": str(self.corpus),
            "route_epoch": 7,
            "routes": {
                "cuda_environment": {"LANG": "C"},
                "op12_telemetry": telemetry,
                "op12_worker": worker_route,
                "op15_telemetry": telemetry,
                "op15_worker": worker_route,
                "relay": {
                    "head_host": "127.0.0.1",
                    "tail_host": "10.0.0.12",
                },
            },
            "runtime": self.runtime,
            "runtime_bundles": {
                "bundle_roots": {
                    "cuda_monolithic": str(self.root / "cuda-mono-runtime"),
                    "cuda_route": str(self.root / "cuda-route-runtime"),
                    "op12_stagenet": str(self.op12_root),
                    "op15_direct_relay": str(self.op15_root / "relay"),
                    "op15_stagenet": str(self.op15_root / "stage"),
                },
                "bundles": bundles,
                "closure_complete": True,
                "components": components,
                "schema": materialize.RUNTIME_INPUT_SCHEMA,
            },
            "schema": materialize.INPUT_SCHEMA,
            "sources": self.sources,
        }

    def write_spec(self):
        self._json(self.spec_path, self.spec)

    def output(self, name="bundle"):
        return self.root / name


class MaterializeRuntimeInputsTests(unittest.TestCase):
    def test_materializes_complete_deterministic_bundle(self):
        fixture = Fixture(self)
        output = fixture.output()
        first, manifest = materialize.build_bundle(fixture.spec_path, output)
        second, _ = materialize.build_bundle(fixture.spec_path, output)
        self.assertEqual(first, second)
        materialize.publish(first, output)
        self.assertEqual(
            set(path.name for path in output.iterdir()),
            {
                materialize.COMMAND_NAME,
                materialize.CUDA_NAME,
                materialize.HISTORIES_NAME,
                materialize.JOINT_NAME,
                materialize.MANIFEST_NAME,
                materialize.MONOLITHIC_NAME,
                materialize.PHONE_NAME,
                materialize.RUNTIME_NAME,
                materialize.SPECS_NAME,
            },
        )
        self.assertEqual(
            (output / materialize.HISTORIES_NAME).read_bytes(),
            fixture.histories.read_bytes(),
        )
        command = json.loads((output / materialize.COMMAND_NAME).read_bytes())
        phone = json.loads((output / materialize.PHONE_NAME).read_bytes())
        self.assertEqual(
            command["mechanism_commands"],
            phone["mechanism_commands"],
        )
        self.assertEqual(
            manifest["bindings"]["mechanism_commands_sha256"],
            materialize.sha256(
                materialize.canonical_bytes(command["mechanism_commands"])
            ),
        )
        relay_source = next(
            item
            for item in manifest["sources"]
            if item["role"] == "relay_process_probe"
        )
        self.assertEqual(
            relay_source["sha256"],
            materialize.sha256(
                Path(fixture.sources["relay_process_probe"]).read_bytes()
            ),
        )

    def test_validate_inputs_creates_no_output(self):
        fixture = Fixture(self)
        output = fixture.output()
        result = materialize.main([
            "--input-spec", str(fixture.spec_path),
            "--output-dir", str(output),
            "--validate-inputs",
        ])
        self.assertEqual(result, 0)
        self.assertFalse(output.exists())

    def test_inline_plans_are_typed_bound_and_capture_compatible(self):
        fixture = Fixture(self)
        outputs, manifest = materialize.build_bundle(
            fixture.spec_path,
            fixture.output(),
        )
        phone = json.loads(outputs[materialize.PHONE_NAME])
        cuda = json.loads(outputs[materialize.CUDA_NAME])
        monolithic = json.loads(outputs[materialize.MONOLITHIC_NAME])
        managed = {
            **phone["processes"],
            "cuda_route": cuda["worker"],
            "cuda_monolithic": {"argv": monolithic["command"]},
        }
        for name, command in managed.items():
            argv = command["argv"]
            self.assertNotIn("--plan", argv)
            self.assertNotIn("--boot-id", argv)
            self.assertEqual(argv.count("--plan-json"), 1)
            inline = argv[argv.index("--plan-json") + 1]
            inline_sha256 = argv[argv.index("--plan-sha256") + 1]
            plan = json.loads(inline)
            self.assertEqual(plan["bundle_id"], name)
            if plan["mode"] == "android":
                self.assertEqual(
                    plan["android"]["boot_id_source"],
                    "phase_fresh_snapshot",
                )
                self.assertNotIn("boot_id", plan["android"])
            self.assertEqual(
                materialize.sha256(inline.encode("ascii")),
                manifest["bindings"]["typed_plan_sha256"][
                    f"managed.{name}"
                ],
            )
            self.assertEqual(
                inline_sha256,
                materialize.sha256(inline.encode("ascii")),
            )
        for phone_name in ("op12", "op15"):
            for moment in ("before_argv", "after_argv"):
                argv = phone["probes"][phone_name][moment]
                self.assertIn("--capture-compatible", argv)
                inline = argv[argv.index("--plan-json") + 1]
                plan = json.loads(inline)
                self.assertNotIn("--boot-id", argv)
                self.assertEqual(
                    plan["android"]["boot_id_source"],
                    "phase_fresh_snapshot",
                )
                self.assertNotIn("boot_id", plan["android"])
                self.assertEqual(
                    plan["stage_v3"],
                    {
                        "expected_active_sequences": 0,
                        "source": "relay_owned_status",
                    },
                )
                self.assertNotIn("host", plan["stage_v3"])
                self.assertNotIn("port", plan["stage_v3"])
                expected_role = (
                    "stagenet_worker"
                    if phone_name == "op12"
                    else "direct_relay"
                )
                self.assertEqual(
                    plan["network_process"]["role"],
                    expected_role,
                )
                self.assertEqual(
                    plan["network_process"]["argv"][0],
                    plan["network_process"]["executable_path"],
                )
                self.assertEqual(
                    plan["network_process"]["artifact"]["path"],
                    plan["network_process"]["executable_path"],
                )
        for phone_name in ("op12", "op15"):
            self.assertEqual(
                phone["phones"][phone_name]["boot_id_source"],
                "phase_fresh_snapshot",
            )
            self.assertNotIn("boot_id", phone["phones"][phone_name])
        self.assertNotIn(
            "--expected-boot-id",
            phone["relay_process_probe"]["argv"],
        )

    def test_inline_plan_mutation_breaks_exact_binding(self):
        fixture = Fixture(self)
        outputs, _ = materialize.build_bundle(
            fixture.spec_path,
            fixture.output(),
        )
        phone = json.loads(outputs[materialize.PHONE_NAME])
        argv = phone["processes"]["op15_stagenet"]["argv"]
        inline_index = argv.index("--plan-json") + 1
        expected = json.loads(argv[inline_index])
        mutated = copy.deepcopy(argv)
        plan = copy.deepcopy(expected)
        plan["route"]["devices"] = "WRONG"
        mutated[inline_index] = materialize.compact_json(plan)
        with self.assertRaisesRegex(
            materialize.MaterializeError,
            "E_INLINE_PLAN_CONTENT",
        ):
            materialize.validate_inline_plan_argv(mutated, expected)

        probe_argv = phone["probes"]["op15"]["before_argv"]
        probe_plan = json.loads(
            probe_argv[probe_argv.index("--plan-json") + 1]
        )
        mutated_probe = copy.deepcopy(probe_argv)
        mutated_probe[mutated_probe.index("--plan-sha256") + 1] = "0" * 64
        with self.assertRaisesRegex(
            materialize.MaterializeError,
            "E_INLINE_PLAN_SHA256",
        ):
            materialize.validate_inline_plan_argv(
                mutated_probe,
                probe_plan,
                probe=True,
            )

    def test_managed_inline_plans_pass_typed_launcher_validator(self):
        fixture = Fixture(self)
        outputs, _ = materialize.build_bundle(
            fixture.spec_path,
            fixture.output(),
        )
        producer_root = (
            SOURCE.parent.parent
            / "a_only_acquisition_driver_v1"
            / "producers_v1"
        )
        launcher = load_module(
            "managed_launcher_materializer_validation",
            producer_root / "managed_runtime_launcher_v1.py",
        )
        phone = json.loads(outputs[materialize.PHONE_NAME])
        cuda = json.loads(outputs[materialize.CUDA_NAME])
        monolithic = json.loads(outputs[materialize.MONOLITHIC_NAME])
        argvs = [
            *[
                process["argv"]
                for process in phone["processes"].values()
            ],
            cuda["worker"]["argv"],
            monolithic["command"],
        ]
        for argv in argvs:
            inline = argv[argv.index("--plan-json") + 1]
            inline_sha256 = argv[argv.index("--plan-sha256") + 1]
            launcher.parse_plan_json(inline, inline_sha256)

    def test_probe_inline_plans_pass_capture_compatible_validator(self):
        fixture = Fixture(self)
        outputs, _ = materialize.build_bundle(
            fixture.spec_path,
            fixture.output(),
        )
        producer_root = (
            SOURCE.parent.parent
            / "a_only_acquisition_driver_v1"
            / "producers_v1"
        )
        probe = load_module(
            "phone_probe_materializer_validation",
            producer_root / "phone_runtime_probe_v1.py",
        )
        phone = json.loads(outputs[materialize.PHONE_NAME])
        for phone_name in ("op12", "op15"):
            argv = phone["probes"][phone_name]["before_argv"]
            inline = argv[argv.index("--plan-json") + 1]
            expected_sha256 = argv[argv.index("--plan-sha256") + 1]
            self.assertEqual(
                probe.parse_plan_json(inline, expected_sha256),
                probe.validate_plan(json.loads(inline)),
            )

    def test_published_graph_passes_existing_structural_validators(self):
        fixture = Fixture(self)
        output = fixture.output()
        outputs, manifest = materialize.build_bundle(fixture.spec_path, output)
        materialize.publish(outputs, output)
        graph = plan_common.validate_nested_execution_graph(
            output / materialize.COMMAND_NAME,
            manifest["bindings"]["contract_sha256"],
            manifest["bindings"]["candidate_sha256"],
        )
        runtime = plan_common.validate_runtime_bundle_plan(
            output / materialize.RUNTIME_NAME,
            manifest["bindings"]["contract_sha256"],
            manifest["bindings"]["candidate_sha256"],
        )
        specs = json.loads((output / materialize.SPECS_NAME).read_bytes())
        for name in ("acquisition", "artifact", "fresh"):
            plan_common.validate_driver_spec(specs[name], name)
        monolithic_argv = graph["command_matrix"]["desktop"][8]
        self.assertEqual(monolithic_argv[1], "--plan-json")
        self.assertEqual(
            json.loads(monolithic_argv[2])["bundle_id"],
            "cuda_monolithic",
        )
        self.assertEqual(
            {item["bundle_id"] for item in runtime["bundles"]},
            materialize.BUNDLE_IDS,
        )

    def test_published_launches_pass_producer_validators(self):
        fixture = Fixture(self)
        output = fixture.output()
        outputs, _ = materialize.build_bundle(fixture.spec_path, output)
        materialize.publish(outputs, output)
        producer_root = (
            SOURCE.parent.parent
            / "a_only_acquisition_driver_v1"
            / "producers_v1"
        )
        phone = load_module(
            "phone_materializer_validation",
            producer_root / "phone_route_capture_v1.py",
        )
        cuda = load_module(
            "cuda_materializer_validation",
            producer_root / "cuda_route_capture_v1.py",
        )
        monolithic = load_module(
            "monolithic_materializer_validation",
            producer_root / "cuda_monolithic_v1.py",
        )
        joint = load_module(
            "joint_materializer_validation",
            producer_root / "joint_phone_cuda_v1.py",
        )
        history_raw = (output / materialize.HISTORIES_NAME).read_bytes()
        model_artifact = json.loads(
            (output / materialize.CUDA_NAME).read_bytes()
        )["model_artifact"]
        quality_digest = json.loads(
            (output / materialize.CUDA_NAME).read_bytes()
        )["quality_corpus_content_sha256"]
        pre = fixture.root / "pre"
        pre.mkdir()
        (pre / "quality_corpus.jsonl").write_bytes(fixture.corpus.read_bytes())
        with mock.patch.multiple(
            phone,
            MODEL_SHA256=fixture.model_sha256,
        ):
            phone.load_plan(output / materialize.PHONE_NAME)
            phone.load_corpus(pre, "phase", quality_digest)
        with mock.patch.multiple(
            cuda,
            MODEL_BYTES=len(fixture.model_raw),
            MODEL_PATH=str(fixture.model),
            MODEL_SHA256=fixture.model_sha256,
        ):
            cuda.load_plan(
                output / materialize.CUDA_NAME,
                output / materialize.HISTORIES_NAME,
                history_raw,
                model_artifact,
            )
            cuda.load_corpus(pre, "phase", quality_digest)
        with mock.patch.object(
            monolithic,
            "MODEL_PATH",
            str(fixture.model),
            create=True,
        ):
            monolithic.load_launch(
                output / materialize.MONOLITHIC_NAME,
                fixture.model_sha256,
            )
        joint.load_plan(output / materialize.JOINT_NAME)

    def test_omitted_nested_input_refuses_without_output(self):
        fixture = Fixture(self)
        del fixture.spec["phones"]["op15"]["serial"]
        fixture.write_spec()
        output = fixture.output()
        with self.assertRaisesRegex(materialize.MaterializeError, "E_INPUT_KEYS"):
            materialize.build_bundle(fixture.spec_path, output)
        self.assertFalse(output.exists())

    def test_mutated_local_runtime_digest_refuses(self):
        fixture = Fixture(self)
        Path(fixture.runtime["cuda_runtime_path"]).write_bytes(b"mutated\n")
        with self.assertRaisesRegex(
            materialize.MaterializeError,
            "E_INPUT_RUNTIME_(BYTES|SHA256)",
        ):
            materialize.build_bundle(fixture.spec_path, fixture.output())

    def test_history_mutation_refuses_without_fabricating_replacement(self):
        fixture = Fixture(self)
        fixture.spec["route_epoch"] = 8
        fixture.write_spec()
        output = fixture.output()
        with self.assertRaisesRegex(
            materialize.MaterializeError,
            "E_INPUT_HISTORY_EPOCH",
        ):
            materialize.build_bundle(fixture.spec_path, output)
        self.assertFalse(output.exists())

    def test_symlink_source_refuses(self):
        fixture = Fixture(self)
        target = Path(fixture.sources["phone_probe"])
        link = fixture.root / "probe-link"
        link.symlink_to(target)
        fixture.spec["sources"]["phone_probe"] = str(link)
        fixture.write_spec()
        with self.assertRaisesRegex(
            materialize.MaterializeError,
            "E_INPUT_SYMLINK",
        ):
            materialize.build_bundle(fixture.spec_path, fixture.output())

    def test_partial_publication_is_removed(self):
        fixture = Fixture(self)
        output = fixture.output()
        outputs, _ = materialize.build_bundle(fixture.spec_path, output)
        original = materialize.write_new
        calls = 0

        def fail_after_first(path, raw):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected")
            original(path, raw)

        with mock.patch.object(materialize, "write_new", fail_after_first):
            with self.assertRaisesRegex(OSError, "injected"):
                materialize.publish(outputs, output)
        self.assertFalse(output.exists())
        self.assertEqual(
            list(output.parent.glob(f".{output.name}.tmp.*")),
            [],
        )

    def test_existing_output_is_never_modified(self):
        fixture = Fixture(self)
        output = fixture.output()
        output.mkdir()
        marker = output / "marker"
        marker.write_text("keep", encoding="ascii")
        with self.assertRaisesRegex(
            materialize.MaterializeError,
            "E_OUTPUT_EXISTS",
        ):
            materialize.build_bundle(fixture.spec_path, output)
        self.assertEqual(marker.read_text(encoding="ascii"), "keep")

    def test_publication_race_never_replaces_destination(self):
        fixture = Fixture(self)
        output = fixture.output()
        outputs, _ = materialize.build_bundle(fixture.spec_path, output)
        output.mkdir()
        marker = output / "marker"
        marker.write_text("keep", encoding="ascii")
        with self.assertRaisesRegex(
            materialize.MaterializeError,
            "E_OUTPUT_PUBLISH",
        ):
            materialize.publish(outputs, output)
        self.assertEqual(marker.read_text(encoding="ascii"), "keep")
        self.assertEqual(
            list(output.parent.glob(f".{output.name}.tmp.*")),
            [],
        )


if __name__ == "__main__":
    unittest.main()
