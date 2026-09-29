"""Graph mode is a process and qualification identity, not an FFN switch."""

from dataclasses import replace
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_dev.scheduler import (
    GGUFModelManifestLoader, DeviceMemoryCapacity, RuntimePlacementSnapshot,
    UnifiedScheduler, UnifiedScheduleError,
)
from research_dev.scheduler.adapters import (
    LlamaServerProcessConfiguration,
    LlamaServerProcessLauncher,
    PhysicalAdapterError,
    llama_server_launch_contract,
    measured_desktop_control_profile,
)
from research_dev.scheduler.adapters.llama_server import (
    _launch_contract_supports_execution,
    llama_server_runtime_timing,
)
from research_dev.scheduler._internal.runtime_capabilities import (
    RuntimeDesktopControlProfile,
    RuntimeCapabilityError,
)
from research_dev.scheduler._internal.desktop_parent import (
    select_desktop_parent_for_live_vram,
)
from research_dev.scheduler._internal.route_generation.candidates import RouteCandidateMixin
from research_dev.scheduler.campaigns.burstgpt.compare_ab import execution_identity
from test_gguf_cost import write_synthetic_gguf
from test_llama_server_adapter import execution_command
import test_desktop_parent_capacity as parent_fixture


class CudaGraphModeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        model = self.root / "model.gguf"
        write_synthetic_gguf(model)
        self.manifest = GGUFModelManifestLoader.load("synthetic-model-id", model)
        self.command = execution_command(self.manifest.artifact_sha256)

    def test_mode_is_separate_from_ffn_and_rejects_invalid_values(self):
        default = llama_server_launch_contract(self.command, self.manifest)
        disabled = llama_server_launch_contract(replace(
            self.command,
            adapter_parameters={**self.command.adapter_parameters,
                                "cuda_graph_mode": "disabled"},
        ), self.manifest)
        self.assertEqual(default.cuda_graph_mode, "default")
        self.assertEqual(disabled.cuda_graph_mode, "disabled")
        self.assertEqual(default.ffn_environment, disabled.ffn_environment)
        defaults = replace(default, desktop_launch_mode="runtime-defaults")
        self.assertFalse(_launch_contract_supports_execution(default, defaults))
        self.assertFalse(_launch_contract_supports_execution(default, disabled))
        self.assertFalse(_launch_contract_supports_execution(disabled, default))
        for mode in ("0", "1", "enabled", None, 0):
            with self.subTest(mode=mode), self.assertRaises(PhysicalAdapterError):
                replace(default, cuda_graph_mode=mode)

    def test_subprocess_environment_never_inherits_presence_based_switch(self):
        path = self.root / "llama-server"
        path.write_text("#!/bin/sh\n", encoding="ascii")
        path.chmod(0o755)
        launcher = LlamaServerProcessLauncher(LlamaServerProcessConfiguration(
            server_path=path,
            model_paths_by_artifact={self.manifest.artifact_sha256: self.root / "model.gguf"},
            library_paths_by_device={}, executable_device_names={},
            output_directory=self.root,
        ))
        contract = replace(llama_server_launch_contract(self.command, self.manifest),
                           gpu_layers=0, phone_device_id=None, ffn_environment={})

        class FakeServer:
            def __init__(self, command, environment, directory, label, launch_contract):
                self.environment = environment
                self.command = command
                self.launch_contract = launch_contract
                self.stderr_lines = []
                self.process = SimpleNamespace(poll=lambda: None)

            def start(self):
                pass

            def stop(self):
                pass

        for inherited in ("0", "1"):
            for mode in ("default", "disabled"):
                with self.subTest(inherited=inherited, mode=mode), patch.dict(
                    os.environ, {"GGML_CUDA_DISABLE_GRAPHS": inherited}
                ), patch(
                    "research_dev.scheduler.adapters.llama_server.ManagedLlamaServer", FakeServer
                ), patch.object(launcher, "_healthy", return_value=True):
                    server = launcher.launch_contract(
                        "http://127.0.0.1:19000", replace(contract, cuda_graph_mode=mode),
                        self.manifest, label="graph-mode", control_check=lambda: None,
                    )
                    self.assertEqual(server.environment.get("GGML_CUDA_DISABLE_GRAPHS"),
                                     "1" if mode == "disabled" else None)
                    self.assertEqual(os.environ["GGML_CUDA_DISABLE_GRAPHS"], inherited)
                    self.assertFalse(server.launch_contract.ffn_environment)
                    reference = launcher.launch_contract(
                        "http://127.0.0.1:19000", replace(contract, cuda_graph_mode=mode,
                                                        desktop_launch_mode="runtime-defaults"),
                        self.manifest, label="reference-mode", control_check=lambda: None,
                    )
                    self.assertNotIn("--fit", reference.command)
                    self.assertNotIn("--flash-attn", reference.command)
                    self.assertNotIn("--split-mode", reference.command)
                    self.assertEqual(reference.environment.get("GGML_CUDA_DISABLE_GRAPHS"),
                                     "1" if mode == "disabled" else None)

    def test_qualification_and_generated_parent_hashes_include_mode(self):
        parent_fixture.DesktopParentCapacityTests.setUpClass()
        fixture = parent_fixture.DesktopParentCapacityTests()
        source = fixture._source()
        manifest = fixture.manifest
        default = measured_desktop_control_profile(
            manifest, executor_id=source.executor_id,
            operator_placements=source.operator_placements,
            evidence_ids=("sha256:" + "1" * 64,),
        )
        disabled = replace(default, cuda_graph_mode="disabled")
        self.assertNotEqual(default.placement_sha256, disabled.placement_sha256)
        self.assertEqual(RuntimeDesktopControlProfile.from_json(disabled.to_json()), disabled)
        self.assertEqual(RuntimeDesktopControlProfile.from_json(default.to_json()), default)
        with self.assertRaises(RuntimeCapabilityError):
            RuntimeDesktopControlProfile.from_json({**default.to_json(), "cuda_graph_mode": "disabled"})
        cache = {}
        for mode, profile in (("default", default), ("disabled", disabled)):
            current = replace(source, adapter_parameters={**source.adapter_parameters,
                                                         "cuda_graph_mode": mode})
            selection = select_desktop_parent_for_live_vram(
                manifest, current, DeviceMemoryCapacity("cuda0-vram", 100_000_000_000, 0, 0),
                memory_snapshot_id="test-graph-mode",
            )
            self.assertEqual(selection.selected.placement_sha256, profile.placement_sha256)
            pattern = SimpleNamespace(desktop_assignments={
                row.operator_id: row.primary_device_id for row in source.operator_placements
            })
            compiler = SimpleNamespace(_coordinator=lambda _: current,
                                       _desktop_placement_hash_cache=cache)
            self.assertEqual(RouteCandidateMixin._desktop_placement_sha256(
                compiler, manifest, pattern), profile.placement_sha256)
        self.assertEqual(len(cache), 2)

    def test_comparison_requires_and_preserves_graph_mode(self):
        base = {"source_manifest_sha256": "sha256:" + "1" * 64,
                "binaries": {"server": "sha256:" + "2" * 64}}
        artifact = self.manifest.artifact_sha256
        default = execution_identity({"execution_identity": {
            **base, "cuda_graph_mode_by_artifact": {artifact: "default"}}})
        disabled = execution_identity({"execution_identity": {
            **base, "cuda_graph_mode_by_artifact": {artifact: "disabled"}}})
        self.assertNotEqual(default, disabled)
        with self.assertRaises(RuntimeError):
            execution_identity({"execution_identity": base})

    def test_public_requalification_preserves_placement_without_old_evidence(self):
        parent_fixture.DesktopParentCapacityTests.setUpClass()
        fixture = parent_fixture.DesktopParentCapacityTests()
        source = fixture._source()
        source = replace(source, adapter_parameters={
            **source.adapter_parameters,
            "capacity_parent_source_placement_sha256": "sha256:" + "f" * 64,
            "capacity_parent_qualification_sha256": "sha256:" + "e" * 64,
        })
        catalog = SimpleNamespace(
            desktop_control_by_artifact={fixture.manifest.artifact_sha256:
                                        SimpleNamespace(executor_id=source.executor_id)},
            composite_executor_by_id={source.executor_id: source},
            placement_profile=SimpleNamespace(devices={
                "desktop-cuda": SimpleNamespace(kind="gpu", memory_pool_id="cuda0-vram")}),
        )
        scheduler = SimpleNamespace(runtime_model_manifest=lambda _: fixture.manifest,
                                    _runtime_capabilities=catalog)
        method = UnifiedScheduler.select_live_vram_desktop_parent.__wrapped__
        memory = RuntimePlacementSnapshot(
            snapshot_id="graph-mode", captured_at_us=0, valid_until_us=1_000_000,
            capacities={"cuda0-vram": DeviceMemoryCapacity("cuda0-vram", 100_000_000_000, 0, 0)},
        )
        result = method(scheduler, fixture.manifest.model_id, memory,
                        cuda_graph_mode="disabled", preserve_placement=True)
        self.assertEqual(result.selected.gpu_layers, source.adapter_parameters["gpu_layers"])
        self.assertEqual(result.selected.adapter_parameters["cuda_graph_mode"], "disabled")
        self.assertNotIn("capacity_parent_qualification_sha256", result.selected.adapter_parameters)
        tight = replace(memory, capacities={
            "cuda0-vram": DeviceMemoryCapacity("cuda0-vram", 10_000_000_000, 0, 0)})
        with self.assertRaisesRegex(UnifiedScheduleError, "exact desktop parent does not fit"):
            method(scheduler, fixture.manifest.model_id, tight,
                   cuda_graph_mode="disabled", preserve_placement=True)

    def test_absolute_timing_keeps_host_gap_separate_from_rpc(self):
        timing = llama_server_runtime_timing([
            "0.1.000.000 D slot process_toke: id  0 | task 1 | n_decoded = 1, x",
            "S41SERVERFFNUSB request=1 layer=11 tokens=1 columns=64 slot=0 started_ns=100 h2d_completed_ns=110 d2h_completed_ns=200 compute_us=1",
            "0.1.100.000 CUDA graph warmup complete",
            "0.1.200.000 D slot process_toke: id  0 | task 1 | n_decoded = 2, x",
            "S41SERVERFFNUSB request=2 layer=0 tokens=1 columns=64 slot=1 started_ns=900 h2d_completed_ns=910 d2h_completed_ns=1000 compute_us=1",
        ])
        self.assertEqual(timing["cuda_capture_log_count"], 1)
        self.assertEqual(timing["rpc_latency_ns"]["maximum"], 100)
        self.assertEqual(next(iter(timing["host_submission_gap_ns_by_call_class"].values()))["maximum"], 700)
        self.assertEqual(timing["decode_token_latency_us"]["median"], 200_000)


if __name__ == "__main__":
    unittest.main()
