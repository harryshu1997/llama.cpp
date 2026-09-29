"""Compatibility and safety boundaries for extracted campaign inputs."""

import ast
import json
import argparse
from dataclasses import replace
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import runpy
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research_dev.scheduler.config import (
    CampaignManifest, PhoneHtpMemoryCapConfiguration, SchedulerConfigurationError,
    StartupDesktopParentConfiguration,
)

from research_dev.scheduler.campaigns.burstgpt import (
    arguments,
    common,
    runner,
    trace_inputs,
)


class CampaignInputTests(unittest.TestCase):
    def test_startup_preload_invokes_scheduler_with_valid_shape_and_preserves_paid_receipts(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "snapshots").mkdir()
            config = StartupDesktopParentConfiguration("model", "cpu:qualified", "sha256:" + "a" * 64)
            captured = []
            ticket = SimpleNamespace(to_json=lambda: {"ticket_id": "startup-ticket"})

            def submit(request, model_id, snapshot, **options):
                request.validate()
                captured.append((request, model_id, options))
                return ticket

            scheduler = SimpleNamespace(submit_startup_parent_preload=submit, runtime_ticket=lambda _request_id: ticket)
            rig = SimpleNamespace(
                snapshot=lambda *_args: SimpleNamespace(captured_at_us=0, to_json=lambda: {}),
                backend=lambda: "backend",
                verify_startup_parent=lambda _command: {"state": "READY", "desktop_placement_sha256": config.desktop_placement_sha256},
            )
            execution = SimpleNamespace(command=SimpleNamespace(to_json=lambda: {}),
                                        observation=SimpleNamespace(energy=None, payload={}))
            adapter = Mock()
            adapter.execute.return_value = execution
            with patch.object(runner, "CanonicalPhysicalAdapter", return_value=adapter):
                result = runner._preload_startup_parents(
                    SimpleNamespace(output=output), scheduler, rig, {"model": "alias"}, output,
                    time.monotonic_ns(), (config,),
                )
            self.assertEqual((captured[0][0].input_tokens, captured[0][0].output_tokens), (1, 2))
            self.assertEqual(captured[0][2]["executor_id"], config.executor_id)
            self.assertEqual(adapter.execute.call_args.args[1].quality_mode, "accounting-only")
            self.assertEqual(result[0]["accounting"], "startup-load-and-verification-in-paid-interval")
            self.assertTrue((output / "STARTUP_PRELOAD_0.json").is_file())

    def test_catalog_script_registers_whole_phone_without_package_context(self):
        namespace = runpy.run_path(str(Path(runner.__file__).with_name("catalog.py")))
        register = namespace["_register_overlay_whole_phone"]
        materialize = Mock(return_value="registered")
        phone = SimpleNamespace(
            whole_server_sha256="sha256:" + "a" * 64,
            whole_control_transport="adb-ncm", whole_ncm_adb_endpoint="192.0.2.1:5555",
            whole_executable_device="GPUOpenCL", whole_forward_port=29382,
            whole_library_directory="/phone/lib", whole_remote_port=18382,
            whole_server_path="/phone/llama-server",
        )
        model = SimpleNamespace(
            endpoint_ids={"whole_phone": "whole"},
            backend_ids={"whole_phone": "android-llama-server-opencl"},
            phone_artifact_path="/phone/model.gguf", phone_adapter_parameters={},
        )
        with patch.dict(register.__globals__, materialize_whole_model_endpoint=materialize):
            result = register(
                "catalog", SimpleNamespace(model_id="small"), model,
                SimpleNamespace(phone=phone),
                SimpleNamespace(cpu_device_id="cpu", phone_device_id="phone"),
                {"whole": "http://127.0.0.1:29382"}, (),
            )
        self.assertEqual(result, "registered")
        parameters = materialize.call_args.kwargs["adapter_parameters"]
        self.assertEqual(parameters["android_control_transport"], "adb-ncm")
        self.assertTrue(parameters["android_control_script_sha256"].startswith("sha256:"))

    def test_runner_retains_canonical_imports(self):
        exports = (
            (common, (
                "CONFIRMATION", "UnifiedTraceError", "require", "canonical",
                "digest", "load_object", "load_rows",
            )),
            (arguments, ("host_dependency", "_build_parser", "_validate_arguments")),
            (trace_inputs, (
                "REPLAY_SCHEDULE_SCHEMA", "REPLAY_START_US", "QWEN_ROLE",
                "GEMMA_ROLE", "TRACE_HOT_MODEL_ID", "TRACE_COLD_MODEL_ID",
                "trace_role", "validate_trace", "merge_rows", "select_rows",
                "scale_replay_arrivals", "apply_named_replay_schedule",
                "persist_replay_schedule",
            )),
        )
        for module, names in exports:
            for name in names:
                with self.subTest(name=name):
                    self.assertIs(getattr(runner, name), getattr(module, name))

    @staticmethod
    def _command(parser, dependency, output):
        command = []
        for action in parser._actions:
            if not action.required:
                continue
            value = (
                str(output) if action.dest == "output"
                else "desktop-baseline" if action.dest == "selection_mode"
                else "1" if action.type is int
                else str(dependency)
            )
            command.extend((action.option_strings[0], value))
        return command

    def test_runner_fails_fast_on_lifecycle_failures_unless_drain_is_requested(self):
        parser = arguments._build_parser()
        command = self._command(parser, Path("/input"), Path("/output"))
        self.assertEqual(parser.parse_args(command).lifecycle_failure_mode, "fail-fast")
        self.assertEqual(parser.parse_args(
            command + ["--lifecycle-failure-mode", "drain"]
        ).lifecycle_failure_mode, "drain")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(command + ["--lifecycle-failure-mode", "never"])
        tree = ast.parse(Path(runner.__file__).read_text(encoding="ascii"))
        (construction,) = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "CanonicalArrivalCoordinator"
        ]
        (fail_fast,) = [row for row in construction.keywords if row.arg == "fail_fast"]
        self.assertEqual(ast.unparse(fail_fast.value), "args.lifecycle_failure_mode == 'fail-fast'")

    def test_argument_defaults_and_dependency_syntax(self):
        parser = arguments._build_parser()
        command = self._command(parser, Path("/input"), Path("/output"))
        args = parser.parse_args(command + [
            "--transport-host-dependency", "ggml=/libggml.so",
        ])
        self.assertFalse(args.execute)
        self.assertIsNone(args.confirm)
        self.assertEqual(args.max_workers, 32)
        self.assertEqual(args.minimum_usb_speed_mbps, 5000)
        self.assertEqual(args.energy_attribution_kind, "diagnostic")
        self.assertIsNone(args.fixed_phone_residency_json)
        self.assertIsNone(args.phone_htp_memory_caps_json)
        self.assertIsNone(args.startup_desktop_parents_json)
        self.assertIsNone(args.llama_ffn_shards)
        self.assertEqual(args.transport_host_dependency, [
            ("ggml", Path("/libggml.so")),
        ])
        for value in ("ggml", "=/libggml.so", "ggml=", "bad:name=/lib"):
            with self.subTest(value=value), self.assertRaises(common.UnifiedTraceError):
                arguments.host_dependency(value)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(command + ["--selection-mode", "force-phone"])

    def test_memory_cap_configuration_retains_bytes_and_rejects_invalid_values(self):
        cap = PhoneHtpMemoryCapConfiguration("phone-a", 768, 32)
        self.assertEqual(PhoneHtpMemoryCapConfiguration.from_json(cap.to_json()), cap)
        for changes in ({"cap_bytes": -1}, {"workspace_bytes": 769}, {"cap_bytes": True},
                        {"phone_device_id": ""}, {"workspace_bytes": -1}):
            with self.subTest(changes=changes), self.assertRaises(SchedulerConfigurationError):
                replace(cap, **changes)

    def test_memory_caps_round_trip_without_changing_default_configuration(self):
        row = {"schema": "s42-campaign-manifest-v1", "campaign_id": "memory-cap-test",
               "rig_manifest_path": "rig.json", "models_manifest_path": "models.json",
               "evidence_manifest_path": "evidence.json", "selection_mode": "energy-aware",
               "maximum_latency_ppm": 1_100_000, "energy_attribution_kind": "diagnostic",
               "trace": {"large_requests_path": "large.json", "overlay_requests_path": "overlay.json",
                         "trace_manifest_path": "trace.json"}}
        from research_dev.scheduler.config import CAMPAIGN_MANIFEST_SCHEMA
        row["schema"] = CAMPAIGN_MANIFEST_SCHEMA
        plain = CampaignManifest.from_json(row, Path("/inputs"))
        self.assertNotIn("phone_htp_memory_caps", plain.to_json())
        caps = [PhoneHtpMemoryCapConfiguration("phone-b", 512).to_json(),
                PhoneHtpMemoryCapConfiguration("phone-a", 768, 32).to_json()]
        configured = CampaignManifest.from_json({**row, "phone_htp_memory_caps": caps}, Path("/inputs"))
        self.assertEqual([cap.phone_device_id for cap in configured.phone_htp_memory_caps], ["phone-a", "phone-b"])
        self.assertEqual(CampaignManifest.from_json(configured.to_json(), Path("/inputs")), configured)
        startup = StartupDesktopParentConfiguration("model", "cpu:qualified", "sha256:" + "a" * 64)
        with self.assertRaisesRegex(SchedulerConfigurationError, "paid preparation"):
            CampaignManifest.from_json({**row, "startup_desktop_parents": [startup.to_json()]}, Path("/inputs"))
        warm = CampaignManifest.from_json({**row, "include_startup_preparation": True,
                                          "startup_desktop_parents": [startup.to_json()]}, Path("/inputs"))
        self.assertEqual(CampaignManifest.from_json(warm.to_json(), Path("/inputs")), warm)
        with self.assertRaisesRegex(SchedulerConfigurationError, "duplicated"):
            CampaignManifest.from_json({**warm.to_json(), "startup_desktop_parents": [startup.to_json()] * 2}, Path("/inputs"))
        for invalid in (caps + caps, {"phone-a": 768}, [{"cap_bytes": 768}]):
            with self.subTest(invalid=invalid), self.assertRaises(SchedulerConfigurationError):
                CampaignManifest.from_json({**row, "phone_htp_memory_caps": invalid}, Path("/inputs"))

    def test_adaptive_decode_overrides_reach_the_controller_configuration(self):
        from research_dev.scheduler.config import CAMPAIGN_MANIFEST_SCHEMA
        row = {"schema": CAMPAIGN_MANIFEST_SCHEMA, "campaign_id": "overrides-test",
               "rig_manifest_path": "rig.json", "models_manifest_path": "models.json",
               "evidence_manifest_path": "evidence.json", "selection_mode": "adaptive-decode",
               "maximum_latency_ppm": 1_100_000, "energy_attribution_kind": "diagnostic",
               "trace": {"large_requests_path": "large.json", "overlay_requests_path": "overlay.json",
                         "trace_manifest_path": "trace.json"}}
        plain = CampaignManifest.from_json(row, Path("/inputs"))
        self.assertIsNone(plain.adaptive_decode_overrides)
        overrides = {"maximum_probe_tokens": 16, "maximum_window_tokens": 8, "maximum_probe_candidates": 1,
                     "coarse_probe_fractions_ppm": [1000000], "allow_assumed_phone_power_for_operational_selection": True}
        configured = CampaignManifest.from_json({**row, "adaptive_decode_overrides": overrides}, Path("/inputs"))
        self.assertEqual(dict(configured.adaptive_decode_overrides), overrides)
        self.assertEqual(configured.to_json()["adaptive_decode_overrides"], overrides)
        for invalid in ({}, {"maximum_probe_tokens": 0}, {"coarse_probe_fractions_ppm": []},
                        {"maximum_probe_tokens": "16"}, {"bad name": 1}, [1]):
            with self.assertRaises(SchedulerConfigurationError):
                CampaignManifest.from_json({**row, "adaptive_decode_overrides": invalid}, Path("/inputs"))
        from research_dev.scheduler.campaigns.burstgpt import runner
        config = runner._configured_adaptive_decode_config(argparse.Namespace(
            adaptive_minimum_remaining_tokens=8, adaptive_maximum_probe_attempts_per_context=None,
            adaptive_decode_overrides_json=json.dumps(overrides)))
        self.assertEqual(config.maximum_probe_tokens, 16)
        self.assertEqual(config.coarse_probe_fractions_ppm, (1000000,))
        self.assertEqual(config.minimum_remaining_tokens, 8)
        with self.assertRaises(runner.UnifiedTraceError):
            runner._configured_adaptive_decode_config(argparse.Namespace(
                adaptive_minimum_remaining_tokens=None, adaptive_maximum_probe_attempts_per_context=None,
                adaptive_decode_overrides_json=json.dumps({"no_such_field": 1})))

    def test_probe_budget_field_reaches_the_adaptive_controller_configuration(self):
        from research_dev.scheduler.config import CAMPAIGN_MANIFEST_SCHEMA
        row = {"schema": CAMPAIGN_MANIFEST_SCHEMA, "campaign_id": "probe-budget-test",
               "rig_manifest_path": "rig.json", "models_manifest_path": "models.json",
               "evidence_manifest_path": "evidence.json", "selection_mode": "adaptive-decode",
               "maximum_latency_ppm": 1_100_000, "energy_attribution_kind": "diagnostic",
               "trace": {"large_requests_path": "large.json", "overlay_requests_path": "overlay.json",
                         "trace_manifest_path": "trace.json"}}
        plain = CampaignManifest.from_json(row, Path("/inputs"))
        self.assertIsNone(plain.adaptive_maximum_probe_attempts_per_context)
        self.assertIsNone(plain.to_json()["adaptive_maximum_probe_attempts_per_context"])
        configured = CampaignManifest.from_json(
            {**row, "adaptive_maximum_probe_attempts_per_context": 4}, Path("/inputs"))
        self.assertEqual(configured.adaptive_maximum_probe_attempts_per_context, 4)
        self.assertEqual(CampaignManifest.from_json(configured.to_json(), Path("/inputs")), configured)
        for invalid in (0, -1, True, "4", 4.0):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                SchedulerConfigurationError, "probe attempts per context"
            ):
                CampaignManifest.from_json(
                    {**row, "adaptive_maximum_probe_attempts_per_context": invalid}, Path("/inputs"))
        parser = arguments._build_parser()
        args = parser.parse_args(self._command(parser, Path("/input"), Path("/output")) + [
            "--adaptive-maximum-probe-attempts-per-context", "4",
        ])
        config = runner._configured_adaptive_decode_config(args)
        self.assertEqual(config.maximum_probe_attempts_per_context, 4)
        self.assertEqual(config.minimum_remaining_tokens, runner.AdaptiveDecodeConfig().minimum_remaining_tokens)
        args.adaptive_maximum_probe_attempts_per_context = None
        self.assertIsNone(runner._configured_adaptive_decode_config(args))
        args.adaptive_minimum_remaining_tokens = 8
        self.assertEqual(runner._configured_adaptive_decode_config(args),
                         runner.AdaptiveDecodeConfig(minimum_remaining_tokens=8))

    def test_argument_validation_still_requires_confirmation_and_new_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dependency = root / "dependency"
            dependency.write_bytes(b"test")
            output = root / "result"
            parser = arguments._build_parser()
            args = parser.parse_args(self._command(parser, dependency, output))
            with self.assertRaisesRegex(common.UnifiedTraceError, "execution confirmation"):
                arguments._validate_arguments(args)
            args.execute = True
            args.confirm = common.CONFIRMATION
            self.assertEqual(arguments._validate_arguments(args), {})
            self.assertFalse(output.exists())
            for changes, reason in (
                ({"llama_ffn_shards": "/index.json=/phone", "llama_phone_model": None},
                 "requires a phone parent artifact path"),
                ({"maximum_phone_sessions": 0}, "maximum phone sessions"),
                ({"adaptive_minimum_remaining_tokens": 0}, "minimum remaining tokens"),
                ({"adaptive_maximum_probe_attempts_per_context": 0}, "probe attempts per context"),
                ({"arrival_scale": 0}, "arrival scale"),
                ({"replay_schedule": dependency, "request_indices": "1"}, "replay transforms"),
                ({"phone_boot_image_sha256": "unqualified"}, "transport qualification"),
                ({"transport_host_dependency": [("ggml", dependency)] * 2}, "host dependencies"),
            ):
                changed = argparse.Namespace(**{**vars(args), **changes})
                with self.subTest(changes=changes), self.assertRaisesRegex(
                    common.UnifiedTraceError, reason
                ):
                    arguments._validate_arguments(changed)
            output.mkdir()
            with self.assertRaisesRegex(common.UnifiedTraceError, "physical output"):
                arguments._validate_arguments(args)


if __name__ == "__main__":
    unittest.main()
