"""The phone-owner preflight never claims a completed physical gate."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from dataclasses import replace

from research_dev.scheduler.adapters import PhysicalAdapterError
from research_dev.scheduler.campaigns.burstgpt import remote_resident_gate as gate
from research_dev.scheduler._internal.route_generation.costing_parameters import RouteParameterMixin
from research_dev.scheduler._internal.route_generation.common import RouteGenerationError
from research_dev.scheduler._internal.route_generation.feasibility import resident_phone_workspace


class ResidentPrefillCapacityTests(unittest.TestCase):
    def transport(self, declared=None):
        parameters = {"phone_device_id": "phone", "ffn_n_embd": 3840,
                      "ubatch_size": 512, "parallel": 1}
        if declared is not None:
            parameters["ffn_max_tokens"] = declared
        compiler = SimpleNamespace(_transport_adapter_parameters=Mock(return_value={}))
        RouteParameterMixin._candidate_transport_parameters(
            compiler, None, (), None, SimpleNamespace(assistance_phase="decode"), parameters,
        )
        return parameters, compiler._transport_adapter_parameters.call_args.args

    def test_explicit_resident_batch_capacity_survives_decode_materialization(self):
        parameters, arguments = self.transport(512)
        self.assertEqual(parameters["ffn_max_tokens"], 512)
        self.assertEqual(arguments[2], 3840 * 512 * 2)

    def test_ordinary_decode_capacity_is_unchanged(self):
        parameters, arguments = self.transport()
        self.assertEqual(parameters["ffn_max_tokens"], 1)
        self.assertEqual(arguments[2], 3840 * 2)

    def test_invalid_explicit_capacity_fails_before_transport(self):
        for value in (0, -1, 513, True, "512"):
            with self.subTest(value=value), self.assertRaises(RouteGenerationError):
                self.transport(value)

    def test_resident_prefill_workspace_is_reserved_without_changing_other_devices(self):
        from test_remote_resident_launch import MANIFEST
        capability = SimpleNamespace(workspace_bytes_per_token=4096)
        parameters = {"phone_device_id": "phone", "ffn_max_tokens": 512}
        expected = 513 * MANIFEST.embedding_length * 2 + 4096 * 512
        self.assertEqual(resident_phone_workspace(MANIFEST, capability, parameters, "phone", 1), expected)
        self.assertEqual(resident_phone_workspace(MANIFEST, capability, parameters, "cpu", 1), 1)
        self.assertEqual(resident_phone_workspace(MANIFEST, capability, parameters, "phone", expected * 2), expected * 2)

    def test_prefill_preparation_without_matching_transport_remains_ineligible(self):
        import test_offline_phone_residency as fixture
        from research_dev.scheduler import UnifiedScheduler, UnifiedScheduleError
        fixture.OfflinePhoneResidencyTests.setUpClass()
        source, requests, snapshot = fixture.OfflinePhoneResidencyTests.replay_runtime(load_evidence=False)
        catalog = source._runtime_capabilities
        catalog = replace(catalog, composite_executors=tuple(
            replace(row, adapter_parameters={**row.adapter_parameters,
                    "ffn_max_tokens": row.adapter_parameters["ubatch_size"],
                    "memory_workspace_minimum_bytes:op15-phone": 805306368})
            if row.assisted_operator_kind == "ffn" else row
            for row in catalog.composite_executors
        ))
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce", maximum_phone_sessions=1)
        scheduler.register_runtime_capabilities(catalog)
        for manifest in source._runtime_manifests.values():
            scheduler.register_model_manifest(manifest)
        with self.assertRaisesRegex(UnifiedScheduleError, "offline phone residency demand is unavailable"):
            scheduler.plan_offline_phone_residency(
                requests, snapshot=snapshot, observed_at_us=snapshot.captured_at_us,
            )


class PhoneOwnerPreflightTests(unittest.TestCase):
    def _preflight(self, directory, error=None):
        args = SimpleNamespace(
            output=Path(directory), usb_qualification_identity=Path("identity.json"),
            server=Path("server"), transport_host_dependency=(("llama", Path("libllama.so")),),
        )
        identity = SimpleNamespace(
            software_identity=(
                {"host_binary_sha256": "old", "host_dependency_sha256:llama": "old-lib"}
                if error is not None else
                {"host_binary_sha256": "actual", "host_dependency_sha256:llama": "actual"}
            ),
            to_json=lambda: {"identity": "original"},
        )
        with (
            patch.object(gate, "load_transport_qualification_identity", return_value=identity),
            patch.object(gate, "_sha256", return_value="actual"),
            patch.object(gate.runner, "_direct_phone_configuration"),
            patch.object(gate, "DirectPhoneFfnSession") as phone,
        ):
            if error is not None:
                phone.return_value.preflight.side_effect = error
            else:
                phone.return_value.preflight.return_value.to_json.return_value = {"verified": True}
            result = gate._phone_owner_preflight(
                args, object(), {}, SimpleNamespace(to_json=lambda: {"gpu_layers": 23}), {}
            )
            phone.return_value.start.assert_not_called()
            phone.return_value.abort.assert_not_called()
        saved = json.loads((args.output / "PHONE_OWNER_PREFLIGHT.json").read_text())
        self.assertEqual(saved, result)
        return result

    def test_stale_identity_is_blocked_and_differences_are_preserved(self):
        with TemporaryDirectory() as directory:
            result = self._preflight(directory, PhysicalAdapterError("qualification software identity differs"))
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("qualification software identity differs", result["error"])
        self.assertEqual(result["host_software_mismatches"]["host_dependency_sha256:llama"],
                         {"expected": "old-lib", "observed": "actual"})
        self.assertFalse(result["physical_inference_started"])
        self.assertEqual(result["gates"], {})

    def test_success_is_only_a_preflight_pass(self):
        with TemporaryDirectory() as directory:
            result = self._preflight(directory)
        self.assertEqual(result["status"], "PREFLIGHT_PASS")
        self.assertFalse(result["physical_inference_started"])
        self.assertEqual(result["gates"], {})


if __name__ == "__main__":
    unittest.main()
