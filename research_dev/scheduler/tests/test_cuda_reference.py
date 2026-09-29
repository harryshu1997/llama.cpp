"""Real graph evidence and immutable reference comparison contracts."""

from copy import deepcopy
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from research_dev.scheduler.campaigns.burstgpt.compare_ab import (
    ComparisonError, cuda_graph_evidence, frozen_reference_identity,
    validate_frozen_reference, _native_fraction_counts,
)
from test_matched_comparison import fixture
from research_dev.scheduler.campaigns.burstgpt.launch import normalized_runner_contract
from research_dev.scheduler.adapters.heterogeneous_rig import HeterogeneousPhysicalRig
from research_dev.scheduler.campaigns.burstgpt.preflight import _phone_state_checks, _endpoint_checks
from research_dev.scheduler.campaigns.burstgpt.runner import (
    measured_energy, _capture_phone_residency_evidence, _direct_phone_configuration,
)
from research_dev.scheduler.adapters import RawEnergyMeasurement


class CudaReferenceTests(unittest.TestCase):
    def test_desktop_without_phone_qualification_does_not_claim_transport_dependencies(self):
        args = Mock(qwen_ffn_shards=None, gemma_ffn_shards=None, llama_ffn_shards=None)
        models = SimpleNamespace(
            expected_qwen=SimpleNamespace(model_id="q"),
            expected_gemma=SimpleNamespace(model_id="g"),
            diagnostic_url=SimpleNamespace(port=18383, hostname="phone"),
            catalog=SimpleNamespace(executors=()),
        )
        manifests = {name: SimpleNamespace(artifact_sha256="sha256:" + value * 64)
                     for name, value in (("q", "1"), ("g", "2"))}
        dependencies = {"cuda": Path("libggml-cuda.so")}
        with patch("research_dev.scheduler.campaigns.burstgpt.runner.DirectPhoneFfnSessionConfiguration") as configuration:
            _direct_phone_configuration(args, models, manifests, dependencies, None)
            self.assertEqual(configuration.call_args.kwargs["transport_host_dependency_paths"], {})
            self.assertIsNone(configuration.call_args.kwargs["transport_host_binary_path"])
            identity = object()
            _direct_phone_configuration(args, models, manifests, dependencies, identity)
            self.assertEqual(configuration.call_args.kwargs["transport_host_dependency_paths"], dependencies)
            self.assertIs(configuration.call_args.kwargs["transport_qualification_identity"], identity)

    def test_residency_evidence_is_captured_before_terminal_cleanup(self):
        class Rig:
            active = True

            @property
            def direct_phone_residency_state(self):
                return {"active": self.active}

            @property
            def phone_residency_phase_events(self):
                if not self.active:
                    raise RuntimeError("session is no longer active")
                return ({"session_id": "S", "phase": "READY"},)

            @property
            def phone_residency_call_events(self):
                if not self.active:
                    raise RuntimeError("session is no longer active")
                return ({"session_id": "S", "calls": 1},)

        rig = Rig()
        evidence = _capture_phone_residency_evidence(rig)
        rig.active = False
        self.assertEqual(evidence["phone_residency_phase_events"][0]["phase"], "READY")
        self.assertEqual(evidence["phone_residency_call_events"][0]["calls"], 1)
        empty = _capture_phone_residency_evidence(rig)
        self.assertEqual(empty["phone_residency_phase_events"], ())
        self.assertEqual(empty["phone_residency_call_events"], ())

    def test_native_coverage_counts_terminal_tail_without_inventing_energy_windows(self):
        request = {"request_id": "r", "physical_execution_proof": {
            "phone_call_count": 2, "phone_calls_by_layer": [{"layer": 2, "calls": 2}],
        }}
        windows = [{"applied_ack": {"plan_generation": 1}, "policy": {
            "layer_indices": [2], "columns": 128, "split_fraction_ppm": 500000,
        }}]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "RESULT.json"
            log = Path(directory) / "server.stderr"
            log.write_text(
                "S41SERVERFFNCALL context=72:0:1:1 request=1 layer=2 tokens=1 columns=128 payload_bytes=256\n"
                "S41SERVERFFNCALL context=72:0:1:1 request=2 layer=2 tokens=1 columns=128 payload_bytes=256\n",
                encoding="ascii",
            )
            self.assertEqual(_native_fraction_counts(path, request, windows), [((2,), 128, 500000, 2)])
            windows[0]["policy"]["columns"] = 256
            with self.assertRaisesRegex(ComparisonError, "fraction/layer/column"):
                _native_fraction_counts(path, request, windows)

    def test_fixed_ffn_preflight_does_not_require_a_whole_model_http_server(self):
        endpoint = "http://192.168.42.1:18382"
        capability = SimpleNamespace(executor_id="physical:op15-phone", device_id="phone",
                                     endpoint=endpoint, phone_sessions=("S",), supports_whole_model=False)
        transition = SimpleNamespace(device_id="cpu", prepares_device_ids=("phone",), maturity="QUALIFIED")
        catalog = SimpleNamespace(executors=(capability,), composite_executors=(), transitions=(transition,),
                                  placement_profile=SimpleNamespace(devices={"phone": SimpleNamespace(device_id="phone", kind="phone")}))
        phone = SimpleNamespace(available_bytes=2_000_000_000, thermal_qualified=True,
                                temperature_millic=45000, task_server_alive=False)
        self.assertTrue(all(row.status == "PASS" for row in _phone_state_checks(catalog, phone)))
        with patch("research_dev.scheduler.campaigns.burstgpt.preflight._local_port_available",
                   return_value=(False, "whole-model endpoint is not serving")):
            rows, _ = _endpoint_checks(catalog, (endpoint,), {"endpoint:" + endpoint: object()})
            self.assertEqual(rows[0].status, "PASS")
            capability.supports_whole_model = True
            rows, _ = _endpoint_checks(catalog, (endpoint,), {"endpoint:" + endpoint: object()})
            self.assertEqual(rows[0].status, "BLOCKED")
        transition.maturity = "SHADOW"
        self.assertEqual(_phone_state_checks(catalog, phone)[-1].status, "BLOCKED")

    def test_energy_export_keeps_assumed_active_and_idle_durations(self):
        value = RawEnergyMeasurement(
            energy_boundary_id="fleet", fleet_energy_uj_by_domain={"cpu": 100},
            measurement_evidence_ids=("physical:cpu",), attribution_kind="diagnostic",
            estimation_metadata={"phone_active_time_ns": 1000, "phone_idle_time_ns": 2000},
        )
        self.assertEqual(measured_energy(value)["estimation_metadata"], dict(value.estimation_metadata))

    def test_resident_but_unused_reference_still_requires_normal_terminal_cleanup(self):
        rig = HeterogeneousPhysicalRig.__new__(HeterogeneousPhysicalRig)
        rig._stop_dynamic_executors = Mock()
        rig.end_trace(require_phone_execution=False)
        rig._stop_dynamic_executors.assert_called_once_with(
            terminate_phone_session=True, require_phone_execution=False,
        )

    def test_startup_accounting_is_an_explicit_boolean_command_option(self):
        contract = normalized_runner_contract((
            "python3", "runner.py", "--include-startup-preparation", "--execute",
            "--selection-mode", "desktop-baseline",
        ))
        self.assertIs(contract["--include-startup-preparation"], True)
        self.assertIs(contract["--execute"], True)
        self.assertEqual(contract["--selection-mode"], "desktop-baseline")

    def test_graph_proof_requires_capture_success_and_gpu_replay(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("CREATE TABLE StringIds(id INTEGER,value TEXT)")
                connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(nameId INTEGER,returnValue INTEGER,start INTEGER,end INTEGER)")
                connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start INTEGER,end INTEGER,graphExecId INTEGER)")
                for name_id, name, count in (
                    (1, "cudaStreamEndCapture_v10000", 2),
                    (2, "cudaGraphInstantiate_v12000", 1),
                    (3, "cudaGraphExecUpdate_v10020", 2),
                    (4, "cudaGraphLaunch_v10000", 3),
                ):
                    connection.execute("INSERT INTO StringIds VALUES (?,?)", (name_id, name))
                    connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,0,1,2)", [(name_id,)] * count)
                connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (1,3,1)", [()] * 3)
            evidence = cuda_graph_evidence(path)
            self.assertEqual((evidence["captures"], evidence["launches"], evidence["recaptures_reusing_executable"]), (2, 3, 1))
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET returnValue=1 WHERE nameId=3")
            with self.assertRaisesRegex(ComparisonError, "API failed"):
                cuda_graph_evidence(path)
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET returnValue=0")
                connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
            with self.assertRaisesRegex(ComparisonError, "not physically demonstrated"):
                cuda_graph_evidence(path)

    def reference(self):
        result, _, replay = fixture()
        result.update(
            fixed_phone_residency=None,
            initial_observation_inputs={"store": "sha256:" + "c" * 64},
            adaptive_controller_configuration={"maximum_probe_tokens": 80},
            preparation_accounting="runtime-start-through-cleanup",
        )
        result["execution_identity"]["cuda_graph_mode_by_artifact"] = {"sha256:" + "4" * 64: "default"}
        for row in result["request_results"]:
            row["execution_command"]["adapter_parameters"]["cuda_graph_mode"] = "default"
        return result, replay

    def test_frozen_contract_rejects_unrelated_comparison_mismatches(self):
        result, replay = self.reference()
        expected = frozen_reference_identity(result)
        validate_frozen_reference(result, expected, expected_replay=replay)
        for key in ("catalog_sha256", "preparation_accounting", "initial_observation_inputs",
                    "execution_identity", "adaptive_controller_configuration"):
            changed = deepcopy(result)
            changed[key] = "different"
            with self.subTest(key=key), self.assertRaises(ComparisonError):
                validate_frozen_reference(changed, expected, expected_replay=replay)
        changed = deepcopy(result)
        changed["request_results"][0]["execution_command"]["adapter_parameters"]["gpu_layers"] += 1
        with self.assertRaises(ComparisonError):
            validate_frozen_reference(changed, expected, expected_replay=replay)

    def test_evidence_id_order_is_canonical_but_missing_ids_reject(self):
        result, replay = self.reference()
        expected = frozen_reference_identity(result)
        expected["energy_boundary"]["measurement_evidence_ids"].reverse()
        validate_frozen_reference(result, expected, expected_replay=replay)
        expected["energy_boundary"]["measurement_evidence_ids"].pop()
        with self.assertRaisesRegex(ComparisonError, "frozen experiment specification"):
            validate_frozen_reference(result, expected, expected_replay=replay)

    def test_default_label_cannot_admit_disabled_graphs(self):
        result, replay = self.reference()
        result["execution_identity"]["cuda_graph_mode_by_artifact"] = {"sha256:" + "4" * 64: "disabled"}
        with self.assertRaisesRegex(ComparisonError, "disabled CUDA graphs"):
            validate_frozen_reference(result, frozen_reference_identity(result), expected_replay=replay)


if __name__ == "__main__":
    unittest.main()
