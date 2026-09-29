#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
for path in (REPO_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from full_fp16_burstgpt_v1.plan_full_fp16_burstgpt import (  # noqa: E402
    canonical,
    compile_plan,
)
from full_fp16_burstgpt_v1.capture_runtime_placement_snapshot import (  # noqa: E402
    parse_gpu,
    parse_meminfo,
)
from research_dev.scheduler.campaigns.burstgpt.admission_profile import (  # noqa: E402
    FEATURES,
    fit_role,
)
from research_dev.scheduler import PhoneResidencyPlan  # noqa: E402
from research_dev.scheduler.adapters import (  # noqa: E402
    AndroidUsbRestorationReceipt,
    EndpointRuntimeSample,
    FunctionFsBridgeTerminalReceipt,
    LlamaCppCompletionPayload,
    PhysicalAdapterError,
    PhysicalParticipantCommand,
    PhysicalTransitionCommand,
    RawEnergyMeasurement,
)
from research_dev.scheduler import (  # noqa: E402
    DeviceMemoryCapacity,
    ModelManifest,
    RuntimeExecutionContract,
    RuntimePlacementSnapshot,
    RuntimeResidencyEviction,
    RuntimeTransitionPlan,
)
from research_dev.scheduler.tests.test_arrival_coordinator import (  # noqa: E402
    REAL_TRACE_ARRIVALS_US,
    REAL_TRACE_ORDER,
)


QWEN = (
    ROOT
    / "multi_session_phone_v1/results/"
      "QWEN_FULL_FFN_M1_M4_ENERGY_SCREEN_ABBA_V2.json"
)
GEMMA = ROOT / "hybrid_overflow_v1/results/physical_pair_r1/PAIR.json"
RESIDENCY = (
    ROOT
    / "multi_session_phone_v1/results/"
      "OP15_THREE_SESSION_RESIDENCY_PLAN_V1.json"
)
PLACEMENT = (
    ROOT
    / "full_fp16_burstgpt_v1/results/FULL_FP16_BURSTGPT_ABBA_V2.json"
)
RUNNER = (
    REPO_ROOT
    / "research_dev/spikes/s41_gemma_qwen_continuous_baseline/"
      "tp_operator_split_v1/mixed_scheduler_v1/run_hierarchical_trace.py"
)
RUN_SCRIPT = ROOT / "full_fp16_burstgpt_v1/run_full_fp16_arm.sh"
UNIFIED_RUNNER = (
    REPO_ROOT / "research_dev/scheduler/campaigns/burstgpt/runner.py"
)
CANONICAL_PHYSICAL_RIG = (
    REPO_ROOT / "research_dev/scheduler/adapters/heterogeneous_rig.py"
)
PHYSICAL_PREFLIGHT = (
    REPO_ROOT / "research_dev/scheduler/campaigns/burstgpt/preflight.py"
)
PHYSICAL_CAMPAIGN = (
    REPO_ROOT
    / "research_dev/scheduler/campaigns/burstgpt/run_physical_campaign.sh"
)
CLOSE_HELPER = (
    REPO_ROOT / "research_dev/scheduler/adapters/close_resident_bridge.py"
)
LARGE_TRACE = (
    REPO_ROOT
    / "research_dev/spikes/s41_gemma_qwen_continuous_baseline/"
      "tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1/"
      "REQUESTS_SEMANTIC_SOURCE.jsonl"
)
OVERLAY_DIR = ROOT / "full_fp16_burstgpt_v1/small_model_overlay_v1"


def load_unified_runner():
    spec = importlib.util.spec_from_file_location(
        "s42_test_unified_fp16_runner", UNIFIED_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unified runner import specification is absent")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_close_helper():
    spec = importlib.util.spec_from_file_location(
        "s42_test_close_resident_bridge", CLOSE_HELPER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("close helper import specification is absent")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_physical_preflight():
    spec = importlib.util.spec_from_file_location(
        "s42_test_physical_preflight", PHYSICAL_PREFLIGHT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("preflight import specification is absent")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FullFp16BurstGptTests(unittest.TestCase):
    @staticmethod
    def runtime_snapshot(
        residency: PhoneResidencyPlan,
    ) -> RuntimePlacementSnapshot:
        return RuntimePlacementSnapshot(
            snapshot_id="test-live-runtime-capacity",
            captured_at_us=1_000,
            valid_until_us=10_000,
            capacities={
                "cuda:GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08:vram": (
                    DeviceMemoryCapacity(
                        "cuda:GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08:vram",
                        17_175_674_880,
                        816_840_704,
                        536_870_912,
                    )
                ),
                "desktop-host-ram": DeviceMemoryCapacity(
                    "desktop-host-ram",
                    68_719_476_736,
                    10_000_000_000,
                    8_589_934_592,
                ),
                f"phone:{residency.phone_serial}:dram": DeviceMemoryCapacity(
                    f"phone:{residency.phone_serial}:dram",
                    residency.memory_capacity_bytes,
                    4_000_000_000,
                    residency.minimum_available_bytes,
                ),
            },
        )

    def test_unified_scheduler_compiles_both_phone_phases(self) -> None:
        qwen = json.loads(QWEN.read_text(encoding="ascii"))
        gemma = json.loads(GEMMA.read_text(encoding="ascii"))
        placement = json.loads(PLACEMENT.read_text(encoding="ascii"))
        residency = PhoneResidencyPlan.from_json(json.loads(
            RESIDENCY.read_text(encoding="ascii")
        ))
        plan = compile_plan(
            qwen,
            gemma,
            placement,
            residency,
            self.runtime_snapshot(residency),
            qwen_result_hash=digest(QWEN),
            gemma_result_hash=digest(GEMMA),
            placement_result_hash=digest(PLACEMENT),
            residency_hash=digest(RESIDENCY),
            runtime_snapshot_hash="f" * 64,
            now_us=2_000,
        )
        supplied_hash = plan.pop("plan_sha256")
        self.assertEqual(
            supplied_hash, hashlib.sha256(canonical(plan)).hexdigest()
        )
        qwen_phase = plan["phases"]["qwen"]
        gemma_phase = plan["phases"]["gemma"]
        self.assertEqual(qwen_phase["phone_policy"], "4:17408,512:0")
        self.assertEqual(gemma_phase["phone_policy"],
                         "1:6144,16:6144,512:0")
        self.assertEqual(
            qwen_phase["decision"]["decision_reason"],
            "ENERGY_POSITIVE_RESIDENT_PHONE_ROUTE",
        )
        self.assertEqual(
            gemma_phase["decision"]["decision_reason"],
            "ENERGY_POSITIVE_RESIDENT_PHONE_ROUTE",
        )
        self.assertEqual(
            [row["session_id"] for row in
             qwen_phase["decision"]["arm"]["signals"]],
            ["htp1", "htp2"],
        )
        self.assertEqual(
            gemma_phase["decision"]["arm"]["session_id"], "htp0"
        )
        runtime = plan["runtime_placement"]
        self.assertEqual(
            runtime["selected"]["candidate_id"],
            "fp16-server-gpu-cpu-op15-switch-v1",
        )
        self.assertGreaterEqual(
            runtime["conservative_energy_saving_ppm"], 200_000
        )
        self.assertLess(runtime["conservative_latency_change_ppm"], 0)
        self.assertEqual(
            plan["placement"],
            {"gemma_gpu_layers": 25, "qwen_gpu_layers": 18},
        )
        self.assertEqual(plan["execution_arm"], "op15")
        self.assertEqual(
            plan["execution_arm_source"], "runtime_placement"
        )

    def test_runtime_placement_falls_back_to_executable_control(self) -> None:
        qwen = json.loads(QWEN.read_text(encoding="ascii"))
        gemma = json.loads(GEMMA.read_text(encoding="ascii"))
        placement = json.loads(PLACEMENT.read_text(encoding="ascii"))
        residency = PhoneResidencyPlan.from_json(json.loads(
            RESIDENCY.read_text(encoding="ascii")
        ))
        live = self.runtime_snapshot(residency)
        phone_resource = f"phone:{residency.phone_serial}:dram"
        capacities = dict(live.capacities)
        capacities[phone_resource] = DeviceMemoryCapacity(
            phone_resource,
            residency.memory_capacity_bytes,
            (
                residency.memory_capacity_bytes
                - residency.minimum_available_bytes
                - 1
            ),
            residency.minimum_available_bytes,
        )
        constrained = RuntimePlacementSnapshot(
            snapshot_id="test-phone-capacity-fallback",
            captured_at_us=live.captured_at_us,
            valid_until_us=live.valid_until_us,
            capacities=capacities,
        )
        plan = compile_plan(
            qwen,
            gemma,
            placement,
            residency,
            constrained,
            qwen_result_hash=digest(QWEN),
            gemma_result_hash=digest(GEMMA),
            placement_result_hash=digest(PLACEMENT),
            residency_hash=digest(RESIDENCY),
            runtime_snapshot_hash="e" * 64,
            now_us=2_000,
        )
        runtime = plan["runtime_placement"]
        self.assertEqual(
            runtime["selected"]["candidate_id"],
            "fp16-server-gpu-cpu-switch-v1",
        )
        self.assertEqual(
            runtime["selection_reason"],
            "BASELINE_FALLBACK_NO_ADMISSIBLE_ALTERNATIVE",
        )
        self.assertEqual(plan["execution_arm"], "control")

    def test_large_runner_delegates_to_one_unified_runtime(self) -> None:
        source = RUNNER.read_text(encoding="ascii")
        unified = UNIFIED_RUNNER.read_text(encoding="ascii")
        canonical_rig = CANONICAL_PHYSICAL_RIG.read_text(encoding="ascii")
        self.assertIn("scheduler.campaigns.burstgpt.runner", source)
        self.assertNotIn("held_rows", source)
        self.assertNotIn("fp16_runtime_binding", source)
        self.assertNotIn("fp16_family_routes", source)
        self.assertIn("CanonicalArrivalCoordinator", unified)
        self.assertIn("UnifiedScheduler.for_runtime_discovery", unified)
        self.assertIn('len(decisions) == 84', unified)
        self.assertIn('len(terminals) == 84', unified)
        self.assertIn("validate_decision_candidate_coverage", unified)
        self.assertIn("request_candidate_coverage", unified)
        self.assertIn("CanonicalTransitionRegistry", canonical_rig)
        self.assertIn("LlamaServerProcessLauncher", canonical_rig)
        self.assertNotIn("decision.route_id", canonical_rig)

    def test_physical_rig_stops_resident_server_after_cleanup_error(self) -> None:
        runner = load_unified_runner()

        class ResidentServer:
            stopped = False

            def stop(self) -> None:
                self.stopped = True

        resident = ResidentServer()
        rig = object.__new__(runner.UnifiedFp16PhysicalRig)
        rig._resident_server = resident
        rig._runtime_monitor_started = False
        rig._phone_sampler_started = False
        rig._server_sampler_started = False

        def fail_current() -> None:
            raise PhysicalAdapterError("synthetic current cleanup failure")

        rig._stop_current = fail_current
        with self.assertRaisesRegex(
            PhysicalAdapterError, "synthetic current cleanup failure"
        ):
            rig.close()
        self.assertTrue(resident.stopped)
        self.assertIsNone(rig._resident_server)

    def test_physical_transition_rejects_changed_eviction_executor(self) -> None:
        runner = load_unified_runner()
        target_artifact = "sha256:" + "1" * 64
        resident_artifact = "sha256:" + "2" * 64
        target = SimpleNamespace(artifact_sha256=target_artifact)
        resident = SimpleNamespace(artifact_sha256=resident_artifact)
        rig = object.__new__(runner.UnifiedFp16PhysicalRig)
        rig.configuration = SimpleNamespace(
            manifests={"synthetic-target": target},
            phone_device_id="phone-a",
        )
        rig._transition_lock = threading.Lock()
        rig._desktop_transition_lock = threading.Lock()
        rig._lock = threading.RLock()
        rig._current_executor_id = "executor:resident"
        rig._current_manifest = resident
        rig._current_server = SimpleNamespace(
            process=SimpleNamespace(poll=lambda: None)
        )
        rig._transition_active = False
        rig._active_transition_count = 0
        rig._generation = 7
        rig._launch_attempt = 0
        rig._stop_current = mock.Mock()
        transition = RuntimeTransitionPlan(
            transition_id="transition:synthetic",
            device_id="accelerator-a",
            source_state="cold",
            target_state="hot",
            latency_us=10,
            energy_uj=20,
            resource_ids=("exclusive:accelerator-a",),
            maturity="QUALIFIED",
            executor_id="executor:target",
            evictions=(RuntimeResidencyEviction(
                model_id="synthetic-resident",
                artifact_sha256=resident_artifact,
                device_id="accelerator-a",
                resident_bytes=100,
                generation=7,
                executor_id="executor:changed",
                reclaimable_bytes=200,
            ),),
        )
        command = PhysicalTransitionCommand(
            ticket_id="ticket:synthetic",
            request_id="request:synthetic",
            artifact_sha256=target_artifact,
            route_id="route:synthetic",
            operator_plan_sha256="sha256:" + "3" * 64,
            participant=PhysicalParticipantCommand(
                executor_id="executor:target",
                device_id="accelerator-a",
                endpoint="http://127.0.0.1:19000",
                backend="backend:synthetic",
                resource_ids=("exclusive:accelerator-a",),
            ),
            transition=transition,
            execution_contract=RuntimeExecutionContract.desktop(),
            adapter_parameters={},
        )
        with tempfile.TemporaryDirectory() as directory:
            payload = LlamaCppCompletionPayload(
                request_id="request:synthetic",
                expected_model_alias="synthetic-target",
                input_tokens=1,
                output_tokens=1,
                prompt_tokens=(1,),
                seed=1,
                stream_path=Path(directory) / "stream.jsonl",
                on_first_token=lambda _value: None,
            )
            with self.assertRaisesRegex(
                PhysicalAdapterError,
                "transition eviction source differs from physical endpoint",
            ):
                rig._execute_transition(command, payload, lambda: None)
        rig._stop_current.assert_not_called()

        rig._stop_current.reset_mock()
        valid_command = replace(
            command,
            transition=replace(
                transition,
                evictions=(replace(
                    transition.evictions[0],
                    executor_id="executor:resident",
                ),),
            ),
        )

        def cancelled() -> None:
            raise PhysicalAdapterError("synthetic scheduler cancellation")

        with self.assertRaisesRegex(
            PhysicalAdapterError, "synthetic scheduler cancellation"
        ):
            rig._execute_transition(valid_command, payload, cancelled)
        rig._stop_current.assert_not_called()

    def test_close_helper_sends_explicit_terminal_session_control(self) -> None:
        helper = load_close_helper()
        response = bytearray(64)
        struct.pack_into(
            "<IHHH",
            response,
            0,
            helper.PROTOCOL_MAGIC,
            helper.PROTOCOL_VERSION,
            2,
            0,
        )

        class Stream:
            def __init__(self) -> None:
                self.sent = []

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def recv(self, _size: int) -> bytes:
                return bytes(response)

            def sendall(self, value: bytes) -> None:
                self.sent.append(value)

        stream = Stream()
        arguments = [
            "close_resident_bridge.py",
            "--port", "1234",
            "--layer-mask", "1",
            "--n-embd", "64",
            "--columns", "128",
            "--terminate-session",
        ]
        with mock.patch.object(
                helper.socket, "create_connection", return_value=stream), \
                mock.patch.object(sys, "argv", arguments):
            self.assertEqual(helper.main(), 0)
        self.assertEqual(len(stream.sent), 2)
        shutdown = struct.unpack("<IHHIiIIIII", stream.sent[1])
        self.assertEqual(shutdown[:5], (
            helper.PROTOCOL_MAGIC,
            helper.PROTOCOL_VERSION,
            3,
            0,
            -1,
        ))
        self.assertEqual(shutdown[5:], (0, 0, 0, 0, 0))

    def test_terminal_retry_only_rechecks_android_usb(self) -> None:
        runner = load_unified_runner()
        physical = sys.modules[runner.UnifiedFp16PhysicalRig.__module__]
        rig = object.__new__(runner.UnifiedFp16PhysicalRig)
        rig._lock = threading.RLock()
        rig._current_bridge = SimpleNamespace(
            process=SimpleNamespace(
                poll=lambda: None,
                wait=lambda timeout: 0,
            ),
            stderr_lines=(),
            terminate=lambda: None,
        )
        rig._current_parameters = {
            "bridge_queue_depth": 1,
            "ffn_activation": "geglu",
            "ffn_bridge_port": 18671,
            "ffn_columns": 6144,
            "ffn_layer_mask": 8_388_607,
            "ffn_n_embd": 3840,
        }
        rig._last_phone_parameters = dict(rig._current_parameters)
        rig._phone_session_active = True
        rig._phone_terminal_sent = False
        rig._terminal_launch_attempt = 0
        rig._launch_attempt = 2
        rig._bridge_sha256 = "sha256:" + "1" * 64
        rig._bridge_terminal_receipts = []
        rig._usb_restore_receipts = []
        rig._transport_identity = mock.Mock(
            return_value="synthetic-functionfs"
        )
        rig._current_transition_resources = mock.Mock(return_value=())
        rig.configuration = SimpleNamespace(
            adb_port=5037,
            catalog=SimpleNamespace(resources={}),
            minimum_usb_speed_mbps=5000,
            phone_usb_serial="SYNTHETIC123",
        )
        terminal = FunctionFsBridgeTerminalReceipt(
            status="ok",
            calls=0,
            allocator="malloc-split",
            reset_recoveries=0,
            values={
                "allocator": "malloc-split",
                "calls": 0,
                "reset_recoveries": 0,
                "status": "ok",
            },
        )
        restored = AndroidUsbRestorationReceipt(
            serial="SYNTHETIC123",
            adb_port=5037,
            sysfs_device="2-2",
            vendor_id="22d9",
            product_id="2772",
            negotiated_speed_mbps=5000,
        )
        with mock.patch.object(
                physical, "close_functionfs_bridge",
                return_value=terminal) as close_bridge, \
                mock.patch.object(
                    physical, "verify_android_usb_restored",
                    side_effect=(
                        PhysicalAdapterError("synthetic ADB delay"),
                        restored,
                    ),
                ) as verify_usb, \
                mock.patch.object(
                    physical, "CapturedProcess",
                    side_effect=AssertionError("unexpected bridge relaunch")):
            with self.assertRaisesRegex(
                PhysicalAdapterError, "synthetic ADB delay"
            ):
                rig._close_bridge(terminate_phone_session=True)
            rig._close_bridge(terminate_phone_session=True)

        close_bridge.assert_called_once()
        self.assertEqual(verify_usb.call_count, 2)
        self.assertEqual(len(rig._bridge_terminal_receipts), 1)
        self.assertEqual(rig._usb_restore_receipts, [restored.to_json()])
        self.assertFalse(rig._phone_session_active)

    def test_direct_session_cleanup_never_relaunches_legacy_bridge(self) -> None:
        runner = load_unified_runner()
        rig = object.__new__(runner.UnifiedFp16PhysicalRig)
        direct = SimpleNamespace(active=True)
        rig._lock = threading.RLock()
        rig._current_bridge = None
        rig._current_parameters = {"ffn_transport": "functionfs-usb"}
        rig._last_phone_parameters = dict(rig._current_parameters)
        rig._phone_session_active = True
        rig._phone_terminal_sent = False
        rig._direct_phone_session = direct
        rig._finish_direct_phone = mock.Mock()

        rig._close_bridge(terminate_phone_session=True)

        rig._finish_direct_phone.assert_called_once_with(direct)

    def test_direct_abort_is_terminal_and_restores_idempotently(self) -> None:
        runner = load_unified_runner()
        restored = AndroidUsbRestorationReceipt(
            serial="SYNTHETIC123",
            adb_port=5037,
            sysfs_device="2-2",
            vendor_id="22d9",
            product_id="2772",
            negotiated_speed_mbps=5000,
        )
        direct = SimpleNamespace(abort=mock.Mock(return_value=restored))
        rig = object.__new__(runner.UnifiedFp16PhysicalRig)
        rig._lock = threading.RLock()
        rig._usb_restore_receipts = []
        rig._phone_session_active = True
        rig._last_phone_parameters = {
            "ffn_transport": "functionfs-usb"
        }
        rig._phone_terminal_sent = False

        rig._abort_direct_phone(direct)

        self.assertTrue(rig._phone_terminal_sent)
        self.assertFalse(rig._phone_session_active)
        self.assertEqual(
            rig._usb_restore_receipts, [restored.to_json()]
        )

    def test_failed_direct_transition_aborts_and_clears_session(self) -> None:
        runner = load_unified_runner()
        rig = object.__new__(runner.UnifiedFp16PhysicalRig)
        direct = SimpleNamespace(active=True)
        rig._lock = threading.RLock()
        rig._current_server = None
        rig._current_direct_phone = direct
        rig._current_executor_id = "physical:synthetic"
        rig._current_manifest = object()
        rig._current_parameters = {"ffn_transport": "functionfs-usb"}
        rig._current_operator_plan = {"route": "synthetic"}
        rig._abort_direct_phone = mock.Mock()
        rig._close_bridge = mock.Mock()

        rig._stop_current(
            terminate_phone_session=False,
            allow_incomplete_direct_phone=True,
        )

        rig._abort_direct_phone.assert_called_once_with(direct)
        rig._close_bridge.assert_not_called()
        self.assertIsNone(rig._current_direct_phone)
        self.assertIsNone(rig._current_executor_id)
        self.assertIsNone(rig._current_manifest)
        self.assertIsNone(rig._current_parameters)
        self.assertIsNone(rig._current_operator_plan)

    def test_preflight_uses_one_remote_endpoint_sample(self) -> None:
        preflight = load_physical_preflight()
        sample = EndpointRuntimeSample("healthy", "live", 1)
        with mock.patch.object(
                preflight, "probe_llama_endpoint",
                side_effect=AssertionError("unexpected second probe")):
            ready, detail = preflight._local_port_available(
                "http://192.0.2.1:8080", sample
            )
        self.assertTrue(ready)
        self.assertEqual(
            detail,
            "remote endpoint health=healthy slots_probe=live free_slots=1",
        )

    def test_preflight_retries_transient_raw_observation(self) -> None:
        preflight = load_physical_preflight()
        probe = mock.Mock(side_effect=(None, "fresh"))
        with mock.patch.object(preflight.time, "sleep") as sleep:
            observed = preflight._retry_raw_observation(
                probe, lambda value: value == "fresh", timeout_s=1
            )

        self.assertEqual(observed, "fresh")
        self.assertEqual(probe.call_count, 2)
        sleep.assert_called_once_with(0.25)

    def test_preflight_refreshes_background_observations_after_hashing(
        self,
    ) -> None:
        preflight = load_physical_preflight()
        monitor = mock.Mock()
        monitor.snapshot.side_effect = (
            SimpleNamespace(
                captured_at_ns=99,
                error=None,
                stale=False,
                value="old-runtime",
            ),
            SimpleNamespace(
                captured_at_ns=99,
                error=None,
                stale=False,
                value="old-power",
            ),
            SimpleNamespace(
                captured_at_ns=101,
                error=None,
                stale=False,
                value="fresh-runtime",
            ),
            SimpleNamespace(
                captured_at_ns=101,
                error=None,
                stale=False,
                value="fresh-power",
            ),
        )
        with mock.patch.object(
            preflight.time, "monotonic_ns", return_value=100
        ), mock.patch.object(
            preflight.time, "monotonic", side_effect=(1.0, 1.1)
        ), mock.patch.object(preflight.time, "sleep") as sleep:
            values = preflight._refresh_background_observations(
                monitor, ("phone-runtime", "phone-power"), timeout_s=1
            )

        monitor.request_refresh.assert_called_once_with()
        self.assertEqual(values, {
            "phone-runtime": "fresh-runtime",
            "phone-power": "fresh-power",
        })
        sleep.assert_called_once_with(0.05)

    def test_preflight_rejects_stale_phone_router_binary(self) -> None:
        preflight = load_physical_preflight()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            router = root / "router.android"
            router.write_bytes(b"stale-router-without-terminal-control")
            router_sha256 = hashlib.sha256(router.read_bytes()).hexdigest()
            receipt = root / "router.sha256"
            receipt.write_text(
                router_sha256 + "  " + str(router) + "\n"
                + router_sha256 + "  /data/local/tmp/router\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                preflight.RigPreflightError,
                "terminal-control capability",
            ):
                preflight.validate_phone_router_deployment(router, receipt)

    def test_request_admission_fit_passes_disjoint_holdout(self) -> None:
        def rows(split: str) -> list[dict[str, object]]:
            result = []
            for index in range(20):
                features = {
                    "active_model_input_tokens": index * 11,
                    "active_model_output_tokens": index * 7,
                    "active_model_requests": index % 5,
                    "input_tokens": 32 + index * 3,
                    "output_tokens": 8 + index * 2,
                }
                result.append({
                    "event_id": f"{split}-{index}",
                    "features": features,
                    "latency_us": (
                        1000
                        + 3 * features["input_tokens"]
                        + 5 * features["output_tokens"]
                        + 7 * features["active_model_requests"]
                    ),
                    "request_index": index,
                    "split": split,
                })
            return result

        latency, audit = fit_role(rows("train"), rows("holdout"))
        self.assertEqual(
            set(latency["cost_us"]["coefficients"]), set(FEATURES)
        )
        self.assertTrue(latency["measured"])
        self.assertEqual(audit["holdout_upper_violations"], 0)
        self.assertEqual(
            audit["target"],
            "controller_wall_us_including_endpoint_queue",
        )

    def test_physical_arm_delegates_control_and_treatment_to_scheduler(self) -> None:
        source = RUN_SCRIPT.read_text(encoding="ascii")
        self.assertIn("run_fp16_small_overlay_arm.sh", source)
        self.assertIn("large_policy=runtime-auto", source)
        self.assertIn("small_policy=runtime-scheduler", source)
        self.assertIn("large_policy=cpu-overflow", source)
        self.assertIn("small_policy=static-cpu", source)
        self.assertNotIn("--hot-n-gpu-layers", source)
        self.assertNotIn("--cold-n-gpu-layers", source)
        self.assertNotIn("--max-columns", source)

    def test_nonexecuting_preflight_gates_the_physical_launcher(self) -> None:
        preflight = PHYSICAL_PREFLIGHT.read_text(encoding="ascii")
        launcher = (
            ROOT
            / "full_fp16_burstgpt_v1/small_model_overlay_v1/"
              "run_fp16_small_overlay_arm.sh"
        ).read_text(encoding="ascii")
        self.assertIn("run_physical_preflight", preflight)
        self.assertIn("probe_phone_runtime", preflight)
        self.assertIn("probe_phone_power", preflight)
        self.assertIn('"physical_inference_executed": False', preflight)
        self.assertNotIn("LlamaCppHttpClient", preflight)
        self.assertLess(
            launcher.index('python3 "$unified_preflight"'),
            launcher.index("systemd-run --user --scope"),
        )
        self.assertIn("PHYSICAL_PREFLIGHT.json", launcher)

    def test_direct_physical_launcher_passes_canonical_session_contract(
        self,
    ) -> None:
        shell = PHYSICAL_CAMPAIGN.read_text(encoding="ascii")
        launcher = (
            PHYSICAL_CAMPAIGN.parent / "launch.py"
        ).read_text(encoding="ascii")
        self.assertIn('"$here/launch.py" "$campaign" "$output"', shell)
        for argument in (
            "--adb",
            "--phone-usb-close",
            "--phone-session",
            "--phone-restore",
            "--phone-worker",
            "--phone-busybox",
            "--qwen-phone-model",
            "--gemma-phone-model",
            "--phone-session-root",
            "--phone-kernel-release",
            "--nmcli",
        ):
            self.assertIn(argument, launcher)
        self.assertIn(
            "configuration.evidence.transport_qualification_directories",
            launcher,
        )
        self.assertNotIn("--bridge-allocator", launcher)
        self.assertIn('"--selection-mode"', launcher)
        self.assertIn("configuration.campaign.selection_mode", launcher)
        self.assertNotIn("--forced", launcher)

    def test_fake_84_order_matches_the_physical_merged_trace(self) -> None:
        runner = load_unified_runner()
        large, overlay = runner.validate_trace(
            LARGE_TRACE,
            OVERLAY_DIR / "REQUESTS_LLAMA1B_10.jsonl",
            json.loads((OVERLAY_DIR / "TRACE_MANIFEST.json").read_text(
                encoding="ascii"
            )),
        )
        qwen = ModelManifest.from_json(json.loads(
            (UNIFIED_RUNNER.parent / "data/QWEN_MANIFEST.json").read_text(
                encoding="ascii"
            )
        ))
        gemma = ModelManifest.from_json(json.loads(
            (UNIFIED_RUNNER.parent / "data/GEMMA_MANIFEST.json").read_text(
                encoding="ascii"
            )
        ))
        llama_model_id = overlay[0]["execution_model_id"]
        merged = runner.merge_rows(large, overlay, {
            runner.QWEN_ROLE: qwen.model_id,
            runner.GEMMA_ROLE: gemma.model_id,
        })
        role = {
            qwen.model_id: "A",
            gemma.model_id: "B",
            llama_model_id: "C",
        }
        self.assertEqual(
            "".join(role[item["model_id"]] for item in merged),
            REAL_TRACE_ORDER,
        )
        self.assertEqual(
            tuple(
                (
                    item["row"]["arrival_us"]
                    - merged[0]["row"]["arrival_us"]
                ) // 100 + 100
                for item in merged
            ),
            REAL_TRACE_ARRIVALS_US,
        )

    def test_physical_smoke_subset_preserves_trace_identity(self) -> None:
        runner = load_unified_runner()
        merged = [
            {"combined_index": index, "row": {"event_id": f"r{index}"}}
            for index in range(6)
        ]
        selected = runner.select_rows(merged, "0,1,5")
        self.assertEqual(
            [row["combined_index"] for row in selected], [0, 1, 5]
        )
        with self.assertRaisesRegex(runner.UnifiedTraceError, "request indices"):
            runner.select_rows(merged, "1,1")
        with self.assertRaisesRegex(runner.UnifiedTraceError, "request indices"):
            runner.select_rows(merged, "6")

    def test_trace_energy_is_authoritative_and_request_windows_are_not_additive(
        self,
    ) -> None:
        runner = load_unified_runner()
        energy = RawEnergyMeasurement(
            energy_boundary_id="synthetic-paid-trace",
            fleet_energy_uj_by_domain={
                "cpu-package": 10,
                "gpu-board": 20,
                "whole-phone": 30,
            },
            measurement_evidence_ids=("synthetic-trace-meter",),
        )
        accounting = runner.trace_energy_accounting(energy)
        self.assertEqual(
            accounting["authoritative_total_field"], "trace_energy"
        )
        self.assertEqual(
            accounting["comparison_scope"], "trace_level_whole_fleet"
        )
        self.assertFalse(accounting["per_request_windows_additive"])
        self.assertEqual(
            accounting["per_request_energy_role"],
            "diagnostic_only",
        )

    def test_live_snapshot_parsers_bind_exact_hardware(self) -> None:
        self.assertEqual(
            parse_meminfo(
                "MemTotal: 15475004 kB\nMemAvailable: 11600000 kB\n"
            ),
            (15_846_404_096, 11_878_400_000),
        )
        gpu = parse_gpu(
            "GPU-other, Other, 100, 1\n"
            "GPU-target, NVIDIA GeForce RTX 4060 Ti, 16380, 779\n",
            expected_uuid="GPU-target",
            expected_name="NVIDIA GeForce RTX 4060 Ti",
        )
        self.assertEqual(gpu, (17_175_674_880, 816_840_704))


if __name__ == "__main__":
    unittest.main()
