#!/usr/bin/env python3

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from research_dev.scheduler import GGUFModelManifestLoader
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeControl,
    AdaptiveDecodeDirective,
    AdaptiveDecodeGroupedObservation,
    AdaptiveDecodePolicy,
    AdaptiveDecodePolicyAck,
    AdaptiveDecodeWindowBoundary,
    AdaptiveDecodeWindowReceipt,
)
from research_dev.scheduler._internal.runtime_plan import (
    RuntimeExecutionContract,
    RuntimeTransitionPlan,
)
from research_dev.scheduler.adapters import (
    CanonicalHttpExecutionBackend,
    LlamaCppCompletionPayload,
    LlamaCppHttpClient,
    LlamaServerProcessConfiguration,
    LlamaServerProcessLauncher,
    ManagedLlamaServer,
    PhysicalAdapterError,
    PhysicalExecutionCommand,
    PhysicalParticipantCommand,
    PhysicalTransitionCommand,
    RawEnergyMeasurement,
    llama_server_launch_contract,
    parse_llama_server_ffn_call,
    phone_ffn_execution_contract,
)
from research_dev.scheduler.adapters.output_quality import (
    accounting_output_assessment,
    assess_semantic_output,
)
from research_dev.scheduler.adapters.contracts import StalePhysicalSlotError
from research_dev.scheduler.adapters.http_backend import _AdaptivePayloadController
from research_dev.scheduler.adapters.residency import (
    physical_residency_parameters_match,
)

try:
    from .test_gguf_cost import write_synthetic_gguf
except ImportError:
    from test_gguf_cost import write_synthetic_gguf


PLAN_SHA256 = "sha256:" + "2" * 64
ARTIFACT_ROUTE = "synthetic:host-accelerator-helper"


def execution_command(artifact_sha256: str) -> PhysicalExecutionCommand:
    parameters = {
        "batch_size": 16,
        "context_size": 128,
        "cpu_device_id": "host-a",
        "ffn_activation": "geglu",
        "ffn_n_embd": 32,
        "ffn_timeout_ms": 5000,
        "ffn_transport": "functionfs-usb",
        "gpu_device_id": "accelerator-b",
        "gpu_layers": 1,
        "model_alias": "synthetic-model",
        "parallel": 1,
        "phone_device_id": "helper-c",
        "ubatch_size": 4,
        "usb_allocator": "devmem",
        "usb_batch_plan": "split-row",
        "usb_full_duplex": 1,
        "usb_max_payload_bytes": 256,
        "usb_product_id": 0x5678,
        "usb_queue_depth": 1,
        "usb_slot_safety_bytes": 64,
        "usb_split_h2d": 0,
        "usb_transport_generation": "synthetic-functionfs-v1",
        "usb_transport_profile_id": "synthetic-functionfs-profile-v1",
        "usb_vendor_id": 0x1234,
        "usbfs_available_bytes": 4096,
    }
    execution_contract = RuntimeExecutionContract(
        execution_mode="static-split",
        initial_split_fraction_ppm=500_000,
        allowed_adaptive_fractions_ppm=(),
        batch_plan="split-row",
        maximum_batch_size=1,
        queue_depth=1,
        phone_device_id="helper-c",
        phone_endpoint="synthetic://helper-c",
        operator_kind="ffn",
    )
    operator_plan = {
        "assisted_operator_kind": "ffn",
        "execution_contract": execution_contract.to_json(),
        "operators": [
            {
                "device_ids": ["accelerator-b"],
                "operator_id": "layer:0:attention",
                "operator_kind": "attention",
                "split_axis": "none",
                "split_fraction_ppm": 0,
            },
            {
                "device_ids": ["accelerator-b", "helper-c"],
                "operator_id": "layer:0:ffn",
                "operator_kind": "ffn",
                "split_axis": "column",
                "split_fraction_ppm": 500_000,
            },
        ],
        "plan_sha256": PLAN_SHA256,
        "route_id": ARTIFACT_ROUTE,
    }
    return PhysicalExecutionCommand(
        ticket_id="synthetic-request:attempt:0",
        request_id="synthetic-request",
        model_id="synthetic-model-id",
        artifact_sha256=artifact_sha256,
        route_id=ARTIFACT_ROUTE,
        executor_id="synthetic:composite",
        endpoint="http://127.0.0.1:19000",
        operator_plan_protocol="synthetic-plan-v1",
        operator_plan_sha256=PLAN_SHA256,
        planned_start_us=10,
        planned_finish_us=20,
        planned_finish_upper_us=30,
        operator_plan=operator_plan,
        participants=(
            PhysicalParticipantCommand(
                executor_id="synthetic:host",
                device_id="host-a",
                endpoint="synthetic://host-a",
                backend="synthetic",
                resource_ids=("compute:host-a",),
            ),
            PhysicalParticipantCommand(
                executor_id="synthetic:accelerator",
                device_id="accelerator-b",
                endpoint="synthetic://accelerator-b",
                backend="synthetic",
                resource_ids=("compute:accelerator-b",),
            ),
            PhysicalParticipantCommand(
                executor_id="synthetic:helper",
                device_id="helper-c",
                endpoint="synthetic://helper-c",
                backend="synthetic",
                resource_ids=("compute:helper-c",),
            ),
        ),
        leases=(),
        memory_reservations=(),
        transitions=(),
        execution_contract=execution_contract,
        adapter_parameters=parameters,
    )


def adaptive_execution_command(
    command: PhysicalExecutionCommand,
    *,
    selection_mode: str = "adaptive-decode",
) -> PhysicalExecutionCommand:
    parameters = {
        **dict(command.adapter_parameters),
        "ffn_assistance_phase": "decode",
        "ffn_runtime_control_protocol": "decode-boundary-v1",
    }
    parameters.setdefault(
        "ffn_max_tokens", command.execution_contract.maximum_batch_size
    )
    maximum_batch_size = parameters.get(
        "ffn_max_tokens",
        command.execution_contract.maximum_batch_size,
    )
    if command.decode_cohort is not None:
        maximum_batch_size = max(
            maximum_batch_size,
            int(command.decode_cohort["maximum_members"]),
        )
    contract = RuntimeExecutionContract(
        execution_mode="adaptive-split",
        initial_split_fraction_ppm=0,
        allowed_adaptive_fractions_ppm=(0, 500_000),
        batch_plan=str(parameters.get(
            "usb_batch_plan", command.execution_contract.batch_plan
        )),
        maximum_batch_size=maximum_batch_size,
        queue_depth=int(parameters.get(
            "usb_queue_depth", command.execution_contract.queue_depth
        )),
        phone_device_id=command.execution_contract.phone_device_id,
        phone_endpoint=command.execution_contract.phone_endpoint,
        operator_kind="ffn",
    )
    return replace(
        command,
        adapter_parameters=parameters,
        execution_contract=contract,
        operator_plan={
            **dict(command.operator_plan),
            "assisted_operator_kind": "ffn",
            "execution_contract": contract.to_json(),
        },
        selection_mode=selection_mode,
    )


class FakeClient(LlamaCppHttpClient):
    def __init__(self):
        super().__init__(lambda *_args: [
            {"id": 2, "id_task": 1, "is_processing": True},
            {"id": 3, "id_task": 2, "is_processing": True},
        ])

    def complete(
        self,
        endpoint,
        payload,
        control_check,
        *,
        scheduler_headers=None,
    ):
        control_check()
        return {"stream_sha256": "3" * 64, "tokens": [1]}


class FakeMeter:
    def measure(self, _started_ns, _finished_ns):
        return RawEnergyMeasurement(
            energy_boundary_id="synthetic-boundary",
            fleet_energy_uj_by_domain={"synthetic-domain": 1},
            transfer_energy_uj_by_link={},
            measurement_evidence_ids=("synthetic-meter",),
            attribution_kind="isolated",
        )


class RecordingMeter(FakeMeter):
    def __init__(self) -> None:
        self.intervals = []
        self.prepare_count = 0

    def prepare(self) -> None:
        self.prepare_count += 1

    def measure(self, started_ns, finished_ns):
        self.intervals.append((started_ns, finished_ns))
        return super().measure(started_ns, finished_ns)


class LlamaServerAdapterTests(unittest.TestCase):
    def test_semantic_quality_accepts_coherent_nonexact_text(self):
        assessment = assess_semantic_output(
            "A concise answer can differ in wording and remain useful.",
            tuple(range(16)),
        )
        self.assertTrue(assessment.accepted)
        self.assertEqual(assessment.to_json()["mode"], "semantic-sanity-v1")

    def test_semantic_quality_rejects_degenerate_output(self):
        assessment = assess_semantic_output("word " * 32, (7,) * 32)
        self.assertFalse(assessment.accepted)
        self.assertIn(
            "DEGENERATE_TOKEN_REPETITION", assessment.reasons
        )

    def test_internal_warmup_uses_accounting_only_quality(self):
        assessment = accounting_output_assessment("!", (7, 8))

        self.assertTrue(assessment.accepted)
        self.assertEqual(
            assessment.to_json()["mode"], "accounting-only-v1"
        )

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        model = self.root / "model.gguf"
        write_synthetic_gguf(model)
        self.manifest = GGUFModelManifestLoader.load(
            "synthetic-model-id", model
        )
        self.command = execution_command(self.manifest.artifact_sha256)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def managed_server(self) -> ManagedLlamaServer:
        server = ManagedLlamaServer(
            ("synthetic-server",),
            {},
            self.root,
            "synthetic",
            llama_server_launch_contract(self.command, self.manifest),
        )
        server.process = SimpleNamespace(poll=lambda: None)
        return server

    def test_cpu_only_server_disables_cuda_kv_offload(self) -> None:
        server_path = self.root / "llama-server"
        server_path.write_text("#!/bin/sh\n", encoding="ascii")
        server_path.chmod(0o755)
        configuration = LlamaServerProcessConfiguration(
            server_path=server_path,
            model_paths_by_artifact={
                self.manifest.artifact_sha256: self.root / "model.gguf",
            },
            library_paths_by_device={},
            executable_device_names={},
            output_directory=self.root,
        )
        contract = replace(
            llama_server_launch_contract(self.command, self.manifest),
            gpu_layers=0,
            phone_device_id=None,
            ffn_environment={},
        )

        class FakeManagedServer:
            def __init__(
                self, command, environment, output_directory, label,
                launch_contract,
            ):
                self.command = command
                self.environment = environment
                self.process = SimpleNamespace(poll=lambda: None)
                self.stderr_lines = []

            def start(self):
                return None

            def stop(self):
                return None

        launcher = LlamaServerProcessLauncher(configuration)
        with patch(
            "research_dev.scheduler.adapters.llama_server.ManagedLlamaServer",
            FakeManagedServer,
        ), patch.object(launcher, "_healthy", return_value=True):
            server = launcher._launch_contract(
                "http://127.0.0.1:19000",
                contract,
                self.manifest,
                label="cpu-only",
                control_check=lambda: None,
            )

        self.assertIn("--no-kv-offload", server.command)
        self.assertEqual(server.command[server.command.index("--device") + 1], "none")
        self.assertEqual(server.environment["CUDA_VISIBLE_DEVICES"], "")

        with patch("research_dev.scheduler.adapters.llama_server.ManagedLlamaServer", FakeManagedServer), patch.object(
                launcher, "_healthy", return_value=True), patch.dict(os.environ, {"S41_SERVER_LOGITS_TRACE": "/tmp/inherited.bin"}):
            ordinary = launcher._launch_contract("http://127.0.0.1:19000", contract, self.manifest,
                label="trace-default", control_check=lambda: None)
            self.assertNotIn("S41_SERVER_LOGITS_TRACE", ordinary.environment)
            with self.assertRaisesRegex(PhysicalAdapterError, "raw logits tracing support"):
                launcher._launch_contract("http://127.0.0.1:19000", replace(contract,
                    logits_trace_path=str(self.root / "LOGITS.bin")), self.manifest,
                    label="trace-unsupported", control_check=lambda: None)

    def test_native_slot_allocation_receipt_ignores_child_and_stream_text(self) -> None:
        server = self.managed_server()
        line = "0.01.000.123 I slot launch_slot_: id  1 | task 7 | processing task, is_child = 0"
        server.stderr_lines.extend([line, line.replace("is_child = 0", "is_child = 1"),
                                    '0.02 I srv operator(): data: {"content": "' + line + '"}'])
        server.stderr_observed_epoch_us.extend([123, 124, 125])
        self.assertEqual(server.slot_allocation_events(), ())
        prompt = "0.01.000.124 I slot operator(): id 1 | task 7 | new prompt, n_ctx_slot = 1024, n_keep = 0, task.n_tokens = 256"
        server.stderr_lines.append(prompt)
        server.stderr_observed_epoch_us.append(126)
        self.assertEqual(server.slot_allocation_events(), ({"slot_id": 1, "task_id": 7,
            "observed_epoch_us": 123, "line": line, "prompt_tokens": 256, "prefill_line": prompt,
            "prefill_observed_epoch_us": 126},))

    def test_layer_zero_ffn_call_is_valid(self) -> None:
        call = parse_llama_server_ffn_call(
            "S41SERVERFFNCALL request=1 layer=0 tokens=1 "
            "columns=64 payload_bytes=64"
        )
        self.assertIsNotNone(call)
        self.assertEqual(call.layer, 0)

    def test_ffn_call_with_exact_llama_log_prefix_is_valid(self) -> None:
        request_hex = "qwen-request".encode("ascii").hex()
        call = parse_llama_server_ffn_call(
            "2.23.969.698 D S41SERVERFFNCALL "
            f"context={request_hex}:2:1:3 request=332 layer=7 "
            "tokens=1 columns=6528 payload_bytes=10240"
        )
        self.assertIsNotNone(call)
        self.assertEqual(call.request_id, 332)
        self.assertEqual(call.contexts[0].scheduler_request_id, "qwen-request")

    def test_managed_server_exposes_timestamped_ffn_calls(self) -> None:
        request_hex = "qwen-request".encode("ascii").hex()
        server = self.managed_server()
        server.stderr_observed_epoch_us.append(123_456)
        server.stderr_lines.append(
            "S41SERVERFFNCALL "
            f"context={request_hex}:2:1:3 request=332 layer=7 "
            "tokens=1 columns=6528 payload_bytes=10240"
        )

        self.assertEqual(server.ffn_call_events(), ({
            "columns": 6528,
            "contexts": [{
                "plan_generation": 3,
                "rows": 1,
                "scheduler_request_id": "qwen-request",
                "server_slot_id": 2,
            }],
            "layer": 7,
            "line_index": 0,
            "observed_epoch_us": 123_456,
            "payload_bytes": 10240,
            "request_id": 332,
            "tokens": 1,
        },))

    def test_ffn_call_after_interleaved_log_timestamp_is_valid(self) -> None:
        request_hex = "gemma-request".encode("ascii").hex()
        call = parse_llama_server_ffn_call(
            "2.00.243.917 S41SERVERFFNCALL "
            f"context={request_hex}:0:1:3 request=1610 layer=1 "
            "tokens=1 columns=15360 payload_bytes=7680"
        )
        self.assertIsNotNone(call)
        self.assertEqual(call.request_id, 1610)
        self.assertEqual(call.layer, 1)
        self.assertEqual(call.contexts[0].scheduler_request_id, "gemma-request")

    def test_ffn_call_with_unknown_prefix_is_not_proof(self) -> None:
        self.assertIsNone(parse_llama_server_ffn_call(
            "noise S41SERVERFFNCALL request=1 layer=0 tokens=1 "
            "columns=64 payload_bytes=64"
        ))

    def test_legacy_negative_progress_slot_uses_unique_active_slot(self) -> None:
        self.assertEqual(
            LlamaCppHttpClient._progress_slot(-1, {6: 17}), 6
        )

    def test_legacy_negative_progress_slot_rejects_ambiguity(self) -> None:
        with self.assertRaisesRegex(
            PhysicalAdapterError, "cannot be resolved"
        ):
            LlamaCppHttpClient._progress_slot(-1, {2: 11, 6: 17})

    def test_runtime_decode_progress_precedes_existing_callback(self) -> None:
        calls = []
        scheduler = type("Scheduler", (), {})()
        for name in (
            "start_adaptive_decode",
            "adaptive_decode_boundary",
            "seal_adaptive_decode_tail",
            "record_adaptive_decode_window",
            "acknowledge_adaptive_decode_control",
            "fail_adaptive_decode_control",
            "preview_adaptive_decode_completion",
            "complete_adaptive_decode",
        ):
            setattr(scheduler, name, lambda *args, **kwargs: None)
        scheduler.record_runtime_decode_progress = (
            lambda request_id, **values: calls.append(
                ("scheduler", request_id, values["token_index"])
            )
        )
        backend = CanonicalHttpExecutionBackend(
            FakeClient(), FakeMeter(), epoch_ns=time.monotonic_ns() - 1_000_000
        )
        backend.bind_scheduler(scheduler)
        payload = LlamaCppCompletionPayload(
            request_id=self.command.request_id,
            expected_model_alias="synthetic-model",
            input_tokens=1,
            output_tokens=4,
            prompt_tokens=(1,),
            seed=1,
            stream_path=self.root / "runtime-progress.raw",
            on_first_token=lambda _value: None,
            on_decode_progress=lambda _slot, token, _at, _terminal: (
                calls.append(("original", self.command.request_id, token))
            ),
        )

        wrapped = backend._runtime_progress_payload(self.command, payload)
        wrapped.on_decode_progress(2, 3, time.monotonic_ns(), False)

        self.assertEqual(
            calls,
            [
                ("scheduler", self.command.request_id, 3),
                ("original", self.command.request_id, 3),
            ],
        )

    def test_first_decode_progress_only_opens_adaptive_window(self) -> None:
        class Scheduler:
            def __init__(self):
                self.starts = []
                self.boundaries = []

            def start_adaptive_decode(self, request_id, **values):
                self.starts.append((request_id, values))

            def adaptive_decode_boundary(self, request_id, **values):
                self.boundaries.append((request_id, values))
                return None

            def seal_adaptive_decode_tail(self, *args, **kwargs):
                return None

            def record_adaptive_decode_window(self, *args, **kwargs):
                raise AssertionError("window measurement was not requested")

            def acknowledge_adaptive_decode_control(self, *args, **kwargs):
                raise AssertionError("control acknowledgement was not requested")

            def fail_adaptive_decode_control(self, *args, **kwargs):
                raise AssertionError("control failure was not requested")

            def preview_adaptive_decode_completion(self, *args, **kwargs):
                raise AssertionError("completion was not requested")

            def complete_adaptive_decode(self, *args, **kwargs):
                raise AssertionError("completion was not requested")

        command = adaptive_execution_command(self.command)
        scheduler = Scheduler()
        backend = CanonicalHttpExecutionBackend(
            FakeClient(), FakeMeter(), epoch_ns=time.monotonic_ns() - 1_000_000
        )
        backend.bind_scheduler(scheduler)
        payload = LlamaCppCompletionPayload(
            request_id=command.request_id,
            expected_model_alias="synthetic-model",
            input_tokens=1,
            output_tokens=4,
            prompt_tokens=(1,),
            seed=1,
            stream_path=self.root / "adaptive-stream.raw",
            on_first_token=lambda _value: None,
        )
        adaptive = backend._adaptive_payload(
            command, payload, time.monotonic_ns()
        )
        self.assertIsNotNone(adaptive.on_decode_progress)
        adaptive.on_decode_progress(2, 1, time.monotonic_ns(), False)
        self.assertEqual(len(scheduler.starts), 1)
        self.assertEqual(scheduler.boundaries, [])
        adaptive.on_decode_progress(2, 2, time.monotonic_ns(), False)
        self.assertEqual(len(scheduler.boundaries), 1)
        timings = tuple(row for row in backend.adaptive_timing_events
                        if row["kind"] == "DECODE_BOUNDARY_OBSERVED")
        self.assertEqual([row["token_index"] for row in timings], [1, 2])
        self.assertTrue(all(row["kind"] == "DECODE_BOUNDARY_OBSERVED"
                            and row["request_id"] == command.request_id
                            and row["ticket_id"] == command.ticket_id
                            for row in timings))

    def test_adaptive_control_and_stats_use_bounded_service_budget(self):
        command = adaptive_execution_command(replace(
            self.command,
            planned_finish_upper_us=self.command.planned_start_us + 20_000_000,
        ))
        for request_timeout in (12, 3):
            with self.subTest(request_timeout=request_timeout):
                budgets = []

                def read_or_apply(*_args, timeout_s=5):
                    budgets.append(timeout_s)
                    prefill_s = min(10.289, request_timeout - 0.001)
                    if timeout_s < prefill_s:
                        raise TimeoutError("control queued behind prefill")
                    return {}, time.monotonic_ns()

                client = FakeClient()
                client.read_ffn_stats = read_or_apply
                client.apply_ffn_control = read_or_apply
                client.apply_ffn_cohort_control = read_or_apply
                backend = CanonicalHttpExecutionBackend(
                    client, FakeMeter(), epoch_ns=time.monotonic_ns()
                )
                backend.bind_scheduler(Mock(runtime_ticket=None))
                payload = LlamaCppCompletionPayload(
                    request_id=command.request_id,
                    expected_model_alias="synthetic-model", input_tokens=1,
                    output_tokens=19, prompt_tokens=(1,), seed=1,
                    stream_path=self.root / ("control-budget-" + str(request_timeout)),
                    on_first_token=lambda _value: None, timeout_s=request_timeout,
                )
                controller = _AdaptivePayloadController(
                    backend, command, payload, None, None
                )
                controller._read_boundary_stats(SimpleNamespace(slot_id=2))
                control = AdaptiveDecodeControl(
                    request_id=command.request_id, slot_id=2, plan_generation=1,
                    policy=AdaptiveDecodePolicy(
                        route_id="desktop-parent", executor_id="desktop",
                        operator_plan_sha256=PLAN_SHA256,
                        desktop_parent_route_id="desktop-parent",
                        desktop_placement_sha256="sha256:" + "4" * 64,
                        layer_indices=(), layer_mask=0, columns=0,
                        split_fraction_ppm=0, baseline=True,
                        resource_ids=("compute:accelerator-b",),
                    ),
                )
                controller._send_control(control, ())
                controller.cohort = object()
                controller._send_control(control, ((command.request_id, 2), ("peer", 3)))
                self.assertEqual(budgets, [request_timeout] * 3)

    def test_delayed_initial_adaptive_ack_records_transition_window(
        self,
    ) -> None:
        placement = "sha256:" + "4" * 64
        policy = AdaptiveDecodePolicy(
            route_id=ARTIFACT_ROUTE,
            executor_id="synthetic:composite",
            operator_plan_sha256=PLAN_SHA256,
            desktop_parent_route_id="synthetic:desktop-parent",
            desktop_placement_sha256=placement,
            layer_indices=(0,),
            layer_mask=1,
            columns=16,
            split_fraction_ppm=500_000,
            resource_ids=("compute:accelerator-b", "compute:helper-c"),
        )

        class Client(FakeClient):
            def __init__(self, acknowledged_ns):
                self.acknowledged_ns = acknowledged_ns

            def apply_ffn_control(self, _endpoint, control, *, timeout_s=5):
                return ({
                    "applied_token_index": 2,
                    "plan_generation": control.plan_generation,
                    "policy_hash": control.policy.policy_hash,
                    "runtime_stats": {
                        "batched_calls": 0,
                        "calls": 1,
                        "configured_queue_depth": 4,
                        "desktop_compute_us": 20,
                        "download_bytes": 64,
                        "exposed_tail_us": 5,
                        "input_rows": 1,
                        "maximum_active_slots": 1,
                        "maximum_outstanding_transfers": 1,
                        "maximum_tokens": 1,
                        "phone_compute_us": 10,
                        "rpc_us": 4,
                        "transfer_subrequests": 1,
                        "upload_bytes": 64,
                        "usb_d2h_us": 3,
                        "usb_h2d_us": 2,
                        "usb_transfer_us": 5,
                        "useful_overlap_us": 1,
                    },
                    "slot_id": control.slot_id,
                    "success": True,
                }, self.acknowledged_ns)

        class Scheduler:
            def __init__(self):
                self.observation = None

            def start_adaptive_decode(self, request_id, **values):
                return SimpleNamespace(
                    boundary=None,
                    control=AdaptiveDecodeControl(
                        request_id=request_id,
                        slot_id=values["slot_id"],
                        plan_generation=1,
                        policy=policy,
                    ),
                )

            def acknowledge_adaptive_decode_control(
                self, _request_id, _acknowledgement, **values
            ):
                self.observation = values["transition_observation"]
                self.assert_transition_observation()

            def assert_transition_observation(self):
                if self.observation is None:
                    raise AssertionError(
                        "delayed initial acknowledgement lacks observation"
                    )

            def adaptive_decode_boundary(self, *_args, **_kwargs):
                return None

            def seal_adaptive_decode_tail(self, *_args, **_kwargs):
                return None

            def record_adaptive_decode_window(self, *_args, **_kwargs):
                raise AssertionError("window measurement was not requested")

            def fail_adaptive_decode_control(self, *_args, **_kwargs):
                raise AssertionError("control failure was not requested")

            def preview_adaptive_decode_completion(self, *_args, **_kwargs):
                raise AssertionError("completion was not requested")

            def complete_adaptive_decode(self, *_args, **_kwargs):
                raise AssertionError("completion was not requested")

        command = adaptive_execution_command(self.command)
        progress_ns = time.monotonic_ns()
        scheduler = Scheduler()
        backend = CanonicalHttpExecutionBackend(
            Client(progress_ns + 1_000_000),
            FakeMeter(),
            epoch_ns=progress_ns - 1_000_000,
        )
        backend.bind_scheduler(scheduler)
        payload = LlamaCppCompletionPayload(
            request_id=command.request_id,
            expected_model_alias="synthetic-model",
            input_tokens=1,
            output_tokens=4,
            prompt_tokens=(1,),
            seed=1,
            stream_path=self.root / "delayed-initial-ack.raw",
            on_first_token=lambda _value: None,
        )

        controlled = backend._adaptive_payload(command, payload, progress_ns)
        controlled.on_decode_progress(2, 1, progress_ns, False)

        self.assertIsNotNone(scheduler.observation)
        self.assertEqual(scheduler.observation.accounting_token_count, None)
        self.assertEqual(scheduler.observation.completed_phone_calls, 1)
        issued = next(row for row in backend.adaptive_timing_events
                      if row["kind"] == "CONTROL_ISSUED")
        self.assertEqual(issued["request_id"], command.request_id)
        self.assertEqual(issued["control"]["policy_hash"], policy.policy_hash)
        self.assertEqual(issued["control"]["layer_mask"], policy.layer_mask)
        self.assertGreaterEqual(issued["observed_at_us"], 0)

    def test_buffered_progress_after_control_ack_is_ignored(self) -> None:
        placement = "sha256:" + "4" * 64
        policy = AdaptiveDecodePolicy(
            route_id=ARTIFACT_ROUTE,
            executor_id="synthetic:composite",
            operator_plan_sha256=PLAN_SHA256,
            desktop_parent_route_id="synthetic:desktop-parent",
            desktop_placement_sha256=placement,
            layer_indices=(0,),
            layer_mask=1,
            columns=16,
            split_fraction_ppm=500_000,
            resource_ids=("compute:accelerator-b", "compute:helper-c"),
        )

        class Client(FakeClient):
            def __init__(self, acknowledged_ns):
                self.acknowledged_ns = acknowledged_ns

            def apply_ffn_control(self, _endpoint, control, *, timeout_s=5):
                return ({
                    "applied_token_index": 2,
                    "plan_generation": control.plan_generation,
                    "policy_hash": control.policy.policy_hash,
                    "runtime_stats": dict.fromkeys((
                        "batched_calls",
                        "calls",
                        "configured_queue_depth",
                        "desktop_compute_us",
                        "download_bytes",
                        "exposed_tail_us",
                        "input_rows",
                        "maximum_active_slots",
                        "maximum_outstanding_transfers",
                        "maximum_tokens",
                        "phone_compute_us",
                        "rpc_us",
                        "transfer_subrequests",
                        "upload_bytes",
                        "usb_d2h_us",
                        "usb_h2d_us",
                        "usb_transfer_us",
                        "useful_overlap_us",
                    ), 0),
                    "slot_id": control.slot_id,
                    "success": True,
                }, self.acknowledged_ns)

        class Scheduler:
            def __init__(self):
                self.boundaries = []

            def start_adaptive_decode(self, request_id, **values):
                return SimpleNamespace(
                    boundary=None,
                    control=AdaptiveDecodeControl(
                        request_id=request_id,
                        slot_id=values["slot_id"],
                        plan_generation=1,
                        policy=policy,
                    ),
                )

            def acknowledge_adaptive_decode_control(self, *_args, **_kwargs):
                return None

            def adaptive_decode_boundary(self, request_id, **values):
                self.boundaries.append((request_id, values))
                return None

            def seal_adaptive_decode_tail(self, *_args, **_kwargs):
                return None

            def record_adaptive_decode_window(self, *_args, **_kwargs):
                raise AssertionError("window measurement was not requested")

            def fail_adaptive_decode_control(self, *_args, **_kwargs):
                raise AssertionError("control failure was not requested")

            def preview_adaptive_decode_completion(self, *_args, **_kwargs):
                raise AssertionError("completion was not requested")

            def complete_adaptive_decode(self, *_args, **_kwargs):
                raise AssertionError("completion was not requested")

        command = adaptive_execution_command(self.command)
        first_ns = time.monotonic_ns()
        acknowledged_ns = first_ns + 1_000_000
        scheduler = Scheduler()
        backend = CanonicalHttpExecutionBackend(
            Client(acknowledged_ns), FakeMeter(), epoch_ns=first_ns
        )
        backend.bind_scheduler(scheduler)
        payload = LlamaCppCompletionPayload(
            request_id=command.request_id,
            expected_model_alias="synthetic-model",
            input_tokens=1,
            output_tokens=8,
            prompt_tokens=(1,),
            seed=1,
            stream_path=self.root / "buffered-progress.raw",
            on_first_token=lambda _value: None,
        )
        controlled = backend._adaptive_payload(command, payload, first_ns)

        self.assertTrue(controlled.on_decode_progress(2, 1, first_ns, False))
        self.assertFalse(controlled.on_decode_progress(
            2, 3, acknowledged_ns + 999, False
        ))
        self.assertTrue(controlled.on_decode_progress(
            2, 4, acknowledged_ns + 2_000, False
        ))
        self.assertEqual(len(scheduler.boundaries), 1)
        self.assertEqual(scheduler.boundaries[0][1]["token_index"], 4)

        with self.subTest(acknowledgement_in_release_guard=True):
            tail = backend._adaptive_payload(
                command, replace(payload, output_tokens=4), first_ns
            )
            with patch.object(
                scheduler, "acknowledge_adaptive_decode_control",
                return_value=AdaptiveDecodeDirective(
                    state="EXPLOITING", reason="WINDOW_OPENED",
                    target_token_index=4,
                ),
            ), patch.object(scheduler, "seal_adaptive_decode_tail") as seal:
                tail.on_decode_progress(2, 1, first_ns, False)
                seal.assert_called_once_with(
                    command.request_id, slot_id=2, token_index=2,
                    reason="server_release_guard",
                )
                self.assertFalse(tail.on_decode_progress(
                    2, 4, acknowledged_ns + 2_000, True
                ))
                self.assertEqual(len(scheduler.boundaries), 1)

        with self.subTest(ready_subset_expands_while_control_is_pending=True):
            expanded = replace(policy, layer_indices=(0, 1), layer_mask=3)
            followup = AdaptiveDecodeControl(
                request_id=command.request_id,
                slot_id=2,
                plan_generation=2,
                policy=expanded,
            )
            controls = backend._adaptive_payload(command, payload, first_ns)
            with patch.object(
                scheduler, "acknowledge_adaptive_decode_control",
                side_effect=(
                    AdaptiveDecodeDirective(
                        state="PROBING", reason="CONTROL_REQUIRED",
                        target_token_index=None, control=followup,
                    ),
                    AdaptiveDecodeDirective(
                        state="PROBING", reason="WINDOW_OPENED",
                        target_token_index=4,
                    ),
                ),
            ) as acknowledge, patch.object(
                backend._client, "apply_ffn_control",
                wraps=backend._client.apply_ffn_control,
            ) as apply:
                controls.on_decode_progress(2, 1, first_ns, False)
                self.assertEqual(apply.call_count, 2)
                self.assertEqual(acknowledge.call_count, 2)
                self.assertEqual(apply.call_args_list[1].args[1], followup)

    def test_calibration_phone_plan_activates_adaptive_physical_path(
        self,
    ) -> None:
        class Scheduler:
            def __init__(self):
                self.starts = []

            def start_adaptive_decode(self, request_id, **values):
                self.starts.append((request_id, values))

            def adaptive_decode_boundary(self, *_args, **_kwargs):
                return None

            def seal_adaptive_decode_tail(self, *_args, **_kwargs):
                return None

            def record_adaptive_decode_window(self, *_args, **_kwargs):
                raise AssertionError("window measurement was not requested")

            def acknowledge_adaptive_decode_control(
                self, *_args, **_kwargs
            ):
                raise AssertionError("control acknowledgement was not requested")

            def fail_adaptive_decode_control(self, *_args, **_kwargs):
                raise AssertionError("control failure was not requested")

            def preview_adaptive_decode_completion(
                self, *_args, **_kwargs
            ):
                raise AssertionError("completion was not requested")

            def complete_adaptive_decode(self, *_args, **_kwargs):
                raise AssertionError("completion was not requested")

        parameters = {
            **dict(self.command.adapter_parameters),
            "ffn_assistance_phase": "decode",
            "ffn_runtime_control_protocol": "decode-boundary-v1",
            "phone_device_id": "helper-c",
        }
        placement = "sha256:" + "4" * 64
        command = adaptive_execution_command(
            replace(
                self.command,
                adapter_parameters=parameters,
                operator_plan={
                    **dict(self.command.operator_plan),
                    "baseline_executor_id": "synthetic:desktop-parent",
                    "desktop_placement_sha256": placement,
                    "resource_ids": [
                        "compute:accelerator-b",
                        "compute:helper-c",
                    ],
                    "split_fraction_ppm": 500_000,
                },
            ),
            selection_mode="calibration",
        )
        scheduler = Scheduler()
        backend = CanonicalHttpExecutionBackend(
            FakeClient(), FakeMeter(), epoch_ns=time.monotonic_ns() - 1_000_000
        )
        backend.bind_scheduler(scheduler)
        payload = LlamaCppCompletionPayload(
            request_id=command.request_id,
            expected_model_alias="synthetic-model",
            input_tokens=1,
            output_tokens=4,
            prompt_tokens=(1,),
            seed=1,
            stream_path=self.root / "calibration-adaptive-stream.raw",
            on_first_token=lambda _value: None,
        )

        controlled = backend._adaptive_payload(
            command, payload, time.monotonic_ns()
        )

        self.assertIsNotNone(controlled.on_decode_progress)
        controlled.on_decode_progress(2, 1, time.monotonic_ns(), False)
        self.assertEqual(len(scheduler.starts), 1)
        self.assertEqual(
            command.execution_contract.execution_mode, "adaptive-split"
        )
        self.assertEqual(
            command.operator_plan["execution_contract"],
            command.execution_contract.to_json(),
        )

        resident = phone_ffn_execution_contract(command, self.manifest)
        baseline = AdaptiveDecodePolicy(
            route_id="synthetic:desktop-parent",
            executor_id="synthetic:desktop",
            operator_plan_sha256=PLAN_SHA256,
            desktop_parent_route_id="synthetic:desktop-parent",
            desktop_placement_sha256=placement,
            layer_indices=(),
            layer_mask=0,
            columns=0,
            split_fraction_ppm=0,
            resource_ids=("compute:accelerator-b",),
            baseline=True,
        )
        phone = AdaptiveDecodePolicy(
            route_id=command.route_id,
            executor_id=command.executor_id,
            operator_plan_sha256=command.operator_plan_sha256,
            desktop_parent_route_id=baseline.route_id,
            desktop_placement_sha256=placement,
            layer_indices=resident.layer_indices,
            layer_mask=resident.layer_mask,
            columns=resident.columns,
            split_fraction_ppm=500_000,
            resource_ids=(
                "compute:accelerator-b",
                "compute:helper-c",
            ),
        )
        first = AdaptiveDecodeWindowReceipt(
            request_id=command.request_id,
            slot_id=2,
            window_index=0,
            token_start=1,
            token_end=2,
            context_length=2,
            active_batch=1,
            started_at_us=100,
            finished_at_us=200,
            policy=baseline,
            applied_ack=None,
            fleet_energy_uj_by_domain={"fleet": 100},
            latency_per_token_us=100,
            phone_compute_us=0,
            usb_transfer_us=0,
            rpc_us=0,
            exposed_tail_us=0,
            output_valid=True,
            evidence_ids=("synthetic:calibration-baseline",),
            energy_boundary_id="synthetic-fleet",
            energy_attribution_kind="isolated",
            failure_reason=None,
            previous_record_sha256="0" * 64,
        )
        second = AdaptiveDecodeWindowReceipt(
            request_id=command.request_id,
            slot_id=2,
            window_index=1,
            token_start=2,
            token_end=3,
            context_length=3,
            active_batch=1,
            started_at_us=200,
            finished_at_us=300,
            policy=phone,
            applied_ack=None,
            fleet_energy_uj_by_domain={"fleet": 80},
            latency_per_token_us=100,
            phone_compute_us=20,
            usb_transfer_us=10,
            rpc_us=5,
            exposed_tail_us=1,
            output_valid=True,
            evidence_ids=("synthetic:calibration-phone",),
            energy_boundary_id="synthetic-fleet",
            energy_attribution_kind="isolated",
            failure_reason=None,
            previous_record_sha256=first.record_sha256.removeprefix(
                "sha256:"
            ),
        )
        grouped = AdaptiveDecodeGroupedObservation(
            request_id=command.request_id,
            ticket_id=command.ticket_id,
            model_artifact_sha256=command.artifact_sha256,
            planning_profile_sha256=PLAN_SHA256,
            desktop_placement_sha256=placement,
            windows=(first, second),
            final_policy=phone,
            terminal_status="COMPLETED",
            state_history=("BASELINE", "PROBING", "COMPLETED"),
        )
        server = ManagedLlamaServer(
            ("synthetic-server",),
            {},
            self.root,
            "synthetic-calibration-adaptive",
            llama_server_launch_contract(command, self.manifest),
        )
        server.process = SimpleNamespace(poll=lambda: None)
        marker = server.begin_execution(command, self.manifest)
        server.stderr_lines.append(
            "S41SERVERFFNCALL request=31 layer=0 tokens=1 "
            f"columns={resident.columns} "
            f"payload_bytes={self.manifest.embedding_length * 2}"
        )

        proof = server.finish_execution(
            marker,
            command,
            self.manifest,
            output_tokens=3,
            adaptive_observation=grouped,
        ).to_json()

        self.assertEqual(proof["phone_call_count"], 1)
        self.assertEqual(
            proof["operator_plan_sha256"], command.operator_plan_sha256
        )
        self.assertEqual(proof["ticket_id"], command.ticket_id)

        tail_ack = AdaptiveDecodePolicyAck(
            request_id=command.request_id, slot_id=2, plan_generation=61,
            applied_token_index=2, applied_at_us=200,
            policy_hash=phone.policy_hash,
        )
        tail_group = replace(
            grouped, windows=(first,), unmeasured_tail_tokens=1,
            unmeasured_tail_reason="server_release_guard",
            final_policy_ack=tail_ack,
        )
        for generation in (61, 62):
            with self.subTest(tail_generation=generation):
                tail_server = ManagedLlamaServer(
                    ("synthetic-server",), {}, self.root,
                    "synthetic-tail-ack-" + str(generation),
                    llama_server_launch_contract(command, self.manifest),
                )
                tail_server.process = SimpleNamespace(poll=lambda: None)
                tail_marker = tail_server.begin_execution(command, self.manifest)
                tail_server.stderr_lines.append(
                    "S41SERVERFFNCALL context="
                    + command.request_id.encode("ascii").hex()
                    + ":2:1:" + str(generation)
                    + " request=31 layer=0 tokens=1 "
                    + f"columns={resident.columns} "
                    + f"payload_bytes={self.manifest.embedding_length * 2}"
                )
                if generation == tail_ack.plan_generation:
                    tail_proof = tail_server.finish_execution(
                        tail_marker, command, self.manifest, output_tokens=3,
                        adaptive_observation=tail_group,
                    )
                    self.assertEqual(tail_proof.phone_call_count, 1)
                else:
                    with self.assertRaisesRegex(
                        PhysicalAdapterError, "phone calls differ from adaptive windows"
                    ):
                        tail_server.finish_execution(
                            tail_marker, command, self.manifest, output_tokens=3,
                            adaptive_observation=tail_group,
                        )

        measured_phone = replace(second, applied_ack=tail_ack)
        released_tail = replace(
            grouped, windows=(first, measured_phone), unmeasured_tail_tokens=3,
            unmeasured_tail_reason="server_release_guard",
        )
        for count, generation in ((3, 61), (2, 61), (4, 61), (3, 62)):
            with self.subTest(released_tail_calls=count, generation=generation):
                tail_server = ManagedLlamaServer(
                    ("synthetic-server",), {}, self.root,
                    f"synthetic-released-tail-{count}-{generation}",
                    llama_server_launch_contract(command, self.manifest),
                )
                tail_server.process = SimpleNamespace(poll=lambda: None)
                tail_marker = tail_server.begin_execution(command, self.manifest)
                for index in range(count):
                    tail_server.stderr_lines.append(
                        "S41SERVERFFNCALL context=" + command.request_id.encode("ascii").hex()
                        + f":2:1:{generation} request={31 + index} layer=0 tokens=1 "
                        + f"columns={resident.columns} "
                        + f"payload_bytes={self.manifest.embedding_length * 2}"
                    )
                if count == 3 and generation == 61:
                    proof = tail_server.finish_execution(
                        tail_marker, command, self.manifest, output_tokens=6,
                        adaptive_observation=released_tail,
                    )
                    self.assertEqual(proof.phone_call_count, 3)
                else:
                    with patch("research_dev.scheduler.adapters.llama_server."
                               "_FFN_PROOF_DRAIN_TIMEOUT_S", 0):
                        with self.assertRaisesRegex(
                            PhysicalAdapterError, "phone calls differ from adaptive windows",
                        ):
                            tail_server.finish_execution(
                                tail_marker, command, self.manifest, output_tokens=6,
                                adaptive_observation=released_tail,
                            )

        for measured_rows in (1, 2):
            counted_phone = replace(
                measured_phone, completed_phone_calls=measured_rows,
                completed_phone_input_rows=measured_rows,
            )
            counted_tail = replace(released_tail, windows=(first, counted_phone))
            for count, generation in ((4, 61), (3, 61), (5, 61), (4, 62)):
                with self.subTest(measured_rows=measured_rows, calls=count, generation=generation):
                    tail_server = ManagedLlamaServer(
                        ("synthetic-server",), {}, self.root,
                        f"synthetic-counted-tail-{measured_rows}-{count}-{generation}",
                        llama_server_launch_contract(command, self.manifest),
                    )
                    tail_server.process = SimpleNamespace(poll=lambda: None)
                    tail_marker = tail_server.begin_execution(command, self.manifest)
                    for index in range(count):
                        tail_server.stderr_lines.append(
                            "S41SERVERFFNCALL context=" + command.request_id.encode("ascii").hex()
                            + f":2:1:{generation} request={31 + index} layer=0 tokens=1 "
                            + f"columns={resident.columns} "
                            + f"payload_bytes={self.manifest.embedding_length * 2}"
                        )
                    with patch("research_dev.scheduler.adapters.llama_server."
                               "_FFN_PROOF_DRAIN_TIMEOUT_S", 0):
                        if count == 4 and generation == 61:
                            proof = tail_server.finish_execution(
                                tail_marker, command, self.manifest, output_tokens=6,
                                adaptive_observation=counted_tail,
                            )
                            self.assertEqual(proof.phone_call_count, 4)
                        else:
                            with self.assertRaisesRegex(
                                PhysicalAdapterError, "phone calls differ from adaptive windows",
                            ):
                                tail_server.finish_execution(
                                    tail_marker, command, self.manifest, output_tokens=6,
                                    adaptive_observation=counted_tail,
                                )

    def test_adaptive_progress_seals_before_released_terminal_slot(self) -> None:
        class Scheduler:
            def __init__(self):
                self.starts = []
                self.boundaries = []
                self.seals = []

            def start_adaptive_decode(self, request_id, **values):
                self.starts.append((request_id, values))

            def adaptive_decode_boundary(self, request_id, **values):
                self.boundaries.append((request_id, values))
                return None

            def seal_adaptive_decode_tail(self, request_id, **values):
                self.seals.append((request_id, values))

            def record_adaptive_decode_window(self, *args, **kwargs):
                raise AssertionError("window measurement was not requested")

            def acknowledge_adaptive_decode_control(self, *args, **kwargs):
                raise AssertionError("control acknowledgement was not requested")

            def fail_adaptive_decode_control(self, *args, **kwargs):
                raise AssertionError("control failure was not requested")

            def preview_adaptive_decode_completion(self, *args, **kwargs):
                raise AssertionError("completion was not requested")

            def complete_adaptive_decode(self, *args, **kwargs):
                raise AssertionError("completion was not requested")

        command = adaptive_execution_command(self.command)
        scheduler = Scheduler()
        backend = CanonicalHttpExecutionBackend(
            FakeClient(), FakeMeter(), epoch_ns=time.monotonic_ns() - 1_000_000
        )
        backend.bind_scheduler(scheduler)
        payload = LlamaCppCompletionPayload(
            request_id=command.request_id,
            expected_model_alias="synthetic-model",
            input_tokens=1,
            output_tokens=8,
            prompt_tokens=(1,),
            seed=1,
            stream_path=self.root / "adaptive-tail.raw",
            on_first_token=lambda _value: None,
        )
        adaptive = backend._adaptive_payload(
            command, payload, time.monotonic_ns()
        )

        adaptive.on_decode_progress(2, 1, time.monotonic_ns(), False)
        for token_index in range(2, 9):
            adaptive.on_decode_progress(
                2,
                token_index,
                time.monotonic_ns(),
                token_index == 8,
            )

        self.assertEqual(len(scheduler.seals), 1)
        self.assertEqual(scheduler.seals[0][1]["token_index"], 6)
        self.assertTrue(scheduler.boundaries[-1][1]["terminal"])
        self.assertEqual(scheduler.boundaries[-1][1]["token_index"], 6)

    def test_adaptive_execution_persists_request_slot_d2h_fence(
        self,
    ) -> None:
        command = adaptive_execution_command(self.command)
        baseline = AdaptiveDecodePolicy(
            route_id="synthetic:desktop-parent",
            executor_id="synthetic:desktop",
            operator_plan_sha256=PLAN_SHA256,
            desktop_parent_route_id="synthetic:desktop-parent",
            desktop_placement_sha256="sha256:" + "4" * 64,
            layer_indices=(),
            layer_mask=0,
            columns=0,
            split_fraction_ppm=0,
            resource_ids=("compute:accelerator-b",),
            baseline=True,
        )

        class Client(FakeClient):
            def complete(
                self,
                endpoint,
                payload,
                control_check,
                *,
                scheduler_headers=None,
            ):
                control_check()
                first_ns = time.monotonic_ns()
                payload.on_active_batch(2)
                payload.on_decode_progress(2, 1, first_ns, False)
                self._slots_probe = lambda *_args: [
                    {"id": 2, "id_task": 1, "is_processing": True},
                ]
                payload.on_decode_progress(
                    2, 2, first_ns + 1_000_000, False
                )
                return {"stream_sha256": "3" * 64, "tokens": [1]}

            def read_ffn_stats(self, *_args, **_kwargs):
                return ({
                    "applied_token_index": 2,
                    "plan_generation": 1,
                    "policy_hash": baseline.policy_hash,
                    "runtime_stats": {
                        "batched_calls": 0,
                        "calls": 1,
                        "configured_queue_depth": 4,
                        "desktop_compute_us": 20,
                        "download_bytes": 64,
                        "exposed_tail_us": 5,
                        "input_rows": 1,
                        "maximum_active_slots": 1,
                        "maximum_outstanding_transfers": 1,
                        "maximum_tokens": 1,
                        "phone_compute_us": 10,
                        "rpc_us": 4,
                        "transfer_subrequests": 1,
                        "upload_bytes": 64,
                        "usb_d2h_us": 3,
                        "usb_h2d_us": 2,
                        "usb_transfer_us": 5,
                        "useful_overlap_us": 1,
                    },
                    "slot_id": 2,
                    "success": True,
                }, time.monotonic_ns())

        class Completed:
            grouped_observation_sha256 = "sha256:" + "5" * 64

            def to_json(self):
                return {"grouped_observation_sha256": (
                    self.grouped_observation_sha256
                )}

        class Scheduler:
            def __init__(self):
                self.started_at_us = None
                self.observation = None
                self.completed = Completed()

            def start_adaptive_decode(self, _request_id, **values):
                self.started_at_us = values["at_us"]
                return AdaptiveDecodeDirective(
                    state="BASELINE",
                    reason="WINDOW_OPENED",
                    target_token_index=2,
                )

            def adaptive_decode_boundary(self, request_id, **values):
                boundary = AdaptiveDecodeWindowBoundary(
                    request_id=request_id,
                    slot_id=values["slot_id"],
                    window_index=0,
                    token_start=1,
                    token_end=values["token_index"],
                    started_at_us=self.started_at_us,
                    finished_at_us=max(
                        self.started_at_us + 1, values["at_us"]
                    ),
                    policy=baseline,
                    applied_ack=None,
                )
                return AdaptiveDecodeDirective(
                    state="BASELINE",
                    reason="WINDOW_MEASUREMENT_REQUIRED",
                    target_token_index=None,
                    boundary=boundary,
                )

            def seal_adaptive_decode_tail(self, *_args, **_kwargs):
                return None

            def record_adaptive_decode_window(
                self, _request_id, _boundary, observation
            ):
                self.observation = observation
                return None

            def acknowledge_adaptive_decode_control(self, *_args, **_kwargs):
                raise AssertionError("control acknowledgement was requested")

            def fail_adaptive_decode_control(self, *_args, **_kwargs):
                raise AssertionError("control failure was requested")

            def preview_adaptive_decode_completion(self, _request_id):
                return self.completed

            def complete_adaptive_decode(self, _request_id):
                return self.completed

        scheduler = Scheduler()
        backend = CanonicalHttpExecutionBackend(
            Client(), FakeMeter(), epoch_ns=time.monotonic_ns() - 1_000_000
        )
        backend.bind_scheduler(scheduler)
        payload = LlamaCppCompletionPayload(
            request_id=command.request_id,
            expected_model_alias="synthetic-model",
            input_tokens=1,
            output_tokens=4,
            prompt_tokens=(1,),
            seed=1,
            stream_path=self.root / "adaptive-fence.raw",
            on_first_token=lambda _value: None,
        )

        observation = backend.execute(command, payload, lambda: None)

        fences = observation.payload["request_slot_final_d2h_fences"]
        self.assertEqual(len(fences), 1)
        self.assertEqual(fences[0]["request_id"], command.request_id)
        self.assertEqual(fences[0]["slot_id"], 2)
        self.assertEqual(fences[0]["applied_token_index"], 2)
        self.assertTrue(fences[0]["final_d2h_completed"])
        self.assertEqual(fences[0]["download_bytes"], 64)
        self.assertEqual(
            observation.payload["physical_execution_timing"][
                "request_slot_final_d2h_fence_count"
            ],
            1,
        )
        self.assertEqual(scheduler.observation.active_batch, 2)
        self.assertEqual(scheduler.observation.next_active_batch, 1)
        self.assertTrue(scheduler.observation.membership_changed)
        self.assertEqual(scheduler.observation.completed_phone_calls, 1)
        self.assertEqual(
            scheduler.observation.completed_phone_input_rows, 1
        )

    def test_terminal_phone_counters_precede_live_membership_refresh(self) -> None:
        command = adaptive_execution_command(self.command)
        policy = AdaptiveDecodePolicy(
            route_id=ARTIFACT_ROUTE, executor_id=command.executor_id,
            operator_plan_sha256=PLAN_SHA256,
            desktop_parent_route_id="synthetic:desktop-parent",
            desktop_placement_sha256="sha256:" + "4" * 64,
            layer_indices=(0,), layer_mask=1, columns=16,
            split_fraction_ppm=500_000,
            resource_ids=("compute:accelerator-b", "compute:helper-c"),
        )
        ack = AdaptiveDecodePolicyAck(
            request_id=command.request_id, slot_id=2, plan_generation=7,
            applied_token_index=3, applied_at_us=100, policy_hash=policy.policy_hash,
        )
        boundary = AdaptiveDecodeWindowBoundary(
            request_id=command.request_id, slot_id=2, window_index=1,
            token_start=3, token_end=6, started_at_us=100, finished_at_us=20_000,
            policy=policy, applied_ack=ack,
        )
        client = FakeClient()
        scheduler = Mock(runtime_ticket=None)
        scheduler.discard_stale_adaptive_decode_window.side_effect = AssertionError(
            "membership refresh lost the terminal phone counters"
        )
        backend = CanonicalHttpExecutionBackend(
            client, FakeMeter(), epoch_ns=time.monotonic_ns() - 30_000_000,
        )
        backend.bind_scheduler(scheduler)
        payload = LlamaCppCompletionPayload(
            request_id=command.request_id, expected_model_alias="synthetic-model",
            input_tokens=1, output_tokens=8, prompt_tokens=(1,), seed=1,
            stream_path=self.root / "terminal-membership.raw",
            on_first_token=lambda _value: None,
        )
        fences = []
        controller = _AdaptivePayloadController(
            backend, command, payload, None, fences.append,
        )
        controller.live_members = ((2, 1),)
        controller.window_active_batch = controller.live_active_batch = 1
        operations = []

        def slots(*_args):
            operations.append("released-membership")
            return []

        def stats(*_args, **_kwargs):
            if operations:
                raise StalePhysicalSlotError("FFN stats request and active slot differ")
            operations.append("phone-counters")
            counters = {
                **controller.last_stats,
                **{key: 1 for key in controller._GAUGE_NAMES},
                "calls": 3, "input_rows": 3, "transfer_subrequests": 3,
                "upload_bytes": 192, "download_bytes": 192,
            }
            return ({
                "slot_id": 2, "success": True, "plan_generation": 7,
                "applied_token_index": 6, "policy_hash": policy.policy_hash,
                "runtime_stats": counters,
            }, time.monotonic_ns())

        client._slots_probe = slots
        client.read_ffn_stats = stats
        controller._process_boundary(boundary)

        self.assertEqual(operations, ["phone-counters", "released-membership"])
        observed = scheduler.record_adaptive_decode_window.call_args.args[2]
        self.assertEqual(observed.completed_phone_calls, 3)
        self.assertEqual(observed.completed_phone_input_rows, 3)
        self.assertFalse(observed.membership_changed)
        self.assertFalse(observed.execution_context_available)
        self.assertIsNone(observed.failure_reason)
        self.assertTrue(observed.output_valid)
        self.assertEqual(fences[0]["plan_generation"], ack.plan_generation)
        self.assertTrue(fences[0]["final_d2h_completed"])
        scheduler.discard_stale_adaptive_decode_window.assert_not_called()

    def test_live_membership_failure_and_recovery_do_not_invent_batch_change(self):
        controller = object.__new__(_AdaptivePayloadController)
        slots = Mock(side_effect=[
            TimeoutError("slots timed out"),
            [{"id": 2, "id_task": 41, "is_processing": True}],
            [{"id": 2, "id_task": 42, "is_processing": True}],
        ])
        controller.backend = SimpleNamespace(
            _client=SimpleNamespace(_slots_probe=slots), _record_adaptive_timing=Mock(),
            _relative_us=lambda _ns: 1000,
        )
        controller.command = self.command
        controller.cohort = None
        controller.live_members = ((2, 41),)
        controller.live_active_batch = controller.window_active_batch = 1
        controller.live_context_available = True
        self.assertFalse(controller._refresh_active_members())
        self.assertFalse(controller.live_context_available)
        self.assertEqual(controller.live_members, ((2, 41),))
        self.assertFalse(controller._refresh_active_members())
        self.assertTrue(controller.live_context_available)
        self.assertTrue(controller._refresh_active_members())
        events = [row.args[0] for row in controller.backend._record_adaptive_timing.call_args_list]
        self.assertEqual([row["kind"] for row in events], [
            "DECODE_CONTEXT_UNAVAILABLE", "DECODE_CONTEXT_RECOVERED", "DECODE_CONTEXT_CHANGED",
        ])
        self.assertEqual(events[-1]["previous_members"], ((2, 41),))
        self.assertEqual(events[-1]["members"], [[2, 42]])
        self.assertEqual(events[-1]["active_batch"], events[-1]["previous_active_batch"])

    def test_released_phone_tail_waits_for_exact_terminal_without_fake_counters(self):
        command = adaptive_execution_command(self.command)
        policy = AdaptiveDecodePolicy(
            route_id=ARTIFACT_ROUTE, executor_id=command.executor_id,
            operator_plan_sha256=PLAN_SHA256, desktop_parent_route_id="synthetic:parent",
            desktop_placement_sha256="sha256:" + "4" * 64,
            layer_indices=(0,), layer_mask=1, columns=16, split_fraction_ppm=500_000,
            resource_ids=("compute:accelerator-b", "compute:helper-c"),
        )
        boundary = AdaptiveDecodeWindowBoundary(
            request_id=command.request_id, slot_id=2, window_index=2,
            token_start=5, token_end=6, started_at_us=500, finished_at_us=600,
            policy=policy, applied_ack=None,
        )
        scheduler = Mock(runtime_ticket=None)
        client = FakeClient()
        client.read_ffn_stats = Mock(side_effect=StalePhysicalSlotError("released"))
        backend = CanonicalHttpExecutionBackend(
            client, FakeMeter(), epoch_ns=time.monotonic_ns() - 1_000_000,
        )
        backend.bind_scheduler(scheduler)
        payload = LlamaCppCompletionPayload(
            request_id=command.request_id, expected_model_alias="synthetic-model",
            input_tokens=1, output_tokens=8, prompt_tokens=(1,), seed=1,
            stream_path=self.root / "released-positive-tail.raw", on_first_token=lambda _: None,
        )
        controller = _AdaptivePayloadController(backend, command, payload, None, None)
        controller.started = True
        controller.last_stats["calls"] = 2
        controller.last_window_end_token = 5
        self.assertIsNone(controller._process_boundary(boundary))
        scheduler.discard_stale_adaptive_decode_window.assert_not_called()
        with self.assertRaisesRegex(PhysicalAdapterError, "terminal identity differs"):
            controller._released_slot_progress(3, 8, 800, True)
        with self.assertRaisesRegex(PhysicalAdapterError, "terminal token count differs"):
            controller._released_slot_progress(2, 7, 800, True)
        controller._released_slot_progress(2, 8, 800, True)
        self.assertTrue(controller.closed)
        self.assertIsNone(controller.pending_stale_boundary)
        call = scheduler.discard_stale_adaptive_decode_window.call_args
        self.assertEqual(call.args[3], "released_slot_phone_tail")
        self.assertEqual(call.kwargs["terminal_token_index"], 8)
        self.assertEqual(call.args[2].completed_phone_calls, 0)
        scheduler.record_adaptive_decode_window.assert_not_called()
        scheduler.acknowledge_adaptive_decode_control.assert_not_called()

    def test_stale_short_tail_stats_are_discarded_without_control_retry(
        self,
    ) -> None:
        command = adaptive_execution_command(self.command)
        baseline = AdaptiveDecodePolicy(
            route_id="synthetic:desktop-parent",
            executor_id="synthetic:desktop",
            operator_plan_sha256=PLAN_SHA256,
            desktop_parent_route_id="synthetic:desktop-parent",
            desktop_placement_sha256="sha256:" + "4" * 64,
            layer_indices=(),
            layer_mask=0,
            columns=0,
            split_fraction_ppm=0,
            resource_ids=("compute:accelerator-b",),
            baseline=True,
        )

        class Client(FakeClient):
            def read_ffn_stats(self, *_args, **_kwargs):
                raise StalePhysicalSlotError(
                    "FFN stats failed: "
                    "FFN stats request and active slot differ"
                )

            def apply_ffn_control(self, *_args, **_kwargs):
                raise AssertionError("short tail retried an FFN control")

        class Scheduler:
            def __init__(self):
                self.boundary = None
                self.discards = []
                self.seals = []
                self.baseline_ack = None
                self.terminal_progress = []

            def start_adaptive_decode(self, request_id, **values):
                return AdaptiveDecodeDirective(
                    state="EXPLOITING",
                    reason="WINDOW_OPENED",
                    target_token_index=3,
                )

            def adaptive_decode_boundary(self, request_id, **values):
                self.boundary = AdaptiveDecodeWindowBoundary(
                    request_id=request_id,
                    slot_id=values["slot_id"],
                    window_index=0,
                    token_start=1,
                    token_end=values["token_index"],
                    started_at_us=1,
                    finished_at_us=max(2, values["at_us"]),
                    policy=baseline,
                    applied_ack=self.baseline_ack,
                )
                return AdaptiveDecodeDirective(
                    state="EXPLOITING",
                    reason="WINDOW_MEASUREMENT_REQUIRED",
                    target_token_index=None,
                    boundary=self.boundary,
                )

            def seal_adaptive_decode_tail(self, request_id, **values):
                self.seals.append((request_id, values))

            def discard_stale_adaptive_decode_window(
                self, request_id, boundary, observation, reason, **terminal
            ):
                self.discards.append(
                    (request_id, boundary, observation, reason)
                )
                self.terminal_progress.append(terminal)
                return AdaptiveDecodeDirective(
                    state="EXPLOITING",
                    reason="STALE_SLOT_DIRECTIVE_DISCARDED",
                    target_token_index=None,
                )

            def record_adaptive_decode_window(self, *_args, **_kwargs):
                raise AssertionError("stale stats were recorded normally")

            def acknowledge_adaptive_decode_control(self, *_args, **_kwargs):
                raise AssertionError("control acknowledgement was requested")

            def fail_adaptive_decode_control(self, *_args, **_kwargs):
                raise AssertionError("stale control was retried")

            def preview_adaptive_decode_completion(self, *_args, **_kwargs):
                raise AssertionError("completion was not requested")

            def complete_adaptive_decode(self, *_args, **_kwargs):
                raise AssertionError("completion was not requested")

        scheduler = Scheduler()
        backend = CanonicalHttpExecutionBackend(
            Client(), FakeMeter(), epoch_ns=time.monotonic_ns() - 1_000_000
        )
        backend.bind_scheduler(scheduler)
        payload = LlamaCppCompletionPayload(
            request_id=command.request_id,
            expected_model_alias="synthetic-model",
            input_tokens=1,
            output_tokens=3,
            prompt_tokens=(1,),
            seed=1,
            stream_path=self.root / "stale-short-tail.raw",
            on_first_token=lambda _value: None,
        )
        adaptive = backend._adaptive_payload(
            command, payload, time.monotonic_ns()
        )

        adaptive.on_decode_progress(2, 1, time.monotonic_ns(), False)
        adaptive.on_decode_progress(2, 2, time.monotonic_ns(), False)

        self.assertEqual(len(scheduler.seals), 1)
        self.assertEqual(len(scheduler.discards), 1)
        self.assertEqual(
            scheduler.discards[0][3], "stale_slot_stats_discarded"
        )
        self.assertTrue(scheduler.discards[0][2].output_valid)

        for stale_token in (3, 31):
            with self.subTest(stale_token=stale_token):
                scheduler = Scheduler()
                scheduler.baseline_ack = AdaptiveDecodePolicyAck(
                    request_id=command.request_id, slot_id=2,
                    plan_generation=2, applied_token_index=1,
                    applied_at_us=1, policy_hash=baseline.policy_hash,
                )
                backend = CanonicalHttpExecutionBackend(
                    Client(), FakeMeter(), epoch_ns=time.monotonic_ns() - 1_000_000,
                )
                backend.bind_scheduler(scheduler)
                adaptive = backend._adaptive_payload(
                    command, replace(payload, output_tokens=31), time.monotonic_ns(),
                )
                adaptive.on_decode_progress(2, 1, time.monotonic_ns(), False)
                adaptive.on_decode_progress(
                    2, stale_token, time.monotonic_ns(), stale_token == 31,
                )
                if stale_token < 31:
                    self.assertEqual(scheduler.discards, [])
                    adaptive.on_decode_progress(2, 4, time.monotonic_ns(), False)
                    with self.assertRaisesRegex(PhysicalAdapterError, "terminal identity differs"):
                        adaptive.on_decode_progress(3, 31, time.monotonic_ns(), True)
                    with self.assertRaisesRegex(PhysicalAdapterError, "terminal token count differs"):
                        adaptive.on_decode_progress(2, 30, time.monotonic_ns(), True)
                    adaptive.on_decode_progress(2, 31, time.monotonic_ns(), True)
                adaptive.on_decode_progress(2, 31, time.monotonic_ns(), True)
                self.assertEqual(len(scheduler.discards), 1)
                self.assertEqual(scheduler.discards[0][3], "released_slot_baseline_tail")
                self.assertEqual(scheduler.terminal_progress[0]["terminal_token_index"], 31)
                self.assertIn("physical:terminal-release-confirmed",
                              scheduler.discards[0][2].evidence_ids)

    def test_phone_contract_uses_request_scoped_maximum_tokens(self) -> None:
        parameters = dict(self.command.adapter_parameters)
        parameters["ffn_max_tokens"] = 2
        parameters["ubatch_size"] = 2
        command = PhysicalExecutionCommand(
            **{
                **self.command.__dict__,
                "adapter_parameters": parameters,
            }
        )
        contract = llama_server_launch_contract(command, self.manifest)
        self.assertEqual(contract.ubatch_size, 2)
        self.assertEqual(
            phone_ffn_execution_contract(command, self.manifest).n_embd,
            self.manifest.embedding_length,
        )
        self.assertTrue(
            contract.ffn_environment["S41_SERVER_FFN_POLICY"].startswith(
                "2:"
            )
        )

    def test_phone_contract_rejects_a_policy_smaller_than_ubatch(self) -> None:
        parameters = dict(self.command.adapter_parameters)
        parameters["ffn_max_tokens"] = 2
        command = PhysicalExecutionCommand(
            **{
                **self.command.__dict__,
                "adapter_parameters": parameters,
            }
        )

        with self.assertRaisesRegex(
            PhysicalAdapterError, "does not cover the maximum batch"
        ):
            llama_server_launch_contract(command, self.manifest)

    def test_decode_only_phone_contract_keeps_desktop_ubatch(self) -> None:
        parameters = dict(self.command.adapter_parameters)
        parameters.update({
            "ffn_assistance_phase": "decode",
            "ffn_max_tokens": 1,
            "ffn_runtime_control_protocol": "decode-boundary-v1",
        })
        command = adaptive_execution_command(
            replace(self.command, adapter_parameters=parameters)
        )

        contract = llama_server_launch_contract(command, self.manifest)

        self.assertEqual(contract.ubatch_size, 4)
        self.assertEqual(
            phone_ffn_execution_contract(command, self.manifest).max_tokens,
            1,
        )
        self.assertEqual(
            contract.ffn_environment["S41_SERVER_FFN_MAX_TOKENS"], "1"
        )
        self.assertEqual(
            contract.ffn_environment["S41_SERVER_FFN_RUNTIME_CONTROL"], "1"
        )
        self.assertNotIn(
            "S41_SERVER_FFN_POLICY", contract.ffn_environment
        )

    def test_dormant_phone_runtime_keeps_launch_contract_stable(self) -> None:
        adaptive = adaptive_execution_command(self.command)
        phone = dict(adaptive.adapter_parameters)
        phone.update({
            "ffn_max_tokens": 1,
            "ffn_resident_columns": 128,
            "ffn_resident_layer_mask": 1,
        })
        dormant_names = {
            "ffn_activation",
            "ffn_assistance_phase",
            "ffn_max_tokens",
            "ffn_n_embd",
            "ffn_resident_columns",
            "ffn_resident_layer_mask",
            "ffn_runtime_control_protocol",
            "ffn_timeout_ms",
            "ffn_transport",
            "phone_device_id",
            "usb_allocator",
            "usb_batch_plan",
            "usb_full_duplex",
            "usb_max_payload_bytes",
            "usb_product_id",
            "usb_queue_depth",
            "usb_slot_safety_bytes",
            "usb_split_h2d",
            "usb_transport_generation",
            "usb_transport_profile_id",
            "usb_vendor_id",
            "usbfs_available_bytes",
        }
        dormant = json.dumps(
            {
                name: value for name, value in phone.items()
                if name in dormant_names
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        parameters = {
            **phone,
            "dormant_phone_ffn_runtime_v1": dormant,
        }
        adaptive = replace(adaptive, adapter_parameters=parameters)
        desktop_contract = RuntimeExecutionContract(
            execution_mode="desktop",
            initial_split_fraction_ppm=0,
            allowed_adaptive_fractions_ppm=(),
            batch_plan="none",
            maximum_batch_size=1,
            queue_depth=1,
        )
        desktop = replace(
            adaptive,
            adapter_parameters={
                name: value for name, value in parameters.items()
                if name != "phone_device_id"
            },
            execution_contract=desktop_contract,
            operator_plan={
                **dict(adaptive.operator_plan),
                "assisted_operator_kind": None,
                "execution_contract": desktop_contract.to_json(),
            },
        )

        desktop_launch = llama_server_launch_contract(
            desktop, self.manifest
        )
        adaptive_launch = llama_server_launch_contract(
            adaptive, self.manifest
        )

        self.assertEqual(desktop_launch, adaptive_launch)
        self.assertEqual(
            desktop_launch.ffn_environment[
                "S41_SERVER_FFN_RUNTIME_CONTROL"
            ],
            "1",
        )
        self.assertNotIn(
            "S41_SERVER_FFN_SHARDS", desktop_launch.ffn_environment
        )
        self.assertTrue(
            CanonicalHttpExecutionBackend._adaptive_enabled(desktop)
        )
        self.assertTrue(
            CanonicalHttpExecutionBackend._adaptive_enabled(adaptive)
        )
        server = ManagedLlamaServer(
            ("synthetic-server",),
            {},
            self.root,
            "synthetic-dormant-phone",
            desktop_launch,
        )
        server.process = SimpleNamespace(poll=lambda: None)
        desktop_marker = server.begin_execution(desktop, self.manifest)
        adaptive_marker = server.begin_execution(adaptive, self.manifest)
        self.assertIsNone(desktop_marker.phone_contract)
        self.assertIsNotNone(adaptive_marker.phone_contract)

        base_parameters = {
            name: value for name, value in desktop.adapter_parameters.items()
            if name != "dormant_phone_ffn_runtime_v1"
        }
        self.assertTrue(physical_residency_parameters_match(
            desktop.adapter_parameters, base_parameters
        ))
        self.assertFalse(physical_residency_parameters_match(
            base_parameters, desktop.adapter_parameters
        ))
        self.assertFalse(physical_residency_parameters_match(
            {
                **base_parameters,
                "dormant_phone_ffn_runtime_v1": dormant + " ",
            },
            desktop.adapter_parameters,
        ))
        dormant_subset = json.dumps(
            {
                **json.loads(dormant),
                "ffn_resident_layer_mask": 1,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        dormant_superset = json.dumps(
            {
                **json.loads(dormant),
                "ffn_resident_layer_mask": 3,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        dormant_disjoint = json.dumps(
            {
                **json.loads(dormant),
                "ffn_resident_layer_mask": 1 << 8,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.assertTrue(physical_residency_parameters_match(
            {
                **base_parameters,
                "dormant_phone_ffn_runtime_v1": dormant_superset,
            },
            {
                **base_parameters,
                "dormant_phone_ffn_runtime_v1": dormant_subset,
            },
        ))
        self.assertFalse(physical_residency_parameters_match(
            {
                **base_parameters,
                "dormant_phone_ffn_runtime_v1": dormant_subset,
            },
            {
                **base_parameters,
                "dormant_phone_ffn_runtime_v1": dormant_superset,
            },
        ))
        self.assertFalse(physical_residency_parameters_match(
            {
                **base_parameters,
                "dormant_phone_ffn_runtime_v1": dormant_superset,
            },
            {
                **base_parameters,
                "dormant_phone_ffn_runtime_v1": dormant_disjoint,
            },
        ))
        superset_command = replace(
            desktop,
            adapter_parameters={
                **base_parameters,
                "dormant_phone_ffn_runtime_v1": dormant_superset,
            },
        )
        subset_command = replace(
            desktop,
            adapter_parameters={
                **base_parameters,
                "dormant_phone_ffn_runtime_v1": dormant_subset,
            },
        )
        disjoint_command = replace(
            desktop,
            adapter_parameters={
                **base_parameters,
                "dormant_phone_ffn_runtime_v1": dormant_disjoint,
            },
        )
        superset_server = ManagedLlamaServer(
            ("synthetic-server",),
            {},
            self.root,
            "synthetic-dormant-phone-superset",
            llama_server_launch_contract(
                superset_command, self.manifest
            ),
        )
        superset_server.process = SimpleNamespace(poll=lambda: None)
        self.assertIsNone(
            superset_server.begin_execution(
                subset_command, self.manifest
            ).phone_contract
        )
        with self.assertRaisesRegex(
            PhysicalAdapterError,
            "execution differs from the launched contract",
        ):
            superset_server.begin_execution(
                disjoint_command, self.manifest
            )
        ordinary = replace(
            desktop,
            adapter_parameters=base_parameters,
            operator_plan={
                **dict(desktop.operator_plan),
                "operators": [
                    {
                        **row,
                        "device_ids": ["accelerator-b"],
                    }
                    if row.get("operator_kind") == "ffn"
                    else dict(row)
                    for row in desktop.operator_plan["operators"]
                ],
            },
        )
        ordinary_marker = server.begin_execution(
            ordinary, self.manifest
        )
        self.assertIsNone(ordinary_marker.phone_contract)

    def test_static_decode_phone_contract_applies_selected_policy(self) -> None:
        parameters = dict(self.command.adapter_parameters)
        parameters.update({
            "ffn_assistance_phase": "decode",
            "ffn_max_tokens": 1,
            "ffn_runtime_control_protocol": "decode-boundary-v1",
            "ffn_resident_columns": 128,
            "ffn_selected_columns": 64,
            "ffn_selected_layer_mask": 1,
        })
        command = replace(
            self.command,
            adapter_parameters=parameters,
            selection_mode="calibration",
        )

        contract = llama_server_launch_contract(command, self.manifest)

        phone = phone_ffn_execution_contract(command, self.manifest)
        self.assertEqual(phone.columns, 64)
        self.assertEqual(
            contract.ffn_environment["S41_SERVER_FFN_RUNTIME_CONTROL"],
            "1",
        )
        self.assertNotIn("S41_SERVER_FFN_POLICY", contract.ffn_environment)

        transition = PhysicalTransitionCommand(
            ticket_id=command.ticket_id,
            request_id=command.request_id,
            artifact_sha256=command.artifact_sha256,
            route_id=command.route_id,
            operator_plan_sha256=command.operator_plan_sha256,
            participant=PhysicalParticipantCommand(
                executor_id=command.executor_id,
                device_id="helper-c",
                endpoint=command.endpoint,
                backend="synthetic",
                resource_ids=("transition:helper-c",),
            ),
            transition=RuntimeTransitionPlan(
                transition_id="synthetic-load",
                device_id="helper-c",
                source_state="cold",
                target_state="hot",
                latency_us=1,
                energy_uj=1,
                resource_ids=("transition:helper-c",),
                maturity="QUALIFIED",
            ),
            execution_contract=command.execution_contract,
            adapter_parameters=parameters,
            operator_plan_protocol=command.operator_plan_protocol,
            operator_plan=command.operator_plan,
            selection_mode=command.selection_mode,
        )
        self.assertEqual(
            llama_server_launch_contract(transition, self.manifest),
            contract,
        )

        adaptive = adaptive_execution_command(command)
        adaptive_transition = replace(
            transition,
            execution_contract=adaptive.execution_contract,
            operator_plan=adaptive.operator_plan,
            selection_mode="adaptive-decode",
        )
        adaptive_contract = llama_server_launch_contract(
            adaptive, self.manifest
        )
        self.assertEqual(
            llama_server_launch_contract(
                adaptive_transition, self.manifest
            ),
            adaptive_contract,
        )
        self.assertIn(
            "S41_SERVER_FFN_RUNTIME_CONTROL",
            adaptive_contract.ffn_environment,
        )
        self.assertNotIn(
            "S41_SERVER_FFN_POLICY", adaptive_contract.ffn_environment
        )

    def test_static_decode_applies_selected_policy_at_first_boundary(
        self,
    ) -> None:
        class StaticControlClient(FakeClient):
            def __init__(self):
                self.controls = []

            def apply_ffn_control(self, _endpoint, control, *, timeout_s=5):
                self.controls.append(control)
                return ({
                    "applied_token_index": 1,
                    "plan_generation": control.plan_generation,
                    "policy_hash": control.policy.policy_hash,
                    "runtime_stats": {},
                    "slot_id": control.slot_id,
                    "success": True,
                }, time.monotonic_ns())

        parameters = dict(self.command.adapter_parameters)
        parameters.update({
            "ffn_assistance_phase": "decode",
            "ffn_max_tokens": 1,
            "ffn_resident_columns": 128,
            "ffn_runtime_control_protocol": "decode-boundary-v1",
            "ffn_selected_columns": 64,
            "ffn_selected_layer_mask": 1,
        })
        operator_plan = {
            **self.command.operator_plan,
            "assisted_operator_kind": "ffn",
            "baseline_executor_id": "synthetic:desktop-parent",
            "desktop_placement_sha256": "sha256:" + "4" * 64,
            "resource_ids": ["compute:host-a", "compute:helper-c"],
            "split_fraction_ppm": 500_000,
        }
        command = replace(
            self.command,
            adapter_parameters=parameters,
            operator_plan=operator_plan,
            selection_mode="calibration",
        )
        client = StaticControlClient()
        backend = CanonicalHttpExecutionBackend(
            client, FakeMeter(), epoch_ns=time.monotonic_ns()
        )
        acknowledgements = []
        payload = LlamaCppCompletionPayload(
            request_id=command.request_id,
            expected_model_alias="synthetic-model",
            input_tokens=1,
            output_tokens=8,
            prompt_tokens=(1,),
            seed=1,
            stream_path=self.root / "static-control.raw",
            on_first_token=lambda _value: None,
        )

        controlled = backend._static_decode_payload(
            command, payload, acknowledgements.append
        )
        controlled.on_decode_progress(2, 1, time.monotonic_ns(), False)
        controlled.on_decode_progress(2, 2, time.monotonic_ns(), False)

        self.assertEqual(len(client.controls), 1)
        self.assertEqual(client.controls[0].policy.columns, 64)
        self.assertEqual(client.controls[0].policy.layer_mask, 1)
        self.assertEqual(acknowledgements[0]["request_id"], command.request_id)
        self.assertEqual(acknowledgements[0]["applied_token_index"], 1)

    def test_static_decode_proof_starts_coverage_at_acknowledged_token(
        self,
    ) -> None:
        parameters = dict(self.command.adapter_parameters)
        parameters.update({
            "ffn_assistance_phase": "decode",
            "ffn_max_tokens": 1,
            "ffn_resident_columns": 128,
            "ffn_runtime_control_protocol": "decode-boundary-v1",
            "ffn_selected_columns": 64,
            "ffn_selected_layer_mask": 1,
        })
        operator_plan = {
            **self.command.operator_plan,
            "assisted_operator_kind": "ffn",
            "baseline_executor_id": "synthetic:desktop-parent",
            "desktop_placement_sha256": "sha256:" + "4" * 64,
            "resource_ids": ["compute:host-a", "compute:helper-c"],
            "split_fraction_ppm": 500_000,
        }
        command = replace(
            self.command,
            adapter_parameters=parameters,
            operator_plan=operator_plan,
            selection_mode="calibration",
        )
        server = ManagedLlamaServer(
            ("synthetic-server",),
            {},
            self.root,
            "synthetic-static-decode",
            llama_server_launch_contract(command, self.manifest),
        )
        server.process = SimpleNamespace(poll=lambda: None)
        marker = server.begin_execution(command, self.manifest)
        server.stderr_lines.extend(
            "S41SERVERFFNCALL request=" + str(index)
            + " layer=0 tokens=1 columns=64 payload_bytes=64"
            for index in range(11, 16)
        )
        policy = CanonicalHttpExecutionBackend._static_decode_policy(command)
        acknowledgement = {
            "applied_token_index": 3,
            "plan_generation": 1,
            "policy_hash": policy.policy_hash,
            "request_id": command.request_id,
            "slot_id": 2,
        }

        proof = server.finish_execution(
            marker,
            command,
            self.manifest,
            output_tokens=8,
            static_control_ack=acknowledgement,
        ).to_json()

        self.assertEqual(proof["phone_call_count"], 5)
        self.assertEqual(proof["static_policy_applied_token_index"], 3)
        self.assertEqual(proof["static_policy_hash"], policy.policy_hash)
        with self.assertRaisesRegex(
            PhysicalAdapterError,
            "static FFN runtime acknowledgement is absent",
        ):
            server.finish_execution(
                marker,
                command,
                self.manifest,
                output_tokens=8,
            )

    def test_static_decode_serializes_control_acknowledgements(self) -> None:
        class ConcurrentControlClient(FakeClient):
            def __init__(self):
                self.active = 0
                self.maximum_active = 0
                self.lock = threading.Lock()

            def apply_ffn_control(self, _endpoint, control, *, timeout_s=5):
                with self.lock:
                    self.active += 1
                    self.maximum_active = max(
                        self.maximum_active, self.active
                    )
                time.sleep(0.01)
                with self.lock:
                    self.active -= 1
                return ({
                    "applied_token_index": 1,
                    "plan_generation": control.plan_generation,
                    "policy_hash": control.policy.policy_hash,
                    "runtime_stats": {},
                    "slot_id": control.slot_id,
                    "success": True,
                }, time.monotonic_ns())

        parameters = dict(self.command.adapter_parameters)
        parameters.update({
            "ffn_assistance_phase": "decode",
            "ffn_max_tokens": 1,
            "ffn_resident_columns": 128,
            "ffn_runtime_control_protocol": "decode-boundary-v1",
            "ffn_selected_columns": 64,
            "ffn_selected_layer_mask": 1,
        })
        operator_plan = {
            **self.command.operator_plan,
            "assisted_operator_kind": "ffn",
            "baseline_executor_id": "synthetic:desktop-parent",
            "desktop_placement_sha256": "sha256:" + "4" * 64,
            "resource_ids": ["compute:host-a", "compute:helper-c"],
            "split_fraction_ppm": 500_000,
        }
        client = ConcurrentControlClient()
        backend = CanonicalHttpExecutionBackend(
            client, FakeMeter(), epoch_ns=time.monotonic_ns()
        )

        def apply(index: int) -> None:
            command = replace(
                self.command,
                request_id="synthetic-request-" + str(index),
                adapter_parameters=parameters,
                operator_plan=operator_plan,
                selection_mode="calibration",
            )
            payload = LlamaCppCompletionPayload(
                request_id=command.request_id,
                expected_model_alias="synthetic-model",
                input_tokens=1,
                output_tokens=8,
                prompt_tokens=(1,),
                seed=1,
                stream_path=self.root / (command.request_id + ".raw"),
                on_first_token=lambda _value: None,
            )
            controlled = backend._static_decode_payload(command, payload)
            controlled.on_decode_progress(
                index, 1, time.monotonic_ns(), False
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            tuple(pool.map(apply, range(4)))

        self.assertEqual(client.maximum_active, 1)

    def test_execution_proof_binds_calls_to_exact_ticket_plan(self) -> None:
        server = self.managed_server()
        marker = server.begin_execution(self.command, self.manifest)
        server.stderr_lines.extend((
            "S41SERVERFFNCALL request=11 layer=0 tokens=1 "
            "columns=64 payload_bytes=64",
            "S41SERVERFFNCALL request=12 layer=0 tokens=1 "
            "columns=64 payload_bytes=64",
        ))

        proof = server.finish_execution(
            marker, self.command, self.manifest, output_tokens=2
        ).to_json()

        self.assertEqual(proof["ticket_id"], self.command.ticket_id)
        self.assertEqual(proof["phone_call_count"], 2)
        self.assertEqual(proof["phone_calls_by_layer"], [
            {"calls": 2, "layer": 0}
        ])
        self.assertEqual(proof["phone_first_request_id"], 11)
        self.assertEqual(proof["phone_last_request_id"], 12)
        self.assertTrue(proof["proof_sha256"].startswith("sha256:"))
        self.assertEqual(len(proof["proof_sha256"]), 71)

    def test_execution_proof_rejects_wrong_split_width(self) -> None:
        server = self.managed_server()
        marker = server.begin_execution(self.command, self.manifest)
        server.stderr_lines.append(
            "S41SERVERFFNCALL request=1 layer=0 tokens=1 "
            "columns=32 payload_bytes=64"
        )

        with self.assertRaisesRegex(
            PhysicalAdapterError, "differ from the operator plan"
        ):
            server.finish_execution(
                marker, self.command, self.manifest, output_tokens=1
            )

    def test_adaptive_execution_proof_matches_window_policies(self) -> None:
        placement = "sha256:" + "4" * 64
        operator_plan = {
            **dict(self.command.operator_plan),
            "assisted_operator_kind": "ffn",
            "desktop_placement_sha256": placement,
        }
        command = adaptive_execution_command(replace(
            self.command,
            operator_plan=operator_plan,
        ))
        contract = phone_ffn_execution_contract(command, self.manifest)
        baseline = AdaptiveDecodePolicy(
            route_id="synthetic-desktop-parent",
            executor_id="synthetic-desktop",
            operator_plan_sha256=PLAN_SHA256,
            desktop_parent_route_id="synthetic-desktop-parent",
            desktop_placement_sha256=placement,
            layer_indices=(),
            layer_mask=0,
            columns=0,
            split_fraction_ppm=0,
            resource_ids=("cpu", "gpu"),
            baseline=True,
        )
        phone = AdaptiveDecodePolicy(
            route_id=command.route_id,
            executor_id=command.executor_id,
            operator_plan_sha256=PLAN_SHA256,
            desktop_parent_route_id=baseline.route_id,
            desktop_placement_sha256=placement,
            layer_indices=(0,),
            layer_mask=1,
            columns=contract.columns,
            split_fraction_ppm=500_000,
            resource_ids=("cpu", "gpu", "phone", "usb"),
        )
        first = AdaptiveDecodeWindowReceipt(
            request_id=command.request_id,
            slot_id=0,
            window_index=0,
            token_start=1,
            token_end=2,
            context_length=9,
            active_batch=1,
            started_at_us=100,
            finished_at_us=200,
            policy=baseline,
            applied_ack=None,
            fleet_energy_uj_by_domain={"fleet": 100},
            latency_per_token_us=100,
            phone_compute_us=0,
            usb_transfer_us=0,
            rpc_us=0,
            exposed_tail_us=0,
            output_valid=True,
            evidence_ids=("synthetic:baseline",),
            energy_boundary_id="synthetic-fleet",
            energy_attribution_kind="isolated",
            failure_reason=None,
            previous_record_sha256="0" * 64,
        )
        acknowledgement = AdaptiveDecodePolicyAck(
            request_id=command.request_id,
            slot_id=0,
            plan_generation=1,
            applied_token_index=2,
            applied_at_us=200,
            policy_hash=phone.policy_hash,
        )
        second = AdaptiveDecodeWindowReceipt(
            request_id=command.request_id,
            slot_id=0,
            window_index=1,
            token_start=2,
            token_end=4,
            context_length=10,
            active_batch=1,
            started_at_us=200,
            finished_at_us=400,
            policy=phone,
            applied_ack=acknowledgement,
            fleet_energy_uj_by_domain={"fleet": 120},
            latency_per_token_us=100,
            phone_compute_us=20,
            usb_transfer_us=10,
            rpc_us=5,
            exposed_tail_us=2,
            output_valid=True,
            evidence_ids=("synthetic:phone",),
            energy_boundary_id="synthetic-fleet",
            energy_attribution_kind="isolated",
            failure_reason=None,
            previous_record_sha256=first.record_sha256.removeprefix(
                "sha256:"
            ),
        )
        third = AdaptiveDecodeWindowReceipt(
            request_id=command.request_id,
            slot_id=0,
            window_index=2,
            token_start=4,
            token_end=5,
            context_length=12,
            active_batch=1,
            started_at_us=400,
            finished_at_us=500,
            policy=phone,
            applied_ack=None,
            fleet_energy_uj_by_domain={"fleet": 60},
            latency_per_token_us=100,
            phone_compute_us=10,
            usb_transfer_us=5,
            rpc_us=3,
            exposed_tail_us=1,
            output_valid=True,
            evidence_ids=("synthetic:control-transition",),
            energy_boundary_id="synthetic-fleet",
            energy_attribution_kind="isolated",
            failure_reason=None,
            previous_record_sha256=second.record_sha256.removeprefix(
                "sha256:"
            ),
        )
        grouped = AdaptiveDecodeGroupedObservation(
            request_id=command.request_id,
            ticket_id=command.ticket_id,
            model_artifact_sha256=command.artifact_sha256,
            planning_profile_sha256=PLAN_SHA256,
            desktop_placement_sha256=placement,
            windows=(first, second, third),
            final_policy=phone,
            terminal_status="COMPLETED",
            state_history=(
                "BASELINE", "PREPARING", "PROBING", "EXPLOITING",
                "COMPLETED",
            ),
            unmeasured_tail_tokens=1,
            unmeasured_tail_reason="server_release_guard",
        )
        server = ManagedLlamaServer(
            ("synthetic-server",),
            {},
            self.root,
            "synthetic-adaptive",
            llama_server_launch_contract(command, self.manifest),
        )
        server.process = SimpleNamespace(poll=lambda: None)
        marker = server.begin_execution(command, self.manifest)
        request_hex = command.request_id.encode("ascii").hex()
        foreign_hex = "synthetic-other".encode("ascii").hex()
        server.stderr_lines.extend((
            f"S41SERVERFFNCALL context={request_hex}:0:1:1 "
            "request=21 layer=0 tokens=1 "
            f"columns={contract.columns} payload_bytes=64",
            f"S41SERVERFFNCALL context={foreign_hex}:1:1:4 "
            "request=91 layer=0 tokens=1 "
            f"columns={contract.columns} payload_bytes=64",
            f"S41SERVERFFNCALL context={request_hex}:0:1:1 "
            "request=22 layer=0 tokens=1 "
            f"columns={contract.columns} payload_bytes=64",
            f"S41SERVERFFNCALL context={request_hex}:0:1:1 "
            "request=23 layer=0 tokens=1 "
            f"columns={contract.columns} payload_bytes=64",
        ))

        proof = server.finish_execution(
            marker,
            command,
            self.manifest,
            output_tokens=6,
            adaptive_observation=grouped,
        ).to_json()

        self.assertEqual(proof["phone_call_count"], 3)
        self.assertEqual(proof["phone_first_request_id"], 21)
        self.assertEqual(proof["phone_last_request_id"], 23)
        self.assertEqual(
            proof["adaptive_grouped_observation_sha256"],
            grouped.grouped_observation_sha256,
        )
        self.assertEqual(proof["adaptive_window_count"], 3)

        zero_ack = AdaptiveDecodePolicyAck(
            request_id=command.request_id, slot_id=0, plan_generation=2,
            applied_token_index=5, applied_at_us=500,
            policy_hash=baseline.policy_hash,
        )
        released_window = replace(
            first, window_index=3, token_start=5, token_end=9,
            started_at_us=500, finished_at_us=900,
            applied_ack=zero_ack, measurement_eligible=False,
            completed_phone_calls=0, completed_phone_input_rows=0,
            failure_reason="released_slot_baseline_tail",
            evidence_ids=("physical:terminal-release-confirmed",),
            previous_record_sha256=third.record_sha256.removeprefix("sha256:"),
        )
        released_group = replace(
            grouped, windows=(*grouped.windows, released_window),
            final_policy=baseline, unmeasured_tail_tokens=0,
            unmeasured_tail_reason=None,
        )
        released_server = ManagedLlamaServer(
            ("synthetic-server",), {}, self.root, "synthetic-released-baseline",
            llama_server_launch_contract(command, self.manifest),
        )
        released_server.process = SimpleNamespace(poll=lambda: None)
        released_marker = released_server.begin_execution(command, self.manifest)
        released_server.stderr_lines.extend(server.stderr_lines)
        released_proof = released_server.finish_execution(
            released_marker, command, self.manifest, output_tokens=9,
            adaptive_observation=released_group,
        ).to_json()
        self.assertEqual(released_proof["phone_call_count"], 3)
        self.assertEqual(released_proof["adaptive_grouped_observation_sha256"],
                         released_group.grouped_observation_sha256)
        zero_window = replace(
            released_window, token_end=7, finished_at_us=700,
            failure_reason=None, evidence_ids=("synthetic:zero-mask-window",),
        )
        continued_window = replace(
            released_window, window_index=4, token_start=7, started_at_us=700,
            applied_ack=None,
            previous_record_sha256=zero_window.record_sha256.removeprefix("sha256:"),
        )
        continued_group = replace(
            released_group, windows=(*grouped.windows, zero_window, continued_window),
        )
        self.assertEqual(released_server.finish_execution(
            released_marker, command, self.manifest, output_tokens=9,
            adaptive_observation=continued_group,
        ).to_json()["phone_call_count"], 3)
        for invalid_ack in (None, replace(zero_ack, plan_generation=1)):
            invalid_zero = replace(zero_window, applied_ack=invalid_ack)
            with self.subTest(invalid_ack=invalid_ack):
                with self.assertRaisesRegex(PhysicalAdapterError, "observation differs"):
                    released_server.finish_execution(
                        released_marker, command, self.manifest, output_tokens=9,
                        adaptive_observation=replace(continued_group, windows=(
                            *grouped.windows, invalid_zero, replace(continued_window,
                                previous_record_sha256=invalid_zero.record_sha256.removeprefix("sha256:")),
                        )),
                    )
        with self.assertRaisesRegex(PhysicalAdapterError, "observation differs"):
            released_server.finish_execution(
                released_marker, command, self.manifest, output_tokens=9,
                adaptive_observation=replace(released_group, windows=(
                    *grouped.windows, replace(released_window, applied_ack=None),
                )),
            )
        released_server.stderr_lines.append(
            f"S41SERVERFFNCALL context={request_hex}:0:2:1 "
            f"request=24 layer=0 tokens=1 columns={contract.columns} payload_bytes=64"
        )
        with self.assertRaises(PhysicalAdapterError):
            released_server.finish_execution(
                released_marker, command, self.manifest, output_tokens=9,
                adaptive_observation=released_group,
            )

        stale_grouped = replace(
            grouped,
            unmeasured_tail_tokens=2,
            unmeasured_tail_reason="stale_slot_control_discarded",
        )
        stale = ManagedLlamaServer(
            ("synthetic-server",),
            {},
            self.root,
            "synthetic-adaptive-stale-control",
            llama_server_launch_contract(command, self.manifest),
        )
        stale.process = SimpleNamespace(poll=lambda: None)
        stale_marker = stale.begin_execution(command, self.manifest)
        stale.stderr_lines.extend(
            f"S41SERVERFFNCALL context={request_hex}:0:1:1 "
            f"request={request_id} layer=0 tokens=1 "
            f"columns={contract.columns} payload_bytes=64"
            for request_id in (31, 32, 33, 34)
        )
        stale_proof = stale.finish_execution(
            stale_marker,
            command,
            self.manifest,
            output_tokens=7,
            adaptive_observation=stale_grouped,
        ).to_json()
        self.assertEqual(stale_proof["phone_call_count"], 4)

    def test_adaptive_proof_ignores_unattributed_calls_after_stale_tail(
        self,
    ) -> None:
        placement = "sha256:" + "4" * 64
        command = adaptive_execution_command(replace(
            self.command,
            operator_plan={
                **dict(self.command.operator_plan),
                "assisted_operator_kind": "ffn",
                "desktop_placement_sha256": placement,
            },
        ))
        baseline = AdaptiveDecodePolicy(
            route_id="synthetic-desktop-parent",
            executor_id="synthetic-desktop",
            operator_plan_sha256=PLAN_SHA256,
            desktop_parent_route_id="synthetic-desktop-parent",
            desktop_placement_sha256=placement,
            layer_indices=(),
            layer_mask=0,
            columns=0,
            split_fraction_ppm=0,
            resource_ids=("cpu", "gpu"),
            baseline=True,
        )
        stale = AdaptiveDecodeWindowReceipt(
            request_id=command.request_id,
            slot_id=0,
            window_index=0,
            token_start=1,
            token_end=2,
            context_length=9,
            active_batch=1,
            started_at_us=100,
            finished_at_us=200,
            policy=baseline,
            applied_ack=None,
            fleet_energy_uj_by_domain={"fleet": 100},
            latency_per_token_us=100,
            phone_compute_us=0,
            usb_transfer_us=0,
            rpc_us=0,
            exposed_tail_us=0,
            output_valid=True,
            evidence_ids=("physical:stale-slot-stats-discarded",),
            energy_boundary_id="synthetic-fleet",
            energy_attribution_kind="diagnostic",
            failure_reason="stale_slot_stats_discarded",
            previous_record_sha256="0" * 64,
            measurement_eligible=False,
            completed_phone_calls=0,
            completed_phone_input_rows=0,
        )
        grouped = AdaptiveDecodeGroupedObservation(
            request_id=command.request_id,
            ticket_id=command.ticket_id,
            model_artifact_sha256=command.artifact_sha256,
            planning_profile_sha256=PLAN_SHA256,
            desktop_placement_sha256=placement,
            windows=(stale,),
            final_policy=baseline,
            terminal_status="COMPLETED",
            state_history=("BASELINE", "EXPLOITING", "COMPLETED"),
            unmeasured_tail_tokens=1,
            unmeasured_tail_reason="stale_slot_stats_discarded",
        )
        server = ManagedLlamaServer(
            ("synthetic-server",),
            {},
            self.root,
            "synthetic-stale-tail",
            llama_server_launch_contract(command, self.manifest),
        )
        server.process = SimpleNamespace(poll=lambda: None)
        marker = server.begin_execution(command, self.manifest)
        server.stderr_lines.append(
            "S41SERVERFFNCALL request=99 layer=0 tokens=1 "
            "columns=32 payload_bytes=64"
        )

        proof = server.finish_execution(
            marker,
            command,
            self.manifest,
            output_tokens=3,
            adaptive_observation=grouped,
        ).to_json()

        self.assertEqual(proof["phone_call_count"], 0)
        self.assertEqual(
            proof["adaptive_grouped_observation_sha256"],
            grouped.grouped_observation_sha256,
        )

    def test_adaptive_cohort_proof_is_shared_and_accepts_batched_calls(
        self,
    ) -> None:
        placement = "sha256:" + "4" * 64
        members = ("synthetic-request", "synthetic-follower")
        cohort = {
            "active_batch": 2,
            "cohort_id": "synthetic-cohort",
            "common_policy_sha256": "sha256:" + "5" * 64,
            "key_sha256": "sha256:" + "6" * 64,
            "leader_request_id": members[0],
            "maximum_members": 4,
            "member_request_ids": list(members),
            "schema": "research-scheduler-decode-cohort-v2",
            "sealed": True,
            "shared_lease_tokens": ["lease-a"],
        }
        parameters = {
            **dict(self.command.adapter_parameters),
            "ffn_column_quantum": 32,
            "ffn_max_tokens": 4,
            "parallel": 4,
        }
        operator_plan = {
            **dict(self.command.operator_plan),
            "assisted_operator_kind": "ffn",
            "desktop_placement_sha256": placement,
        }
        leader = adaptive_execution_command(replace(
            self.command,
            adapter_parameters=parameters,
            decode_cohort=cohort,
            operator_plan=operator_plan,
        ))
        follower = PhysicalExecutionCommand(**{
            **leader.__dict__,
            "request_id": members[1],
            "ticket_id": members[1] + ":attempt:0",
        })
        contract = phone_ffn_execution_contract(leader, self.manifest)
        baseline = AdaptiveDecodePolicy(
            route_id="synthetic-desktop-parent",
            executor_id="synthetic-desktop",
            operator_plan_sha256=PLAN_SHA256,
            desktop_parent_route_id="synthetic-desktop-parent",
            desktop_placement_sha256=placement,
            layer_indices=(),
            layer_mask=0,
            columns=0,
            split_fraction_ppm=0,
            resource_ids=("cpu", "gpu"),
            baseline=True,
        )
        phone = AdaptiveDecodePolicy(
            route_id=leader.route_id,
            executor_id=leader.executor_id,
            operator_plan_sha256=PLAN_SHA256,
            desktop_parent_route_id=baseline.route_id,
            desktop_placement_sha256=placement,
            layer_indices=(0,),
            layer_mask=1,
            columns=contract.columns,
            split_fraction_ppm=500_000,
            resource_ids=("cpu", "gpu", "phone", "usb"),
        )

        def window(
            index: int,
            start: int,
            end: int,
            policy: AdaptiveDecodePolicy,
            previous: str,
        ) -> AdaptiveDecodeWindowReceipt:
            return AdaptiveDecodeWindowReceipt(
                request_id=members[0],
                slot_id=0,
                window_index=index,
                token_start=start,
                token_end=end,
                context_length=9 + start,
                active_batch=2,
                started_at_us=start * 100,
                finished_at_us=end * 100,
                policy=policy,
                applied_ack=None,
                fleet_energy_uj_by_domain={"fleet": 100},
                latency_per_token_us=100,
                phone_compute_us=10 if not policy.baseline else 0,
                usb_transfer_us=5 if not policy.baseline else 0,
                rpc_us=2 if not policy.baseline else 0,
                exposed_tail_us=1 if not policy.baseline else 0,
                output_valid=True,
                evidence_ids=("synthetic:cohort",),
                energy_boundary_id="synthetic-fleet",
                energy_attribution_kind="isolated",
                failure_reason=None,
                previous_record_sha256=previous,
                accounting_token_count=(
                    (end - start) * 2
                    + (2 if not policy.baseline else 0)
                ),
                cohort_id="synthetic-cohort",
                cohort_member_request_ids=members,
                energy_owner_request_id=members[0],
                transfer_subrequests=(
                    0 if policy.baseline else end - start + 1
                ),
                completed_phone_calls=(
                    0 if policy.baseline else end - start
                ),
                completed_phone_input_rows=(
                    0 if policy.baseline else (end - start) * 2
                ),
            )

        first = window(0, 1, 2, baseline, "0" * 64)
        second = window(
            1,
            2,
            4,
            phone,
            first.record_sha256.removeprefix("sha256:"),
        )
        singleton = AdaptiveDecodeWindowReceipt(
            request_id=members[0],
            slot_id=0,
            window_index=2,
            token_start=4,
            token_end=5,
            context_length=13,
            active_batch=1,
            started_at_us=400,
            finished_at_us=500,
            policy=phone,
            applied_ack=None,
            fleet_energy_uj_by_domain={"fleet": 50},
            latency_per_token_us=100,
            phone_compute_us=10,
            usb_transfer_us=5,
            rpc_us=2,
            exposed_tail_us=1,
            output_valid=True,
            evidence_ids=("synthetic:singleton",),
            energy_boundary_id="synthetic-fleet",
            energy_attribution_kind="isolated",
            failure_reason=None,
            previous_record_sha256=second.record_sha256.removeprefix(
                "sha256:"
            ),
            transfer_subrequests=2,
            completed_phone_calls=1,
            completed_phone_input_rows=1,
        )
        grouped = AdaptiveDecodeGroupedObservation(
            request_id=members[0],
            ticket_id=leader.ticket_id,
            model_artifact_sha256=leader.artifact_sha256,
            planning_profile_sha256=PLAN_SHA256,
            desktop_placement_sha256=placement,
            windows=(first, second, singleton),
            final_policy=phone,
            terminal_status="COMPLETED",
            state_history=(
                "BASELINE", "PREPARING", "PROBING", "COMPLETED",
            ),
            unmeasured_tail_tokens=1,
            unmeasured_tail_reason="server_release_guard",
        )

        early = ManagedLlamaServer(
            ("synthetic-server",),
            {},
            self.root,
            "synthetic-cohort-early",
            llama_server_launch_contract(leader, self.manifest),
        )
        early.process = SimpleNamespace(poll=lambda: None)
        early_marker = early.begin_execution(leader, self.manifest)
        early.stderr_lines.append(
            "S41SERVERFFNCALL request=31 layer=0 tokens=2 "
            "columns=32 payload_bytes=128"
        )
        early_proof = early.finish_execution(
            early_marker, leader, self.manifest, output_tokens=2
        ).to_json()
        self.assertNotIn(
            "adaptive_grouped_observation_sha256", early_proof
        )

        last = ManagedLlamaServer(
            ("synthetic-server",),
            {},
            self.root,
            "synthetic-cohort-last",
            llama_server_launch_contract(follower, self.manifest),
        )
        last.process = SimpleNamespace(poll=lambda: None)
        last_marker = last.begin_execution(follower, self.manifest)
        last.stderr_lines.append(
            "S41SERVERFFNCALL request=32 layer=0 tokens=4 "
            f"columns={contract.columns} payload_bytes=256"
        )
        def append_terminal_call() -> None:
            time.sleep(1.1)
            with last._stderr_lock:
                last.stderr_lines.append(
                    "S41SERVERFFNCALL request=33 layer=0 tokens=1 "
                    f"columns={contract.columns} payload_bytes=64"
                )
                notify = getattr(last._stderr_lock, "notify_all", None)
                if callable(notify):
                    notify()

        terminal_call = threading.Thread(target=append_terminal_call)
        terminal_call.start()
        last_proof = last.finish_execution(
            last_marker,
            follower,
            self.manifest,
            output_tokens=8,
            adaptive_observation=grouped,
        ).to_json()
        terminal_call.join()
        self.assertEqual(
            last_proof["adaptive_group_owner_request_id"], members[0]
        )
        self.assertEqual(last_proof["phone_call_count"], 2)
        self.assertEqual(last_proof["adaptive_window_count"], 3)

        multi_member_tail = AdaptiveDecodeGroupedObservation(
            request_id=members[0],
            ticket_id=leader.ticket_id,
            model_artifact_sha256=leader.artifact_sha256,
            planning_profile_sha256=PLAN_SHA256,
            desktop_placement_sha256=placement,
            windows=(first, second),
            final_policy=phone,
            terminal_status="COMPLETED",
            state_history=(
                "BASELINE", "PREPARING", "PROBING", "COMPLETED",
            ),
            unmeasured_tail_tokens=1,
            unmeasured_tail_reason="server_release_guard",
        )
        invalid = ManagedLlamaServer(
            ("synthetic-server",),
            {},
            self.root,
            "synthetic-cohort-invalid-tail",
            llama_server_launch_contract(follower, self.manifest),
        )
        invalid.process = SimpleNamespace(poll=lambda: None)
        invalid_marker = invalid.begin_execution(follower, self.manifest)
        with self.assertRaisesRegex(
            PhysicalAdapterError,
            "adaptive cohort tail requires singleton ownership",
        ):
            invalid.finish_execution(
                invalid_marker,
                follower,
                self.manifest,
                output_tokens=8,
                adaptive_observation=multi_member_tail,
            )

    def test_http_backend_persists_success_callback_proof(self) -> None:
        proof = {
            "executor_id": self.command.executor_id,
            "ticket_id": self.command.ticket_id,
        }
        finished = []
        backend = CanonicalHttpExecutionBackend(
            FakeClient(),
            FakeMeter(),
            epoch_ns=time.monotonic_ns() - 1_000_000,
            on_execution_success=lambda command, _value: (
                proof if command == self.command else None
            ),
            on_execution_finish=finished.append,
        )
        payload = LlamaCppCompletionPayload(
            request_id=self.command.request_id,
            expected_model_alias="synthetic-model",
            input_tokens=1,
            output_tokens=1,
            prompt_tokens=(1,),
            seed=1,
            stream_path=self.root / "stream.raw",
            on_first_token=lambda _value: None,
        )

        observation = backend.execute(
            self.command, payload, lambda: None
        )

        self.assertEqual(
            observation.payload["physical_execution_proof"], proof
        )
        self.assertEqual(finished, [self.command])

    def test_http_backend_excludes_proof_drain_from_inference_window(
        self,
    ) -> None:
        meter = RecordingMeter()
        epoch_ns = time.monotonic_ns() - 1_000_000

        def delayed_proof(command, _value):
            time.sleep(0.05)
            return {
                "executor_id": command.executor_id,
                "ticket_id": command.ticket_id,
            }

        backend = CanonicalHttpExecutionBackend(
            FakeClient(),
            meter,
            epoch_ns=epoch_ns,
            on_execution_success=delayed_proof,
        )
        payload = LlamaCppCompletionPayload(
            request_id=self.command.request_id,
            expected_model_alias="synthetic-model",
            input_tokens=1,
            output_tokens=1,
            prompt_tokens=(1,),
            seed=1,
            stream_path=self.root / "proof-drain-timing.raw",
            on_first_token=lambda _value: None,
        )

        observation = backend.execute(
            self.command, payload, lambda: None
        )

        timing = observation.payload["physical_execution_timing"]
        self.assertGreaterEqual(
            timing["proof_drain_duration_us"], 40_000
        )
        self.assertGreaterEqual(
            timing["proof_drain_finished_us"]
            - observation.finished_us,
            40_000,
        )
        self.assertEqual(
            (meter.intervals[0][1] - epoch_ns) // 1000,
            observation.finished_us,
        )

    def test_http_backend_does_not_finish_an_execution_that_never_started(
        self,
    ) -> None:
        finished = []

        def fail_start(_command):
            raise PhysicalAdapterError("primary start failure")

        def fail_finish(command):
            finished.append(command)
            raise PhysicalAdapterError("secondary finish failure")

        backend = CanonicalHttpExecutionBackend(
            FakeClient(),
            FakeMeter(),
            epoch_ns=time.monotonic_ns() - 1_000_000,
            on_execution_start=fail_start,
            on_execution_finish=fail_finish,
        )
        payload = LlamaCppCompletionPayload(
            request_id=self.command.request_id,
            expected_model_alias="synthetic-model",
            input_tokens=1,
            output_tokens=1,
            prompt_tokens=(1,),
            seed=1,
            stream_path=self.root / "stream.raw",
            on_first_token=lambda _value: None,
        )

        with self.assertRaisesRegex(
            PhysicalAdapterError, "primary start failure"
        ):
            backend.execute(self.command, payload, lambda: None)

        self.assertEqual(finished, [])

    def test_http_backend_uses_an_independent_energy_window_per_ticket(
        self,
    ) -> None:
        meter = RecordingMeter()
        backend = CanonicalHttpExecutionBackend(
            FakeClient(),
            meter,
            epoch_ns=time.monotonic_ns() - 1_000_000,
        )
        payload = LlamaCppCompletionPayload(
            request_id=self.command.request_id,
            expected_model_alias="synthetic-model",
            input_tokens=1,
            output_tokens=1,
            prompt_tokens=(1,),
            seed=1,
            stream_path=self.root / "stream.raw",
            on_first_token=lambda _value: None,
        )
        second = PhysicalExecutionCommand(
            **{
                **self.command.__dict__,
                "request_id": "synthetic-request-2",
                "ticket_id": "synthetic-request-2:attempt:0",
            }
        )

        backend.execute(self.command, payload, lambda: None)
        backend.execute(second, payload, lambda: None)

        self.assertEqual(len(meter.intervals), 2)
        self.assertEqual(meter.prepare_count, 2)
        self.assertGreater(meter.intervals[1][0], meter.intervals[0][0])


if __name__ == "__main__":
    unittest.main()
