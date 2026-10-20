#!/usr/bin/env python3

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from research_dev.scheduler._internal.policy import LeaseRecord
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeDirective,
    AdaptiveDecodePolicy,
    AdaptiveDecodeWindowBoundary,
)
from research_dev.scheduler._internal.runtime_cost import (
    RuntimeExecutorBinding,
    RuntimeParticipantBinding,
)
from research_dev.scheduler._internal.runtime_decode_cohort import (
    RuntimeDecodeCohortManager,
)
from research_dev.scheduler._internal.runtime_plan import (
    RuntimeExecutionContract,
    RuntimeExecutionPlan,
    RuntimeOperatorAssignment,
    RuntimeTransitionPlan,
)
from research_dev.scheduler.adapters.contracts import RawEnergyMeasurement
from research_dev.scheduler.adapters.decode_cohort import (
    DecodeCohortExecutionTracker,
    DecodeCohortPolicyCoordinator,
)
from research_dev.scheduler.adapters.http_backend import (
    CanonicalHttpExecutionBackend,
    LlamaCppCompletionPayload,
    LlamaCppHttpClient,
)
from research_dev.scheduler.adapters.transitions import (
    CanonicalTransitionRegistry,
)
from research_dev.scheduler.adapters.ticket import PhysicalExecutionCommand
from research_dev.scheduler.adapters.ticket import (
    PhysicalParticipantCommand,
    PhysicalTransitionCommand,
)


ARTIFACT = "sha256:" + "a" * 64
PLACEMENT = "sha256:" + "b" * 64
GEOMETRY = "sha256:" + "c" * 64


def plan() -> RuntimeExecutionPlan:
    return RuntimeExecutionPlan(
        route_id="synthetic-cohort-route",
        route_family="operator_split",
        device_ids=("desktop", "helper"),
        assisted_operator_kind="ffn",
        split_axis="column",
        split_fraction_ppm=500_000,
        residency_variant="hot",
        overlap_kind="parallel_join",
        operators=(RuntimeOperatorAssignment(
            operator_id="layer:0:ffn",
            operator_kind="ffn",
            candidate_id="synthetic-cohort-route",
            device_ids=("desktop", "helper"),
            split_axis="column",
            split_fraction_ppm=500_000,
        ),),
        transitions=(),
        resource_ids=("compute:desktop", "link:helper"),
        memory_demands=(),
        execution_contract=RuntimeExecutionContract(
            execution_mode="adaptive-split",
            initial_split_fraction_ppm=0,
            allowed_adaptive_fractions_ppm=(0, 500_000),
            batch_plan="split-row",
            maximum_batch_size=4,
            queue_depth=4,
            phone_device_id="helper",
            phone_endpoint="http://127.0.0.1:12345",
            operator_kind="ffn",
        ),
        adapter_parameters={
            "decode_cohort_formation_us": 100_000,
            "ffn_assistance_phase": "decode",
            "ffn_column_quantum": 32,
            "ffn_resident_columns": 256,
            "ffn_resident_geometry_sha256": GEOMETRY,
            "ffn_resident_layer_mask": 1,
            "ffn_runtime_control_protocol": "decode-boundary-v1",
            "ffn_weight_buffer_layout": "resident-superset",
            "parallel": 4,
            "phone_device_id": "helper",
            "usb_concurrent_streams": 4,
            "usb_queue_depth": 4,
        },
        baseline_executor_id="desktop-parent",
        desktop_placement_sha256=PLACEMENT,
    )


def transition_command(
    request_id: str,
    members: tuple[str, ...],
) -> PhysicalTransitionCommand:
    transition = RuntimeTransitionPlan(
        transition_id="load:synthetic-helper",
        device_id="helper",
        source_state="cold",
        target_state="hot",
        latency_us=1,
        energy_uj=1,
        resource_ids=("helper-transition",),
        maturity="QUALIFIED",
        executor_id="synthetic-cohort",
    )
    return PhysicalTransitionCommand(
        ticket_id="ticket-" + request_id,
        request_id=request_id,
        artifact_sha256=ARTIFACT,
        route_id="synthetic-cohort-route",
        operator_plan_sha256="sha256:" + "d" * 64,
        participant=PhysicalParticipantCommand(
            executor_id="synthetic-cohort",
            device_id="helper",
            endpoint="http://127.0.0.1:12345",
            backend="synthetic",
            resource_ids=("helper-transition",),
        ),
        transition=transition,
        execution_contract=plan().execution_contract,
        adapter_parameters={"resident_geometry": GEOMETRY},
        decode_cohort={
            "cohort_id": "cohort-physical",
            "leader_request_id": members[0],
            "member_request_ids": list(members),
            "sealed": True,
        },
    )


def binding(value: RuntimeExecutionPlan) -> RuntimeExecutorBinding:
    return RuntimeExecutorBinding(
        executor_id="synthetic-cohort-executor",
        route_id=value.route_id,
        model_id="synthetic-model",
        artifact_sha256=ARTIFACT,
        artifact_bytes=1024,
        backend="synthetic-composite",
        resource_ids=value.resource_ids,
        memory_resource_id="memory:desktop",
        resident=True,
        ready=True,
        route_family=value.route_family,
        operator_plan_sha256=value.plan_sha256,
        endpoint="http://127.0.0.1:18080",
        operator_plan_protocol="synthetic-v1",
    )


def command(
    request_id: str,
    members: tuple[str, ...],
    lease_tokens: tuple[str, ...],
) -> PhysicalExecutionCommand:
    execution_contract = plan().execution_contract
    return PhysicalExecutionCommand(
        ticket_id=request_id + ":attempt:0",
        request_id=request_id,
        model_id="synthetic-model",
        artifact_sha256=ARTIFACT,
        route_id="synthetic-cohort-route",
        executor_id="synthetic-cohort-executor",
        endpoint="http://127.0.0.1:18080",
        operator_plan_protocol="synthetic-v1",
        operator_plan_sha256="sha256:" + "d" * 64,
        planned_start_us=1,
        planned_finish_us=2,
        planned_finish_upper_us=3,
        operator_plan={
            "assisted_operator_kind": "ffn",
            "execution_contract": execution_contract.to_json(),
            "plan_sha256": "sha256:" + "d" * 64,
        },
        participants=(
            PhysicalParticipantCommand(
                executor_id="synthetic-desktop",
                device_id="desktop",
                endpoint="http://127.0.0.1:18080",
                backend="synthetic",
                resource_ids=("compute:desktop",),
            ),
            PhysicalParticipantCommand(
                executor_id="synthetic-helper",
                device_id="helper",
                endpoint="http://127.0.0.1:12345",
                backend="synthetic",
                resource_ids=("link:helper",),
            ),
        ),
        leases=(),
        memory_reservations=(),
        transitions=(),
        execution_contract=execution_contract,
        adapter_parameters={
            "ffn_assistance_phase": "decode",
            "ffn_max_tokens": 4,
            "ffn_runtime_control_protocol": "decode-boundary-v1",
            "phone_device_id": "helper",
            "usb_batch_plan": "split-row",
            "usb_queue_depth": 4,
        },
        selection_mode="adaptive-decode",
        decode_cohort={
            "active_batch": len(members),
            "cohort_id": "decode-cohort-1",
            "common_policy_sha256": "sha256:" + "e" * 64,
            "key_sha256": "sha256:" + "f" * 64,
            "leader_request_id": members[0],
            "maximum_members": 4,
            "member_request_ids": list(members),
            "schema": "research-scheduler-decode-cohort-v2",
            "sealed": True,
            "shared_lease_tokens": list(lease_tokens),
        },
    )


class Meter:
    def __init__(self, attribution_kind="isolated") -> None:
        self.prepare_count = 0
        self.measure_count = 0
        self.attribution_kind = attribution_kind

    def prepare(self) -> None:
        self.prepare_count += 1

    def measure(self, _started_ns: int, _finished_ns: int):
        self.measure_count += 1
        return RawEnergyMeasurement(
            energy_boundary_id="synthetic-fleet",
            fleet_energy_uj_by_domain={"fleet": 100},
            measurement_evidence_ids=("synthetic-cohort-meter",),
            attribution_kind=self.attribution_kind,
        )


class CohortClient(LlamaCppHttpClient):
    def __init__(self, members: tuple[str, ...]) -> None:
        super().__init__()
        self._members = members
        self._slots = {
            request_id: index for index, request_id in enumerate(members)
        }
        self._tokens = {request_id: 0 for request_id in members}
        self._lock = threading.Lock()
        self._token_barrier = threading.Barrier(len(members))

    @staticmethod
    def _stats() -> dict[str, int]:
        return {
            "batched_calls": 0,
            "calls": 0,
            "configured_queue_depth": 4,
            "desktop_compute_us": 0,
            "download_bytes": 0,
            "exposed_tail_us": 0,
            "input_rows": 0,
            "maximum_active_slots": 4,
            "maximum_outstanding_transfers": 4,
            "maximum_tokens": 4,
            "phone_compute_us": 0,
            "rpc_us": 0,
            "transfer_subrequests": 0,
            "upload_bytes": 0,
            "usb_d2h_us": 0,
            "usb_h2d_us": 0,
            "usb_transfer_us": 0,
            "useful_overlap_us": 0,
        }

    def read_ffn_cohort_stats(self, _endpoint, members, **_kwargs):
        with self._lock:
            rows = [
                {
                    "applied_token_index": self._tokens[request_id],
                    "plan_generation": 0,
                    "request_id": request_id,
                    "slot_id": slot_id,
                }
                for request_id, slot_id in members
            ]
        return ({
            "applied_token_index": min(
                row["applied_token_index"] for row in rows
            ),
            "cohort_members": rows,
            "plan_generation": 0,
            "policy_hash": "",
            "runtime_stats": self._stats(),
            "slot_id": rows[0]["slot_id"],
            "success": True,
        }, time.monotonic_ns())

    def read_ffn_stats(
        self, _endpoint, request_id, slot_id, **_kwargs
    ):
        with self._lock:
            token = self._tokens[request_id]
        return ({
            "applied_token_index": token,
            "plan_generation": 0,
            "policy_hash": "",
            "runtime_stats": self._stats(),
            "slot_id": slot_id,
            "success": True,
        }, time.monotonic_ns())

    def complete(
        self,
        _endpoint,
        payload,
        control_check,
        *,
        scheduler_headers=None,
    ):
        control_check()
        request_id = payload.request_id
        slot_id = self._slots[request_id]
        if payload.on_active_batch is not None:
            payload.on_active_batch(4)
        for token, terminal in ((1, False), (2, False), (3, True)):
            with self._lock:
                self._tokens[request_id] = token
            payload.on_decode_progress(
                slot_id, token, time.monotonic_ns(), terminal
            )
            self._token_barrier.wait()
        return {"stream_sha256": "3" * 64, "tokens": [1, 2, 3]}


class SchedulerSink:
    def __init__(self) -> None:
        self.starts = []
        self.boundaries = []
        self.seals = []
        self.previews = []
        self.completions = []
        self.cohort_receipts = []

    def start_adaptive_decode(self, request_id, **values):
        self.starts.append((request_id, values))
        return None

    def adaptive_decode_boundary(self, request_id, **values):
        self.boundaries.append((request_id, values))
        return None

    def seal_adaptive_decode_tail(self, request_id, **values):
        self.seals.append((request_id, values))

    def record_adaptive_decode_window(self, *_args, **_kwargs):
        raise AssertionError("no synthetic window was requested")

    def acknowledge_adaptive_decode_control(self, *_args, **_kwargs):
        raise AssertionError("no synthetic control was requested")

    def fail_adaptive_decode_control(self, *_args, **_kwargs):
        raise AssertionError("no synthetic control failed")

    def preview_adaptive_decode_completion(self, request_id):
        self.previews.append(request_id)
        return SimpleNamespace(
            grouped_observation_sha256="sha256:" + "4" * 64,
            to_json=lambda: {"request_id": request_id},
        )

    def complete_adaptive_decode(self, request_id):
        self.completions.append(request_id)
        return SimpleNamespace(
            grouped_observation_sha256="sha256:" + "4" * 64,
        )

    def record_decode_cohort_measurement(self, cohort_id, **values):
        self.cohort_receipts.append((cohort_id, values))


class DecodeCohortTests(unittest.TestCase):
    def test_coalesced_capacity_uses_worker_rows_not_usb_stream_count(self):
        original = plan()
        self.assertEqual(RuntimeDecodeCohortManager.capacity(original), 4)
        coalesced = replace(original,
            execution_contract=replace(original.execution_contract, batch_plan="coalesced-batch", maximum_batch_size=8),
            adapter_parameters={**dict(original.adapter_parameters), "parallel": 8,
                "ffn_max_tokens": 8, "usb_batch_plan": "coalesced-batch", "usb_concurrent_streams": 1})
        self.assertEqual(RuntimeDecodeCohortManager.capacity(coalesced), 4)
        for maximum in (1, 2, 4, 8):
            changed = replace(coalesced, adapter_parameters={**dict(coalesced.adapter_parameters), "ffn_max_tokens": maximum})
            self.assertEqual(RuntimeDecodeCohortManager.capacity(changed), min(4, maximum))
        changed = replace(coalesced, execution_contract=original.execution_contract)
        self.assertEqual(RuntimeDecodeCohortManager.capacity(changed), 1)
        manager = RuntimeDecodeCohortManager()
        for index in range(4):
            admitted = manager.admit(f"request-{index}", coalesced, binding(coalesced), quality_requirement="exact",
                observed_at_us=100 + index, service_upper_us=500, now_ns=1000 + index)
            self.assertEqual(admitted.binding.active_batch, index + 1)
            if index == 0:
                leases = tuple(LeaseRecord(token="shared-" + resource, owner_id=admitted.binding.cohort_id,
                    lease_id="execution", resource_id=resource, lanes=(0,), start_us=100,
                    predicted_end_us=500, reserved_until_us=10_000) for resource in coalesced.resource_ids)
                manager.bind_leases(admitted.binding.cohort_id, leases)
            else:
                self.assertEqual(admitted.shared_leases, leases)
        self.assertEqual(manager.wait_until_sealed("request-0").active_batch, 4)
        fifth = manager.admit("request-4", coalesced, binding(coalesced), quality_requirement="exact",
            observed_at_us=104, service_upper_us=500, now_ns=1004)
        self.assertNotEqual(fifth.binding.cohort_id, admitted.binding.cohort_id)
        self.assertTrue(fifth.leader)
        self.assertEqual(fifth.shared_leases, ())
        for index in range(4):
            self.assertEqual(manager.mark_terminal(f"request-{index}")[0], index == 3)

    def test_http_cohort_supports_eight_distinct_slots_and_rejects_overflow(self):
        members = tuple((f"request-{index}", index) for index in range(8))
        self.assertEqual(LlamaCppHttpClient._cohort_members(members), members)
        for invalid in (members + (("request-8", 8),), members[:-1] + (("request-0", 7),),
                        members[:-1] + (("request-7", 0),)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(Exception, "members are invalid"):
                LlamaCppHttpClient._cohort_members(invalid)

    def test_cold_preparation_cannot_share_decode_leases(self):
        hot = plan()
        cold = replace(hot, transitions=(RuntimeTransitionPlan(
            transition_id="cold-load",
            device_id="desktop",
            source_state="absent",
            target_state="hot",
            latency_us=10,
            energy_uj=1,
            resource_ids=("compute:desktop",),
            resource_slots={"compute:desktop": 4},
            maturity="QUALIFIED",
            executor_id="synthetic-cohort-executor",
        ),))
        manager = RuntimeDecodeCohortManager()
        self.assertEqual(
            manager.candidate_key(cold, binding(cold), quality_requirement="exact"),
            manager.candidate_key(hot, binding(hot), quality_requirement="exact"),
        )
        self.assertFalse(manager.can_join(cold, binding(cold), quality_requirement="exact"))
        self.assertIsNone(manager.admit(
            "cold", cold, binding(cold), quality_requirement="exact",
            observed_at_us=0, service_upper_us=100,
        ))
        self.assertIsNotNone(manager.admit(
            "ready", hot, binding(hot), quality_requirement="exact",
            observed_at_us=10, service_upper_us=90,
        ))

    def test_completion_declared_limit_reports_one_terminal_boundary(self):
        lines = (
            b'data: {"content":"one ","id_slot":0,"stop":false,'
            b'"tokens":[11],"tokens_predicted":1}\n',
            b'data: {"content":"two","id_slot":0,"stop":false,'
            b'"tokens":[12],"tokens_predicted":2}\n',
            b'data: {"content":"","id_slot":0,"model":"synthetic-model",'
            b'"stop":true,"timings":{"predicted_ms":2.0,'
            b'"predicted_n":2,"prompt_ms":1.0,"prompt_n":1},'
            b'"tokens":[],"tokens_predicted":2}\n',
        )

        class Response:
            status = 200

            def __init__(self):
                self._lines = iter(lines)

            def readline(self):
                return next(self._lines, b"")

        class Connection:
            def __init__(self, *_args, **_kwargs):
                self.response = Response()

            def connect(self):
                pass

            def request(self, *_args, **_kwargs):
                pass

            def getresponse(self):
                return self.response

            def close(self):
                pass

        progress = []
        with tempfile.TemporaryDirectory() as root:
            stream = Path(root) / "request.raw"
            payload = LlamaCppCompletionPayload(
                request_id="request-limit",
                expected_model_alias="synthetic-model",
                input_tokens=1,
                output_tokens=2,
                prompt_tokens=(1,),
                seed=1,
                stream_path=stream,
                on_first_token=lambda _value: None,
                on_decode_progress=lambda slot, token, _at, terminal:
                    progress.append((slot, token, terminal)),
                quality_mode="semantic",
            )
            with patch(
                "research_dev.scheduler.adapters.http_backend.http.client."
                "HTTPConnection",
                Connection,
            ):
                result = LlamaCppHttpClient(
                    slots_probe=lambda _host, _port, _timeout_s: [{
                        "id": 0,
                        "id_task": 7,
                        "is_processing": True,
                    }]
                ).complete(
                    "http://127.0.0.1:18080",
                    payload,
                    lambda: None,
                )

        self.assertEqual(progress, [(0, 1, False), (0, 2, True)])
        self.assertEqual(result["tokens"], [11, 12])

    def test_cold_decode_plan_can_form_a_cohort(self):
        value = plan()
        cold = replace(
            value,
            residency_variant="cold",
            transitions=(
                transition_command("request-a", ("request-a", "request-b"))
                    .transition,
            ),
        )

        key = RuntimeDecodeCohortManager.candidate_key(
            cold,
            replace(
                binding(value),
                operator_plan_sha256=cold.plan_sha256,
            ),
            quality_requirement="exact",
        )

        self.assertIsNotNone(key)

    def test_cold_cohort_requires_identical_transition_and_participant(self):
        hot = plan()
        transition = transition_command(
            "first", ("first", "second")
        ).transition
        cold = replace(hot, transitions=(transition,))
        participant = RuntimeParticipantBinding(
            executor_id="synthetic-helper",
            device_id="helper",
            endpoint="http://127.0.0.1:12345",
            backend="synthetic",
            resource_ids=transition.resource_ids,
        )
        executor = replace(binding(cold), participants=(participant,))
        manager = RuntimeDecodeCohortManager()
        first = manager.admit(
            "first", cold, executor, quality_requirement="exact",
            observed_at_us=0, service_upper_us=100, now_ns=0,
        )
        self.assertIsNotNone(first)
        leases = tuple(
            LeaseRecord(
                "shared-" + resource, first.binding.cohort_id,
                "execution", resource, (0,), 0, 100, 1000,
            )
            for resource in cold.resource_ids
        )
        manager.bind_leases(first.binding.cohort_id, leases)
        self.assertTrue(manager.can_join(
            cold, executor, quality_requirement="exact", now_ns=1,
        ))
        variants = (
            replace(cold, transitions=(replace(transition, latency_us=2),)),
            hot,
            replace(cold, adapter_parameters={
                **dict(cold.adapter_parameters), "request_nonce": 7,
            }),
        )
        for value in variants:
            with self.subTest(plan_sha256=value.plan_sha256):
                self.assertFalse(manager.can_join(
                    value,
                    replace(executor, operator_plan_sha256=value.plan_sha256),
                    quality_requirement="exact", now_ns=1,
                ))
        self.assertFalse(manager.can_join(
            cold,
            replace(executor, participants=(
                replace(participant, endpoint="http://127.0.0.1:12346"),
            )),
            quality_requirement="exact", now_ns=1,
        ))
        second = manager.admit(
            "second", cold, executor, quality_requirement="exact",
            observed_at_us=1, service_upper_us=100, now_ns=1,
        )
        self.assertEqual(second.shared_leases, leases)
        self.assertEqual(second.binding.cohort_id, first.binding.cohort_id)

    def test_static_cohort_applies_one_control_to_all_slots(self):
        members = tuple(f"request-{index}" for index in range(4))

        class StaticCohortClient(CohortClient):
            def __init__(self):
                super().__init__(members)
                self.controls = []

            def apply_ffn_cohort_control(
                self, _endpoint, control, rows, **_kwargs
            ):
                self.controls.append((control, tuple(rows)))
                cohort_members = [
                    {
                        "applied_token_index": self._tokens[request_id],
                        "plan_generation": control.plan_generation,
                        "request_id": request_id,
                        "slot_id": slot_id,
                    }
                    for request_id, slot_id in rows
                ]
                return ({
                    "applied_token_index": min(
                        row["applied_token_index"]
                        for row in cohort_members
                    ),
                    "cohort_members": cohort_members,
                    "plan_generation": control.plan_generation,
                    "policy_hash": control.policy.policy_hash,
                    "runtime_stats": self._stats(),
                    "slot_id": cohort_members[0]["slot_id"],
                    "success": True,
                }, time.monotonic_ns())

        def static_command(request_id: str, index: int):
            value = command(request_id, members, ("lease",))
            operator_plan_sha256 = (
                "sha256:" + str(index + 1) * 64
            )
            execution_contract = RuntimeExecutionContract(
                execution_mode="static-split",
                initial_split_fraction_ppm=500_000,
                allowed_adaptive_fractions_ppm=(),
                batch_plan="split-row",
                maximum_batch_size=4,
                queue_depth=4,
                phone_device_id="helper",
                phone_endpoint="http://127.0.0.1:12345",
                operator_kind="ffn",
            )
            return replace(
                value,
                operator_plan_sha256=operator_plan_sha256,
                operator_plan={
                    **dict(value.operator_plan),
                    "assisted_operator_kind": "ffn",
                    "baseline_executor_id": "desktop-parent",
                    "desktop_placement_sha256": PLACEMENT,
                    "resource_ids": list(plan().resource_ids),
                    "split_fraction_ppm": 500_000,
                    "execution_contract": execution_contract.to_json(),
                    "plan_sha256": operator_plan_sha256,
                },
                adapter_parameters={
                    **dict(value.adapter_parameters),
                    "ffn_assistance_phase": "decode",
                    "ffn_max_tokens": 4,
                    "ffn_resident_columns": 256,
                    "ffn_runtime_control_protocol": "decode-boundary-v1",
                    "ffn_selected_columns": 128,
                    "ffn_selected_layer_mask": 1,
                },
                execution_contract=execution_contract,
                selection_mode="calibration",
            )

        client = StaticCohortClient()
        acknowledgements = {}
        acknowledgement_lock = threading.Lock()

        def success(value, payload):
            acknowledgement = payload["static_ffn_control_ack"]
            with acknowledgement_lock:
                acknowledgements[value.request_id] = acknowledgement
            return {"static_ffn_control_ack": acknowledgement}

        backend = CanonicalHttpExecutionBackend(
            client,
            Meter(),
            epoch_ns=time.monotonic_ns() - 1_000_000,
            on_execution_success=success,
        )
        scheduler = SchedulerSink()
        backend.bind_scheduler(scheduler)
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)

            def execute(index: int):
                request_id = members[index]
                return backend.execute(
                    static_command(request_id, index),
                    LlamaCppCompletionPayload(
                        request_id=request_id,
                        expected_model_alias="synthetic-model",
                        input_tokens=1,
                        output_tokens=4,
                        prompt_tokens=(1,),
                        seed=1,
                        stream_path=root_path / (request_id + ".raw"),
                        on_first_token=lambda _value: None,
                    ),
                    lambda: None,
                )

            with ThreadPoolExecutor(max_workers=4) as pool:
                observations = tuple(pool.map(execute, range(4)))

        self.assertEqual(len(observations), 4)
        self.assertEqual(len(client.controls), 1)
        self.assertEqual(
            {request_id for request_id, _slot in client.controls[0][1]},
            set(members),
        )
        self.assertEqual(set(acknowledgements), set(members))
        self.assertEqual(
            len({
                value["policy_hash"]
                for value in acknowledgements.values()
            }),
            1,
        )

    def test_shared_cold_transition_executes_once_for_four_members(self):
        members = tuple(f"request-{index}" for index in range(4))
        commands = tuple(
            transition_command(request_id, members)
            for request_id in members
        )
        calls = []
        call_lock = threading.Lock()

        def handler(command, _payload, _control_check):
            with call_lock:
                calls.append(command.request_id)
            time.sleep(0.05)

        registry = CanonicalTransitionRegistry({
            "load:synthetic-helper": handler,
        })
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = tuple(pool.map(
                lambda row: registry.execute(row, object(), lambda: None),
                commands,
            ))

        self.assertEqual(results, (True, True, True, True))
        self.assertEqual(len(calls), 1)
        self.assertEqual(registry._shared, {})

    def test_shared_cold_transition_failure_reaches_every_member(self):
        members = ("request-a", "request-b")
        commands = tuple(
            transition_command(request_id, members)
            for request_id in members
        )

        def handler(_command, _payload, _control_check):
            time.sleep(0.05)
            raise RuntimeError("synthetic transition failure")

        registry = CanonicalTransitionRegistry({
            "load:synthetic-helper": handler,
        })

        def execute(row):
            with self.assertRaisesRegex(
                Exception, "physical shared transition failed"
            ):
                registry.execute(row, object(), lambda: None)

        with ThreadPoolExecutor(max_workers=2) as pool:
            tuple(pool.map(execute, commands))
        self.assertEqual(registry._shared, {})

    def test_shared_transition_control_failure_retires_every_member(self):
        members = ("request-a", "request-b")
        commands = tuple(
            transition_command(request_id, members)
            for request_id in members
        )
        entered = threading.Event()
        release = threading.Event()

        def handler(_command, _payload, _control_check):
            entered.set()
            release.wait(1)

        registry = CanonicalTransitionRegistry({
            "load:synthetic-helper": handler,
        })
        checks = {"request-a": 0, "request-b": 0}

        def execute(row):
            def control_check():
                checks[row.request_id] += 1
                if (
                    row.request_id == "request-b"
                    and checks[row.request_id] >= 2
                ):
                    raise RuntimeError("synthetic lease failure")

            try:
                registry.execute(row, object(), control_check)
            except Exception:
                return
            self.fail("shared transition unexpectedly completed")

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(execute, commands[0])
            self.assertTrue(entered.wait(1))
            second = pool.submit(execute, commands[1])
            second.result(timeout=1)
            release.set()
            first.result(timeout=1)
        self.assertEqual(registry._shared, {})

    def test_singleton_is_dissolved_and_rejected_by_cohort_endpoint(self):
        value = replace(
            plan(),
            adapter_parameters={
                **dict(plan().adapter_parameters),
                "decode_cohort_formation_us": 0,
            },
        )
        manager = RuntimeDecodeCohortManager()
        admitted = manager.admit(
            "request-single",
            value,
            binding(value),
            quality_requirement="exact",
            observed_at_us=100,
            service_upper_us=500,
            now_ns=1_000,
        )
        leases = tuple(
            LeaseRecord(
                token="single-" + resource,
                owner_id=admitted.binding.cohort_id,
                lease_id="execution",
                resource_id=resource,
                lanes=(0,),
                start_us=100,
                predicted_end_us=500,
                reserved_until_us=600,
            )
            for resource in value.resource_ids
        )
        manager.bind_leases(admitted.binding.cohort_id, leases)
        sealed = manager.wait_until_sealed("request-single")
        self.assertEqual(sealed.active_batch, 1)
        request_leases = manager.dissolve_singleton("request-single")
        self.assertTrue(all(
            row.owner_id == "request-single" for row in request_leases
        ))
        self.assertIsNone(manager.binding("request-single"))

        singleton = command(
            "request-single", ("request-single",), ("lease",)
        )
        with self.assertRaisesRegex(
            Exception, "physical decode cohort identity differs"
        ):
            DecodeCohortExecutionTracker(Meter()).begin(
                singleton, input_tokens=1, output_tokens=1
            )
        normal = replace(singleton, decode_cohort=None)
        scheduler = SchedulerSink()
        backend = CanonicalHttpExecutionBackend(
            CohortClient(("request-single",)),
            Meter(),
            epoch_ns=time.monotonic_ns() - 1_000_000,
        )
        backend.bind_scheduler(scheduler)
        with tempfile.TemporaryDirectory() as root:
            observation = backend.execute(
                normal,
                LlamaCppCompletionPayload(
                    request_id="request-single",
                    expected_model_alias="synthetic-model",
                    input_tokens=1,
                    output_tokens=4,
                    prompt_tokens=(1,),
                    seed=1,
                    stream_path=Path(root) / "request-single.raw",
                    on_first_token=lambda _value: None,
                ),
                lambda: None,
            )
        self.assertIsNotNone(observation.energy)
        self.assertEqual(scheduler.starts[0][0], "request-single")
        self.assertEqual(scheduler.completions, ["request-single"])

    def test_compatibility_ignores_request_plan_hash(self):
        value = plan()
        changed = replace(
            value,
            adapter_parameters={
                **dict(value.adapter_parameters),
                "request_nonce": 7,
            },
        )
        self.assertNotEqual(value.plan_sha256, changed.plan_sha256)
        first = RuntimeDecodeCohortManager.candidate_key(
            value, binding(value), quality_requirement="exact"
        )
        second = RuntimeDecodeCohortManager.candidate_key(
            changed,
            replace(
                binding(value),
                operator_plan_sha256=changed.plan_sha256,
            ),
            quality_requirement="exact",
        )
        self.assertEqual(first, second)

    def test_manager_shares_one_physical_lease_until_last_terminal(self):
        value = plan()
        executor = binding(value)
        manager = RuntimeDecodeCohortManager()
        first = manager.admit(
            "request-0",
            value,
            executor,
            quality_requirement="exact",
            observed_at_us=100,
            service_upper_us=500,
            now_ns=1_000,
        )
        self.assertTrue(first.leader)
        leases = tuple(
            LeaseRecord(
                token="shared-" + resource,
                owner_id=first.binding.cohort_id,
                lease_id="execution",
                resource_id=resource,
                lanes=(0,),
                start_us=100,
                predicted_end_us=500,
                reserved_until_us=10_000,
            )
            for resource in value.resource_ids
        )
        manager.bind_leases(first.binding.cohort_id, leases)
        for index in range(1, 4):
            admitted = manager.admit(
                f"request-{index}",
                value,
                executor,
                quality_requirement="exact",
                observed_at_us=100 + index,
                service_upper_us=500,
                now_ns=1_000 + index,
            )
            self.assertFalse(admitted.leader)
            self.assertEqual(admitted.shared_leases, leases)
        sealed = manager.wait_until_sealed("request-0")
        self.assertEqual(sealed.active_batch, 4)
        self.assertEqual(sealed.shared_lease_tokens, tuple(
            row.token for row in leases
        ))
        for index in range(3):
            self.assertEqual(
                manager.mark_terminal(f"request-{index}"),
                (False, first.binding.cohort_id),
            )
        self.assertEqual(
            manager.mark_terminal("request-3"),
            (True, first.binding.cohort_id),
        )

    def test_forming_cohort_reports_only_compatible_joinable_plan(self):
        value = plan()
        executor = binding(value)
        manager = RuntimeDecodeCohortManager()
        first = manager.admit(
            "request-0",
            value,
            executor,
            quality_requirement="semantic",
            observed_at_us=100,
            service_upper_us=500,
            now_ns=1_000,
        )
        leases = tuple(
            LeaseRecord(
                token="joinable-" + resource,
                owner_id=first.binding.cohort_id,
                lease_id="execution",
                resource_id=resource,
                lanes=(0,),
                start_us=100,
                predicted_end_us=500,
                reserved_until_us=10_000,
            )
            for resource in value.resource_ids
        )
        manager.bind_leases(first.binding.cohort_id, leases)

        self.assertTrue(manager.can_join(
            value,
            executor,
            quality_requirement="semantic",
            now_ns=1_001,
        ))
        self.assertFalse(manager.can_join(
            value,
            executor,
            quality_requirement="exact",
            now_ns=1_001,
        ))
        self.assertFalse(manager.can_join(
            replace(
                value,
                resource_ids=value.resource_ids + ("link:other",),
            ),
            executor,
            quality_requirement="semantic",
            now_ns=1_001,
        ))

    def test_different_output_lengths_share_compatible_cohort(self):
        value = plan()
        executor = binding(value)
        manager = RuntimeDecodeCohortManager()
        first = manager.admit(
            "request-a",
            value,
            executor,
            quality_requirement="exact",
            observed_at_us=100,
            service_upper_us=500,
            now_ns=1_000,
        )
        leases = (LeaseRecord(
            token="shared",
            owner_id=first.binding.cohort_id,
            lease_id="execution",
            resource_id="compute:desktop",
            lanes=(0,),
            start_us=100,
            predicted_end_us=500,
            reserved_until_us=10_000,
        ), LeaseRecord(
            token="shared-link",
            owner_id=first.binding.cohort_id,
            lease_id="execution",
            resource_id="link:helper",
            lanes=(0,),
            start_us=100,
            predicted_end_us=500,
            reserved_until_us=10_000,
        ))
        manager.bind_leases(first.binding.cohort_id, leases)
        other = manager.admit(
            "request-b",
            value,
            executor,
            quality_requirement="exact",
            observed_at_us=101,
            service_upper_us=500,
            now_ns=1_001,
        )
        self.assertFalse(other.leader)
        self.assertEqual(
            other.binding.cohort_id, first.binding.cohort_id
        )

    def test_longest_decode_becomes_policy_leader_at_seal(self):
        value = replace(
            plan(),
            adapter_parameters={
                **dict(plan().adapter_parameters),
                "parallel": 2,
            },
        )
        executor = binding(value)
        manager = RuntimeDecodeCohortManager()
        first = manager.admit(
            "request-short",
            value,
            executor,
            quality_requirement="semantic",
            observed_at_us=100,
            service_upper_us=500,
            output_tokens=20,
            now_ns=1_000,
        )
        manager.bind_leases(first.binding.cohort_id, (
            LeaseRecord(
                token="long-leader-compute",
                owner_id=first.binding.cohort_id,
                lease_id="execution",
                resource_id="compute:desktop",
                lanes=(0,),
                start_us=100,
                predicted_end_us=500,
                reserved_until_us=10_000,
            ),
            LeaseRecord(
                token="long-leader-link",
                owner_id=first.binding.cohort_id,
                lease_id="execution",
                resource_id="link:helper",
                lanes=(0,),
                start_us=100,
                predicted_end_us=500,
                reserved_until_us=10_000,
            ),
        ))
        manager.admit(
            "request-long",
            value,
            executor,
            quality_requirement="semantic",
            observed_at_us=101,
            service_upper_us=500,
            output_tokens=100,
            now_ns=1_001,
        )
        sealed = manager.wait_until_sealed("request-short")
        self.assertEqual(
            sealed.member_request_ids,
            ("request-long", "request-short"),
        )
        self.assertEqual(sealed.leader_request_id, "request-long")

    def test_policy_coordinator_forwards_each_membership_shrink(self):
        members = tuple(f"request-{index}" for index in range(4))
        commands = tuple(command(value, members, ("lease",)) for value in members)
        coordinator = DecodeCohortPolicyCoordinator()
        events = []
        event_lock = threading.Lock()

        def factory(_command, _payload, _started_ns, _view):
            def progress(slot, token, _at_ns, terminal):
                with event_lock:
                    events.append((slot, token, terminal))

            return progress, lambda _value: None

        with ThreadPoolExecutor(max_workers=4) as pool:
            callbacks = tuple(pool.map(
                lambda row: coordinator.register(
                    row, object(), time.monotonic_ns(), factory
                ),
                commands,
            ))
        for index, (progress, _) in enumerate(callbacks):
            progress(index, 1, time.monotonic_ns(), False)
        for index, (progress, _) in enumerate(callbacks):
            progress(index, 2, time.monotonic_ns(), False)
        for index, (progress, _) in enumerate(callbacks):
            progress(index, 3, time.monotonic_ns(), True)
        self.assertEqual(
            [(token, terminal) for _, token, terminal in events],
            [
                (1, False),
                (2, False),
                (2, True),
                (2, True),
                (2, True),
                (3, True),
            ],
        )
        with ThreadPoolExecutor(max_workers=4) as pool:
            leaders = tuple(pool.map(coordinator.complete_http, commands))
        self.assertEqual(leaders.count(members[0]), 1)
        self.assertEqual(leaders.count(None), 3)
        for row in commands:
            coordinator.release(row)

    def test_policy_continues_after_member_retires(self):
        members = ("request-short", "request-long")
        commands = tuple(command(value, members, ("lease",)) for value in members)
        coordinator = DecodeCohortPolicyCoordinator()
        events = []
        views = []

        def factory(_command, _payload, _started_ns, view):
            views.append(view)
            return (
                lambda _slot, token, _at_ns, terminal: events.append(
                    (token, terminal)
                ),
                lambda _value: None,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            callbacks = tuple(pool.map(
                lambda row: coordinator.register(
                    row, object(), time.monotonic_ns(), factory
                ),
                commands,
            ))
        callbacks[0][0](0, 1, time.monotonic_ns(), False)
        callbacks[1][0](1, 1, time.monotonic_ns(), False)
        callbacks[0][0](0, 2, time.monotonic_ns(), True)
        callbacks[1][0](1, 2, time.monotonic_ns(), False)
        self.assertEqual(views[0].active_batch, 1)
        self.assertFalse(views[0].membership_changed)
        callbacks[1][0](1, 3, time.monotonic_ns(), False)
        callbacks[1][0](1, 4, time.monotonic_ns(), True)

        self.assertEqual(
            events,
            [(1, False), (1, True), (2, False), (3, False), (4, True)],
        )
        self.assertEqual(views[0].active_batch, 0)
        self.assertFalse(views[0].membership_changed)
        self.assertIsNone(coordinator.complete_http(commands[0]))
        self.assertEqual(
            coordinator.complete_http(commands[1]), members[0]
        )
        for row in commands:
            coordinator.release(row)

    def test_policy_does_not_retire_member_before_terminal_progress(self):
        members = ("request-short", "request-long")
        commands = tuple(command(value, members, ("lease",)) for value in members)
        coordinator = DecodeCohortPolicyCoordinator()
        events = []
        views = []
        payloads = {
            "request-short": SimpleNamespace(output_tokens=20),
            "request-long": SimpleNamespace(output_tokens=100),
        }

        def factory(_command, _payload, _started_ns, view):
            views.append(view)
            return (
                lambda _slot, token, _at_ns, terminal: events.append(
                    (token, terminal)
                ),
                lambda _value: None,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            callbacks = tuple(pool.map(
                lambda row: coordinator.register(
                    row,
                    payloads[row.request_id],
                    time.monotonic_ns(),
                    factory,
                ),
                commands,
            ))
        callbacks[0][0](0, 1, 1_000_000_000, False)
        callbacks[1][0](1, 1, 1_000_000_000, False)
        callbacks[0][0](0, 2, 1_500_000_000, False)
        callbacks[1][0](1, 2, 1_500_000_000, False)
        callbacks[0][0](0, 10, 5_500_000_000, False)
        callbacks[1][0](1, 3, 2_000_000_000, False)

        self.assertEqual(events, [(1, False), (2, False), (3, False)])
        self.assertFalse(views[0].membership_changed)
        self.assertEqual(views[0].active_batch, 2)
        callbacks[0][0](0, 11, 6_000_000_000, False)
        callbacks[1][0](1, 4, 2_500_000_000, False)
        self.assertEqual(
            events,
            [(1, False), (2, False), (3, False), (4, False)],
        )
        for row in commands:
            coordinator.release(row)

    def test_buffered_progress_does_not_retire_member_early(self):
        members = ("request-short", "request-long")
        commands = tuple(
            replace(
                command(value, members, ("lease",)),
                planned_finish_us=(
                    342_000_001 if value == members[0]
                    else 781_000_001
                ),
                planned_finish_upper_us=(
                    513_000_001 if value == members[0]
                    else 1_171_500_001
                ),
            )
            for value in members
        )
        coordinator = DecodeCohortPolicyCoordinator()
        events = []
        views = []
        payloads = {
            "request-short": SimpleNamespace(output_tokens=342),
            "request-long": SimpleNamespace(output_tokens=781),
        }

        def factory(_command, _payload, _started_ns, view):
            views.append(view)
            return (
                lambda _slot, token, _at_ns, terminal: events.append(
                    (token, terminal)
                ),
                lambda _value: None,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            callbacks = tuple(pool.map(
                lambda row: coordinator.register(
                    row,
                    payloads[row.request_id],
                    time.monotonic_ns(),
                    factory,
                ),
                commands,
            ))
        callbacks[0][0](0, 1, 1_000_000_000, False)
        callbacks[1][0](1, 1, 1_000_000_000, False)
        callbacks[0][0](0, 15, 1_014_000_000, False)
        callbacks[1][0](1, 15, 1_014_000_000, False)

        self.assertEqual(events, [(1, False), (15, False)])
        self.assertFalse(views[0].membership_changed)
        callbacks[0][0](0, 342, 2_000_000_000, True)
        callbacks[1][0](1, 16, 1_015_000_000, False)
        self.assertEqual(events[-1], (16, False))
        self.assertFalse(views[0].membership_changed)
        self.assertEqual(views[0].active_batch, 1)
        for row in commands:
            coordinator.release(row)

    def test_stale_membership_frontier_does_not_stop_adaptation(self):
        members = ("request-short", "request-long")
        commands = tuple(command(value, members, ("lease",)) for value in members)
        coordinator = DecodeCohortPolicyCoordinator()
        events = []
        payloads = {
            "request-short": SimpleNamespace(output_tokens=20),
            "request-long": SimpleNamespace(output_tokens=100),
        }

        def factory(_command, _payload, _started_ns, _view):
            def progress(_slot, token, _at_ns, terminal):
                events.append((token, terminal))
                return not (terminal and len(events) == 3)

            return progress, lambda _value: None

        with ThreadPoolExecutor(max_workers=2) as pool:
            callbacks = tuple(pool.map(
                lambda row: coordinator.register(
                    row,
                    payloads[row.request_id],
                    time.monotonic_ns(),
                    factory,
                ),
                commands,
            ))
        callbacks[0][0](0, 1, 1_000_000_000, False)
        callbacks[1][0](1, 1, 1_000_000_000, False)
        callbacks[0][0](0, 2, 1_500_000_000, False)
        callbacks[1][0](1, 2, 1_500_000_000, False)
        callbacks[0][0](0, 10, 5_500_000_000, False)
        callbacks[1][0](1, 3, 2_000_000_000, False)
        callbacks[1][0](1, 4, 2_500_000_000, False)
        callbacks[1][0](1, 5, 3_000_000_000, False)

        self.assertEqual(
            events,
            [
                (1, False),
                (2, False),
                (3, False),
                (4, False),
                (5, False),
            ],
        )
        for row in commands:
            coordinator.release(row)

    def test_execution_energy_is_measured_once_and_owned_by_leader(self):
        members = tuple(f"request-{index}" for index in range(4))
        commands = tuple(command(value, members, ("lease",)) for value in members)
        meter = Meter()
        tracker = DecodeCohortExecutionTracker(meter)

        def run(row):
            tracker.begin(row, input_tokens=8, output_tokens=16)
            return tracker.finish(row, time.monotonic_ns())

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = tuple(pool.map(run, commands))
        self.assertEqual(meter.prepare_count, 1)
        self.assertEqual(meter.measure_count, 1)
        measured = tuple(row for row in results if row.energy is not None)
        self.assertEqual(len(measured), 1)
        self.assertEqual(measured[0].total_input_tokens, 32)
        self.assertEqual(measured[0].total_output_tokens, 64)

    def test_unequal_members_retire_without_completion_wait(self):
        members = ("request-short", "request-long")
        commands = tuple(command(value, members, ("lease",)) for value in members)
        meter = Meter()
        tracker = DecodeCohortExecutionTracker(meter)
        with ThreadPoolExecutor(max_workers=2) as pool:
            tuple(pool.map(
                lambda pair: tracker.begin(
                    pair[0],
                    input_tokens=8,
                    output_tokens=pair[1],
                ),
                zip(commands, (8, 32)),
            ))

        started = time.monotonic()
        first = tracker.finish(commands[0], time.monotonic_ns())
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertIsNone(first.energy)
        self.assertEqual(meter.measure_count, 0)
        second = tracker.finish(commands[1], time.monotonic_ns())
        self.assertIsNotNone(second.energy)
        self.assertEqual(second.total_input_tokens, 16)
        self.assertEqual(second.total_output_tokens, 40)
        self.assertEqual(meter.measure_count, 1)

    def test_member_failure_aborts_cohort_without_stale_state(self):
        members = ("request-a", "request-b")
        commands = tuple(command(value, members, ("lease",)) for value in members)
        tracker = DecodeCohortExecutionTracker(Meter())
        with ThreadPoolExecutor(max_workers=2) as pool:
            tuple(pool.map(
                lambda row: tracker.begin(
                    row, input_tokens=8, output_tokens=16
                ),
                commands,
            ))
        error = RuntimeError("synthetic member failure")
        tracker.fail(commands[0], error)
        tracker.abort(commands[0])
        with self.assertRaisesRegex(
            Exception, "physical decode cohort measurement failed"
        ):
            tracker.finish(commands[1], time.monotonic_ns())
        tracker.abort(commands[1])
        self.assertEqual(tracker._states, {})

    def test_cohort_control_response_requires_common_frontier(self):
        members = (("request-0", 2), ("request-1", 5))
        value = {
            "applied_token_index": 7,
            "cohort_members": [
                {
                    "applied_token_index": 7,
                    "plan_generation": 2,
                    "request_id": "request-0",
                    "slot_id": 2,
                },
                {
                    "applied_token_index": 8,
                    "plan_generation": 2,
                    "request_id": "request-1",
                    "slot_id": 5,
                },
            ],
            "plan_generation": 2,
            "policy_hash": "sha256:" + "1" * 64,
            "runtime_stats": {
                "calls": 0,
                "desktop_compute_us": 0,
                "download_bytes": 0,
                "exposed_tail_us": 0,
                "phone_compute_us": 0,
                "rpc_us": 0,
                "upload_bytes": 0,
                "usb_transfer_us": 0,
                "useful_overlap_us": 0,
            },
            "slot_id": 2,
            "success": True,
        }
        parsed = LlamaCppHttpClient._parse_cohort_response(
            value,
            members,
            policy_hash="sha256:" + "1" * 64,
            plan_generation=2,
        )
        self.assertEqual(parsed["applied_token_index"], 7)
        with self.assertRaisesRegex(
            Exception, "acknowledgement identity differs"
        ):
            LlamaCppHttpClient._parse_cohort_response(
                {**value, "applied_token_index": 8},
                members,
                policy_hash="sha256:" + "1" * 64,
                plan_generation=2,
            )

    def test_http_backend_uses_one_policy_and_one_energy_receipt(self):
        members = tuple(f"request-{index}" for index in range(4))
        commands = tuple(command(value, members, ("lease",)) for value in members)
        meter = Meter()
        scheduler = SchedulerSink()
        backend = CanonicalHttpExecutionBackend(
            CohortClient(members),
            meter,
            epoch_ns=time.monotonic_ns() - 1_000_000,
        )
        backend.bind_scheduler(scheduler)
        with tempfile.TemporaryDirectory() as root:
            payloads = tuple(
                LlamaCppCompletionPayload(
                    request_id=request_id,
                    expected_model_alias="synthetic-model",
                    input_tokens=1,
                    output_tokens=4,
                    prompt_tokens=(1,),
                    seed=1,
                    stream_path=Path(root) / (request_id + ".raw"),
                    on_first_token=lambda _value: None,
                )
                for request_id in members
            )
            with ThreadPoolExecutor(max_workers=4) as pool:
                observations = tuple(pool.map(
                    lambda pair: backend.execute(
                        pair[0], pair[1], lambda: None
                    ),
                    zip(commands, payloads),
                ))
        self.assertEqual([row[0] for row in scheduler.starts], [members[0]])
        self.assertEqual(scheduler.previews, [members[0]])
        self.assertEqual(scheduler.completions, [members[0]])
        self.assertEqual(len(scheduler.cohort_receipts), 1)
        self.assertEqual(meter.prepare_count, 1)
        self.assertEqual(meter.measure_count, 1)
        self.assertEqual(
            sum(row.energy is not None for row in observations), 1
        )
        values = scheduler.cohort_receipts[0][1]
        self.assertEqual(values["energy_boundary_id"], "synthetic-fleet")
        self.assertEqual(values["attribution_kind"], "isolated")
        self.assertEqual(values["total_input_tokens"], 4)
        self.assertEqual(values["total_output_tokens"], 16)

    def test_physical_stats_follow_live_members_at_singleton_handoff(self):
        members = ("request-long", "request-short")

        class ShrinkingClient(CohortClient):
            def __init__(self):
                super().__init__(members)
                self.active = set(members)
                self.long_at_two = threading.Event()
                self.short_finished = threading.Event()
                self.cohort_stats_members = []
                self.singleton_stats_members = []

            def read_ffn_cohort_stats(self, endpoint, rows, **kwargs):
                with self._lock:
                    if any(request_id not in self.active for request_id, _ in rows):
                        raise RuntimeError(
                            "synthetic cohort includes a released slot"
                        )
                    self.cohort_stats_members.append(tuple(rows))
                return super().read_ffn_cohort_stats(endpoint, rows, **kwargs)

            def read_ffn_stats(
                self, _endpoint, request_id, slot_id, **_kwargs
            ):
                with self._lock:
                    if request_id not in self.active:
                        raise RuntimeError("synthetic singleton slot is released")
                    self.singleton_stats_members.append((request_id, slot_id))
                    token = self._tokens[request_id] + 1
                return ({
                    "applied_token_index": token,
                    "plan_generation": 0,
                    "policy_hash": "",
                    "runtime_stats": self._stats(),
                    "slot_id": slot_id,
                    "success": True,
                }, time.monotonic_ns())

            def complete(
                self,
                _endpoint,
                payload,
                control_check,
                *,
                scheduler_headers=None,
            ):
                control_check()
                request_id = payload.request_id
                slot_id = self._slots[request_id]
                if payload.on_active_batch is not None:
                    payload.on_active_batch(2)
                with self._lock:
                    self._tokens[request_id] = 1
                payload.on_decode_progress(
                    slot_id, 1, time.monotonic_ns(), False
                )
                self._token_barrier.wait()
                if request_id == "request-long":
                    with self._lock:
                        self._tokens[request_id] = 2
                    payload.on_decode_progress(
                        slot_id, 2, time.monotonic_ns(), False
                    )
                    self.long_at_two.set()
                    self.short_finished.wait(1)
                    for token in range(3, 11):
                        terminal = token == 10
                        with self._lock:
                            self._tokens[request_id] = token
                            if terminal:
                                self.active.remove(request_id)
                        payload.on_decode_progress(
                            slot_id, token, time.monotonic_ns(), terminal
                        )
                else:
                    self.long_at_two.wait(1)
                    with self._lock:
                        self._tokens[request_id] = 2
                        self.active.remove(request_id)
                    payload.on_decode_progress(
                        slot_id, 2, time.monotonic_ns(), True
                    )
                    self.short_finished.set()
                return {
                    "stream_sha256": "3" * 64,
                    "tokens": list(range(
                        1, 11 if request_id == "request-long" else 3
                    )),
                }

        class ShrinkScheduler(SchedulerSink):
            def __init__(self):
                super().__init__()
                self.windows = []
                self.observations = []
                self.policy = AdaptiveDecodePolicy(
                    route_id="desktop-parent",
                    executor_id="desktop-parent",
                    operator_plan_sha256="sha256:" + "d" * 64,
                    desktop_parent_route_id="desktop-parent",
                    desktop_placement_sha256=PLACEMENT,
                    layer_indices=(),
                    layer_mask=0,
                    columns=0,
                    split_fraction_ppm=0,
                    resource_ids=("compute:desktop",),
                    baseline=True,
                    predicted_latency_per_token_us=1,
                    predicted_energy_per_token_uj=1,
                )

            def adaptive_decode_boundary(self, request_id, **values):
                self.boundaries.append((request_id, values))
                if (
                    (not values["terminal"] and not self.windows)
                    or len(self.windows) >= 2
                ):
                    return None
                token_start = 1 if not self.windows else 2
                window = AdaptiveDecodeWindowBoundary(
                    request_id=request_id,
                    slot_id=values["slot_id"],
                    window_index=len(self.windows),
                    token_start=token_start,
                    token_end=values["token_index"],
                    started_at_us=1,
                    finished_at_us=values["at_us"],
                    policy=self.policy,
                    applied_ack=None,
                )
                self.windows.append(window)
                return AdaptiveDecodeDirective(
                    state="BASELINE",
                    reason="WINDOW_MEASUREMENT_REQUIRED",
                    target_token_index=None,
                    boundary=window,
                )

            def record_adaptive_decode_window(
                self, _request_id, _boundary, observation
            ):
                self.observations.append(observation)
                return None

        commands = tuple(command(value, members, ("lease",)) for value in members)
        client = ShrinkingClient()
        scheduler = ShrinkScheduler()
        backend = CanonicalHttpExecutionBackend(
            client,
            Meter(),
            epoch_ns=time.monotonic_ns() - 1_000_000,
        )
        backend.bind_scheduler(scheduler)
        with tempfile.TemporaryDirectory() as root:
            payloads = (
                LlamaCppCompletionPayload(
                    request_id=members[0],
                    expected_model_alias="synthetic-model",
                    input_tokens=1,
                    output_tokens=10,
                    prompt_tokens=(1,),
                    seed=1,
                    stream_path=Path(root) / "long.raw",
                    on_first_token=lambda _value: None,
                ),
                LlamaCppCompletionPayload(
                    request_id=members[1],
                    expected_model_alias="synthetic-model",
                    input_tokens=1,
                    output_tokens=2,
                    prompt_tokens=(1,),
                    seed=1,
                    stream_path=Path(root) / "short.raw",
                    on_first_token=lambda _value: None,
                ),
            )
            with ThreadPoolExecutor(max_workers=2) as pool:
                tuple(pool.map(
                    lambda pair: backend.execute(
                        pair[0], pair[1], lambda: None
                    ),
                    zip(commands, payloads),
                ))

        self.assertEqual(client.cohort_stats_members, [])
        self.assertEqual(
            client.singleton_stats_members,
            [
                ("request-long", 0),
                ("request-long", 0),
                ("request-long", 0),
            ],
        )
        self.assertEqual(scheduler.observations[0].active_batch, 2)
        self.assertEqual(scheduler.observations[0].next_active_batch, 1)
        self.assertEqual(
            scheduler.observations[0].cohort_member_request_ids,
            members,
        )
        self.assertEqual(scheduler.observations[1].active_batch, 1)
        self.assertEqual(scheduler.observations[1].accounting_token_count, 1)

    def test_stats_lookahead_does_not_advance_stream_cursor(self):
        members = ("request-a", "request-b")

        class AheadClient(CohortClient):
            def read_ffn_stats(
                self, endpoint, request_id, slot_id, **kwargs
            ):
                value, observed_ns = super().read_ffn_stats(
                    endpoint, request_id, slot_id, **kwargs
                )
                return ({
                    **value,
                    "applied_token_index":
                        value["applied_token_index"] + 2,
                }, observed_ns)

        class StrictScheduler(SchedulerSink):
            def __init__(self):
                super().__init__()
                self.initial_token = None

            def start_adaptive_decode(self, request_id, **values):
                self.initial_token = values["first_token_index"]
                return super().start_adaptive_decode(
                    request_id, **values
                )

            def adaptive_decode_boundary(self, request_id, **values):
                if values["token_index"] <= self.initial_token:
                    raise RuntimeError("synthetic progress moved backward")
                return super().adaptive_decode_boundary(
                    request_id, **values
                )

        commands = tuple(command(value, members, ("lease",)) for value in members)
        scheduler = StrictScheduler()
        backend = CanonicalHttpExecutionBackend(
            AheadClient(members),
            Meter(),
            epoch_ns=time.monotonic_ns() - 1_000_000,
        )
        backend.bind_scheduler(scheduler)
        with tempfile.TemporaryDirectory() as root:
            payloads = tuple(
                LlamaCppCompletionPayload(
                    request_id=request_id,
                    expected_model_alias="synthetic-model",
                    input_tokens=1,
                    output_tokens=4,
                    prompt_tokens=(1,),
                    seed=1,
                    stream_path=Path(root) / (request_id + ".raw"),
                    on_first_token=lambda _value: None,
                )
                for request_id in members
            )
            with ThreadPoolExecutor(max_workers=2) as pool:
                tuple(pool.map(
                    lambda pair: backend.execute(
                        pair[0], pair[1], lambda: None
                    ),
                    zip(commands, payloads),
                ))
        self.assertEqual(scheduler.initial_token, 1)
        self.assertTrue(all(
            values["token_index"] > scheduler.initial_token
            for _, values in scheduler.boundaries
        ))

    def test_diagnostic_cohort_energy_retains_boundary_without_qualification(self):
        members = ("request-a", "request-b")
        commands = tuple(command(value, members, ("lease",)) for value in members)
        scheduler = SchedulerSink()
        backend = CanonicalHttpExecutionBackend(
            CohortClient(members),
            Meter("diagnostic"),
            epoch_ns=time.monotonic_ns() - 1_000_000,
        )
        backend.bind_scheduler(scheduler)
        with tempfile.TemporaryDirectory() as root:
            payloads = tuple(
                LlamaCppCompletionPayload(
                    request_id=request_id,
                    expected_model_alias="synthetic-model",
                    input_tokens=1,
                    output_tokens=4,
                    prompt_tokens=(1,),
                    seed=1,
                    stream_path=Path(root) / (request_id + ".raw"),
                    on_first_token=lambda _value: None,
                )
                for request_id in members
            )
            with ThreadPoolExecutor(max_workers=2) as pool:
                tuple(pool.map(
                    lambda pair: backend.execute(
                        pair[0], pair[1], lambda: None
                    ),
                    zip(commands, payloads),
                ))
        self.assertEqual(len(scheduler.cohort_receipts), 1)
        values = scheduler.cohort_receipts[0][1]
        self.assertEqual(values["attribution_kind"], "diagnostic")
        self.assertEqual(values["energy_boundary_id"], "synthetic-fleet")


if __name__ == "__main__":
    unittest.main()
