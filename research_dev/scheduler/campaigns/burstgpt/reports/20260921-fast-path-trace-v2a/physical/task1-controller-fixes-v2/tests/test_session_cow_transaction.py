#!/usr/bin/env python3
"""Copy-on-write phone session replacement: scoping, rollback, and proofs."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
import hashlib
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from research_dev.scheduler import UnifiedScheduler
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeConfig,
    AdaptiveDecodePolicyAck,
)
from research_dev.scheduler._internal.runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeExecutorState,
)
from research_dev.scheduler._internal.runtime_plan import (
    PhoneSessionReplacementAuthorization,
    RuntimeHelperExecutionEnvelope,
    RuntimePlanError,
    RuntimeTransitionPlan,
    helper_preparation_changed_session_ids,
    phone_session_map_sha256,
)
from research_dev.scheduler._internal.runtime_plan import RuntimePhoneShard
from research_dev.scheduler.adapters.contracts import (
    PhysicalAdapterError,
    RawTransitionObservation,
)
from research_dev.scheduler.adapters.heterogeneous_rig import (
    HeterogeneousPhysicalRig,
    _HelperReconfiguration,
    _PersistentPhoneResidency,
)
from research_dev.scheduler.adapters.llama_server import (
    LlamaServerPhoneSessionProof,
)
from research_dev.scheduler.adapters.phone_session import (
    DirectPhoneFfnReconfigurationReceipt,
    DirectPhoneFfnSession,
    DirectPhoneFfnSessionProof,
    DirectPhoneFfnTerminalReceipt,
    _ShardResidencyWindow,
)
from research_dev.scheduler.adapters.runtime import (
    CanonicalPhysicalAdapter,
    HelperPreparationFaultInjector,
)
from research_dev.scheduler.adapters.ticket import (
    validate_phone_session_replacement_command,
)
from research_dev.scheduler._unified.common import _StalePhoneSessionAssignment


QWEN = "sha256:" + "a" * 64
GEMMA = "sha256:" + "b" * 64


@dataclass(frozen=True)
class _LaunchState:
    phone_shards: tuple[RuntimePhoneShard, ...]
    shard_manifest_sha256: str
    weight_sources: tuple = ()


class _ResidencyControlConnection:
    def __init__(self, selected_session_id: str, load_count: int) -> None:
        self.selected_session_id = selected_session_id
        self.load_count = load_count
        self.request = b""
        self.response = b""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def settimeout(self, _timeout):
        return None

    def shutdown(self, _how):
        return None

    def sendall(self, request: bytes) -> None:
        self.request = request
        header, payload = request.split(b"\n", 1)
        schema, size, digest, quantum, max_tokens = header.decode(
            "ascii"
        ).split()
        if schema != "S42RESIDENCY_V3" or int(size) != len(payload):
            raise AssertionError("residency request framing differs")
        selected = next(
            row for row in payload.decode("ascii").splitlines()
            if row.split(",", 1)[0] == self.selected_session_id
        )
        selected_fields = selected.split(",")
        if int(selected_fields[5]) % int(quantum):
            self.response = b"S42RESIDENCY_ERROR column_quantum\n"
            return
        session_generation = selected_fields[11]
        self.response = (
            "S42RESIDENCY_READY "
            + digest
            + " "
            + self.selected_session_id
            + " "
            + str(self.load_count)
            + " "
            + quantum
            + " "
            + max_tokens
            + " "
            + session_generation
            + "\n"
        ).encode("ascii")

    def recv(self, _size: int) -> bytes:
        response, self.response = self.response, b""
        return response


def shard(
    session_id: str,
    artifact: str,
    layer_mask: int,
    session_generation: int = 1,
) -> RuntimePhoneShard:
    suffix = session_id[-1].lower()
    return RuntimePhoneShard(
        session_id=session_id,
        endpoint="session://phone/" + session_id,
        layer_mask=layer_mask,
        maximum_columns=128,
        resident_bytes=100,
        resident_geometry_sha256="sha256:" + suffix * 64,
        operator_plan_sha256="sha256:" + (
            "d" if artifact == QWEN else "e"
        ) * 64,
        artifact_sha256=artifact,
        session_generation=session_generation,
    )


def proof(row: RuntimePhoneShard, calls: int = 3) -> LlamaServerPhoneSessionProof:
    return LlamaServerPhoneSessionProof(
        session_id=row.session_id,
        endpoint=row.endpoint,
        artifact_sha256=str(row.artifact_sha256),
        resident_geometry_sha256=row.resident_geometry_sha256,
        operator_plan_sha256=row.operator_plan_sha256,
        session_generation=row.session_generation,
        layer_mask=row.layer_mask,
        calls=calls,
        rows=calls,
        payload_bytes=calls * 16,
    )


def terminal_proof(
    row: RuntimePhoneShard, calls: int = 0
) -> DirectPhoneFfnSessionProof:
    return DirectPhoneFfnSessionProof(
        session_id=row.session_id,
        endpoint_sha256="sha256:" + hashlib.sha256(
            row.endpoint.encode("ascii")
        ).hexdigest(),
        artifact_sha256=row.artifact_sha256,
        resident_geometry_sha256=row.resident_geometry_sha256,
        operator_plan_sha256=row.operator_plan_sha256,
        session_generation=row.session_generation,
        layer_mask=row.layer_mask,
        calls=calls,
        rows=calls,
        h2d_bytes=calls * 16,
        d2h_bytes=calls * 16,
        h2d_us=calls,
        d2h_us=calls,
        compute_us=calls,
        rpc_us=calls,
    )


def session_with(shards: tuple[RuntimePhoneShard, ...]) -> DirectPhoneFfnSession:
    session = object.__new__(DirectPhoneFfnSession)
    session._launch = SimpleNamespace(phone_shards=shards)
    session._remote_root = "/data/local/tmp/resident"
    session._proof_shards = list(shards)
    session._executed_proof_shards = []
    session._bound_ticket_ids = ["layout-ticket"]
    session._residency_generation = 1
    session._shard_residency_windows = [
        _ShardResidencyWindow(shard=row, first_generation=1) for row in shards
    ]
    session._ticket_bind_generations = {"layout-ticket": 1}
    session._historical_execution_proofs = []
    session._column_quantum_by_session = {
        row.session_id: 32 for row in shards
    }
    return session


class ChangedSessionScopeTests(unittest.TestCase):
    def test_retained_helper_gets_no_changed_sessions(self) -> None:
        self.assertEqual(
            helper_preparation_changed_session_ids(
                ("HTP2",), ("HTP0", "HTP1")
            ),
            (),
        )

    def test_new_helper_gets_only_its_own_changed_session(self) -> None:
        self.assertEqual(
            helper_preparation_changed_session_ids(
                ("HTP0", "HTP2"), ("HTP2",)
            ),
            ("HTP2",),
        )
        self.assertEqual(
            helper_preparation_changed_session_ids(
                ("HTP2", "HTP0"), ("HTP1", "HTP0", "HTP2")
            ),
            ("HTP0", "HTP2"),
        )

    def test_scope_rejects_invalid_session_names(self) -> None:
        with self.assertRaises(RuntimePlanError):
            helper_preparation_changed_session_ids(("",), ("HTP0",))

    def test_retained_helper_transition_keeps_empty_changed_scope(self) -> None:
        transition = RuntimeTransitionPlan(
            transition_id="replace-htp0",
            device_id="phone",
            source_state="warm",
            target_state="hot",
            latency_us=10,
            energy_uj=10,
            resource_ids=("phone-htp",),
            maturity="QUALIFIED",
            prepares_device_ids=("phone",),
            phone_shards=(shard("HTP0", GEMMA, 1),),
            changed_phone_session_ids=("HTP0",),
        )
        holder = SimpleNamespace(
            helper_plan=SimpleNamespace(
                execution_contract=SimpleNamespace(
                    phone_device_id="phone"
                ),
                transitions=(transition,),
            ),
            helper_binding=SimpleNamespace(participants=(
                SimpleNamespace(
                    device_id="phone",
                    resource_ids=("phone-htp",),
                ),
            )),
            preparation_changed_session_ids=(),
        )

        scoped = RuntimeHelperExecutionEnvelope\
            .preparation_transitions.fget(holder)

        self.assertEqual(scoped[0].changed_phone_session_ids, ())


class ReplacementAuthorizationTests(unittest.TestCase):
    def test_selected_session_is_validated_without_substitution(self) -> None:
        source = tuple(
            shard("HTP" + str(index), QWEN, 1 << index)
            for index in range(3)
        )
        for selected_index in range(len(source)):
            with self.subTest(selected_index=selected_index):
                selected = source[selected_index].session_id
                replacement = replace(
                    source[selected_index],
                    artifact_sha256=GEMMA,
                    operator_plan_sha256="sha256:" + "e" * 64,
                    session_generation=2,
                )
                target = tuple(
                    replacement if index == selected_index else row
                    for index, row in enumerate(source)
                )
                authorization = PhoneSessionReplacementAuthorization.create(
                    selected_session_id=selected,
                    source_shards=source,
                    target_shards=target,
                )
                self.assertEqual(
                    authorization.source_layout_hash,
                    phone_session_map_sha256(source),
                )
                command = SimpleNamespace(
                    replacement_authorization=authorization,
                    transition=SimpleNamespace(
                        changed_phone_session_ids=(selected,),
                        phone_shards=target,
                    ),
                )
                session = object.__new__(DirectPhoneFfnSession)
                session._launch = SimpleNamespace(phone_shards=source)
                session._remote_root = "/data/local/tmp/resident"
                session._validate_partial_reconfiguration_authority(command)

                retained_index = (selected_index + 1) % len(source)
                stale = tuple(
                    replace(row, session_generation=2)
                    if index == retained_index else row
                    for index, row in enumerate(source)
                )
                session._launch = SimpleNamespace(phone_shards=stale)
                with self.assertRaisesRegex(
                    PhysicalAdapterError,
                    "stale phone session replacement authorization",
                ):
                    session._validate_partial_reconfiguration_authority(
                        command
                    )
                self.assertEqual(
                    command.replacement_authorization.selected_session_id,
                    selected,
                )
                self.assertEqual(
                    command.transition.changed_phone_session_ids,
                    (selected,),
                )

    def test_replacement_authorization_is_required_only_for_partial(self) -> None:
        source = tuple(
            shard("HTP" + str(index), QWEN, 1 << index)
            for index in range(3)
        )
        selected = source[-1].session_id
        replacement = replace(
            source[-1],
            artifact_sha256=GEMMA,
            operator_plan_sha256="sha256:" + "e" * 64,
            session_generation=2,
        )
        target = (*source[:-1], replacement)
        legacy = SimpleNamespace(
            replacement_authorization=None,
            transition=SimpleNamespace(
                changed_phone_session_ids=(selected,),
                phone_shards=target,
                evictions=(),
            ),
        )
        validate_phone_session_replacement_command(legacy)

        partial = SimpleNamespace(
            replacement_authorization=None,
            transition=SimpleNamespace(
                changed_phone_session_ids=(selected,),
                phone_shards=target,
                evictions=(SimpleNamespace(session_id=selected),),
            ),
        )
        with self.assertRaisesRegex(
            PhysicalAdapterError, "lacks exact replacement authority"
        ):
            validate_phone_session_replacement_command(partial)
        partial.replacement_authorization = (
            PhoneSessionReplacementAuthorization.create(
                selected_session_id=selected,
                source_shards=source,
                target_shards=target,
            )
        )
        validate_phone_session_replacement_command(partial)

    def test_stale_scheduler_authorization_defers_and_requests_replan(
        self,
    ) -> None:
        selected = "session-selected"
        stale_generation = 2
        geometry = "sha256:" + "7" * 64
        artifact = "sha256:" + "8" * 64
        target_shard = shard(selected, artifact, 1, session_generation=2)
        layout = SimpleNamespace(
            changed_session_ids=(selected,),
            geometry_sha256=geometry,
        )
        state = SimpleNamespace(
            generation=stale_generation,
            layout=layout,
            state="PROPOSED",
            covers_artifact=lambda value: value == artifact,
        )
        ready = SimpleNamespace(layout=SimpleNamespace(
            geometry_sha256="sha256:" + "9" * 64,
        ))
        transition = SimpleNamespace(
            transition_id="replace-selected",
            latency_us=10,
            resource_ids=(),
            resource_slots={},
        )
        memory_demand = SimpleNamespace(
            device_id="phone",
            kind="resident",
            required_bytes=100,
        )
        helper = SimpleNamespace(
            phone_layout_generation=stale_generation,
            phone_layout_geometry_sha256=geometry,
            operator_plan_sha256="sha256:" + "a" * 64,
            replacement_authorization=SimpleNamespace(
                selected_session_id=selected,
            ),
            preparation_transitions=(transition,),
            helper_binding=object(),
            helper_plan=SimpleNamespace(
                execution_contract=SimpleNamespace(
                    phone_shards=(target_shard,),
                    phone_device_id="phone",
                ),
                memory_demands=(memory_demand,),
            ),
            preparation_ticket_id=lambda _ticket_id: "preparation-ticket",
        )
        ticket = SimpleNamespace(
            ticket_id="request-ticket",
            dispatch_state="ACQUIRED",
            execution_plan=SimpleNamespace(helper_envelope=helper),
            request=SimpleNamespace(request_id="request"),
            model=SimpleNamespace(
                model_id="model",
                artifact_sha256=artifact,
            ),
            binding=SimpleNamespace(executor_id="desktop"),
        )
        controller = mock.Mock()
        controller.phone_layout.return_value = state
        controller.phone_layout_sessions_are_usable.return_value = False
        controller.phone_layout_transition_blockers.return_value = ()
        controller.ready_phone_layout.return_value = ready
        controller.reject_phone_layout_proposal.return_value = ready
        compiler = mock.Mock()
        scheduler = object.__new__(UnifiedScheduler)
        scheduler._runtime_lock = threading.RLock()
        scheduler._model_placement_controller = controller
        scheduler._request_helper_preparations = {}
        scheduler._request_helper_preparation_envelopes = {
            ("request", ticket.ticket_id, stale_generation): helper,
        }
        scheduler._automated_route_compiler = compiler
        scheduler._runtime_epoch_route_compiler = None
        scheduler._runtime_capabilities = SimpleNamespace(
            executors=(), exclusive_residency_resources={},
        )
        scheduler.runtime_ticket = mock.Mock(return_value=ticket)
        scheduler.runtime_model_manifest = mock.Mock(return_value=object())
        scheduler._phone_layout_snapshot_verification = mock.Mock(
            return_value=None
        )
        scheduler._copy_on_write_preparation_yielding_resources = mock.Mock(
            return_value=()
        )
        scheduler._partial_phone_replacement_memory_demands = mock.Mock(
            return_value=(memory_demand,)
        )
        scheduler._runtime_transaction_checkpoint = mock.Mock(
            return_value=object()
        )
        scheduler._restore_runtime_transaction = mock.Mock()
        scheduler._validate_phone_session_replacement_authorization = (
            mock.Mock(side_effect=_StalePhoneSessionAssignment("stale"))
        )
        scheduler._update_phone_residency_portfolio = mock.Mock()
        scheduler._defer_phone_layout_revalidation = mock.Mock(return_value=None)
        scheduler._reject_stale_phone_layout_proposal = mock.Mock(return_value=None)
        snapshot = mock.Mock(spec=HeterogeneousRuntimeSnapshot)

        result = scheduler.begin_request_helper_preparation(
            "request",
            observed_at_us=100,
            snapshot=snapshot,
            expected_phone_layout_generation=stale_generation,
        )

        self.assertEqual(result["status"], "DEFERRED")
        self.assertEqual(result["reason"], "STALE_PHONE_SESSION_ASSIGNMENT")
        self.assertEqual(result["replan_status"], "REQUESTED")
        controller.reject_phone_layout_proposal.assert_called_once_with(
            stale_generation,
            observed_at_us=100,
            reason="STALE_PHONE_SESSION_ASSIGNMENT",
        )
        compiler.set_phone_residency_layout.assert_called_once_with(
            ready.layout
        )
        scheduler._update_phone_residency_portfolio.assert_called_once_with(
            ticket.request,
            scheduler.runtime_model_manifest.return_value,
            100,
            snapshot,
        )
        self.assertEqual(scheduler._request_helper_preparation_envelopes, {})
        self.assertEqual(
            helper.replacement_authorization.selected_session_id, selected
        )


class SessionMaskAcknowledgementTests(unittest.TestCase):
    def test_pending_drain_retry_preserves_the_bound_policy(self) -> None:
        scheduler = object.__new__(UnifiedScheduler)
        controller = mock.Mock()
        controller.request_helper_rebind.return_value = SimpleNamespace(
            state="REQUESTED",
            target_allowed_session_ids=("session-a", "session-b"),
            retained_layer_mask=3,
            drain_policy_sha256="sha256:" + "c" * 64,
        )
        scheduler._model_placement_controller = controller
        scheduler._adaptive_decode = mock.Mock()
        scheduler._adaptive_decode.request_helper_session_drain.return_value = {
            "already_applied": False,
            "policy_hash": "sha256:" + "d" * 64,
        }
        scheduler._transaction = mock.Mock(return_value=nullcontext())

        result = scheduler._defer_preparation_for_blockers(
            "replacement-request", SimpleNamespace(generation=2),
            ("serving-request",), 20_000,
        )

        self.assertEqual(result["status"], "DEFERRED")
        scheduler._adaptive_decode.request_helper_session_drain\
            .assert_not_called()
        controller.bind_request_helper_rebind_drain_policy\
            .assert_not_called()
        controller.mark_request_helper_rebind_quiesced.assert_not_called()

    def test_already_quiesced_blocker_needs_no_new_drain_policy(self) -> None:
        scheduler = object.__new__(UnifiedScheduler)
        controller = mock.Mock()
        controller.request_helper_rebind.return_value = SimpleNamespace(
            state="QUIESCED",
            target_allowed_session_ids=("session-a", "session-b"),
            retained_layer_mask=3,
        )
        scheduler._model_placement_controller = controller
        scheduler._adaptive_decode = mock.Mock()
        scheduler._transaction = mock.Mock(return_value=nullcontext())

        result = scheduler._defer_preparation_for_blockers(
            "gemma-request",
            SimpleNamespace(generation=2),
            ("qwen-request",),
            20_000,
        )

        self.assertEqual(result["status"], "DEFERRED")
        scheduler._adaptive_decode.request_helper_session_drain\
            .assert_not_called()
        controller.bind_request_helper_rebind_drain_policy\
            .assert_not_called()

    def test_server_ack_publishes_the_exact_retained_session_mask(self) -> None:
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        controller = mock.Mock()
        selected = "session-selected"
        retained = ("session-a", "session-b")
        policy_sha256 = "sha256:" + "c" * 64
        controller.request_helper_rebind_state.return_value = {
            "drain_policy_sha256": policy_sha256,
            "removed_session_ids": [selected],
            "target_allowed_session_ids": list(retained),
            "target_generation": 2,
        }
        scheduler._model_placement_controller = controller
        directive = SimpleNamespace(reason="MASK_ACKNOWLEDGED")
        scheduler._adaptive_decode = mock.Mock()
        scheduler._adaptive_decode.acknowledge.return_value = directive
        acknowledgement = AdaptiveDecodePolicyAck(
            request_id="qwen-request",
            slot_id=7,
            plan_generation=3,
            applied_token_index=19,
            applied_at_us=20_000,
            policy_hash=policy_sha256,
        )
        baseline = SimpleNamespace(policy_hash="sha256:" + "d" * 64)
        with (
            mock.patch.object(
                scheduler,
                "_runtime_transaction_checkpoint",
                return_value=object(),
            ),
            mock.patch.object(
                scheduler,
                "runtime_execution_ticket",
                return_value=object(),
            ),
            mock.patch.object(
                scheduler,
                "_adaptive_policies_from_ticket",
                return_value=(baseline, (), None),
            ),
            mock.patch.object(
                scheduler,
                "_track_adaptive_directive",
                return_value=directive,
            ),
        ):
            result = scheduler.acknowledge_adaptive_decode_control(
                "qwen-request", acknowledgement
            )

        self.assertIs(result, directive)
        controller.mark_request_helper_rebind_quiesced.assert_called_once_with(
            "qwen-request",
            2,
            observed_at_us=20_000,
            allowed_session_ids=retained,
            drain_policy_sha256=policy_sha256,
        )


class ReadyHelperRefreshTests(unittest.TestCase):
    def test_ready_publication_resumes_or_rebinds_the_current_helper(self) -> None:
        scheduler = object.__new__(UnifiedScheduler)
        scheduler._request_helper_opportunities = {}
        scheduler._late_request_helper_contexts = {}
        scheduler._remember_request_helper_envelope = mock.Mock()
        scheduler._record_ready_helper_event_once = mock.Mock()
        scheduler._model_placement_controller = mock.Mock()
        scheduler._adaptive_decode = mock.Mock()
        scheduler._adaptive_decode.snapshot.return_value = {
            "helper_available": False,
            "allow_assumed_phone_power_for_operational_selection": False,
        }
        ticket = SimpleNamespace(
            request=SimpleNamespace(request_id="serving-request"),
            decision=SimpleNamespace(route_id="desktop-parent"),
            execution_plan=SimpleNamespace(),
        )
        layout = SimpleNamespace(
            generation=3,
            layout=SimpleNamespace(geometry_sha256="sha256:" + "3" * 64),
        )
        materialized = SimpleNamespace(
            helper=SimpleNamespace(
                operator_plan_sha256="sha256:" + "4" * 64,
                helper_plan=SimpleNamespace(execution_contract=SimpleNamespace(
                    phone_shards=(SimpleNamespace(session_id="session-a"),),
                    phone_device_id=None,
                )),
            ),
            opportunity=SimpleNamespace(evidence_state="LEARNING"),
            baseline=mock.sentinel.baseline,
            policies=(mock.sentinel.policy,),
            ticket_policy=None,
            component=SimpleNamespace(identity_sha256="sha256:" + "5" * 64),
            generated_parent_route_id="desktop-parent",
        )

        scheduler._publish_ready_helper_materialization(
            ticket, layout, materialized, True, 20_000
        )

        scheduler._adaptive_decode.helper_rebound.assert_not_called()
        scheduler._adaptive_decode.helper_ready.assert_called_once_with(
            "serving-request",
            phone_layout_generation=layout.generation,
            phone_layout_geometry_sha256=layout.layout.geometry_sha256,
            candidates=materialized.policies,
            component_capability_sha256=materialized.component.identity_sha256,
            ticket_policy=None,
            helper_evidence_state="LEARNING",
            allow_assumed_phone_power_for_operational_selection=False,
        )
        scheduler._adaptive_decode.helper_ready.reset_mock()
        scheduler._adaptive_decode.snapshot.return_value["helper_available"] = True
        scheduler._publish_ready_helper_materialization(
            ticket, layout, materialized, False, 21_000
        )
        scheduler._adaptive_decode.helper_ready.assert_not_called()
        scheduler._adaptive_decode.helper_rebound.assert_called_once_with(
            "serving-request", phone_layout_generation=layout.generation,
            phone_layout_geometry_sha256=layout.layout.geometry_sha256,
            candidates=materialized.policies,
            component_capability_sha256=materialized.component.identity_sha256,
            ticket_policy=None, helper_evidence_state="LEARNING", compatible_layers_by_plan={},
        )

    def test_decode_boundary_attaches_new_ready_helper_once(self) -> None:
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        request_id = "late-ready-request"
        ticket = SimpleNamespace(
            request=SimpleNamespace(request_id=request_id),
            ticket_id="late-ready-request:attempt:0",
            execution_plan=SimpleNamespace(),
        )
        helper = SimpleNamespace(
            phone_layout_generation=2,
            phone_layout_geometry_sha256="sha256:" + "8" * 64,
            operator_plan_sha256="sha256:" + "9" * 64,
            helper_plan=SimpleNamespace(
                execution_contract=SimpleNamespace(phone_device_id=None),
            ),
        )
        ready = (helper, mock.sentinel.layout, mock.sentinel.component)
        directive = mock.sentinel.directive
        controller = mock.Mock()
        controller.request_binding.side_effect = (
            {"helper_attachment": None},
            {"helper_attachment": None},
            {"helper_attachment": {
                "start_token_index": 11,
                "phone_layout_generation": 2,
                "phone_layout_geometry_sha256": (
                    helper.phone_layout_geometry_sha256
                ),
                "operator_plan_sha256": helper.operator_plan_sha256,
            }},
        )
        controller.request_helper_rebind_state.return_value = None
        scheduler._model_placement_controller = controller
        scheduler._late_request_helper_contexts = {}
        scheduler._adaptive_decode = mock.Mock()
        scheduler._adaptive_decode.snapshot.return_value = {
            "helper_available": False,
            "allow_assumed_phone_power_for_operational_selection": False,
        }
        scheduler._adaptive_decode.boundary.return_value = directive

        with (
            mock.patch.object(
                scheduler,
                "runtime_execution_ticket",
                return_value=ticket,
            ),
            mock.patch.object(
                scheduler,
                "_reevaluate_pending_phone_layout_at_boundary",
            ),
            mock.patch.object(
                scheduler,
                "_request_helper_envelope",
                side_effect=(None, helper, helper),
            ),
            mock.patch.object(
                scheduler,
                "_ready_request_helper",
                return_value=ready,
            ) as resolve_ready,
            mock.patch.object(
                scheduler,
                "_attach_ready_request_helper",
                return_value=True,
            ) as attach,
            mock.patch.object(
                scheduler,
                "_track_adaptive_directive",
                return_value=directive,
            ),
        ):
            scheduler.adaptive_decode_boundary(
                request_id,
                slot_id=4,
                token_index=10,
                at_us=10_000,
            )
            scheduler.adaptive_decode_boundary(
                request_id,
                slot_id=4,
                token_index=11,
                at_us=11_000,
            )
            scheduler.adaptive_decode_boundary(
                request_id,
                slot_id=4,
                token_index=12,
                at_us=12_000,
            )

        self.assertEqual(resolve_ready.call_count, 2)
        self.assertTrue(all(
            row.args == (ticket, helper)
            for row in resolve_ready.call_args_list
        ))
        attach.assert_called_once_with(
            ticket,
            slot_id=4,
            token_index=11,
            at_us=11_000,
            fraction_ppm=0,
            ready_helper=ready,
        )

    def test_ready_replacement_reuses_its_exact_load_authorization(self) -> None:
        scheduler = object.__new__(UnifiedScheduler)
        source_shards = tuple(
            shard("HTP" + str(index), QWEN, 1 << index)
            for index in range(3)
        )
        target_shards = (
            shard("HTP0", GEMMA, 1, session_generation=2),
            *source_shards[1:],
        )
        authorization = PhoneSessionReplacementAuthorization.create(
            selected_session_id="HTP0",
            source_shards=source_shards,
            target_shards=target_shards,
        )
        geometry = "sha256:" + "8" * 64
        target = SimpleNamespace(
            generation=2,
            state="READY",
            layout=SimpleNamespace(
                changed_session_ids=("HTP0",),
                geometry_sha256=geometry,
                shards=target_shards,
                session_generation_by_id={
                    row.session_id: row.session_generation
                    for row in target_shards
                },
                replacement_source_identities=(
                    SimpleNamespace(session_id="HTP0"),
                ),
            ),
        )
        envelope = SimpleNamespace(
            phone_layout_generation=2,
            phone_layout_geometry_sha256=geometry,
            replacement_authorization=authorization,
        )
        scheduler._model_placement_controller = mock.Mock()
        scheduler._model_placement_controller.ready_phone_layout.return_value = (
            target
        )
        scheduler._request_helper_preparation_envelopes = {
            ("owner", "ticket", 2): envelope,
        }
        scheduler._request_helper_preparations = {}

        self.assertEqual(
            scheduler._phone_session_replacement_authorization(target),
            authorization,
        )

    def test_ready_helper_generation_restores_the_planning_compiler_view(
        self,
    ) -> None:
        class Compiler:
            def __init__(self, value):
                self.phone_residency_layout = value

            def set_phone_residency_layout(self, value):
                self.phone_residency_layout = value

        scheduler = object.__new__(UnifiedScheduler)
        target_layout = object()
        ready_layout = object()
        primary = Compiler(target_layout)
        epoch = Compiler(target_layout)
        scheduler._runtime_capabilities = object()
        scheduler._automated_route_compiler = primary
        scheduler._runtime_epoch_route_compiler = epoch
        ready = SimpleNamespace(layout=ready_layout)

        with scheduler._ready_layout_compiler_view(ready):
            self.assertIs(primary.phone_residency_layout, ready_layout)
            self.assertIs(epoch.phone_residency_layout, ready_layout)

        self.assertIs(primary.phone_residency_layout, target_layout)
        self.assertIs(epoch.phone_residency_layout, target_layout)

    def test_rollback_retains_rebind_when_only_retained_sessions_are_ready(
        self,
    ) -> None:
        scheduler = object.__new__(UnifiedScheduler)
        controller = mock.Mock()
        controller.request_binding.return_value = {
            "fraction_ppm": 0,
            "helper_attachment": {
                "allowed_session_ids": ["HTP0", "HTP1", "HTP2"],
                "fraction_ppm": 0,
                "lease_tokens": [],
                "lease_reserved_until_us": None,
            },
        }
        controller.phone_layout_sessions_are_usable.return_value = True
        scheduler._model_placement_controller = controller
        scheduler._request_helper_opportunities = {}
        scheduler._late_request_helper_contexts = {}
        scheduler._adaptive_decode = mock.Mock()
        rebind = {
            "state": "QUIESCED",
            "source_generation": 1,
            "source_geometry_sha256": "sha256:" + "1" * 64,
            "target_allowed_session_ids": ["HTP1", "HTP2"],
            "removed_session_ids": ["HTP0"],
            "previous_fraction_ppm": 0,
        }

        scheduler._settle_rebind_after_transition_failure(
            "qwen-request",
            rebind,
            preparation_ticket_id="replacement-ticket",
            failed_at_us=30_000,
            reason="injected_post_load_failure",
            unavailable=(),
        )

        controller.phone_layout_sessions_are_usable.assert_called_once_with(
            1,
            "sha256:" + "1" * 64,
            ("HTP1", "HTP2"),
        )
        controller.cancel_request_helper_rebind.assert_not_called()
        controller.detach_request_helper.assert_not_called()
        self.assertEqual(
            controller.record_request_helper_event.call_args.args[1],
            "REBIND_RETAINED_AFTER_ROLLBACK",
        )

    def test_ready_layout_reuses_its_verified_safety_sample(self) -> None:
        scheduler = object.__new__(UnifiedScheduler)
        safety = RuntimeExecutorState(
            executor_id="phone-executor",
            healthy=True,
            ready=True,
            temperature_millic=42_000,
            battery_ppm=800_000,
            free_slots=1,
            busy_until_us=0,
            thermal_qualified=True,
        )
        geometry = "sha256:" + "2" * 64
        scheduler._request_helper_preparations = {
            "old": SimpleNamespace(
                state="READY",
                phone_layout_generation=1,
                phone_layout_geometry_sha256=geometry,
                verification_sha256="sha256:" + "3" * 64,
                phone_safety_state=safety,
                ready_at_us=10,
                started_at_us=1,
                preparation_ticket_id="old",
            ),
            "exact": SimpleNamespace(
                state="READY",
                phone_layout_generation=2,
                phone_layout_geometry_sha256=geometry,
                verification_sha256="sha256:" + "4" * 64,
                phone_safety_state=safety,
                ready_at_us=20,
                started_at_us=2,
                preparation_ticket_id="exact",
            ),
        }
        layout = SimpleNamespace(
            generation=2,
            layout=SimpleNamespace(geometry_sha256=geometry),
        )

        self.assertIs(
            scheduler._ready_layout_phone_safety_state(layout),
            safety,
        )

    def test_preparation_owner_does_not_own_the_ready_route_template(
        self,
    ) -> None:
        scheduler = object.__new__(UnifiedScheduler)
        geometry = "sha256:" + "5" * 64
        desktop = "sha256:" + "6" * 64
        phone_shard = shard("HTP0", GEMMA, 1, session_generation=2)
        layout = SimpleNamespace(
            generation=2,
            state="READY",
            layout=SimpleNamespace(
                geometry_sha256=geometry,
                shards=(phone_shard,),
            ),
            covers_artifact=lambda artifact: artifact == GEMMA,
        )
        helper_plan = SimpleNamespace(
            baseline_executor_id="desktop-executor",
            execution_contract=SimpleNamespace(
                phone_shards=(phone_shard,),
            ),
        )
        helper_binding = object()
        envelope = SimpleNamespace(
            artifact_sha256=GEMMA,
            desktop_parent_route_id="desktop-route",
            desktop_placement_sha256=desktop,
            phone_layout_generation=2,
            phone_layout_geometry_sha256=geometry,
            helper_plan=helper_plan,
            helper_binding=helper_binding,
            operator_plan_sha256="sha256:" + "7" * 64,
        )
        scheduler._model_placement_controller = mock.Mock()
        scheduler._model_placement_controller.ready_phone_layout.return_value = (
            layout
        )
        scheduler._request_helper_opportunities = {"second-request": ()}
        scheduler._request_helper_envelope_history = {}
        scheduler._offline_phone_residency_plans = {}
        scheduler._request_helper_preparation_envelopes = {
            ("preparation-owner", "ticket", 2): envelope,
        }
        ticket = SimpleNamespace(
            request=SimpleNamespace(request_id="second-request"),
            model=SimpleNamespace(artifact_sha256=GEMMA),
            decision=SimpleNamespace(route_id="desktop-route"),
            binding=SimpleNamespace(executor_id="desktop-executor"),
            execution_plan=SimpleNamespace(
                helper_envelope=None,
                desktop_placement_sha256=desktop,
            ),
        )

        actual_plan, actual_binding = (
            scheduler._resolve_verified_ready_helper_plan(ticket, layout)
        )
        self.assertIs(actual_plan, helper_plan)
        self.assertIs(actual_binding, helper_binding)

    def test_retained_helper_history_bootstraps_its_ready_subset(self) -> None:
        scheduler = object.__new__(UnifiedScheduler)
        geometry = "sha256:" + "9" * 64
        desktop = "sha256:" + "a" * 64
        phone_shard = shard("HTP1", QWEN, 2)
        layout = SimpleNamespace(
            generation=3,
            state="READY",
            layout=SimpleNamespace(
                geometry_sha256=geometry,
                shards=(phone_shard,),
            ),
            covers_artifact=lambda artifact: artifact == QWEN,
        )
        helper_plan = SimpleNamespace(
            baseline_executor_id="desktop-executor",
            execution_contract=SimpleNamespace(
                phone_shards=(phone_shard,),
            ),
        )
        participant = SimpleNamespace(
            executor_id="phone-executor",
            device_id="phone",
            endpoint="phone://endpoint",
            backend="phone-backend",
            resource_ids=("phone-htp",),
        )
        helper_binding = SimpleNamespace(
            executor_id="helper-executor",
            endpoint="helper://endpoint",
            backend="helper-backend",
            operator_plan_protocol="operator-plan-v1",
            participants=(participant,),
        )
        historical = SimpleNamespace(
            artifact_sha256=QWEN,
            desktop_parent_route_id="desktop-route",
            desktop_placement_sha256=desktop,
            phone_layout_generation=1,
            operator_plan_sha256="sha256:" + "b" * 64,
            helper_plan=helper_plan,
            helper_binding=helper_binding,
        )
        scheduler._model_placement_controller = mock.Mock()
        scheduler._model_placement_controller.ready_phone_layout.return_value = (
            layout
        )
        scheduler._request_helper_opportunities = {"qwen-request": ()}
        scheduler._request_helper_envelope_history = {
            "qwen-request": {historical.operator_plan_sha256: historical},
        }
        scheduler._offline_phone_residency_plans = {}
        scheduler._request_helper_preparation_envelopes = {}
        ticket = SimpleNamespace(
            request=SimpleNamespace(request_id="qwen-request"),
            model=SimpleNamespace(artifact_sha256=QWEN),
            decision=SimpleNamespace(route_id="desktop-route"),
            binding=SimpleNamespace(executor_id="desktop-executor"),
            execution_plan=SimpleNamespace(
                helper_envelope=None,
                desktop_placement_sha256=desktop,
            ),
        )

        actual_plan, actual_binding = (
            scheduler._resolve_verified_ready_helper_plan(ticket, layout)
        )
        self.assertIs(actual_plan, helper_plan)
        self.assertIs(actual_binding, helper_binding)

    def test_system_endpoint_template_bootstraps_late_request(self) -> None:
        scheduler = object.__new__(UnifiedScheduler)
        old_geometry = "sha256:" + "8" * 64
        ready_geometry = "sha256:" + "9" * 64
        desktop = "sha256:" + "a" * 64
        old_shard = shard("HTP0", QWEN, 1)
        ready_shard = shard("HTP1", QWEN, 2)
        layout = SimpleNamespace(
            generation=2,
            state="READY",
            layout=SimpleNamespace(
                geometry_sha256=ready_geometry,
                shards=(ready_shard,),
            ),
            covers_artifact=lambda artifact: artifact == QWEN,
        )
        helper_plan = SimpleNamespace(
            baseline_executor_id="desktop-executor",
            execution_contract=SimpleNamespace(
                phone_shards=(old_shard,),
            ),
        )
        participant = SimpleNamespace(
            executor_id="phone-executor",
            device_id="phone",
            endpoint="phone://endpoint",
            backend="phone-backend",
            resource_ids=("phone-htp",),
        )
        helper_binding = SimpleNamespace(
            artifact_sha256=QWEN,
            model_id="qwen",
            executor_id="helper-executor",
            endpoint="helper://endpoint",
            backend="helper-backend",
            operator_plan_protocol="operator-plan-v1",
            participants=(participant,),
        )
        template = SimpleNamespace(
            artifact_sha256=QWEN,
            desktop_parent_route_id="desktop-route-cold",
            desktop_placement_sha256=desktop,
            phone_layout_generation=1,
            phone_layout_geometry_sha256=old_geometry,
            operator_plan_sha256="sha256:" + "b" * 64,
            helper_plan=helper_plan,
            helper_binding=helper_binding,
            preparation_changed_session_ids=(),
            replacement_authorization=None,
        )
        scheduler._phone_helper_endpoint_templates = {}
        scheduler._remember_phone_helper_endpoint_template(template)
        scheduler._model_placement_controller = mock.Mock()
        scheduler._model_placement_controller.ready_phone_layout.return_value = (
            layout
        )
        scheduler._request_helper_opportunities = {"late-request": ()}
        scheduler._request_helper_envelope_history = {}
        scheduler._offline_phone_residency_plans = {}
        scheduler._request_helper_preparation_envelopes = {}
        ticket = SimpleNamespace(
            request=SimpleNamespace(request_id="late-request"),
            model=SimpleNamespace(
                artifact_sha256=QWEN,
                model_id="qwen",
            ),
            decision=SimpleNamespace(route_id="desktop-route-hot"),
            binding=SimpleNamespace(executor_id="desktop-executor"),
            execution_plan=SimpleNamespace(
                helper_envelope=None,
                desktop_placement_sha256=desktop,
            ),
        )

        actual_plan, actual_binding = (
            scheduler._resolve_verified_ready_helper_plan(ticket, layout)
        )

        self.assertIs(actual_plan, helper_plan)
        self.assertIs(actual_binding, helper_binding)

    def test_ready_template_rejects_stale_generation_and_parent(self) -> None:
        scheduler = object.__new__(UnifiedScheduler)
        geometry = "sha256:" + "8" * 64
        desktop = "sha256:" + "9" * 64
        stale_shard = shard("HTP0", QWEN, 1, session_generation=1)
        ready_shard = replace(stale_shard, session_generation=2)
        layout = SimpleNamespace(
            generation=2,
            state="READY",
            layout=SimpleNamespace(
                geometry_sha256=geometry,
                shards=(ready_shard,),
                session_generation_by_id={"HTP0": 2},
                session_identities=(),
            ),
            covers_artifact=lambda artifact: artifact == QWEN,
        )
        helper_plan = SimpleNamespace(
            baseline_executor_id="desktop-executor",
            execution_contract=SimpleNamespace(
                phone_shards=(stale_shard,),
            ),
            adapter_parameters={},
        )
        binding = SimpleNamespace(
            executor_id="helper-executor",
            endpoint="helper://endpoint",
            backend="helper-backend",
            operator_plan_protocol="operator-plan-v1",
            participants=(),
        )
        template = SimpleNamespace(
            artifact_sha256=QWEN,
            desktop_parent_route_id="desktop-route",
            desktop_placement_sha256=desktop,
            phone_layout_generation=1,
            phone_layout_geometry_sha256=geometry,
            operator_plan_sha256="sha256:" + "a" * 64,
            helper_plan=helper_plan,
            helper_binding=binding,
            preparation_changed_session_ids=(),
            replacement_authorization=None,
        )
        scheduler._model_placement_controller = mock.Mock()
        scheduler._model_placement_controller.ready_phone_layout.return_value = (
            layout
        )
        scheduler._offline_phone_residency_plans = {}
        scheduler._phone_helper_endpoint_templates = {("stale",): template}

        lookup = {
            "artifact_sha256": QWEN,
            "desktop_parent_route_id": "desktop-route",
            "desktop_placement_sha256": desktop,
            "baseline_executor_id": "desktop-executor",
        }
        self.assertIsNone(
            scheduler._authoritative_ready_helper_template(**lookup)
        )
        scheduler._phone_helper_endpoint_templates = {
            ("current",): SimpleNamespace(
                **{
                    **vars(template),
                    "phone_layout_generation": 2,
                    "helper_plan": SimpleNamespace(
                        **{
                            **vars(helper_plan),
                            "execution_contract": SimpleNamespace(
                                phone_shards=(ready_shard,)
                            ),
                        }
                    ),
                }
            ),
        }
        self.assertIsNone(scheduler._authoritative_ready_helper_template(
            **{**lookup, "desktop_placement_sha256": "sha256:" + "7" * 64}
        ))

    def test_insufficient_remaining_opportunity_is_a_clean_rejection(
        self,
    ) -> None:
        scheduler = object.__new__(UnifiedScheduler)
        request_id = "short-request"
        ticket = SimpleNamespace(
            ticket_id="short-ticket",
            dispatch_state="ACQUIRED",
            transition_status="NOT_REQUIRED",
            request=SimpleNamespace(request_id=request_id, output_tokens=32),
            model=SimpleNamespace(artifact_sha256=QWEN),
            execution_plan=SimpleNamespace(
                desktop_placement_sha256="sha256:" + "4" * 64,
            ),
        )
        layout = SimpleNamespace(
            generation=2,
            state="READY",
            layout=SimpleNamespace(
                geometry_sha256="sha256:" + "5" * 64,
                shards=(SimpleNamespace(
                    artifact_sha256=QWEN,
                    session_id="HTP0",
                ),),
                session_generation_by_id={"HTP0": 2},
            ),
        )
        controller = mock.Mock()
        controller.remaining_request_decode_tokens.return_value = 3
        controller.request_helper_events.return_value = ()
        scheduler._model_placement_controller = controller
        scheduler._runtime_controller = mock.Mock()
        scheduler._runtime_controller.current_tickets.return_value = (ticket,)
        scheduler._adaptive_envelope_minimum_remaining_tokens = 4
        scheduler._late_request_helper_contexts = {}
        scheduler._request_helper_preparations = {}

        updated = scheduler._rematerialize_ready_layout_helpers(
            layout,
            mock.Mock(spec=HeterogeneousRuntimeSnapshot),
            50_000,
            phone_safety_state=mock.sentinel.safety,
        )

        self.assertEqual(updated, ())
        controller.record_request_helper_event.assert_called_once()
        args = controller.record_request_helper_event.call_args.args
        self.assertEqual(args[:3], (request_id, "REJECTED", 50_000))
        self.assertFalse(args[3]["accepted"])
        self.assertEqual(
            args[3]["reason"], "INSUFFICIENT_REMAINING_OPPORTUNITY"
        )

    def test_ready_publication_waits_for_placement_acquire_ack(self) -> None:
        scheduler = object.__new__(UnifiedScheduler)
        request_id = "pending-acquire"
        ticket = SimpleNamespace(
            ticket_id="pending-ticket",
            dispatch_state="ACQUIRED",
            transition_status="NOT_REQUIRED",
            request=SimpleNamespace(request_id=request_id, output_tokens=32),
            model=SimpleNamespace(artifact_sha256=QWEN),
            execution_plan=SimpleNamespace(
                desktop_placement_sha256="sha256:" + "4" * 64,
            ),
        )
        layout = SimpleNamespace(
            generation=2,
            state="READY",
            layout=SimpleNamespace(
                geometry_sha256="sha256:" + "5" * 64,
                shards=(SimpleNamespace(
                    artifact_sha256=QWEN,
                    session_id="HTP0",
                ),),
            ),
        )
        controller = mock.Mock()
        controller.request_is_acquired.return_value = False
        scheduler._model_placement_controller = controller
        scheduler._runtime_controller = mock.Mock()
        scheduler._runtime_controller.current_tickets.return_value = (ticket,)
        scheduler._late_request_helper_contexts = {}
        scheduler._request_helper_preparations = {}

        updated = scheduler._rematerialize_ready_layout_helpers(
            layout,
            mock.Mock(spec=HeterogeneousRuntimeSnapshot),
            50_000,
            phone_safety_state=mock.sentinel.safety,
        )

        self.assertEqual(updated, ())
        controller.record_request_helper_event.assert_not_called()

    def test_short_shared_policy_request_can_materialize_a_ready_helper(self) -> None:
        scheduler = object.__new__(UnifiedScheduler)
        ticket = SimpleNamespace(
            ticket_id="short-ticket", dispatch_state="ACQUIRED", transition_status="NOT_REQUIRED",
            request=SimpleNamespace(request_id="short-request", output_tokens=3),
            model=SimpleNamespace(artifact_sha256=QWEN),
            execution_plan=SimpleNamespace(desktop_placement_sha256="sha256:" + "4" * 64),
        )
        layout = SimpleNamespace(
            generation=2, state="READY", layout=SimpleNamespace(
                geometry_sha256="sha256:" + "5" * 64,
                shards=(SimpleNamespace(artifact_sha256=QWEN, session_id="HTP0"),),
                session_generation_by_id={"HTP0": 2},
            ),
        )
        placement = mock.Mock()
        placement.remaining_request_decode_tokens.return_value = 2
        placement.request_helper_events.return_value = ()
        placement.request_helper_rebind_state.return_value = None
        scheduler._model_placement_controller = placement
        scheduler._runtime_controller = mock.Mock()
        scheduler._runtime_controller.current_tickets.return_value = (ticket,)
        scheduler._adaptive_envelope_minimum_remaining_tokens = 24
        scheduler._adaptive_decode_config = AdaptiveDecodeConfig(server_policy_coherence=True)
        scheduler._retain_usable_ready_layout_helper = mock.Mock(return_value=False)
        scheduler._ready_helper_parent_rejection_unchanged = mock.Mock(return_value=False)
        scheduler._rematerialize_ready_layout_helper = mock.Mock()
        snapshot = mock.Mock(spec=HeterogeneousRuntimeSnapshot)
        updated = scheduler._rematerialize_ready_layout_helpers(
            layout, snapshot, 50_000, phone_safety_state=mock.sentinel.safety)
        self.assertEqual(updated, ("short-request",))
        scheduler._rematerialize_ready_layout_helper.assert_called_once_with(
            ticket, layout, snapshot, 50_000, mock.sentinel.safety)

        placement.remaining_request_decode_tokens.return_value = 0
        scheduler._rematerialize_ready_layout_helper.reset_mock()
        self.assertEqual(scheduler._rematerialize_ready_layout_helpers(
            layout, snapshot, 50_001, phone_safety_state=mock.sentinel.safety), ())
        scheduler._rematerialize_ready_layout_helper.assert_not_called()

    def test_ready_refresh_waits_for_retained_rebind_retry(self) -> None:
        scheduler = object.__new__(UnifiedScheduler)
        request_id = "qwen-request"
        ticket = SimpleNamespace(
            ticket_id="qwen-ticket",
            transition_status="NOT_REQUIRED",
            request=SimpleNamespace(
                request_id=request_id,
                output_tokens=32,
            ),
            model=SimpleNamespace(artifact_sha256=QWEN),
            dispatch_state="ACQUIRED",
            execution_plan=SimpleNamespace(
                desktop_placement_sha256="sha256:" + "c" * 64,
            ),
        )
        controller = mock.Mock()
        controller.remaining_request_decode_tokens.return_value = 32
        controller.request_helper_events.return_value = ()
        controller.request_helper_rebind_state.return_value = {
            "target_generation": 2,
        }
        scheduler._model_placement_controller = controller
        scheduler._runtime_controller = mock.Mock()
        scheduler._runtime_controller.current_tickets.return_value = (ticket,)
        scheduler._late_request_helper_contexts = {request_id: object()}
        scheduler._request_helper_preparations = {}
        scheduler._adaptive_envelope_minimum_remaining_tokens = 24
        layout = SimpleNamespace(
            generation=1,
            state="READY",
            layout=SimpleNamespace(
                geometry_sha256="sha256:" + "d" * 64,
                shards=(SimpleNamespace(
                    artifact_sha256=QWEN,
                    session_id="HTP0",
                ),),
                session_generation_by_id={"HTP0": 1},
            ),
        )
        snapshot = mock.Mock(spec=HeterogeneousRuntimeSnapshot)

        updated = scheduler._rematerialize_ready_layout_helpers(
            layout, snapshot, 40_000
        )

        self.assertEqual(updated, ())
        self.assertIn(request_id, scheduler._late_request_helper_contexts)
        self.assertEqual(
            tuple(
                call.args[1]
                for call in controller.record_request_helper_event.call_args_list
            ),
            ("ELIGIBLE", "HELPER_REMATERIALIZATION_DEFERRED"),
        )
        self.assertEqual(
            controller.record_request_helper_event.call_args.args,
            (
                request_id,
                "HELPER_REMATERIALIZATION_DEFERRED",
                40_000,
                {
                    "phone_layout_generation": 1,
                    "reason": "REBIND_AWAITS_RETRY_TARGET",
                    "rebind_target_generation": 2,
                },
            ),
        )


class GenerationScopedProofTests(unittest.TestCase):
    def setUp(self) -> None:
        self.qwen = tuple(
            replace(
                shard("HTP" + str(index), QWEN, 1 << index),
                maximum_columns=17_408,
            )
            for index in range(3)
        )
        self.gemma = replace(
            shard("HTP2", GEMMA, 1 << 5, session_generation=2),
            maximum_columns=15_360,
        )

    def replace_htp2(self, session: DirectPhoneFfnSession) -> None:
        session._proof_shards = [*self.qwen[:2], self.gemma]
        session._replace_resident_shard(self.qwen[2], self.gemma)

    def test_earlier_ticket_may_report_calls_on_replaced_shard(self) -> None:
        session = session_with(self.qwen)
        session._ticket_bind_generations["qwen-ticket"] = 1
        self.replace_htp2(session)
        self.assertEqual(session.residency_generation, 2)
        session.record_execution_proof(
            "qwen-ticket", QWEN, tuple(proof(row) for row in self.qwen)
        )
        self.assertEqual(session._executed_proof_shards, list(self.qwen[:2]))
        historical = session._historical_execution_proofs
        self.assertEqual(len(historical), 1)
        self.assertEqual(historical[0]["session_id"], "HTP2")
        self.assertEqual(historical[0]["artifact_sha256"], QWEN)
        self.assertEqual(historical[0]["last_residency_generation"], 1)
        self.assertEqual(historical[0]["ticket_bound_generation"], 1)

    def test_later_ticket_cannot_claim_calls_on_replaced_shard(self) -> None:
        session = session_with(self.qwen)
        self.replace_htp2(session)
        session._ticket_bind_generations["late-ticket"] = 2
        with self.assertRaisesRegex(
            PhysicalAdapterError, "differs from loaded residency"
        ):
            session.record_execution_proof(
                "late-ticket", QWEN, (proof(self.qwen[2]),)
            )

    def test_current_shard_proof_still_binds_to_resident_shard(self) -> None:
        session = session_with(self.qwen)
        self.replace_htp2(session)
        session._ticket_bind_generations["gemma-ticket"] = 2
        session.record_execution_proof(
            "gemma-ticket", GEMMA, (proof(self.gemma),)
        )
        self.assertEqual(session._executed_proof_shards, [self.gemma])
        self.assertEqual(session._historical_execution_proofs, [])

    def test_executed_shard_retires_into_history_when_replaced(self) -> None:
        session = session_with(self.qwen)
        session.record_execution_proof(
            "layout-ticket", QWEN, tuple(proof(row) for row in self.qwen)
        )
        self.replace_htp2(session)
        self.assertEqual(session._executed_proof_shards, list(self.qwen[:2]))
        retired = [
            row for row in session._historical_execution_proofs
            if row["kind"] == "retired_executed_shard"
        ]
        self.assertEqual(len(retired), 1)
        self.assertEqual(retired[0]["session_id"], "HTP2")

    def test_rollback_reopens_source_window(self) -> None:
        session = session_with(self.qwen)
        self.replace_htp2(session)
        session._proof_shards = list(self.qwen)
        session._replace_resident_shard(self.gemma, self.qwen[2])
        self.assertEqual(session.residency_generation, 3)
        session._ticket_bind_generations["after-rollback"] = 3
        session.record_execution_proof(
            "after-rollback", QWEN, (proof(self.qwen[2]),)
        )
        self.assertEqual(session._executed_proof_shards, [self.qwen[2]])
        with self.assertRaisesRegex(
            PhysicalAdapterError, "differs from loaded residency"
        ):
            session.record_execution_proof(
                "after-rollback", GEMMA, (proof(self.gemma),)
            )

    def test_physical_rollback_manifests_and_acks_bumped_epoch(self) -> None:
        session = session_with(self.qwen)
        selected = self.gemma.session_id
        self.replace_htp2(session)
        session._ticket_bind_generations["historical-qwen"] = 1
        session.record_execution_proof(
            "historical-qwen", QWEN, (proof(self.qwen[2]),)
        )
        historical = tuple(session._historical_execution_proofs)
        session._launch = _LaunchState(
            phone_shards=(*self.qwen[:2], self.gemma),
            shard_manifest_sha256="sha256:" + "2" * 64,
        )
        session._load_count_by_session = {
            row.session_id: (2 if row.session_id == selected else 1)
            for row in session._launch.phone_shards
        }
        session._column_quantum_by_session = {
            row.session_id: (
                1_280 if row.session_id == selected else 2_176
            )
            for row in session._launch.phone_shards
        }
        session._max_tokens_by_session = {
            row.session_id: (2 if row.session_id == selected else 4)
            for row in session._launch.phone_shards
        }
        session.configuration = SimpleNamespace(
            diagnostic_host="192.0.2.1",
            diagnostic_port=20_000,
            launch_timeout_s=1,
            multi_session_port_base=21_000,
            multi_session_device_count=3,
            model_paths_by_artifact={QWEN: "/data/qwen.gguf"},
        )
        receipt = DirectPhoneFfnReconfigurationReceipt(
            ticket_id="replace-selected",
            previous_manifest_sha256="sha256:" + "1" * 64,
            target_manifest_sha256="sha256:" + "2" * 64,
            changed_session_id=selected,
            load_count=2,
            previous_load_count=1,
            column_quantum=1_280,
            previous_column_quantum=2_176,
            max_tokens=2,
            previous_max_tokens=4,
            previous_shard=self.qwen[2],
            target_shard=self.gemma,
            target_shards=(*self.qwen[:2], self.gemma),
        )
        connection = _ResidencyControlConnection(selected, load_count=3)
        with mock.patch(
            "research_dev.scheduler.adapters.phone_session."
            "socket.create_connection",
            return_value=connection,
        ):
            session.rollback_reconfiguration(receipt)

        _header, payload = connection.request.split(b"\n", 1)
        header_fields = _header.decode("ascii").split()
        self.assertEqual(header_fields[3], "2176")
        rows = {
            fields[0]: fields
            for fields in (
                row.split(",")
                for row in payload.decode("ascii").splitlines()
            )
        }
        self.assertTrue(all(len(fields) == 12 for fields in rows.values()))
        self.assertEqual(int(rows[selected][11]), 3)
        self.assertTrue(all(
            int(fields[11]) == 1
            for session_id, fields in rows.items()
            if session_id != selected
        ))
        restored = {
            row.session_id: row for row in session.phone_shards
        }
        self.assertEqual(restored[selected].artifact_sha256, QWEN)
        self.assertEqual(restored[selected].session_generation, 3)
        self.assertEqual(
            session.column_quantum_by_session[selected], 2_176
        )
        self.assertTrue(all(
            restored[row.session_id] == row for row in self.qwen[:2]
        ))
        self.assertEqual(session._historical_execution_proofs, list(historical))
        selected_windows = [
            row for row in session._shard_residency_windows
            if row.shard.session_id == selected
        ]
        self.assertEqual(
            [row.shard.session_generation for row in selected_windows],
            [1, 2, 3],
        )

    def test_terminal_proofs_keep_original_and_restored_epochs_distinct(
        self,
    ) -> None:
        selected = self.gemma.session_id
        restored = replace(self.qwen[2], session_generation=3)
        current = (*self.qwen[:2], restored)
        history = (self.qwen[2], self.gemma)
        proven = (
            *(terminal_proof(row) for row in self.qwen[:2]),
            terminal_proof(self.qwen[2], calls=2),
            terminal_proof(self.gemma),
            terminal_proof(restored, calls=3),
        )
        terminal = DirectPhoneFfnTerminalReceipt(
            transport="session-router",
            requests=5,
            reset_recoveries=0,
            status=0,
            queue_depth=4,
            maximum_pending_outputs=0,
            phone_payload_copies=20,
            d2h_completions=5,
            session_proofs=proven,
        )

        DirectPhoneFfnSession._validate_shard_terminal(
            terminal,
            current,
            historical_shards=history,
            executed_shards=(restored,),
        )
        epochs = sorted(
            row.session_generation
            for row in terminal.session_proofs
            if row.session_id == selected
            and row.artifact_sha256 == QWEN
        )
        self.assertEqual(epochs, [1, 3])


class RigRollbackTests(unittest.TestCase):
    def residency(self, shards) -> _PersistentPhoneResidency:
        return _PersistentPhoneResidency(
            executor_id="phone-helper",
            endpoint="http://desktop.invalid:1",
            phone_shards=shards,
            layout_geometry_sha256="sha256:" + "8" * 64,
            manifests_by_artifact={
                artifact: SimpleNamespace(artifact_sha256=artifact)
                for artifact in {row.artifact_sha256 for row in shards}
            },
            parameters_by_artifact={
                artifact: {"old": 1}
                for artifact in {row.artifact_sha256 for row in shards}
            },
            operator_plans_by_artifact={
                artifact: {"old": 1}
                for artifact in {row.artifact_sha256 for row in shards}
            },
            executions_by_artifact={
                artifact: SimpleNamespace(max_tokens=2)
                for artifact in {row.artifact_sha256 for row in shards}
            },
            load_count_by_session={row.session_id: 1 for row in shards},
            column_quantum_by_session={
                row.session_id: 32 for row in shards
            },
            max_tokens_by_session={row.session_id: 2 for row in shards},
            generation=1,
            participant_device_ids=("phone",),
            replacement_resource_ids=("phone-memory",),
            session_resource_ids=("htp", "usb"),
        )

    def make_rig(self, direct_phone, record):
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig._transition_lock = threading.Lock()
        rig._helper_reconfigurations = {"prep-ticket": record}
        rig._current_direct_phone = direct_phone
        rig._direct_phone_receipts = []
        rig._phone_residency = None
        return rig

    def command(self):
        return SimpleNamespace(
            ticket_id="prep-ticket",
            phone_layout_generation=2,
            transition=SimpleNamespace(
                transition_id="replace-htp2",
                changed_phone_session_ids=("HTP2",),
            ),
        )

    def test_rollback_restores_captured_source_and_verifies_it(self) -> None:
        qwen = tuple(shard("HTP" + str(i), QWEN, 1 << i) for i in range(3))
        gemma = shard("HTP2", GEMMA, 1 << 5, session_generation=2)
        previous = self.residency(qwen)
        receipt = DirectPhoneFfnReconfigurationReceipt(
            ticket_id="prep-ticket",
            previous_manifest_sha256="sha256:" + "1" * 64,
            target_manifest_sha256="sha256:" + "2" * 64,
            changed_session_id="HTP2",
            load_count=2,
            previous_load_count=1,
            column_quantum=32,
            previous_column_quantum=32,
            max_tokens=4,
            previous_max_tokens=2,
            previous_shard=qwen[2],
            target_shard=gemma,
            target_shards=(*qwen[:2], gemma),
        )
        direct_phone = SimpleNamespace(
            active=True,
            phone_shards=(*qwen[:2], gemma),
            load_count_by_session={"HTP0": 1, "HTP1": 1, "HTP2": 2},
            max_tokens_by_session={"HTP0": 2, "HTP1": 2, "HTP2": 4},
            column_quantum_by_session={
                "HTP0": 32, "HTP1": 32, "HTP2": 32,
            },
            residency_generation=2,
        )

        def rollback(value):
            self.assertIs(value, receipt)
            direct_phone.phone_shards = (
                *qwen[:2],
                replace(qwen[2], session_generation=3),
            )
            direct_phone.load_count_by_session = {
                "HTP0": 1, "HTP1": 1, "HTP2": 3,
            }
            direct_phone.max_tokens_by_session = {
                "HTP0": 2, "HTP1": 2, "HTP2": 2,
            }
            direct_phone.column_quantum_by_session = {
                "HTP0": 32, "HTP1": 32, "HTP2": 32,
            }
            direct_phone.residency_generation = 3

        direct_phone.rollback_reconfiguration = rollback
        rig = self.make_rig(
            direct_phone,
            _HelperReconfiguration(
                transition_id="replace-htp2",
                changed_session_ids=("HTP2",),
                receipt=receipt,
                previous_phone_residency=previous,
            ),
        )
        result = rig.rollback_helper_transition(self.command())
        self.assertTrue(result["physical_change"])
        self.assertEqual(result["changed_session_id"], "HTP2")
        self.assertEqual(result["command_changed_session_ids"], ["HTP2"])
        self.assertEqual(result["restored_shard"]["artifact_sha256"], QWEN)
        self.assertEqual(
            result["restored_session_generations"],
            {"HTP0": 1, "HTP1": 1, "HTP2": 3},
        )
        self.assertEqual(
            rig._phone_residency.phone_shards,
            (*qwen[:2], replace(qwen[2], session_generation=3)),
        )
        self.assertEqual(
            rig._phone_residency.load_count_by_session["HTP2"], 3
        )
        self.assertEqual(rig._helper_reconfigurations, {})
        self.assertEqual(
            rig._direct_phone_receipts[-1]["kind"],
            "helper_transition_rollback",
        )

    def test_failed_rollback_leaves_residency_unknown(self) -> None:
        qwen = tuple(shard("HTP" + str(i), QWEN, 1 << i) for i in range(3))
        gemma = shard("HTP2", GEMMA, 1 << 5, session_generation=2)
        receipt = DirectPhoneFfnReconfigurationReceipt(
            ticket_id="prep-ticket",
            previous_manifest_sha256="sha256:" + "1" * 64,
            target_manifest_sha256="sha256:" + "2" * 64,
            changed_session_id="HTP2",
            load_count=2,
            previous_load_count=1,
            column_quantum=32,
            previous_column_quantum=32,
            max_tokens=4,
            previous_max_tokens=2,
            previous_shard=qwen[2],
            target_shard=gemma,
            target_shards=(*qwen[:2], gemma),
        )
        direct_phone = SimpleNamespace(
            active=True,
            phone_shards=(*qwen[:2], gemma),
            rollback_reconfiguration=mock.Mock(
                side_effect=PhysicalAdapterError("phone unreachable")
            ),
        )
        rig = self.make_rig(
            direct_phone,
            _HelperReconfiguration(
                transition_id="replace-htp2",
                changed_session_ids=("HTP2",),
                receipt=receipt,
                previous_phone_residency=self.residency(qwen),
            ),
        )
        rig._phone_residency = self.residency((*qwen[:2], gemma))
        with self.assertRaisesRegex(PhysicalAdapterError, "unreachable"):
            rig.rollback_helper_transition(self.command())
        self.assertIsNone(rig._phone_residency)

    def test_rollback_of_fresh_start_stops_the_phone_session(self) -> None:
        rig = self.make_rig(
            SimpleNamespace(active=True),
            _HelperReconfiguration(
                transition_id="replace-htp2",
                changed_session_ids=("HTP0", "HTP1", "HTP2"),
                receipt=None,
                previous_phone_residency=None,
                fresh_start=True,
            ),
        )
        rig._phone_residency = object()
        rig._phone_executor_id = "phone-helper"
        rig._phone_parameters = {"x": 1}
        with mock.patch.object(rig, "_stop_phone_session") as stop:
            result = rig.rollback_helper_transition(self.command())
        stop.assert_called_once_with(
            terminate_phone_session=True,
            allow_incomplete_direct_phone=True,
        )
        self.assertTrue(result["physical_change"])
        self.assertIsNone(rig._phone_residency)
        self.assertIsNone(rig._phone_executor_id)

    def test_rollback_without_physical_change_is_a_noop_receipt(self) -> None:
        rig = self.make_rig(
            SimpleNamespace(active=True),
            _HelperReconfiguration(
                transition_id="replace-htp2",
                changed_session_ids=("HTP2",),
                receipt=None,
                previous_phone_residency=None,
            ),
        )
        result = rig.rollback_helper_transition(self.command())
        self.assertFalse(result["physical_change"])

    def test_rollback_requires_a_matching_reconfiguration(self) -> None:
        rig = self.make_rig(SimpleNamespace(active=True), None)
        rig._helper_reconfigurations = {}
        with self.assertRaisesRegex(
            PhysicalAdapterError, "no reconfiguration receipt"
        ):
            rig.rollback_helper_transition(self.command())


class _CompensationScheduler:
    def __init__(self, *, fail_completion: bool) -> None:
        self.fail_completion = fail_completion
        self.completions = 0
        self.failures: list[dict[str, object]] = []
        self.begin_calls = 0
        self.ticket = SimpleNamespace(
            request=SimpleNamespace(
                request_id="gemma-request", arrival_us=0
            ),
            dispatch_state="ACQUIRED",
            ticket_id="gemma-request:attempt:0",
        )

    def runtime_ticket(self, _request_id):
        return self.ticket

    def runtime_background_helper_preparation_allowed(self, *_a, **_k):
        return True

    def runtime_request_helper_preparation_envelope(self, *_a, **_k):
        # A rolled-back proposal is replaced by a fresh generation.
        generation = 2 + len(self.failures)
        return SimpleNamespace(
            phone_layout_generation=generation,
            phone_layout_geometry_sha256="sha256:" + "3" * 64,
            operator_plan_sha256="sha256:" + "4" * 64,
            preparation_ticket_id=(
                lambda _ticket: "phone-helper-layout-" + str(generation)
            ),
        )

    def begin_request_helper_preparation(self, *_a, **_k):
        self.begin_calls += 1
        return {"status": "OWNER"}

    def check_request_helper_preparation(self, *_a, **_k):
        return None

    def complete_request_helper_preparation(self, *_a, **_k):
        self.completions += 1
        if self.fail_completion:
            raise ValueError("helper envelope changed sessions are invalid")

    def fail_request_helper_preparation(self, request_id, ticket_id, **kw):
        self.failures.append({"request_id": request_id, **kw})
        self.ticket = SimpleNamespace(
            request=self.ticket.request,
            dispatch_state="COMPLETED",
            ticket_id=self.ticket.ticket_id,
        )


class _CompensationBackend:
    def __init__(self, *, rollback_error: Exception | None = None) -> None:
        self.applied = []
        self.rolled_back = []
        self.rollback_error = rollback_error

    def apply_transition(self, command, _payload, control_check):
        control_check()
        self.applied.append(command)
        return RawTransitionObservation(
            started_us=1_000, finished_us=1_001, status="COMPLETED"
        )

    def rollback_transition(self, command):
        self.rolled_back.append(command)
        if self.rollback_error is not None:
            raise self.rollback_error
        return {
            "physical_change": True,
            "restored_session_generations": {"HTP2": 3},
        }


def _transition_command():
    transition = SimpleNamespace(
        transition_id="replace-htp2",
        changed_phone_session_ids=("HTP2",),
        phone_shards=("HTP0", "HTP1", "HTP2"),
        evictions=(),
    )
    return SimpleNamespace(
        ticket_id="phone-helper-layout-2",
        transition=transition,
        participant=SimpleNamespace(executor_id="phone", endpoint="x"),
        helper_only=True,
    )


class TransactionalPreparationTests(unittest.TestCase):
    def run_worker(self, scheduler, backend, *, injector=None):
        adapter = object.__new__(CanonicalPhysicalAdapter)
        adapter._scheduler = scheduler
        adapter._backend = backend
        adapter._epoch_ns = time.monotonic_ns()
        adapter._helper_preparation_lock = threading.Lock()
        adapter._helper_preparation_watchers = set()
        adapter._helper_preparation_rejections = []
        adapter._started_helper_preparations = set()
        adapter._helper_preparations = []
        adapter._fault_injector = injector
        adapter._snapshot = lambda _ticket, at_us: SimpleNamespace(
            captured_at_us=at_us
        )
        helper_command = SimpleNamespace(
            helper_envelope=None,
            helper_transitions=(_transition_command(),),
            adapter_parameters={},
        )
        receipt = SimpleNamespace(
            transition_id="replace-htp2",
            status="COMPLETED",
            finished_us=1_001,
        )
        with mock.patch(
            "research_dev.scheduler.adapters.runtime."
            "interpret_runtime_ticket",
            return_value=helper_command,
        ), mock.patch(
            "research_dev.scheduler.adapters.runtime."
            "bind_ready_helper_to_physical_command",
            return_value=helper_command,
        ), mock.patch(
            "research_dev.scheduler.adapters.runtime."
            "transition_receipt_from_observation",
            return_value=receipt,
        ):
            handle = adapter._start_helper_preparation(
                scheduler.ticket, helper_command, object()
            )
            self.assertIsNotNone(handle)
            handle.thread.join(timeout=2.0)
        self.assertFalse(handle.thread.is_alive())

    def test_ready_refresh_keeps_watching_pending_preload_stages(self) -> None:
        class ProgressiveScheduler(_CompensationScheduler):
            refreshed = False

            def model_placement_controller_stats(self):
                return {"target_phone_layout_state": "PROPOSED"}

            def runtime_ready_helper_refresh_needed(self, *_a, **_k):
                return self.completions == 1 and not self.refreshed

            def refresh_ready_request_helper(self, *_a, **_k):
                self.refreshed = True
                return True

            def runtime_request_helper_preparation_envelope(self, *_a, **_k):
                helper = super().runtime_request_helper_preparation_envelope()
                helper.phone_layout_generation += self.completions
                return helper

            def complete_request_helper_preparation(self, *_a, **_k):
                self.completions += 1
                if self.completions == 3:
                    self.ticket.dispatch_state = "COMPLETED"

        scheduler = ProgressiveScheduler(fail_completion=False)
        backend = _CompensationBackend()
        self.run_worker(scheduler, backend)
        self.assertTrue(scheduler.refreshed)
        self.assertEqual(len(backend.applied), 3)
        self.assertFalse(backend.rolled_back)
        self.assertFalse(scheduler.failures)

    def test_pending_desktop_does_not_delay_next_preload_stage(self) -> None:
        class PendingDesktopScheduler(_CompensationScheduler):
            refresh_attempts = 0

            def runtime_ready_helper_refresh_needed(self, *_a, **_k):
                return self.completions > 0

            def refresh_ready_request_helper(self, *_a, **_k):
                self.refresh_attempts += 1
                return False

            def runtime_request_helper_preparation_envelope(self, *_a, **_k):
                helper = super().runtime_request_helper_preparation_envelope()
                helper.phone_layout_generation += self.completions
                return helper

            def complete_request_helper_preparation(self, *_a, **_k):
                self.completions += 1
                if self.completions == 3:
                    self.ticket.dispatch_state = "COMPLETED"

        scheduler = PendingDesktopScheduler(fail_completion=False)
        scheduler.ticket.transition_status = "PENDING"
        backend = _CompensationBackend()
        self.run_worker(scheduler, backend)
        self.assertGreater(scheduler.refresh_attempts, 0)
        self.assertEqual(scheduler.ticket.transition_status, "PENDING")
        self.assertEqual(len(backend.applied), 3)
        self.assertFalse(backend.rolled_back)
        self.assertFalse(scheduler.failures)

    def test_commit_failure_rolls_back_physically_then_logically(self) -> None:
        scheduler = _CompensationScheduler(fail_completion=True)
        backend = _CompensationBackend()
        self.run_worker(scheduler, backend)
        self.assertEqual(len(backend.applied), 1)
        self.assertEqual(len(backend.rolled_back), 1)
        self.assertEqual(len(scheduler.failures), 1)
        failure = scheduler.failures[0]
        self.assertNotIn("unavailable_session_ids", failure)
        self.assertIn(
            "changed sessions are invalid", str(failure["reason"])
        )

    def test_failed_rollback_marks_session_unavailable(self) -> None:
        scheduler = _CompensationScheduler(fail_completion=True)
        backend = _CompensationBackend(
            rollback_error=PhysicalAdapterError("phone unreachable")
        )
        self.run_worker(scheduler, backend)
        self.assertEqual(
            scheduler.failures[0]["unavailable_session_ids"], ("HTP2",)
        )

    def test_backend_without_rollback_marks_session_unavailable(self) -> None:
        scheduler = _CompensationScheduler(fail_completion=True)
        backend = _CompensationBackend()
        backend.rollback_transition = None
        self.run_worker(scheduler, backend)
        self.assertEqual(
            scheduler.failures[0]["unavailable_session_ids"], ("HTP2",)
        )

    def test_successful_commit_does_not_roll_back(self) -> None:
        scheduler = _CompensationScheduler(fail_completion=False)
        original_complete = scheduler.complete_request_helper_preparation

        def complete(*args, **kwargs):
            original_complete(*args, **kwargs)
            scheduler.ticket = SimpleNamespace(
                request=scheduler.ticket.request,
                dispatch_state="COMPLETED",
                ticket_id=scheduler.ticket.ticket_id,
            )

        scheduler.complete_request_helper_preparation = complete
        backend = _CompensationBackend()
        self.run_worker(scheduler, backend)
        self.assertEqual(scheduler.completions, 1)
        self.assertEqual(backend.rolled_back, [])
        self.assertEqual(scheduler.failures, [])

    def test_post_load_fault_is_injected_exactly_once(self) -> None:
        injector = HelperPreparationFaultInjector("post-load-once")
        scheduler = _CompensationScheduler(fail_completion=False)
        backend = _CompensationBackend()
        original_fail = scheduler.fail_request_helper_preparation

        def fail(request_id, ticket_id, **kw):
            original_fail(request_id, ticket_id, **kw)
            # The retry re-arms the same ticket instead of completing it.
            scheduler.ticket = SimpleNamespace(
                request=scheduler.ticket.request,
                dispatch_state="ACQUIRED",
                ticket_id=scheduler.ticket.ticket_id + "r",
            )
            adapter_started.clear()

        adapter_started = set()
        scheduler.fail_request_helper_preparation = fail
        original_complete = scheduler.complete_request_helper_preparation

        def complete(*args, **kwargs):
            original_complete(*args, **kwargs)
            scheduler.ticket = SimpleNamespace(
                request=scheduler.ticket.request,
                dispatch_state="COMPLETED",
                ticket_id=scheduler.ticket.ticket_id,
            )

        scheduler.complete_request_helper_preparation = complete
        self.run_worker(scheduler, backend, injector=injector)
        self.assertEqual(len(injector.records), 1)
        self.assertEqual(injector.records[0]["point"], "post-load")
        self.assertEqual(len(backend.rolled_back), 1)
        self.assertEqual(len(scheduler.failures), 1)
        self.assertIn(
            "injected_helper_preparation_fault", scheduler.failures[0]["reason"]
        )
        self.assertEqual(scheduler.completions, 1)
        self.assertEqual(len(backend.applied), 2)

    def test_invalid_injection_mode_is_rejected(self) -> None:
        with self.assertRaises(PhysicalAdapterError):
            HelperPreparationFaultInjector("random")

    def test_fault_skips_the_initial_whole_layout_load(self) -> None:
        injector = HelperPreparationFaultInjector("post-load-once")
        whole = SimpleNamespace(transition=SimpleNamespace(
            changed_phone_session_ids=("HTP0", "HTP1", "HTP2"),
            phone_shards=("HTP0", "HTP1", "HTP2"),
        ))
        partial = SimpleNamespace(transition=SimpleNamespace(
            changed_phone_session_ids=("HTP2",),
            phone_shards=("HTP0", "HTP1", "HTP2"),
        ))
        self.assertFalse(injector.is_partial_replacement([whole]))
        self.assertTrue(injector.is_partial_replacement([partial]))
        injector.check("post-load", partial_replacement=False)
        self.assertEqual(injector.records, ())
        with self.assertRaises(PhysicalAdapterError):
            injector.check("post-load", partial_replacement=True)
        self.assertEqual(len(injector.records), 1)


if __name__ == "__main__":
    unittest.main()
