#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from research_dev.scheduler import (
    HeterogeneousRuntimeSnapshot,
    ModelManifest,
    PhoneFfnShardStorageMetadata,
    RuntimeCapabilityCatalog,
    RuntimeExecutionReceipt,
    UnifiedScheduler,
)
from research_dev.scheduler._internal.model_placement_controller import (
    ModelPlacementController,
)
from research_dev.scheduler._internal.capacity import DeviceMemoryCapacity
from research_dev.scheduler._internal.adaptive_decode_contracts import (
    AdaptiveDecodeConfig,
    AdaptiveDecodePolicyAck,
    AdaptiveDecodeRawWindowObservation,
)
from research_dev.scheduler._internal.offline_phone_residency import (
    select_offline_resident_superset,
    verify_offline_phone_layout,
)
from research_dev.scheduler._internal.phone_shards import (
    PhoneFfnResidencyDemand,
    generate_mixed_ffn_residency_layouts,
    progressive_ffn_residency_layouts,
)
from research_dev.scheduler._internal.runtime_capabilities import (
    PhoneSessionResidencyObservation,
)
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler.config import PhoneResidentModelReprovisioningConfiguration
from research_dev.scheduler.adapters import (
    CanonicalOfflinePhoneResidencyPreloader,
    RawTransitionObservation,
    interpret_runtime_ticket,
    transition_receipt_from_observation,
)
from research_dev.scheduler.adapters.ticket import (
    bind_ready_helper_to_physical_command,
    validate_phone_session_replacement_command,
)

try:
    from .test_automated_runtime import runtime_snapshot
    from .test_multi_session_phone import manifest, session
    from . import test_replay_determinism as replay_fixture
except ImportError:
    from test_automated_runtime import runtime_snapshot
    from test_multi_session_phone import manifest, session
    import test_replay_determinism as replay_fixture


SHA_PROJECTION = "sha256:" + "9" * 64


def _hide_phone_telemetry(snapshot, phone_executor_id):
    return replace(
        snapshot,
        executors={
            **snapshot.executors,
            phone_executor_id: replace(
                snapshot.executors[phone_executor_id],
                battery_ppm=0,
                temperature_millic=100_000,
                thermal_qualified=None,
            ),
        },
        memory=replace(
            snapshot.memory,
            capacities={
                **snapshot.memory.capacities,
                "op15-ram": DeviceMemoryCapacity(
                    "op15-ram", 10_000_000_000,
                    10_000_000_000, 0,
                ),
            },
        ),
    )


class _OfflinePhoneBackend:
    def __init__(self, snapshot: HeterogeneousRuntimeSnapshot) -> None:
        self.source_snapshot = snapshot
        self.phone_shards = {}
        self.commands = []
        self.source_by_ticket = {}
        self.load_count_by_session = {}
        self.clock_us = snapshot.captured_at_us
        self.fail_next_snapshot = False

    def apply_transition(self, command, _payload, _control_check):
        validate_phone_session_replacement_command(command)
        self.source_by_ticket[command.ticket_id] = dict(self.phone_shards)
        self.phone_shards = {
            row.session_id: row
            for row in command.transition.phone_shards
        }
        for session_id in command.transition.changed_phone_session_ids:
            self.load_count_by_session[session_id] = (
                self.load_count_by_session.get(session_id, 0) + 1
            )
        self.commands.append(command)
        started_us = self.clock_us
        self.clock_us += 10
        return RawTransitionObservation(
            started_us=started_us,
            finished_us=self.clock_us,
            status="COMPLETED",
            evicted_artifact_sha256s=tuple(sorted({
                row.artifact_sha256
                for row in command.transition.evictions
            })),
        )

    def rollback_transition(self, command):
        source = self.source_by_ticket[command.ticket_id]
        changed = tuple(command.transition.changed_phone_session_ids)
        target = {
            row.session_id: row
            for row in command.transition.phone_shards
        }
        restored_empty = tuple(
            session_id for session_id in changed
            if session_id not in source
        )
        restored_generations = {}
        restored = dict(source)
        for session_id in changed:
            if session_id not in source:
                self.load_count_by_session.pop(session_id, None)
                continue
            restored_generation = (
                target[session_id].session_generation + 1
            )
            restored[session_id] = replace(
                source[session_id],
                session_generation=restored_generation,
            )
            restored_generations[session_id] = restored_generation
            self.load_count_by_session[session_id] = (
                self.load_count_by_session.get(session_id, 0) + 1
            )
        self.phone_shards = restored
        return {
            "physical_change": True,
            "restored_empty_session_ids": restored_empty,
            "restored_session_generations": restored_generations,
        }

    def snapshot(self, stage, at_us):
        if self.fail_next_snapshot:
            self.fail_next_snapshot = False
            raise RuntimeError("injected post-load verification failure")
        snapshot = replay_fixture.ReplayDeterminismTests._retime_snapshot(
            self.source_snapshot,
            max(at_us, self.clock_us),
            "offline-phone-snapshot-" + str(len(self.commands)),
        )
        phone_device_id = (
            stage.helper_envelope.helper_plan.execution_contract
                .phone_device_id
        )
        return replace(
            snapshot,
            executors={
                key: replace(
                    value,
                    healthy=True,
                    ready=True,
                    free_slots=max(1, value.free_slots),
                    busy_until_us=0,
                    temperature_millic=max(1, value.temperature_millic),
                    battery_ppm=max(1, value.battery_ppm),
                    thermal_qualified=True,
                )
                for key, value in snapshot.executors.items()
            },
            links={
                key: replace(value, ready=True, busy_until_us=0)
                for key, value in snapshot.links.items()
            },
            phone_session_residency=tuple(
                PhoneSessionResidencyObservation(
                    session_id=row.session_id,
                    device_id=phone_device_id,
                    executor_id=(
                        stage.helper_envelope.helper_binding.executor_id
                    ),
                    endpoint=row.endpoint,
                    artifact_sha256=row.artifact_sha256,
                    resident_geometry_sha256=(
                        row.resident_geometry_sha256
                    ),
                    operator_plan_sha256=row.operator_plan_sha256,
                    session_generation=row.session_generation,
                    resident_bytes=row.resident_bytes,
                )
                for row in sorted(
                    self.phone_shards.values(),
                    key=lambda value: value.session_id,
                )
            ),
        )


class OfflinePhoneResidencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        replay_fixture.ReplayDeterminismTests.setUpClass()

    @staticmethod
    def replay_runtime(*, load_evidence: bool = True):
        case = next(
            row for row in replay_fixture.ReplayDeterminismTests.fixture["cases"]
            if row["case_id"] == "session_cow_gate_v3"
        )
        replay = replay_fixture.ReplayDeterminismTests(methodName="runTest")
        if load_evidence:
            scheduler = replay._new_scheduler(case)
        else:
            scheduler = UnifiedScheduler.for_runtime_discovery(
                "enforce",
                maximum_phone_sessions=case["maximum_phone_sessions"],
            )
            scheduler.register_runtime_capabilities(
                RuntimeCapabilityCatalog.from_json(case["runtime_catalog"])
            )
            for value in replay_fixture.ReplayDeterminismTests.fixture[
                "common"
            ]["model_manifests"]:
                scheduler.register_model_manifest(
                    ModelManifest.from_json(value)
                )
        requests_by_model = {}
        for row in case["requests"]:
            if not row["model_id"].startswith("qwen"):
                continue
            requests_by_model.setdefault(row["model_id"], []).append(
                replay._request(row)
            )
        snapshot = HeterogeneousRuntimeSnapshot.from_json(
            case["requests"][0]["snapshot"]
        )
        return scheduler, requests_by_model, snapshot

    def test_offline_learning_preload_preserves_route_qualification(self) -> None:
        scheduler, requests_by_model, snapshot = self.replay_runtime(
            load_evidence=False
        )

        plan = scheduler.plan_offline_phone_residency(
            requests_by_model,
            snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )

        artifact = next(iter(plan.request_by_artifact))
        self.assertEqual(len(plan.target_layout.shards), 3)
        self.assertEqual(
            plan.metadata["route_evidence_state_by_artifact"][artifact],
            "LEARNING",
        )
        self.assertEqual(
            plan.metadata["demand_selection_rule"],
            "measured-queue-energy-first;"
            "learning-rough-compute-only-without-measured-demand",
        )
        self.assertEqual(
            plan.target_layout.objective_kind,
            "queue_rough_compute_ops",
        )
        self.assertGreater(plan.target_layout.transition_cost, 0)
        self.assertIn(
            "ROUTE_NOT_QUALIFIED",
            plan.current_stage.helper_envelope.helper_binding
                .eligibility_reasons,
        )
        backend = _OfflinePhoneBackend(snapshot)
        preloader = CanonicalOfflinePhoneResidencyPreloader(
            scheduler,
            backend,
            epoch_ns=(
                time.monotonic_ns() - snapshot.captured_at_us * 1_000
            ),
            snapshot_provider=backend.snapshot,
        )
        first = preloader.execute_next_stage(
            plan.plan_id,
            object(),
            snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        self.assertEqual(first.plan.current_stage.state, "READY")
        self.assertTrue(all(
            shard.session_generation > 0
            for command in first.commands
            for shard in command.transition.phone_shards
        ))

    def test_deferred_preload_uses_refreshed_snapshot_capture_time(
        self,
    ) -> None:
        scheduler, requests_by_model, snapshot = self.replay_runtime()
        plan = scheduler.plan_offline_phone_residency(
            requests_by_model,
            snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        backend = _OfflinePhoneBackend(snapshot)
        original_begin = scheduler.begin_offline_phone_residency_stage
        begin_count = 0

        def defer_once(*args, **kwargs):
            nonlocal begin_count
            begin_count += 1
            if begin_count == 1:
                return {"status": "DEFERRED"}
            return original_begin(*args, **kwargs)

        scheduler.begin_offline_phone_residency_stage = defer_once

        def captured_later(stage, observed_at_us):
            backend.clock_us = max(
                backend.clock_us, observed_at_us + 1
            )
            return backend.snapshot(stage, observed_at_us + 1)

        preloader = CanonicalOfflinePhoneResidencyPreloader(
            scheduler,
            backend,
            epoch_ns=(
                time.monotonic_ns() - snapshot.captured_at_us * 1_000
            ),
            snapshot_provider=captured_later,
            preparation_wait_timeout_s=1,
        )

        result = preloader.execute_next_stage(
            plan.plan_id,
            object(),
            snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )

        self.assertEqual(begin_count, 2)
        self.assertEqual(result.plan.current_stage.state, "READY")

    def test_progressive_preload_uses_each_snapshot_capture_time(self) -> None:
        scheduler, requests_by_model, snapshot = self.replay_runtime()
        plan = scheduler.plan_offline_phone_residency(
            requests_by_model, snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        backend = _OfflinePhoneBackend(snapshot)

        def captured_later(stage, observed_at_us):
            backend.clock_us = max(backend.clock_us, observed_at_us + 1)
            return backend.snapshot(stage, observed_at_us + 1)

        preloader = CanonicalOfflinePhoneResidencyPreloader(
            scheduler, backend,
            epoch_ns=time.monotonic_ns() - snapshot.captured_at_us * 1000,
            snapshot_provider=captured_later,
        )
        result = preloader.preload(
            plan, lambda _stage: object(), initial_snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        self.assertEqual(result.plan.state, "READY")
        self.assertEqual(len(result.commands), 3)
        self.assertTrue(all(stage.state == "READY" for stage in result.plan.stages))
        self.assertEqual(set(backend.load_count_by_session.values()), {1})

    def test_expired_preload_snapshot_still_rejects_before_loading(self) -> None:
        scheduler, requests_by_model, snapshot = self.replay_runtime()
        plan = scheduler.plan_offline_phone_residency(
            requests_by_model, snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        backend = _OfflinePhoneBackend(snapshot)
        preloader = CanonicalOfflinePhoneResidencyPreloader(
            scheduler, backend,
            epoch_ns=time.monotonic_ns() - snapshot.captured_at_us * 1000,
            snapshot_provider=backend.snapshot,
        )
        with self.assertRaisesRegex(ValueError, "system snapshot is stale"):
            preloader.execute_next_stage(
                plan.plan_id, object(), snapshot=snapshot,
                observed_at_us=snapshot.valid_until_us,
            )
        self.assertEqual(backend.commands, [])
        self.assertEqual(scheduler.offline_phone_residency_stage(plan.plan_id).state, "PROPOSED")

    def test_desktop_baseline_cannot_prepare_a_proposed_phone_layout(self) -> None:
        scheduler, requests_by_model, snapshot = self.replay_runtime(
            load_evidence=False
        )
        model_id, requests = next(iter(requests_by_model.items()))
        request = replace(
            requests[0],
            arrival_us=snapshot.captured_at_us,
            deadline_us=max(
                requests[0].deadline_us,
                snapshot.captured_at_us + 600_000_000,
            ),
        )
        submitted = scheduler.submit_automated_request(
            request,
            model_id,
            snapshot,
            observed_at_us=snapshot.captured_at_us,
            selection_mode="desktop-baseline",
        )
        target = scheduler._model_placement_controller.target_phone_layout()
        self.assertIsNotNone(target)
        self.assertEqual(target.state, "PROPOSED")
        for acquired in (False, True):
            with self.subTest(acquired=acquired):
                ticket = (
                    scheduler.wait_runtime_request(request.request_id, 0)
                    if acquired else submitted
                )
                self.assertFalse(
                    scheduler.runtime_background_helper_preparation_allowed(
                        request.request_id,
                        expected_ticket_id=ticket.ticket_id,
                    )
                )
                self.assertIsNone(
                    scheduler.runtime_request_helper_preparation_envelope(
                        request.request_id,
                        expected_ticket_id=ticket.ticket_id,
                        observed_at_us=snapshot.captured_at_us,
                        snapshot=snapshot,
                    )
                )
        self.assertFalse(scheduler.request_helper_events())
        self.assertEqual(target.state, "PROPOSED")

    def test_online_learning_demand_can_authorize_initial_layout(self) -> None:
        scheduler, requests_by_model, snapshot = self.replay_runtime(
            load_evidence=False
        )
        model_id, requests = next(iter(requests_by_model.items()))
        request = replace(
            requests[0],
            arrival_us=snapshot.captured_at_us,
            deadline_us=max(
                requests[0].deadline_us,
                snapshot.captured_at_us + 600_000_000,
            ),
        )

        candidates = scheduler._generate_automated_candidate_set(
            request,
            scheduler.runtime_model_manifest(model_id),
            snapshot,
            snapshot.captured_at_us,
            use_residency_holds=False,
        )
        target = (
            scheduler._model_placement_controller.target_phone_layout()
        )

        self.assertIsNotNone(target)
        self.assertEqual(target.state, "PROPOSED")
        self.assertEqual(target.generation, 1)
        self.assertEqual(
            target.layout.objective_kind, "queue_rough_compute_ops"
        )
        self.assertEqual(len(target.layout.shards), 3)
        self.assertEqual(
            target.layout.changed_session_ids,
            ("HTP0", "HTP1", "HTP2"),
        )
        self.assertTrue(any(
            row.evidence_state == "LEARNING"
            for row in scheduler._compact_helper_opportunities(
                candidates, request
            )
        ))
        self.assertTrue(any(
            "ROUTE_NOT_QUALIFIED" in row.rejection_reasons
            for row in candidates.candidates
        ))
        self.assertEqual(
            scheduler.phone_residency_events()[-1]["reason"],
            "PHONE_RESIDENCY_LEARNING_EXPLORATION",
        )
        submitted = scheduler.submit_automated_request(
            request,
            model_id,
            snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        self.assertTrue(
            scheduler.runtime_background_helper_preparation_allowed(
                request.request_id,
                expected_ticket_id=submitted.ticket_id,
            )
        )
        ticket = scheduler.wait_runtime_request(request.request_id, 0)
        if ticket.transition_status == "PENDING":
            command = interpret_runtime_ticket(ticket)
            receipts = tuple(
                transition_receipt_from_observation(
                    transition,
                    RawTransitionObservation(
                        started_us=ticket.decision.start_us,
                        finished_us=ticket.decision.start_us + 1,
                        status="COMPLETED",
                        evicted_artifact_sha256s=tuple(sorted({
                            row.artifact_sha256
                            for row in transition.transition.evictions
                        })),
                    ),
                )
                for transition in command.transitions
            )
            scheduler.record_automated_transition_receipts(
                request.request_id, receipts
            )
        ticket = scheduler.runtime_execution_ticket(request.request_id)
        helper = scheduler.runtime_request_helper_preparation_envelope(
            request.request_id,
            expected_ticket_id=ticket.ticket_id,
            observed_at_us=snapshot.captured_at_us,
            snapshot=snapshot,
        )
        self.assertIsNotNone(helper)
        self.assertEqual(
            tuple(
                shard.session_id
                for shard in helper.helper_plan.execution_contract
                    .phone_shards
            ),
            ("HTP0", "HTP1", "HTP2"),
        )
        self.assertEqual(
            scheduler._request_helper_opportunities[
                request.request_id
            ][0].evidence_state,
            "LEARNING",
        )

    def test_partial_storage_never_generates_an_unbacked_session(self) -> None:
        scheduler, requests_by_model, snapshot = self.replay_runtime(load_evidence=False)
        model_id, requests = next(iter(requests_by_model.items()))
        model = scheduler.runtime_model_manifest(model_id)
        stored = PhoneFfnShardStorageMetadata(
            parent_artifact_sha256=model.artifact_sha256,
            shard_sha256=canonical_sha256([model_id, "single-shard"]),
            path="/phone/single.ffn.gguf", layer_mask=63,
            maximum_columns=model.feed_forward_length, session_id="HTP0",
        )
        scheduler.register_phone_ffn_shard_storage((stored,))
        scheduler.register_phone_ffn_shard_storage((stored,))
        with self.assertRaisesRegex(Exception, "references undiscovered sessions"):
            scheduler.register_phone_ffn_shard_storage((replace(stored, session_id="missing"),))
        request = replace(requests[0], arrival_us=snapshot.captured_at_us,
                          deadline_us=snapshot.captured_at_us + 600_000_000)
        candidates = scheduler.generate_automated_candidates(
            request, model_id, snapshot, observed_at_us=snapshot.captured_at_us,
        )
        shards = [shard for row in candidates.candidates
                  for shard in row.plan.execution_contract.phone_shards]
        self.assertTrue(shards)
        self.assertEqual({row.session_id for row in shards}, {stored.session_id})
        self.assertTrue(all(row.layer_mask & ~stored.layer_mask == 0 for row in shards))
        self.assertTrue(all(row.maximum_columns <= stored.maximum_columns for row in shards))
        target = scheduler._model_placement_controller.target_phone_layout()
        self.assertIsNotNone(target)
        self.assertEqual(tuple(row.session_id for row in target.layout.shards), (stored.session_id,))

    def test_online_shard_preparation_publishes_a_usable_subset(self) -> None:
        scheduler, requests_by_model, snapshot = self.replay_runtime(
            load_evidence=False
        )
        model_id, requests = next(iter(requests_by_model.items()))
        model = scheduler.runtime_model_manifest(model_id)
        scheduler.register_phone_ffn_shard_storage(tuple(
            PhoneFfnShardStorageMetadata(
                parent_artifact_sha256=model.artifact_sha256,
                shard_sha256=canonical_sha256([model_id, index]),
                path="/phone/HTP" + str(index) + ".gguf",
                layer_mask=63 << (6 * index),
                maximum_columns=model.feed_forward_length,
                session_id="HTP" + str(index),
            )
            for index in range(3)
        ))
        request = replace(
            requests[0], arrival_us=snapshot.captured_at_us,
            deadline_us=snapshot.captured_at_us + 600_000_000,
        )
        ticket = self.acquire_desktop_request(
            scheduler, request, model_id, snapshot, selection_mode="calibration"
        )
        dormant = json.loads(ticket.execution_plan.adapter_parameters[
            "dormant_phone_ffn_runtime_v1"
        ])
        self.assertEqual(dormant["ffn_resident_layer_mask"], (1 << 18) - 1)
        backend = _OfflinePhoneBackend(snapshot)
        for index in range(3):
            target = scheduler._model_placement_controller.target_phone_layout()
            self.assertEqual(len(target.layout.shards), index + 1)
            helper = scheduler.runtime_request_helper_preparation_envelope(
                request.request_id, expected_ticket_id=ticket.ticket_id,
                observed_at_us=snapshot.captured_at_us, snapshot=snapshot,
            )
            self.assertIsNotNone(helper)
            decision = scheduler.begin_request_helper_preparation(
                request.request_id, observed_at_us=snapshot.captured_at_us,
                snapshot=snapshot,
                expected_phone_layout_generation=helper.phone_layout_generation,
                expected_phone_layout_geometry_sha256=helper.phone_layout_geometry_sha256,
                expected_operator_plan_sha256=helper.operator_plan_sha256,
            )
            self.assertEqual(decision["status"], "OWNER")
            command = interpret_runtime_ticket(ticket)
            command = bind_ready_helper_to_physical_command(
                replace(command, helper_envelope=None, helper_transitions=()), helper
            )
            receipts = tuple(
                transition_receipt_from_observation(
                    row, backend.apply_transition(row, object(), lambda: None)
                )
                for row in command.helper_transitions
            )
            snapshot = backend.snapshot(
                SimpleNamespace(helper_envelope=helper), backend.clock_us
            )
            completed = scheduler.complete_request_helper_preparation(
                request.request_id, decision["preparation_ticket_id"], receipts,
                snapshot=snapshot,
            )
            self.assertEqual(completed["state"], "READY")
            ready = scheduler._model_placement_controller.ready_phone_layout()
            self.assertEqual(len(ready.layout.shards), index + 1)
            self.assertTrue(scheduler.refresh_ready_request_helper(
                request.request_id, expected_ticket_id=ticket.ticket_id,
                observed_at_us=snapshot.captured_at_us, snapshot=snapshot,
            ))
            ticket = scheduler.runtime_execution_ticket(request.request_id)
            envelope = scheduler._late_request_helper_contexts[
                request.request_id
            ].helper
            self.assertIsNotNone(envelope)
            shards = envelope.helper_plan.execution_contract.phone_shards
            self.assertEqual(len(shards), index + 1)
            self.assertTrue(all(
                row.session_generation == 1
                for row in shards
            ))
        self.assertEqual(sorted(backend.load_count_by_session.values()), [1, 1, 1])

        pending = replace(ticket, transition_status="PENDING", transition_receipts=())
        with patch.object(
            scheduler._runtime_controller, "current_tickets", return_value=(pending,)
        ):
            self.assertEqual(scheduler._rematerialize_ready_layout_helpers(
                ready, snapshot, snapshot.captured_at_us
            ), ())

    def test_online_learning_accepts_missing_marginal_energy_evidence(
        self,
    ) -> None:
        scheduler, requests_by_model, snapshot = self.replay_runtime(
            load_evidence=False
        )
        model_id, requests = next(iter(requests_by_model.items()))
        request = replace(
            requests[0],
            arrival_us=snapshot.captured_at_us,
            deadline_us=max(
                requests[0].deadline_us,
                snapshot.captured_at_us + 600_000_000,
            ),
        )
        manifest = scheduler.runtime_model_manifest(model_id)
        candidates = scheduler._generate_automated_candidate_set(
            request,
            manifest,
            snapshot,
            snapshot.captured_at_us,
            use_residency_holds=False,
            update_phone_residency_portfolio=False,
        )
        candidates = replace(
            candidates,
            candidates=tuple(
                replace(
                    row,
                    rejection_reasons=tuple(sorted({
                        *row.rejection_reasons,
                        "ROUTE_MARGINAL_ENERGY_EVIDENCE_ABSENT",
                    })),
                )
                if (
                    row.plan.execution_contract.phone_device_id is not None
                    and not {
                        "MEMORY_CAPACITY",
                        "TRANSPORT_PROFILE_INCOMPLETE",
                    }.intersection(row.rejection_reasons)
                ) else row
                for row in candidates.candidates
            ),
        )
        helper_route_ids = {
            row.helper_operator_plan.route_id
            for row in scheduler._compact_helper_opportunities(
                candidates, request
            )
            if row.evidence_state == "LEARNING"
        }

        learning = scheduler._learning_phone_demand(
            candidates,
            request,
            manifest,
            request.output_tokens,
        )

        self.assertIsNotNone(learning)
        self.assertTrue(helper_route_ids)
        self.assertEqual(learning.status["evidence_state"], "LEARNING")
        self.assertIn(
            "ROUTE_MARGINAL_ENERGY_EVIDENCE_ABSENT",
            next(iter(learning.status["candidate_rejections"].values())),
        )
        self.assertTrue(all(
            not row.admitted
            for row in candidates.candidates
            if row.candidate_id in helper_route_ids
        ))

    def test_queued_demand_learning_tolerates_unknown_marginal_system_cost(
        self,
    ) -> None:
        """s1a: while Gemma 000 ran, the queued Qwen 001 switch's phone routes carried
        MARGINAL_SYSTEM_COST_UNKNOWN, so Qwen had no learning demand, was not phone-capable
        and the phone followed it only once its desktop load finished (+417 s)."""
        scheduler, requests_by_model, snapshot = self.replay_runtime(
            load_evidence=False
        )
        model_id, requests = next(iter(requests_by_model.items()))
        request = replace(
            requests[0], arrival_us=snapshot.captured_at_us,
            deadline_us=snapshot.captured_at_us + 600_000_000,
        )
        manifest = scheduler.runtime_model_manifest(model_id)
        generated = scheduler._generate_automated_candidate_set(
            request, manifest, snapshot, snapshot.captured_at_us,
            use_residency_holds=False, update_phone_residency_portfolio=False,
        )
        protected = replace(generated, candidates=tuple(
            replace(row, rejection_reasons=tuple(sorted({
                *row.rejection_reasons, "MARGINAL_SYSTEM_COST_UNKNOWN",
            })))
            if row.plan.execution_contract.phone_device_id is not None
            and not {"MEMORY_CAPACITY", "TRANSPORT_PROFILE_INCOMPLETE"}.intersection(
                row.rejection_reasons
            ) else row
            for row in generated.candidates
        ))

        def learned():
            return scheduler._learning_phone_demand(
                protected, request, manifest, request.output_tokens
            )

        self.assertIsNone(learned())
        scheduler.configure_phone_resident_model_reprovisioning(
            PhoneResidentModelReprovisioningConfiguration(early_on_transition=True)
        )
        self.assertIsNone(learned())
        scheduler.configure_phone_resident_model_reprovisioning(
            PhoneResidentModelReprovisioningConfiguration(count_queued_demand=True)
        )
        learning = learned()
        self.assertIsNotNone(learning)
        self.assertEqual(learning.status["evidence_state"], "LEARNING")
        self.assertIn(
            "MARGINAL_SYSTEM_COST_UNKNOWN",
            next(iter(learning.status["candidate_rejections"].values())),
        )
        # Other physical rejections still refuse the learning demand.
        broken = replace(protected, candidates=tuple(
            replace(row, rejection_reasons=tuple(sorted({
                *row.rejection_reasons, "MEMORY_CAPACITY",
            }))) if row.rejection_reasons else row
            for row in protected.candidates
        ))
        self.assertIsNone(scheduler._learning_phone_demand(
            broken, request, manifest, request.output_tokens
        ))

    def test_online_learning_replacement_survives_boundary_reevaluation(
        self,
    ) -> None:
        scheduler, requests_by_model, snapshot = self.replay_runtime(
            load_evidence=False
        )
        qwen_model_id, qwen_requests = next(iter(requests_by_model.items()))
        qwen = replace(
            qwen_requests[0],
            arrival_us=snapshot.captured_at_us,
            deadline_us=snapshot.captured_at_us + 1_000_000_000,
        )
        scheduler._generate_automated_candidate_set(
            qwen,
            scheduler.runtime_model_manifest(qwen_model_id),
            snapshot,
            snapshot.captured_at_us,
            use_residency_holds=False,
        )
        proposed = scheduler._model_placement_controller.target_phone_layout()
        self.assertIsNotNone(proposed)
        projection = canonical_sha256({"test": "learning-qqq"})
        scheduler._model_placement_controller.begin_phone_layout_transition(
            proposed.generation,
            ticket_id="learning-qqq",
            transition_ids=("learning-qqq",),
            ready_at_us=snapshot.captured_at_us + 2,
            projection_token_sha256=projection,
            workspace_bytes=proposed.workspace_bytes,
            observed_at_us=snapshot.captured_at_us + 1,
        )
        ready = (
            scheduler._model_placement_controller
            .complete_phone_layout_transition(
                generation=proposed.generation,
                ticket_id="learning-qqq",
                transition_ids=("learning-qqq",),
                geometry_sha256=proposed.layout.geometry_sha256,
                projection_token_sha256=projection,
                finished_at_us=snapshot.captured_at_us + 2,
            )
        )
        scheduler._publish_phone_layout_view(ready.layout, True, {})

        case = next(
            row for row in replay_fixture.ReplayDeterminismTests.fixture[
                "cases"
            ]
            if row["case_id"] == "session_cow_gate_v3"
        )
        gemma_row = next(
            row for row in case["requests"]
            if row["model_id"].startswith("gemma")
        )
        replay = replay_fixture.ReplayDeterminismTests(methodName="runTest")
        minimum_residency = (
            scheduler._model_placement_controller.policy
            .phone_minimum_residency_us
        )
        observed_at_us = (
            snapshot.captured_at_us + minimum_residency + 10
        )
        gemma_snapshot = replay._retime_snapshot(
            HeterogeneousRuntimeSnapshot.from_json(gemma_row["snapshot"]),
            observed_at_us,
            "online-learning-replacement",
        )
        gemma_snapshot = replace(
            gemma_snapshot,
            executors={
                key: replace(
                    value,
                    healthy=True,
                    ready=True,
                    free_slots=max(1, value.free_slots),
                    busy_until_us=0,
                    temperature_millic=35_000,
                    battery_ppm=800_000,
                    thermal_qualified=True,
                )
                for key, value in gemma_snapshot.executors.items()
            },
            links={
                key: replace(value, ready=True, busy_until_us=0)
                for key, value in gemma_snapshot.links.items()
            },
        )
        gemma = replace(
            replay._request(gemma_row),
            arrival_us=observed_at_us,
            deadline_us=observed_at_us + 1_000_000_000,
        )
        gemma_manifest = scheduler.runtime_model_manifest(
            gemma_row["model_id"]
        )
        scheduler._generate_automated_candidate_set(
            gemma,
            gemma_manifest,
            gemma_snapshot,
            observed_at_us,
            use_residency_holds=False,
            update_phone_residency_portfolio=False,
        )

        for index, remaining in enumerate((292, 291, 290, 289)):
            scheduler._update_phone_residency_portfolio(
                replace(gemma, output_tokens=remaining),
                gemma_manifest,
                observed_at_us + index + 1,
                gemma_snapshot if index == 0 else None,
            )

        replacement = (
            scheduler._model_placement_controller.target_phone_layout()
        )
        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.generation, 2)
        self.assertEqual(len(replacement.layout.changed_session_ids), 1)
        selected = replacement.layout.changed_session_ids[0]
        retained = {"HTP0", "HTP1", "HTP2"} - {selected}
        self.assertEqual(
            dict(replacement.layout.session_generation_by_id),
            {
                session_id: 2 if session_id == selected else 1
                for session_id in ("HTP0", "HTP1", "HTP2")
            },
        )
        self.assertTrue(all(
            scheduler._model_placement_controller
                .phone_session_state(session_id).state == "READY"
            for session_id in retained
        ))
        self.assertEqual(
            scheduler._model_placement_controller.ready_phone_layout()
                .generation,
            1,
        )

    @staticmethod
    def layouts():
        model = manifest()
        sessions = tuple(session(index) for index in range(3))
        candidates = generate_mixed_ffn_residency_layouts(
            (PhoneFfnResidencyDemand(
                model,
                1_000,
                model.feed_forward_length,
                "coalesced-batch",
            ),),
            sessions,
            phone_wide_limit_bytes=sum(
                row.resident_memory_limit_bytes for row in sessions
            ),
            transition_energy_uj_by_session={
                row.session_id: 1 for row in sessions
            },
        )
        return model, sessions, candidates

    @staticmethod
    def acquire_desktop_request(
        scheduler,
        request,
        model_id,
        snapshot,
        *,
        selection_mode="energy-aware",
    ):
        ticket = scheduler.submit_automated_request(
            request,
            model_id,
            snapshot,
            observed_at_us=snapshot.captured_at_us,
            selection_mode=selection_mode,
        )
        ticket = scheduler.wait_runtime_request(request.request_id, 0)
        if ticket.transition_status == "PENDING":
            command = interpret_runtime_ticket(ticket)
            receipts = tuple(
                transition_receipt_from_observation(
                    transition,
                    RawTransitionObservation(
                        started_us=ticket.decision.start_us,
                        finished_us=ticket.decision.start_us + 1,
                        status="COMPLETED",
                        evicted_artifact_sha256s=tuple(sorted({
                            row.artifact_sha256
                            for row in transition.transition.evictions
                        })),
                    ),
                )
                for transition in command.transitions
            )
            ticket = scheduler.record_automated_transition_receipts(
                request.request_id, receipts
            )
        return scheduler.runtime_execution_ticket(request.request_id)

    def test_superset_is_published_one_verified_session_at_a_time(self) -> None:
        _model, sessions, candidates = self.layouts()
        target = select_offline_resident_superset(candidates)
        stages = progressive_ffn_residency_layouts(target)

        self.assertEqual(len(target.shards), 3)
        self.assertEqual(tuple(len(row.shards) for row in stages), (1, 2, 3))
        self.assertTrue(all(len(row.changed_session_ids) == 1 for row in stages))

        controller = ModelPlacementController()
        controller.register_empty_phone_sessions(
            {row.session_id: row.endpoint for row in sessions},
            observed_at_us=0,
        )
        previous_sessions = set()
        for index, structural in enumerate(stages):
            proposed = controller.propose_phone_layout(
                structural,
                workspace_bytes=0,
                shared_compute_resource_id="phone-htp",
                shared_transport_resource_ids=(
                    "phone-functionfs", "phone-usb",
                ),
                observed_at_us=index * 10,
                selection_reason="OFFLINE_RESIDENT_SUPERSET",
                queue_work_by_artifact=(
                    structural.queued_work_by_artifact
                ),
                queue_benefit_uj=structural.queue_benefit,
                transition_cost_uj=structural.transition_cost,
                switching_margin_uj=0,
                minimum_residency_us=0,
                force=True,
            )
            changed = proposed.layout.changed_session_ids
            controller.begin_phone_layout_transition(
                proposed.generation,
                ticket_id="offline-stage-" + str(index),
                transition_ids=("load-stage-" + str(index),),
                ready_at_us=index * 10 + 5,
                projection_token_sha256=SHA_PROJECTION,
                workspace_bytes=0,
                observed_at_us=index * 10,
            )
            if previous_sessions:
                self.assertEqual(
                    controller.ready_phone_layout().state,
                    "READY",
                )
            for retained in previous_sessions:
                self.assertEqual(
                    controller.phone_session_state(retained).state,
                    "READY",
                )
            ready = controller.complete_phone_layout_transition(
                generation=proposed.generation,
                ticket_id="offline-stage-" + str(index),
                transition_ids=("load-stage-" + str(index),),
                geometry_sha256=proposed.layout.geometry_sha256,
                projection_token_sha256=SHA_PROJECTION,
                finished_at_us=index * 10 + 5,
            )
            self.assertIsNotNone(ready)
            self.assertEqual(
                controller.phone_session_state(changed[0]).state,
                "READY",
            )
            previous_sessions.add(changed[0])

        self.assertEqual(
            {row.state for row in controller.phone_session_states()},
            {"READY"},
        )
        self.assertEqual(
            dict(controller.ready_phone_layout().layout.session_generation_by_id),
            {row.session_id: 1 for row in sessions},
        )

    def test_online_superset_advances_only_after_individual_readiness(self) -> None:
        _model, sessions, candidates = self.layouts()
        target = select_offline_resident_superset(candidates)
        controller = ModelPlacementController()
        controller.register_empty_phone_sessions(
            {row.session_id: row.endpoint for row in sessions},
            observed_at_us=0,
        )
        first = controller.propose_phone_layout(
            target,
            workspace_bytes=0,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-usb",),
            observed_at_us=0,
            progressive=True,
        )
        checkpoint = controller.checkpoint()
        retained = {}
        for index in range(3):
            proposed = controller.target_phone_layout()
            self.assertEqual(len(proposed.layout.shards), index + 1)
            self.assertEqual(len(proposed.layout.changed_session_ids), 1)
            selected = proposed.layout.changed_session_ids[0]
            controller.begin_phone_layout_transition(
                proposed.generation,
                ticket_id="stage-" + str(index),
                transition_ids=("load-" + str(index),),
                ready_at_us=index * 10 + 5,
                projection_token_sha256=SHA_PROJECTION,
                workspace_bytes=0,
                observed_at_us=index * 10,
            )
            self.assertEqual(controller.phone_session_state(selected).state, "LOADING")
            for session_id, state in retained.items():
                self.assertEqual(controller.phone_session_state(session_id), state)
            controller.complete_phone_layout_transition(
                generation=proposed.generation,
                ticket_id="stage-" + str(index),
                transition_ids=("load-" + str(index),),
                geometry_sha256=proposed.layout.geometry_sha256,
                projection_token_sha256=SHA_PROJECTION,
                finished_at_us=index * 10 + 5,
            )
            retained[selected] = controller.phone_session_state(selected)
            self.assertEqual(retained[selected].state, "READY")
            self.assertEqual(retained[selected].session_generation, 1)
            self.assertEqual(controller.phone_preload_inflight(), index < 2)
        self.assertEqual(
            controller.ready_phone_layout().layout.geometry_sha256,
            target.geometry_sha256,
        )
        controller.restore(checkpoint)
        self.assertEqual(controller.target_phone_layout(), first)
        self.assertTrue(controller.phone_preload_inflight())
        self.assertIsNone(controller.ready_phone_layout())

    def test_failed_progressive_add_keeps_the_published_subset(self) -> None:
        _model, _sessions, candidates = self.layouts()
        controller = ModelPlacementController()
        first = controller.propose_phone_layout(
            select_offline_resident_superset(candidates),
            workspace_bytes=0,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-usb",),
            observed_at_us=0,
            progressive=True,
        )
        controller.begin_phone_layout_transition(
            first.generation, ticket_id="first", transition_ids=("first",),
            ready_at_us=1, projection_token_sha256=SHA_PROJECTION,
            workspace_bytes=0, observed_at_us=0,
        )
        ready = controller.complete_phone_layout_transition(
            generation=first.generation, ticket_id="first",
            transition_ids=("first",),
            geometry_sha256=first.layout.geometry_sha256,
            projection_token_sha256=SHA_PROJECTION, finished_at_us=1,
        )
        retained_id = ready.layout.shards[0].session_id
        retained = controller.phone_session_state(retained_id)
        second = controller.target_phone_layout()
        controller.begin_phone_layout_transition(
            second.generation, ticket_id="second", transition_ids=("second",),
            ready_at_us=3, projection_token_sha256=SHA_PROJECTION,
            workspace_bytes=0, observed_at_us=2,
        )
        controller.fail_phone_layout_transition(
            "second", generation=second.generation,
            projection_token_sha256=SHA_PROJECTION, failed_at_us=3,
            reason="injected-progressive-load-failure",
            unavailable_session_ids=second.layout.changed_session_ids,
        )
        self.assertEqual(controller.phone_session_state(retained_id), retained)
        self.assertEqual(controller.ready_phone_layout(), ready)
        self.assertFalse(controller.phone_preload_inflight())
        self.assertIsNone(controller.target_phone_layout())

    def test_failed_add_preserves_ready_session_and_shared_inference(self) -> None:
        _model, sessions, candidates = self.layouts()
        stages = progressive_ffn_residency_layouts(
            select_offline_resident_superset(candidates)
        )
        controller = ModelPlacementController()
        controller.register_empty_phone_sessions(
            {row.session_id: row.endpoint for row in sessions},
            observed_at_us=0,
        )
        first = controller.propose_phone_layout(
            stages[0],
            workspace_bytes=0,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-usb",),
            observed_at_us=0,
            minimum_residency_us=0,
            force=True,
        )
        controller.begin_phone_layout_transition(
            first.generation,
            ticket_id="first",
            transition_ids=("load-first",),
            ready_at_us=1,
            projection_token_sha256=SHA_PROJECTION,
            workspace_bytes=0,
            observed_at_us=0,
        )
        ready = controller.complete_phone_layout_transition(
            generation=first.generation,
            ticket_id="first",
            transition_ids=("load-first",),
            geometry_sha256=first.layout.geometry_sha256,
            projection_token_sha256=SHA_PROJECTION,
            finished_at_us=1,
        )
        second = controller.propose_phone_layout(
            stages[1],
            workspace_bytes=0,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-usb",),
            observed_at_us=2,
            minimum_residency_us=0,
            force=True,
        )
        self.assertEqual(
            UnifiedScheduler._copy_on_write_preparation_yielding_resources(
                ready, second
            ),
            ("phone-htp", "phone-usb"),
        )
        controller.begin_phone_layout_transition(
            second.generation,
            ticket_id="second",
            transition_ids=("load-second",),
            ready_at_us=3,
            projection_token_sha256=SHA_PROJECTION,
            workspace_bytes=0,
            observed_at_us=2,
        )
        retained = next(iter({row.session_id for row in ready.layout.shards}))
        changed = second.layout.changed_session_ids[0]
        self.assertEqual(controller.phone_session_state(retained).state, "READY")
        controller.fail_phone_layout_transition(
            "second",
            generation=second.generation,
            projection_token_sha256=SHA_PROJECTION,
            failed_at_us=3,
            reason="injected-offline-load-failure",
            unavailable_session_ids=(changed,),
        )
        self.assertEqual(controller.phone_session_state(retained).state, "READY")
        self.assertEqual(
            controller.phone_session_state(changed).state, "UNAVAILABLE"
        )
        self.assertEqual(
            controller.ready_phone_layout().layout.geometry_sha256,
            ready.layout.geometry_sha256,
        )

    def test_exact_physical_map_verification(self) -> None:
        model, _sessions, candidates = self.layouts()
        structural = select_offline_resident_superset(candidates)
        controller = ModelPlacementController()
        proposed = controller.propose_phone_layout(
            structural,
            workspace_bytes=0,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-usb",),
            observed_at_us=0,
            minimum_residency_us=0,
            force=True,
        )
        observations = tuple(
            PhoneSessionResidencyObservation(
                session_id=shard.session_id,
                device_id="phone-a",
                executor_id="executor:phone-a",
                endpoint=shard.endpoint,
                artifact_sha256=shard.artifact_sha256,
                resident_geometry_sha256=(
                    shard.resident_geometry_sha256
                ),
                operator_plan_sha256=shard.operator_plan_sha256,
                session_generation=(
                    proposed.layout.session_generation_by_id[
                        shard.session_id
                    ]
                ),
                resident_bytes=shard.resident_bytes,
            )
            for shard in proposed.layout.shards
        )
        snapshot = replace(
            runtime_snapshot(model),
            phone_session_residency=observations,
        )
        proof = verify_offline_phone_layout(
            proposed,
            snapshot,
            phone_device_id="phone-a",
            executor_id="executor:phone-a",
        )
        self.assertTrue(proof.startswith("sha256:"))
        with self.assertRaisesRegex(
            ValueError, "physical session identity differs"
        ):
            verify_offline_phone_layout(
                proposed,
                replace(
                    snapshot,
                    phone_session_residency=(
                        replace(observations[0], resident_bytes=1),
                        *observations[1:],
                    ),
                ),
                phone_device_id="phone-a",
                executor_id="executor:phone-a",
            )

    def test_scheduler_preloads_three_sessions_and_restart_reuses_them(
        self,
    ) -> None:
        scheduler, requests_by_model, snapshot = self.replay_runtime()
        plan = scheduler.plan_offline_phone_residency(
            requests_by_model,
            snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        backend = _OfflinePhoneBackend(snapshot)
        preloader = CanonicalOfflinePhoneResidencyPreloader(
            scheduler,
            backend,
            epoch_ns=(
                time.monotonic_ns() - snapshot.captured_at_us * 1_000
            ),
            snapshot_provider=backend.snapshot,
        )
        result = preloader.preload(
            plan,
            lambda _stage: object(),
            initial_snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )

        self.assertEqual(result.plan.state, "READY")
        self.assertEqual(len(result.plan.target_layout.shards), 3)
        self.assertEqual(len(result.commands), 3)
        self.assertEqual(
            tuple(
                command.transition.changed_phone_session_ids
                for command in result.commands
            ),
            tuple(
                (stage.selected_session_id,)
                for stage in result.plan.stages
            ),
        )
        self.assertEqual(
            {
                row.session_id: result.plan.target_layout
                    .session_generation_by_id[row.session_id]
                for row in result.plan.target_layout.shards
            },
            {row.session_id: 1 for row in result.plan.target_layout.shards},
        )
        self.assertTrue(all(
            row.session_generation > 0
            for command in result.commands
            for row in command.transition.phone_shards
        ))
        self.assertEqual(
            {
                row["state"]
                for row in scheduler.phone_residency_session_states()
            },
            {"READY"},
        )

        hot_snapshot = backend.snapshot(
            result.plan.current_stage,
            result.plan.finished_at_us,
        )
        final_stage = result.plan.current_stage
        self.assertIsNotNone(final_stage)
        helper = final_stage.helper_envelope
        rebound_helper = replace(
            helper,
            desktop_parent_route_id=(
                helper.desktop_parent_route_id + ":pre-restart"
            ),
        )
        persisted_plan = replace(
            result.plan,
            stages=(
                *result.plan.stages[:-1],
                replace(final_stage, helper_envelope=rebound_helper),
            ),
        )
        restarted, _requests, _initial = self.replay_runtime()
        adopted = restarted.adopt_offline_phone_residency(
            persisted_plan,
            requests_by_model,
            snapshot=hot_snapshot,
            observed_at_us=hot_snapshot.captured_at_us,
        )
        self.assertEqual(adopted.state, "READY")
        self.assertEqual(len(backend.commands), 3)
        self.assertEqual(
            dict(
                restarted._model_placement_controller.ready_phone_layout()
                    .layout.session_generation_by_id
            ),
            dict(result.plan.target_layout.session_generation_by_id),
        )

        hidden = _hide_phone_telemetry(hot_snapshot, next(
            row.executor_id
            for row in restarted._runtime_capabilities.executors
            if row.phone_sessions
        ))
        model_id, qwen_rows = next(iter(requests_by_model.items()))
        qwen_request = qwen_rows[0]
        qwen_artifact = restarted.runtime_model_manifest(
            model_id
        ).artifact_sha256
        lookup = {
            "artifact_sha256": qwen_artifact,
            "desktop_parent_route_id": helper.desktop_parent_route_id,
            "desktop_placement_sha256": helper.desktop_placement_sha256,
            "baseline_executor_id": (
                helper.helper_plan.baseline_executor_id
            ),
        }
        self.assertIsNone(
            restarted._authoritative_ready_helper_template(**lookup)
        )
        self.assertEqual(
            restarted._authoritative_ready_helper_template(
                **lookup,
                allow_parent_route_rebind=True,
            ),
            restarted._reusable_phone_helper_template(rebound_helper),
        )
        ticket = self.acquire_desktop_request(
            restarted, qwen_request, model_id, hot_snapshot
        )
        dormant = json.loads(ticket.execution_plan.adapter_parameters[
            "dormant_phone_ffn_runtime_v1"
        ])
        self.assertEqual(
            dormant["ffn_resident_layer_mask"],
            sum(
                row.layer_mask for row in result.plan.target_layout.shards
                if row.artifact_sha256 == qwen_artifact
            ),
        )
        self.assertTrue(restarted.refresh_ready_request_helper(
            qwen_request.request_id,
            expected_ticket_id=ticket.ticket_id,
            observed_at_us=hidden.captured_at_us,
            snapshot=hidden,
        ))
        attachment = restarted._model_placement_controller\
            .request_binding(qwen_request.request_id)["helper_attachment"]
        self.assertEqual(
            attachment["allowed_session_ids"],
            ["HTP0", "HTP1", "HTP2"],
        )
        self.assertIn(
            "ATTACHED",
            tuple(
                row["kind"] for row in restarted.request_helper_events()
                if row["request_id"] == qwen_request.request_id
            ),
        )
        restarted.cancel_runtime_request(
            qwen_request.request_id,
            hidden.captured_at_us + 1,
            "replacement planning fixture",
        )
        case = next(
            row for row in replay_fixture.ReplayDeterminismTests.fixture[
                "cases"
            ]
            if row["case_id"] == "session_cow_gate_v3"
        )
        replay = replay_fixture.ReplayDeterminismTests(
            methodName="runTest"
        )
        gemma_requests = {}
        for row in case["requests"]:
            if row["model_id"].startswith("gemma"):
                gemma_requests.setdefault(row["model_id"], []).append(
                    replay._request(row)
                )
        replacement = restarted.plan_offline_phone_residency(
            gemma_requests,
            snapshot=hidden,
            observed_at_us=hidden.captured_at_us,
        )
        selected = replacement.current_stage.selected_session_id
        gemma_artifact = next(iter(replacement.request_by_artifact))
        artifacts = [
            row.artifact_sha256 for row in replacement.target_layout.shards
        ]
        self.assertEqual(
            replacement.current_stage.layout.layout.changed_session_ids,
            (selected,),
        )
        self.assertEqual(artifacts.count(qwen_artifact), 2)
        self.assertEqual(artifacts.count(gemma_artifact), 1)
        self.assertLessEqual(
            replacement.target_layout.resident_bytes,
            replacement.phone_wide_limit_bytes,
        )

    def test_completed_preparation_owner_does_not_own_ready_helper(
        self,
    ) -> None:
        scheduler, requests_by_model, snapshot = self.replay_runtime()
        model_id, source_requests = next(iter(requests_by_model.items()))
        owner = replace(
            source_requests[0],
            request_id="ready-layout-owner",
            arrival_us=snapshot.captured_at_us,
            deadline_us=snapshot.captured_at_us + 1_000_000_000,
            output_tokens=341,
        )
        plan = scheduler.plan_offline_phone_residency(
            {model_id: (owner,)},
            snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        owner_ticket = self.acquire_desktop_request(
            scheduler, owner, model_id, snapshot
        )
        backend = _OfflinePhoneBackend(snapshot)
        preloader = CanonicalOfflinePhoneResidencyPreloader(
            scheduler,
            backend,
            epoch_ns=(
                time.monotonic_ns() - snapshot.captured_at_us * 1_000
            ),
            snapshot_provider=backend.snapshot,
        )
        result = preloader.preload(
            plan,
            lambda _stage: object(),
            initial_snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        ready = scheduler._model_placement_controller.ready_phone_layout()
        self.assertIsNotNone(ready)
        load_count = len(backend.commands)
        hot = backend.snapshot(
            result.plan.current_stage, result.plan.finished_at_us + 1
        )
        phone_executor_id = next(
            row.executor_id
            for row in scheduler._runtime_capabilities.executors
            if row.phone_sessions
        )
        hot = _hide_phone_telemetry(hot, phone_executor_id)
        self.assertTrue(scheduler.refresh_ready_request_helper(
            owner.request_id,
            expected_ticket_id=owner_ticket.ticket_id,
            observed_at_us=hot.captured_at_us,
            snapshot=hot,
        ))
        scheduler.start_adaptive_decode(
            owner.request_id,
            slot_id=2,
            first_token_index=1,
            at_us=hot.captured_at_us + 1,
        )
        attached = scheduler._try_attach_ready_request_helper(
            owner_ticket,
            slot_id=2,
            token_index=1,
            at_us=hot.captured_at_us + 2,
            fraction_ppm=250_000,
        )
        self.assertTrue(attached.attached, attached)
        owner_binding = scheduler._model_placement_controller.request_binding(
            owner.request_id
        )
        owner_leases = tuple(
            owner_binding["helper_attachment"]["lease_tokens"]
        )
        self.assertTrue(owner_leases)
        scheduler._model_placement_controller.record_request_helper_work(
            owner.request_id, 1, observed_at_us=hot.captured_at_us + 3
        )
        finished_at_us = hot.captured_at_us + 4
        scheduler.complete_automated_request(
            owner.request_id,
            RuntimeExecutionReceipt(
                ticket_id=owner_ticket.ticket_id,
                request_id=owner.request_id,
                artifact_sha256=owner_ticket.model.artifact_sha256,
                operator_plan_sha256=(
                    owner_ticket.execution_plan.plan_sha256
                ),
                executor_id=owner_ticket.binding.executor_id,
                endpoint=owner_ticket.binding.endpoint,
                operator_plan_protocol=(
                    owner_ticket.binding.operator_plan_protocol
                ),
                participant_executor_ids=tuple(
                    row.executor_id
                    for row in owner_ticket.binding.participants
                ),
                started_us=owner_ticket.decision.start_us,
                finished_us=finished_at_us,
                output_sha256="sha256:" + "f" * 64,
                status="COMPLETED",
            ),
        )
        self.assertEqual(
            scheduler.runtime_ticket(owner.request_id).dispatch_state,
            "COMPLETED",
        )

        follower = replace(
            owner,
            request_id="ready-layout-follower",
            arrival_us=finished_at_us + 1,
            deadline_us=finished_at_us + 1_000_000_001,
        )
        follower_snapshot = replay_fixture.ReplayDeterminismTests\
            ._retime_snapshot(
                hot, follower.arrival_us, "ready-layout-follower"
            )
        follower_ticket = self.acquire_desktop_request(
            scheduler, follower, model_id, follower_snapshot
        )
        self.assertNotEqual(owner_ticket.ticket_id, follower_ticket.ticket_id)
        self.assertEqual(
            owner_ticket.execution_plan.desktop_placement_sha256,
            follower_ticket.execution_plan.desktop_placement_sha256,
        )
        self.assertTrue(scheduler.refresh_ready_request_helper(
            follower.request_id,
            expected_ticket_id=follower_ticket.ticket_id,
            observed_at_us=follower_snapshot.captured_at_us,
            snapshot=follower_snapshot,
        ))
        started = scheduler.start_adaptive_decode(
            follower.request_id,
            slot_id=3,
            first_token_index=1,
            at_us=follower_snapshot.captured_at_us + 1,
            config=AdaptiveDecodeConfig(
                minimum_remaining_tokens=4,
                minimum_window_tokens=2,
                maximum_window_tokens=4,
                maximum_probe_tokens=20,
                maximum_probe_candidates=2,
                measurement_resolution_us=1,
                transition_cost_us=1,
                transition_energy_uj=1,
                minimum_energy_saving_ppm=0,
                maximum_latency_ppm=10_000_000,
                uncertainty_ppm=0,
                exploration_latency_budget_ppm=1_000_000,
                exploration_energy_budget_ppm=1_000_000,
                warmup_windows_per_policy=0,
                allow_assumed_phone_power_for_operational_selection=True,
            ),
        )
        boundary = scheduler.adaptive_decode_boundary(
            follower.request_id,
            slot_id=3,
            token_index=started.target_token_index,
            at_us=follower_snapshot.captured_at_us + 10,
        ).boundary
        self.assertIsNotNone(boundary)
        directive = scheduler.record_adaptive_decode_window(
            follower.request_id,
            boundary,
            AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": 1_000_000},
                phone_compute_us=0,
                usb_transfer_us=0,
                rpc_us=0,
                exposed_tail_us=0,
                output_valid=True,
                evidence_ids=("ready-layout-reuse-baseline",),
                energy_boundary_id="ready-layout-reuse",
                energy_attribution_kind="isolated",
                desktop_compute_us=1_000_000,
            ),
        )
        self.assertIsNotNone(directive.control)
        self.assertGreater(directive.control.policy.split_fraction_ppm, 0)
        scheduler.acknowledge_adaptive_decode_control(
            follower.request_id,
            AdaptiveDecodePolicyAck(
                request_id=follower.request_id,
                slot_id=3,
                plan_generation=directive.control.plan_generation,
                applied_token_index=boundary.token_end,
                applied_at_us=follower_snapshot.captured_at_us + 11,
                policy_hash=directive.control.policy.policy_hash,
            ),
        )
        follower_binding = (
            scheduler._model_placement_controller.request_binding(
                follower.request_id
            )
        )
        follower_leases = tuple(
            follower_binding["helper_attachment"]["lease_tokens"]
        )
        self.assertTrue(follower_leases)
        self.assertTrue(set(owner_leases).isdisjoint(follower_leases))
        self.assertEqual(len(backend.commands), load_count)
        self.assertEqual(
            {
                row["session_id"]: row["session_generation"]
                for row in follower_binding["helper_attachment"]
                    ["phone_session_identities"]
            },
            {
                row.session_id: ready.layout.session_generation_by_id[
                    row.session_id
                ]
                for row in ready.layout.shards
                if row.artifact_sha256
                    == follower_ticket.model.artifact_sha256
            },
        )
        self.assertGreater(follower_binding["fraction_ppm"], 0)
        follower_events = tuple(
            row for row in scheduler.request_helper_events()
            if row["request_id"] == follower.request_id
        )
        self.assertTrue({
            "ELIGIBLE",
            "TEMPLATE_FOUND",
            "MATERIALIZED",
            "ATTACHED",
            "FRACTION_APPLIED",
        }.issubset({row["kind"] for row in follower_events}))
        self.assertFalse(any(
            "opportunity is not exact" in str(row.get("reason", ""))
            for row in follower_events
        ))

    def test_active_desktop_request_attaches_after_first_ready_session(
        self,
    ) -> None:
        scheduler, requests_by_model, snapshot = self.replay_runtime()
        plan = scheduler.plan_offline_phone_residency(
            requests_by_model,
            snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        model_id = next(iter(requests_by_model))
        request = requests_by_model[model_id][0]
        ticket = self.acquire_desktop_request(
            scheduler,
            request,
            model_id,
            snapshot,
            selection_mode="calibration",
        )
        self.assertEqual(
            ticket.execution_plan.execution_contract.execution_mode,
            "desktop",
        )
        self.assertIsNone(ticket.execution_plan.helper_envelope)
        self.assertEqual(plan.current_stage.state, "PROPOSED")

        backend = _OfflinePhoneBackend(snapshot)
        preloader = CanonicalOfflinePhoneResidencyPreloader(
            scheduler,
            backend,
            epoch_ns=(
                time.monotonic_ns() - snapshot.captured_at_us * 1_000
            ),
            snapshot_provider=backend.snapshot,
        )
        first = preloader.execute_next_stage(
            plan.plan_id,
            object(),
            snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        first_stage = first.plan.current_stage
        first_session = first_stage.selected_session_id
        events = scheduler._model_placement_controller\
            .request_helper_events(request.request_id)
        kinds = tuple(row["kind"] for row in events)
        all_events = scheduler._model_placement_controller\
            .request_helper_events()
        ready_event = next(
            row for row in all_events
            if row["kind"] == "OFFLINE_RESIDENCY_STAGE_READY"
            and row["request_id"] == first_stage.request_id
        )
        attached_event = next(
            row for row in events if row["kind"] == "ATTACHED"
        )
        self.assertLess(
            ready_event["event_index"], attached_event["event_index"]
        )
        self.assertIn("HELPER_REMATERIALIZED", kinds)
        self.assertEqual(
            scheduler._model_placement_controller
                .request_binding(request.request_id)["fraction_ppm"],
            0,
        )

        ready_at_us = first_stage.verified_at_us
        scheduler.start_adaptive_decode(
            request.request_id,
            slot_id=2,
            first_token_index=0,
            at_us=ready_at_us + 1,
        )
        attached = scheduler._try_attach_ready_request_helper(
            ticket,
            slot_id=2,
            token_index=1,
            at_us=ready_at_us + 2,
            fraction_ppm=250_000,
        )
        self.assertTrue(attached.attached, attached)
        scheduler._model_placement_controller.record_request_helper_work(
            request.request_id, 3, observed_at_us=ready_at_us + 3
        )

        hot = backend.snapshot(first_stage, ready_at_us + 4)
        phone_executor_id = next(
            row.executor_id
            for row in scheduler._runtime_capabilities.executors
            if row.phone_sessions
        )
        hot = _hide_phone_telemetry(hot, phone_executor_id)
        second = scheduler.next_offline_phone_residency_stage(
            plan.plan_id,
            snapshot=hot,
            observed_at_us=hot.captured_at_us,
        )
        self.assertNotEqual(
            second.current_stage.selected_session_id, first_session
        )
        self.assertGreater(
            second.current_stage.phone_safety_state.battery_ppm, 0
        )
        self.assertLess(
            second.current_stage.phone_safety_state.temperature_millic,
            100_000,
        )
        second_result = preloader.execute_next_stage(
            plan.plan_id,
            object(),
            snapshot=hot,
            observed_at_us=hot.captured_at_us,
        )
        expanded_at_us = (
            second_result.plan.current_stage.verified_at_us + 1
        )
        scheduler.adaptive_decode_boundary(
            request.request_id,
            slot_id=2,
            token_index=2,
            at_us=expanded_at_us,
        )
        binding = scheduler._model_placement_controller.request_binding(
            request.request_id
        )
        self.assertEqual(binding["fraction_ppm"], 250_000)
        self.assertEqual(
            binding["helper_attachment"]["completed_phone_calls"], 3
        )
        self.assertEqual(
            binding["helper_attachment"]["allowed_session_ids"],
            sorted({
                first_session,
                second_result.plan.current_stage.selected_session_id,
            }),
        )
        scheduler.adaptive_decode_boundary(
            request.request_id,
            slot_id=2,
            token_index=3,
            at_us=expanded_at_us + 1,
        )
        expanded_events = tuple(
            row for row in scheduler._model_placement_controller
                .request_helper_events(request.request_id)
            if row["kind"] == "HELPER_EXPANDED"
        )
        self.assertEqual(len(expanded_events), 1)
        self.assertEqual(
            scheduler._model_placement_controller
                .phone_session_state(first_session).state,
            "READY",
        )
        self.assertEqual(second_result.plan.state, "PARTIAL")
        self.assertIn(
            "HELPER_EXPANDED",
            tuple(
                row["kind"] for row in scheduler
                    ._model_placement_controller
                    .request_helper_events(request.request_id)
            ),
        )

        second_stage = second_result.plan.current_stage
        scheduler.cancel_runtime_request(
            request.request_id,
            second_stage.verified_at_us + 1,
            "desktop preparation owner completed",
        )
        hot = _hide_phone_telemetry(
            backend.snapshot(
                second_stage, second_stage.verified_at_us + 2
            ),
            phone_executor_id,
        )
        inactive_executor_ids = {
            second_stage.helper_envelope.helper_plan.baseline_executor_id,
            second_stage.helper_envelope.helper_binding.executor_id,
        }
        stored_plan = scheduler._offline_phone_residency_plans[plan.plan_id]
        materialization = stored_plan.materialization_snapshot
        scheduler._offline_phone_residency_plans[plan.plan_id] = replace(
            stored_plan,
            materialization_snapshot=replace(
                materialization,
                executors={
                    key: (
                        replace(value, ready=False, free_slots=0)
                        if key in inactive_executor_ids else value
                    )
                    for key, value in materialization.executors.items()
                },
                memory=replace(
                    materialization.memory,
                    capacities={
                        **materialization.memory.capacities,
                        "cuda0-vram": DeviceMemoryCapacity(
                            "cuda0-vram", 17_175_674_880,
                            3_999_268_864, 536_870_912,
                        ),
                    },
                ),
            ),
        )
        hot = replace(
            hot,
            executors={
                key: (
                    replace(value, ready=False, free_slots=0)
                    if key in inactive_executor_ids else value
                )
                for key, value in hot.executors.items()
            },
            memory=replace(
                hot.memory,
                capacities={
                    **hot.memory.capacities,
                    "cuda0-vram": DeviceMemoryCapacity(
                        "cuda0-vram", 17_175_674_880,
                        16_638_803_968, 536_870_912,
                    ),
                },
            ),
        )
        third = scheduler.next_offline_phone_residency_stage(
            plan.plan_id,
            snapshot=hot,
            observed_at_us=hot.captured_at_us,
        )
        third_result = preloader.execute_next_stage(
            plan.plan_id,
            object(),
            snapshot=hot,
            observed_at_us=hot.captured_at_us,
        )
        self.assertEqual(third.current_stage.state, "PROPOSED")
        self.assertEqual(third_result.plan.state, "READY")
        self.assertEqual(
            {
                stage.selected_session_id
                for stage in third_result.plan.stages
            },
            {"HTP0", "HTP1", "HTP2"},
        )

    def test_ready_during_decode_starts_bounded_learning_probe(self) -> None:
        scheduler, requests_by_model, snapshot = self.replay_runtime()
        scheduler._adaptive_decode_config = AdaptiveDecodeConfig(
            minimum_remaining_tokens=16,
            minimum_window_tokens=32,
            maximum_window_tokens=128,
            maximum_probe_tokens=1_024,
            maximum_probe_candidates=4,
            measurement_resolution_us=5_000_000,
            transition_cost_us=2_000,
            transition_energy_uj=20_000,
            minimum_energy_saving_ppm=0,
            maximum_latency_ppm=10_000_000,
            uncertainty_ppm=0,
            exploration_latency_budget_ppm=1_000_000,
            exploration_energy_budget_ppm=1_000_000,
            warmup_windows_per_policy=0,
            allow_assumed_phone_power_for_operational_selection=True,
            coarse_probe_fractions_ppm=(
                250_000, 500_000, 750_000, 1_000_000
            ),
        )
        model_id, original_requests = next(iter(requests_by_model.items()))
        request = replace(
            original_requests[0],
            output_tokens=341,
            deadline_us=max(
                original_requests[0].deadline_us,
                snapshot.captured_at_us + 1_000_000_000,
            ),
        )
        plan = scheduler.plan_offline_phone_residency(
            {model_id: (request,)},
            snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        self.acquire_desktop_request(
            scheduler, request, model_id, snapshot
        )
        started = scheduler.start_adaptive_decode(
            request.request_id,
            slot_id=2,
            first_token_index=1,
            at_us=snapshot.captured_at_us + 1,
        )
        self.assertGreater(started.target_token_index, 2)

        backend = _OfflinePhoneBackend(snapshot)
        preloader = CanonicalOfflinePhoneResidencyPreloader(
            scheduler,
            backend,
            epoch_ns=(
                time.monotonic_ns() - snapshot.captured_at_us * 1_000
            ),
            snapshot_provider=backend.snapshot,
        )
        first = preloader.execute_next_stage(
            plan.plan_id,
            object(),
            snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        self.assertEqual(first.plan.current_stage.state, "READY")

        ready_boundary = scheduler.adaptive_decode_boundary(
            request.request_id,
            slot_id=2,
            token_index=2,
            at_us=snapshot.captured_at_us + 2_000_001,
        )
        self.assertIsNotNone(ready_boundary.boundary)
        self.assertEqual(ready_boundary.boundary.token_end, 2)
        self.assertLess(
            ready_boundary.boundary.token_end,
            started.target_token_index,
        )
        control = scheduler.record_adaptive_decode_window(
            request.request_id,
            ready_boundary.boundary,
            AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": 1_000_000},
                phone_compute_us=0,
                usb_transfer_us=0,
                rpc_us=0,
                exposed_tail_us=0,
                output_valid=True,
                evidence_ids=("bounded-learning-probe-test",),
                energy_boundary_id="bounded-learning-probe-test",
                energy_attribution_kind="isolated",
                desktop_compute_us=2_000_000,
            ),
        )
        self.assertIsNotNone(control.control)
        self.assertEqual(control.control.policy.split_fraction_ppm, 250_000)
        self.assertEqual(
            scheduler._model_placement_controller
                .request_binding(request.request_id)["fraction_ppm"],
            250_000,
        )
        selected = tuple(
            row for row in scheduler._model_placement_controller
                .request_helper_events(request.request_id)
            if row["kind"] == "WINDOW_LEASE_SELECTED"
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            selected[0]["selection_kind"],
            "BOUNDED_LEARNING_EXPLORATION",
        )

        acknowledged_at = snapshot.captured_at_us + 2_000_002
        opened = scheduler.acknowledge_adaptive_decode_control(
            request.request_id,
            AdaptiveDecodePolicyAck(
                request_id=request.request_id,
                slot_id=2,
                plan_generation=control.control.plan_generation,
                applied_token_index=2,
                applied_at_us=acknowledged_at,
                policy_hash=control.control.policy.policy_hash,
            ),
        )
        finished_at = acknowledged_at + 32_000_000
        phone_boundary = scheduler.adaptive_decode_boundary(
            request.request_id,
            slot_id=2,
            token_index=opened.target_token_index,
            at_us=finished_at,
        ).boundary
        next_probe = scheduler.record_adaptive_decode_window(
            request.request_id,
            phone_boundary,
            AdaptiveDecodeRawWindowObservation(
                fleet_energy_uj_by_domain={"fleet": 16_000_000},
                phone_compute_us=1_000_000,
                usb_transfer_us=100_000,
                rpc_us=100_000,
                exposed_tail_us=0,
                output_valid=True,
                evidence_ids=("bounded-learning-probe-test",),
                energy_boundary_id="bounded-learning-probe-test",
                energy_attribution_kind="isolated",
                desktop_compute_us=30_000_000,
                completed_phone_calls=32,
                completed_phone_input_rows=32,
            ),
        )
        self.assertIsNotNone(next_probe.control)
        self.assertEqual(next_probe.control.policy.split_fraction_ppm, 500_000)
        self.assertTrue(scheduler._adaptive_decode.snapshot(
            request.request_id
        )["helper_available"])
        selected = tuple(
            row for row in scheduler._model_placement_controller
                .request_helper_events(request.request_id)
            if row["kind"] == "WINDOW_LEASE_SELECTED"
        )
        self.assertEqual(selected[-1]["requested_bid"]["phone_window_count"], 1)
        self.assertEqual(
            selected[-1]["requested_bid"]["requested_fraction_ppm"], 500_000
        )

    def test_post_load_failure_restores_only_the_added_session(self) -> None:
        scheduler, requests_by_model, snapshot = self.replay_runtime()
        plan = scheduler.plan_offline_phone_residency(
            requests_by_model,
            snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        backend = _OfflinePhoneBackend(snapshot)
        preloader = CanonicalOfflinePhoneResidencyPreloader(
            scheduler,
            backend,
            epoch_ns=(
                time.monotonic_ns() - snapshot.captured_at_us * 1_000
            ),
            snapshot_provider=backend.snapshot,
        )
        first = preloader.execute_next_stage(
            plan.plan_id,
            object(),
            snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        first_ready = first.plan.current_stage
        next_snapshot = backend.snapshot(
            first_ready, first_ready.verified_at_us
        )
        plan = scheduler.next_offline_phone_residency_stage(
            plan.plan_id,
            snapshot=next_snapshot,
            observed_at_us=next_snapshot.captured_at_us,
        )
        retained = set(backend.phone_shards)
        backend.fail_next_snapshot = True
        with self.assertRaisesRegex(
            Exception, "physical offline phone preparation failed"
        ):
            preloader.execute_next_stage(
                plan.plan_id,
                object(),
                snapshot=next_snapshot,
                observed_at_us=next_snapshot.captured_at_us,
            )
        self.assertEqual(set(backend.phone_shards), retained)
        self.assertEqual(
            {
                row["session_id"]
                for row in scheduler.phone_residency_session_states()
                if row["state"] == "READY"
            },
            retained,
        )
        self.assertEqual(
            scheduler.offline_phone_residency_snapshot(plan.plan_id)[
                "state"
            ],
            "PARTIAL",
        )

    def test_offline_replacement_respects_deployed_shard_masks(self) -> None:
        scheduler, qwen_requests, snapshot = self.replay_runtime()
        case = next(
            row for row in replay_fixture.ReplayDeterminismTests.fixture["cases"]
            if row["case_id"] == "session_cow_gate_v3"
        )
        replay = replay_fixture.ReplayDeterminismTests(methodName="runTest")
        gemma_row = next(
            row for row in case["requests"]
            if row["model_id"].startswith("gemma")
        )
        qwen_id = next(iter(qwen_requests))
        gemma_id = gemma_row["model_id"]
        storage = tuple(
            PhoneFfnShardStorageMetadata(
                parent_artifact_sha256=model.artifact_sha256,
                shard_sha256=canonical_sha256([model_id, index]),
                path="/phone/" + model_id + "/HTP" + str(index) + ".gguf",
                layer_mask=((1 << count) - 1) << (count * index),
                maximum_columns=model.feed_forward_length,
                session_id="HTP" + str(index),
            )
            for model_id, count in ((qwen_id, 6), (gemma_id, 8))
            for model in (scheduler.runtime_model_manifest(model_id),)
            for index in range(3)
        )
        scheduler.register_phone_ffn_shard_storage(storage)
        backend = _OfflinePhoneBackend(snapshot)
        preloader = CanonicalOfflinePhoneResidencyPreloader(
            scheduler,
            backend,
            epoch_ns=time.monotonic_ns() - snapshot.captured_at_us * 1_000,
            snapshot_provider=backend.snapshot,
        )
        initial = scheduler.plan_offline_phone_residency(
            qwen_requests,
            snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        ready = preloader.preload(
            initial,
            lambda _stage: object(),
            initial_snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        ).plan
        self.assertEqual(len(ready.target_layout.shards), 3)
        previous = dict(backend.phone_shards)
        hot = backend.snapshot(ready.current_stage, ready.finished_at_us)
        replacement = scheduler.plan_offline_phone_residency(
            {gemma_id: (replay._request(gemma_row),)},
            snapshot=hot,
            observed_at_us=hot.captured_at_us,
        )
        selected = replacement.current_stage.selected_session_id
        self.assertEqual(replacement.target_layout.changed_session_ids, (selected,))
        self.assertEqual(backend.phone_shards, previous)
        stored = {(row.parent_artifact_sha256, row.session_id): row for row in storage}
        for shard in replacement.target_layout.shards:
            source = stored[(shard.artifact_sha256, shard.session_id)]
            self.assertEqual(shard.layer_mask & ~source.layer_mask, 0)
            self.assertLessEqual(shard.maximum_columns, source.maximum_columns)
        target = next(
            row for row in replacement.target_layout.shards
            if row.session_id == selected
        )
        self.assertEqual(target.layer_mask.bit_count(), 8)
        self.assertEqual(
            target.artifact_sha256,
            scheduler.runtime_model_manifest(gemma_id).artifact_sha256,
        )
        finished = preloader.execute_next_stage(
            replacement.plan_id,
            object(),
            snapshot=hot,
            observed_at_us=hot.captured_at_us,
        )
        self.assertEqual(finished.plan.state, "READY")
        self.assertEqual(backend.phone_shards[selected].session_generation, 2)
        for session_id in set(previous) - {selected}:
            self.assertEqual(backend.phone_shards[session_id], previous[session_id])
            self.assertEqual(backend.load_count_by_session[session_id], 1)

    def test_model_generic_replacement_and_epoch_bumped_rollback(
        self,
    ) -> None:
        scheduler, qwen_requests, snapshot = self.replay_runtime()
        backend = _OfflinePhoneBackend(snapshot)
        preloader = CanonicalOfflinePhoneResidencyPreloader(
            scheduler,
            backend,
            epoch_ns=(
                time.monotonic_ns() - snapshot.captured_at_us * 1_000
            ),
            snapshot_provider=backend.snapshot,
        )
        qwen_plan = scheduler.plan_offline_phone_residency(
            qwen_requests,
            snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        qwen_result = preloader.preload(
            qwen_plan,
            lambda _stage: object(),
            initial_snapshot=snapshot,
            observed_at_us=snapshot.captured_at_us,
        )
        qwen_artifact = next(iter(qwen_plan.request_by_artifact))
        qwen_loads = dict(backend.load_count_by_session)

        case = next(
            row for row in replay_fixture.ReplayDeterminismTests.fixture[
                "cases"
            ]
            if row["case_id"] == "session_cow_gate_v3"
        )
        replay = replay_fixture.ReplayDeterminismTests(
            methodName="runTest"
        )
        gemma_requests = {}
        for row in case["requests"]:
            if row["model_id"].startswith("gemma"):
                gemma_requests.setdefault(row["model_id"], []).append(
                    replay._request(row)
                )
        hot = backend.snapshot(
            qwen_result.plan.current_stage,
            qwen_result.plan.finished_at_us,
        )
        gemma_plan = scheduler.plan_offline_phone_residency(
            gemma_requests,
            snapshot=hot,
            observed_at_us=hot.captured_at_us,
        )
        selected = gemma_plan.current_stage.selected_session_id
        retained = set(backend.phone_shards) - {selected}
        source_generation = backend.phone_shards[
            selected
        ].session_generation
        gemma_result = preloader.execute_next_stage(
            gemma_plan.plan_id,
            object(),
            snapshot=hot,
            observed_at_us=hot.captured_at_us,
        )

        self.assertEqual(gemma_result.plan.state, "READY")
        self.assertNotEqual(
            backend.phone_shards[selected].artifact_sha256,
            qwen_artifact,
        )
        self.assertEqual(
            backend.phone_shards[selected].session_generation,
            source_generation + 1,
        )
        self.assertEqual(
            {
                session_id: backend.load_count_by_session[session_id]
                for session_id in retained
            },
            {session_id: qwen_loads[session_id] for session_id in retained},
        )

        mixed = backend.snapshot(
            gemma_result.plan.current_stage,
            gemma_result.plan.finished_at_us,
        )
        model_id, requests = next(iter(gemma_requests.items()))
        follower = replace(
            requests[0], request_id="replacement-ready-follower",
            arrival_us=mixed.captured_at_us,
            deadline_us=mixed.captured_at_us + 1_000_000_000,
        )
        ticket = self.acquire_desktop_request(
            scheduler, follower, model_id, mixed
        )
        scheduler._request_helper_preparation_envelopes.clear()
        scheduler._request_helper_preparations.clear()
        load_count = len(backend.commands)
        self.assertTrue(scheduler.refresh_ready_request_helper(
            follower.request_id, expected_ticket_id=ticket.ticket_id,
            observed_at_us=mixed.captured_at_us, snapshot=mixed,
        ))
        scheduler.start_adaptive_decode(
            follower.request_id, slot_id=3, first_token_index=1,
            at_us=mixed.captured_at_us + 1,
        )
        attached = scheduler._try_attach_ready_request_helper(
            ticket, slot_id=3, token_index=1,
            at_us=mixed.captured_at_us + 2, fraction_ppm=250_000,
        )
        self.assertTrue(attached.attached, attached)
        helper = scheduler._request_helper_envelope(ticket)
        self.assertEqual(helper.preparation_changed_session_ids, ())
        self.assertIsNone(helper.replacement_authorization)
        self.assertEqual(
            [(row.session_id, row.session_generation) for row
             in helper.helper_plan.execution_contract.phone_shards],
            [(selected, source_generation + 1)],
        )
        binding = scheduler._model_placement_controller.request_binding(
            follower.request_id
        )
        self.assertTrue(binding["helper_attachment"]["lease_tokens"])
        self.assertEqual(len(backend.commands), load_count)
        scheduler.cancel_runtime_request(
            follower.request_id, mixed.captured_at_us + 3, "TEST_FINISHED"
        )
        restore_plan = scheduler.plan_offline_phone_residency(
            qwen_requests,
            snapshot=mixed,
            observed_at_us=mixed.captured_at_us,
        )
        self.assertEqual(
            restore_plan.current_stage.selected_session_id, selected
        )
        backend.fail_next_snapshot = True
        with self.assertRaisesRegex(
            Exception, "physical offline phone preparation failed"
        ):
            preloader.execute_next_stage(
                restore_plan.plan_id,
                object(),
                snapshot=mixed,
                observed_at_us=mixed.captured_at_us,
            )
        restored_generation = source_generation + 3
        self.assertEqual(
            backend.phone_shards[selected].session_generation,
            restored_generation,
        )
        logical = {
            row["session_id"]: row
            for row in scheduler.phone_residency_session_states()
        }
        self.assertEqual(logical[selected]["state"], "READY")
        self.assertEqual(
            logical[selected]["session_generation"], restored_generation
        )
        self.assertTrue(all(
            logical[session_id]["state"] == "READY"
            and logical[session_id]["session_generation"] == 1
            for session_id in retained
        ))
        source_map = dict(backend.phone_shards)
        source_loads = dict(backend.load_count_by_session)
        restored_snapshot = backend.snapshot(
            restore_plan.current_stage, backend.clock_us
        )
        retry = scheduler.plan_offline_phone_residency(
            qwen_requests,
            snapshot=restored_snapshot,
            observed_at_us=restored_snapshot.captured_at_us,
        )
        self.assertEqual(retry.current_stage.selected_session_id, selected)
        self.assertEqual(retry.current_stage.layout.layout.changed_session_ids,
                         (selected,))
        authorization = retry.current_stage.helper_envelope.replacement_authorization
        self.assertEqual(authorization.source_generation,
                         restored_generation)
        self.assertEqual(authorization.target_generation, restored_generation + 1)
        self.assertEqual(backend.phone_shards, source_map)
        retried = preloader.execute_next_stage(
            retry.plan_id,
            object(),
            snapshot=restored_snapshot,
            observed_at_us=restored_snapshot.captured_at_us,
        )
        self.assertEqual(retried.plan.state, "READY")
        self.assertEqual(backend.phone_shards[selected].artifact_sha256,
                         qwen_artifact)
        self.assertEqual(backend.phone_shards[selected].session_generation,
                         restored_generation + 1)
        self.assertEqual(backend.load_count_by_session[selected],
                         source_loads[selected] + 1)
        for session_id in retained:
            self.assertEqual(backend.phone_shards[session_id], source_map[session_id])
            self.assertEqual(backend.load_count_by_session[session_id],
                             source_loads[session_id])
        self.assertEqual(
            gemma_result.plan.current_stage.layout.layout
                .session_generation_by_id[selected],
            source_generation + 1,
        )



class LateHelperAdoptionReadyPathTests(unittest.TestCase):
    """s1a Qwen 001: started helper-less, adopted the first READY shard on the READY
    publication path (no HELPER_ADOPTED_LATE there), then lost the helper when the READY
    refresh 1 s after the second shard's publication rebuilt it and hit
    ``runtime helper plan history changed identity``."""

    @classmethod
    def setUpClass(cls) -> None:
        replay_fixture.ReplayDeterminismTests.setUpClass()

    def running_helper_less(self, *, late_helper_adoption: bool):
        scheduler, requests_by_model, snapshot = OfflinePhoneResidencyTests.replay_runtime(
            load_evidence=False
        )
        scheduler._adaptive_decode_config = replace(
            scheduler._adaptive_decode_config,
            late_helper_adoption=late_helper_adoption,
            late_helper_adoption_minimum_tokens=4,
        )
        model_id, requests = next(iter(requests_by_model.items()))
        model = scheduler.runtime_model_manifest(model_id)
        scheduler.register_phone_ffn_shard_storage(tuple(
            PhoneFfnShardStorageMetadata(
                parent_artifact_sha256=model.artifact_sha256,
                shard_sha256=canonical_sha256([model_id, index]),
                path="/phone/HTP" + str(index) + ".gguf",
                layer_mask=63 << (6 * index),
                maximum_columns=model.feed_forward_length,
                session_id="HTP" + str(index),
            )
            for index in range(3)
        ))
        request = replace(
            requests[0], arrival_us=snapshot.captured_at_us,
            deadline_us=snapshot.captured_at_us + 600_000_000, output_tokens=64,
        )
        ticket = OfflinePhoneResidencyTests.acquire_desktop_request(
            scheduler, request, model_id, snapshot, selection_mode="calibration"
        )
        scheduler.start_adaptive_decode(
            request.request_id, slot_id=0, first_token_index=0,
            at_us=snapshot.captured_at_us + 1,
        )
        self.assertFalse(
            scheduler._adaptive_decode.snapshot(request.request_id)["helper_available"]
        )
        return scheduler, ticket, _OfflinePhoneBackend(snapshot), snapshot

    @staticmethod
    def publish_next_shard(scheduler, ticket, backend, snapshot):
        """The request's preparation loads one more session and it becomes READY."""
        request_id = ticket.request.request_id
        helper = scheduler.runtime_request_helper_preparation_envelope(
            request_id, expected_ticket_id=ticket.ticket_id,
            observed_at_us=snapshot.captured_at_us, snapshot=snapshot,
        )
        decision = scheduler.begin_request_helper_preparation(
            request_id, observed_at_us=snapshot.captured_at_us, snapshot=snapshot,
            expected_phone_layout_generation=helper.phone_layout_generation,
            expected_phone_layout_geometry_sha256=helper.phone_layout_geometry_sha256,
            expected_operator_plan_sha256=helper.operator_plan_sha256,
        )
        command = bind_ready_helper_to_physical_command(
            replace(interpret_runtime_ticket(ticket), helper_envelope=None,
                    helper_transitions=()),
            helper,
        )
        receipts = tuple(
            transition_receipt_from_observation(
                row, backend.apply_transition(row, object(), lambda: None)
            )
            for row in command.helper_transitions
        )
        snapshot = backend.snapshot(SimpleNamespace(helper_envelope=helper), backend.clock_us)
        completed = scheduler.complete_request_helper_preparation(
            request_id, decision["preparation_ticket_id"], receipts, snapshot=snapshot,
        )
        assert completed["state"] == "READY"
        return helper, snapshot

    @staticmethod
    def kinds(scheduler, request_id):
        return [
            row["kind"] for row in
            scheduler._model_placement_controller.request_helper_events(request_id)
        ]

    def test_ready_publication_adoption_records_helper_adopted_late(self) -> None:
        for enabled in (True, False):
            with self.subTest(late_helper_adoption=enabled):
                scheduler, ticket, backend, snapshot = self.running_helper_less(
                    late_helper_adoption=enabled
                )
                request_id = ticket.request.request_id
                self.publish_next_shard(scheduler, ticket, backend, snapshot)
                state = scheduler._adaptive_decode.snapshot(request_id)
                self.assertTrue(state["helper_available"])
                adopted = [
                    row for row in
                    scheduler._model_placement_controller.request_helper_events(request_id)
                    if row["kind"] == "HELPER_ADOPTED_LATE"
                ]
                if not enabled:
                    self.assertEqual(adopted, [])
                    continue
                self.assertEqual(len(adopted), 1)
                self.assertEqual(adopted[0]["source"], "READY_LAYOUT_PUBLISHED")
                self.assertEqual(adopted[0]["phone_layout_generation"], 1)
                self.assertEqual(adopted[0]["token_index"], 0)

    def refreshed_after_a_history_conflict(self, *, late_helper_adoption: bool):
        """Second shard READY, then the READY refresh sees another identity for the plan."""
        scheduler, ticket, backend, snapshot = self.running_helper_less(
            late_helper_adoption=late_helper_adoption
        )
        request_id = ticket.request.request_id
        helper, snapshot = self.publish_next_shard(scheduler, ticket, backend, snapshot)
        scheduler.adaptive_decode_boundary(
            request_id, slot_id=0, token_index=4, at_us=snapshot.captured_at_us + 4,
        )
        helper, snapshot = self.publish_next_shard(scheduler, ticket, backend, snapshot)
        context = scheduler._late_request_helper_contexts[request_id]
        self.assertEqual(context.helper.phone_layout_generation, 2)
        binding = scheduler._model_placement_controller.request_binding(request_id)
        self.assertEqual(binding["helper_attachment"]["phone_layout_generation"], 1)
        # What s1a's refresh rebuilt differed from the recorded envelope of the same plan.
        history = scheduler._request_helper_envelope_history[request_id]
        plan_sha256 = context.helper.operator_plan_sha256
        history[plan_sha256] = replace(
            history[plan_sha256],
            desktop_parent_route_id=history[plan_sha256].desktop_parent_route_id + ":rebuilt",
        )
        later = backend.snapshot(
            SimpleNamespace(helper_envelope=helper), backend.clock_us + 1_000_000
        )
        refreshed = scheduler.refresh_ready_request_helper(
            request_id, expected_ticket_id=ticket.ticket_id,
            observed_at_us=later.captured_at_us, snapshot=later,
        )
        scheduler.adaptive_decode_boundary(
            request_id, slot_id=0, token_index=8, at_us=later.captured_at_us + 8,
        )
        return scheduler, request_id, refreshed

    def test_ready_refresh_keeps_the_helper_materialized_for_the_ready_layout(self) -> None:
        scheduler, request_id, refreshed = self.refreshed_after_a_history_conflict(
            late_helper_adoption=True
        )
        self.assertTrue(refreshed)
        kinds = self.kinds(scheduler, request_id)
        self.assertIn("HELPER_MATERIALIZATION_RETAINED", kinds)
        self.assertNotIn("HELPER_REMATERIALIZATION_FAILED", kinds)
        # The next boundary expands the attachment onto the second READY shard.
        self.assertEqual(kinds.count("HELPER_EXPANDED"), 1)
        binding = scheduler._model_placement_controller.request_binding(request_id)
        self.assertEqual(binding["helper_attachment"]["phone_layout_generation"], 2)
        self.assertEqual(scheduler._late_request_helper_contexts[request_id]
                         .helper.phone_layout_generation, 2)

    def test_without_late_adoption_the_refresh_still_rebuilds_and_fails(self) -> None:
        scheduler, request_id, refreshed = self.refreshed_after_a_history_conflict(
            late_helper_adoption=False
        )
        self.assertFalse(refreshed)
        kinds = self.kinds(scheduler, request_id)
        self.assertIn("HELPER_REMATERIALIZATION_FAILED", kinds)
        self.assertNotIn("HELPER_MATERIALIZATION_RETAINED", kinds)
        self.assertNotIn(request_id, scheduler._late_request_helper_contexts)

if __name__ == "__main__":
    unittest.main()
