#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import unittest

from research_dev.scheduler._internal.model_placement_controller import (
    ModelDemandSnapshot,
    ModelPlacementController,
    ModelPlacementControllerError,
    ModelPlacementPolicy,
    ModelPlacementTrigger,
    PhoneLayoutRequestImpact,
    RequestHelperEnvelopeBinding,
)
from research_dev.scheduler._internal.phone_shards import (
    PhoneFfnResidencyLayout,
    PhoneFfnShardPlacement,
    _residency_geometry_sha256,
)
from research_dev.scheduler._internal.types import canonical_sha256


def identity(label: str) -> str:
    return canonical_sha256({"identity": label})


def phone_layout(label: str = "a") -> PhoneFfnResidencyLayout:
    artifact = identity("phone-artifact-" + label)
    geometry = identity("phone-shard-" + label)
    shard = PhoneFfnShardPlacement(
        artifact_sha256=artifact,
        session_id="HTP0",
        endpoint="session://phone/HTP0",
        memory_resource_id="phone-session-0",
        operator_ids=("blk.0.ffn",),
        layer_mask=1,
        maximum_columns=1024,
        resident_bytes=4096,
        resident_geometry_sha256=geometry,
        operator_plan_sha256=identity("phone-plan-" + label),
    )
    return PhoneFfnResidencyLayout(
        shards=(shard,),
        queued_work_by_artifact={artifact: 10},
        queue_benefit_by_artifact={artifact: 1000},
        queue_benefit_by_session={"HTP0": 1000},
        queue_benefit=1000,
        transition_cost=100,
        transition_cost_by_session={"HTP0": 100},
        objective=-900,
        objective_kind="queue_energy_delta_uj",
        changed_session_ids=("HTP0",),
        geometry_sha256=canonical_sha256({
            "artifact_sha256": artifact,
            "shards": [{
                "geometry_sha256": geometry,
                "session_id": "HTP0",
            }],
        }),
    )


def mixed_phone_layout(
    assignments: tuple[str, str, str],
    *,
    current_assignments: tuple[str, str, str] | None = None,
    benefit_by_session: tuple[int, int, int] = (1000, 1000, 1000),
    transition_uj: int = 100,
) -> PhoneFfnResidencyLayout:
    artifact_by_label = {
        label: identity("mixed-artifact-" + label)
        for label in set(assignments)
    }
    if current_assignments is not None:
        artifact_by_label.update({
            label: identity("mixed-artifact-" + label)
            for label in set(current_assignments)
        })
    shards = tuple(
        PhoneFfnShardPlacement(
            artifact_sha256=artifact_by_label[label],
            session_id="HTP" + str(index),
            endpoint="session://phone/HTP" + str(index),
            memory_resource_id="phone-session-" + str(index),
            operator_ids=("blk." + str(index) + ".ffn",),
            layer_mask=1 << index,
            maximum_columns=1024,
            resident_bytes=4096,
            resident_geometry_sha256=identity(
                "mixed-geometry-" + label + "-" + str(index)
            ),
            operator_plan_sha256=identity(
                "mixed-plan-" + label
            ),
        )
        for index, label in enumerate(assignments)
    )
    changed = tuple(
        "HTP" + str(index)
        for index, label in enumerate(assignments)
        if current_assignments is None
        or current_assignments[index] != label
    )
    benefit_by_artifact: dict[str, int] = {}
    benefit_by_session = {
        "HTP" + str(index): benefit
        for index, benefit in enumerate(benefit_by_session)
    }
    for shard in shards:
        benefit_by_artifact[shard.artifact_sha256] = (
            benefit_by_artifact.get(shard.artifact_sha256, 0)
            + benefit_by_session[shard.session_id]
        )
    total_benefit = sum(benefit_by_session.values())
    total_transition = transition_uj * len(changed)
    return PhoneFfnResidencyLayout(
        shards=shards,
        queued_work_by_artifact={
            artifact: 100 for artifact in benefit_by_artifact
        },
        queue_benefit_by_artifact=benefit_by_artifact,
        queue_benefit_by_session=benefit_by_session,
        queue_benefit=total_benefit,
        transition_cost=total_transition,
        transition_cost_by_session={
            session_id: transition_uj for session_id in changed
        },
        objective=total_transition - total_benefit,
        objective_kind="queue_energy_delta_uj",
        changed_session_ids=changed,
        geometry_sha256=canonical_sha256(
            {
                "artifact_sha256": shards[0].artifact_sha256,
                "shards": [
                    {
                        "geometry_sha256": (
                            row.resident_geometry_sha256
                        ),
                        "session_id": row.session_id,
                    }
                    for row in shards
                ],
            }
            if len({row.artifact_sha256 for row in shards}) == 1 else
            {
                "shards": [
                    {
                        "artifact_sha256": row.artifact_sha256,
                        "geometry_sha256": (
                            row.resident_geometry_sha256
                        ),
                        "session_id": row.session_id,
                    }
                    for row in shards
                ],
            }
        ),
    )


def partial_phone_layout(
    assignments: dict[str, str],
    *,
    current: PhoneFfnResidencyLayout | None = None,
    benefit_by_session: dict[str, int] | None = None,
    transition_uj: int = 100,
    reshape: tuple[str, ...] = (),
) -> PhoneFfnResidencyLayout:
    """A learning layout over an arbitrary subset of HTP sessions.

    Shards shared with ``current`` keep their identity unless the session is
    listed in ``reshape``; ``changed_session_ids`` are the sessions whose shard
    differs from ``current``.
    """
    current_by_session = {} if current is None else {row.session_id: row for row in current.shards}
    shards = []
    for session_id, label in sorted(assignments.items()):
        existing = current_by_session.get(session_id)
        if (existing is not None and session_id not in reshape
                and existing.artifact_sha256 == identity("mixed-artifact-" + label)):
            shards.append(existing)
            continue
        shards.append(PhoneFfnShardPlacement(
            artifact_sha256=identity("mixed-artifact-" + label),
            session_id=session_id,
            endpoint="session://phone/" + session_id,
            memory_resource_id="phone-session-" + session_id,
            operator_ids=("blk." + session_id[-1] + ".ffn",),
            layer_mask=1 << int(session_id[-1]),
            maximum_columns=1024,
            resident_bytes=4096,
            resident_geometry_sha256=identity(
                "mixed-geometry-" + label + "-" + session_id
                + ("-reshaped" if session_id in reshape else "")
            ),
            operator_plan_sha256=identity("mixed-plan-" + label),
        ))
    shards = tuple(shards)
    changed = tuple(
        row.session_id for row in shards
        if current_by_session.get(row.session_id) is None
        or current_by_session[row.session_id] != row
    )
    benefits = dict(benefit_by_session or {})
    by_session = {row.session_id: benefits.get(row.session_id, 1000) for row in shards}
    by_artifact: dict[str, int] = {}
    for row in shards:
        by_artifact[row.artifact_sha256] = by_artifact.get(row.artifact_sha256, 0) + by_session[row.session_id]
    total_benefit = sum(by_session.values())
    return PhoneFfnResidencyLayout(
        shards=shards,
        queued_work_by_artifact={artifact: 100 for artifact in by_artifact},
        queue_benefit_by_artifact=by_artifact,
        queue_benefit_by_session=by_session,
        queue_benefit=total_benefit,
        transition_cost=transition_uj * len(changed),
        transition_cost_by_session={session_id: transition_uj for session_id in changed},
        objective=-total_benefit,
        objective_kind="queue_rough_compute_ops",
        changed_session_ids=changed,
        geometry_sha256=_residency_geometry_sha256(shards),
    )


def snapshot(
    observed_at_us: int = 1_000_000,
    *,
    active: int = 1,
    queued: int = 0,
    queued_input_tokens: int = 0,
    queued_output_tokens: int = 0,
    oldest_wait_us: int = 0,
    drain_us: int = 0,
    devices: tuple[str, ...] = ("compute-a", "helper-b"),
    sessions: tuple[str, ...] = ("session-0", "session-1", "session-2"),
    resident: str | None = None,
    memory_generation: str = "memory-1",
    resource_generation: str = "resources-1",
    capability_generation: str = "capability-1",
    profile_generation: str = "profile-1",
    transport_generation: str = "transport-1",
    residency_generation: str = "residency-1",
    learning_generation: str = "learning-1",
) -> ModelDemandSnapshot:
    return ModelDemandSnapshot(
        artifact_sha256=identity("artifact"),
        observed_at_us=observed_at_us,
        active_request_count=active,
        queued_request_count=queued,
        queued_input_tokens=queued_input_tokens,
        queued_output_tokens=queued_output_tokens,
        oldest_queued_wait_us=oldest_wait_us,
        predicted_queue_drain_us=drain_us,
        current_resident_component_identity_sha256=resident,
        available_device_ids=devices,
        available_session_ids=sessions,
        memory_generation_sha256=identity(memory_generation),
        resource_calendar_generation_sha256=identity(resource_generation),
        capability_generation_sha256=identity(capability_generation),
        profile_generation_sha256=identity(profile_generation),
        transport_generation_sha256=identity(transport_generation),
        residency_generation_sha256=identity(residency_generation),
        learning_generation_sha256=identity(learning_generation),
    )


def published_trigger(
    row: ModelDemandSnapshot,
    **kwargs: object,
) -> ModelPlacementTrigger:
    return ModelPlacementTrigger(
        snapshot=row,
        epoch_sha256=identity("epoch"),
        epoch_demand_generation_sha256=row.demand_generation_sha256,
        epoch_pressure_bucket=row.pressure_bucket,
        epoch_valid_until_us=row.observed_at_us + 10_000_000,
        **kwargs,
    )


class ModelPlacementControllerTests(unittest.TestCase):
    def test_layout_publication_time_is_separate_from_snapshot_time(self) -> None:
        controller = ModelPlacementController()
        controller.record_phone_layout_evaluation(100, {"reason": "before"})
        original = dict(controller.phone_layout_events()[0])
        self.assertNotIn("published_at_us", original)
        controller.set_phone_layout_event_clock(lambda: 1200)
        controller.record_phone_layout_evaluation(100, {"reason": "after"})
        event = dict(controller.phone_layout_events()[-1])
        self.assertEqual(event["observed_at_us"], 100)
        self.assertEqual(event["published_at_us"], 1200)
        self.assertEqual(controller.phone_layout_events()[0], original)
        event_hash = event.pop("event_sha256")
        self.assertEqual(event_hash, canonical_sha256(event))
        controller.set_phone_layout_event_clock(None)
        controller.record_phone_layout_evaluation(200, {"reason": "replay"})
        self.assertNotIn("published_at_us", controller.phone_layout_events()[-1])

    def controller(self) -> ModelPlacementController:
        return ModelPlacementController(ModelPlacementPolicy(
            debounce_us=100_000,
            switch_hysteresis_us=100_000,
            phone_layout_confirmation_snapshots=2,
            phone_minimum_residency_us=100_000,
        ))

    def establish(self, controller: ModelPlacementController) -> ModelDemandSnapshot:
        row = snapshot()
        self.assertEqual(
            controller.evaluate(ModelPlacementTrigger(snapshot=row)).kind,
            "RECOMPUTE_NOW",
        )
        return row

    def active_ggg_to_ggq(self):
        controller = self.controller()
        current = mixed_phone_layout(("a", "a", "a"))
        target_layout = mixed_phone_layout(
            ("a", "a", "b"),
            current_assignments=("a", "a", "a"),
            benefit_by_session=(1000, 1000, 4500),
            transition_uj=500,
        )
        proposed = controller.propose_phone_layout(
            current,
            workspace_bytes=1024,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-functionfs", "phone-usb"),
            observed_at_us=100,
        )
        controller.begin_phone_layout_transition(
            proposed.generation,
            ticket_id="load-ggg",
            transition_ids=("load-ggg",),
            ready_at_us=200,
            projection_token_sha256=identity("load-ggg-projection"),
            workspace_bytes=1024,
            observed_at_us=110,
        )
        ready = controller.complete_phone_layout_transition(
            generation=proposed.generation,
            ticket_id="load-ggg",
            transition_ids=("load-ggg",),
            geometry_sha256=current.geometry_sha256,
            projection_token_sha256=identity("load-ggg-projection"),
            finished_at_us=200,
        )
        desktop = identity("ggg-desktop")
        artifact = current.shards[0].artifact_sha256
        helper = RequestHelperEnvelopeBinding(
            route_id="ggg-helper",
            operator_plan_sha256=current.shards[0].operator_plan_sha256,
            desktop_parent_route_id="desktop-parent",
            desktop_placement_sha256=desktop,
            phone_layout_generation=ready.generation,
            phone_layout_geometry_sha256=current.geometry_sha256,
            activation_dtype="f16",
            assisted_layer_mask=sum(row.layer_mask for row in current.shards),
            maximum_columns=1024,
            allowed_fractions_ppm=(0, 500_000, 750_000, 1_000_000),
            phone_session_ids=("HTP0", "HTP1", "HTP2"),
            resource_ids=("phone-htp", "phone-usb"),
        )
        controller.bind_dispatched_request(
            "gemma-request",
            artifact,
            "desktop-parent",
            identity("ggg-base-component"),
            0,
            desktop_placement_sha256=desktop,
            kv_cache_owner_id="gemma-kv",
            sequence_identity="gemma-sequence",
            server_slot_id=2,
            helper_envelope=helper,
            output_tokens=500,
        )
        controller.mark_request_acquired("gemma-request")
        controller.attach_request_helper(
            "gemma-request",
            phone_layout_generation=ready.generation,
            phone_layout_geometry_sha256=current.geometry_sha256,
            resident_component_identity_sha256=(
                ready.resident_component_identity_sha256
            ),
            operator_plan_sha256=helper.operator_plan_sha256,
            start_token_index=10,
            fraction_ppm=750_000,
            lease_tokens=("ggg-helper-lease",),
            lease_reserved_until_us=5_000_000,
            observed_at_us=250,
        )
        controller.record_request_helper_work(
            "gemma-request", 30, observed_at_us=260
        )
        target = controller.propose_phone_layout(
            target_layout,
            workspace_bytes=1024,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-functionfs", "phone-usb"),
            observed_at_us=200_000,
        )
        return controller, current, target_layout, ready, target, helper, desktop

    def test_phone_layout_is_ready_only_after_physical_proof(self) -> None:
        controller = self.controller()
        proposed = controller.propose_phone_layout(
            phone_layout(),
            workspace_bytes=1024,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=(
                "phone-functionfs", "phone-usb"
            ),
            observed_at_us=100,
        )
        self.assertEqual(proposed.state, "PROPOSED")
        transitioning = controller.begin_phone_layout_transition(
            proposed.generation,
            ticket_id="request:attempt:0",
            transition_ids=("load-phone",),
            ready_at_us=200,
            projection_token_sha256=identity("projection"),
            workspace_bytes=2048,
            observed_at_us=110,
        )
        self.assertEqual(transitioning.state, "PREPARING")
        self.assertIsNone(controller.ready_phone_layout())
        with self.assertRaisesRegex(
            ModelPlacementControllerError,
            "completion differs",
        ):
            controller.complete_phone_layout_transition(
                generation=proposed.generation,
                ticket_id="wrong-ticket",
                transition_ids=("load-phone",),
                geometry_sha256=proposed.layout.geometry_sha256,
                projection_token_sha256=identity("projection"),
                finished_at_us=200,
            )
        ready = controller.complete_phone_layout_transition(
            generation=proposed.generation,
            ticket_id="request:attempt:0",
            transition_ids=("load-phone",),
            geometry_sha256=proposed.layout.geometry_sha256,
            projection_token_sha256=identity("projection"),
            finished_at_us=200,
        )
        self.assertEqual(ready.state, "READY")
        self.assertEqual(ready.ready_at_us, 200)
        self.assertEqual(
            controller.ready_phone_layout().generation,
            proposed.generation,
        )
        self.assertIsNone(controller.complete_phone_layout_transition(
            generation=proposed.generation,
            ticket_id="request:attempt:0",
            transition_ids=("load-phone",),
            geometry_sha256=proposed.layout.geometry_sha256,
            projection_token_sha256=identity("projection"),
            finished_at_us=210,
        ))
        self.assertEqual(
            controller.phone_layout_events()[-1]["kind"],
            "STALE_PREPARATION_COMPLETION_IGNORED",
        )

    def test_no_refresh_while_queue_pressure_remains_in_same_bucket(self) -> None:
        controller = self.controller()
        first = self.establish(controller)
        second = replace(
            first,
            observed_at_us=2_000_000,
            queued_request_count=1,
            queued_input_tokens=100,
            queued_output_tokens=40,
            oldest_queued_wait_us=10,
            predicted_queue_drain_us=20,
        )
        controller.evaluate(published_trigger(second))
        third = replace(
            second,
            observed_at_us=3_000_000,
            queued_input_tokens=127,
            queued_output_tokens=63,
            oldest_queued_wait_us=15,
            predicted_queue_drain_us=31,
        )
        action = controller.evaluate(published_trigger(third))
        self.assertEqual(action.kind, "KEEP_EPOCH")
        self.assertEqual(action.trigger_reasons, ("NO_MATERIAL_CHANGE",))

    def test_refresh_when_pressure_bucket_changes(self) -> None:
        controller = self.controller()
        first = self.establish(controller)
        changed = replace(
            first,
            observed_at_us=2_000_000,
            queued_request_count=4,
            queued_input_tokens=2048,
            queued_output_tokens=1024,
            oldest_queued_wait_us=1_000_000,
            predicted_queue_drain_us=8_000_000,
        )
        action = controller.evaluate(published_trigger(changed))
        self.assertEqual(action.kind, "REFRESH_IN_BACKGROUND")
        self.assertIn("QUEUE_PRESSURE_BUCKET_CHANGED", action.trigger_reasons)

    def test_device_session_removal_invalidates_epoch(self) -> None:
        controller = self.controller()
        first = self.establish(controller)
        changed = replace(
            first,
            observed_at_us=2_000_000,
            available_session_ids=("session-0", "session-1"),
        )
        action = controller.evaluate(published_trigger(
            changed,
            selected_route_feasible=False,
        ))
        self.assertEqual(action.kind, "RECOMPUTE_NOW")
        self.assertIn("SESSION_AVAILABILITY_CHANGED", action.trigger_reasons)
        self.assertIn("SELECTED_ROUTE_INFEASIBLE", action.trigger_reasons)

    def test_three_session_to_two_session_degradation(self) -> None:
        controller = self.controller()
        first = self.establish(controller)
        changed = replace(
            first,
            observed_at_us=2_000_000,
            available_session_ids=("session-0", "session-1"),
            capability_generation_sha256=identity("capability-2"),
        )
        action = controller.evaluate(published_trigger(
            changed,
            selected_route_feasible=False,
            old_component_identity_sha256=identity("three-session"),
            proposed_component_identity_sha256=identity("two-session"),
            old_warm_energy_upper_uj=10_000,
            new_warm_energy_upper_uj=8_000,
            new_transition_energy_upper_uj=1_000,
            old_restore_energy_upper_uj=0,
            desktop_latency_upper_us=100_000,
            proposed_latency_upper_us=110_000,
            maximum_latency_ppm=1_250_000,
            predicted_reuse_count=1,
        ))
        self.assertEqual(action.kind, "RECOMPUTE_NOW")
        self.assertIsNone(action.break_even_use_count)
        self.assertEqual(action.new_component_identity_sha256, identity("two-session"))
        publication = controller.authorize_publication(published_trigger(
            changed,
            selected_route_feasible=False,
            old_component_identity_sha256=identity("three-session"),
            proposed_component_identity_sha256=identity("two-session"),
            old_warm_energy_upper_uj=10_000,
            new_warm_energy_upper_uj=8_000,
            new_transition_energy_upper_uj=1_000,
            old_restore_energy_upper_uj=0,
            desktop_latency_upper_us=100_000,
            proposed_latency_upper_us=110_000,
            maximum_latency_ppm=1_250_000,
            predicted_reuse_count=1,
        ))
        self.assertEqual(publication.break_even_use_count, 1)
        self.assertEqual(publication.kind, "RECOMPUTE_NOW")

    def test_transition_cost_prevents_low_reuse_switch(self) -> None:
        controller = self.controller()
        row = snapshot(resident=identity("old"))
        action = controller.authorize_publication(published_trigger(
            row,
            old_component_identity_sha256=identity("old"),
            proposed_component_identity_sha256=identity("new"),
            old_warm_energy_upper_uj=10_000,
            new_warm_energy_upper_uj=9_000,
            new_transition_energy_upper_uj=5_000,
            old_restore_energy_upper_uj=2_000,
            predicted_reuse_count=1,
        ))
        self.assertEqual(action.break_even_use_count, 7)
        self.assertEqual(action.expected_reuse_count, 2)
        self.assertEqual(action.kind, "KEEP_EPOCH")
        self.assertIn("TRANSITION_BREAK_EVEN_NOT_MET", action.trigger_reasons)

    def test_sufficient_queued_reuse_enables_switch(self) -> None:
        controller = self.controller()
        row = snapshot(
            queued=7,
            queued_input_tokens=4096,
            queued_output_tokens=2048,
            resident=identity("old"),
        )
        action = controller.authorize_publication(published_trigger(
            row,
            old_component_identity_sha256=identity("old"),
            proposed_component_identity_sha256=identity("new"),
            old_warm_energy_upper_uj=10_000,
            new_warm_energy_upper_uj=9_000,
            new_transition_energy_upper_uj=5_000,
            old_restore_energy_upper_uj=2_000,
        ))
        self.assertEqual(action.break_even_use_count, 7)
        self.assertEqual(action.expected_reuse_count, 8)
        self.assertEqual(action.kind, "RECOMPUTE_NOW")

    def test_refresh_trigger_does_not_apply_publication_economics(self) -> None:
        controller = self.controller()
        first = self.establish(controller)
        changed = replace(
            first,
            observed_at_us=2_000_000,
            learning_generation_sha256=identity("learning-2"),
            current_resident_component_identity_sha256=identity("old"),
        )
        action = controller.evaluate(published_trigger(
            changed,
            old_component_identity_sha256=identity("old"),
            proposed_component_identity_sha256=identity("new"),
            old_warm_energy_lower_uj=10_000,
            old_warm_energy_upper_uj=11_000,
            new_warm_energy_upper_uj=9_000,
            new_transition_energy_upper_uj=20_000,
            old_restore_energy_upper_uj=0,
        ))
        self.assertEqual(action.kind, "REFRESH_IN_BACKGROUND")
        self.assertIsNone(action.break_even_use_count)
        self.assertIsNone(action.selected_transition_energy_uj)

    def test_publication_rejects_energy_negative_switch(self) -> None:
        controller = self.controller()
        row = snapshot(
            active=1,
            queued=2,
            resident=identity("old"),
        )
        action = controller.authorize_publication(published_trigger(
            row,
            old_component_identity_sha256=identity("old"),
            proposed_component_identity_sha256=identity("new"),
            old_warm_energy_lower_uj=9_000,
            old_warm_energy_upper_uj=10_000,
            new_warm_energy_upper_uj=9_500,
            new_transition_energy_upper_uj=1_000,
            old_restore_energy_upper_uj=500,
            desktop_latency_upper_us=100_000,
            proposed_latency_upper_us=100_000,
        ))
        self.assertEqual(action.kind, "KEEP_EPOCH")
        self.assertIsNone(action.break_even_use_count)
        self.assertEqual(action.expected_reuse_count, 3)
        self.assertIn(
            "PROPOSED_WARM_ENERGY_NOT_LOWER",
            action.trigger_reasons,
        )

    def test_publication_uses_queue_reuse_once(self) -> None:
        controller = self.controller()
        row = snapshot(
            active=1,
            queued=2,
            resident=identity("old"),
        )
        action = controller.authorize_publication(published_trigger(
            row,
            old_component_identity_sha256=identity("old"),
            proposed_component_identity_sha256=identity("new"),
            old_warm_energy_lower_uj=10_000,
            old_warm_energy_upper_uj=11_000,
            new_warm_energy_upper_uj=9_000,
            new_transition_energy_upper_uj=2_500,
            old_restore_energy_upper_uj=0,
            desktop_latency_upper_us=100_000,
            proposed_latency_upper_us=110_000,
            maximum_latency_ppm=1_250_000,
        ))
        self.assertEqual(action.expected_reuse_count, 3)
        self.assertEqual(action.break_even_use_count, 3)
        self.assertEqual(action.kind, "RECOMPUTE_NOW")

    def test_repeated_learning_notifications_coalesce(self) -> None:
        controller = self.controller()
        first = self.establish(controller)
        controller.mark_background_complete(first.artifact_sha256, 1_100_000)
        second = replace(
            first,
            observed_at_us=2_000_000,
            learning_generation_sha256=identity("learning-2"),
        )
        first_action = controller.evaluate(published_trigger(second))
        self.assertEqual(first_action.kind, "REFRESH_IN_BACKGROUND")
        third = replace(
            second,
            observed_at_us=2_010_000,
            learning_generation_sha256=identity("learning-3"),
        )
        second_action = controller.evaluate(published_trigger(third))
        self.assertEqual(second_action.kind, "KEEP_EPOCH")
        self.assertTrue(second_action.coalesced)
        self.assertEqual(controller.stats()["REFRESH_IN_BACKGROUND"], 1)

    def test_no_future_request_is_used(self) -> None:
        current = snapshot(queued=1, queued_input_tokens=256)
        left = self.controller().evaluate(published_trigger(current))
        future_trace_suffix = ((9_000_000, "future-request"),)
        self.assertEqual(len(future_trace_suffix), 1)
        right = self.controller().evaluate(published_trigger(current))
        self.assertEqual(left, right)
        self.assertNotIn(
            "future",
            ModelDemandSnapshot.__dataclass_fields__,
        )

    def test_request_route_remains_fixed_after_dispatch(self) -> None:
        controller = self.controller()
        controller.bind_dispatched_request(
            "request-1",
            identity("artifact"),
            "route-a",
            identity("resident-a"),
            0,
        )
        controller.mark_request_acquired("request-1")
        with self.assertRaisesRegex(
            ModelPlacementControllerError,
            "placement is immutable",
        ):
            controller.bind_dispatched_request(
                "request-1",
                identity("artifact"),
                "route-b",
                identity("resident-a"),
                0,
            )

    def test_queued_request_route_can_change_before_acquisition(self) -> None:
        controller = self.controller()
        controller.bind_dispatched_request(
            "request-1",
            identity("artifact"),
            "route-a",
            identity("portfolio-a"),
            0,
        )

        controller.bind_dispatched_request(
            "request-1",
            identity("artifact"),
            "route-b",
            identity("portfolio-b"),
            750_000,
        )

        self.assertEqual(
            controller.request_binding("request-1")["route_id"],
            "route-b",
        )
        self.assertEqual(
            controller.request_binding("request-1")[
                "resident_component_identity_sha256"
            ],
            identity("portfolio-b"),
        )

    def test_fraction_change_does_not_change_residency_identity(self) -> None:
        controller = self.controller()
        component = identity("resident-a")
        controller.bind_dispatched_request(
            "request-1", identity("artifact"), "route-a", component, 0
        )
        controller.update_request_fraction(
            "request-1",
            750_000,
            resident_component_identity_sha256=component,
        )
        self.assertEqual(
            controller.request_binding("request-1")["fraction_ppm"],
            750_000,
        )
        with self.assertRaisesRegex(
            ModelPlacementControllerError,
            "changed residency identity",
        ):
            controller.update_request_fraction(
                "request-1",
                500_000,
                resident_component_identity_sha256=identity("resident-b"),
            )

    def test_ready_helper_attaches_without_changing_base_placement(
        self,
    ) -> None:
        controller = self.controller()
        layout = phone_layout()
        state = controller.propose_phone_layout(
            layout,
            workspace_bytes=1024,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=(
                "phone-functionfs", "phone-usb"
            ),
            observed_at_us=100,
        )
        state = controller.begin_phone_layout_transition(
            state.generation,
            ticket_id="helper-preparation",
            transition_ids=("load-phone",),
            ready_at_us=200,
            projection_token_sha256=identity("helper-projection"),
            workspace_bytes=1024,
            observed_at_us=110,
        )
        state = controller.complete_phone_layout_transition(
            generation=state.generation,
            ticket_id="helper-preparation",
            transition_ids=("load-phone",),
            geometry_sha256=layout.geometry_sha256,
            projection_token_sha256=identity("helper-projection"),
            finished_at_us=200,
        )
        artifact = layout.shards[0].artifact_sha256
        desktop = identity("desktop-placement")
        base_component = identity("desktop-component")
        helper = RequestHelperEnvelopeBinding(
            route_id="phone-helper",
            operator_plan_sha256=layout.shards[0].operator_plan_sha256,
            desktop_parent_route_id="desktop-parent",
            desktop_placement_sha256=desktop,
            phone_layout_generation=state.generation,
            phone_layout_geometry_sha256=layout.geometry_sha256,
            activation_dtype="f16",
            assisted_layer_mask=1,
            maximum_columns=1024,
            allowed_fractions_ppm=(0, 500_000, 1_000_000),
            phone_session_ids=("HTP0",),
            resource_ids=("phone-htp", "phone-usb"),
        )
        controller.bind_dispatched_request(
            "request-1",
            artifact,
            "desktop-parent",
            base_component,
            0,
            desktop_placement_sha256=desktop,
            kv_cache_owner_id="kv-owner",
            sequence_identity="sequence-a",
            helper_envelope=helper,
        )
        controller.mark_request_acquired("request-1")
        controller.bind_request_server_slot(
            "request-1", sequence_identity="sequence-a", server_slot_id=3
        )
        controller.attach_request_helper(
            "request-1",
            phone_layout_generation=state.generation,
            phone_layout_geometry_sha256=layout.geometry_sha256,
            resident_component_identity_sha256=(
                state.resident_component_identity_sha256
            ),
            operator_plan_sha256=helper.operator_plan_sha256,
            start_token_index=7,
            fraction_ppm=500_000,
            lease_tokens=("helper-window-lease",),
            lease_reserved_until_us=500,
            observed_at_us=250,
        )
        controller.renew_request_helper_leases(
            "request-1",
            lease_tokens=("helper-window-lease",),
            reserved_until_us=700,
            observed_at_us=300,
        )
        controller.update_request_fraction(
            "request-1",
            1_000_000,
            resident_component_identity_sha256=(
                state.resident_component_identity_sha256
            ),
            observed_at_us=350,
        )
        binding = controller.request_binding("request-1")
        self.assertEqual(binding["base"]["route_id"], "desktop-parent")
        self.assertEqual(binding["base"]["kv_cache_owner_id"], "kv-owner")
        self.assertEqual(binding["base"]["server_slot_id"], 3)
        self.assertEqual(binding["fraction_ppm"], 1_000_000)
        self.assertEqual(
            binding["helper_attachment"]["start_token_index"], 7
        )
        self.assertEqual(
            [row["kind"] for row in controller.request_helper_events(
                "request-1"
            )],
            ["ATTACHED", "LEASES_RENEWED", "FRACTION_CHANGED"],
        )

    def test_helper_event_cap_coalesces_renewals_and_keeps_lifecycle_events(
        self,
    ) -> None:
        controller = ModelPlacementController(ModelPlacementPolicy(
            maximum_events=4,
        ))
        controller.record_request_helper_event(
            "request-1", "ATTACHED", 100, {"fraction_ppm": 500_000},
        )
        for index in range(6):
            controller.record_request_helper_event(
                "request-1", "LEASES_RENEWED", 200 + index, {"lease": index},
            )
        events = controller.request_helper_events("request-1")
        self.assertEqual([row["kind"] for row in events], ["ATTACHED", "LEASES_RENEWED"])
        renewal = events[1]
        self.assertEqual(renewal["event_index"], 6)
        self.assertEqual(renewal["first_event_index"], 1)
        self.assertEqual(renewal["renewal_count"], 6)
        self.assertEqual(renewal["first_observed_at_us"], 200)
        self.assertEqual(renewal["observed_at_us"], 205)
        self.assertEqual(renewal["lease"], 5)
        # A renewal of another request keeps both requests' latest lease state.
        controller.record_request_helper_event(
            "request-2", "LEASES_RENEWED", 300, {"lease": 0},
        )
        controller.record_request_helper_event(
            "request-1", "LEASES_RENEWED", 301, {"lease": 6},
        )
        controller.record_request_helper_event(
            "request-1", "ASSISTANCE_DECISION", 900, {"reason": "MEASURED_REJECTION"},
        )
        controller.record_request_helper_event(
            "request-1", "DETACHED", 1_000, {"fallback_outcome": "REQUEST_COMPLETED"},
        )
        events = controller.request_helper_events()
        self.assertEqual(len(events), 4)
        # Renewals are evicted before any lifecycle or reason event.
        self.assertEqual(
            [(row["request_id"], row["kind"]) for row in events],
            [("request-1", "ATTACHED"), ("request-1", "LEASES_RENEWED"),
             ("request-1", "ASSISTANCE_DECISION"), ("request-1", "DETACHED")],
        )
        self.assertEqual([row["event_index"] for row in events], [0, 8, 9, 10])
        self.assertNotIn("renewal_count", events[1])
        self.assertEqual(events[1]["lease"], 6)
        restored = ModelPlacementController(ModelPlacementPolicy(maximum_events=4))
        restored.restore(controller.checkpoint())
        restored.record_request_helper_event(
            "request-1", "DETACHED", 1_100, {"fallback_outcome": "REQUEST_COMPLETED"},
        )
        restored.record_request_helper_event(
            "request-1", "DETACHED", 1_200, {"fallback_outcome": "REQUEST_COMPLETED"},
        )
        events = restored.request_helper_events("request-1")
        # Without renewals to spare, the oldest lifecycle event goes; indices
        # stay monotonic regardless of buffer length.
        self.assertEqual(
            [row["kind"] for row in events],
            ["ASSISTANCE_DECISION", "DETACHED", "DETACHED", "DETACHED"],
        )
        self.assertEqual([row["event_index"] for row in events], [9, 10, 11, 12])

    def test_ready_helper_envelope_binds_after_desktop_acquisition(
        self,
    ) -> None:
        controller = self.controller()
        layout = phone_layout("late")
        state = controller.propose_phone_layout(
            layout,
            workspace_bytes=1024,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=(
                "phone-functionfs", "phone-usb"
            ),
            observed_at_us=100,
        )
        state = controller.begin_phone_layout_transition(
            state.generation,
            ticket_id="late-helper-preparation",
            transition_ids=("load-phone",),
            ready_at_us=200,
            projection_token_sha256=identity("late-projection"),
            workspace_bytes=1024,
            observed_at_us=110,
        )
        state = controller.complete_phone_layout_transition(
            generation=state.generation,
            ticket_id="late-helper-preparation",
            transition_ids=("load-phone",),
            geometry_sha256=layout.geometry_sha256,
            projection_token_sha256=identity("late-projection"),
            finished_at_us=200,
        )
        artifact = layout.shards[0].artifact_sha256
        desktop = identity("late-desktop-placement")
        base_component = identity("late-desktop-component")
        controller.bind_dispatched_request(
            "late-request",
            artifact,
            "desktop-parent",
            base_component,
            0,
            desktop_placement_sha256=desktop,
            kv_cache_owner_id="late-kv-owner",
            sequence_identity="late-sequence",
        )
        controller.mark_request_acquired("late-request")
        controller.bind_request_server_slot(
            "late-request",
            sequence_identity="late-sequence",
            server_slot_id=2,
        )
        helper = RequestHelperEnvelopeBinding(
            route_id="phone-helper",
            operator_plan_sha256=layout.shards[0].operator_plan_sha256,
            desktop_parent_route_id="desktop-parent",
            desktop_placement_sha256=desktop,
            phone_layout_generation=state.generation,
            phone_layout_geometry_sha256=layout.geometry_sha256,
            activation_dtype="f16",
            assisted_layer_mask=1,
            maximum_columns=1024,
            allowed_fractions_ppm=(0, 500_000, 1_000_000),
            phone_session_ids=("HTP0",),
            resource_ids=("phone-htp", "phone-usb"),
        )

        controller.bind_request_helper_envelope(
            "late-request", helper, observed_at_us=210
        )
        controller.attach_request_helper(
            "late-request",
            phone_layout_generation=state.generation,
            phone_layout_geometry_sha256=layout.geometry_sha256,
            resident_component_identity_sha256=(
                state.resident_component_identity_sha256
            ),
            operator_plan_sha256=helper.operator_plan_sha256,
            start_token_index=9,
            fraction_ppm=500_000,
            lease_tokens=("late-helper-lease",),
            lease_reserved_until_us=500,
            observed_at_us=220,
        )

        binding = controller.request_binding("late-request")
        self.assertEqual(binding["base"]["route_id"], "desktop-parent")
        self.assertEqual(binding["base"]["kv_cache_owner_id"], "late-kv-owner")
        self.assertEqual(binding["base"]["server_slot_id"], 2)
        self.assertEqual(binding["helper_attachment"]["start_token_index"], 9)
        self.assertEqual(binding["fraction_ppm"], 500_000)
        self.assertEqual(
            [row["kind"] for row in controller.request_helper_events(
                "late-request"
            )],
            ["ENVELOPE_BOUND", "ATTACHED"],
        )

    def test_zero_work_helper_envelope_follows_ready_layout(self) -> None:
        controller = self.controller()
        current = mixed_phone_layout(("a", "a", "b"))
        proposed = controller.propose_phone_layout(
            current,
            workspace_bytes=1024,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=(
                "phone-functionfs", "phone-usb"
            ),
            observed_at_us=100,
        )
        controller.begin_phone_layout_transition(
            proposed.generation,
            ticket_id="load-ggq",
            transition_ids=("load-ggq",),
            ready_at_us=200,
            projection_token_sha256=identity("load-ggq-projection"),
            workspace_bytes=1024,
            observed_at_us=110,
        )
        ready = controller.complete_phone_layout_transition(
            generation=proposed.generation,
            ticket_id="load-ggq",
            transition_ids=("load-ggq",),
            geometry_sha256=current.geometry_sha256,
            projection_token_sha256=identity("load-ggq-projection"),
            finished_at_us=200,
        )
        artifact = current.shards[2].artifact_sha256
        desktop = identity("inactive-helper-desktop")
        helper = RequestHelperEnvelopeBinding(
            route_id="q-helper",
            operator_plan_sha256=current.shards[2].operator_plan_sha256,
            desktop_parent_route_id="desktop-parent",
            desktop_placement_sha256=desktop,
            phone_layout_generation=ready.generation,
            phone_layout_geometry_sha256=current.geometry_sha256,
            activation_dtype="f16",
            assisted_layer_mask=current.shards[2].layer_mask,
            maximum_columns=1024,
            allowed_fractions_ppm=(0, 500_000, 1_000_000),
            phone_session_ids=("HTP2",),
            resource_ids=("phone-htp", "phone-usb"),
        )
        controller.bind_dispatched_request(
            "inactive-helper-request",
            artifact,
            "desktop-parent",
            identity("inactive-helper-base"),
            0,
            desktop_placement_sha256=desktop,
            kv_cache_owner_id="inactive-helper-kv",
            sequence_identity="inactive-helper-sequence",
            server_slot_id=1,
            helper_envelope=helper,
        )
        controller.mark_request_acquired("inactive-helper-request")
        controller.attach_request_helper(
            "inactive-helper-request",
            phone_layout_generation=ready.generation,
            phone_layout_geometry_sha256=current.geometry_sha256,
            resident_component_identity_sha256=(
                ready.resident_component_identity_sha256
            ),
            operator_plan_sha256=helper.operator_plan_sha256,
            start_token_index=1,
            fraction_ppm=0,
            lease_tokens=(),
            lease_reserved_until_us=None,
            observed_at_us=210,
        )

        replacement = mixed_phone_layout(
            ("a", "b", "b"),
            current_assignments=("a", "a", "b"),
        )
        target = controller.propose_phone_layout(
            replacement,
            workspace_bytes=1024,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=(
                "phone-functionfs", "phone-usb"
            ),
            observed_at_us=200_000,
        )
        controller.begin_phone_layout_transition(
            target.generation,
            ticket_id="replace-htp1",
            transition_ids=("replace-htp1",),
            ready_at_us=201_000,
            projection_token_sha256=identity("replace-htp1-projection"),
            workspace_bytes=1024,
            observed_at_us=200_010,
        )
        target = controller.complete_phone_layout_transition(
            generation=target.generation,
            ticket_id="replace-htp1",
            transition_ids=("replace-htp1",),
            geometry_sha256=replacement.geometry_sha256,
            projection_token_sha256=identity("replace-htp1-projection"),
            finished_at_us=201_000,
        )
        target_shards = tuple(
            row for row in replacement.shards
            if row.artifact_sha256 == artifact
        )
        target_helper = RequestHelperEnvelopeBinding(
            route_id="qq-helper",
            operator_plan_sha256=target_shards[0].operator_plan_sha256,
            desktop_parent_route_id="desktop-parent",
            desktop_placement_sha256=desktop,
            phone_layout_generation=target.generation,
            phone_layout_geometry_sha256=replacement.geometry_sha256,
            activation_dtype="f16",
            assisted_layer_mask=sum(
                row.layer_mask for row in target_shards
            ),
            maximum_columns=1024,
            allowed_fractions_ppm=(0, 500_000, 1_000_000),
            phone_session_ids=("HTP1", "HTP2"),
            resource_ids=("phone-htp", "phone-usb"),
        )
        base = controller.request_binding(
            "inactive-helper-request"
        )["base"]
        rebound = controller.bind_request_helper_envelope(
            "inactive-helper-request",
            target_helper,
            observed_at_us=201_010,
        )
        self.assertEqual(rebound.base.to_json(), base)
        self.assertIsNone(rebound.helper_attachment)
        self.assertEqual(rebound.fraction_ppm, 0)
        controller.attach_request_helper(
            "inactive-helper-request",
            phone_layout_generation=target.generation,
            phone_layout_geometry_sha256=replacement.geometry_sha256,
            resident_component_identity_sha256=(
                target.resident_component_identity_sha256
            ),
            operator_plan_sha256=target_helper.operator_plan_sha256,
            start_token_index=20,
            fraction_ppm=0,
            lease_tokens=(),
            lease_reserved_until_us=None,
            observed_at_us=201_020,
        )
        binding = controller.request_binding("inactive-helper-request")
        self.assertEqual(binding["base"], base)
        self.assertEqual(
            binding["helper_attachment"]["phone_session_ids"],
            ["HTP1", "HTP2"],
        )
        self.assertIn(
            "INACTIVE_ENVELOPE_REPLACED",
            [row["kind"] for row in controller.request_helper_events(
                "inactive-helper-request"
            )],
        )

    def test_phone_layout_replacement_needs_distinct_live_snapshots(
        self,
    ) -> None:
        controller = self.controller()
        current = phone_layout("current")
        state = controller.propose_phone_layout(
            current,
            workspace_bytes=1024,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-functionfs", "phone-usb"),
            observed_at_us=100,
        )
        controller.begin_phone_layout_transition(
            state.generation,
            ticket_id="prepare-current",
            transition_ids=("load-current",),
            ready_at_us=200,
            projection_token_sha256=identity("projection-current"),
            workspace_bytes=1024,
            observed_at_us=110,
        )
        controller.complete_phone_layout_transition(
            generation=state.generation,
            ticket_id="prepare-current",
            transition_ids=("load-current",),
            geometry_sha256=current.geometry_sha256,
            projection_token_sha256=identity("projection-current"),
            finished_at_us=200,
        )
        replacement = phone_layout("replacement")
        first = identity("live-pressure-1")
        self.assertEqual(
            controller.confirm_phone_layout_candidate(
                replacement.geometry_sha256,
                first,
                observed_at_us=300,
            ),
            (False, 1),
        )
        self.assertEqual(
            dict(controller.pending_phone_layout_candidate()),
            {
                "geometry_sha256": replacement.geometry_sha256,
                "snapshot_count": 1,
                "snapshot_sha256": first,
            },
        )
        self.assertEqual(
            controller.confirm_phone_layout_candidate(
                replacement.geometry_sha256,
                first,
                observed_at_us=301,
            ),
            (False, 1),
        )
        self.assertEqual(
            controller.confirm_phone_layout_candidate(
                replacement.geometry_sha256,
                identity("live-pressure-2"),
                observed_at_us=400,
            ),
            (True, 2),
        )
        self.assertIsNone(controller.pending_phone_layout_candidate())

    def test_phone_layout_decision_log_is_deterministic(self) -> None:
        def events() -> tuple[dict[str, object], ...]:
            controller = self.controller()
            layout = phone_layout("deterministic")
            controller.confirm_phone_layout_candidate(
                layout.geometry_sha256,
                identity("deterministic-snapshot-1"),
                observed_at_us=100,
            )
            controller.confirm_phone_layout_candidate(
                layout.geometry_sha256,
                identity("deterministic-snapshot-2"),
                observed_at_us=200,
            )
            controller.propose_phone_layout(
                layout,
                workspace_bytes=1024,
                shared_compute_resource_id="phone-htp",
                shared_transport_resource_ids=(
                    "phone-functionfs", "phone-usb"
                ),
                observed_at_us=200,
                selection_reason=(
                    "PHONE_RESIDENCY_SESSION_MARGINAL_GAIN"
                ),
                queue_work_by_artifact={
                    layout.shards[0].artifact_sha256: 10
                },
                queue_benefit_uj=1000,
                transition_cost_uj=100,
                switching_margin_uj=10,
            )
            return tuple(
                dict(row) for row in controller.phone_layout_events()
            )

        first = events()
        second = events()
        self.assertEqual(first, second)
        self.assertTrue(all(
            row["event_sha256"] == canonical_sha256({
                key: value
                for key, value in row.items()
                if key != "event_sha256"
            })
            for row in first
        ))

    def test_replacement_waits_for_all_admitted_helper_bindings(self) -> None:
        controller = self.controller()
        current = phone_layout("draining")
        state = controller.propose_phone_layout(
            current,
            workspace_bytes=1024,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-functionfs", "phone-usb"),
            observed_at_us=100,
        )
        controller.begin_phone_layout_transition(
            state.generation,
            ticket_id="prepare-draining",
            transition_ids=("load-draining",),
            ready_at_us=200,
            projection_token_sha256=identity("projection-draining"),
            workspace_bytes=1024,
            observed_at_us=110,
        )
        state = controller.complete_phone_layout_transition(
            generation=state.generation,
            ticket_id="prepare-draining",
            transition_ids=("load-draining",),
            geometry_sha256=current.geometry_sha256,
            projection_token_sha256=identity("projection-draining"),
            finished_at_us=200,
        )
        artifact = current.shards[0].artifact_sha256
        desktop = identity("draining-desktop")
        helper = RequestHelperEnvelopeBinding(
            route_id="desktop-parent-helper",
            operator_plan_sha256=current.shards[0].operator_plan_sha256,
            desktop_parent_route_id="desktop-parent",
            desktop_placement_sha256=desktop,
            phone_layout_generation=state.generation,
            phone_layout_geometry_sha256=current.geometry_sha256,
            activation_dtype="f16",
            assisted_layer_mask=1,
            maximum_columns=1024,
            allowed_fractions_ppm=(0, 500_000),
            phone_session_ids=("HTP0",),
            resource_ids=("phone-htp", "phone-usb"),
        )
        controller.bind_dispatched_request(
            "active-request",
            artifact,
            "desktop-parent",
            identity("draining-base-component"),
            0,
            desktop_placement_sha256=desktop,
            helper_envelope=helper,
        )
        controller.mark_request_acquired("active-request")
        controller.attach_request_helper(
            "active-request",
            phone_layout_generation=state.generation,
            phone_layout_geometry_sha256=current.geometry_sha256,
            resident_component_identity_sha256=(
                state.resident_component_identity_sha256
            ),
            operator_plan_sha256=helper.operator_plan_sha256,
            start_token_index=4,
            fraction_ppm=500_000,
            lease_tokens=("active-helper-lease",),
            lease_reserved_until_us=500,
            observed_at_us=250,
        )
        controller.bind_dispatched_request(
            "queued-request",
            artifact,
            "desktop-parent",
            identity("queued-base-component"),
            0,
            desktop_placement_sha256=desktop,
            helper_envelope=helper,
        )
        replacement = phone_layout("next")
        target = controller.propose_phone_layout(
            replacement,
            workspace_bytes=1024,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-functionfs", "phone-usb"),
            observed_at_us=200_000,
        )
        self.assertEqual(controller.ready_phone_layout().state, "READY")
        self.assertEqual(target.state, "PROPOSED")
        self.assertEqual(
            controller.phone_layout_transition_blockers(target.generation),
            ("active-request",),
        )
        controller.record_request_helper_work(
            "active-request", 8, observed_at_us=200_010
        )
        self.assertEqual(
            controller.phone_layout_transition_blockers(target.generation),
            ("active-request",),
        )
        with self.assertRaisesRegex(
            ModelPlacementControllerError, "admitted blockers"
        ):
            controller.begin_phone_layout_transition(
                target.generation,
                ticket_id="prepare-next",
                transition_ids=("load-next",),
                ready_at_us=200_200,
                projection_token_sha256=identity("projection-next"),
                workspace_bytes=1024,
                observed_at_us=200_100,
            )
        controller.update_request_fraction(
            "active-request", 0, observed_at_us=200_050
        )
        self.assertEqual(
            controller.phone_layout_transition_blockers(target.generation),
            ("active-request",),
        )
        controller.release_request("active-request")
        self.assertEqual(
            controller.phone_layout_transition_blockers(target.generation),
            (),
        )
        controller.release_request("queued-request")
        self.assertEqual(
            controller.phone_layout_transition_blockers(target.generation),
            (),
        )
        transitioning = controller.begin_phone_layout_transition(
            target.generation,
            ticket_id="prepare-next",
            transition_ids=("load-next",),
            ready_at_us=200_200,
            projection_token_sha256=identity("projection-next"),
            workspace_bytes=1024,
            observed_at_us=200_100,
        )
        self.assertEqual(transitioning.state, "PREPARING")
        self.assertEqual(controller.ready_phone_layout().state, "DRAINING")
        proposed = controller.reset_phone_layout_transition(
            "prepare-next",
            generation=target.generation,
            projection_token_sha256=identity("projection-next"),
            observed_at_us=200_150,
            reason="retry",
        )
        self.assertEqual(proposed.state, "PROPOSED")
        self.assertEqual(controller.ready_phone_layout().state, "READY")
        controller.begin_phone_layout_transition(
            target.generation,
            ticket_id="prepare-next",
            transition_ids=("load-next",),
            ready_at_us=200_300,
            projection_token_sha256=identity("projection-next-retry"),
            workspace_bytes=1024,
            observed_at_us=200_200,
        )
        controller.fail_phone_layout_transition(
            "prepare-next",
            generation=target.generation,
            projection_token_sha256=identity("projection-next-retry"),
            failed_at_us=200_250,
            reason="injected",
        )
        self.assertIsNone(controller.target_phone_layout())
        self.assertEqual(controller.ready_phone_layout().state, "READY")

    def test_active_ggg_helper_quiesces_and_rebinds_to_gg(self) -> None:
        (
            controller, _current, target_layout, _ready, target,
            _helper, desktop,
        ) = self.active_ggg_to_ggq()
        rebind = controller.request_helper_rebind(
            "gemma-request", target.generation, observed_at_us=200_010
        )
        self.assertEqual(rebind.retained_session_ids, ("HTP0", "HTP1"))
        self.assertEqual(rebind.removed_session_ids, ("HTP2",))
        self.assertEqual(
            controller.phone_layout_transition_blockers(target.generation),
            ("gemma-request",),
        )
        controller.update_request_fraction(
            "gemma-request", 0, observed_at_us=200_020
        )
        controller.mark_request_helper_rebind_quiesced(
            "gemma-request", target.generation, observed_at_us=200_030
        )
        self.assertEqual(
            controller.phone_layout_transition_blockers(target.generation),
            (),
        )
        controller.begin_phone_layout_transition(
            target.generation,
            ticket_id="replace-htp2",
            transition_ids=("replace-htp2",),
            ready_at_us=201_000,
            projection_token_sha256=identity("replace-htp2-projection"),
            workspace_bytes=1024,
            observed_at_us=200_040,
        )
        ready = controller.complete_phone_layout_transition(
            generation=target.generation,
            ticket_id="replace-htp2",
            transition_ids=("replace-htp2",),
            geometry_sha256=target_layout.geometry_sha256,
            projection_token_sha256=identity("replace-htp2-projection"),
            finished_at_us=201_000,
        )
        gemma_shards = tuple(
            row for row in target_layout.shards
            if row.artifact_sha256 == target_layout.shards[0].artifact_sha256
        )
        envelope = RequestHelperEnvelopeBinding(
            route_id="gg-helper",
            operator_plan_sha256=identity("gg-helper-plan"),
            desktop_parent_route_id="desktop-parent",
            desktop_placement_sha256=desktop,
            phone_layout_generation=ready.generation,
            phone_layout_geometry_sha256=target_layout.geometry_sha256,
            activation_dtype="f16",
            assisted_layer_mask=sum(row.layer_mask for row in gemma_shards),
            maximum_columns=1024,
            allowed_fractions_ppm=(0, 500_000, 750_000, 1_000_000),
            phone_session_ids=("HTP0", "HTP1"),
            resource_ids=("phone-htp", "phone-usb"),
        )
        before = controller.request_binding("gemma-request")["base"]
        controller.commit_request_helper_rebind(
            "gemma-request",
            envelope,
            resident_component_identity_sha256=identity("gg-component"),
            start_token_index=50,
            observed_at_us=201_010,
        )
        after = controller.request_binding("gemma-request")
        self.assertEqual(after["base"], before)
        self.assertEqual(
            after["helper_attachment"]["phone_session_ids"],
            ["HTP0", "HTP1"],
        )
        self.assertEqual(after["helper_attachment"]["completed_phone_calls"], 30)
        self.assertEqual(after["fraction_ppm"], 0)

    def test_rebind_cannot_clear_blocker_before_baseline_ack(self) -> None:
        controller, _current, _target_layout, _ready, target, _helper, _desktop = (
            self.active_ggg_to_ggq()
        )
        controller.request_helper_rebind(
            "gemma-request", target.generation, observed_at_us=200_010
        )
        with self.assertRaisesRegex(
            ModelPlacementControllerError, "not safely quiesced"
        ):
            controller.mark_request_helper_rebind_quiesced(
                "gemma-request",
                target.generation,
                observed_at_us=200_020,
            )
        self.assertEqual(
            controller.phone_layout_transition_blockers(target.generation),
            ("gemma-request",),
        )

    def test_acknowledged_session_mask_keeps_retained_assistance(self) -> None:
        controller, current, _layout, ready, target, helper, _desktop = (
            self.active_ggg_to_ggq()
        )
        previous_attachment = controller.request_binding("gemma-request")[
            "helper_attachment"
        ]
        previous_states = {
            row.session_id: row for row in controller.phone_session_states()
        }
        selected = target.layout.changed_session_ids[0]
        rebind = controller.request_helper_rebind(
            "gemma-request", target.generation, observed_at_us=200_010
        )
        self.assertEqual(rebind.removed_session_ids, (selected,))
        drain_policy = identity("retained-session-drain")
        controller.bind_request_helper_rebind_drain_policy(
            "gemma-request",
            target.generation,
            drain_policy,
            observed_at_us=200_020,
        )
        with self.assertRaisesRegex(
            ModelPlacementControllerError, "admitted blockers"
        ):
            controller.begin_phone_layout_transition(
                target.generation,
                ticket_id="replace-selected-session",
                transition_ids=("replace-selected-session",),
                ready_at_us=201_000,
                projection_token_sha256=identity("masked-drain-projection"),
                workspace_bytes=1024,
                observed_at_us=200_025,
            )
        self.assertEqual(controller.request_binding("gemma-request")[
            "helper_attachment"
        ], previous_attachment)
        with self.assertRaisesRegex(
            ModelPlacementControllerError, "drain acknowledgement differs"
        ):
            controller.mark_request_helper_rebind_quiesced(
                "gemma-request", target.generation,
                observed_at_us=200_026,
                allowed_session_ids=rebind.retained_session_ids,
                drain_policy_sha256=identity("unacknowledged-drain"),
            )
        self.assertEqual(controller.request_binding("gemma-request")[
            "helper_attachment"
        ], previous_attachment)
        controller.mark_request_helper_rebind_quiesced(
            "gemma-request",
            target.generation,
            observed_at_us=200_030,
            allowed_session_ids=rebind.retained_session_ids,
            drain_policy_sha256=drain_policy,
        )
        rebound = controller.bind_request_helper_rebind_drain_policy(
            "gemma-request",
            target.generation,
            drain_policy,
            observed_at_us=200_040,
        )
        self.assertEqual(rebound.state, "QUIESCED")
        self.assertEqual(rebound.drain_policy_sha256, drain_policy)

        binding = controller.request_binding("gemma-request")
        self.assertEqual(binding["fraction_ppm"], 750_000)
        for field in ("lease_tokens", "lease_reserved_until_us"):
            self.assertEqual(binding["helper_attachment"][field],
                             previous_attachment[field])
        self.assertEqual(
            tuple(binding["helper_attachment"]["allowed_session_ids"]),
            rebind.retained_session_ids,
        )
        self.assertNotIn(
            selected,
            binding["helper_attachment"]["allowed_session_ids"],
        )
        self.assertEqual(
            controller.phone_layout_transition_blockers(target.generation),
            (),
        )

        controller.begin_phone_layout_transition(
            target.generation,
            ticket_id="replace-selected-session",
            transition_ids=("replace-selected-session",),
            ready_at_us=201_000,
            projection_token_sha256=identity("masked-drain-projection"),
            workspace_bytes=1024,
            observed_at_us=200_040,
        )
        states = {
            row.session_id: row for row in controller.phone_session_states()
        }
        self.assertEqual(states[selected].state, "LOADING")
        self.assertTrue(all(
            states[session_id].state == "READY"
            for session_id in rebind.retained_session_ids
        ))
        for session_id in rebind.retained_session_ids:
            self.assertEqual(states[session_id], previous_states[session_id])
        self.assertTrue(all(
            "gemma-request" in states[session_id].active_helper_references
            for session_id in rebind.retained_session_ids
        ))
        repeated = controller.attach_request_helper(
            "gemma-request",
            phone_layout_generation=ready.generation,
            phone_layout_geometry_sha256=current.geometry_sha256,
            resident_component_identity_sha256=(
                ready.resident_component_identity_sha256
            ),
            operator_plan_sha256=helper.operator_plan_sha256,
            start_token_index=60,
            fraction_ppm=750_000,
            lease_tokens=("phone-lease",),
            lease_reserved_until_us=2_000_000,
            observed_at_us=200_050,
        )
        self.assertEqual(
            repeated.helper_attachment.allowed_session_ids,
            rebind.retained_session_ids,
        )
        controller.record_request_helper_work(
            "gemma-request", 1, observed_at_us=200_060
        )
        self.assertEqual(
            controller.request_binding("gemma-request")
                ["helper_attachment"]["completed_phone_calls"],
            31,
        )

    def test_rebind_preserves_base_ticket_kv_slot_and_sequence(self) -> None:
        controller, _current, _target_layout, _ready, target, _helper, _desktop = (
            self.active_ggg_to_ggq()
        )
        before = controller.request_binding("gemma-request")["base"]
        controller.request_helper_rebind(
            "gemma-request", target.generation, observed_at_us=200_010
        )
        controller.update_request_fraction(
            "gemma-request", 0, observed_at_us=200_020
        )
        controller.mark_request_helper_rebind_quiesced(
            "gemma-request", target.generation, observed_at_us=200_030
        )
        self.assertEqual(
            controller.request_binding("gemma-request")["base"], before
        )

    def test_rebind_rollback_restores_binding_and_all_leases(self) -> None:
        controller, _current, _target_layout, _ready, target, _helper, desktop = (
            self.active_ggg_to_ggq()
        )
        checkpoint = controller.checkpoint()
        before = controller.request_binding("gemma-request")
        controller.request_helper_rebind(
            "gemma-request", target.generation, observed_at_us=200_010
        )
        controller.restore(checkpoint)
        self.assertEqual(controller.request_binding("gemma-request"), before)
        self.assertIsNone(
            controller.request_helper_rebind_state("gemma-request")
        )
        self.assertEqual(
            controller.request_binding("gemma-request")["base"]
                ["desktop_placement_sha256"],
            desktop,
        )

    def test_ggg_to_ggq_changes_only_htp2(self) -> None:
        (
            controller, current, target_layout, _ready, target,
            _helper, _desktop,
        ) = self.active_ggg_to_ggq()
        rebind = controller.request_helper_rebind(
            "gemma-request", target.generation, observed_at_us=200_010
        )
        current_by_session = {
            row.session_id: row for row in current.shards
        }
        target_by_session = {
            row.session_id: row for row in target_layout.shards
        }
        self.assertEqual(rebind.retained_session_ids, ("HTP0", "HTP1"))
        self.assertEqual(rebind.removed_session_ids, ("HTP2",))
        self.assertEqual(target_layout.changed_session_ids, ("HTP2",))
        for session_id in rebind.retained_session_ids:
            self.assertEqual(
                current_by_session[session_id],
                target_by_session[session_id],
            )

    def test_ready_layout_is_used_as_current_marginal_baseline(self) -> None:
        controller = self.controller()
        current = mixed_phone_layout(
            ("a", "a", "a"),
            benefit_by_session=(4_000, 4_000, 4_000),
        )
        reevaluated = mixed_phone_layout(
            ("a", "a", "a"),
            current_assignments=("a", "a", "a"),
            benefit_by_session=(4_000, 4_000, 4_000),
        )
        lower_value = mixed_phone_layout(
            ("a", "a", "b"),
            current_assignments=("a", "a", "a"),
            benefit_by_session=(4_000, 4_000, 3_500),
            transition_uj=100,
        )
        selected, reason, evidence = (
            controller.select_phone_layout_candidate(
                (reevaluated, lower_value),
                current_layout=current,
                minimum_energy_saving_ppm=0,
            )
        )
        self.assertEqual(selected.geometry_sha256, current.geometry_sha256)
        self.assertEqual(reason, "PHONE_RESIDENCY_SESSION_HYSTERESIS")
        self.assertLess(evidence[0].gain_over_current_uj, 0)

    def test_higher_incremental_gain_replaces_profitable_session(self) -> None:
        controller = ModelPlacementController(ModelPlacementPolicy(
            phone_session_hysteresis_uj=30,
            phone_session_latency_penalty_uj=20,
            phone_session_interference_energy_uj=40,
            phone_session_safety_margin_uj=10,
        ))
        current = mixed_phone_layout(("a", "a", "a"))
        reevaluated = mixed_phone_layout(
            ("a", "a", "a"),
            current_assignments=("a", "a", "a"),
            benefit_by_session=(1_000, 1_000, 500),
        )
        target = mixed_phone_layout(
            ("a", "a", "b"),
            current_assignments=("a", "a", "a"),
            benefit_by_session=(1_000, 1_000, 4_500),
            transition_uj=100,
        )

        selected, reason, evidence = (
            controller.select_phone_layout_candidate(
                (reevaluated, target),
                current_layout=current,
                minimum_energy_saving_ppm=100_000,
            )
        )

        self.assertEqual(selected.geometry_sha256, target.geometry_sha256)
        self.assertEqual(reason, "PHONE_RESIDENCY_SESSION_MARGINAL_GAIN")
        gain = evidence[0]
        self.assertEqual(gain.current_warm_energy_saved_uj, 500)
        self.assertEqual(gain.proposed_warm_energy_saved_uj, 4_500)
        self.assertEqual(gain.session_load_energy_uj, 100)
        self.assertEqual(gain.session_eviction_energy_uj, 0)
        self.assertEqual(gain.latency_penalty_uj, 20)
        self.assertEqual(gain.interference_energy_uj, 40)
        self.assertEqual(gain.safety_margin_uj, 50)
        self.assertEqual(gain.gain_over_current_uj, 3_790)

    def test_learning_expansion_adds_coverage_without_evicting_resident_shards(
        self,
    ) -> None:
        controller = ModelPlacementController()
        current = partial_phone_layout({"HTP0": "a"})
        expansion = partial_phone_layout(
            {"HTP0": "a", "HTP1": "a"}, current=current,
            benefit_by_session={"HTP0": 1_000, "HTP1": 900},
        )
        self.assertEqual(expansion.changed_session_ids, ("HTP1",))
        self.assertEqual(expansion.shards[0], current.shards[0])
        reshaped = partial_phone_layout(
            {"HTP0": "a", "HTP1": "a"}, current=current, reshape=("HTP0",),
            benefit_by_session={"HTP0": 1_500, "HTP1": 900},
        )
        self.assertEqual(reshaped.changed_session_ids, ("HTP0", "HTP1"))
        # Demand for "a" is covered, so reshaping the resident shard stays
        # retained, but adding a session for "a" while keeping the resident
        # shard byte-identical is allowed and is named as an expansion.
        selected, reason, _ = controller.select_phone_layout_candidate(
            (reshaped,), current_layout=current,
            minimum_energy_saving_ppm=0,
        )
        self.assertEqual(selected, current)
        self.assertEqual(reason, "PHONE_RESIDENCY_LEARNING_RETAINED")
        selected, reason, _ = controller.select_phone_layout_candidate(
            (reshaped, expansion), current_layout=current,
            minimum_energy_saving_ppm=0,
        )
        self.assertEqual(selected, expansion)
        self.assertEqual(reason, "PHONE_RESIDENCY_LEARNING_EXPANSION")
        # A further single-session expansion from the two-session layout.
        third = partial_phone_layout(
            {"HTP0": "a", "HTP1": "a", "HTP2": "a"}, current=expansion,
            benefit_by_session={"HTP0": 1_000, "HTP1": 900, "HTP2": 800},
        )
        self.assertEqual(third.changed_session_ids, ("HTP2",))
        selected, reason, _ = controller.select_phone_layout_candidate(
            (third,), current_layout=expansion, minimum_energy_saving_ppm=0,
        )
        self.assertEqual(selected, third)
        self.assertEqual(reason, "PHONE_RESIDENCY_LEARNING_EXPANSION")

    def test_learning_expansion_keeps_normal_checks(self) -> None:
        controller = ModelPlacementController()
        current = partial_phone_layout({"HTP0": "a"})
        # Expansion still proceeds one session at a time.
        two_at_once = partial_phone_layout(
            {"HTP0": "a", "HTP1": "a", "HTP2": "a"}, current=current,
            benefit_by_session={"HTP0": 1_000, "HTP1": 900, "HTP2": 800},
        )
        self.assertEqual(two_at_once.changed_session_ids, ("HTP1", "HTP2"))
        selected, reason, _ = controller.select_phone_layout_candidate(
            (two_at_once,), current_layout=current, minimum_energy_saving_ppm=0,
        )
        self.assertEqual(selected, current)
        self.assertEqual(reason, "PHONE_RESIDENCY_LEARNING_RETAINED")
        expansion = partial_phone_layout(
            {"HTP0": "a", "HTP1": "a"}, current=current,
            benefit_by_session={"HTP0": 1_000, "HTP1": 900},
        )
        blocked = PhoneLayoutRequestImpact(
            request_id="request-1", remaining_tokens=50, current_layer_mask=1,
            retained_layer_mask=1, evidence_reused=True, verification_feasible=False,
            verification_tokens=8,
        )
        selected, reason, _ = controller.select_phone_layout_candidate(
            (expansion,), current_layout=current, minimum_energy_saving_ppm=0,
            request_impacts_by_geometry={expansion.geometry_sha256: (blocked,)},
        )
        self.assertEqual(selected, current)
        self.assertEqual(reason, "PHONE_RESIDENCY_REVALIDATION_UNAFFORDABLE")

    def test_learning_replaces_until_each_demanded_artifact_is_covered(
        self,
    ) -> None:
        controller = ModelPlacementController()
        current = mixed_phone_layout(("a", "a", "a"))
        first_probe = replace(
            mixed_phone_layout(
                ("b", "a", "a"),
                current_assignments=("a", "a", "a"),
                benefit_by_session=(4_000, 1_000, 1_000),
            ),
            objective_kind="queue_rough_compute_ops",
        )
        expanded_probe = replace(
            mixed_phone_layout(
                ("b", "b", "a"),
                current_assignments=("b", "a", "a"),
                benefit_by_session=(4_000, 4_000, 1_000),
            ),
            objective_kind="queue_rough_compute_ops",
        )

        selected, reason, _ = controller.select_phone_layout_candidate(
            (first_probe,),
            current_layout=current,
            minimum_energy_saving_ppm=0,
        )
        self.assertEqual(selected, first_probe)
        self.assertEqual(reason, "PHONE_RESIDENCY_LEARNING_EXPLORATION")

        selected, reason, _ = controller.select_phone_layout_candidate(
            (expanded_probe,),
            current_layout=first_probe,
            minimum_energy_saving_ppm=0,
        )
        self.assertEqual(selected, first_probe)
        self.assertEqual(reason, "PHONE_RESIDENCY_LEARNING_RETAINED")

    def test_transition_failure_keeps_both_requests_on_desktop(self) -> None:
        (
            controller, _current, _target_layout, _ready, target,
            _helper, _desktop,
        ) = self.active_ggg_to_ggq()
        controller.bind_dispatched_request(
            "qwen-request",
            target.layout.shards[-1].artifact_sha256,
            "desktop-qwen",
            identity("desktop-qwen-component"),
            0,
        )
        controller.mark_request_acquired("qwen-request")
        controller.request_helper_rebind(
            "gemma-request", target.generation, observed_at_us=200_010
        )
        controller.update_request_fraction(
            "gemma-request", 0, observed_at_us=200_020
        )
        controller.mark_request_helper_rebind_quiesced(
            "gemma-request", target.generation, observed_at_us=200_030
        )
        controller.begin_phone_layout_transition(
            target.generation,
            ticket_id="replace-htp2",
            transition_ids=("replace-htp2",),
            ready_at_us=201_000,
            projection_token_sha256=identity("failure-projection"),
            workspace_bytes=1024,
            observed_at_us=200_040,
        )
        controller.fail_phone_layout_transition(
            "replace-htp2",
            generation=target.generation,
            projection_token_sha256=identity("failure-projection"),
            failed_at_us=200_500,
            reason="injected",
        )
        restored_states = {
            row.session_id: row
            for row in controller.phone_session_states()
        }
        self.assertEqual(
            {key: row.state for key, row in restored_states.items()},
            {"HTP0": "READY", "HTP1": "READY", "HTP2": "READY"},
        )
        self.assertEqual(
            {
                key: row.session_generation
                for key, row in restored_states.items()
            },
            {"HTP0": 1, "HTP1": 1, "HTP2": 1},
        )
        controller.cancel_request_helper_rebind(
            "gemma-request",
            observed_at_us=200_500,
            reason="PHONE_LAYOUT_TRANSITION_FAILED",
        )
        controller.detach_request_helper(
            "gemma-request",
            fallback_outcome="PHONE_LAYOUT_TRANSITION_FAILED",
            observed_at_us=200_500,
        )
        self.assertEqual(
            controller.request_binding("gemma-request")["fraction_ppm"],
            0,
        )
        self.assertEqual(
            controller.request_binding("qwen-request")["fraction_ppm"],
            0,
        )
        self.assertEqual(controller.ready_phone_layout().state, "READY")

    def test_active_remaining_decode_work_is_monotonic(self) -> None:
        controller = self.controller()
        controller.bind_dispatched_request(
            "decode-work",
            identity("decode-work-artifact"),
            "desktop-parent",
            identity("decode-work-component"),
            0,
            output_tokens=100,
        )
        controller.mark_request_acquired("decode-work")

        self.assertEqual(
            controller.remaining_request_decode_tokens(
                "decode-work", 100
            ),
            100,
        )
        self.assertEqual(
            controller.record_request_decode_progress("decode-work", 30),
            30,
        )
        self.assertEqual(
            controller.remaining_request_decode_tokens(
                "decode-work", 100
            ),
            70,
        )
        self.assertEqual(
            controller.record_request_decode_progress("decode-work", 20),
            30,
        )
        self.assertEqual(
            controller.remaining_request_decode_tokens(
                "decode-work", 100
            ),
            70,
        )
        controller.record_request_decode_progress("decode-work", 100)
        self.assertEqual(
            controller.remaining_request_decode_tokens(
                "decode-work", 100
            ),
            0,
        )

    def test_one_session_replacement_preserves_active_mixed_helpers(
        self,
    ) -> None:
        controller = self.controller()
        current = mixed_phone_layout(("a", "a", "a"))
        mixed = mixed_phone_layout(
            ("a", "a", "b"),
            current_assignments=("a", "a", "a"),
            benefit_by_session=(1000, 1000, 4500),
            transition_uj=500,
        )
        initial = controller.propose_phone_layout(
            current,
            workspace_bytes=1024,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=(
                "phone-functionfs", "phone-usb"
            ),
            observed_at_us=100,
        )
        controller.begin_phone_layout_transition(
            initial.generation,
            ticket_id="load-a",
            transition_ids=("load-a-shards",),
            ready_at_us=200,
            projection_token_sha256=identity("load-a-projection"),
            workspace_bytes=1024,
            observed_at_us=110,
        )
        ready_a = controller.complete_phone_layout_transition(
            generation=initial.generation,
            ticket_id="load-a",
            transition_ids=("load-a-shards",),
            geometry_sha256=current.geometry_sha256,
            projection_token_sha256=identity("load-a-projection"),
            finished_at_us=200,
        )
        self.assertIsNotNone(ready_a)
        self.assertEqual(
            {
                row.session_id: row.session_generation
                for row in controller.phone_session_states()
            },
            {"HTP0": 1, "HTP1": 1, "HTP2": 1},
        )
        selected, reason, gains = controller.select_phone_layout_candidate(
            (mixed,),
            current_layout=current,
            minimum_energy_saving_ppm=0,
        )
        self.assertEqual(selected, mixed)
        self.assertEqual(reason, "PHONE_RESIDENCY_SESSION_MARGINAL_GAIN")
        self.assertEqual(mixed.changed_session_ids, ("HTP2",))
        self.assertEqual([row.session_id for row in gains], ["HTP2"])
        self.assertGreater(gains[0].gain_over_current_uj, 0)

        artifact_a = current.shards[0].artifact_sha256
        artifact_b = mixed.shards[2].artifact_sha256
        desktop_a = identity("mixed-desktop-a")
        desktop_b = identity("mixed-desktop-b")
        helper_a = RequestHelperEnvelopeBinding(
            route_id="helper-a",
            operator_plan_sha256=current.shards[0].operator_plan_sha256,
            desktop_parent_route_id="desktop-a",
            desktop_placement_sha256=desktop_a,
            phone_layout_generation=ready_a.generation,
            phone_layout_geometry_sha256=current.geometry_sha256,
            activation_dtype="f16",
            assisted_layer_mask=(1 << 0) | (1 << 1),
            maximum_columns=1024,
            allowed_fractions_ppm=(0, 500_000, 1_000_000),
            phone_session_ids=("HTP0", "HTP1"),
            resource_ids=("phone-htp", "phone-usb"),
        )
        controller.bind_dispatched_request(
            "request-a",
            artifact_a,
            "desktop-a",
            identity("mixed-base-a"),
            0,
            desktop_placement_sha256=desktop_a,
            kv_cache_owner_id="kv-a",
            sequence_identity="sequence-a",
            helper_envelope=helper_a,
        )
        controller.mark_request_acquired("request-a")
        controller.attach_request_helper(
            "request-a",
            phone_layout_generation=ready_a.generation,
            phone_layout_geometry_sha256=current.geometry_sha256,
            resident_component_identity_sha256=(
                ready_a.resident_component_identity_sha256
            ),
            operator_plan_sha256=helper_a.operator_plan_sha256,
            start_token_index=4,
            fraction_ppm=500_000,
            lease_tokens=("lease-a",),
            lease_reserved_until_us=5_000_000,
            observed_at_us=250,
        )
        controller.record_request_helper_work(
            "request-a", 10, observed_at_us=260
        )

        controller.bind_dispatched_request(
            "request-b",
            artifact_b,
            "desktop-b",
            identity("mixed-base-b"),
            0,
            desktop_placement_sha256=desktop_b,
            kv_cache_owner_id="kv-b",
            sequence_identity="sequence-b",
        )
        controller.mark_request_acquired("request-b")
        self.assertEqual(
            controller.request_binding("request-b")["base"]["route_id"],
            "desktop-b",
        )

        target = controller.propose_phone_layout(
            mixed,
            workspace_bytes=1024,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=(
                "phone-functionfs", "phone-usb"
            ),
            observed_at_us=3_000_000,
        )
        self.assertEqual(
            {
                row.session_id: row.session_generation
                for row in target.session_identities
            },
            {"HTP0": 1, "HTP1": 1, "HTP2": 2},
        )
        self.assertEqual(
            controller.phone_layout_transition_blockers(target.generation),
            (),
        )
        controller.begin_phone_layout_transition(
            target.generation,
            ticket_id="replace-htp2",
            transition_ids=("replace-htp2",),
            ready_at_us=3_100_000,
            projection_token_sha256=identity("replace-htp2-projection"),
            workspace_bytes=1024,
            observed_at_us=3_000_100,
        )
        states_during_load = {
            row.session_id: row
            for row in controller.phone_session_states()
        }
        self.assertEqual(states_during_load["HTP0"].state, "READY")
        self.assertEqual(states_during_load["HTP1"].state, "READY")
        self.assertEqual(states_during_load["HTP2"].state, "LOADING")
        self.assertEqual(
            states_during_load["HTP0"].active_helper_references,
            ("request-a",),
        )
        self.assertEqual(
            states_during_load["HTP1"].active_helper_references,
            ("request-a",),
        )
        self.assertTrue(controller.request_helper_layout_is_usable(
            "request-a", ready_a.generation, current.geometry_sha256
        ))
        controller.bind_dispatched_request(
            "request-a-late",
            artifact_a,
            "desktop-a",
            identity("mixed-base-a-late"),
            0,
            desktop_placement_sha256=desktop_a,
            kv_cache_owner_id="kv-a-late",
            sequence_identity="sequence-a-late",
            helper_envelope=helper_a,
        )
        controller.mark_request_acquired("request-a-late")
        controller.attach_request_helper(
            "request-a-late",
            phone_layout_generation=ready_a.generation,
            phone_layout_geometry_sha256=current.geometry_sha256,
            resident_component_identity_sha256=(
                ready_a.resident_component_identity_sha256
            ),
            operator_plan_sha256=helper_a.operator_plan_sha256,
            start_token_index=6,
            fraction_ppm=500_000,
            lease_tokens=("lease-a-late",),
            lease_reserved_until_us=5_000_000,
            observed_at_us=3_000_110,
        )
        self.assertTrue(controller.request_helper_layout_is_usable(
            "request-a-late", ready_a.generation,
            current.geometry_sha256,
        ))
        ready_mixed = controller.complete_phone_layout_transition(
            generation=target.generation,
            ticket_id="replace-htp2",
            transition_ids=("replace-htp2",),
            geometry_sha256=mixed.geometry_sha256,
            projection_token_sha256=identity(
                "replace-htp2-projection"
            ),
            finished_at_us=3_100_000,
        )
        self.assertIsNotNone(ready_mixed)
        self.assertEqual(
            {
                row.session_id: row.session_generation
                for row in controller.phone_session_states()
            },
            {"HTP0": 1, "HTP1": 1, "HTP2": 2},
        )
        self.assertTrue(controller.request_helper_layout_is_usable(
            "request-a", ready_a.generation, current.geometry_sha256
        ))
        self.assertTrue(controller.request_helper_layout_is_usable(
            "request-a-late", ready_a.generation,
            current.geometry_sha256,
        ))

        helper_b = RequestHelperEnvelopeBinding(
            route_id="helper-b",
            operator_plan_sha256=mixed.shards[2].operator_plan_sha256,
            desktop_parent_route_id="desktop-b",
            desktop_placement_sha256=desktop_b,
            phone_layout_generation=ready_mixed.generation,
            phone_layout_geometry_sha256=mixed.geometry_sha256,
            activation_dtype="f16",
            assisted_layer_mask=1 << 2,
            maximum_columns=1024,
            allowed_fractions_ppm=(0, 500_000, 1_000_000),
            phone_session_ids=("HTP2",),
            resource_ids=("phone-htp", "phone-usb"),
        )
        controller.bind_request_helper_envelope(
            "request-b", helper_b, observed_at_us=3_100_010
        )
        controller.attach_request_helper(
            "request-b",
            phone_layout_generation=ready_mixed.generation,
            phone_layout_geometry_sha256=mixed.geometry_sha256,
            resident_component_identity_sha256=(
                ready_mixed.resident_component_identity_sha256
            ),
            operator_plan_sha256=helper_b.operator_plan_sha256,
            start_token_index=8,
            fraction_ppm=500_000,
            lease_tokens=("lease-b",),
            lease_reserved_until_us=5_000_000,
            observed_at_us=3_100_020,
        )
        controller.record_request_helper_work(
            "request-b", 5, observed_at_us=3_100_030
        )
        binding_a = controller.request_binding("request-a")
        binding_b = controller.request_binding("request-b")
        self.assertEqual(binding_a["base"]["kv_cache_owner_id"], "kv-a")
        self.assertEqual(binding_a["helper_attachment"]["phone_session_ids"], [
            "HTP0", "HTP1"
        ])
        self.assertEqual(
            [
                row["session_generation"]
                for row in binding_a["helper_attachment"]
                    ["phone_session_identities"]
            ],
            [1, 1],
        )
        self.assertEqual(binding_b["base"]["kv_cache_owner_id"], "kv-b")
        self.assertEqual(binding_b["helper_attachment"]["phone_session_ids"], [
            "HTP2"
        ])


if __name__ == "__main__":
    unittest.main()


class SessionCowRollbackTests(ModelPlacementControllerTests):
    """Rebind idempotence and rollback publication for one replaced session."""

    def quiesced_ggq(self):
        (
            controller, current, target_layout, ready, target,
            helper, desktop,
        ) = self.active_ggg_to_ggq()
        controller.request_helper_rebind(
            "gemma-request", target.generation, observed_at_us=200_010
        )
        controller.update_request_fraction(
            "gemma-request", 0, observed_at_us=200_020
        )
        controller.mark_request_helper_rebind_quiesced(
            "gemma-request", target.generation, observed_at_us=200_030
        )
        controller.begin_phone_layout_transition(
            target.generation,
            ticket_id="replace-htp2",
            transition_ids=("replace-htp2",),
            ready_at_us=201_000,
            projection_token_sha256=identity("replace-htp2-projection"),
            workspace_bytes=1024,
            observed_at_us=200_040,
        )
        return controller, current, target_layout, ready, target, helper, desktop

    def test_quiesced_rebind_resumes_without_positive_fraction(self) -> None:
        controller, _c, _tl, _ready, target, _h, _d = self.quiesced_ggq()
        first = controller.request_helper_rebind_state("gemma-request")
        self.assertEqual(first["state"], "QUIESCED")
        resumed = controller.request_helper_rebind(
            "gemma-request", target.generation, observed_at_us=200_050
        )
        self.assertEqual(resumed.state, "QUIESCED")
        self.assertEqual(
            controller.request_helper_rebind_state("gemma-request"), first
        )

    def test_rolled_back_transition_retargets_quiesced_rebind(self) -> None:
        controller, current, target_layout, ready, target, _h, _d = (
            self.quiesced_ggq()
        )
        selected = target.layout.changed_session_ids[0]
        self.assertTrue(controller.fail_phone_layout_transition(
            "replace-htp2",
            generation=target.generation,
            projection_token_sha256=identity("replace-htp2-projection"),
            failed_at_us=200_500,
            reason="injected_post_load_failure",
            restored_session_generations={selected: 3},
        ))
        # Logical rollback republishes the source layout with every session
        # READY, and the quiesced helper remains bound to it at 0%.
        self.assertEqual(controller.ready_phone_layout().generation, ready.generation)
        self.assertEqual(
            {row.session_id: row.state for row in controller.phone_session_states()},
            {"HTP0": "READY", "HTP1": "READY", "HTP2": "READY"},
        )
        self.assertEqual(
            controller.ready_phone_layout().layout
                .session_generation_by_id[selected],
            3,
        )
        self.assertTrue(all(
            generation == (3 if session_id == selected else 1)
            for session_id, generation in (
                controller.ready_phone_layout().layout
                    .session_generation_by_id.items()
            )
        ))
        self.assertFalse(controller.request_helper_layout_is_usable(
            "gemma-request", ready.generation, current.geometry_sha256
        ))
        retry = controller.propose_phone_layout(
            target_layout,
            workspace_bytes=1024,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-functionfs", "phone-usb"),
            observed_at_us=300_000,
            force=True,
        )
        self.assertNotEqual(retry.generation, target.generation)
        self.assertEqual(
            controller.phone_layout_transition_blockers(retry.generation),
            ("gemma-request",),
        )
        rebind = controller.request_helper_rebind(
            "gemma-request", retry.generation, observed_at_us=300_010
        )
        self.assertEqual(rebind.state, "QUIESCED")
        self.assertEqual(rebind.target_generation, retry.generation)
        self.assertEqual(rebind.removed_session_ids, (selected,))
        self.assertEqual(
            controller.phone_layout_transition_blockers(retry.generation),
            (),
        )
        kinds = [
            row["kind"]
            for row in controller.request_helper_events("gemma-request")
        ]
        self.assertIn("REBIND_RETARGETED", kinds)
        controller.begin_phone_layout_transition(
            retry.generation,
            ticket_id="replace-htp2-retry",
            transition_ids=("replace-htp2-retry",),
            ready_at_us=301_000,
            projection_token_sha256=identity("retry-projection"),
            workspace_bytes=1024,
            observed_at_us=300_020,
        )
        ready_retry = controller.complete_phone_layout_transition(
            generation=retry.generation,
            ticket_id="replace-htp2-retry",
            transition_ids=("replace-htp2-retry",),
            geometry_sha256=target_layout.geometry_sha256,
            projection_token_sha256=identity("retry-projection"),
            finished_at_us=301_000,
        )
        self.assertEqual(ready_retry.state, "READY")
        self.assertEqual(
            ready_retry.layout.session_generation_by_id[selected], 4
        )
        self.assertTrue(all(
            generation == (4 if session_id == selected else 1)
            for session_id, generation in (
                ready_retry.layout.session_generation_by_id.items()
            )
        ))
        with self.assertRaisesRegex(
            ModelPlacementControllerError,
            "phone layout generation is absent",
        ):
            controller.phone_layout(target.generation)

    def test_restored_failure_reauthorizes_one_fresh_retry(self) -> None:
        controller, _current, target_layout, _ready, target, _h, _d = (
            self.quiesced_ggq()
        )
        selected = target.layout.changed_session_ids[0]
        controller.fail_phone_layout_transition(
            "replace-htp2",
            generation=target.generation,
            projection_token_sha256=identity("replace-htp2-projection"),
            failed_at_us=200_500,
            reason="injected_post_load_failure",
            restored_session_generations={selected: 3},
        )
        pending = controller.pending_phone_layout_candidate()
        self.assertEqual(
            pending["geometry_sha256"], target_layout.geometry_sha256
        )
        confirmed, count = controller.confirm_phone_layout_candidate(
            target_layout.geometry_sha256,
            identity("post-rollback-snapshot"),
            observed_at_us=200_510,
        )
        self.assertTrue(confirmed)
        self.assertGreaterEqual(
            count, controller.policy.phone_layout_confirmation_snapshots
        )
        retry = controller.propose_phone_layout(
            target_layout,
            workspace_bytes=1024,
            shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-functionfs", "phone-usb"),
            observed_at_us=200_520,
        )
        self.assertEqual(retry.layout.changed_session_ids, (selected,))
        self.assertEqual(
            retry.layout.session_generation_by_id[selected], 4
        )
        self.assertIsNone(controller.pending_phone_layout_candidate())

    def test_rebind_cannot_switch_to_a_second_live_proposal(self) -> None:
        controller, _c, _tl, _ready, target, _h, _d = self.quiesced_ggq()
        with self.assertRaisesRegex(
            ModelPlacementControllerError, "already active"
        ):
            controller.request_helper_rebind(
                "gemma-request", target.generation + 7, observed_at_us=200_050
            )

    def test_detached_fallback_helper_is_not_a_transition_blocker(self) -> None:
        controller, _c, _tl, _ready, target, _h, _d = self.active_ggg_to_ggq()
        self.assertEqual(
            controller.phone_layout_transition_blockers(target.generation),
            ("gemma-request",),
        )
        controller.update_request_fraction(
            "gemma-request", 0, observed_at_us=200_020
        )
        controller.detach_request_helper(
            "gemma-request",
            fallback_outcome="PHONE_LAYOUT_TRANSITION_FAILED",
            observed_at_us=200_030,
        )
        self.assertEqual(
            controller.phone_layout_transition_blockers(target.generation),
            (),
        )

    def test_unproven_restoration_marks_session_unavailable(self) -> None:
        controller, current, _tl, ready, target, _h, _d = self.quiesced_ggq()
        controller.fail_phone_layout_transition(
            "replace-htp2",
            generation=target.generation,
            projection_token_sha256=identity("replace-htp2-projection"),
            failed_at_us=200_500,
            reason="rollback_failed",
            unavailable_session_ids=("HTP2",),
        )
        states = {
            row.session_id: row.state
            for row in controller.phone_session_states()
        }
        self.assertEqual(states["HTP2"], "UNAVAILABLE")
        self.assertEqual(states["HTP0"], "READY")
        self.assertFalse(controller.request_helper_layout_is_usable(
            "gemma-request", ready.generation, current.geometry_sha256
        ))
        self.assertFalse(controller.phone_layout_sessions_are_usable(
            ready.generation, current.geometry_sha256, ("HTP2",)
        ))
        events = controller.phone_layout_events()
        self.assertIn("SESSION_UNAVAILABLE", [row["kind"] for row in events])
        self.assertEqual(
            events[-1]["unavailable_session_ids"], ["HTP2"]
        )

    def test_unavailable_sessions_must_belong_to_the_transition(self) -> None:
        controller, _c, _tl, _ready, target, _h, _d = self.quiesced_ggq()
        with self.assertRaisesRegex(
            ModelPlacementControllerError, "unavailable sessions differ"
        ):
            controller.fail_phone_layout_transition(
                "replace-htp2",
                generation=target.generation,
                projection_token_sha256=identity("replace-htp2-projection"),
                failed_at_us=200_500,
                reason="rollback_failed",
                unavailable_session_ids=("HTP0",),
            )
