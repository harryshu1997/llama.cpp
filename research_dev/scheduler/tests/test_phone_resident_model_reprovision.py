"""Phone re-provisioning for the desktop's model: configuration, demand, selection, triggers."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

from research_dev.scheduler import UnifiedScheduleError, UnifiedScheduler
from research_dev.scheduler._internal.model_placement_controller import (
    ModelPlacementController, ModelPlacementPolicy,
)
from research_dev.scheduler._internal.phone_shards import (
    PhoneFfnResidencyDemand, generate_mixed_ffn_residency_layouts,
)
from research_dev.scheduler._unified.phone_residency_ops import economics, reprovision
from research_dev.scheduler._unified.phone_residency_ops.common import (
    _PhoneDemandDiscovery, _PhoneMemoryBudget, _PhoneQueueDemand,
)
from research_dev.scheduler.campaigns.burstgpt import arguments
from research_dev.scheduler.config import (
    CAMPAIGN_MANIFEST_SCHEMA, CampaignManifest, PhoneResidentModelReprovisioningConfiguration,
    SchedulerConfigurationError,
)

try:
    from .test_multi_session_phone import manifest, session
    from .test_automated_runtime import request
except ImportError:
    from test_multi_session_phone import manifest, session
    from test_automated_runtime import request

MODEL_A = manifest(6)
MODEL_B = replace(manifest(6), model_id="synthetic-session-model-b",
                  artifact_sha256="sha256:" + "e" * 64)
A = MODEL_A.artifact_sha256
B = MODEL_B.artifact_sha256
SESSIONS = tuple(session(index) for index in range(3))
LIMIT = 768
CONFIG = PhoneResidentModelReprovisioningConfiguration(load_bytes_per_second=32)
EARLY = replace(CONFIG, early_on_transition=True, count_queued_demand=True)


def order_view(*rows):
    """A dispatch order view: (request_id, queue state, predecessor request ids) per row."""
    return {request_id: {"state": state, "predecessor_request_ids": tuple(predecessors)}
            for request_id, state, predecessors in rows}


def demand_row(model, work):
    return PhoneFfnResidencyDemand(model, work, 8, "split-row")


def desktop_load(latency_us=47_000_000):
    return SimpleNamespace(device_id="desktop-gpu", source_state="cold", target_state="hot",
                           latency_us=latency_us, prepares_device_ids=("desktop-gpu",))


def phone_load():
    return SimpleNamespace(device_id="phone-a", source_state="cold", target_state="hot",
                           latency_us=12_000_000, prepares_device_ids=("phone-a",))


def ticket(request_id, model, *, state="QUEUED", transition=None, output_tokens=100,
           dispatched_at=None, devices=("desktop-gpu",)):
    return SimpleNamespace(
        request=request(request_id, output_tokens=output_tokens),
        model=SimpleNamespace(model_id=model.model_id, artifact_sha256=model.artifact_sha256),
        dispatch_state=state,
        transition_status="PENDING" if transition is not None else "NOT_REQUIRED",
        execution_plan=SimpleNamespace(
            device_ids=devices, transitions=() if transition is None else (transition,),
            helper_envelope=None,
        ),
        decision=SimpleNamespace(start_us=dispatched_at or 0),
        dispatch_receipt=None if dispatched_at is None else SimpleNamespace(observed_at_us=dispatched_at),
    )


def queue_demand(tickets):
    active, queued, active_count, queued_count = {}, {}, {}, {}
    for row in tickets:
        artifact = row.model.artifact_sha256
        tokens, number = ((active, active_count) if row.dispatch_state == "ACQUIRED"
                          else (queued, queued_count))
        tokens[artifact] = tokens.get(artifact, 0) + row.request.output_tokens
        number[artifact] = number.get(artifact, 0) + 1
    work = {key: active.get(key, 0) + queued.get(key, 0) for key in set(active) | set(queued)}
    return _PhoneQueueDemand(active_count, queued_count, active, queued, work)


def discovery_for(demand):
    models = {A: MODEL_A, B: MODEL_B}
    return _PhoneDemandDiscovery(
        demand_rows=tuple(demand_row(models[key], work)
                          for key, work in sorted(demand.queued_work_by_artifact.items()) if work > 0),
        sessions=SESSIONS, helper_id="phone-a", route_evidence_by_artifact={},
    )


def budget(current, limit=LIMIT):
    return _PhoneMemoryBudget(
        current=current, planning=current,
        accepted_resident_bytes=0 if current is None else current.resident_bytes,
        live_capacity=None, persistent_service_reserve_by_artifact={},
        persistent_service_reserve_bytes=0, live_phone_wide_limit=None, phone_wide_limit=limit,
    )


def layout_with_counts(rows, counts, limit=LIMIT):
    return next(row for row in generate_mixed_ffn_residency_layouts(rows, SESSIONS, phone_wide_limit_bytes=limit)
                if reprovision._session_counts(row) == counts)


def resident(rows, counts, limit=LIMIT):
    layout = layout_with_counts(rows, counts, limit)
    return layout.with_session_generations({row.session_id: 1 for row in layout.shards})


FULL_A = resident((demand_row(MODEL_A, 100),), {A: 3})
FULL_B = resident((demand_row(MODEL_B, 100),), {B: 3})


def layers(layout):
    return reprovision._layers_by_artifact(layout)


def host(tickets, *, current=None, states=(), bindings=None, events=None, configuration=CONFIG,
         view=None):
    desktop = SimpleNamespace(device_id="desktop-gpu", executor_id="executor:desktop-gpu",
                              phone_sessions=(), adapter_parameters={})
    phone = SimpleNamespace(device_id="phone-a", executor_id="executor:phone-a",
                            phone_sessions=SESSIONS, adapter_parameters={})
    real = ModelPlacementController(ModelPlacementPolicy())
    ready = None if current is None else SimpleNamespace(layout=current)
    placement = SimpleNamespace(
        policy=real.policy,
        phone_layout_events=lambda: tuple(events or ()),
        phone_session_states=lambda: tuple(states),
        request_binding=lambda request_id: (bindings or {}).get(request_id),
        ready_phone_layout=lambda: ready,
        planning_phone_layout=lambda: ready,
        select_phone_layout_candidate=real.select_phone_layout_candidate,
        record_request_helper_event=lambda *args: None,
    )
    controller = SimpleNamespace(
        _runtime_capabilities=SimpleNamespace(
            executors=(desktop, phone), executor_by_device={"desktop-gpu": desktop, "phone-a": phone},
            transitions=(), minimum_energy_saving_ppm=10_000, phone_power_profile_by_device={},
        ),
        _runtime_controller=SimpleNamespace(
            current_tickets=lambda: tuple(tickets),
            **({} if view is None else {"dispatch_order_view": lambda: view}),
        ),
        _model_placement_controller=placement,
        _phone_reprovisioning=configuration,
        _phone_ffn_shard_storage=(),
        _runtime_manifests={MODEL_A.model_id: MODEL_A, MODEL_B.model_id: MODEL_B},
        _phone_layout_request_impacts=lambda *args: (),
    )
    controller._phone_transition_estimates = (
        lambda *args: economics._phone_transition_estimates(controller, *args)
    )
    return controller


def plan(controller, tickets, *, snapshot=None, observed_at_us=1_000_000, discovery=None):
    demand = queue_demand(tickets)
    return reprovision._phone_reprovision_demand(
        controller, demand, discovery or discovery_for(demand), snapshot, observed_at_us,
    )


def choose(controller, tickets, current, *, snapshot=None, limit=LIMIT, observed_at_us=1_000_000,
           sessions=SESSIONS):
    layout_demand, layout_discovery = plan(controller, tickets, snapshot=snapshot, observed_at_us=observed_at_us)
    return economics._phone_candidate_choice(
        controller, layout_discovery, layout_demand, sessions, budget(current, limit), observed_at_us,
    )


def residency(*rows):
    return SimpleNamespace(residency=tuple(
        SimpleNamespace(device_id=device, artifact_sha256=artifact, state=state, resident_bytes=1)
        for device, artifact, state in rows
    ))


class ConfigurationTests(unittest.TestCase):
    def test_round_trip_and_defaults(self):
        default = PhoneResidentModelReprovisioningConfiguration()
        self.assertEqual(default.to_json(), {"mode": "resident-model", "load_bytes_per_second": 200_000_000,
                                             "minimum_learned_samples": 2,
                                             "boundary_reevaluation_interval_us": 10_000_000})
        self.assertEqual(PhoneResidentModelReprovisioningConfiguration.from_json({}), default)
        self.assertEqual(PhoneResidentModelReprovisioningConfiguration.from_json(CONFIG.to_json()), CONFIG)
        every_boundary = PhoneResidentModelReprovisioningConfiguration.from_json(
            {"boundary_reevaluation_interval_us": 0})
        self.assertEqual(every_boundary.boundary_reevaluation_interval_us, 0)
        self.assertEqual(PhoneResidentModelReprovisioningConfiguration.from_json(every_boundary.to_json()),
                         every_boundary)

    def test_early_transition_flags_are_opt_in_and_round_trip(self):
        default = PhoneResidentModelReprovisioningConfiguration()
        self.assertFalse(default.early_on_transition)
        self.assertFalse(default.count_queued_demand)
        self.assertNotIn("early_on_transition", default.to_json())
        self.assertNotIn("count_queued_demand", default.to_json())
        early = PhoneResidentModelReprovisioningConfiguration.from_json(
            {"early_on_transition": True, "count_queued_demand": True})
        self.assertEqual((early.early_on_transition, early.count_queued_demand), (True, True))
        self.assertEqual(early.to_json()["early_on_transition"], True)
        self.assertEqual(early.to_json()["count_queued_demand"], True)
        self.assertEqual(PhoneResidentModelReprovisioningConfiguration.from_json(early.to_json()), early)
        only = PhoneResidentModelReprovisioningConfiguration.from_json({"count_queued_demand": True})
        self.assertNotIn("early_on_transition", only.to_json())
        self.assertEqual(PhoneResidentModelReprovisioningConfiguration.from_json(EARLY.to_json()), EARLY)

    def test_invalid_values_are_rejected(self):
        for invalid in ({"mode": "static"}, {"load_bytes_per_second": 0}, {"minimum_learned_samples": 0},
                        {"unknown": 1}, {"load_bytes_per_second": "270"}, {"load_bytes_per_second": True},
                        {"boundary_reevaluation_interval_us": -1}, {"boundary_reevaluation_interval_us": 1.5}, [],
                        {"early_on_transition": 1}, {"count_queued_demand": "true"},
                        {"early_on_transition": None}):
            with self.subTest(invalid=invalid), self.assertRaises(SchedulerConfigurationError):
                PhoneResidentModelReprovisioningConfiguration.from_json(invalid)

    def test_campaign_manifest_field_is_opt_in_and_round_trips(self):
        row = {"schema": CAMPAIGN_MANIFEST_SCHEMA, "campaign_id": "reprovision-test",
               "rig_manifest_path": "rig.json", "models_manifest_path": "models.json",
               "evidence_manifest_path": "evidence.json", "selection_mode": "energy-aware",
               "maximum_latency_ppm": 1_100_000, "energy_attribution_kind": "diagnostic",
               "trace": {"large_requests_path": "large.json", "overlay_requests_path": "overlay.json",
                         "trace_manifest_path": "trace.json"}}
        plain = CampaignManifest.from_json(row, Path("/inputs"))
        self.assertIsNone(plain.phone_resident_model_reprovisioning)
        self.assertNotIn("phone_resident_model_reprovisioning", plain.to_json())
        configured = CampaignManifest.from_json(
            {**row, "phone_resident_model_reprovisioning": CONFIG.to_json()}, Path("/inputs"))
        self.assertEqual(configured.phone_resident_model_reprovisioning, CONFIG)
        self.assertEqual(CampaignManifest.from_json(configured.to_json(), Path("/inputs")), configured)
        fixed = {"mode": "fixed", "reference_id": "ref", "assignments": [
            {"session_id": "HTP0", "artifact_sha256": A, "layer_mask": 1, "maximum_columns": 8}]}
        with self.assertRaisesRegex(SchedulerConfigurationError, "re-provisioned"):
            CampaignManifest.from_json({**row, "fixed_phone_residency": fixed,
                                        "phone_resident_model_reprovisioning": {}}, Path("/inputs"))

    def test_runner_argument_is_optional(self):
        parser = arguments._build_parser()
        option = next(action for action in parser._actions
                      if "--phone-resident-model-reprovisioning-json" in action.option_strings)
        self.assertIsNone(option.default)
        self.assertFalse(option.required)

    def test_scheduler_setter_validates_and_records(self):
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        self.assertIsNone(scheduler._phone_reprovisioning)
        with self.assertRaisesRegex(UnifiedScheduleError, "invalid"):
            scheduler.configure_phone_resident_model_reprovisioning({"mode": "resident-model"})
        scheduler.configure_phone_resident_model_reprovisioning(CONFIG)
        self.assertEqual(scheduler._phone_reprovisioning, CONFIG)
        event = scheduler.phone_residency_events()[-1]
        self.assertEqual(event["reason"], "PHONE_RESIDENT_MODEL_REPROVISIONING_CONFIGURED")
        self.assertEqual(event["configuration"], CONFIG.to_json())
        scheduler._fixed_phone_residency = object()
        with self.assertRaisesRegex(UnifiedScheduleError, "fixed phone residency"):
            scheduler.configure_phone_resident_model_reprovisioning(CONFIG)
        scheduler.configure_phone_resident_model_reprovisioning(None)
        self.assertIsNone(scheduler._phone_reprovisioning)


class DemandTests(unittest.TestCase):
    def test_knob_off_returns_the_arrived_work_demand_unchanged(self):
        tickets = [ticket("a-1", MODEL_A, state="ACQUIRED"), ticket("b-1", MODEL_B)]
        demand = queue_demand(tickets)
        discovery = discovery_for(demand)
        result = reprovision._phone_reprovision_demand(
            host(tickets, configuration=None), demand, discovery, None, 5)
        self.assertIs(result[0], demand)
        self.assertIs(result[1], discovery)
        self.assertIsNone(result[0].reprovision)

    def test_executing_model_is_followed_alone(self):
        tickets = [ticket("a-1", MODEL_A, state="ACQUIRED"), ticket("b-1", MODEL_B), ticket("b-2", MODEL_B)]
        demand, discovery = plan(host(tickets), tickets)
        context = demand.reprovision
        self.assertEqual((context.mode, context.commitment_source), ("FOLLOW", "executing"))
        self.assertEqual(context.split_artifacts, (A,))
        self.assertEqual(demand.queued_work_by_artifact, {A: 100})
        self.assertEqual([row.manifest.artifact_sha256 for row in discovery.demand_rows], [A])
        self.assertTrue(context.confirmed)

    def test_loading_model_is_followed_with_its_load_window(self):
        leader = ticket("b-1", MODEL_B, state="ACQUIRED", transition=desktop_load(47_000_000),
                        dispatched_at=100_000_000)
        context = plan(host([leader, ticket("a-2", MODEL_A)]), [leader, ticket("a-2", MODEL_A)])[0].reprovision
        self.assertEqual((context.mode, context.commitment_source, context.leader_request_id),
                         ("FOLLOW", "loading", "b-1"))
        self.assertEqual((context.load_window_start_us, context.load_window_end_us), (100_000_000, 147_000_000))

    def test_models_served_together_or_unknown_are_split(self):
        both = [ticket("a-1", MODEL_A, state="ACQUIRED"), ticket("b-1", MODEL_B, state="ACQUIRED")]
        context = plan(host(both), both)[0].reprovision
        self.assertEqual((context.mode, context.split_artifacts), ("PROPORTIONAL", (A, B)))
        queued = [ticket("a-1", MODEL_A), ticket("b-1", MODEL_B)]
        context = plan(host(queued), queued)[0].reprovision
        self.assertEqual((context.mode, context.commitment_source), ("PROPORTIONAL", None))
        self.assertFalse(context.confirmed)

    def test_idle_followed_model_holds_the_layout(self):
        queued = [ticket("b-1", MODEL_B)]
        snapshot = residency(("desktop-gpu", A, "hot"), ("phone-a", B, "hot"))
        context = plan(host(queued, current=FULL_A), queued, snapshot=snapshot)[0].reprovision
        self.assertEqual((context.mode, context.commitment_source), ("HOLD", "resident"))
        events = ({"kind": "EVALUATED", "desktop_reprovision": {"followed_artifact_sha256s": [A]}},)
        context = plan(host(queued, current=FULL_A, events=events), queued)[0].reprovision
        self.assertEqual((context.mode, context.commitment_source), ("HOLD", "retained"))
        # a model the phone neither holds nor is demanded for is not followed
        context = plan(host(queued, events=events), queued)[0].reprovision
        self.assertEqual((context.mode, context.commitment_source), ("PROPORTIONAL", None))

    def test_hot_desktop_residency_wins_over_warm(self):
        queued = [ticket("a-1", MODEL_A), ticket("b-1", MODEL_B)]
        snapshot = residency(("desktop-gpu", A, "warm"), ("desktop-gpu", B, "hot"))
        context = plan(host(queued), queued, snapshot=snapshot)[0].reprovision
        self.assertEqual((context.mode, context.followed_artifacts), ("FOLLOW", (B,)))

    def test_phone_only_plans_and_models_without_phone_demand_are_not_followed(self):
        tickets = [ticket("a-1", MODEL_A, state="ACQUIRED", transition=phone_load(), devices=("phone-a",)),
                   ticket("b-1", MODEL_B)]
        self.assertEqual(reprovision._desktop_commitment(host(tickets), frozenset({A, B}), None),
                         ((), None, None, None, None))
        leader = ticket("b-1", MODEL_B, state="ACQUIRED", transition=desktop_load(), dispatched_at=5)
        self.assertEqual(reprovision._desktop_commitment(host([leader]), frozenset({A}), None),
                         ((), None, None, None, None))


    def test_queued_next_model_is_followed_only_with_count_queued_demand(self):
        decided = ticket("b-1", MODEL_B, transition=desktop_load(47_000_000), dispatched_at=100_000_000)
        view = order_view(("b-1", "QUEUED", ()))
        self.assertEqual(
            reprovision._desktop_commitment(host([decided], configuration=EARLY, view=view), frozenset({A, B}), None),
            ((B,), "queued", "b-1", 100_000_000, 147_000_000),
        )
        # Knob off, or no dispatch order to prove the request is next: unchanged / fail-closed.
        self.assertEqual(reprovision._desktop_commitment(host([decided], view=view), frozenset({A, B}), None),
                         ((), None, None, None, None))
        self.assertEqual(reprovision._desktop_commitment(host([decided], configuration=EARLY), frozenset({A, B}), None),
                         ((), None, None, None, None))
        # A queued plan without a desktop load (resident model, idle server) is followed without a window.
        hot = ticket("a-1", MODEL_A, dispatched_at=700)
        self.assertEqual(
            reprovision._desktop_commitment(
                host([hot], configuration=EARLY, view=order_view(("a-1", "QUEUED", ()))), frozenset({A, B}), None),
            ((A,), "queued", "a-1", 700, None),
        )

    def test_queued_model_behind_live_work_is_not_followed(self):
        running = ticket("a-1", MODEL_A, state="ACQUIRED")
        decided = ticket("b-1", MODEL_B, transition=desktop_load(), dispatched_at=5)
        view = order_view(("a-1", "ACTIVE", ()), ("b-1", "QUEUED", ("a-1",)))
        self.assertEqual(
            reprovision._desktop_commitment(host([running, decided], configuration=EARLY, view=view),
                                            frozenset({A, B}), None),
            ((A,), "executing", None, None, None),
        )
        # Once its predecessor is finishing, the decided model joins the followed set.
        view = order_view(("a-1", "FINISHING", ()), ("b-1", "QUEUED", ("a-1",)))
        self.assertEqual(
            reprovision._desktop_commitment(host([running, decided], configuration=EARLY, view=view),
                                            frozenset({A, B}), None),
            ((A, B), "executing", None, None, None),
        )

    def test_queued_next_demand_counts_every_queued_request_and_is_confirmed(self):
        tickets = [ticket("b-1", MODEL_B, transition=desktop_load(47_000_000), dispatched_at=100_000_000),
                   ticket("b-2", MODEL_B)]
        view = order_view(("b-1", "QUEUED", ()), ("b-2", "QUEUED", ("b-1",)))
        demand, discovery = plan(host(tickets, configuration=EARLY, view=view), tickets)
        context = demand.reprovision
        self.assertEqual((context.mode, context.commitment_source, context.leader_request_id),
                         ("FOLLOW", "queued", "b-1"))
        self.assertEqual((context.load_window_start_us, context.load_window_end_us), (100_000_000, 147_000_000))
        self.assertEqual(demand.queued_work_by_artifact, {B: 200})
        self.assertEqual(context.work_by_artifact, {B: 200})
        self.assertTrue(context.confirmed)
        # Knob off: the queued model is not followed, the split is unconfirmed.
        context = plan(host(tickets, view=view), tickets)[0].reprovision
        self.assertEqual((context.mode, context.commitment_source), ("PROPORTIONAL", None))
        self.assertFalse(context.confirmed)


class SelectionTests(unittest.TestCase):
    def test_decided_transition_reprovisions_before_the_first_admission(self):
        """The Qwen->Gemma switch is decided (queued, next) while the phone still holds model A's
        shards: with the knobs the swap to model B is selected and confirmed at once."""
        decided = ticket("b-1", MODEL_B, transition=desktop_load(47_000_000), dispatched_at=100_000_000)
        view = order_view(("b-1", "QUEUED", ()))
        choice = choose(host([decided], current=FULL_A, configuration=EARLY, view=view), [decided], FULL_A)
        self.assertEqual(choice.reason, reprovision.REASON_REPROVISION)
        self.assertTrue(choice.confirmed)
        self.assertEqual(choice.reprovision["desktop_commitment_source"], "queued")
        self.assertEqual(choice.reprovision["mode"], "FOLLOW")
        self.assertEqual(choice.reprovision["target_layers_by_artifact"], layers(FULL_B))
        self.assertIsNotNone(choice.reprovision["fits_load_window"])
        plain = choose(host([decided], current=FULL_A, view=view), [decided], FULL_A)
        self.assertFalse(plain.confirmed)
        self.assertIsNone(plain.reprovision["desktop_commitment_source"])

    def test_single_model_demand_fills_the_whole_phone(self):
        tickets = [ticket("a-1", MODEL_A), ticket("a-2", MODEL_A)]
        choice = choose(host(tickets), tickets, None)
        self.assertEqual(choice.reason, reprovision.REASON_PROPORTIONAL)
        self.assertEqual(layers(choice.selected), {A: 6})
        self.assertEqual(choice.selected.resident_bytes, LIMIT)
        self.assertEqual(choice.reprovision["target_session_counts_by_artifact"], {A: 3})

    def test_followed_model_gets_every_session_one_at_a_time(self):
        tickets = [ticket("a-1", MODEL_A, state="ACQUIRED"), ticket("b-1", MODEL_B), ticket("b-2", MODEL_B)]
        current = resident((demand_row(MODEL_A, 100), demand_row(MODEL_B, 200)), {A: 1, B: 2})
        controller = host(tickets, current=current)
        for step in range(2):
            choice = choose(controller, tickets, current)
            self.assertEqual(choice.reason, reprovision.REASON_REPROVISION)
            self.assertTrue(choice.confirmed)
            self.assertEqual(choice.reprovision["target_layers_by_artifact"], {A: 6})
            self.assertEqual(len(choice.selected.changed_session_ids), 1)
            self.assertLessEqual(choice.selected.resident_bytes, LIMIT)
            source = {row.session_id: row for row in current.shards}
            for row in choice.selected.shards:
                if row.session_id not in choice.selected.changed_session_ids:
                    self.assertEqual(row, source[row.session_id])
            self.assertEqual(layers(choice.selected)[A], 2 * (step + 2))
            current = choice.selected.with_session_generations({row.session_id: 1 for row in choice.selected.shards})
            controller = host(tickets, current=current)
        self.assertEqual(choose(controller, tickets, current).reason, reprovision.REASON_RETAINED)

    def test_two_model_demand_splits_in_proportion_to_remaining_work(self):
        for work_a, work_b, expected in ((2000, 1000, {A: 2, B: 1}), (1000, 3000, {A: 1, B: 2})):
            tickets = [ticket("a-1", MODEL_A, output_tokens=work_a), ticket("b-1", MODEL_B, output_tokens=work_b)]
            choice = choose(host(tickets), tickets, None)
            self.assertEqual(choice.reason, reprovision.REASON_PROPORTIONAL)
            self.assertEqual(reprovision._session_counts(choice.selected), expected)
            self.assertEqual(choice.reprovision["target_session_counts_by_artifact"], expected)
        self.assertEqual(reprovision._proportional_session_counts({A: 1000, B: 1000}, 3), {A: 2, B: 1})
        self.assertEqual(reprovision._proportional_session_counts({A: 7}, 3), {A: 3})
        self.assertEqual(reprovision._proportional_session_counts({A: 1, B: 1000}, 3), {B: 3})
        self.assertEqual(reprovision._proportional_session_counts({}, 3), {})

    def test_swap_is_timed_against_the_desktop_load_window(self):
        dispatched_at = 100_000_000
        leader = ticket("b-1", MODEL_B, state="ACQUIRED", transition=desktop_load(47_000_000),
                        dispatched_at=dispatched_at)
        tickets = [leader, ticket("a-2", MODEL_A, output_tokens=5000)]
        choice = choose(host(tickets, current=FULL_A), tickets, FULL_A, observed_at_us=dispatched_at)
        self.assertEqual(choice.reason, reprovision.REASON_REPROVISION)
        self.assertTrue(choice.confirmed)
        recorded = choice.reprovision
        self.assertEqual(recorded["followed_artifact_sha256s"], [B])
        self.assertEqual(recorded["stage_swap_latency_us"], 8_000_000)
        self.assertEqual(recorded["target_swap_latency_us"], 24_000_000)
        self.assertTrue(recorded["fits_load_window"])
        self.assertEqual(recorded["target_layers_by_artifact"], {B: 6})
        self.assertEqual(layers(choice.selected), {A: 4, B: 2})
        changed = choice.selected.changed_session_ids[0]
        self.assertEqual(choice.transition_latencies[changed], 8_000_000)
        slow = PhoneResidentModelReprovisioningConfiguration(load_bytes_per_second=4)
        choice = choose(host(tickets, current=FULL_A, configuration=slow), tickets, FULL_A,
                        observed_at_us=dispatched_at + 10_000_000)
        self.assertEqual(choice.reprovision["target_swap_latency_us"], 192_000_000)
        self.assertFalse(choice.reprovision["fits_load_window"])

    def test_in_use_sessions_are_kept_and_the_change_is_deferred(self):
        # A still decodes while B loads: split A:1 / B:2, but A's helper pins HTP0 and HTP1
        leader = ticket("b-1", MODEL_B, state="ACQUIRED", transition=desktop_load(), dispatched_at=100,
                        output_tokens=200)
        using = ticket("a-1", MODEL_A, state="ACQUIRED")
        tickets = [using, leader]
        in_use = {"a-1": {"fraction_ppm": 0, "helper_attachment": {
            "fraction_ppm": 0, "lease_tokens": [], "completed_phone_calls": 3, "lease_reserved_until_us": None,
            "fallback_outcome": None, "allowed_session_ids": ["HTP0", "HTP1"]}}}
        choice = choose(host(tickets, current=FULL_A, bindings=in_use), tickets, FULL_A, observed_at_us=100)
        self.assertEqual(choice.selected.changed_session_ids, ("HTP2",))
        self.assertEqual(choice.reprovision["in_use_session_ids"], ["HTP0", "HTP1"])
        self.assertTrue(set(choice.reprovision["blocked_session_ids"]) <= {"HTP0", "HTP1"})
        # every session referenced by a helper: nothing changes, the deferral is recorded
        states = tuple(SimpleNamespace(session_id=f"HTP{index}", state="READY", active_helper_references=("a-1",))
                       for index in range(3))
        choice = choose(host(tickets, current=FULL_A, states=states), tickets, FULL_A, observed_at_us=100)
        self.assertEqual(choice.reason, reprovision.REASON_DEFERRED_IN_USE)
        self.assertIs(choice.selected, FULL_A)
        self.assertTrue(choice.reprovision["blocked_session_ids"])
        # a session mid-transition counts as in use
        states = (SimpleNamespace(session_id="HTP1", state="LOADING", active_helper_references=()),)
        choice = choose(host(tickets, current=FULL_A, states=states), tickets, FULL_A, observed_at_us=100)
        self.assertNotIn("HTP1", choice.selected.changed_session_ids)
        # a helper that never called the phone, or fell back without leases, does not pin sessions
        for attachment in ({"fraction_ppm": 0, "lease_tokens": [], "completed_phone_calls": 0},
                           {"fraction_ppm": 0, "lease_tokens": [], "completed_phone_calls": 9,
                            "fallback_outcome": "LATENCY_BOUND_EXCEEDED", "lease_reserved_until_us": None}):
            idle = {"a-1": {"fraction_ppm": 0, "helper_attachment": {
                **attachment, "allowed_session_ids": ["HTP0", "HTP1"]}}}
            self.assertEqual(reprovision._in_use_session_ids(host(tickets, bindings=idle)), ())

    def test_revalidation_that_cannot_be_verified_defers_the_stage(self):
        tickets = [ticket("a-1", MODEL_A, state="ACQUIRED")]
        controller = host(tickets, current=FULL_B)
        controller._phone_layout_request_impacts = lambda *args: (
            SimpleNamespace(request_id="x", verification_feasible=False),)
        choice = choose(controller, tickets, FULL_B)
        self.assertEqual(choice.reason, reprovision.REASON_DEFERRED_REVALIDATION)
        self.assertIs(choice.selected, FULL_B)

    def test_ram_and_session_caps_bound_every_stage(self):
        tickets = [ticket("a-1", MODEL_A, state="ACQUIRED")]
        current = resident((demand_row(MODEL_B, 100),), {B: 2}, limit=512)
        for _ in range(5):
            choice = choose(host(tickets, current=current), tickets, current, limit=640)
            if choice.reason != reprovision.REASON_REPROVISION:
                break
            self.assertEqual(choice.reprovision["target_layers_by_artifact"].get(A), 5)
            self.assertLessEqual(choice.selected.resident_bytes, 640)
            for row in choice.selected.shards:
                self.assertLessEqual(row.resident_bytes, 256)
            for row in choice.layouts:
                self.assertLessEqual(row.resident_bytes, 640)
            current = choice.selected.with_session_generations({row.session_id: 1 for row in choice.selected.shards})
        self.assertEqual(choice.reason, reprovision.REASON_RETAINED)
        self.assertEqual(layers(current), {A: 5})

    def test_unavailable_session_defers_to_the_forced_default_selector(self):
        tickets = [ticket("b-1", MODEL_B, state="ACQUIRED")]
        sessions = (session(0), session(1), session(2, ready=False))
        demand, discovery = plan(host(tickets, current=FULL_A), tickets)
        self.assertIsNone(reprovision._reprovision_candidate_choice(
            host(tickets, current=FULL_A), demand.reprovision, (), sessions, budget(FULL_A), {}, 5))

    def test_idle_followed_model_keeps_the_layout(self):
        queued = [ticket("b-1", MODEL_B)]
        events = ({"kind": "EVALUATED", "desktop_reprovision": {"followed_artifact_sha256s": [A]}},)
        choice = choose(host(queued, current=FULL_A, events=events), queued, FULL_A)
        self.assertEqual(choice.reason, reprovision.REASON_HOLD)
        self.assertIs(choice.selected, FULL_A)

    def test_selection_is_deterministic(self):
        leader = ticket("b-1", MODEL_B, state="ACQUIRED", transition=desktop_load(), dispatched_at=100)
        tickets = [leader, ticket("a-2", MODEL_A)]
        first = choose(host(tickets, current=FULL_A), tickets, FULL_A, observed_at_us=100)
        second = choose(host(tickets, current=FULL_A), tickets, FULL_A, observed_at_us=100)
        self.assertEqual(first.selected, second.selected)
        self.assertEqual(dict(first.reprovision), dict(second.reprovision))
        demand, _ = plan(host(tickets, current=FULL_A), tickets, observed_at_us=100)
        reversed_choice = reprovision._reprovision_candidate_choice(
            host(tickets, current=FULL_A), demand.reprovision, tuple(reversed(first.layouts)), SESSIONS,
            budget(FULL_A), {}, 100)
        self.assertEqual(reversed_choice.selected, first.selected)
        self.assertEqual(reversed_choice.reprovision["target_geometry_sha256"],
                         first.reprovision["target_geometry_sha256"])


class LearnedLoadRateTests(unittest.TestCase):
    def controller(self, *events):
        placement = ModelPlacementController(ModelPlacementPolicy())
        for kind, at_us, generation, session_id, resident_bytes in events:
            placement._record_phone_layout_event(kind, at_us, {
                "layout_generation": generation,
                "session": {"session_id": session_id, "resident_bytes": resident_bytes},
            })
        return SimpleNamespace(_model_placement_controller=placement)

    def test_prior_until_enough_windows_then_bytes_over_wall_time(self):
        one = self.controller(("SESSION_LOADING", 1_000_000, 1, "HTP0", 3_200_000_000),
                              ("SESSION_VERIFIED", 13_000_000, 1, "HTP0", 3_200_000_000))
        self.assertEqual(reprovision._learned_phone_load_rate(one, CONFIG), (32, 0))
        self.assertEqual(reprovision._learned_phone_load_rate(
            one, PhoneResidentModelReprovisioningConfiguration(minimum_learned_samples=1)),
            (3_200_000_000 // 12, 1))
        # two sessions of one transition share its wall window
        two = self.controller(("SESSION_LOADING", 1_000_000, 1, "HTP0", 3_200_000_000),
                              ("SESSION_VERIFIED", 13_000_000, 1, "HTP0", 3_200_000_000),
                              ("SESSION_LOADING", 20_000_000, 2, "HTP1", 2_800_000_000),
                              ("SESSION_LOADING", 20_000_000, 2, "HTP2", 2_800_000_000),
                              ("SESSION_VERIFIED", 40_000_000, 2, "HTP1", 2_800_000_000),
                              ("SESSION_VERIFIED", 40_000_000, 2, "HTP2", 2_800_000_000))
        rate, samples = reprovision._learned_phone_load_rate(two, CONFIG)
        self.assertEqual((rate, samples), ((3_200_000_000 + 5_600_000_000) // 32, 2))
        self.assertEqual(reprovision._load_latency_us(3_200_000_000, rate),
                         (3_200_000_000 * 1_000_000 + rate - 1) // rate)

    def test_run5_windows_give_the_measured_rate(self):
        run5 = self.controller(
            ("SESSION_LOADING", 13_665_149, 1, "HTP0", 3_208_642_560),
            ("SESSION_VERIFIED", 33_105_523, 1, "HTP0", 3_208_642_560),
            ("SESSION_LOADING", 33_231_841, 2, "HTP1", 3_208_642_560),
            ("SESSION_VERIFIED", 45_485_166, 2, "HTP1", 3_208_642_560),
            ("SESSION_LOADING", 79_865_719, 3, "HTP2", 2_673_868_800),
            ("SESSION_VERIFIED", 94_190_716, 3, "HTP2", 2_673_868_800),
            ("SESSION_LOADING", 251_776_056, 4, "HTP2", 2_831_155_200),
            ("SESSION_VERIFIED", 263_892_157, 4, "HTP2", 2_831_155_200))
        rate, samples = reprovision._learned_phone_load_rate(run5, CONFIG)
        self.assertEqual(samples, 4)
        self.assertEqual(rate // 1_000_000, 205)

    def test_a_retried_load_restarts_its_window(self):
        controller = self.controller(("SESSION_LOADING", 1_000_000, 1, "HTP0", 3_200_000_000),
                                     ("SESSION_LOADING", 50_000_000, 1, "HTP0", 3_200_000_000),
                                     ("SESSION_VERIFIED", 66_000_000, 1, "HTP0", 3_200_000_000))
        self.assertEqual(reprovision._learned_phone_load_rate(
            controller, PhoneResidentModelReprovisioningConfiguration(minimum_learned_samples=1)),
            (200_000_000, 1))

    def test_incomplete_or_malformed_windows_are_ignored(self):
        controller = self.controller(("SESSION_VERIFIED", 5, 1, "HTP0", 10),
                                     ("SESSION_LOADING", 9, 2, "HTP0", 10),
                                     ("SESSION_VERIFIED", 9, 2, "HTP0", 10),
                                     ("SESSION_LOADING", 1, 3, "HTP1", 0),
                                     ("SESSION_VERIFIED", 2, 3, "HTP1", 0),
                                     ("SESSION_LOADING", 1, 4, "HTP1", 10),
                                     ("SESSION_LOADING", 1, 4, "HTP2", 10),
                                     ("SESSION_VERIFIED", 3, 4, "HTP1", 10))
        self.assertEqual(reprovision._learned_phone_load_rate(
            controller, PhoneResidentModelReprovisioningConfiguration(minimum_learned_samples=1)),
            (200_000_000, 0))


class TriggerTests(unittest.TestCase):
    def controller(self, tickets, *, fail=None, configuration=None):
        calls, events = [], []

        @contextmanager
        def transaction(**_kwargs):
            yield

        def update(request_value, model, at_us, snapshot):
            calls.append((request_value.request_id, model.model_id, at_us, snapshot))
            if fail is not None:
                raise fail

        desktop = SimpleNamespace(device_id="desktop-gpu", phone_sessions=())
        phone = SimpleNamespace(device_id="phone-a", phone_sessions=SESSIONS)
        controller = SimpleNamespace(
            _runtime_capabilities=SimpleNamespace(executors=(desktop, phone), phone_power_profile_by_device={}),
            _runtime_controller=SimpleNamespace(current_tickets=lambda: tuple(tickets)),
            _model_placement_controller=SimpleNamespace(
                phone_layout_events=lambda: tuple(events),
                record_request_helper_event=lambda *args: events.append(("helper", *args)),
            ),
            _transaction=transaction,
            _update_phone_residency_portfolio=update,
            _phone_reprovisioning=configuration,
            runtime_model_manifest=lambda model_id: {MODEL_A.model_id: MODEL_A, MODEL_B.model_id: MODEL_B}[model_id],
        )
        return controller, calls, events

    def test_decided_transition_reevaluates_a_queued_desktop_load_once(self):
        decided = ticket("b-1", MODEL_B, transition=desktop_load(), dispatched_at=500)
        controller, calls, _events = self.controller([decided], configuration=EARLY)
        reprovision._reevaluate_phone_layout_for_decided_transition(controller, decided, 321)
        self.assertEqual(calls, [("b-1", MODEL_B.model_id, 321, None)])
        for other, configuration in (
            (decided, CONFIG),
            (ticket("b-2", MODEL_B, state="ACQUIRED", transition=desktop_load(), dispatched_at=5), EARLY),
            (ticket("a-1", MODEL_A), EARLY),
            (ticket("a-2", MODEL_A, transition=phone_load(), devices=("phone-a",)), EARLY),
        ):
            controller, calls, _events = self.controller([other], configuration=configuration)
            reprovision._reevaluate_phone_layout_for_decided_transition(controller, other, 321)
            self.assertEqual(calls, [])

    def test_failed_decided_reevaluation_is_recorded_not_raised(self):
        decided = ticket("b-1", MODEL_B, transition=desktop_load(), dispatched_at=500)
        controller, _calls, events = self.controller(
            [decided], fail=UnifiedScheduleError("no phone"), configuration=EARLY)
        reprovision._reevaluate_phone_layout_for_decided_transition(controller, decided, 321)
        self.assertEqual(events[-1][:3], ("helper", "b-1", "DECIDED_TRANSITION_LAYOUT_REEVALUATION_FAILED"))

    def test_release_reevaluates_for_a_decided_or_pending_load_under_early_on_transition(self):
        done = ticket("a-1", MODEL_A, state="COMPLETED")
        for waiting in (ticket("b-1", MODEL_B, transition=desktop_load(), dispatched_at=5),
                        ticket("b-1", MODEL_B, state="ACQUIRED", transition=desktop_load(), dispatched_at=5)):
            controller, calls, _events = self.controller([done, waiting], configuration=EARLY)
            reprovision._reevaluate_phone_layout_after_release(controller, done, 900)
            self.assertEqual(calls, [("b-1", MODEL_B.model_id, 900, None)])
        # Without a decided load (and no in-use deferral) the release changes nothing.
        controller, calls, _events = self.controller([done, ticket("b-1", MODEL_B)], configuration=EARLY)
        reprovision._reevaluate_phone_layout_after_release(controller, done, 900)
        self.assertEqual(calls, [])

    def test_desktop_load_dispatch_reevaluates_once_at_the_dispatch_time(self):
        leader = ticket("b-1", MODEL_B, state="ACQUIRED", transition=desktop_load(), dispatched_at=777)
        controller, calls, _events = self.controller([leader])
        reprovision._reevaluate_phone_layout_for_desktop_load(controller, leader)
        self.assertEqual(calls, [("b-1", MODEL_B.model_id, 777, None)])
        for other in (ticket("a-1", MODEL_A, state="ACQUIRED"),
                      ticket("a-2", MODEL_A, state="ACQUIRED", transition=phone_load(), devices=("phone-a",)),
                      ticket("a-3", MODEL_A, transition=desktop_load(), dispatched_at=5)):
            controller, calls, _events = self.controller([other])
            reprovision._reevaluate_phone_layout_for_desktop_load(controller, other)
            self.assertEqual(calls, [])

    def test_failed_reevaluation_is_recorded_not_raised(self):
        leader = ticket("b-1", MODEL_B, state="ACQUIRED", transition=desktop_load(), dispatched_at=777)
        controller, _calls, events = self.controller([leader], fail=UnifiedScheduleError("no phone"))
        reprovision._reevaluate_phone_layout_for_desktop_load(controller, leader)
        self.assertEqual(events[-1][:3], ("helper", "b-1", "DESKTOP_LOAD_LAYOUT_REEVALUATION_FAILED"))
        self.assertEqual(events[-1][4], {"reason": "no phone"})

    def test_physical_preparation_waits_for_release_instead_of_draining(self):
        placement = ModelPlacementController(ModelPlacementPolicy())
        controller = SimpleNamespace(_phone_reprovisioning=CONFIG, _model_placement_controller=placement)
        state = SimpleNamespace(generation=4, selection_reason=reprovision.REASON_REPROVISION)
        for _ in range(2):
            decision = reprovision._defer_preparation_until_release(controller, "b-1", state, ("a-2", "a-1"), 50)
            self.assertEqual(decision["status"], "DEFERRED")
            self.assertEqual(decision["reason"], "WAITING_FOR_HELPER_RELEASE")
            self.assertEqual(decision["blocking_request_ids"], ("a-1", "a-2"))
        self.assertEqual(len(placement.request_helper_events("b-1")), 1)
        # other proposals, and the knob off, keep the default drain path
        learning = SimpleNamespace(generation=4, selection_reason="PHONE_RESIDENCY_LEARNING_EXPLORATION")
        self.assertIsNone(reprovision._defer_preparation_until_release(controller, "b-1", learning, ("a-1",), 50))
        controller._phone_reprovisioning = None
        self.assertIsNone(reprovision._defer_preparation_until_release(controller, "b-1", state, ("a-1",), 50))

    def test_release_retries_only_a_recorded_in_use_deferral(self):
        done = ticket("a-1", MODEL_A, state="COMPLETED")
        waiting = ticket("b-1", MODEL_B, state="ACQUIRED", transition=desktop_load(), dispatched_at=5)
        controller, calls, events = self.controller([done, waiting])
        reprovision._reevaluate_phone_layout_after_release(controller, done, 900)
        self.assertEqual(calls, [])
        events.append({"kind": "EVALUATED", "desktop_reprovision": {"blocked_session_ids": ["HTP0"]}})
        reprovision._reevaluate_phone_layout_after_release(controller, done, 900)
        self.assertEqual(calls, [("b-1", MODEL_B.model_id, 900, None)])
        controller, calls, events = self.controller([done])
        events.append({"kind": "EVALUATED", "desktop_reprovision": {"blocked_session_ids": ["HTP0"]}})
        reprovision._reevaluate_phone_layout_after_release(controller, done, 900)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
