"""Phone re-provisioning for the desktop's model through the real portfolio path.

Only base-tree names are imported and the knob is duck-typed, so the base tree runs the
same scenarios; there the arrived-work portfolio keeps its static split
(``PHONE_RESIDENCY_LEARNING_RETAINED``) and the re-provisioning assertions fail."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_dev.scheduler import UnifiedScheduler
from research_dev.scheduler._internal.phone_shards import (
    PhoneFfnResidencyDemand, generate_mixed_ffn_residency_layouts,
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
A, B = MODEL_A.artifact_sha256, MODEL_B.artifact_sha256
SESSIONS = tuple(session(index) for index in range(3))
LIMIT = 768
PROJECTION = "sha256:" + "f" * 64
KNOB = SimpleNamespace(mode="resident-model", load_bytes_per_second=32, minimum_learned_samples=2)


def demand_row(model, work):
    return PhoneFfnResidencyDemand(model, work, 8, "split-row")


def counts(layout):
    result = {}
    for shard in layout.shards:
        result[shard.artifact_sha256] = result.get(shard.artifact_sha256, 0) + 1
    return result


def layers(layout):
    result = {}
    for shard in layout.shards:
        result[shard.artifact_sha256] = result.get(shard.artifact_sha256, 0) + bin(shard.layer_mask).count("1")
    return result


def layout_with_counts(rows, wanted):
    return next(
        row for row in generate_mixed_ffn_residency_layouts(rows, SESSIONS, phone_wide_limit_bytes=LIMIT)
        if counts(row) == wanted
    )


def desktop_load(latency_us=47_000_000):
    return SimpleNamespace(device_id="desktop-gpu", source_state="cold", target_state="hot",
                           latency_us=latency_us, prepares_device_ids=("desktop-gpu",))


def ticket(request_id, model, *, state="QUEUED", transition=None, output_tokens=100, dispatched_at=None):
    return SimpleNamespace(
        request=request(request_id, output_tokens=output_tokens),
        model=SimpleNamespace(model_id=model.model_id, artifact_sha256=model.artifact_sha256),
        dispatch_state=state,
        transition_status="PENDING" if transition is not None else "NOT_REQUIRED",
        execution_plan=SimpleNamespace(
            device_ids=("desktop-gpu",), transitions=() if transition is None else (transition,),
            helper_envelope=None,
        ),
        decision=SimpleNamespace(start_us=dispatched_at or 0),
        dispatch_receipt=None if dispatched_at is None else SimpleNamespace(observed_at_us=dispatched_at),
    )


def helper_binding(*session_ids):
    return {"fraction_ppm": 1_000_000, "helper_attachment": {
        "fraction_ppm": 1_000_000, "lease_tokens": ["lease"], "completed_phone_calls": 5,
        "fallback_outcome": None, "lease_reserved_until_us": 9, "allowed_session_ids": list(session_ids),
    }}


class PortfolioReprovisionTests(unittest.TestCase):
    def scheduler(self, tickets, *, knob=KNOB):
        desktop = SimpleNamespace(device_id="desktop-gpu", executor_id="executor:desktop-gpu",
                                  phone_sessions=(), adapter_parameters={},
                                  memory_resource_id="desktop-memory", workspace_bytes_per_token=0)
        phone = SimpleNamespace(device_id="phone-a", executor_id="executor:phone-a",
                                phone_sessions=SESSIONS, adapter_parameters={},
                                memory_resource_id="phone-memory", workspace_bytes_per_token=32)
        catalog = SimpleNamespace(
            executors=(desktop, phone), executor_by_device={"desktop-gpu": desktop, "phone-a": phone},
            transitions=(), minimum_energy_saving_ppm=10_000, phone_power_profile_by_device={},
            placement_profile=SimpleNamespace(
                devices={"phone-a": SimpleNamespace(allocation_limit_bytes=LIMIT, kind="phone")},
                memory_pools={"phone-memory": SimpleNamespace(capacity_bytes=LIMIT, reserved_bytes=0)},
            ),
        )
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler._runtime_capabilities = catalog
        scheduler._runtime_manifests = {MODEL_A.model_id: MODEL_A, MODEL_B.model_id: MODEL_B}
        scheduler._runtime_controller = SimpleNamespace(
            current_tickets=lambda: tuple(tickets), checkpoint=lambda: None, restore=lambda _: None,
        )
        scheduler._phone_reprovisioning = knob
        compiler = SimpleNamespace(
            phone_residency_demand=lambda model, work: (demand_row(model, work), SESSIONS, "phone-a"),
            phone_residency_evidence_status=lambda artifact: {"reason": "PHONE_RESIDENCY_ROUTE_EVIDENCE_READY"},
        )
        return scheduler, compiler

    @staticmethod
    def load(scheduler, state, at_us, duration_us):
        """Physically complete one PROPOSED layout (the helper-preparation transaction)."""
        placement = scheduler._model_placement_controller
        ticket_id = f"prepare-{state.generation}"
        placement.begin_phone_layout_transition(
            state.generation, ticket_id=ticket_id, transition_ids=("load",), ready_at_us=at_us + duration_us,
            workspace_bytes=32, observed_at_us=at_us, projection_token_sha256=PROJECTION,
        )
        placement.complete_phone_layout_transition(
            generation=state.generation, ticket_id=ticket_id, transition_ids=("load",),
            geometry_sha256=state.layout.geometry_sha256, projection_token_sha256=PROJECTION,
            finished_at_us=at_us + duration_us,
        )
        return placement.ready_phone_layout()

    def install_ready(self, scheduler, layout, at_us=1_000_000):
        state = scheduler._model_placement_controller.propose_phone_layout(
            layout, workspace_bytes=32, shared_compute_resource_id="phone-htp",
            shared_transport_resource_ids=("phone-functionfs", "phone-usb"), observed_at_us=at_us,
        )
        return self.load(scheduler, state, at_us, layout.resident_bytes * 1_000_000 // 32)

    @staticmethod
    def evaluate(scheduler, compiler, source, model, at_us):
        with patch.object(UnifiedScheduler, "_automated_compiler", return_value=compiler):
            scheduler._update_phone_residency_portfolio(source.request, model, at_us, None)
        return next(row for row in reversed(scheduler.phone_residency_events()) if row["kind"] == "EVALUATED")

    @staticmethod
    def target(scheduler):
        return scheduler._model_placement_controller.target_phone_layout()

    def test_executing_model_takes_the_static_split_session(self):
        # base tree: both models demanded and every session occupied -> LEARNING_RETAINED forever
        tickets = [ticket("a-1", MODEL_A, state="ACQUIRED"), ticket("b-1", MODEL_B)]
        scheduler, compiler = self.scheduler(tickets)
        split = layout_with_counts((demand_row(MODEL_A, 100), demand_row(MODEL_B, 100)), {A: 2, B: 1})
        b_session = next(row.session_id for row in split.shards if row.artifact_sha256 == B)
        self.install_ready(scheduler, split)
        event = self.evaluate(scheduler, compiler, tickets[0], MODEL_A, 61_000_000)
        self.assertEqual(event["reason"], "PHONE_RESIDENCY_DESKTOP_REPROVISION")
        recorded = event["desktop_reprovision"]
        self.assertEqual((recorded["mode"], recorded["desktop_commitment_source"]), ("FOLLOW", "executing"))
        self.assertEqual(recorded["followed_artifact_sha256s"], [A])
        self.assertEqual(recorded["target_layers_by_artifact"], {A: 6})
        self.assertTrue(event["selection_confirmed"])
        target = self.target(scheduler)
        self.assertEqual(target.state, "PROPOSED")
        self.assertEqual(target.layout.changed_session_ids, (b_session,))
        self.assertEqual(layers(target.layout), {A: 6})
        self.assertLessEqual(target.layout.resident_bytes, LIMIT)

    def test_desktop_load_starts_the_swap_inside_the_load_window_and_converges(self):
        dispatched_at = 100_000_000
        leader = ticket("b-1", MODEL_B, state="ACQUIRED", transition=desktop_load(47_000_000),
                        dispatched_at=dispatched_at)
        tickets = [leader, ticket("a-2", MODEL_A)]
        scheduler, compiler = self.scheduler(tickets)
        self.install_ready(scheduler, layout_with_counts((demand_row(MODEL_A, 100),), {A: 3}))
        event = self.evaluate(scheduler, compiler, leader, MODEL_B, dispatched_at)
        self.assertEqual(event["reason"], "PHONE_RESIDENCY_DESKTOP_REPROVISION")
        recorded = event["desktop_reprovision"]
        self.assertEqual(recorded["desktop_commitment_source"], "loading")
        self.assertEqual(recorded["leader_request_id"], "b-1")
        self.assertEqual(recorded["desktop_load_window_end_us"], dispatched_at + 47_000_000)
        self.assertEqual(recorded["stage_swap_latency_us"], 8_000_000)  # 256 B at 32 B/s
        self.assertEqual(recorded["target_swap_latency_us"], 24_000_000)
        self.assertTrue(recorded["fits_load_window"])
        self.assertEqual(recorded["target_layers_by_artifact"], {B: 6})
        self.assertTrue(event["selection_confirmed"])
        at_us = dispatched_at
        for stage in range(3):
            target = self.target(scheduler)
            self.assertEqual(target.state, "PROPOSED")
            self.assertEqual(len(target.layout.changed_session_ids), 1)
            self.assertEqual(layers(target.layout).get(B), 2 * (stage + 1))
            self.assertLessEqual(target.layout.resident_bytes, LIMIT)
            ready = self.load(scheduler, target, at_us, 8_000_000)
            at_us += 8_000_000
            event = self.evaluate(scheduler, compiler, leader, MODEL_B, at_us)
        self.assertEqual(layers(ready.layout), {B: 6})
        self.assertLessEqual(at_us, dispatched_at + 47_000_000)
        self.assertEqual(event["reason"], "PHONE_RESIDENCY_REPROVISION_RETAINED")
        self.assertEqual(event["desktop_reprovision"]["learned_load_samples"], 4)
        self.assertEqual(event["desktop_reprovision"]["load_bytes_per_second"], 32)
        self.assertIsNone(self.target(scheduler))

    def test_two_models_served_together_split_by_remaining_work(self):
        tickets = [ticket("a-1", MODEL_A, state="ACQUIRED", output_tokens=100),
                   ticket("b-1", MODEL_B, state="ACQUIRED", output_tokens=200)]
        scheduler, compiler = self.scheduler(tickets)
        self.install_ready(scheduler, layout_with_counts((demand_row(MODEL_A, 100),), {A: 3}))
        event = self.evaluate(scheduler, compiler, tickets[0], MODEL_A, 61_000_000)
        self.assertEqual(event["reason"], "PHONE_RESIDENCY_PROPORTIONAL_SPLIT")
        recorded = event["desktop_reprovision"]
        self.assertEqual(recorded["mode"], "PROPORTIONAL")
        self.assertEqual(recorded["target_session_counts_by_artifact"], {A: 1, B: 2})
        self.assertEqual(counts(self.target(scheduler).layout), {A: 2, B: 1})

    def test_in_use_sessions_are_never_evicted_and_the_swap_resumes_on_release(self):
        dispatched_at = 100_000_000
        using = ticket("a-1", MODEL_A, state="ACQUIRED", output_tokens=100)
        leader = ticket("b-1", MODEL_B, state="ACQUIRED", transition=desktop_load(), dispatched_at=dispatched_at,
                        output_tokens=200)
        tickets = [using, leader]
        scheduler, compiler = self.scheduler(tickets)
        self.install_ready(scheduler, layout_with_counts((demand_row(MODEL_A, 100),), {A: 3}))
        bindings = {"a-1": helper_binding("HTP0", "HTP1")}
        scheduler._model_placement_controller.request_binding = bindings.get
        event = self.evaluate(scheduler, compiler, leader, MODEL_B, dispatched_at)
        recorded = event["desktop_reprovision"]
        self.assertEqual(recorded["in_use_session_ids"], ["HTP0", "HTP1"])
        self.assertEqual(recorded["target_session_counts_by_artifact"], {A: 1, B: 2})
        self.assertTrue(recorded["blocked_session_ids"])
        self.assertLessEqual(set(recorded["blocked_session_ids"]), {"HTP0", "HTP1"})
        self.assertEqual(self.target(scheduler).layout.changed_session_ids, ("HTP2",))
        self.load(scheduler, self.target(scheduler), dispatched_at, 8_000_000)
        event = self.evaluate(scheduler, compiler, leader, MODEL_B, dispatched_at + 8_000_000)
        self.assertEqual(event["reason"], "PHONE_RESIDENCY_REPROVISION_DEFERRED_IN_USE")
        self.assertIsNone(self.target(scheduler))
        ready = scheduler._model_placement_controller.ready_phone_layout().layout
        self.assertEqual([row.artifact_sha256 for row in ready.shards if row.session_id != "HTP2"], [A, A])
        # the helper releases HTP0/HTP1 when a-1 completes: the swap continues without a new arrival
        del tickets[0]
        del bindings["a-1"]
        with patch.object(UnifiedScheduler, "_automated_compiler", return_value=compiler):
            scheduler._reevaluate_phone_layout_after_release(using, dispatched_at + 9_000_000)
        event = next(row for row in reversed(scheduler.phone_residency_events()) if row["kind"] == "EVALUATED")
        self.assertEqual(event["reason"], "PHONE_RESIDENCY_DESKTOP_REPROVISION")
        self.assertEqual(event["desktop_reprovision"]["in_use_session_ids"], [])
        self.assertEqual(event["desktop_reprovision"]["target_layers_by_artifact"], {B: 6})
        self.assertEqual(layers(self.target(scheduler).layout), {A: 2, B: 4})

    def test_arrival_of_the_other_model_does_not_swap_an_idle_phone(self):
        # run-5: the first Gemma arrival took a Qwen session 220 s before Gemma ran
        tickets = [ticket("a-1", MODEL_A, state="ACQUIRED")]
        scheduler, compiler = self.scheduler(tickets)
        full_a = self.install_ready(scheduler, layout_with_counts((demand_row(MODEL_A, 100),), {A: 3}))
        self.evaluate(scheduler, compiler, tickets[0], MODEL_A, 61_000_000)
        tickets[0] = ticket("b-1", MODEL_B)
        event = self.evaluate(scheduler, compiler, tickets[0], MODEL_B, 62_000_000)
        self.assertEqual(event["reason"], "PHONE_RESIDENCY_REPROVISION_HOLD_IDLE_MODEL")
        self.assertEqual(event["desktop_reprovision"]["desktop_commitment_source"], "retained")
        self.assertIsNone(self.target(scheduler))
        self.assertEqual(scheduler._model_placement_controller.ready_phone_layout().layout, full_a.layout)

    def test_dispatch_with_a_desktop_load_triggers_the_reevaluation_only_when_configured(self):
        leader = ticket("b-1", MODEL_B, state="ACQUIRED", transition=desktop_load(), dispatched_at=5)
        for knob, expected in ((KNOB, [leader]), (None, [])):
            scheduler, _compiler = self.scheduler([leader], knob=knob)
            scheduler._runtime_controller.wait_ready = lambda request_id, epoch_ns: "receipt"
            scheduler._runtime_controller.commit_wait = lambda receipt, commit_acquired: leader
            calls = []
            with patch.object(UnifiedScheduler, "_reevaluate_phone_layout_for_desktop_load", create=True,
                              new=lambda _self, value: calls.append(value)):
                self.assertIs(scheduler.wait_runtime_request("b-1", 0), leader)
            self.assertEqual(calls, expected)

    def test_knob_off_keeps_the_base_selection(self):
        tickets = [ticket("a-1", MODEL_A, state="ACQUIRED"), ticket("b-1", MODEL_B)]
        scheduler, compiler = self.scheduler(tickets, knob=None)
        split = layout_with_counts((demand_row(MODEL_A, 100), demand_row(MODEL_B, 100)), {A: 2, B: 1})
        self.install_ready(scheduler, split)
        event = self.evaluate(scheduler, compiler, tickets[0], MODEL_A, 61_000_000)
        self.assertEqual(event["reason"], "PHONE_RESIDENCY_LEARNING_RETAINED")
        self.assertNotIn("desktop_reprovision", event)
        self.assertIsNone(self.target(scheduler))


if __name__ == "__main__":
    unittest.main()
