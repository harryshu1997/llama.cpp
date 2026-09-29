"""Receipt attribution uses acquisition history, not campaign labels."""

from types import SimpleNamespace
from dataclasses import replace
import time
import threading
import unittest
from unittest.mock import patch

from research_dev.scheduler import UnifiedScheduler
from research_dev.scheduler.adapters.energy import (
    PhoneActivityIntervalTracker, PolledPhonePowerSampler,
    RaplNvmlPhoneEnergyMeter,
)
from research_dev.scheduler._internal.runtime_capabilities import RuntimePhonePowerProfile
from research_dev.scheduler.tests.test_runtime_controller import (
    profile, request, model, cpu_binding, phone_binding, snapshot, wait_acquired,
)
from research_dev.scheduler.tests import test_automated_runtime as automated
from research_dev.scheduler._unified.automated_selection_ops.objectives import protected_work_budget_allows
from research_dev.scheduler._internal.policy import LeaseDemand, ResourceTimeline, ResourceProfile
from research_dev.scheduler._internal.runtime_resources import runtime_preparation_windows
from research_dev.scheduler.adapters.heterogeneous_rig import HeterogeneousPhysicalRig
from research_dev.scheduler.adapters.runtime import CanonicalPhysicalAdapter
from research_dev.scheduler.adapters.ticket import interpret_runtime_ticket
from research_dev.scheduler.adapters.contracts import RawTransitionObservation
from research_dev.scheduler._internal.runtime_system_cost import calculate_marginal_system_cost
from research_dev.scheduler._internal.route_generation.costing import _physical_interference_resources


class ReceiptEnergyTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = UnifiedScheduler((profile(),), "enforce")
        self.tracker = PhoneActivityIntervalTracker()
        rows = tuple({
            "gpu": {"sample_t_ns": t}, "rapl_package": {"sample_t_ns": t},
        } for t in (0, 4_000_000_000))
        self.meter = RaplNvmlPhoneEnergyMeter(
            lambda: rows,
            lambda *_: {"cpu_package_energy_j": 1., "gpu_board_energy_j": 2.},
            PolledPhonePowerSampler(lambda: None),
            energy_boundary_id="fleet", attribution_kind="matched_abba",
            phone_power_profile=RuntimePhonePowerProfile.assumed_4p5w(
                device_id="phone", domain_id="phone-system"),
            phone_activity=self.tracker,
        )
        self.meter.bind_scheduler(self, 0)

    def runtime_receipt_energy_context(self, ticket_id, devices, start, end):
        if "phone" in devices:
            return {"reason": "ENERGY_ROUTE_TOUCHES_PHONE"}
        overlaps = self.scheduler._runtime_controller.acquired_lease_overlaps(
            ticket_id, ("desktop-cpu", "phone-compute"), start, end)
        return {
            "reason": "ENERGY_LEDGER_WINDOW_UNAVAILABLE" if overlaps is None else (
                "ENERGY_SERVER_LEASE_OVERLAP" if overlaps else "ENERGY_ISOLATED_SERVER_RECEIPT"),
            "overlapping_ticket_ids": overlaps or (),
        }

    def acquire(self, identity, binding):
        self.scheduler.submit_runtime_request(
            request(identity), model(), (
                cpu_binding(), phone_binding(ready=binding.route_id == "phone-full")),
            snapshot=snapshot(1000), observed_at_us=1000)
        return wait_acquired(self.scheduler, identity)

    def receipt(self, ticket, devices=("cpu",)):
        return SimpleNamespace(ticket_id=ticket.ticket_id,
                               participants=tuple(SimpleNamespace(device_id=d) for d in devices))

    def test_solo_receipt_is_attributable_with_assumed_idle_phone(self):
        owner = self.acquire("solo", cpu_binding())
        result = self.meter.measure_receipt(self.receipt(owner), 1_000_000_000, 2_000_000_000)
        self.assertEqual(result.attribution_kind, "isolated")
        self.assertEqual(result.estimation_metadata["phone_active_time_ns"], 0)
        self.assertIn("ASSUMED_4P5W", result.measurement_evidence_ids)

    def test_whole_campaign_does_not_inherit_attribution_label(self):
        self.assertEqual(self.meter.measure(1_000_000_000, 2_000_000_000).attribution_kind,
                         "diagnostic")

    def test_phone_activity_or_phone_route_is_diagnostic(self):
        owner = self.acquire("solo", cpu_binding())
        result = self.meter.measure_receipt(self.receipt(owner, ("phone",)),
                                           1_000_000_000, 2_000_000_000)
        self.assertEqual(result.estimation_metadata["energy_attribution_reason"], "ENERGY_ROUTE_TOUCHES_PHONE")
        self.tracker.record("phone-load", "load", 1_100_000_000, 1_200_000_000)
        result = self.meter.measure_receipt(self.receipt(owner), 1_000_000_000, 2_000_000_000)
        self.assertEqual(result.estimation_metadata["energy_attribution_reason"], "ENERGY_PHONE_ACTIVITY_OVERLAP")

    def test_completed_peer_still_makes_overlapping_receipt_diagnostic(self):
        owner = self.acquire("owner", cpu_binding())
        peer = self.acquire("peer", phone_binding())
        self.scheduler.complete_runtime_request(peer.request.request_id, 1_500_000)
        result = self.meter.measure_receipt(self.receipt(owner), 1_000_000_000, 2_000_000_000)
        self.assertEqual(result.attribution_kind, "diagnostic")
        self.assertEqual(result.estimation_metadata["energy_attribution_overlapping_tickets"], peer.ticket_id)
        result = self.meter.measure_receipt(self.receipt(owner), 2_000_000_000, 3_000_000_000)
        self.assertEqual(result.attribution_kind, "isolated")

    def test_unacquired_ticket_cannot_claim_isolation(self):
        receipt = SimpleNamespace(ticket_id="unknown", participants=(SimpleNamespace(device_id="cpu"),))
        result = self.meter.measure_receipt(receipt, 1_000_000_000, 2_000_000_000)
        self.assertEqual(result.attribution_kind, "diagnostic")


class PhaseLeaseTests(unittest.TestCase):
    setUp = automated.AutomatedRuntimeTests.setUp
    tearDown = automated.AutomatedRuntimeTests.tearDown
    scheduler_and_manifest = automated.AutomatedRuntimeTests.scheduler_and_manifest

    def cold_ticket(self):
        source = automated.catalog()
        resources = {name: replace(row, capacity=4) if name.startswith("compute:") else row
                     for name, row in source.resources.items()}
        transitions = tuple(replace(row, resource_slots={r: 4 for r in row.resource_ids})
                            for row in source.transitions)
        scheduler, manifest = self.scheduler_and_manifest(replace(source, resources=resources, transitions=transitions))
        ticket = scheduler.submit_automated_request(
            automated.request("phase"), manifest.model_id,
            automated.runtime_snapshot(manifest, resident_devices=()),
            selection_mode="desktop-baseline")
        ticket = scheduler.wait_runtime_request(
            "phase", time.monotonic_ns() - ticket.decision.start_us * 1000)
        return scheduler, ticket

    def test_receipt_releases_prepare_capacity_and_keeps_execution_leases(self):
        scheduler, ticket = self.cold_ticket()
        self.assertTrue(ticket.prepare_lease_tokens)
        self.assertTrue(all(len(row.lanes) == 4 for row in ticket.decision.leases
                            if row.token in ticket.prepare_lease_tokens))
        receipts = automated.FakeAutomatedPhysicalAdapter._transition_receipts(ticket)
        updated = scheduler.record_automated_transition_receipts("phase", receipts)
        self.assertTrue(all(len(row.lanes) == 1 for row in updated.live_leases))
        finished = max(row.finished_us for row in receipts)
        calendar = scheduler.timeline.causal_state()
        self.assertTrue(calendar)
        renewed = scheduler.extend_runtime_request(
            "phase", at_us=finished, reserved_until_us=ticket.decision.finish_upper_us + 1000)
        self.assertTrue(renewed.extended_leases)
        self.assertFalse(ticket.prepare_lease_tokens & {row["token"] for row in renewed.extended_leases})
        released = scheduler.release_automated_runtime_capacity(
            "phase", expected_ticket_id=ticket.ticket_id, actual_end_us=finished + 1)
        self.assertTrue(released.lease_coverage.covered)
        self.assertFalse(ticket.prepare_lease_tokens & set(released.released_tokens))

    def test_load_overrun_moves_future_execution_without_self_overlap(self):
        scheduler, ticket = self.cold_ticket()
        end = ticket.decision.finish_upper_us + 1000
        scheduler.extend_runtime_request("phase", at_us=ticket.decision.start_us, reserved_until_us=end)
        receipts = tuple(replace(row, finished_us=end - 1) for row in
                         automated.FakeAutomatedPhysicalAdapter._transition_receipts(ticket))
        updated = scheduler.record_automated_transition_receipts("phase", receipts)
        self.assertEqual(updated.prepare_lease_tokens, ticket.prepare_lease_tokens)
        self.assertEqual({row.token for row in updated.live_leases},
                         {row.token for row in ticket.decision.leases} - ticket.prepare_lease_tokens)

    def test_phase_commit_failure_restores_ticket_queue_and_calendar(self):
        scheduler, ticket = self.cold_ticket()
        before = scheduler.timeline.checkpoint()
        queue = scheduler._runtime_controller.queue.checkpoint()
        with patch.object(scheduler.timeline, "retime_many", side_effect=ValueError("injected retime")):
            with self.assertRaisesRegex(ValueError, "injected retime"):
                scheduler.record_automated_transition_receipts(
                    "phase", automated.FakeAutomatedPhysicalAdapter._transition_receipts(ticket),
                )
        self.assertEqual(scheduler.timeline.checkpoint(), before)
        self.assertEqual(scheduler._runtime_controller.queue.checkpoint(), queue)
        self.assertEqual(scheduler.runtime_ticket("phase"), ticket)

    def test_unknown_protected_horizon_does_not_masquerade_as_idle(self):
        scheduler, manifest = self.scheduler_and_manifest()
        current = automated.protected_snapshot(manifest)
        current = replace(current, protected_work=replace(
            current.protected_work, critical_path_end_us=0, measured=False,
        ))
        candidates = scheduler.generate_automated_candidates(
            automated.request("unknown-horizon"), manifest.model_id, current,
        )
        self.assertTrue(candidates.baseline.admitted)
        self.assertTrue(any(
            "MARGINAL_SYSTEM_COST_UNKNOWN" in row.rejection_reasons
            for row in candidates.candidates if not row.baseline
        ))

    def test_budget_policy_is_explicit_and_calibration_stays_protected(self):
        source = automated.catalog(
            phone_ops_per_s=6_000_000_000, phone_power_mw=1500,
            phone_bandwidth=8_000_000_000,
            system_cost_profiles=(automated.system_cost_profile(helper_interference_ppm=1),),
        )
        for policy, mode in (("strict", "energy-aware"),
                             ("energy-budgeted", "energy-aware"),
                             ("energy-budgeted", "calibration")):
            with self.subTest(policy=policy, mode=mode):
                scheduler = UnifiedScheduler.for_runtime_discovery(
                    "enforce", protected_work_policy=policy,
                )
                scheduler.register_runtime_capabilities(source)
                manifest = scheduler.register_gguf_model("model", self.path)
                ticket = scheduler.submit_automated_request(
                    automated.request("policy"), manifest.model_id,
                    automated.protected_snapshot(manifest), selection_mode=mode,
                )
                if policy == "energy-budgeted" and mode == "energy-aware":
                    self.assertEqual(ticket.decision.reason, "BUDGETED_PROTECTED_WORK_ENERGY_SAVING")
                    self.assertGreater(ticket.decision.marginal_system_cost["interference_upper_us"], 0)
                else:
                    self.assertNotEqual(ticket.decision.reason, "BUDGETED_PROTECTED_WORK_ENERGY_SAVING")
                    self.assertEqual(ticket.decision.marginal_system_cost["interference_upper_us"], 0)

    def test_receipt_context_uses_real_catalog_and_acquired_plan(self):
        scheduler, ticket = self.cold_ticket()
        at = ticket.dispatch_receipt.observed_at_us
        context = scheduler.runtime_receipt_energy_context(ticket.ticket_id,
                                                           ticket.execution_plan.device_ids, at, at + 1)
        self.assertEqual(context["reason"], "ENERGY_ISOLATED_SERVER_RECEIPT")
        self.assertEqual(scheduler.runtime_receipt_energy_context(
            ticket.ticket_id, ("helper-c",), at, at + 1)["reason"], "ENERGY_LEDGER_IDENTITY_UNAVAILABLE")
        self.assertGreater(scheduler.runtime_protected_work_end_us((ticket.ticket_id,)), at)

    def test_multi_phase_context_allocation_skips_occupied_combinations(self):
        timeline = ResourceTimeline({"ctx": ResourceProfile("ctx", "compute", 512, True, "ctx")})
        timeline.commit_leases(timeline.preview_leases((LeaseDemand("busy", "ctx", 1, 0, 1000, 1000),),
                                                       0, 1000, 1000), "busy")
        preview = timeline.preview_leases((LeaseDemand("load", "ctx", 1, 0, 10, 10),
                                           LeaseDemand("decode", "ctx", 8, 10, 100, 100)), 0, 110, 110)
        self.assertEqual(preview.start_us, 0)
        self.assertEqual(preview.plans[1].lanes, tuple(range(1, 9)))


class SequentialPreparationTests(unittest.TestCase):
    setUp = PhaseLeaseTests.setUp
    tearDown = PhaseLeaseTests.tearDown
    scheduler_and_manifest = PhaseLeaseTests.scheduler_and_manifest
    def two_loads(self):
        source = automated.catalog()
        source = replace(
            source,
            resources={key: replace(row, capacity=4) if key.startswith("compute:") else row
                       for key, row in source.resources.items()},
            transitions=tuple(replace(
                row, resource_ids=tuple(sorted(set(row.resource_ids) | {"compute:host-a"})),
                resource_slots={key: 4 for key in set(row.resource_ids) | {"compute:host-a"}},
            ) for row in source.transitions),
        )
        scheduler, manifest = self.scheduler_and_manifest(source)
        self.transition_snapshot = automated.runtime_snapshot(manifest, resident_devices=())

        def select(candidates, *_args, **_kwargs):
            candidate = next(row for row in candidates.candidates
                             if row.admitted and row.binding.route_id.startswith(
                                 "auto:layers:accelerator-b+host-a:"))
            return candidate, (), "SEQUENTIAL_LOAD_TEST_FIXTURE"

        with patch.object(scheduler, "_select_automated_candidate", side_effect=select):
            ticket = scheduler.submit_automated_request(
                automated.request("two-loads"), manifest.model_id,
                automated.runtime_snapshot(manifest, resident_devices=()),
            )
        ticket = scheduler.wait_runtime_request(
            "two-loads", time.monotonic_ns() - ticket.decision.start_us * 1000)
        self.assertEqual(len(ticket.execution_plan.transitions), 2)
        receipts = automated.FakeAutomatedPhysicalAdapter._transition_receipts(ticket)
        cursor = ticket.decision.start_us
        ordered = []
        for row, transition in zip(receipts, ticket.execution_plan.transitions):
            end = cursor + transition.latency_us
            ordered.append(replace(row, started_us=cursor, finished_us=end))
            cursor = end
        return scheduler, ticket, tuple(ordered)

    def test_two_loads_share_resource_in_adapter_order_and_release_individually(self):
        scheduler, ticket, receipts = self.two_loads()
        leases = [ticket.transition_lease_by_resource(row)["compute:host-a"]
                  for row in ticket.execution_plan.transitions]
        self.assertEqual(leases[0].reserved_until_us, leases[1].start_us)
        early = ticket.decision.start_us + 1
        receipts = (replace(receipts[0], finished_us=early),
                    replace(receipts[1], started_us=early,
                            finished_us=early + ticket.execution_plan.transitions[1].latency_us))
        memory = scheduler._runtime_memory.checkpoint()
        partial = scheduler.record_automated_transition_receipts("two-loads", receipts[:1])
        self.assertEqual(partial.transition_status, "PENDING")
        self.assertIn(leases[0].token, partial.completed_prepare_lease_tokens)
        self.assertIn(leases[1], partial.live_leases)
        with self.assertRaisesRegex(ValueError, "transitions are not complete"):
            scheduler.runtime_execution_ticket("two-loads")
        ready = scheduler.record_automated_transition_receipts("two-loads", receipts)
        self.assertEqual(ready.transition_status, "COMPLETED")
        self.assertEqual(scheduler._runtime_memory.checkpoint(), memory)
        windows = runtime_preparation_windows(ready)
        calendar = {row["token"]: (row["start_us"], row["end_us"])
                    for resource in scheduler.timeline.causal_state()["resources"].values()
                    for lane in resource["lanes"] for row in lane}
        for lease, receipt in zip(leases, receipts):
            self.assertEqual(windows[lease.token][1], receipt.finished_us)
            self.assertEqual(calendar[lease.token], windows[lease.token])
        self.assertTrue(all(len(row.lanes) == 1 for row in ready.live_leases))

    def test_overrun_renews_active_load_and_shifts_successors_atomically(self):
        scheduler, ticket, receipts = self.two_loads()
        first, second = ticket.execution_plan.transitions
        first_end = receipts[0].finished_us + 10
        scheduler.extend_runtime_request("two-loads", at_us=first_end - 1,
                                         reserved_until_us=first_end)
        updated = scheduler.runtime_ticket("two-loads")
        windows = runtime_preparation_windows(updated)
        first_tokens = {row.token for row in ticket.transition_lease_by_resource(first).values()
                        if row.token in ticket.prepare_lease_tokens and first.transition_id in row.lease_id}
        for token in first_tokens:
            self.assertEqual(windows[token][1], first_end)
        second_lease = ticket.transition_lease_by_resource(second)["compute:host-a"]
        self.assertEqual(windows[second_lease.token], (first_end, first_end + second.latency_us))
        first_receipt = replace(receipts[0], finished_us=first_end)
        partial = scheduler.record_automated_transition_receipts("two-loads", (first_receipt,))
        second_end = first_end + second.latency_us + 50
        renewal = scheduler.extend_runtime_request("two-loads", at_us=second_end - 1,
                                                   reserved_until_us=second_end)
        self.assertFalse(partial.completed_prepare_lease_tokens &
                         {row["token"] for row in renewal.extended_leases})
        final = scheduler.record_automated_transition_receipts("two-loads", (
            first_receipt, replace(receipts[1], started_us=first_end, finished_us=second_end),
        ))
        windows = runtime_preparation_windows(final)
        self.assertEqual(windows[second_lease.token], (first_end, second_end))
        for lease in final.live_leases:
            self.assertEqual(windows[lease.token][0], second_end)
            self.assertGreaterEqual(windows[lease.token][1] - second_end,
                                    lease.reserved_until_us - lease.start_us)

    def test_partial_receipt_transaction_rolls_back_and_prefix_cannot_change(self):
        scheduler, ticket, receipts = self.two_loads()
        partial = scheduler.record_automated_transition_receipts("two-loads", receipts[:1])
        before = (scheduler.timeline.checkpoint(), scheduler._runtime_controller.queue.checkpoint(),
                  scheduler._runtime_memory.checkpoint())
        release = scheduler._runtime_controller.queue.release_prepare_leases

        def fail_after_release(*args, **kwargs):
            release(*args, **kwargs)
            raise ValueError("injected post-release failure")

        with patch.object(scheduler._runtime_controller.queue, "release_prepare_leases",
                          side_effect=fail_after_release):
            with self.assertRaisesRegex(ValueError, "post-release"):
                scheduler.record_automated_transition_receipts("two-loads", receipts)
        self.assertEqual(scheduler.runtime_ticket("two-loads"), partial)
        self.assertEqual(before, (scheduler.timeline.checkpoint(),
                                 scheduler._runtime_controller.queue.checkpoint(),
                                 scheduler._runtime_memory.checkpoint()))
        with self.assertRaisesRegex(ValueError, "receipt prefix changed"):
            scheduler.record_automated_transition_receipts("two-loads", (
                replace(receipts[0], finished_us=receipts[0].finished_us + 1), receipts[1],
            ))
        scheduler.record_automated_transition_receipts("two-loads", receipts)

    def test_adapter_publishes_verified_prefix_before_next_load(self):
        scheduler, ticket, receipts = self.two_loads()
        commands = interpret_runtime_ticket(ticket)
        seen = []

        def apply(command, _payload, _check):
            index = len(seen)
            current = scheduler.runtime_ticket("two-loads")
            self.assertEqual(len(current.transition_receipts), index)
            if index:
                self.assertTrue(current.completed_prepare_lease_tokens)
                self.assertEqual(current.transition_status, "PENDING")
            seen.append(command.transition.transition_id)
            return RawTransitionObservation(
                started_us=receipts[index].started_us, finished_us=receipts[index].finished_us,
                status="COMPLETED",
            )

        backend = SimpleNamespace(apply_transition=apply, execute=lambda *_: None)
        adapter = CanonicalPhysicalAdapter(
            scheduler, backend, epoch_ns=0, snapshot_provider=lambda *_: self.transition_snapshot)
        with patch.object(scheduler, "start_runtime_lease_renewal"), \
             patch.object(scheduler, "stop_runtime_lease_renewal"), \
             patch.object(scheduler, "observe_automated_runtime_snapshot"):
            ready, recovery = adapter._apply_transitions(ticket, commands, {})
        self.assertIsNone(recovery)
        self.assertEqual(ready.transition_status, "COMPLETED")
        self.assertEqual(seen, [row.transition_id for row in ticket.execution_plan.transitions])


class ProtectedWorkBudgetTests(unittest.TestCase):
    def transport_scope(self):
        shared = ("transport:root", "transport:session")
        links = tuple(SimpleNamespace(
            link_id="generated:" + direction, source_device=source, target_device=target,
            status="measured", transport_generation="wire-v1", transport_profile_id="physical-link",
            qualification_identity_sha256="sha256:" + "a" * 64,
        ) for direction, source, target in (("out", "host", "edge"), ("in", "edge", "host")))
        resources = {key: SimpleNamespace(kind=kind) for key, kind in (
            ("host", "compute"), ("helper", "compute"), ("coordinator", "coordinator"),
            *((key, "transport") for key in shared),
            *(("link:" + link.link_id, "transport") for link in links),
        )}
        catalog = SimpleNamespace(
            resources=resources, placement_profile=SimpleNamespace(links=links),
            executor_by_device={"edge": SimpleNamespace(phone_sessions=(
                SimpleNamespace(shared_transport_resource_ids=shared),))},
        )
        coordinator = SimpleNamespace(assisted_operator_kind="ffn", adapter_parameters={
            "phone_device_id": "edge", "functionfs_resource_id": "transport:session",
        })
        profile = replace(automated.system_cost_profile(helper_interference_ppm=0),
                          interference_ppm_by_resource={"host": 0, "helper": 0,
                                                        shared[0]: 100_000, shared[1]: 200_000})
        return catalog, coordinator, profile

    def test_qualified_ffn_link_aliases_charge_shared_transport_once(self):
        catalog, coordinator, profile = self.transport_scope()
        resource_ids = tuple(catalog.resources)
        links = tuple(row.link_id for row in catalog.placement_profile.links)
        scoped = _physical_interference_resources(catalog, resource_ids, links, coordinator, profile)
        self.assertEqual(scoped, ("helper", "host", "transport:root", "transport:session"))
        self.assertEqual(profile.missing_resources(scoped), ())
        observed, _ = self.rig()._protected_work_observation(
            (SimpleNamespace(ticket_id="protected"),), False, 1_000_000)
        cost = profile.route_cost(scoped, observed, service_us=1000, service_upper_us=1000,
                                  finish_us=2000, finish_upper_us=2000)
        self.assertEqual(cost["interference_ppm"], 300_000)
        self.assertGreater(cost["upper_uj"], 0)
        self.assertEqual(tuple(catalog.resources), resource_ids)

    def test_transport_scope_does_not_invent_missing_or_unqualified_evidence(self):
        for case in ("unqualified-link", "missing-physical", "unmapped-device", "wrong-contract", "unknown-link"):
            with self.subTest(case=case):
                catalog, coordinator, profile = self.transport_scope()
                if case == "unqualified-link":
                    catalog.placement_profile.links[0].qualification_identity_sha256 = None
                elif case == "missing-physical":
                    profile = replace(profile, interference_ppm_by_resource={"host": 0, "helper": 0})
                elif case == "unmapped-device":
                    coordinator.adapter_parameters["phone_device_id"] = "different"
                elif case == "wrong-contract":
                    coordinator.adapter_parameters["functionfs_resource_id"] = "different"
                else:
                    catalog.resources["unmapped-transport"] = SimpleNamespace(kind="transport")
                scoped = _physical_interference_resources(
                    catalog, tuple(catalog.resources),
                    tuple(row.link_id for row in catalog.placement_profile.links), coordinator, profile)
                self.assertTrue(profile.missing_resources(scoped))

    def rig(self):
        rig = object.__new__(HeterogeneousPhysicalRig)
        rig._lock = threading.RLock()
        rig.epoch_ns = 1_000_000_000
        rig._active_large_since_ns = {"protected": 1_000_000_000}
        rig._execution_markers = {"protected": object()}
        self.power_rows = [{
            "gpu": {"sample_t_ns": at, "power_mw": 60_000},
            "rapl_package": {"sample_t_ns": at, "name": "package-0",
                             "energy_uj": energy, "max_energy_range_uj": 100_000_000},
        } for at, energy in ((1_000_000_000, 1000), (2_000_000_000, 40_001_000))]
        rig._sampler = SimpleNamespace(latest_rows=lambda: self.power_rows)
        rig._scheduler = SimpleNamespace(
            runtime_protected_work_end_us=lambda _: 10_000_000,
            runtime_protected_work_power_context=lambda *_: {"reason": "PROTECTED_POWER_ISOLATED"},
        )
        return rig

    def test_owned_helper_does_not_compete_with_its_own_execution(self):
        rig = self.rig()
        owner = SimpleNamespace(ticket_id="protected", request_id="owner")
        observed, features = rig._protected_work_observation(
            (owner,), False, 1_000_000, request_id="owner")
        self.assertIsNone(observed)
        self.assertEqual(features["protected_work_own_ticket_count"], 1)
        self.assertEqual(features["protected_work_peer_count"], 0)
        self.assertEqual(features["protected_power_valid"], 0)

    def test_protected_power_uses_only_samples_at_or_before_snapshot(self):
        rig = self.rig()
        self.power_rows.append({
            "gpu": {"sample_t_ns": 2_100_000_000, "power_mw": 900_000},
            "rapl_package": {"sample_t_ns": 2_100_000_000, "name": "package-0",
                             "energy_uj": 90_001_000, "max_energy_range_uj": 100_000_000},
        })
        observed, features = rig._protected_work_observation(
            (SimpleNamespace(ticket_id="protected"),), False, 1_000_000)
        self.assertTrue(observed.measured)
        self.assertEqual(observed.phase_power_mw, 100_000)
        self.assertEqual(features["protected_power_sample_end_us"], 1_000_000)
        self.assertEqual(features["protected_power_sample_age_us"], 0)

    def test_owned_helper_keeps_peers_and_unknown_or_switching_work_protected(self):
        rig = self.rig()
        owner = SimpleNamespace(ticket_id="owner-ticket", request_id="owner")
        peer = SimpleNamespace(ticket_id="protected", request_id="peer")
        rig._execution_markers[owner.ticket_id] = object()
        observed, features = rig._protected_work_observation(
            (owner, peer), False, 1_000_000, request_id="owner")
        self.assertTrue(observed.measured)
        self.assertEqual(features["protected_work_peer_count"], 1)
        rig._execution_markers["unscoped-peer"] = object()
        observed, _ = rig._protected_work_observation(
            (owner,), False, 1_000_000, request_id="owner")
        self.assertFalse(observed.measured)
        self.assertEqual(observed.observation_id, "PROTECTED_POWER_PEER_SCOPE_UNAVAILABLE")
        rig._execution_markers = {owner.ticket_id: object()}
        observed, _ = rig._protected_work_observation(
            (owner,), True, 1_000_000, request_id="owner")
        self.assertFalse(observed.measured)

    def test_protected_power_includes_cpu_and_gpu_extension_without_route_tail_double_charge(self):
        rig = self.rig()
        observed, features = rig._protected_work_observation(
            (SimpleNamespace(ticket_id="protected"),), False, 1_000_000)
        self.assertTrue(observed.measured)
        self.assertEqual(observed.phase_power_mw, 100_000)
        self.assertEqual(features["protected_cpu_power_mw"], 40_000)
        self.assertEqual(features["protected_gpu_power_mw"], 60_000)
        cost = calculate_marginal_system_cost(
            critical_path_end_us=observed.critical_path_end_us,
            phase_power_mw=observed.phase_power_mw,
            stranded_idle_power_mw=observed.stranded_idle_power_mw,
            causal_tail_power_mw=observed.causal_tail_power_mw,
            interference_ppm=100_000, lower_error_ppm=0, upper_error_ppm=0,
            service_us=1_000_000, service_upper_us=1_000_000,
            finish_us=12_000_000, finish_upper_us=12_000_000,
        )
        self.assertEqual(cost["phase_interference_uj"], 10_000_000)
        self.assertEqual(cost["total_uj"], 10_000_000)
        self.assertEqual(cost["causal_tail_uj"], 0)

    def test_missing_stale_or_contaminated_power_remains_unknown(self):
        for case in ("missing", "stale", "overlap", "switching", "identity"):
            with self.subTest(case=case):
                rig = self.rig()
                if case == "missing":
                    self.power_rows[-1]["rapl_package"] = None
                elif case == "overlap":
                    rig._scheduler.runtime_protected_work_power_context = lambda *_: {
                        "reason": "PROTECTED_POWER_OVERLAP"}
                elif case == "identity":
                    self.power_rows[-1]["rapl_package"]["name"] = "different-package"
                observed, features = rig._protected_work_observation(
                    (SimpleNamespace(ticket_id="protected"),), case == "switching",
                    4_000_001 if case == "stale" else 1_000_000,
                )
                self.assertFalse(observed.measured)
                self.assertEqual(features["protected_power_valid"], 0)
                self.assertNotEqual(observed.observation_id, "RAPL_NVML_PROTECTED_POWER")

    def test_strict_default_and_budgeted_upper_bound(self):
        baseline = SimpleNamespace(marginal_system_cost=None, cost=SimpleNamespace(fleet_energy_lower_uj=1000))
        alternative = SimpleNamespace(marginal_system_cost={
            "route_energy_upper_uj": 600, "upper_uj": 200, "measured": True,
        })
        self.assertFalse(protected_work_budget_allows("strict", alternative, baseline, 100_000))
        self.assertTrue(protected_work_budget_allows("energy-budgeted", alternative, baseline, 100_000))
        alternative.marginal_system_cost["upper_uj"] = 300
        self.assertFalse(protected_work_budget_allows("energy-budgeted", alternative, baseline, 100_000))
        alternative.marginal_system_cost["upper_uj"] = None
        self.assertFalse(protected_work_budget_allows("energy-budgeted", alternative, baseline, 100_000))
        alternative.marginal_system_cost.update(upper_uj=100, measured=False)
        self.assertFalse(protected_work_budget_allows("energy-budgeted", alternative, baseline, 100_000))

    def test_budget_cancels_common_idle_and_charges_lost_assistance(self):
        # Waiting alternative: 5,500 J route energy of which 5,300 J is idle
        # accrued while the protected work runs anyway; its own work is 200 J.
        baseline = SimpleNamespace(
            marginal_system_cost={
                "route_energy_lower_uj": 5_500, "route_energy_incremental_lower_uj": 200,
                "protected_overlap_idle_uj": 5_300, "measured": True,
            },
            cost=SimpleNamespace(fleet_energy_lower_uj=5_500),
        )
        alternative = SimpleNamespace(marginal_system_cost={
            "route_energy_upper_uj": 1_200, "route_energy_incremental_upper_uj": 1_200,
            "protected_overlap_idle_uj": 0, "upper_uj": 380, "measured": True,
        })
        # The old comparison would have admitted this (5,500 * 0.99 - 1,200 > 380).
        self.assertFalse(protected_work_budget_allows(
            "energy-budgeted", alternative, baseline, 10_000, assistance_loss_upper_uj=0))
        # A genuinely cheaper start-now still passes on incremental energy.
        alternative.marginal_system_cost.update(
            route_energy_upper_uj=100, route_energy_incremental_upper_uj=100, upper_uj=20)
        self.assertTrue(protected_work_budget_allows(
            "energy-budgeted", alternative, baseline, 10_000, assistance_loss_upper_uj=0))
        # Lost assistance is charged to the start-now side.
        self.assertTrue(protected_work_budget_allows(
            "energy-budgeted", alternative, baseline, 10_000, assistance_loss_upper_uj=70))
        self.assertFalse(protected_work_budget_allows(
            "energy-budgeted", alternative, baseline, 10_000, assistance_loss_upper_uj=80))
        # Unknown loss stays unknown: never admitted, under either policy.
        self.assertFalse(protected_work_budget_allows(
            "energy-budgeted", alternative, baseline, 10_000, assistance_loss_upper_uj=None))
        self.assertFalse(protected_work_budget_allows(
            "strict", alternative, baseline, 10_000, assistance_loss_upper_uj=0))
        # Records without incremental keys fall back to the route energies.
        legacy_baseline = SimpleNamespace(marginal_system_cost=None,
                                          cost=SimpleNamespace(fleet_energy_lower_uj=1000))
        legacy = SimpleNamespace(marginal_system_cost={
            "route_energy_upper_uj": 600, "upper_uj": 200, "measured": True})
        self.assertTrue(protected_work_budget_allows(
            "energy-budgeted", legacy, legacy_baseline, 100_000, assistance_loss_upper_uj=0))
        self.assertFalse(protected_work_budget_allows(
            "energy-budgeted", legacy, legacy_baseline, 100_000, assistance_loss_upper_uj=100))

    def test_lost_assistance_upper_bound_from_active_sessions(self):
        from research_dev.scheduler._unified.automated_selection import AutomatedSelectionMixin
        summaries = (
            {"request_id": "self", "assisted": True, "loss_upper_per_token_uj": 999,
             "remaining_tokens": 10, "assisted_latency_per_token_us": 1_000},
            {"request_id": "idle-peer", "assisted": False, "loss_upper_per_token_uj": None,
             "remaining_tokens": 50, "assisted_latency_per_token_us": None},
            {"request_id": "gemma", "assisted": True, "loss_upper_per_token_uj": 24_000_000,
             "remaining_tokens": 180, "assisted_latency_per_token_us": 420_000},
        )
        host = SimpleNamespace(_adaptive_decode=SimpleNamespace(assistance_summaries=lambda: summaries))
        row = SimpleNamespace(cost=SimpleNamespace(service_upper_us=25_319_638))
        request = SimpleNamespace(request_id="self")
        loss = AutomatedSelectionMixin._protected_assistance_loss_upper_uj(host, request, row)
        # ceil(25.32 s / 0.42 s) = 61 overlapping tokens of a 180-token remainder.
        self.assertEqual(loss, 24_000_000 * 61)
        # A remainder shorter than the overlap is charged in full.
        short = ({**summaries[2], "remaining_tokens": 30},)
        self.assertEqual(AutomatedSelectionMixin._protected_assistance_loss_upper_uj(
            host, request, row, short), 24_000_000 * 30)
        # Assisted work without operational bounds makes the loss unknown.
        unknown = ({**summaries[2], "loss_upper_per_token_uj": None},)
        self.assertIsNone(AutomatedSelectionMixin._protected_assistance_loss_upper_uj(
            host, request, row, unknown))
        # No adaptive controller or no assisted peers: nothing to lose.
        self.assertEqual(AutomatedSelectionMixin._protected_assistance_loss_upper_uj(
            SimpleNamespace(), request, row), 0)
        self.assertEqual(AutomatedSelectionMixin._protected_assistance_loss_upper_uj(
            host, request, row, (summaries[0], summaries[1])), 0)


if __name__ == "__main__":
    unittest.main()
