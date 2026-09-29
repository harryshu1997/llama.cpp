#!/usr/bin/env python3

from __future__ import annotations

import threading
import time
import unittest
from dataclasses import replace
from unittest import mock

import research_dev.scheduler as public
from research_dev.scheduler import (
    DeviceMemoryCapacity,
    ProfileBundle,
    Request,
    RuntimeExecutorBinding,
    RuntimeMemoryDemand,
    RuntimeModelArtifact,
    RuntimePlacementSnapshot,
    UnifiedScheduler,
)


MODEL_HASH = "sha256:" + "c" * 64
MODEL_BYTES = 400_000_000
WORKLOAD = "runtime-controller-task"


def profile(*, include_alternate_phone: bool = False) -> ProfileBundle:
    def route(
        route_id: str,
        resource_slots: dict[str, int],
        *,
        baseline: bool,
        latency: int,
        energy: int,
    ) -> dict[str, object]:
        return {
            "baseline": baseline,
            "energy": {
                "boundary_id": "cpu+gpu+phone",
                "cost_uj": {
                    "fixed": 0,
                    "input_token": energy,
                    "kind": "affine_tokens_v1",
                    "output_token": energy,
                },
                "lower_error_ppm": 0,
                "status": "measured",
                "upper_error_ppm": 0,
            },
            "evidence_ids": [route_id + "-qualified"],
            "granularity": "task",
            "latency": {
                "cost_us": {
                    "fixed": 0,
                    "input_token": latency,
                    "kind": "affine_tokens_v1",
                    "output_token": latency,
                },
                "measured": True,
                "sample_count": 4,
                "ucb_add_us": 100,
            },
            "overlap": {"status": "not_applicable"},
            "placement_verified": True,
            "quality_class": "bounded_numeric",
            "resident": True,
            "resource_slots": resource_slots,
            "route_id": route_id,
            "server_busy_ppm": 1_000_000,
            "server_memory_bytes": MODEL_BYTES,
            "workload_id": WORKLOAD,
        }

    routes = [
        route(
            "desktop-cpu",
            {"desktop-cpu": 1},
            baseline=True,
            latency=100,
            energy=1_000,
        ),
        route(
            "phone-full",
            {"phone-compute": 1, "phone-link": 1},
            baseline=False,
            latency=20,
            energy=100,
        ),
    ]
    if include_alternate_phone:
        routes.append(route(
            "phone-full-alternate",
            {
                "desktop-cpu": 1,
                "phone-compute": 1,
                "phone-link": 1,
            },
            baseline=False,
            latency=30,
            energy=150,
        ))
    return ProfileBundle.from_json({
        "policy": {
            "energy_saving_ppm": 50_000,
            "latency_limit_ppm": 20_000_000,
        },
        "profile_id": "runtime-controller-profile",
        "resources": [
            {
                "capacity": 2,
                "identity": "cpu",
                "kind": "cpu",
                "ready": True,
                "resource_id": "desktop-cpu",
            },
            {
                "capacity": 1,
                "identity": "phone",
                "kind": "phone-gpu",
                "ready": True,
                "resource_id": "phone-compute",
            },
            {
                "capacity": 1,
                "identity": "link",
                "kind": "transport",
                "ready": True,
                "resource_id": "phone-link",
            },
        ],
        "routes": routes,
        "schema": "s42-general-scheduler-profile-v1",
        "trace_workload_map": {"test-model": WORKLOAD},
    })


def model() -> RuntimeModelArtifact:
    return RuntimeModelArtifact("test-model", MODEL_HASH, MODEL_BYTES)


def request(request_id: str, arrival_us: int = 1_000) -> Request:
    return Request(
        request_id=request_id,
        workload_id=WORKLOAD,
        arrival_us=arrival_us,
        deadline_us=arrival_us + 1_000_000,
        input_tokens=100,
        output_tokens=20,
        quality_requirement="bounded_numeric",
    )


def snapshot(now_us: int) -> RuntimePlacementSnapshot:
    return RuntimePlacementSnapshot(
        snapshot_id=f"snapshot-{now_us}",
        captured_at_us=max(0, now_us - 100),
        valid_until_us=now_us + 1_000_000,
        capacities={
            "host-ram": DeviceMemoryCapacity(
                "host-ram", 32_000_000_000, 4_000_000_000, 2_000_000_000
            ),
            "phone-ram": DeviceMemoryCapacity(
                "phone-ram", 12_000_000_000, 4_000_000_000, 2_000_000_000
            ),
        },
    )


def cpu_binding(*, ready: bool = True) -> RuntimeExecutorBinding:
    return RuntimeExecutorBinding(
        executor_id="http://127.0.0.1:18080",
        route_id="desktop-cpu",
        model_id="test-model",
        artifact_sha256=MODEL_HASH,
        artifact_bytes=MODEL_BYTES,
        backend="cpu",
        resource_ids=("desktop-cpu",),
        memory_resource_id="host-ram",
        resident=True,
        ready=ready,
        route_family="cpu",
    )


def phone_binding(
    *,
    ready: bool = True,
    queueable: bool = False,
    memory_demands: tuple[RuntimeMemoryDemand, ...] = (),
) -> RuntimeExecutorBinding:
    return RuntimeExecutorBinding(
        executor_id="http://phone.test:19090",
        route_id="phone-full",
        model_id="test-model",
        artifact_sha256=MODEL_HASH,
        artifact_bytes=MODEL_BYTES,
        backend="phone-gpu",
        resource_ids=("phone-compute", "phone-link"),
        memory_resource_id=(None if memory_demands else "phone-ram"),
        resident=True,
        ready=ready,
        queueable=queueable,
        memory_demands=memory_demands,
        route_family="phone",
    )


def bindings(**phone_options: object) -> tuple[RuntimeExecutorBinding, ...]:
    return (cpu_binding(), phone_binding(**phone_options))


def wait_acquired(scheduler: UnifiedScheduler, request_id: str):
    ticket = scheduler.runtime_ticket(request_id)
    epoch_ns = time.monotonic_ns() - (ticket.decision.start_us + 10_000) * 1000
    return scheduler.wait_runtime_request(request_id, epoch_ns)


class RuntimeControllerTests(unittest.TestCase):
    def test_not_ready_queueable_executor_fails_closed(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        estimates = scheduler.estimate_runtime_costs(
            request("not-ready"),
            model(),
            bindings(ready=False, queueable=True),
            snapshot=snapshot(1_000),
            now_us=1_000,
        )
        phone = next(
            row for row in estimates.estimates
            if row.route_id == "phone-full"
        )
        self.assertFalse(phone.admitted)
        self.assertEqual(phone.reason, "EXECUTOR_NOT_READY")

    def test_one_arrival_evaluates_every_binding(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        ticket = scheduler.submit_runtime_request(
            request("all-bindings"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        self.assertEqual(
            {row.route_id for row in ticket.cost_estimates.estimates},
            {"desktop-cpu", "phone-full"},
        )
        self.assertEqual(
            {
                row["route_id"]
                for row in ticket.online_placement_receipt.causal_input[
                    "bindings"
                ]
            },
            {"desktop-cpu", "phone-full"},
        )

    def test_selected_binding_is_the_selected_route_endpoint(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        ticket = scheduler.submit_runtime_request(
            request("binding"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        self.assertEqual(ticket.decision.route_id, "phone-full")
        self.assertEqual(ticket.binding.route_id, ticket.decision.route_id)
        self.assertEqual(ticket.binding.executor_id, "http://phone.test:19090")
        self.assertIs(
            ticket.binding,
            scheduler.runtime_executor_registry.binding("phone-full"),
        )

    def test_dispatch_queue_is_owned_by_unified_scheduler(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        ticket = scheduler.submit_runtime_request(
            request("queued"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        state = scheduler.runtime_controller_snapshot()
        self.assertEqual(ticket.dispatch_state, "QUEUED")
        self.assertEqual(
            state["dispatch_queue"]["queued"]["phone-full"],
            ["queued"],
        )
        acquired = wait_acquired(scheduler, "queued")
        self.assertEqual(acquired.dispatch_state, "ACQUIRED")

    def test_early_completion_replans_frontier_at_release_time(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        first_request = request("early-owner")
        second_request = request("early-successor", 1_001)
        scheduler.submit_runtime_request(
            first_request,
            model(),
            bindings(),
            snapshot=snapshot(1_000),
            observed_at_us=1_000,
        )
        second = scheduler.submit_runtime_request(
            second_request,
            model(),
            bindings(),
            snapshot=snapshot(1_001),
            observed_at_us=1_001,
        )
        active = wait_acquired(scheduler, first_request.request_id)
        completed_at_us = active.decision.start_us + 50

        completion = scheduler.complete_runtime_request(
            first_request.request_id, completed_at_us
        )
        wake = scheduler.wait_runtime_request(
            second_request.request_id, time.monotonic_ns()
        )

        self.assertEqual(
            completion.completion_event_replans,
            (second_request.request_id,),
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
        self.assertEqual(
            wake.dispatch_receipt.wake_reason,
            "capacity_released_early",
        )
        self.assertEqual(wake.lease_status, "CANCELLED")
        self.assertEqual(
            scheduler.cancel(second_request.request_id, completed_at_us),
            (),
        )
        replanned = scheduler.replan_runtime_request(
            second_request.request_id,
            second_request,
            model(),
            bindings(),
            snapshot=snapshot(completed_at_us),
            observed_at_us=completed_at_us,
            reason="capacity_released_early",
        )
        self.assertEqual(replanned.binding.executor_id, second.binding.executor_id)
        self.assertEqual(replanned.decision.start_us, completed_at_us)

    def test_priority_compaction_promotes_resident_memory_follower(
        self,
    ) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        root = scheduler.submit_runtime_request(
            request("resident-compaction-root"),
            model(),
            bindings(),
            snapshot=snapshot(1_000),
            observed_at_us=1_000,
        )
        follower = scheduler.submit_runtime_request(
            request("resident-compaction-follower", 1_001),
            model(),
            bindings(),
            snapshot=snapshot(1_001),
            observed_at_us=1_001,
        )
        self.assertEqual(
            follower.memory_reservation_status,
            "NOT_REQUIRED_RESIDENT",
        )
        self.assertEqual(follower.memory_reservations, ())
        with mock.patch.object(
            scheduler._runtime_memory,
            "release_owner",
            wraps=scheduler._runtime_memory.release_owner,
        ) as release_owner, mock.patch.object(
            scheduler._runtime_controller.queue,
            "priority_compaction_followers",
            return_value=(follower.request.request_id,),
        ):
            identities = (
                scheduler._runtime_controller
                .prepare_priority_compaction_followers(
                    root.request.request_id,
                    1_050,
                    scheduler.cancel,
                    release_owner,
                    scheduler._runtime_memory.owner_tokens,
                )
            )
            promoted = (
                scheduler._runtime_controller
                .promote_priority_compaction_follower(
                    follower.request.request_id,
                    follower.ticket_id,
                    1_050,
                    scheduler.cancel,
                    release_owner,
                    scheduler._runtime_memory.owner_tokens,
                )
            )

        self.assertEqual(
            identities,
            ((follower.request.request_id, follower.ticket_id),),
        )
        self.assertEqual(promoted.dispatch_state, "REPLAN_REQUIRED")
        self.assertEqual(
            promoted.memory_reservation_status,
            "NOT_REQUIRED_RESIDENT",
        )
        self.assertEqual(
            scheduler._runtime_memory.owner_tokens(
                follower.request.request_id
            ),
            (),
        )
        release_owner.assert_not_called()

    def test_early_completion_defers_downstream_projection_chain(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        requests = tuple(
            request("early-chain-" + str(index), 1_000 + index)
            for index in range(3)
        )
        tickets = tuple(
            scheduler.submit_runtime_request(
                row,
                model(),
                bindings(),
                snapshot=snapshot(row.arrival_us),
                observed_at_us=row.arrival_us,
            )
            for row in requests
        )
        active = wait_acquired(scheduler, requests[0].request_id)
        completed_at_us = active.decision.start_us + 50

        completion = scheduler.complete_runtime_request(
            requests[0].request_id, completed_at_us
        )

        self.assertEqual(
            completion.completion_event_replans,
            (requests[1].request_id,),
        )
        queue = scheduler.runtime_controller_snapshot()["dispatch_queue"]
        self.assertEqual(
            queue["entry_states"][requests[1].request_id]["state"],
            "REPLAN_REQUIRED",
        )
        self.assertEqual(
            queue["entry_states"][requests[2].request_id]["state"],
            "DEFERRED_REPLAN",
        )
        self.assertEqual(
            scheduler.runtime_ticket(requests[2].request_id).lease_status,
            "CANCELLED",
        )
        self.assertEqual(tickets[2].lease_status, "RESERVED")

    def test_physical_failure_creates_a_new_baseline_decision(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        first = scheduler.submit_runtime_request(
            request("failure"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        first = wait_acquired(scheduler, "failure")
        failed_at_us = first.decision.start_us + 1
        recovery = scheduler.fail_runtime_request(
            "failure",
            failed_at_us=failed_at_us,
            reason="transport_before_connect",
            request=request("failure"),
            model=model(),
            bindings=bindings(),
            snapshot=snapshot(failed_at_us),
        )
        self.assertEqual(recovery.failed_ticket_id, first.ticket_id)
        self.assertNotEqual(recovery.fallback.ticket_id, first.ticket_id)
        self.assertEqual(recovery.fallback.previous_ticket_id, first.ticket_id)
        self.assertEqual(recovery.fallback.decision.route_id, "desktop-cpu")
        self.assertEqual(recovery.fallback.binding.route_id, "desktop-cpu")

    def test_failed_shared_resource_excludes_every_dependent_route(self) -> None:
        shared_phone = replace(
            phone_binding(),
            executor_id="http://phone.test:19091",
            route_id="phone-full-alternate",
            resource_ids=(
                "desktop-cpu", "phone-compute", "phone-link"
            ),
            route_family="cpu-phone",
        )
        all_bindings = bindings() + (shared_phone,)
        scheduler = UnifiedScheduler(
            (profile(include_alternate_phone=True),), "enforce"
        )
        original = request("shared-resource-failure")
        scheduler.submit_runtime_request(
            original,
            model(),
            all_bindings,
            snapshot=snapshot(1_000),
            observed_at_us=1_000,
        )
        active = wait_acquired(scheduler, original.request_id)
        failed_at_us = active.decision.start_us + 1

        recovery = scheduler.fail_runtime_request(
            original.request_id,
            failed_at_us=failed_at_us,
            reason="shared_transport_failed",
            request=original,
            model=model(),
            bindings=all_bindings,
            snapshot=snapshot(failed_at_us),
            physical_failure=public.RuntimeExecutionFailure(
                "transport", True, False, ("phone-link",)
            ),
        )

        self.assertEqual(recovery.fallback.decision.route_id, "desktop-cpu")
        self.assertEqual(
            scheduler.runtime_controller_snapshot()[
                "quarantined_resources"
            ],
            ["phone-link"],
        )
        reasons = dict(recovery.fallback.decision.rejected)
        self.assertEqual(
            reasons["phone-full-alternate"], "EXECUTOR_NOT_READY"
        )
        self.assertIn(
            "RESOURCE_QUARANTINED",
            scheduler.runtime_executor_registry.binding(
                "phone-full-alternate"
            ).eligibility_reasons,
        )

    def test_quarantine_changes_later_scheduler_decisions(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        scheduler.submit_runtime_request(
            request("first"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        first = wait_acquired(scheduler, "first")
        failed_at_us = first.decision.start_us + 1
        recovery = scheduler.fail_runtime_request(
            "first",
            failed_at_us=failed_at_us,
            reason="transport_before_connect",
            request=request("first"),
            model=model(),
            bindings=bindings(),
            snapshot=snapshot(failed_at_us),
        )
        fallback = wait_acquired(scheduler, "first")
        scheduler.complete_runtime_request(
            "first", fallback.decision.finish_us
        )
        later_us = fallback.decision.finish_us + 1
        later = scheduler.submit_runtime_request(
            request("later", later_us), model(), bindings(),
            snapshot=snapshot(later_us), observed_at_us=later_us,
        )
        phone = next(
            row for row in later.cost_estimates.estimates
            if row.route_id == "phone-full"
        )
        self.assertEqual(recovery.quarantine_action, "disabled_for_remaining_run")
        self.assertEqual(phone.reason, "EXECUTOR_NOT_READY")
        self.assertEqual(later.decision.route_id, "desktop-cpu")

    def test_prediction_and_final_lease_coverage_are_separate(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        scheduler.submit_runtime_request(
            request("late"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        ticket = wait_acquired(scheduler, "late")
        actual_end_us = ticket.decision.finish_upper_us + 1_000
        scheduler.extend_runtime_request(
            "late",
            at_us=ticket.decision.start_us,
            reserved_until_us=actual_end_us + 1_000,
        )
        completion = scheduler.complete_runtime_request(
            "late", actual_end_us
        )
        self.assertTrue(completion.lease_coverage.covered)
        self.assertFalse(completion.latency_upper_bound.met)
        self.assertEqual(completion.latency_upper_bound.overrun_us, 1_000)
        self.assertTrue(scheduler.runtime_route_is_quarantined("phone-full"))
        self.assertFalse(
            scheduler.runtime_executor_registry.binding("phone-full").ready
        )

    def test_lease_extension_cancels_replanning_follower(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        scheduler.submit_runtime_request(
            request("renewal-owner"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        active = wait_acquired(scheduler, "renewal-owner")
        follower = scheduler.submit_runtime_request(
            request("renewal-follower", 1_001), model(), bindings(),
            snapshot=snapshot(1_001), observed_at_us=1_001,
        )
        self.assertTrue(
            scheduler._runtime_controller.require_queued_replan(
                follower.request.request_id,
                "residency_projection_invalid",
            )
        )
        wake = scheduler.wait_runtime_request(
            follower.request.request_id, time.monotonic_ns()
        )
        scheduler._runtime_controller.queue.retire_replan(
            follower.request.request_id,
            wake.dispatch_receipt.queue_generation,
        )

        receipt = scheduler.extend_runtime_request(
            active.request.request_id,
            at_us=active.decision.start_us,
            reserved_until_us=follower.decision.start_us + 1,
        )

        self.assertIn(
            follower.request.request_id,
            receipt.cancelled_queued_tokens,
        )
        current = scheduler.runtime_ticket(follower.request.request_id)
        self.assertEqual(current.lease_status, "CANCELLED")
        self.assertEqual(
            current.memory_reservation_status,
            follower.memory_reservation_status,
        )
        self.assertEqual(
            scheduler.runtime_controller_snapshot()["dispatch_queue"]
                ["entry_states"][follower.request.request_id]["state"],
            "DEFERRED_REPLAN",
        )

    def test_cancellation_and_completion_release_owned_resources(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        cancelled = scheduler.submit_runtime_request(
            request("cancelled"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        tokens = scheduler.cancel_runtime_request(
            "cancelled", cancelled.decision.start_us, "caller_cancelled"
        )
        self.assertEqual(set(tokens), {
            lease.token for lease in cancelled.decision.leases
        })
        self.assertEqual(
            scheduler.runtime_controller_snapshot()["dispatch_queue"]["queued"],
            {},
        )

        next_us = cancelled.decision.start_us + 1
        scheduler.submit_runtime_request(
            request("completed", next_us), model(), bindings(),
            snapshot=snapshot(next_us), observed_at_us=next_us,
        )
        active = wait_acquired(scheduler, "completed")
        completion = scheduler.complete_runtime_request(
            "completed", active.decision.finish_us
        )
        self.assertEqual(completion.status, "released")
        state = scheduler.runtime_controller_snapshot()["dispatch_queue"]
        self.assertEqual(state["active"], {})
        self.assertEqual(state["queued"], {})

    def test_additional_multi_resource_memory_fails_closed(self) -> None:
        demands = (
            RuntimeMemoryDemand(
                "phone-weights", "phone-ram", "model_weights",
                MODEL_BYTES, MODEL_BYTES, "resident",
            ),
            RuntimeMemoryDemand(
                "host-kv", "host-ram", "kv_cache",
                10_000_000, 0, "request",
            ),
            RuntimeMemoryDemand(
                "phone-workspace", "phone-ram", "workspace",
                20_000_000, 0, "request",
            ),
        )
        scheduler = UnifiedScheduler((profile(),), "enforce")
        ticket = scheduler.submit_runtime_request(
            request("memory"),
            model(),
            bindings(memory_demands=demands),
            snapshot=snapshot(1_000),
            observed_at_us=1_000,
        )
        phone = next(
            row for row in ticket.cost_estimates.estimates
            if row.route_id == "phone-full"
        )
        self.assertFalse(phone.admitted)
        self.assertEqual(phone.reason, "MEMORY_RESERVATION_UNAVAILABLE")
        self.assertEqual(
            phone.memory_demands,
            tuple(sorted(demands, key=lambda row: row.demand_id)),
        )
        self.assertEqual(
            dict(phone.additional_bytes_by_resource),
            {"host-ram": 10_000_000, "phone-ram": 20_000_000},
        )
        self.assertEqual(ticket.decision.route_id, "desktop-cpu")

    def test_existing_and_new_public_symbols_have_single_identity(self) -> None:
        from research_dev.scheduler._internal import decision_log
        from research_dev.scheduler._internal import runtime_admission
        from research_dev.scheduler._internal import runtime_controller
        from research_dev.scheduler._internal import runtime_cost
        from research_dev.scheduler._internal import runtime_execution
        from research_dev.scheduler._internal import runtime_phase
        from research_dev.scheduler._internal import runtime_queue

        self.assertIs(
            public.RuntimeExecutorBinding,
            runtime_cost.RuntimeExecutorBinding,
        )
        self.assertIs(
            public.RuntimeExecutorRegistry,
            runtime_cost.RuntimeExecutorRegistry,
        )
        self.assertIs(
            public.RuntimeDispatchQueue,
            runtime_queue.RuntimeDispatchQueue,
        )
        self.assertIs(
            public.RuntimeRequestTicket,
            runtime_controller.RuntimeRequestTicket,
        )
        self.assertIs(
            public.RuntimeExecutorObservation,
            runtime_admission.RuntimeExecutorObservation,
        )
        self.assertIs(
            public.RuntimeRequestObservation,
            runtime_admission.RuntimeRequestObservation,
        )
        self.assertIs(
            public.RuntimeExecutionFailure,
            runtime_execution.RuntimeExecutionFailure,
        )
        self.assertIs(
            public.RuntimePhaseObservation,
            runtime_phase.RuntimePhaseObservation,
        )
        self.assertIs(public.DecisionLogError, decision_log.DecisionLogError)

    def test_failed_composite_renewal_is_a_no_op(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        scheduler.submit_runtime_request(
            request("atomic-renewal"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        active = wait_acquired(scheduler, "atomic-renewal")
        old_end = min(active.final_reserved_until_us.values())
        requested_end = old_end + 10_000
        scheduler.reserve_external_resource(
            "phone-link",
            "renewal-conflict",
            old_end,
            requested_end + 10_000,
        )
        calendar_before = scheduler.timeline.causal_state()
        ticket_before = scheduler.runtime_ticket("atomic-renewal").to_json()
        log_before = scheduler.runtime_decision_log_bytes()
        with self.assertRaises(public.UnifiedScheduleError):
            scheduler.extend_runtime_request(
                "atomic-renewal",
                at_us=active.decision.start_us,
                reserved_until_us=requested_end,
            )
        self.assertEqual(scheduler.timeline.causal_state(), calendar_before)
        self.assertEqual(
            scheduler.runtime_ticket("atomic-renewal").to_json(),
            ticket_before,
        )
        self.assertEqual(scheduler.runtime_decision_log_bytes(), log_before)

    def test_renewal_failure_reaches_execution_control(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        scheduler.submit_runtime_request(
            request("renewal-control"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        active = wait_acquired(scheduler, "renewal-control")
        old_end = min(active.final_reserved_until_us.values())
        scheduler.reserve_external_resource(
            "phone-link",
            "coordinator-conflict",
            old_end,
            old_end + 1_000_000,
        )
        epoch_ns = time.monotonic_ns() - old_end * 1_000
        scheduler.start_runtime_lease_renewal(
            active,
            epoch_ns=epoch_ns,
            guard_us=1,
            quantum_us=100_000,
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                scheduler.check_runtime_lease_renewal(
                    active.request.request_id
                )
            except public.RuntimeExecutionCoordinatorError:
                break
            time.sleep(0.005)
        else:
            self.fail("renewal failure did not reach execution control")
        with self.assertRaises(public.RuntimeExecutionCoordinatorError):
            scheduler.stop_runtime_lease_renewal(active.request.request_id)

    def test_overdue_active_lease_renews_for_one_quantum(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        scheduler.submit_runtime_request(
            request("overdue-renewal"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        active = wait_acquired(scheduler, "overdue-renewal")
        old_end = min(active.final_reserved_until_us.values())
        epoch_ns = time.monotonic_ns() - (old_end + 1_000) * 1_000
        receipts = []
        scheduler.start_runtime_lease_renewal(
            active,
            epoch_ns=epoch_ns,
            guard_us=1,
            quantum_us=100,
            on_renewal=receipts.append,
        )
        deadline = time.monotonic() + 2
        while not receipts and time.monotonic() < deadline:
            time.sleep(0.005)
        scheduler.stop_runtime_lease_renewal(active.request.request_id)
        self.assertTrue(receipts)
        first = receipts[0]
        self.assertEqual(
            min(
                row["reserved_until_us"]
                for row in first.extended_leases
            ),
            max(old_end + 100, first.at_us + 100),
        )

    def test_completed_preparation_leases_do_not_drive_renewal_wakeups(self) -> None:
        """Physical runs v7-v12: the coordinator took its horizon over every
        lease, so a completed preparation phase whose predicted end had
        passed made the wait zero and leases were renewed ~2,600 times per
        second for the rest of the request."""
        from types import SimpleNamespace
        from research_dev.scheduler._internal.runtime_execution import (
            RuntimeLeaseRenewalCoordinator,
        )

        scheduler = UnifiedScheduler((profile(),), "enforce")
        scheduler.submit_runtime_request(
            request("dead-prepare-lease"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        active = wait_acquired(scheduler, "dead-prepare-lease")
        epoch_ns = time.monotonic_ns() - active.decision.start_us * 1_000
        now_us = active.decision.start_us
        horizon = {"live": now_us + 60_000_000, "dead": now_us - 5_000_000}
        extends = []

        def current_ticket(_request_id):
            return SimpleNamespace(
                ticket_id=active.ticket_id,
                live_leases=(SimpleNamespace(token="live"),),
                final_reserved_until_us=dict(horizon),
            )

        def extend(_request_id, *, at_us, reserved_until_us):
            extends.append((at_us, reserved_until_us))
            horizon["live"] = max(horizon["live"], reserved_until_us)
            return SimpleNamespace(at_us=at_us, extended_leases=())

        coordinator = RuntimeLeaseRenewalCoordinator(
            active, epoch_ns=epoch_ns, guard_us=250_000, quantum_us=2_000_000,
            current_ticket=current_ticket, extend=extend, on_renewal=None,
        )
        coordinator.start()
        try:
            time.sleep(0.6)
        finally:
            coordinator.stop()
        # The dead preparation lease is ignored; the live horizon is a minute
        # away, so nothing is renewed at all.
        self.assertEqual(extends, [])
        self.assertEqual(coordinator.diagnostics(),
                         {"expired_horizons": 0, "renewals": 0, "stalled_renewals": 0})

    def _renewal_fixture(self, name):
        from types import SimpleNamespace
        scheduler = UnifiedScheduler((profile(),), "enforce")
        scheduler.submit_runtime_request(
            request(name), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        active = wait_acquired(scheduler, name)
        now0_us = 10_000_000  # coordinator clock starts at 10 s
        epoch_ns = time.monotonic_ns() - now0_us * 1_000
        return SimpleNamespace, active, epoch_ns, now0_us

    def test_stalled_helper_horizon_with_healthy_base_leases_does_not_spin(self) -> None:
        """Reproduced by review: 40,307 renewals in 0.161 s when only the
        attached-helper horizon was stale (leases held at 0%, never renewed)
        while the base leases were healthy."""
        from research_dev.scheduler._internal.runtime_execution import (
            RuntimeLeaseRenewalCoordinator,
        )
        SimpleNamespace, active, epoch_ns, now0_us = self._renewal_fixture("stale-helper-horizon")
        base = {"live": now0_us + 60_000_000}
        helper = {"horizon": now0_us - 1_000_000, "released": False}
        extends, expiries = [], []

        def current_ticket(_request_id):
            return SimpleNamespace(ticket_id=active.ticket_id,
                                   live_leases=(SimpleNamespace(token="live"),),
                                   final_reserved_until_us=dict(base))

        def extend(_request_id, *, at_us, reserved_until_us):
            extends.append(at_us); base["live"] = max(base["live"], reserved_until_us)
            return SimpleNamespace(at_us=at_us, extended_leases=())

        def current_horizon(ticket):
            return base["live"] if helper["released"] else min(base["live"], helper["horizon"])

        def on_expired(ticket, at_us):
            expiries.append(at_us); helper["released"] = True
            return True

        coordinator = RuntimeLeaseRenewalCoordinator(
            active, epoch_ns=epoch_ns, guard_us=250_000, quantum_us=2_000_000,
            current_ticket=current_ticket, extend=extend, on_renewal=None,
            current_horizon=current_horizon, on_expired=on_expired,
        )
        coordinator.start()
        try:
            time.sleep(0.6)
        finally:
            coordinator.stop()
        # One renewal, one stall, one expiry that detached the idle helper;
        # afterwards the healthy base horizon governs and nothing else fires.
        self.assertEqual(len(extends), 1)
        self.assertEqual(len(expiries), 1)
        self.assertEqual(coordinator.diagnostics(),
                         {"expired_horizons": 1, "renewals": 1, "stalled_renewals": 1})

    def test_stalled_helper_horizon_recovers_when_it_advances_again(self) -> None:
        from research_dev.scheduler._internal.runtime_execution import (
            RuntimeLeaseRenewalCoordinator,
        )
        SimpleNamespace, active, epoch_ns, now0_us = self._renewal_fixture("recovering-helper-horizon")
        base = {"live": now0_us + 60_000_000}
        helper = {"horizon": now0_us + 300_000}
        extends, expiries = [], []

        def current_ticket(_request_id):
            return SimpleNamespace(ticket_id=active.ticket_id,
                                   live_leases=(SimpleNamespace(token="live"),),
                                   final_reserved_until_us=dict(base))

        def extend(_request_id, *, at_us, reserved_until_us):
            extends.append(at_us); base["live"] = max(base["live"], reserved_until_us)
            if len(extends) >= 2:
                # The helper leases renew again from the second attempt on.
                helper["horizon"] = at_us + 1_800_000
            return SimpleNamespace(at_us=at_us, extended_leases=())

        coordinator = RuntimeLeaseRenewalCoordinator(
            active, epoch_ns=epoch_ns, guard_us=100_000, quantum_us=2_000_000,
            current_ticket=current_ticket, extend=extend, on_renewal=None,
            current_horizon=lambda ticket: min(base["live"], helper["horizon"]),
            on_expired=lambda ticket, at_us: expiries.append(at_us) or True,
        )
        coordinator.start()
        try:
            time.sleep(0.8)
        finally:
            coordinator.stop()
        # First renewal at T-100 ms stalled (helper horizon unchanged) and was
        # retried once the guard had elapsed, still inside the authorization;
        # the retry advanced the horizon, so nothing expired and no more
        # renewals were needed for the rest of the test.
        self.assertEqual(len(extends), 2)
        self.assertEqual(expiries, [])
        self.assertGreaterEqual(extends[1] - extends[0], 90_000)
        self.assertEqual(coordinator.diagnostics(),
                         {"expired_horizons": 0, "renewals": 2, "stalled_renewals": 1})

    def test_expired_helper_horizon_without_release_surfaces_failure(self) -> None:
        from research_dev.scheduler._internal.runtime_execution import (
            RuntimeExecutionCoordinatorError, RuntimeLeaseRenewalCoordinator,
        )
        SimpleNamespace, active, epoch_ns, now0_us = self._renewal_fixture("expired-no-release")
        base = {"live": now0_us + 60_000_000}
        extends = []

        def current_ticket(_request_id):
            return SimpleNamespace(ticket_id=active.ticket_id,
                                   live_leases=(SimpleNamespace(token="live"),),
                                   final_reserved_until_us=dict(base))

        def extend(_request_id, *, at_us, reserved_until_us):
            extends.append(at_us); base["live"] = max(base["live"], reserved_until_us)
            return SimpleNamespace(at_us=at_us, extended_leases=())

        # No expiry handler: an expired, unrenewable helper horizon is a failure.
        coordinator = RuntimeLeaseRenewalCoordinator(
            active, epoch_ns=epoch_ns, guard_us=100_000, quantum_us=2_000_000,
            current_ticket=current_ticket, extend=extend, on_renewal=None,
            current_horizon=lambda ticket: now0_us - 5_000_000,
        )
        coordinator.start()
        time.sleep(0.3)
        with self.assertRaisesRegex(RuntimeExecutionCoordinatorError, "renewal failed"):
            coordinator.stop()
        self.assertEqual(len(extends), 1)
        self.assertEqual(coordinator.diagnostics()["expired_horizons"], 1)
        # A handler that cannot release anything is also a failure, and the
        # handler is never hammered: at most one call per guard interval.
        SimpleNamespace, active2, epoch2, now2_us = self._renewal_fixture("expired-unhandled")
        calls = []
        coordinator = RuntimeLeaseRenewalCoordinator(
            active2, epoch_ns=epoch2, guard_us=100_000, quantum_us=2_000_000,
            current_ticket=lambda _r: SimpleNamespace(
                ticket_id=active2.ticket_id, live_leases=(SimpleNamespace(token="live"),),
                final_reserved_until_us={"live": now2_us + 60_000_000}),
            extend=lambda _r, *, at_us, reserved_until_us: SimpleNamespace(at_us=at_us, extended_leases=()),
            on_renewal=None, current_horizon=lambda ticket: now2_us - 5_000_000,
            on_expired=lambda ticket, at_us: calls.append(at_us) or False,
        )
        coordinator.start()
        time.sleep(0.3)
        with self.assertRaisesRegex(RuntimeExecutionCoordinatorError, "renewal failed"):
            coordinator.stop()
        self.assertEqual(len(calls), 1)

    def test_late_helper_attachment_renews_once_at_its_horizon(self) -> None:
        from research_dev.scheduler._internal.runtime_execution import (
            RuntimeLeaseRenewalCoordinator,
        )
        SimpleNamespace, active, epoch_ns, now0_us = self._renewal_fixture("late-attachment")
        base = {"live": now0_us + 60_000_000}
        helper = {"horizon": None}
        extends = []

        def now_us():
            return max(0, (time.monotonic_ns() - epoch_ns) // 1000)

        def current_ticket(_request_id):
            return SimpleNamespace(ticket_id=active.ticket_id,
                                   live_leases=(SimpleNamespace(token="live"),),
                                   final_reserved_until_us=dict(base))

        def current_horizon(ticket):
            return base["live"] if helper["horizon"] is None else min(base["live"], helper["horizon"])

        def extend(_request_id, *, at_us, reserved_until_us):
            extends.append(at_us); base["live"] = max(base["live"], reserved_until_us)
            # The scheduler renews the helper leases with the base extension.
            helper["horizon"] = at_us + 1_800_000
            return SimpleNamespace(at_us=at_us, extended_leases=())

        coordinator = RuntimeLeaseRenewalCoordinator(
            active, epoch_ns=epoch_ns, guard_us=100_000, quantum_us=2_000_000,
            current_ticket=current_ticket, extend=extend, on_renewal=None,
            current_horizon=current_horizon,
            on_expired=lambda ticket, at_us: False,
        )
        coordinator.start()
        try:
            time.sleep(0.15)
            self.assertEqual(extends, [])
            helper["horizon"] = now_us() + 150_000   # attached late, 150 ms of authorization
            coordinator.wake()
            time.sleep(0.4)
        finally:
            coordinator.stop()
        # Renewed once, about 50 ms before the late horizon (guard 100 ms),
        # then the renewed 1.8 s helper horizon governs and nothing spins.
        self.assertEqual(len(extends), 1)
        self.assertEqual(coordinator.diagnostics(),
                         {"expired_horizons": 0, "renewals": 1, "stalled_renewals": 0})

    def test_expire_request_helper_authorization_releases_or_detaches(self) -> None:
        from types import SimpleNamespace
        from research_dev.scheduler._unified.runtime_requests import RuntimeRequestMixin
        calls = []
        def host(binding):
            placement = SimpleNamespace(
                request_binding=lambda request_id: binding,
                record_request_helper_event=lambda request_id, kind, at_us, payload: calls.append((kind, dict(payload))),
            )
            return SimpleNamespace(
                _runtime_lock=threading.RLock(),
                _model_placement_controller=placement,
                _release_request_helper_leases=lambda request_id, at_us: calls.append(("release", request_id, at_us)),
                _adaptive_decode=SimpleNamespace(helper_unavailable=lambda request_id: calls.append(("unavailable", request_id))),
            )
        ticket = SimpleNamespace(request=SimpleNamespace(request_id="gemma-36"))
        attached = {"helper_attachment": {"lease_tokens": ["lease-a"], "lease_reserved_until_us": 70_000_000}}
        # Idle helper leases are released.
        self.assertTrue(RuntimeRequestMixin._expire_request_helper_authorization(
            host({**attached, "fraction_ppm": 0}), ticket, 73_000_000))
        self.assertEqual(calls[0], ("release", "gemma-36", 73_000_000))
        self.assertEqual(calls[1][0], "HELPER_AUTHORIZATION_EXPIRED")
        self.assertEqual(calls[1][1]["reason"], "IDLE_HELPER_LEASES_RELEASED")
        calls.clear()
        # Active assistance is told to detach at its next boundary.
        self.assertTrue(RuntimeRequestMixin._expire_request_helper_authorization(
            host({**attached, "fraction_ppm": 1_000_000}), ticket, 73_000_000))
        self.assertEqual(calls[0], ("unavailable", "gemma-36"))
        self.assertEqual(calls[1][1]["reason"], "ASSISTANCE_DETACHING_AT_NEXT_BOUNDARY")
        self.assertEqual(calls[1][1]["fraction_ppm"], 1_000_000)
        calls.clear()
        # Nothing attached: nothing to detach, the coordinator surfaces it.
        self.assertFalse(RuntimeRequestMixin._expire_request_helper_authorization(host(None), ticket, 1))
        self.assertFalse(RuntimeRequestMixin._expire_request_helper_authorization(
            host({"helper_attachment": {"lease_tokens": []}, "fraction_ppm": 0}), ticket, 1))
        self.assertEqual(calls, [])

    def test_stalled_live_horizon_is_paced_at_the_guard_not_spun(self) -> None:
        from types import SimpleNamespace
        from research_dev.scheduler._internal.runtime_execution import (
            RuntimeExecutionCoordinatorError, RuntimeLeaseRenewalCoordinator,
        )

        scheduler = UnifiedScheduler((profile(),), "enforce")
        scheduler.submit_runtime_request(
            request("stalled-horizon"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        active = wait_acquired(scheduler, "stalled-horizon")
        now0_us = 10_000_000
        epoch_ns = time.monotonic_ns() - now0_us * 1_000
        stuck_end = now0_us + 300_000  # valid for 300 ms, never moves
        extends = []

        def current_ticket(_request_id):
            # An extension that never moves the live horizon (refused retime).
            return SimpleNamespace(
                ticket_id=active.ticket_id,
                live_leases=(SimpleNamespace(token="stuck"),),
                final_reserved_until_us={"stuck": stuck_end},
            )

        def extend(_request_id, *, at_us, reserved_until_us):
            extends.append(at_us)
            return SimpleNamespace(at_us=at_us, extended_leases=())

        coordinator = RuntimeLeaseRenewalCoordinator(
            active, epoch_ns=epoch_ns, guard_us=100_000, quantum_us=2_000_000,
            current_ticket=current_ticket, extend=extend, on_renewal=None,
        )
        coordinator.start()
        time.sleep(0.55)
        # Sleeping never extended the authorization: once the stuck horizon
        # passed, the expiry surfaced as a failure (no detach handler here).
        with self.assertRaisesRegex(RuntimeExecutionCoordinatorError, "renewal failed"):
            coordinator.stop()
        # Renewed at T-100 ms (stall), retried at the guard while still valid,
        # then expired: a handful of attempts, never a hot loop.
        self.assertGreaterEqual(len(extends), 2)
        self.assertLessEqual(len(extends), 4)
        self.assertGreaterEqual(extends[1] - extends[0], 90_000)
        self.assertTrue(all(at_us <= stuck_end + 50_000 for at_us in extends), extends)
        diagnostics = coordinator.diagnostics()
        self.assertEqual(diagnostics["renewals"], len(extends))
        self.assertEqual(diagnostics["stalled_renewals"], len(extends))
        self.assertEqual(diagnostics["expired_horizons"], 1)

    def test_late_helper_horizon_wakes_periodic_renewal(self) -> None:
        from research_dev.scheduler._internal.runtime_execution import (
            RuntimeLeaseRenewalCoordinator,
        )

        scheduler = UnifiedScheduler((profile(),), "enforce")
        scheduler.submit_runtime_request(
            request("late-helper-renewal"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        active = wait_acquired(scheduler, "late-helper-renewal")
        scheduler.extend_runtime_request(
            active.request.request_id,
            at_us=active.decision.start_us,
            reserved_until_us=(
                min(active.final_reserved_until_us.values()) + 1_000_000
            ),
        )
        active = scheduler.runtime_ticket(active.request.request_id)
        base_end_us = min(active.final_reserved_until_us.values())
        epoch_ns = (
            time.monotonic_ns() - active.decision.start_us * 1_000
        )
        helper_horizon = [None]
        receipts = []

        def renewed(receipt) -> None:
            helper_horizon[0] = None
            receipts.append(receipt)

        def current_horizon(ticket) -> int:
            return (
                min(ticket.final_reserved_until_us.values())
                if helper_horizon[0] is None else helper_horizon[0]
            )

        coordinator = RuntimeLeaseRenewalCoordinator(
            active,
            epoch_ns=epoch_ns,
            guard_us=1_000,
            quantum_us=100_000,
            current_ticket=scheduler.runtime_ticket,
            extend=scheduler.extend_runtime_request,
            on_renewal=renewed,
            current_horizon=current_horizon,
        )
        coordinator.start()
        try:
            time.sleep(0.01)
            self.assertFalse(receipts)
            helper_horizon[0] = active.decision.start_us + 30_000
            coordinator.wake()
            deadline = time.monotonic() + 1
            while not receipts and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(receipts)
            self.assertLess(receipts[0].at_us, base_end_us)
        finally:
            coordinator.stop()

    def test_terminal_ticket_cannot_be_rewritten(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        scheduler.submit_runtime_request(
            request("terminal"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        active = wait_acquired(scheduler, "terminal")
        scheduler.complete_runtime_request(
            "terminal", active.decision.finish_us
        )
        terminal = scheduler.runtime_ticket("terminal")
        log_before = scheduler.runtime_decision_log_bytes()
        with self.assertRaises(public.UnifiedScheduleError):
            scheduler.cancel_runtime_request(
                "terminal", active.decision.finish_us, "late_cancel"
            )
        self.assertEqual(scheduler.runtime_ticket("terminal"), terminal)
        self.assertEqual(scheduler.runtime_decision_log_bytes(), log_before)

    def test_failure_rejects_changed_request_and_model_identity(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        original_request = request("identity")
        original_model = model()
        scheduler.submit_runtime_request(
            original_request, original_model, bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        active = wait_acquired(scheduler, "identity")
        changed_request = replace(
            original_request,
            deadline_us=original_request.deadline_us + 1,
        )
        changed_model = RuntimeModelArtifact(
            original_model.model_id,
            "sha256:" + "d" * 64,
            original_model.artifact_bytes,
        )
        ticket_before = scheduler.runtime_ticket("identity")
        log_before = scheduler.runtime_decision_log_bytes()
        with self.assertRaises(public.UnifiedScheduleError):
            scheduler.fail_runtime_request(
                "identity",
                failed_at_us=active.decision.start_us + 1,
                reason="synthetic_transport_failure",
                request=changed_request,
                model=original_model,
                bindings=bindings(),
                snapshot=snapshot(active.decision.start_us + 1),
            )
        with self.assertRaises(public.UnifiedScheduleError):
            scheduler.fail_runtime_request(
                "identity",
                failed_at_us=active.decision.start_us + 1,
                reason="synthetic_transport_failure",
                request=original_request,
                model=changed_model,
                bindings=bindings(),
                snapshot=snapshot(active.decision.start_us + 1),
            )
        self.assertEqual(scheduler.runtime_ticket("identity"), ticket_before)
        self.assertEqual(scheduler.runtime_decision_log_bytes(), log_before)

    def test_replan_rejects_changed_request_and_model_identity(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        scheduler.submit_runtime_request(
            request("identity-blocker"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        blocker = wait_acquired(scheduler, "identity-blocker")
        original = request("identity-replan", 1_001)
        scheduler.submit_runtime_request(
            original, model(), bindings(),
            snapshot=snapshot(1_001), observed_at_us=1_001,
        )
        queued = scheduler.runtime_ticket("identity-replan")
        scheduler.extend_runtime_request(
            "identity-blocker",
            at_us=blocker.decision.start_us,
            reserved_until_us=queued.decision.start_us + 1,
        )
        scheduler.complete_runtime_request(
            "identity-blocker", blocker.decision.finish_us
        )
        wake = scheduler.wait_runtime_request(
            "identity-replan", time.monotonic_ns()
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
        changed_request = replace(
            original, output_tokens=original.output_tokens + 1
        )
        changed_model = RuntimeModelArtifact(
            model().model_id, "sha256:" + "e" * 64, MODEL_BYTES
        )
        ticket_before = scheduler.runtime_ticket("identity-replan")
        log_before = scheduler.runtime_decision_log_bytes()
        for request_row, model_row in (
            (changed_request, model()),
            (original, changed_model),
        ):
            with self.assertRaises(public.UnifiedScheduleError):
                scheduler.replan_runtime_request(
                    "identity-replan",
                    request_row,
                    model_row,
                    bindings(),
                    snapshot=snapshot(queued.decision.start_us + 1),
                    observed_at_us=queued.decision.start_us + 1,
                    reason="identity_test",
                )
        self.assertEqual(
            scheduler.runtime_ticket("identity-replan"), ticket_before
        )
        self.assertEqual(scheduler.runtime_decision_log_bytes(), log_before)

    def test_failed_and_cancelled_tickets_are_immutable(self) -> None:
        scheduler = UnifiedScheduler((profile(),), "enforce")
        cancelled = scheduler.submit_runtime_request(
            request("terminal-cancelled"), model(), bindings(),
            snapshot=snapshot(1_000), observed_at_us=1_000,
        )
        scheduler.cancel_runtime_request(
            "terminal-cancelled",
            cancelled.decision.start_us,
            "synthetic_cancel",
        )
        cancelled_terminal = scheduler.runtime_ticket("terminal-cancelled")
        with self.assertRaises(public.UnifiedScheduleError):
            scheduler.cancel_runtime_request(
                "terminal-cancelled",
                cancelled.decision.start_us,
                "second_cancel",
            )
        self.assertEqual(
            scheduler.runtime_ticket("terminal-cancelled"),
            cancelled_terminal,
        )

        failed_request = request(
            "terminal-failed", cancelled.decision.start_us + 1
        )
        scheduler.submit_runtime_request(
            failed_request, model(), bindings(),
            snapshot=snapshot(failed_request.arrival_us),
            observed_at_us=failed_request.arrival_us,
        )
        active = wait_acquired(scheduler, "terminal-failed")
        result = scheduler.fail_runtime_request(
            "terminal-failed",
            failed_at_us=active.decision.start_us + 1,
            reason="unsafe_failure",
            request=failed_request,
            model=model(),
            bindings=bindings(),
            snapshot=snapshot(active.decision.start_us + 1),
            physical_failure=public.RuntimeExecutionFailure(
                "response", False, True
            ),
        )
        self.assertIsNone(result.fallback)
        failed_terminal = scheduler.runtime_ticket("terminal-failed")
        with self.assertRaises(public.UnifiedScheduleError):
            scheduler.cancel_runtime_request(
                "terminal-failed",
                active.decision.start_us + 1,
                "late_cancel",
            )
        self.assertEqual(
            scheduler.runtime_ticket_by_id(failed_terminal.ticket_id),
            failed_terminal,
        )


if __name__ == "__main__":
    unittest.main()
