"""Automated runtime: arrivals, queues, leases, replans, controls and completion.

Split from test_automated_runtime.py on 2026-09-13; fixtures and the base class stay there."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
import threading
import time
from unittest import mock
from research_dev.scheduler import (
    DeviceMemoryCapacity,
    ModelResidencyObservation,
    RuntimeCapabilityCatalog,
    RuntimeCompositeExecutorCapability,
    RuntimeCompositeOperatorPlacement,
    RuntimeExecutorState,
    RuntimeExecutionReceipt,
    RuntimePlacementSnapshot,
    RuntimeResourceError,
    RuntimeTransitionCapability,
    UnifiedScheduleError,
    UnifiedScheduler,
)
from research_dev.scheduler._internal.decision_log import DecisionLogError
from research_dev.scheduler._internal.runtime_controller import RuntimeReplanRetryRequired
from research_dev.scheduler._internal.route_generation import (
    RouteGenerationError,
    candidate_set_to_runtime_costs,
)
from research_dev.scheduler._internal.runtime_search import request_shape_bucket
from research_dev.scheduler._internal.runtime_residency_projection import (
    RuntimeResidencyProjectionError,
    project_scheduler_residency,
)
from research_dev.scheduler.adapters import (
    catalog_preloaded_residency_samples,
    EndpointRuntimeSample,
    ExecutorResidencySample,
    live_executor_residency_sample,
    PhysicalAdapterError,
    UnifiedRuntimeSnapshotBuilder,
)

try:
    from .test_automated_runtime import (
        AutomatedRuntimeTests,
        catalog,
        executor_state,
        protected_snapshot,
        request,
        runtime_snapshot,
        system_cost_profile,
        write_synthetic_gguf,
    )
except ImportError:  # run from the tests directory
    from test_automated_runtime import (
        AutomatedRuntimeTests,
        catalog,
        executor_state,
        protected_snapshot,
        request,
        runtime_snapshot,
        system_cost_profile,
        write_synthetic_gguf,
    )


class AutomatedRuntimeRuntimeTests(AutomatedRuntimeTests):
    """Automated runtime: arrivals, queues, leases, replans, controls and completion."""

    def test_live_residency_observation_wakes_scheduler_owned_queue(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        cold = runtime_snapshot(
            manifest,
            phone_bandwidth=8_000_000_000,
            resident_devices=(),
        )
        first = scheduler.submit_automated_request(
            request("residency-owner", arrival_us=1_000),
            manifest.model_id,
            cold,
        )
        second = scheduler.submit_automated_request(
            request("burstgpt-v2:88139", arrival_us=1_100),
            manifest.model_id,
            cold,
        )
        self.assertTrue(first.execution_plan.transitions)
        self.assertTrue(second.execution_plan.transitions)
        first_active = scheduler.wait_runtime_request(
            first.request.request_id,
            time.monotonic_ns() - first.decision.start_us * 1_000,
        )
        self.assertEqual(first_active.dispatch_state, "ACQUIRED")

        hot = replace(
            runtime_snapshot(
                manifest, phone_bandwidth=8_000_000_000
            ),
            captured_at_us=2_000,
            memory=replace(
                runtime_snapshot(manifest).memory,
                captured_at_us=2_000,
            ),
        )
        transition_executor_id = (
            first.execution_plan.transitions[0].executor_id
            or first.binding.executor_id
        )
        prepared_device_ids = set(
            first.execution_plan.transitions[0].prepares_device_ids
        )
        hot = replace(
            hot,
            residency=tuple(
                replace(row, executor_id=transition_executor_id)
                if row.device_id in prepared_device_ids
                else row
                for row in hot.residency
            ),
        )
        changed = scheduler.observe_automated_runtime_snapshot(
            hot, observed_at_us=2_000
        )
        self.assertEqual(changed, (second.request.request_id,))
        wake = scheduler.wait_runtime_request(
            second.request.request_id,
            time.monotonic_ns() - 2_000 * 1_000,
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")
        with mock.patch.object(
            scheduler._automated_compiler(),
            "generate",
            side_effect=AssertionError("full placement generation ran"),
        ):
            replanned = scheduler.replan_automated_request(
                second.request.request_id,
                observed_at_us=2_000,
                reason=wake.dispatch_receipt.wake_reason,
                snapshot=hot,
            )
        self.assertEqual(replanned.previous_ticket_id, second.ticket_id)
        self.assertFalse(replanned.execution_plan.transitions)
        timing = scheduler.runtime_decision_timings()[-1]
        self.assertEqual(timing["event_kind"], "REPLAN")
        self.assertTrue(timing["model_placement_epoch_fast_path"])
        record = scheduler.runtime_decision_log()["records"][-1]
        self.assertEqual(record["event_kind"], "REPLAN")
        self.assertEqual(
            len(record["candidates"]),
            len(replanned.cost_estimates.estimates),
        )
        self.assertIn("model_placement_epoch", record["selected"])
        resolution = record["selected"][
            "model_placement_resolution"
        ]
        self.assertEqual(resolution["pass_count"], 1)
        self.assertEqual(
            resolution["passes"][0]["proposed_route_id"],
            second.decision.route_id,
        )
        self.assertEqual(
            resolution["passes"][0]["live_selected_route_id"],
            replanned.decision.route_id,
        )
        self.assertNotEqual(
            second.decision.route_id, replanned.decision.route_id
        )
        self.assertEqual(
            resolution["passes"][0][
                "proposed_component_identity_sha256"
            ],
            resolution["passes"][0][
                "live_component_identity_sha256"
            ],
        )
        self.assertEqual(
            resolution["passes"][0]["rejection_reasons"], []
        )

    def test_new_arrival_repairs_stale_projection_in_queue_order(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        first_path = Path(self.directory.name) / "arrival-first.gguf"
        write_synthetic_gguf(first_path, block_count=2, sliding_window=32)
        first_model = scheduler.register_gguf_model(
            "arrival-model-a", first_path
        )
        second_model = scheduler.register_gguf_model(
            "arrival-model-b", self.path
        )
        hot = replace(
            runtime_snapshot(
                first_model,
                include_phone=False,
                resident_devices=("accelerator-b",),
            ),
            executors={
                gpu.executor_id: executor_state(gpu.executor_id)
            },
        )
        current = replace(
            hot.residency[0],
            executor_id=gpu.executor_id,
            reclaimable_bytes=hot.residency[0].resident_bytes,
        )
        hot = replace(hot, residency=(current,))
        first = scheduler.submit_automated_request(
            request("arrival-model-b-first", arrival_us=1_000),
            second_model.model_id,
            hot,
            selection_mode="desktop-baseline",
        )
        second = scheduler.submit_automated_request(
            request("arrival-model-a-second", arrival_us=1_100),
            first_model.model_id,
            hot,
            selection_mode="desktop-baseline",
        )
        capacity = hot.memory.capacities["gpu-memory"]
        changed = replace(
            hot,
            snapshot_id="arrival-changed-reclaimable",
            captured_at_us=1_200,
            memory=replace(
                hot.memory,
                snapshot_id="arrival-changed-reclaimable-memory",
                captured_at_us=1_200,
                capacities={
                    "gpu-memory": replace(
                        capacity,
                        occupied_bytes=capacity.occupied_bytes + 1,
                    )
                },
            ),
            residency=(replace(
                current,
                reclaimable_bytes=current.reclaimable_bytes + 1,
            ),),
        )

        third = scheduler.submit_automated_request(
            request("arrival-model-b-third", arrival_us=1_200),
            second_model.model_id,
            changed,
            observed_at_us=1_200,
            selection_mode="desktop-baseline",
        )

        first_replan = scheduler.runtime_ticket(
            first.request.request_id
        )
        second_replan = scheduler.runtime_ticket(
            second.request.request_id
        )
        self.assertEqual(first_replan.attempt_index, 1)
        self.assertEqual(second_replan.attempt_index, 0)
        self.assertEqual(second_replan.lease_status, "CANCELLED")
        self.assertEqual(
            scheduler.runtime_controller_snapshot()["dispatch_queue"]
                ["entry_states"][second.request.request_id]["state"],
            "DEFERRED_REPLAN",
        )
        self.assertGreaterEqual(
            third.decision.start_us,
            max(
                row.reserved_until_us
                for row in first_replan.decision.leases
            ),
        )
        replans = tuple(
            row["request_ids"][0]
            for row in scheduler.runtime_decision_log()["records"]
            if row["event_kind"] == "REPLAN"
        )
        self.assertEqual(
            replans,
            (first.request.request_id,),
        )
        eviction = first_replan.execution_plan.transitions[0].evictions[0]
        self.assertEqual(
            eviction.reclaimable_bytes, current.reclaimable_bytes + 1
        )

    def test_arrival_repair_defers_after_snapshot_expiration(self) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
        ).to_json())
        scheduler, manifest = self.scheduler_and_manifest(profile)
        snapshot = runtime_snapshot(
            manifest,
            include_phone=False,
            resident_devices=(),
        )
        predecessor = scheduler.submit_automated_request(
            request("expired-repair-predecessor", arrival_us=1_000),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        follower = scheduler.submit_automated_request(
            request("expired-repair-follower", arrival_us=1_100),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        error = RuntimeResidencyProjectionError(
            "projected transition eviction is stale: accelerator-b",
            request_id=predecessor.request.request_id,
            ticket_id=predecessor.ticket_id,
        )
        original_wait = scheduler._runtime_controller.wait

        def expired_wait(request_id, epoch_ns):
            ticket = original_wait(request_id, epoch_ns)
            return replace(
                ticket,
                dispatch_receipt=replace(
                    ticket.dispatch_receipt,
                    observed_at_us=snapshot.valid_until_us,
                ),
            )

        with mock.patch.object(
            scheduler._runtime_controller,
            "wait",
            side_effect=expired_wait,
        ), self.assertRaises(RuntimeReplanRetryRequired):
            scheduler._repair_stale_projection_chain(
                error, snapshot, snapshot.valid_until_us
            )

        self.assertEqual(
            scheduler.runtime_ticket(
                predecessor.request.request_id
            ).dispatch_state,
            "REPLAN_REQUIRED",
        )
        self.assertEqual(
            scheduler.runtime_ticket(
                follower.request.request_id
            ).dispatch_state,
            "QUEUED",
        )
        self.assertNotIn(
            predecessor.request.request_id,
            scheduler._runtime_controller.projection_request_ids(),
        )
        self.assertNotIn(
            "FAILED",
            tuple(
                row["event_kind"]
                for row in scheduler.runtime_decision_log()["records"]
            ),
        )

    def test_published_epoch_reuses_static_audit_and_live_costs(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        first = scheduler.submit_automated_request(
            request("published-epoch-first"),
            manifest.model_id,
            snapshot,
        )
        compiler = scheduler._automated_compiler()
        prospective = scheduler._select_model_placement_candidate
        with mock.patch.object(
            compiler,
            "generate",
            side_effect=AssertionError("full placement generation ran"),
        ), mock.patch.object(
            scheduler,
            "_select_model_placement_candidate",
            wraps=prospective,
        ) as prospective_selection:
            second = scheduler.submit_automated_request(
                request("published-epoch-second", arrival_us=1_100),
                manifest.model_id,
                snapshot,
                observed_at_us=1_100,
            )

        self.assertEqual(prospective_selection.call_count, 0)

        self.assertEqual(
            len(second.cost_estimates.estimates),
            len(first.cost_estimates.estimates),
        )
        selected = next(
            row for row in second.cost_estimates.estimates
            if row.route_id == second.decision.route_id
        )
        epoch = selected.details["model_placement_epoch"]
        self.assertEqual(epoch["artifact_sha256"], manifest.artifact_sha256)
        self.assertEqual(epoch["selected_route_id"], first.decision.route_id)
        self.assertEqual(
            epoch["selected_executor_id"], first.binding.executor_id
        )
        self.assertEqual(
            epoch["selected_operator_plan_sha256"],
            first.execution_plan.plan_sha256,
        )
        self.assertEqual(epoch["objective"], "energy-aware")
        self.assertTrue(epoch["allowed_adaptive_fractions_ppm"])
        self.assertTrue(all(
            epoch[name].startswith("sha256:")
            for name in (
                "capability_generation_sha256",
                "profile_generation_sha256",
                "transport_generation_sha256",
                "learning_generation_sha256",
                "route_template_identity_sha256",
                "selected_component_identity_sha256",
            )
        ))
        timing = scheduler.runtime_decision_timings()[-1]
        self.assertTrue(timing["model_placement_epoch_fast_path"])
        self.assertEqual(
            timing["model_placement_epoch_invalidation_reason"], "NONE"
        )
        records = scheduler.runtime_decision_log()["records"]
        self.assertEqual(
            len(records[-1]["candidates"]),
            len(second.cost_estimates.estimates),
        )
        fallback = tuple(
            row for row in second.cost_estimates.estimates
            if row.details.get("recovery_fallback")
        )
        self.assertEqual(len(fallback), 1)
        self.assertFalse(fallback[0].admitted)
        self.assertIn(
            "MODEL_EPOCH_AUDIT_ONLY",
            fallback[0].details["rejection_reasons"],
        )
        self.assertEqual(
            records[-1]["selected"]["model_placement_epoch"][
                "epoch_sha256"
            ],
            epoch["epoch_sha256"],
        )
        self.assertEqual(
            dict(scheduler.model_placement_epoch_stats()),
            {
                "background_refresh_candidates": 0,
                "background_refresh_failures": 0,
                "background_refresh_total_us": 0,
                "background_refreshes": 0,
                "entries": 1,
                "evictions": 0,
                "hits": 1,
                "invalidations": 0,
                "invalidation_reasons": {},
                "misses": 1,
            },
        )

    def test_replan_regenerates_after_live_epoch_revalidation_fails(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        other_path = Path(self.directory.name) / "live-epoch-other.gguf"
        write_synthetic_gguf(other_path, sliding_window=16)
        other = scheduler.register_gguf_model(
            "live-epoch-other", other_path
        )
        hot = runtime_snapshot(
            manifest,
            include_phone=False,
            resident_devices=("host-a", "accelerator-b"),
        )
        queued = scheduler.submit_automated_request(
            request("live-epoch-replan", arrival_us=1_000),
            manifest.model_id,
            hot,
            selection_mode="desktop-baseline",
        )
        observed_at_us = 1_100
        changed_residency = tuple(
            row for row in hot.residency
            if row.device_id != "accelerator-b"
        ) + (ModelResidencyObservation(
            model_id=other.model_id,
            artifact_sha256=other.artifact_sha256,
            device_id="accelerator-b",
            state="hot",
            resident_tensor_ids=tuple(
                tensor.tensor_id for tensor in other.tensors
            ),
            resident_bytes=other.tensor_bytes,
            generation=2,
        ),)
        refreshed = replace(
            hot,
            snapshot_id="live-epoch-refreshed",
            captured_at_us=observed_at_us,
            valid_until_us=observed_at_us + 10_000_000,
            memory=replace(
                hot.memory,
                snapshot_id="live-epoch-refreshed-memory",
                captured_at_us=observed_at_us,
                valid_until_us=observed_at_us + 10_000_000,
            ),
            residency=changed_residency,
        )
        self.assertEqual(
            scheduler.observe_automated_runtime_snapshot(
                refreshed, observed_at_us=observed_at_us
            ),
            (queued.request.request_id,),
        )
        wake = scheduler.wait_runtime_request(
            queued.request.request_id,
            time.monotonic_ns() - observed_at_us * 1_000,
        )
        compiler = scheduler._automated_compiler()
        original_generate = compiler.generate
        with mock.patch.object(
            compiler,
            "reuse_exact_route_template_set",
            return_value=None,
        ), mock.patch.object(
            compiler,
            "materialize_route_template_set",
            side_effect=RouteGenerationError(
                "published desktop baseline live revalidation failed"
            ),
        ), mock.patch.object(
            compiler,
            "generate",
            wraps=original_generate,
        ) as regenerate:
            replanned = scheduler.replan_automated_request(
                queued.request.request_id,
                observed_at_us=observed_at_us,
                reason=wake.dispatch_receipt.wake_reason,
                snapshot=refreshed,
            )

        self.assertGreaterEqual(regenerate.call_count, 1)
        self.assertEqual(replanned.selection_mode, "desktop-baseline")
        self.assertIn("accelerator-b", replanned.execution_plan.device_ids)
        selected = next(
            row for row in replanned.cost_estimates.estimates
            if row.route_id == replanned.decision.route_id
        )
        self.assertTrue(selected.details["model_placement_epoch"])
        self.assertFalse(
            selected.details["model_placement_epoch_fast_path"]
        )
        self.assertEqual(
            selected.details["model_placement_epoch_invalidation_reason"],
            "ROUTE_TEMPLATE_LIVE_REVALIDATION_FAILED",
        )

    def test_cross_shape_epoch_remaps_audit_parent_after_residency_change(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        cold = runtime_snapshot(
            manifest,
            resident_devices=("host-a", "helper-c"),
        )
        first_request = request(
            "cross-shape-cold-parent",
            input_tokens=12,
            output_tokens=4,
        )
        candidates = scheduler.generate_automated_candidates(
            first_request, manifest.model_id, cold
        )
        baseline = candidates.baseline
        audit = next(
            row for row in candidates.candidates
            if row.candidate_id != baseline.candidate_id
            and row.candidate_id
                != candidates.recovery_fallback_route_id
        )
        paired_plan = replace(
            audit.plan,
            baseline_executor_id=baseline.binding.executor_id,
            desktop_placement_sha256=(
                baseline.plan.desktop_placement_sha256
            ),
        )
        audit = replace(
            audit,
            plan=paired_plan,
            binding=replace(
                audit.binding,
                operator_plan_sha256=paired_plan.plan_sha256,
            ),
            paired_baseline_route_id=baseline.candidate_id,
        )
        candidates = replace(
            candidates,
            candidates=tuple(
                audit if row.candidate_id == audit.candidate_id else row
                for row in candidates.candidates
            ),
        )
        compiler = scheduler._automated_compiler()
        templates = compiler.compile_route_template_set(
            candidates,
            baseline,
            manifest,
            input_token_bucket=request_shape_bucket(12, 4)[0],
            output_token_bucket=request_shape_bucket(12, 4)[1],
            quality_requirement=first_request.quality_requirement,
        )
        hot = replace(
            runtime_snapshot(manifest),
            snapshot_id="cross-shape-hot-runtime",
        )
        second_request = request(
            "cross-shape-hot-parent",
            arrival_us=1_100,
            input_tokens=65,
            output_tokens=33,
        )
        materialize_route = compiler._materialize_route_key
        with mock.patch.object(
            compiler,
            "_materialize_route_key",
            wraps=materialize_route,
        ) as live_materialize:
            rematerialized = compiler.materialize_route_template_set(
                templates,
                second_request,
                manifest,
                hot,
                observed_at_us=1_100,
                residency_holds={},
                search_metadata={},
                additional_live_route_ids=(),
            )
        estimates = candidate_set_to_runtime_costs(
            rematerialized,
            second_request,
            manifest,
            hot,
            compiler.catalog,
        )

        rows = {
            row.route_id: row for row in estimates.estimates
        }
        self.assertEqual(
            len(rematerialized.candidates), len(candidates.candidates)
        )
        self.assertLess(
            live_materialize.call_count, len(candidates.candidates)
        )
        self.assertNotEqual(
            baseline.candidate_id, rematerialized.baseline_route_id
        )
        self.assertEqual(len(rows), len(candidates.candidates))
        self.assertTrue(all(
            row.details["paired_baseline_route_id"] is None
            or row.details["paired_baseline_route_id"] in rows
            for row in rows.values()
        ))
        self.assertTrue(
            next(iter(rows.values())).details[
                "route_template_cross_shape_reuse"
            ]
        )

    def test_published_epoch_template_is_not_evicted(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        scheduler._runtime_route_template_cache_maximum = 1
        scheduler.submit_automated_request(
            request("pinned-template-first", arrival_us=1_000),
            manifest.model_id,
            snapshot,
        )
        published = (
            scheduler._runtime_residency_cohorts
            .published_model_placement_epoch_hashes()
        )
        self.assertEqual(len(published), 1)
        first_epoch_sha256 = next(iter(published))

        with self.assertRaisesRegex(
            UnifiedScheduleError,
            "published model placement template capacity is exhausted",
        ):
            scheduler.submit_automated_request(
                replace(request(
                    "pinned-template-second",
                    arrival_us=1_100,
                ), quality_requirement="semantic"),
                manifest.model_id,
                snapshot,
                observed_at_us=1_100,
            )

        self.assertEqual(
            scheduler._runtime_residency_cohorts
                .published_model_placement_epoch_hashes(),
            published,
        )
        self.assertIn(
            first_epoch_sha256, scheduler._runtime_route_template_sets
        )

    def test_pending_hot_reuse_preserves_measured_eviction_allocation(self) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source, executors=(gpu,), composite_executors=(),
            transitions=tuple(
                replace(row, executor_id=gpu.executor_id,
                        prepares_device_ids=(gpu.device_id,))
                for row in source.transitions if row.device_id == gpu.device_id
            ),
        ).to_json())
        scheduler, manifest = self.scheduler_and_manifest(profile)
        cold = replace(
            runtime_snapshot(manifest, include_phone=False, resident_devices=()),
            executors={gpu.executor_id: executor_state(gpu.executor_id)},
        )
        ticket = scheduler.submit_automated_request(
            request("pending-hot-reuse"), manifest.model_id, cold,
            selection_mode="desktop-baseline",
        )
        plan = replace(
            ticket.execution_plan, residency_variant="hot",
            transitions=tuple(
                replace(row, source_state="hot", evictions=())
                for row in ticket.execution_plan.transitions
            ),
        )
        ticket = replace(
            ticket, execution_plan=plan,
            binding=replace(ticket.binding, operator_plan_sha256=plan.plan_sha256),
            executor_bindings=tuple(
                replace(row, operator_plan_sha256=plan.plan_sha256)
                if row == ticket.binding else row
                for row in ticket.executor_bindings
            ),
        )
        self.assertEqual(ticket.transition_status, "PENDING")
        weights = next(row.required_bytes for row in plan.memory_demands
                       if row.kind == "model_weights")
        modelled_allocation = sum(row.required_bytes for row in plan.memory_demands)
        for measured_allocation in (None, modelled_allocation + 240_479_756):
            with self.subTest(reclaimable_bytes=measured_allocation):
                resident = ModelResidencyObservation(
                    model_id=manifest.model_id,
                    artifact_sha256=manifest.artifact_sha256,
                    device_id=gpu.device_id, state="hot",
                    resident_tensor_ids=tuple(row.tensor_id for row in manifest.tensors),
                    resident_bytes=weights, generation=3,
                    executor_id=gpu.executor_id,
                    reclaimable_bytes=measured_allocation,
                )
                hot = replace(cold, residency=(resident,))
                projected = project_scheduler_residency(
                    hot, profile, (ticket,), {manifest.model_id: manifest},
                )
                self.assertEqual(projected.residency, hot.residency)
                self.assertEqual(projected.memory, hot.memory)
                self.assertEqual(ticket.transition_status, "PENDING")

    def test_stale_predecessor_projection_defers_follower_replan(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        predecessor = scheduler.submit_automated_request(
            request("stale-predecessor", arrival_us=1_000),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        follower = scheduler.submit_automated_request(
            request("stale-follower", arrival_us=1_100),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        self.assertTrue(
            scheduler._runtime_controller.require_queued_replan(
                follower.request.request_id,
                "runtime_state_changed",
            )
        )
        wake = scheduler.wait_runtime_request(
            follower.request.request_id,
            time.monotonic_ns() - follower.decision.start_us * 1_000,
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")

        with mock.patch.object(
            scheduler,
            "_automated_snapshot_for_request",
            side_effect=RuntimeResidencyProjectionError(
                "projected transition eviction is stale: accelerator-b",
                request_id=predecessor.request.request_id,
                ticket_id=predecessor.ticket_id,
            ),
        ):
            deferred = scheduler.replan_automated_request(
                follower.request.request_id,
                observed_at_us=wake.dispatch_receipt.observed_at_us,
                reason=wake.dispatch_receipt.wake_reason,
                snapshot=snapshot,
            )

        self.assertEqual(deferred.ticket_id, follower.ticket_id)
        self.assertEqual(deferred.dispatch_state, "REPLAN_REQUIRED")
        self.assertEqual(
            scheduler.runtime_controller_snapshot()["dispatch_queue"]
                ["entry_states"][follower.request.request_id]["state"],
            "DEFERRED_REPLAN",
        )
        predecessor_wake = scheduler.wait_runtime_request(
            predecessor.request.request_id,
            time.monotonic_ns() - predecessor.decision.start_us * 1_000,
        )
        self.assertEqual(
            predecessor_wake.dispatch_state, "REPLAN_REQUIRED"
        )
        predecessor_replan = scheduler.replan_automated_request(
            predecessor.request.request_id,
            observed_at_us=predecessor_wake.dispatch_receipt.observed_at_us,
            reason=predecessor_wake.dispatch_receipt.wake_reason,
            snapshot=snapshot,
        )
        self.assertEqual(predecessor_replan.attempt_index, 1)
        self.assertEqual(
            scheduler.runtime_controller_snapshot()["dispatch_queue"]
                ["entry_states"][follower.request.request_id]["state"],
            "DEFERRED_REPLAN",
        )
        predecessor_active = scheduler.wait_runtime_request(
            predecessor.request.request_id,
            time.monotonic_ns()
            - predecessor_replan.decision.start_us * 1_000,
        )
        self.assertEqual(predecessor_active.dispatch_state, "ACQUIRED")
        predecessor_finished_us = max(
            predecessor_active.decision.start_us + 1,
            wake.dispatch_receipt.observed_at_us,
        )
        scheduler.complete_automated_request(
            predecessor.request.request_id,
            self.execution_receipt(
                predecessor_active,
                finished_us=predecessor_finished_us,
            ),
        )
        follower_wake = scheduler.wait_runtime_request(
            follower.request.request_id,
            time.monotonic_ns() - predecessor_finished_us * 1_000,
        )
        self.assertEqual(
            follower_wake.dispatch_state, "REPLAN_REQUIRED"
        )
        follower_replan = scheduler.replan_automated_request(
            follower.request.request_id,
            observed_at_us=(
                follower_wake.dispatch_receipt.observed_at_us
            ),
            reason=follower_wake.dispatch_receipt.wake_reason,
            snapshot=snapshot,
        )
        self.assertEqual(follower_replan.attempt_index, 1)
        self.assertNotIn(
            "FAILED",
            tuple(
                row["event_kind"]
                for row in scheduler.runtime_decision_log()["records"]
            ),
        )

    def test_arrival_projection_repair_defers_until_causal_predecessor(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        predecessor = scheduler.submit_automated_request(
            request("projection-owner", arrival_us=1_000),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        stale = scheduler.submit_automated_request(
            request("projection-stale", arrival_us=1_100),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        before = len(scheduler.runtime_decision_log()["records"])

        scheduler._repair_stale_projection_chain(
            RuntimeResidencyProjectionError(
                "projected transition eviction is stale: accelerator-b",
                request_id=stale.request.request_id,
                ticket_id=stale.ticket_id,
            ),
            snapshot,
            stale.request.arrival_us,
        )

        current = scheduler.runtime_ticket(stale.request.request_id)
        queue = scheduler.runtime_controller_snapshot()["dispatch_queue"]
        self.assertEqual(current.ticket_id, stale.ticket_id)
        self.assertEqual(current.lease_status, "CANCELLED")
        self.assertEqual(
            current.memory_reservation_status, "CANCELLED"
        )
        self.assertFalse(any(
            row["owner_id"] == stale.request.request_id
            and row["end_us"] > row["start_us"]
            for resource in scheduler.timeline.causal_state()[
                "resources"
            ].values()
            for lane in resource["lanes"]
            for row in lane
        ))
        self.assertFalse(any(
            row["owner_id"] == stale.request.request_id
            for row in scheduler.runtime_memory_state()["reservations"]
        ))
        self.assertEqual(
            queue["entry_states"][stale.request.request_id]["state"],
            "DEFERRED_REPLAN",
        )
        self.assertEqual(
            queue["causal_predecessors"][stale.request.request_id],
            [predecessor.request.request_id],
        )
        records = scheduler.runtime_decision_log()["records"]
        self.assertEqual(len(records), before)
        self.assertFalse(any(
            row["event_kind"] == "REPLAN"
            and stale.request.request_id in row["request_ids"]
            for row in records
        ))

    def test_arrival_projection_repair_replans_ready_predecessor(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
        ).to_json())
        scheduler, manifest = self.scheduler_and_manifest(profile)
        snapshot = replace(
            runtime_snapshot(
                manifest,
                include_phone=False,
                resident_devices=(),
            ),
            executors={
                gpu.executor_id: executor_state(gpu.executor_id)
            },
        )
        predecessor = scheduler.submit_automated_request(
            request("projection-ready-replan", arrival_us=1_000),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        self.assertTrue(
            scheduler._runtime_controller.require_queued_replan(
                predecessor.request.request_id,
                "residency_projection_invalid",
            )
        )
        wake = scheduler.wait_runtime_request(
            predecessor.request.request_id,
            time.monotonic_ns()
            - predecessor.decision.start_us * 1_000,
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")

        barrier = scheduler._repair_stale_projection_chain(
            RuntimeResidencyProjectionError(
                "projected transition eviction is stale: accelerator-b",
                request_id=predecessor.request.request_id,
                ticket_id=predecessor.ticket_id,
            ),
            snapshot,
            predecessor.request.arrival_us,
        )

        replacement = scheduler.runtime_ticket(
            predecessor.request.request_id
        )
        self.assertEqual(replacement.attempt_index, 1)
        self.assertEqual(
            replacement.previous_ticket_id, predecessor.ticket_id
        )
        self.assertTrue(barrier)
        self.assertTrue(any(
            row["event_kind"] == "REPLAN"
            and replacement.request.request_id in row["request_ids"]
            for row in scheduler.runtime_decision_log()["records"]
        ))

    def test_arrival_projection_repair_detects_repeated_attempt_cycle(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(manifest)
        predecessor = scheduler.submit_automated_request(
            request("projection-cycle-owner", arrival_us=1_000),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )

        def fail_projection(*_args, **_kwargs):
            current = scheduler._runtime_controller.ticket(
                predecessor.request.request_id
            )
            error = RuntimeResidencyProjectionError(
                "projected eviction exceeds occupied memory: gpu-memory",
                request_id=current.request.request_id,
                ticket_id=current.ticket_id,
            )
            raise UnifiedScheduleError(str(error)) from error

        def replace_attempt(*_args, **_kwargs):
            current = scheduler._runtime_controller.ticket(
                predecessor.request.request_id
            )
            replacement = replace(
                current,
                ticket_id=(
                    current.request.request_id
                    + ":attempt:"
                    + str(current.attempt_index + 1)
                ),
                attempt_index=current.attempt_index + 1,
                previous_ticket_id=current.ticket_id,
            )
            scheduler._runtime_controller._tickets[
                current.request.request_id
            ] = replacement
            return {"compute:accelerator-b": 10_000}

        with mock.patch.object(
            scheduler,
            "_submit_automated_request_once",
            side_effect=fail_projection,
        ), mock.patch.object(
            scheduler,
            "_repair_stale_projection_chain",
            side_effect=replace_attempt,
        ) as repair:
            with self.assertRaisesRegex(
                UnifiedScheduleError,
                "runtime stale projection repair made no progress",
            ):
                scheduler.submit_automated_request(
                    request("projection-cycle-arrival", arrival_us=1_100),
                    manifest.model_id,
                    snapshot,
                    selection_mode="desktop-baseline",
                )

        self.assertEqual(repair.call_count, 2)

    def test_arrival_projection_repair_waits_for_replanning_predecessor(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        transition = replace(
            next(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
            executor_id=gpu.executor_id,
            prepares_device_ids=("accelerator-b",),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            composite_executors=(),
            transitions=(transition,),
        ).to_json())
        scheduler, manifest = self.scheduler_and_manifest(profile)
        snapshot = replace(
            runtime_snapshot(
                manifest,
                include_phone=False,
                resident_devices=(),
            ),
            executors={
                gpu.executor_id: executor_state(gpu.executor_id)
            },
        )
        predecessor = scheduler.submit_automated_request(
            request("projection-replanning", arrival_us=1_000),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        self.assertTrue(
            scheduler._runtime_controller.require_queued_replan(
                predecessor.request.request_id,
                "residency_observation_changed",
            )
        )
        wake = scheduler.wait_runtime_request(
            predecessor.request.request_id,
            time.monotonic_ns()
            - predecessor.decision.start_us * 1_000,
        )
        scheduler._runtime_controller.queue.retire_replan(
            predecessor.request.request_id,
            wake.dispatch_receipt.queue_generation,
        )
        records_before = len(
            scheduler.runtime_decision_log()["records"]
        )

        barrier = scheduler._repair_stale_projection_chain(
            RuntimeResidencyProjectionError(
                "projected transition eviction is stale: accelerator-b",
                request_id=predecessor.request.request_id,
                ticket_id=predecessor.ticket_id,
            ),
            snapshot,
            predecessor.request.arrival_us,
        )

        self.assertTrue(barrier)
        queue = scheduler.runtime_controller_snapshot()["dispatch_queue"]
        self.assertEqual(
            queue["entry_states"][predecessor.request.request_id]["state"],
            "REPLANNING",
        )
        self.assertEqual(
            scheduler.runtime_ticket(
                predecessor.request.request_id
            ).ticket_id,
            predecessor.ticket_id,
        )
        self.assertEqual(
            len(scheduler.runtime_decision_log()["records"]),
            records_before,
        )

    def test_active_stale_predecessor_defers_follower_until_completion(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(
            manifest,
            include_phone=False,
            resident_devices=("host-a", "accelerator-b"),
        )
        predecessor = scheduler.submit_automated_request(
            request("active-stale-predecessor", arrival_us=1_000),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        follower = scheduler.submit_automated_request(
            request("active-stale-follower", arrival_us=1_100),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        active = scheduler.wait_runtime_request(
            predecessor.request.request_id,
            time.monotonic_ns() - predecessor.decision.start_us * 1_000,
        )
        self.assertEqual(active.dispatch_state, "ACQUIRED")
        self.assertTrue(
            scheduler._runtime_controller.require_queued_replan(
                follower.request.request_id,
                "runtime_state_changed",
            )
        )
        wake = scheduler.wait_runtime_request(
            follower.request.request_id,
            time.monotonic_ns() - follower.decision.start_us * 1_000,
        )
        self.assertEqual(wake.dispatch_state, "REPLAN_REQUIRED")

        with mock.patch.object(
            scheduler,
            "_automated_snapshot_for_request",
            side_effect=RuntimeResidencyProjectionError(
                "projected transition eviction is stale: accelerator-b",
                request_id=predecessor.request.request_id,
                ticket_id=predecessor.ticket_id,
            ),
        ):
            deferred = scheduler.replan_automated_request(
                follower.request.request_id,
                observed_at_us=wake.dispatch_receipt.observed_at_us,
                reason=wake.dispatch_receipt.wake_reason,
                snapshot=snapshot,
            )

        self.assertEqual(deferred.ticket_id, follower.ticket_id)
        self.assertEqual(deferred.dispatch_state, "REPLAN_REQUIRED")
        with self.assertRaises(RuntimeReplanRetryRequired) as raised:
            scheduler.replan_automated_request(
                follower.request.request_id,
                observed_at_us=wake.dispatch_receipt.observed_at_us,
                reason=wake.dispatch_receipt.wake_reason,
                snapshot=snapshot,
                expected_ticket_id=follower.ticket_id,
                expected_queue_generation=(
                    wake.dispatch_receipt.queue_generation
                ),
            )
        self.assertEqual(raised.exception.reason, "REPLAN_WAKE_ROLLED_BACK")
        self.assertNotIn(
            "FAILED",
            tuple(
                row["event_kind"]
                for row in scheduler.runtime_decision_log()["records"]
            ),
        )
        follower_wakes = []
        waiter = threading.Thread(
            target=lambda: follower_wakes.append(
                scheduler.wait_runtime_request(
                    follower.request.request_id,
                    time.monotonic_ns()
                    - follower.decision.start_us * 1_000,
                )
            )
        )
        waiter.start()
        time.sleep(0.01)
        self.assertTrue(waiter.is_alive())

        completed_at_us = max(
            active.decision.start_us + 1,
            wake.dispatch_receipt.observed_at_us,
        )
        scheduler.complete_automated_request(
            predecessor.request.request_id,
            self.execution_receipt(active, finished_us=completed_at_us),
        )
        waiter.join(timeout=1)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(len(follower_wakes), 1)
        self.assertEqual(
            follower_wakes[0].dispatch_state, "REPLAN_REQUIRED"
        )
        self.assertEqual(
            follower_wakes[0].dispatch_receipt.wake_reason,
            "predecessor_completion",
        )
        replanned = scheduler.replan_automated_request(
            follower.request.request_id,
            observed_at_us=completed_at_us,
            reason=follower_wakes[0].dispatch_receipt.wake_reason,
            snapshot=snapshot,
        )
        self.assertEqual(replanned.attempt_index, 1)
        self.assertNotIn(
            "FAILED",
            tuple(
                row["event_kind"]
                for row in scheduler.runtime_decision_log()["records"]
            ),
        )

    def test_stale_replan_wake_observes_newer_scheduler_attempt(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        snapshot = runtime_snapshot(
            manifest,
            include_phone=False,
            resident_devices=("host-a", "accelerator-b"),
        )
        original = scheduler.submit_automated_request(
            request("superseded-replan-wake", arrival_us=1_000),
            manifest.model_id,
            snapshot,
            selection_mode="desktop-baseline",
        )
        self.assertTrue(
            scheduler._runtime_controller.require_queued_replan(
                original.request.request_id,
                "runtime_state_changed",
            )
        )
        wake = scheduler.wait_runtime_request(
            original.request.request_id,
            time.monotonic_ns() - original.decision.start_us * 1_000,
        )
        replacement = scheduler.replan_automated_request(
            original.request.request_id,
            observed_at_us=wake.dispatch_receipt.observed_at_us,
            reason=wake.dispatch_receipt.wake_reason,
            snapshot=snapshot,
            expected_ticket_id=original.ticket_id,
        )

        observed = scheduler.replan_automated_request(
            original.request.request_id,
            observed_at_us=wake.dispatch_receipt.observed_at_us,
            reason=wake.dispatch_receipt.wake_reason,
            snapshot=snapshot,
            expected_ticket_id=original.ticket_id,
        )

        self.assertEqual(observed.ticket_id, replacement.ticket_id)
        self.assertEqual(observed.attempt_index, 1)
        self.assertEqual(observed.dispatch_state, "QUEUED")
        events = tuple(
            row["event_kind"]
            for row in scheduler.runtime_decision_log()["records"]
        )
        self.assertEqual(events.count("REPLAN"), 1)
        self.assertNotIn("FAILED", events)

    def test_failed_replan_terminalizes_owner_and_wakes_follower(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        hot = runtime_snapshot(
            manifest,
            include_phone=False,
            resident_devices=("host-a", "accelerator-b"),
        )
        tickets = tuple(
            scheduler.submit_automated_request(
                request(
                    "failed-replan-" + str(index),
                    arrival_us=1_000 + index,
                ),
                manifest.model_id,
                hot,
                selection_mode="desktop-baseline",
            )
            for index in range(3)
        )
        first = scheduler.wait_runtime_request(
            tickets[0].request.request_id,
            time.monotonic_ns() - tickets[0].decision.start_us * 1_000,
        )
        extended_until_us = tickets[-1].decision.finish_upper_us + 1_000
        scheduler.extend_runtime_request(
            first.request.request_id,
            at_us=first.decision.start_us + 1,
            reserved_until_us=extended_until_us,
        )
        observed_at_us = max(
            tickets[-1].request.arrival_us,
            first.decision.start_us + 2,
        )
        scheduler.complete_automated_request(
            first.request.request_id,
            self.execution_receipt(first, finished_us=observed_at_us),
        )
        failed_wake = scheduler.wait_runtime_request(
            tickets[1].request.request_id,
            time.monotonic_ns() - extended_until_us * 1_000,
        )
        follower_results = []
        follower = threading.Thread(
            target=lambda: follower_results.append(
                scheduler.wait_runtime_request(
                    tickets[2].request.request_id,
                    time.monotonic_ns() - extended_until_us * 1_000,
                )
            )
        )
        follower.start()
        time.sleep(0.01)
        self.assertTrue(follower.is_alive())

        with mock.patch.object(
            scheduler,
            "_select_automated_candidate",
            side_effect=RuntimeError("synthetic replan failure"),
        ):
            with self.assertRaisesRegex(
                UnifiedScheduleError, "synthetic replan failure"
            ):
                scheduler.replan_automated_request(
                    tickets[1].request.request_id,
                    observed_at_us=observed_at_us,
                    reason=failed_wake.dispatch_receipt.wake_reason,
                    snapshot=hot,
                )

        follower.join(1)
        self.assertFalse(follower.is_alive())
        self.assertEqual(len(follower_results), 1)
        self.assertEqual(
            follower_results[0].dispatch_state, "REPLAN_REQUIRED"
        )
        failed = scheduler.runtime_ticket(tickets[1].request.request_id)
        self.assertEqual(failed.dispatch_state, "FAILED")
        self.assertEqual(failed.lease_status, "CANCELLED")
        self.assertEqual(failed.memory_reservation_status, "CANCELLED")
        self.assertEqual(
            scheduler.wait_runtime_request(
                failed.request.request_id, time.monotonic_ns()
            ),
            failed,
        )
        log_before = scheduler.runtime_decision_log_bytes()
        with self.assertRaisesRegex(
            UnifiedScheduleError,
            "runtime terminal ticket cannot be replanned",
        ):
            scheduler.replan_automated_request(
                failed.request.request_id,
                observed_at_us=observed_at_us,
                reason="late_cleanup_replan",
                snapshot=hot,
            )
        self.assertEqual(
            scheduler.runtime_ticket(failed.request.request_id), failed
        )
        self.assertEqual(
            scheduler.runtime_decision_log_bytes(), log_before
        )
        self.assertFalse(any(
            row["owner_id"] == failed.request.request_id
            for row in scheduler.runtime_memory_state()["reservations"]
        ))
        terminal_rows = tuple(
            row for row in scheduler.runtime_decision_log()["records"]
            if row["event_kind"] == "FAILED"
            and row["request_ids"] == [failed.request.request_id]
        )
        self.assertEqual(len(terminal_rows), 1)

    def test_queue_released_compaction_follower_detaches_ownership(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        hot = runtime_snapshot(
            manifest,
            include_phone=False,
            resident_devices=("host-a", "accelerator-b"),
        )
        root = scheduler.submit_automated_request(
            request("released-compaction-root", arrival_us=1_000),
            manifest.model_id,
            hot,
            selection_mode="desktop-baseline",
        )
        follower = scheduler.submit_automated_request(
            request("released-compaction-follower", arrival_us=1_001),
            manifest.model_id,
            hot,
            selection_mode="desktop-baseline",
        )
        recorded_memory_tokens = tuple(
            row.token for row in follower.memory_reservations
        )
        self.assertTrue(recorded_memory_tokens)

        def replacement(request_id, **_kwargs):
            if request_id == root.request.request_id:
                changed = (
                    scheduler._runtime_controller.queue.require_replan(
                        follower.request.request_id,
                        "capacity_released_early",
                    )
                )
                self.assertTrue(changed)
                return root
            return scheduler.runtime_ticket(request_id)

        with mock.patch.object(
            scheduler._runtime_controller,
            "priority_compaction_followers",
            return_value=(),
        ), mock.patch.object(
            scheduler,
            "_replan_automated_request_once",
            side_effect=replacement,
        ) as replan, mock.patch.object(
            scheduler._runtime_memory,
            "release_owner",
            wraps=scheduler._runtime_memory.release_owner,
        ) as release_owner:
            result = scheduler.replan_automated_request(
                root.request.request_id,
                observed_at_us=2_000,
                reason="capacity_released_early",
                snapshot=hot,
                expected_ticket_id=root.ticket_id,
            )

        self.assertEqual(result, root)
        self.assertEqual(replan.call_count, 2)
        detached = scheduler.runtime_ticket(follower.request.request_id)
        self.assertEqual(detached.dispatch_state, "REPLAN_REQUIRED")
        self.assertEqual(detached.lease_status, "CANCELLED")
        self.assertEqual(
            detached.memory_reservation_status, "CANCELLED"
        )
        self.assertEqual(
            release_owner.call_args_list,
            [mock.call(follower.request.request_id)],
        )
        self.assertEqual(
            scheduler._runtime_memory.owner_tokens(
                follower.request.request_id
            ),
            (),
        )
        self.assertEqual(
            tuple(row.token for row in detached.memory_reservations),
            recorded_memory_tokens,
        )

    def test_live_transport_measurement_overrides_catalog_bandwidth(
        self,
    ) -> None:
        profile = catalog()
        _, manifest = self.scheduler_and_manifest(profile)
        snapshot = UnifiedRuntimeSnapshotBuilder(profile).build(
            snapshot_id="live-link-bandwidth",
            captured_at_us=0,
            valid_until_us=10_000_000,
            memory=runtime_snapshot(manifest).memory,
            executor_samples={
                row.executor_id: EndpointRuntimeSample(
                    "healthy", "live", 1
                )
                for row in profile.executors
            },
            link_bandwidth_samples={"pcie-out": 123_456_789},
        )
        self.assertEqual(
            snapshot.links["pcie-out"].measured_bandwidth_bytes_per_s,
            123_456_789,
        )

    def test_live_composite_does_not_restate_preloaded_allocation(
        self,
    ) -> None:
        source = catalog()
        _, manifest = self.scheduler_and_manifest(source)
        operator = next(
            row for row in manifest.operators if row.kind == "ffn"
        )
        composite = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:preloaded-live",
            endpoint="synthetic://preloaded-live",
            backend="backend:composite",
            coordinator_device_id="host-a",
            participant_device_ids=("host-a", "helper-c"),
            participant_resource_ids={
                "host-a": ("compute:host-a",),
                "helper-c": ("compute:helper-c",),
            },
            route_family="operator_split",
            assisted_operator_kind="ffn",
            split_axis="column",
            split_fractions_ppm=(500_000,),
            layer_fractions_ppm=(),
            residency_states=("cold", "hot", "warm"),
            resource_ids=("compute:host-a", "compute:helper-c"),
            operator_plan_protocol="synthetic-composite-v1",
            maturity="QUALIFIED",
            evidence_ids=("synthetic-preloaded-live",),
            artifact_sha256=manifest.artifact_sha256,
            operator_placements=(RuntimeCompositeOperatorPlacement(
                operator_id=operator.operator_id,
                primary_device_id="host-a",
                helper_device_id="helper-c",
                split_axis="column",
                split_fraction_ppm=500_000,
                assisted=True,
            ),),
            adapter_parameters={
                "preloaded_resident_device_ids": "helper-c",
            },
        )
        profile = replace(source, composite_executors=(composite,))
        preloaded = catalog_preloaded_residency_samples(
            profile, {manifest.model_id: manifest}
        )
        live = live_executor_residency_sample(
            profile,
            manifest,
            composite.executor_id,
            generation=2,
        )
        fallback = ExecutorResidencySample(
            manifest,
            "executor:host-a",
            generation=1,
        )

        self.assertEqual(len(preloaded), 1)
        self.assertIsNotNone(live)
        self.assertEqual(preloaded[0].resident_device_ids, ("helper-c",))
        self.assertEqual(live.resident_device_ids, ("host-a",))
        snapshot = UnifiedRuntimeSnapshotBuilder(profile).build(
            snapshot_id="preloaded-live-allocation",
            captured_at_us=0,
            valid_until_us=10_000_000,
            memory=runtime_snapshot(manifest).memory,
            executor_samples={
                composite.executor_id: EndpointRuntimeSample(
                    "healthy", "live", 1
                ),
            },
            residencies=preloaded + (fallback, live),
        )
        by_device = {row.device_id: row for row in snapshot.residency}
        self.assertEqual(by_device["helper-c"].generation, 1)
        self.assertEqual(by_device["host-a"].generation, 2)
        self.assertEqual(
            by_device["host-a"].executor_id,
            composite.executor_id,
        )
        with self.assertRaisesRegex(
            PhysicalAdapterError,
            "runtime residency endpoint allocation conflicts",
        ):
            UnifiedRuntimeSnapshotBuilder(profile).build(
                snapshot_id="conflicting-live-allocation",
                captured_at_us=0,
                valid_until_us=10_000_000,
                memory=runtime_snapshot(manifest).memory,
                executor_samples={
                    composite.executor_id: EndpointRuntimeSample(
                        "healthy", "live", 1
                    ),
                },
                residencies=(replace(fallback, generation=2), live),
            )

    def test_control_handoff_bound_does_not_delay_dispatch(self) -> None:
        delay_upper_us = 250_000
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
            system_cost_profiles=(system_cost_profile(
                helper_interference_ppm=0,
                control_delay_us=100_000,
                control_delay_upper_us=delay_upper_us,
            ),),
        ))
        source = request("control-handoff")
        ticket = scheduler.submit_automated_request(
            source,
            manifest.model_id,
            protected_snapshot(manifest),
        )
        self.assertEqual(ticket.decision.start_us, source.arrival_us)
        self.assertEqual(
            ticket.decision.marginal_system_cost[
                "control_delay_upper_us"
            ],
            delay_upper_us,
        )
        self.assertGreaterEqual(
            ticket.decision.finish_upper_us,
            ticket.decision.finish_us + delay_upper_us,
        )
        self.assertTrue(all(
            lease.predicted_end_us == ticket.decision.finish_us
            and lease.reserved_until_us == lease.predicted_end_us
            and lease.reserved_until_us < ticket.decision.finish_upper_us
            for lease in ticket.decision.leases
        ))

    def test_tardy_baseline_does_not_bypass_phone_energy_gate(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=12_000_000_000,
            phone_power_mw=200_000,
            phone_bandwidth=8_000_000_000,
        ))
        tardy = replace(
            request("tardy-energy-gate"), deadline_us=1_001
        )
        ticket = scheduler.submit_automated_request(
            tardy,
            manifest.model_id,
            runtime_snapshot(
                manifest,
                phone_bandwidth=8_000_000_000,
            ),
        )
        baseline = next(
            row for row in ticket.cost_estimates.estimates
            if row.baseline
        )
        unsafe_phone = [
            row for row in ticket.cost_estimates.estimates
            if row.admitted
            and "helper-c" in row.details.get("device_ids", [])
            and row.details["cost_breakdown"]["finish_upper_us"]
                <= baseline.details["cost_breakdown"]["finish_upper_us"]
            and row.fleet_energy_upper_uj
                > baseline.fleet_energy_lower_uj
        ]

        self.assertTrue(unsafe_phone)
        self.assertEqual(ticket.decision.route_id, baseline.route_id)
        self.assertEqual(
            ticket.decision.reason,
            "QUALIFIED_FALLBACK_NO_SAFE_ALTERNATIVE",
        )
        self.assertEqual(ticket.selection_mode, "energy-aware")
        rejected = dict(ticket.decision.rejected)
        self.assertTrue(all(
            rejected[row.route_id] == "ENERGY_NOT_POSITIVE"
            for row in unsafe_phone
        ))

    def test_tardy_baseline_selects_energy_positive_no_slower_route(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        tardy = replace(
            request("tardy-slo-gate"), deadline_us=1_001
        )
        ticket = scheduler.submit_automated_request(
            tardy,
            manifest.model_id,
            runtime_snapshot(
                manifest,
                phone_bandwidth=8_000_000_000,
                gpu_busy_until_us=2_000_000,
            ),
        )
        baseline = next(
            row for row in ticket.cost_estimates.estimates
            if row.baseline
        )
        selected = next(
            row for row in ticket.cost_estimates.estimates
            if row.route_id == ticket.decision.route_id
        )
        qualified_phone = [
            row for row in ticket.cost_estimates.estimates
            if row.admitted
            and "helper-c" in row.details.get("device_ids", [])
            and row.details["cost_breakdown"]["finish_upper_us"]
                <= baseline.details["cost_breakdown"]["finish_upper_us"]
            and row.fleet_energy_upper_uj * 1_000_000
                <= baseline.fleet_energy_lower_uj * 990_000
        ]

        self.assertTrue(qualified_phone)
        self.assertNotEqual(ticket.decision.route_id, baseline.route_id)
        self.assertEqual(
            ticket.decision.reason,
            "CONSERVATIVE_FLEET_ENERGY_SAVING",
        )
        self.assertEqual(ticket.selection_mode, "energy-aware")
        self.assertIn("helper-c", ticket.execution_plan.device_ids)
        self.assertLessEqual(
            selected.details["cost_breakdown"]["finish_upper_us"],
            baseline.details["cost_breakdown"]["finish_upper_us"],
        )
        self.assertLessEqual(
            selected.fleet_energy_upper_uj * 1_000_000,
            baseline.fleet_energy_lower_uj * 990_000,
        )

    def test_tardy_desktop_baseline_controls_relative_slo_admission(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        tardy = replace(
            request("tardy-desktop-relative-admission"),
            deadline_us=1_001,
        )
        compiler = scheduler._automated_compiler()
        generate_one = compiler._one

        def recovery_finishes_at_deadline(*args, **kwargs):
            row = generate_one(*args, **kwargs)
            if (
                row.route_family == "whole_model"
                and row.binding.executor_id == "executor:host-a"
            ):
                row = replace(row, cost=replace(
                    row.cost,
                    service_us=1,
                    service_upper_us=1,
                    finish_us=tardy.deadline_us,
                    finish_upper_us=tardy.deadline_us,
                ))
            elif "helper-c" not in row.device_ids:
                reasons = tuple(sorted(set(
                    row.rejection_reasons + ("ENERGY_UNKNOWN",)
                )))
                row = replace(
                    row,
                    binding=replace(
                        row.binding,
                        ready=False,
                        eligibility_reasons=reasons,
                    ),
                    admitted=False,
                    rejection_reasons=reasons,
                )
            return row

        with mock.patch.object(
            compiler, "_one", side_effect=recovery_finishes_at_deadline
        ):
            candidates = scheduler.generate_automated_candidates(
                tardy,
                manifest.model_id,
                runtime_snapshot(
                    manifest,
                    phone_bandwidth=8_000_000_000,
                ),
            )

        self.assertLessEqual(
            candidates.recovery_fallback.cost.finish_upper_us,
            tardy.deadline_us,
        )
        self.assertGreater(
            candidates.baseline.cost.finish_upper_us,
            tardy.deadline_us,
        )
        relative_phone = tuple(
            row for row in candidates.candidates
            if "helper-c" in row.device_ids
            and row.maturity == "QUALIFIED"
            and row.cost.finish_upper_us > tardy.deadline_us
        )
        self.assertTrue(relative_phone)
        self.assertTrue(all(
            "SLO_UPPER_BOUND" not in row.rejection_reasons
            for row in relative_phone
        ))

    def test_tardy_baseline_does_not_bypass_interference_gate(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
            system_cost_profiles=(system_cost_profile(
                helper_interference_ppm=100_000_000,
            ),),
        ))
        tardy = replace(
            request("tardy-interference-gate"), deadline_us=1_001
        )
        ticket = scheduler.submit_automated_request(
            tardy,
            manifest.model_id,
            protected_snapshot(manifest),
        )
        baseline = next(
            row for row in ticket.cost_estimates.estimates
            if row.baseline
        )
        interfering = [
            row for row in ticket.cost_estimates.estimates
            if row.admitted
            and row.details.get("marginal_system_cost") is not None
            and row.details["marginal_system_cost"][
                "interference_upper_us"
            ] > 0
        ]

        self.assertTrue(interfering)
        self.assertGreater(
            baseline.details["cost_breakdown"]["finish_upper_us"],
            tardy.deadline_us,
        )
        self.assertNotIn(
            ticket.decision.route_id,
            {row.route_id for row in interfering},
        )
        self.assertEqual(ticket.selection_mode, "energy-aware")
        rejected = dict(ticket.decision.rejected)
        self.assertTrue(all(
            rejected[row.route_id] == "PROTECTED_WORK_DELAY"
            for row in interfering
        ))

    def test_tardy_fallback_remains_executable_and_minimizes_tardiness(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        tardy = replace(request("all-routes-tardy"), deadline_us=1_001)
        ticket = scheduler.submit_automated_request(
            tardy,
            manifest.model_id,
            runtime_snapshot(manifest, phone_bandwidth=8_000_000_000),
            selection_mode="deadline-first",
        )
        estimates = {
            row.route_id: row for row in ticket.cost_estimates.estimates
        }
        baseline = next(
            row for row in estimates.values()
            if row.baseline
        )
        selected = estimates[ticket.decision.route_id]
        admitted = [
            row for row in estimates.values()
            if row.admitted
        ]
        self.assertGreater(
            baseline.details["cost_breakdown"]["finish_upper_us"],
            tardy.deadline_us,
        )
        self.assertGreater(
            selected.details["cost_breakdown"]["finish_upper_us"],
            tardy.deadline_us,
        )
        self.assertEqual(
            selected.details["cost_breakdown"]["finish_upper_us"],
            min(
                row.details["cost_breakdown"]["finish_upper_us"]
                for row in admitted
            ),
        )
        self.assertEqual(
            ticket.decision.reason, "MINIMUM_CONSERVATIVE_TARDINESS"
        )
        self.assertEqual(ticket.selection_mode, "deadline-first")

    def test_arrival_identity_is_separate_from_observation_time(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        original = request("delayed-observation", arrival_us=1_000)
        ticket = scheduler.submit_automated_request(
            original,
            manifest.model_id,
            runtime_snapshot(manifest),
            observed_at_us=1_600,
        )
        self.assertEqual(ticket.request, original)
        self.assertGreaterEqual(ticket.decision.start_us, 1_600)
        selected = next(
            row for row in ticket.cost_estimates.estimates
            if row.route_id == ticket.decision.route_id
        )
        self.assertGreaterEqual(
            selected.details["cost_breakdown"]["queue_delay_us"],
            600,
        )
        record = scheduler.runtime_decision_log()["records"][0]
        self.assertEqual(record["event_time_us"], 1_600)

    def test_completion_evidence_uses_ticket_executor_without_plan_cache(
        self,
    ) -> None:
        profile = catalog()
        scheduler, manifest = self.scheduler_and_manifest(profile)
        value = request("executor-bound-completion")
        snapshot = runtime_snapshot(manifest)
        selected = scheduler.generate_automated_candidates(
            value, manifest.model_id, snapshot
        ).baseline
        fresh, _ = self.scheduler_and_manifest(profile)
        compiler = fresh._automated_compiler()
        receipt = RuntimeExecutionReceipt(
            ticket_id="executor-bound-ticket",
            request_id=value.request_id,
            artifact_sha256=manifest.artifact_sha256,
            operator_plan_sha256=selected.plan.plan_sha256,
            executor_id=selected.binding.executor_id,
            endpoint=selected.binding.endpoint,
            operator_plan_protocol=(
                selected.binding.operator_plan_protocol
            ),
            participant_executor_ids=tuple(
                row.executor_id for row in selected.binding.participants
            ),
            started_us=0,
            finished_us=max(1, selected.cost.component_service_us),
            output_sha256="sha256:" + "f" * 64,
            status="COMPLETED",
        )

        self.assertTrue(compiler.record_execution_observation(
            value,
            manifest,
            selected.plan,
            receipt,
            snapshot.cost_features,
            selected.cost.component_service_us,
            selected.cost.component_energy_uj,
            selected.binding.executor_id,
        ))

    def test_failed_transition_atomically_replans_and_is_journaled(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        snapshot = runtime_snapshot(
            manifest,
            phone_bandwidth=8_000_000_000,
            resident_devices=(),
        )
        ticket = scheduler.submit_automated_request(
            request("failed-transition"), manifest.model_id, snapshot
        )
        epoch_ns = time.monotonic_ns() - ticket.decision.start_us * 1_000
        active = scheduler.wait_runtime_request(ticket.request.request_id, epoch_ns)
        receipts = self.transition_receipts(active, failed=True)
        recovery = scheduler.fail_automated_request(
            active.request.request_id,
            failed_at_us=active.decision.start_us + 1,
            reason="synthetic_transition_failure",
            snapshot=snapshot,
            transition_receipts=receipts,
        )
        self.assertIsNotNone(recovery.fallback)
        self.assertEqual(
            recovery.fallback.previous_transition_receipts, receipts
        )
        self.assertNotEqual(
            recovery.fallback.decision.route_id, active.decision.route_id
        )
        fallback_record = scheduler.runtime_decision_log()["records"][-1]
        self.assertEqual(fallback_record["event_kind"], "FALLBACK")
        self.assertEqual(
            fallback_record["selected"]["previous_transition_receipts"],
            [row.to_json() for row in receipts],
        )

    def test_automated_completion_requires_the_exact_physical_receipt(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        ticket = scheduler.submit_automated_request(
            request("physical-receipt"), manifest.model_id, runtime_snapshot(manifest)
        )
        epoch_ns = time.monotonic_ns() - ticket.decision.start_us * 1_000
        active = scheduler.wait_runtime_request(ticket.request.request_id, epoch_ns)
        before = (
            scheduler.timeline.causal_state(),
            scheduler.runtime_memory_state(),
            scheduler.runtime_controller_snapshot(),
            scheduler.runtime_decision_log_bytes(),
        )
        with self.assertRaises(UnifiedScheduleError):
            scheduler.complete_runtime_request(
                active.request.request_id, active.decision.finish_us
            )
        invalid = self.execution_receipt(
            active, endpoint="synthetic://wrong-endpoint"
        )
        with self.assertRaises(UnifiedScheduleError):
            scheduler.complete_automated_request(
                active.request.request_id, invalid
            )
        self.assertEqual(scheduler.timeline.causal_state(), before[0])
        self.assertEqual(scheduler.runtime_memory_state(), before[1])
        self.assertEqual(scheduler.runtime_controller_snapshot(), before[2])
        self.assertEqual(scheduler.runtime_decision_log_bytes(), before[3])

        physical = self.execution_receipt(active)
        completion = scheduler.complete_automated_request(
            active.request.request_id, physical
        )
        self.assertEqual(completion.execution_receipt, physical)
        terminal = scheduler.runtime_decision_log()["records"][-1]
        self.assertEqual(
            terminal["selected"]["execution_receipt"], physical.to_json()
        )

    def test_failed_journal_transaction_rolls_back_memory_and_execution(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        before_timeline = scheduler.timeline.causal_state()
        before_memory = scheduler.runtime_memory_state()
        before_controller = scheduler.runtime_controller_snapshot()
        before_log = scheduler.runtime_decision_log_bytes()
        with mock.patch.object(
            scheduler._runtime_decision_log,
            "append",
            side_effect=DecisionLogError("synthetic journal failure"),
        ):
            with self.assertRaises(UnifiedScheduleError):
                scheduler.submit_automated_request(
                    request("journal-rollback"),
                    manifest.model_id,
                    runtime_snapshot(manifest, phone_bandwidth=8_000_000_000),
                )
        self.assertEqual(scheduler.timeline.causal_state(), before_timeline)
        self.assertEqual(scheduler.runtime_memory_state(), before_memory)
        self.assertEqual(
            scheduler.runtime_controller_snapshot(), before_controller
        )
        self.assertEqual(scheduler.runtime_decision_log_bytes(), before_log)

    def test_failed_acquisition_journal_write_rolls_back_dispatch(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        ticket = scheduler.submit_automated_request(
            request("acquire-journal-rollback"),
            manifest.model_id,
            runtime_snapshot(manifest),
        )
        before_controller = scheduler.runtime_controller_snapshot()
        before_log = scheduler.runtime_decision_log_bytes()
        epoch_ns = time.monotonic_ns() - ticket.decision.start_us * 1_000
        with mock.patch.object(
            scheduler._runtime_decision_log,
            "append",
            side_effect=DecisionLogError("synthetic acquisition journal failure"),
        ):
            with self.assertRaises(UnifiedScheduleError):
                scheduler.wait_runtime_request(
                    ticket.request.request_id, epoch_ns
                )
        self.assertEqual(
            scheduler.runtime_controller_snapshot(), before_controller
        )
        self.assertEqual(scheduler.runtime_decision_log_bytes(), before_log)

    def test_failed_composite_memory_reservation_is_atomic(self) -> None:
        profile = catalog(
            phone_ops_per_s=3_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
            phone_whole_model=False,
            gpu_whole_model=False,
        )
        probe, probe_manifest = self.scheduler_and_manifest(profile)
        probe_ticket = probe.submit_automated_request(
            request("composite-probe"),
            probe_manifest.model_id,
            runtime_snapshot(probe_manifest, phone_bandwidth=8_000_000_000),
        )
        self.assertGreater(len(probe_ticket.execution_plan.device_ids), 1)
        self.assertGreater(len({
            row.resource_id for row in probe_ticket.memory_demands
        }), 1)

        scheduler, manifest = self.scheduler_and_manifest(profile)
        scheduler._runtime_memory.fail_after_mutations_for_test(1)
        before = scheduler.runtime_memory_state()
        with self.assertRaises(UnifiedScheduleError):
            scheduler.submit_automated_request(
                request("memory-rollback"),
                manifest.model_id,
                runtime_snapshot(manifest, phone_bandwidth=8_000_000_000),
            )
        self.assertEqual(scheduler.runtime_memory_state(), before)
        self.assertEqual(
            scheduler.runtime_controller_snapshot()["tickets"], {}
        )
        self.assertEqual(
            scheduler.runtime_decision_log()["records"], []
        )

    def test_pending_projection_does_not_hide_structural_memory_failure(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        snapshot = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        original_preview = scheduler._runtime_memory.preview

        def preview(demands, memory_snapshot, **interval):
            if any(
                row.resource_id == "phone-memory" for row in demands
            ):
                raise RuntimeResourceError(
                    "memory capacity is insufficient: phone-memory"
                )
            return original_preview(
                demands, memory_snapshot, **interval
            )

        with mock.patch.object(
            scheduler._runtime_memory, "preview", side_effect=preview
        ):
            ticket = scheduler.submit_automated_request(
                request("pending-projection-structural-memory"),
                manifest.model_id,
                snapshot,
            )

        self.assertNotIn("helper-c", ticket.execution_plan.device_ids)
        self.assertTrue(any(
            reason == "MEMORY_CAPACITY_CURRENT:phone-memory"
            for _, reason in ticket.decision.rejected
        ))

    def test_projected_live_capacity_reselects_before_atomic_reservation(
        self,
    ) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        snapshot = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        original_preview = scheduler._runtime_memory.preview

        def preview(demands, memory_snapshot, **interval):
            if (
                interval.get("enforce_live_capacity", True)
                and any(
                    row.resource_id == "phone-memory" for row in demands
                )
            ):
                raise RuntimeResourceError(
                    "memory capacity is insufficient: phone-memory"
                )
            return original_preview(
                demands, memory_snapshot, **interval
            )

        with mock.patch.object(
            scheduler._runtime_memory, "preview", side_effect=preview
        ):
            ticket = scheduler.submit_automated_request(
                request("projected-live-capacity-fallback"),
                manifest.model_id,
                snapshot,
            )

        self.assertNotIn("helper-c", ticket.execution_plan.device_ids)
        self.assertTrue(any(
            reason == "MEMORY_CAPACITY_CURRENT:phone-memory"
            for _, reason in ticket.decision.rejected
        ))
        self.assertEqual(
            len(scheduler.runtime_controller_snapshot()["tickets"]), 1
        )
        self.assertEqual(
            len(scheduler.runtime_decision_log()["records"]), 1
        )

    def test_pending_exclusive_transition_projects_exact_replacement(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            transitions=tuple(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        first_path = Path(self.directory.name) / "first-model.gguf"
        write_synthetic_gguf(first_path, sliding_window=64)
        first_model = scheduler.register_gguf_model(
            "synthetic-first", first_path
        )
        second_model = scheduler.register_gguf_model(
            "synthetic-second", self.path
        )
        cold = runtime_snapshot(
            first_model,
            include_phone=False,
            resident_devices=(),
        )
        cold = replace(
            cold,
            executors={
                "executor:accelerator-b": executor_state(
                    "executor:accelerator-b"
                )
            },
        )
        first_probe = scheduler.generate_automated_candidates(
            request("pending-first-probe"), first_model.model_id, cold
        ).baseline
        second_shape = request(
            "pending-second-probe",
            input_tokens=2_048,
            output_tokens=256,
        )
        second_probe = scheduler.generate_automated_candidates(
            second_shape,
            second_model.model_id,
            replace(cold, residency=()),
        ).baseline
        first_required = sum(
            row.required_bytes for row in first_probe.plan.memory_demands
            if row.resource_id == "gpu-memory"
        )
        second_required = sum(
            row.required_bytes for row in second_probe.plan.memory_demands
            if row.resource_id == "gpu-memory"
        )
        measured_process_overhead = 100
        self.assertGreater(
            second_required,
            first_required + measured_process_overhead,
        )
        capacity_bytes = second_required
        cold = replace(
            cold,
            memory=replace(
                cold.memory,
                capacities={
                    "gpu-memory": DeviceMemoryCapacity(
                        "gpu-memory", capacity_bytes, 0, 0
                    )
                },
            ),
        )
        first = scheduler.submit_automated_request(
            request("pending-first", arrival_us=1_000),
            first_model.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        self.assertEqual(first.transition_status, "PENDING")
        self.assertEqual(first.dispatch_state, "QUEUED")

        pending_snapshot = replace(
            cold,
            snapshot_id="synthetic-pending-publication",
            captured_at_us=1_100,
            valid_until_us=10_000_000,
            memory=RuntimePlacementSnapshot(
                snapshot_id="synthetic-pending-memory",
                captured_at_us=1_100,
                valid_until_us=10_000_000,
                capacities={
                    "gpu-memory": DeviceMemoryCapacity(
                        "gpu-memory",
                        capacity_bytes,
                        first_required + measured_process_overhead,
                        0,
                    )
                },
            ),
            residency=(),
        )
        second = scheduler.submit_automated_request(
            request(
                "pending-second",
                arrival_us=1_100,
                input_tokens=second_shape.input_tokens,
                output_tokens=second_shape.output_tokens,
            ),
            second_model.model_id,
            pending_snapshot,
            selection_mode="desktop-baseline",
        )

        self.assertGreaterEqual(
            second.decision.start_us,
            max(row.reserved_until_us for row in first.decision.leases),
        )
        self.assertEqual(second.execution_plan.device_ids, ("accelerator-b",))
        self.assertEqual(len(second.execution_plan.transitions), 1)
        evictions = second.execution_plan.transitions[0].evictions
        self.assertEqual(len(evictions), 1)
        self.assertEqual(evictions[0].model_id, first_model.model_id)
        self.assertEqual(
            evictions[0].artifact_sha256, first_model.artifact_sha256
        )
        self.assertEqual(evictions[0].generation, 1)
        self.assertEqual(
            evictions[0].reclaimable_bytes,
            first_required + measured_process_overhead,
        )
        weights = next(
            row for row in second.execution_plan.memory_demands
            if row.demand_id == "weights:accelerator-b"
        )
        self.assertGreater(weights.replaceable_bytes, 0)
        self.assertEqual(weights.replacement_group, "compute:accelerator-b")
        self.assertEqual(pending_snapshot.residency, ())

        changed_pending_snapshot = replace(
            pending_snapshot,
            snapshot_id="synthetic-pending-publication-changed",
            captured_at_us=1_200,
            memory=replace(
                pending_snapshot.memory,
                snapshot_id="synthetic-pending-memory-changed",
                captured_at_us=1_200,
                capacities={
                    "gpu-memory": replace(
                        pending_snapshot.memory.capacities["gpu-memory"],
                        occupied_bytes=(
                            first_required + measured_process_overhead + 1
                        ),
                    )
                },
            ),
        )
        third = scheduler.submit_automated_request(
            request(
                "pending-third",
                arrival_us=1_200,
                input_tokens=second_shape.input_tokens,
                output_tokens=second_shape.output_tokens,
            ),
            second_model.model_id,
            changed_pending_snapshot,
            selection_mode="desktop-baseline",
        )

        self.assertEqual(third.dispatch_state, "QUEUED")
        self.assertGreaterEqual(
            third.decision.start_us,
            max(row.reserved_until_us for row in second.decision.leases),
        )
        self.assertEqual(third.execution_plan.device_ids, ("accelerator-b",))

    def test_pending_replacement_does_not_double_count_loaded_target(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            qualified_fallback=True,
            exclusive_residency_resource_id="compute:accelerator-b",
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=(gpu,),
            transitions=tuple(
                row for row in source.transitions
                if row.device_id == "accelerator-b"
            ),
        ).to_json())
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        target = scheduler.register_gguf_model(
            "pending-loaded-target", self.path
        )
        old_path = Path(self.directory.name) / "pending-loaded-old.gguf"
        write_synthetic_gguf(old_path, sliding_window=64)
        old = scheduler.register_gguf_model(
            "pending-loaded-old", old_path
        )
        cold = replace(
            runtime_snapshot(
                target,
                include_phone=False,
                resident_devices=(),
            ),
            executors={
                "executor:accelerator-b": executor_state(
                    "executor:accelerator-b"
                )
            },
        )
        probe = scheduler.generate_automated_candidates(
            request("pending-loaded-probe"), target.model_id, cold
        ).baseline
        required = sum(
            row.required_bytes for row in probe.plan.memory_demands
            if row.resource_id == "gpu-memory"
        )
        request_bytes = sum(
            row.required_bytes for row in probe.plan.memory_demands
            if row.resource_id == "gpu-memory"
            and row.kind != "model_weights"
        )
        capacity_bytes = required + request_bytes
        old_bytes = 100
        initial = replace(
            cold,
            memory=replace(
                cold.memory,
                capacities={
                    "gpu-memory": DeviceMemoryCapacity(
                        "gpu-memory", capacity_bytes, old_bytes, 0
                    )
                },
            ),
            residency=(ModelResidencyObservation(
                model_id=old.model_id,
                artifact_sha256=old.artifact_sha256,
                device_id="accelerator-b",
                state="hot",
                resident_tensor_ids=tuple(
                    row.tensor_id for row in old.tensors
                ),
                resident_bytes=old_bytes,
                generation=7,
            ),),
        )
        first = scheduler.submit_automated_request(
            request("pending-loaded-first", arrival_us=1_000),
            target.model_id,
            initial,
            selection_mode="desktop-baseline",
        )
        first = scheduler.wait_runtime_request(
            first.request.request_id,
            time.monotonic_ns() - first.decision.start_us * 1_000,
        )
        self.assertEqual(first.dispatch_state, "ACQUIRED")
        self.assertEqual(first.transition_status, "PENDING")

        in_flight = replace(
            initial,
            snapshot_id="synthetic-pending-loaded-target",
            captured_at_us=1_100,
            valid_until_us=10_000_000,
            memory=RuntimePlacementSnapshot(
                snapshot_id="synthetic-pending-loaded-target-memory",
                captured_at_us=1_100,
                valid_until_us=10_000_000,
                capacities={
                    "gpu-memory": DeviceMemoryCapacity(
                        "gpu-memory", capacity_bytes, required, 0
                    )
                },
            ),
        )
        second = scheduler.submit_automated_request(
            request("pending-loaded-second", arrival_us=1_100),
            target.model_id,
            in_flight,
            selection_mode="desktop-baseline",
        )

        self.assertEqual(second.binding.executor_id, first.binding.executor_id)
        self.assertEqual(second.execution_plan.residency_variant, "hot")
        self.assertEqual(second.execution_plan.transitions, ())
        self.assertGreaterEqual(
            second.decision.start_us,
            max(row.reserved_until_us for row in first.decision.leases),
        )

    def test_pending_composite_transition_replaces_owned_host_and_gpu(
        self,
    ) -> None:
        source = catalog()
        gpu = replace(
            source.executor_by_device["accelerator-b"],
            supports_whole_model=False,
            exclusive_residency_resource_id="compute:accelerator-b",
            layer_fractions_ppm=(),
        )
        host = replace(
            source.executor_by_device["host-a"],
            exclusive_residency_resource_id="compute:host-a",
            layer_fractions_ppm=(),
        )
        composite = RuntimeCompositeExecutorCapability(
            executor_id="coordinator:host-accelerator",
            endpoint="synthetic://host-accelerator",
            backend="backend:composite",
            coordinator_device_id="host-a",
            participant_device_ids=("host-a", "accelerator-b"),
            participant_resource_ids={
                "host-a": ("compute:host-a",),
                "accelerator-b": ("compute:accelerator-b",),
            },
            route_family="layer_placement",
            assisted_operator_kind=None,
            split_axis="none",
            split_fractions_ppm=(),
            layer_fractions_ppm=(500_000,),
            residency_states=("cold", "hot", "warm"),
            resource_ids=("compute:host-a", "compute:accelerator-b"),
            operator_plan_protocol="synthetic-composite-v1",
            maturity="QUALIFIED",
            evidence_ids=("synthetic-composite-replacement",),
            replacement_group_by_device={
                "host-a": "compute:host-a",
                "accelerator-b": "compute:accelerator-b",
            },
        )
        transition = RuntimeTransitionCapability(
            transition_id="load:host-accelerator",
            device_id="accelerator-b",
            source_state="cold",
            target_state="hot",
            fixed_latency_us=100,
            bandwidth_bytes_per_s=1_000_000_000,
            fixed_energy_uj=100,
            dynamic_pj_per_byte=100,
            resource_ids=("compute:host-a", "compute:accelerator-b"),
            maturity="QUALIFIED",
            evidence_ids=("synthetic-composite-replacement",),
            executor_id=composite.executor_id,
            prepares_device_ids=("host-a", "accelerator-b"),
        )
        profile = RuntimeCapabilityCatalog.from_json(replace(
            source,
            executors=tuple(
                gpu if row.device_id == gpu.device_id else
                host if row.device_id == host.device_id else row
                for row in source.executors
            ),
            composite_executors=(composite,),
            transitions=source.transitions + (transition,),
        ).to_json())
        restored = profile.composite_executor_by_id[composite.executor_id]
        self.assertEqual(
            dict(restored.replacement_group_by_device),
            dict(composite.replacement_group_by_device),
        )
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(profile)
        first_path = Path(self.directory.name) / "composite-first.gguf"
        second_path = Path(self.directory.name) / "composite-second.gguf"
        write_synthetic_gguf(first_path, sliding_window=64)
        write_synthetic_gguf(second_path, sliding_window=96)
        first_model = scheduler.register_gguf_model(
            "composite-first", first_path
        )
        second_model = scheduler.register_gguf_model(
            "composite-second", second_path
        )
        cold = runtime_snapshot(
            first_model,
            include_phone=False,
            resident_devices=(),
        )
        states = dict(cold.executors)
        states[composite.executor_id] = RuntimeExecutorState(
            executor_id=composite.executor_id,
            healthy=True,
            ready=False,
            temperature_millic=40_000,
            battery_ppm=900_000,
            free_slots=0,
            busy_until_us=0,
        )
        cold = replace(cold, executors=states)
        first = scheduler.submit_automated_request(
            request("pending-composite-first", arrival_us=1_000),
            first_model.model_id,
            cold,
            selection_mode="desktop-baseline",
        )
        self.assertEqual(first.binding.executor_id, composite.executor_id)
        self.assertEqual(first.transition_status, "PENDING")
        by_resource = {}
        weights_by_resource = {}
        for demand in first.execution_plan.memory_demands:
            by_resource[demand.resource_id] = (
                by_resource.get(demand.resource_id, 0)
                + demand.required_bytes
            )
            if demand.kind == "model_weights":
                weights_by_resource[demand.resource_id] = (
                    weights_by_resource.get(demand.resource_id, 0)
                    + demand.required_bytes
                )
        pending = replace(
            cold,
            snapshot_id="synthetic-pending-composite",
            captured_at_us=1_100,
            valid_until_us=10_000_000,
            memory=RuntimePlacementSnapshot(
                snapshot_id="synthetic-pending-composite-memory",
                captured_at_us=1_100,
                valid_until_us=10_000_000,
                capacities={
                    resource_id: DeviceMemoryCapacity(
                        resource_id,
                        required * 2,
                        weights_by_resource.get(resource_id, 0),
                        0,
                    )
                    for resource_id, required in by_resource.items()
                },
            ),
            residency=(),
        )
        second = scheduler.submit_automated_request(
            request("pending-composite-second", arrival_us=1_100),
            second_model.model_id,
            pending,
            selection_mode="desktop-baseline",
        )

        self.assertEqual(second.binding.executor_id, composite.executor_id)
        self.assertGreaterEqual(
            second.decision.start_us,
            max(row.reserved_until_us for row in first.decision.leases),
        )
        transition_plan = second.execution_plan.transitions[0]
        self.assertEqual(
            {row.device_id for row in transition_plan.evictions},
            {"host-a", "accelerator-b"},
        )
        self.assertEqual(
            {row.replacement_group for row in transition_plan.evictions},
            {"compute:accelerator-b", "compute:host-a"},
        )
        replacement = {
            row.device_id: row.replaceable_bytes
            for row in second.execution_plan.memory_demands
            if row.kind == "model_weights"
        }
        self.assertGreater(replacement["host-a"], 0)
        self.assertGreater(replacement["accelerator-b"], 0)

        later_pending = replace(
            pending,
            snapshot_id="synthetic-pending-composite-later",
            captured_at_us=1_200,
            valid_until_us=10_000_000,
            memory=replace(
                cold.memory,
                snapshot_id="synthetic-pending-composite-memory-later",
                captured_at_us=1_200,
                valid_until_us=10_000_000,
            ),
        )
        third = scheduler.submit_automated_request(
            request("pending-composite-third", arrival_us=1_200),
            second_model.model_id,
            later_pending,
            selection_mode="desktop-baseline",
        )

        self.assertEqual(third.binding.executor_id, composite.executor_id)
        self.assertGreaterEqual(
            third.decision.start_us,
            max(row.reserved_until_us for row in second.decision.leases),
        )
        self.assertEqual(third.execution_plan.residency_variant, "hot")
        self.assertEqual(third.execution_plan.transitions, ())

    def test_tardy_memory_rejected_baseline_is_not_reintroduced(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest(catalog(
            phone_ops_per_s=6_000_000_000,
            phone_power_mw=1_500,
            phone_bandwidth=8_000_000_000,
        ))
        snapshot = runtime_snapshot(
            manifest, phone_bandwidth=8_000_000_000
        )
        original_preview = scheduler._runtime_memory.preview

        def preview(demands, memory_snapshot, **interval):
            if any(row.resource_id == "gpu-memory" for row in demands):
                raise RuntimeResourceError(
                    "memory capacity is insufficient: gpu-memory"
                )
            return original_preview(
                demands, memory_snapshot, **interval
            )

        late = replace(
            request("tardy-memory-selection-fallback"),
            deadline_us=1_001,
        )
        with mock.patch.object(
            scheduler._runtime_memory, "preview", side_effect=preview
        ):
            ticket = scheduler.submit_automated_request(
                late,
                manifest.model_id,
                snapshot,
                selection_mode="deadline-first",
            )

        self.assertNotIn(
            "accelerator-b", ticket.execution_plan.device_ids
        )
        self.assertEqual(ticket.selection_mode, "deadline-first")
        self.assertTrue(any(
            reason == "MEMORY_CAPACITY_CURRENT:gpu-memory"
            for _, reason in ticket.decision.rejected
        ))

    def test_cancel_releases_memory_and_terminal_is_immutable(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        ticket = scheduler.submit_automated_request(
            request("automated-cancel"),
            manifest.model_id,
            runtime_snapshot(manifest),
        )
        self.assertTrue(scheduler.runtime_memory_state()["reservations"])
        scheduler.cancel_runtime_request(
            ticket.request.request_id,
            ticket.request.arrival_us,
            "synthetic_cancel",
        )
        self.assertEqual(scheduler.runtime_memory_state()["reservations"], [])
        with self.assertRaises(UnifiedScheduleError):
            scheduler.cancel_runtime_request(
                ticket.request.request_id,
                ticket.request.arrival_us,
                "duplicate_cancel",
            )
        events = [
            row["event_kind"]
            for row in scheduler.runtime_decision_log()["records"]
        ]
        self.assertEqual(events, ["DECISION", "CANCELLED"])

    def test_failed_terminal_journal_write_rolls_back_release(self) -> None:
        scheduler, manifest = self.scheduler_and_manifest()
        ticket = scheduler.submit_automated_request(
            request("terminal-rollback"),
            manifest.model_id,
            runtime_snapshot(manifest),
        )
        epoch_ns = time.monotonic_ns() - ticket.decision.start_us * 1_000
        active = scheduler.wait_runtime_request(ticket.request.request_id, epoch_ns)
        before_timeline = scheduler.timeline.causal_state()
        before_memory = scheduler.runtime_memory_state()
        before_controller = scheduler.runtime_controller_snapshot()
        before_log = scheduler.runtime_decision_log_bytes()
        with mock.patch.object(
            scheduler._runtime_decision_log,
            "append",
            side_effect=DecisionLogError("synthetic terminal journal failure"),
        ):
            with self.assertRaises(UnifiedScheduleError):
                scheduler.complete_automated_request(
                    active.request.request_id, self.execution_receipt(active)
                )
        self.assertEqual(scheduler.timeline.causal_state(), before_timeline)
        self.assertEqual(scheduler.runtime_memory_state(), before_memory)
        self.assertEqual(
            scheduler.runtime_controller_snapshot(), before_controller
        )
        self.assertEqual(scheduler.runtime_decision_log_bytes(), before_log)


if __name__ == "__main__":
    unittest.main()
